"""
noce_net.layers
===============
Building blocks of NOCE-Net.

* :class:`MultiPatchEmbedding`  - multi-kernel Conv1D patch embedding of the
  intra-cycle inputs (Patch Encoder input).
* :class:`MultiPatchIndexing`   - time-index embedding pooled to the same patches.
* :class:`TaskEmbeddingCyc`     - Task-Emb of the task vector ``eta``.
* :class:`FECEmbedding`         - FEC-Emb of ``gamma_c = [c, Delta_c]``.
* :class:`PatchEncoder`         - stack of transformer blocks (used both as
  Patch Encoder and Patch Decoder).
* :class:`OutputRecon`          - output reconstruction layer (ORL).
* :class:`CycleBlock`           - complete NOCE-Net forward pass.
* :class:`Predictor`, :class:`StateMixer`, :class:`TemporalLayerNorm` -
  auxiliary capacity-tracking head that reads the recurrent states.

Notation (shared with the rest of the package): ``B`` batteries, ``L``
FECs, ``T`` time steps per FEC, ``d_u`` input features, ``D = dmodel``,
``K`` patch sizes ``s_1 < ... < s_K``, ``T_p = sum_k ceil(T / s_k)``
patch tokens per FEC and ``d_h = 3 D`` recurrent state size.

``CycleBlock.forward`` returns ``(V_hat, rul_pred, H, h_last)``. The
auxiliary ``rul_pred`` output is computed but no loss is attached to it
in this repository, so NOCE-Net is trained on voltage response only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from noce_net.transformer import ParallelTransformerBlock

class SwiGLU(nn.Module):
    """SwiGLU activation: splits the last dimension in two halves ``(x, g)`` and returns ``silu(g) * x``."""
    def forward(self, x):
        """(..., 2F) -> (..., F)."""
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class TemporalLayerNorm(nn.Module):
    """
    Normalisation along the temporal axis of a (batch, time, features) tensor.

    Each feature is standardised with its mean and standard deviation over
    time and, if ``affine`` is True, rescaled with learnable ``gamma``/``beta``.

    Args:
        feature_dim: number of features (last dimension).
        eps: numerical stabiliser added to the standard deviation.
        affine: apply the learnable affine transform.
    """

    def __init__(self, feature_dim, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.gamma = nn.Parameter(torch.ones(1, 1, feature_dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, feature_dim))

    def forward(self, x):
        # x: (batch_size, time_steps, features)
        """Normalise ``x`` of shape (batch, time, features) along time; returns the same shape."""
        if x.dim() != 3:
            raise ValueError("Input must be 3D: (batch_size, time_steps, features)")

        # Compute mean and std along time dimension (dim=1)
        mean = x.mean(dim=1, keepdim=True)  # shape: (batch_size, 1, features)
        std = x.std(dim=1, keepdim=True)    # shape: (batch_size, 1, features)

        x_norm = (x - mean) / (std + self.eps)

        # Initialize gamma and beta lazily based on feature size
        #if self.affine and self.gamma is None:
        #    feature_dim = x.size(2)
        #    self.gamma = nn.Parameter(torch.ones(1, 1, feature_dim))
        #    self.beta = nn.Parameter(torch.zeros(1, 1, feature_dim))

        if self.affine:
            x_norm = x_norm * self.gamma + self.beta

        return x_norm


class StateMixer(nn.Module):
  """
  MLP-Mixer style block for the auxiliary capacity head.

  Mixes across the ``N`` state features and across the ``D`` channels
  of a (B, L, N, D) tensor, each followed by temporal normalisation and a
  residual connection.

  Args:
      configs: ``model_configs`` (uses ``N`` and ``dmodel``).
  """

  def __init__(self, configs):
    super().__init__()

    self.configs = configs

    self.feature_mixer = nn.Sequential(nn.Linear(configs['N'], configs['dmodel']),
                                       nn.GELU(),
                                       nn.Linear(configs['dmodel'], configs['N']))
    self.channel_mixer = nn.Sequential(nn.Linear(configs['dmodel'], 2*configs['dmodel']),
                                       nn.GELU(),
                                       nn.Linear(2*configs['dmodel'], configs['dmodel']))

    self.feature_norm = TemporalLayerNorm(configs['N'])
    self.channel_norm = TemporalLayerNorm(configs['dmodel'])

  def forward(self, x):
      """Feature mixing then channel mixing of ``x`` (B, L, N, D); returns (B, L, N, D)."""
      B, L, F, D = x.shape

      res = x

      x = self.feature_mixer(x.permute(0,3,1,2))
      x = self.feature_norm(x.reshape(-1, L, F)).reshape(B, D, L, F).permute(0,2,3,1)
      x = x + res

      x = self.channel_mixer(x)
      x = self.channel_norm(x.permute(0,2,1,3).reshape(-1, L, D)).reshape(B, F, L, D).permute(0,2,1,3)
      x = x + res

      return x

class Predictor(nn.Module):
  """
  Auxiliary capacity-tracking head operating on the recurrent states.

  The state sequence (B, L+1, 3D) is reshaped into its three D-dimensional
  tokens; first and second differences across FECs are appended
  (``N = 9`` features), processed by ``num_mixer_layers`` :class:`StateMixer`
  blocks and mapped to one scalar per FEC.

  Args:
      configs: ``model_configs`` (uses ``N``, ``dmodel``, ``num_mixer_layers``).

  Returns (forward):
      (B, L, 1) tensor.
  """
  def __init__(self, configs):

    super().__init__()

    self.configs = configs
    #self.feature_init_mix = nn.Linear(configs[''] , configs['dmodel'])
    self.MLP = [StateMixer(configs) for _ in range(configs['num_mixer_layers'])]
    self.MLP = nn.Sequential(*self.MLP)

    #self.feature_combiner = nn.Linear(configs['dmodel'], 1)
    self.head = nn.Sequential(nn.Linear(configs['dmodel']*configs['N'], 4*configs['dmodel']),
                              nn.GELU(),
                              nn.Linear(4*configs['dmodel'], configs['dmodel']),
                              nn.GELU(),
                              nn.Linear(configs['dmodel'], 1))


  def forward(self, x, is_norm = None):

    """Map recurrent states (B, L+1, 3D) to one value per FEC (B, L, 1); ``is_norm`` is unused."""
    B, L = x.shape[0], x.shape[1]
    x = x.reshape(B, L, 3, -1)
    #print(x.shape)
    xdiff = x[:,1:,:] - x[:,:-1,:]
    xdiff2 = torch.cat((torch.zeros_like(xdiff[:,:1,:]), xdiff[:,1:,:] - xdiff[:,:-1,:]) , dim = 1)
    x = torch.cat([x[:,1:], xdiff, xdiff2], dim = 2)

    #x = self.feature_init_mix(x.permute(0,1,3,2)).permute(0,1,3,2)
    x = self.MLP(x)
    #print(x.shape)
    #x = self.feature_combiner(x.permute(0,1,3,2)).squeeze(-1)
    x = self.head(x.reshape(B, L-1, -1))
    return x


class MultiPatchEmbedding(nn.Module):
    """
    Multi-patch embedding of the intra-cycle inputs.

    For every patch size ``s_k`` a ``Conv1d(d_u, D, kernel=s_k, stride=s_k)``
    embeds non-overlapping patches; the token sequences of all patch sizes
    are concatenated along time. The padding mask is max-pooled with the
    same patch sizes so that a token is valid if any of its steps is valid.

    Args:
        din: input features ``d_u``.
        dmodel: embedding dimension ``D``.
        patches: patch sizes ``[s_1, ..., s_K]``.

    Forward:
        x: (B*L, T, d_u), mask: (B*L, T, 1)
        returns tokens (B*L, T_p, D) and token mask (B*L, T_p, 1).
    """

    def __init__(self, din, dmodel, patches):
        super().__init__()


        self.dmodel = dmodel
        self.patches = patches
        self.din = din

        patch_embs = []
        for p in self.patches:
            patch_embs.append(nn.Conv1d(self.din, self.dmodel, p, stride = p))

        mask_embs = []
        for p in self.patches:
            mask_embs.append(nn.MaxPool1d(p))

        self.patch_embs = nn.ModuleList(patch_embs)
        self.mask_embs = nn.ModuleList(mask_embs)

    def forward(self, x, mask):

        """Embed ``x`` (B*L, T, d_u) with mask (B*L, T, 1); returns tokens (B*L, T_p, D) and token mask (B*L, T_p, 1)."""
        B, T, Din = x.shape
        assert mask.shape == (B, T, 1)

        x_emb = []
        mask_emb = []

        for patch in self.patch_embs:
            x_emb.append(patch(x.permute(0,2,1).contiguous()))

        for mask_aggr in self.mask_embs:
            mask_emb.append(mask_aggr(mask.permute(0,2,1).float()))

        return torch.cat(x_emb, dim = -1).contiguous().permute(0,2,1), torch.cat(mask_emb, dim = -1).contiguous().permute(0,2,1).int()

class TaskEmbeddingCyc(nn.Module):
    """
    Task-Emb: embedding of the task vector ``eta``.

    Categorical entries (form factor, anode, cathode) use ``nn.Embedding``
    and continuous entries (Q_nom and the two voltage limits) use linear
    projections; the six resulting D-dimensional vectors are combined by a
    learned linear mixing over the six features.

    Args:
        task_dim: length of ``eta`` (6).
        dmodel: output dimension.

    Forward:
        task: (..., 6) -> (..., dmodel)
    """

    def __init__(self, task_dim, dmodel):
        super().__init__()
        self.formfactor_emb = nn.Embedding(4, dmodel)
        self.anode_emb = nn.Embedding(4, dmodel)
        self.cathode_emb = nn.Embedding(6, dmodel)
        self.q_emb = nn.Linear(1, dmodel)
        self.vmin_emb = nn.Linear(1, dmodel)
        self.vmax_emb = nn.Linear(1, dmodel)
        self.task_mixer = nn.Linear(task_dim, 1, bias = False)

    def forward(self, task):
        """Embed ``task`` (..., 6) into (..., dmodel)."""
        Dtsk = task.shape[-1]
        task_shape = task.shape
        task = task.view(-1, Dtsk)
        formfactor_emb = self.formfactor_emb(task[:,0].int())
        anode_emb = self.anode_emb(task[:,1].int())
        cathode_emb = self.cathode_emb(task[:,2].int())
        q_emb = self.q_emb(task[:,3:4].float())
        vmin_emb = self.vmin_emb(task[:,4:5].float())
        vmax_emb = self.vmax_emb(task[:,5:].float())
        task_emb = self.task_mixer(torch.cat((formfactor_emb.unsqueeze(1),
                                              anode_emb.unsqueeze(1) ,
                                              cathode_emb.unsqueeze(1) ,
                                              q_emb.unsqueeze(1),
                                              vmin_emb.unsqueeze(1),
                                              vmax_emb.unsqueeze(1)), dim = 1).permute(0,2,1))
        return task_emb.view(*task_shape[:-1], -1)

class FECEmbedding(nn.Module):
    """
    FEC-Emb: embedding of ``gamma_c = [c, Delta_c]``.

    The FEC index ``c`` is embedded with ``nn.Embedding(5000, D)``, the
    duration ``Delta_c`` with a linear projection, and the two vectors are
    combined by a learned linear weighting.

    Forward:
        fec_var: (B, L, 2) -> (B, L, D)
    """
    def __init__(self, fec_dim, dmodel):
        super().__init__()
        self.dmodel = dmodel
        self.fec_dim = fec_dim
        self.fec_dur = nn.Linear(self.fec_dim-1, self.dmodel, bias = False)
        self.fec_id = nn.Embedding(5000, dmodel)
        self.combiner = nn.Linear(fec_dim, 1)

    def forward(self, fec_var):
      """Embed ``fec_var`` (B, L, 2) into (B, L, D)."""
      B, L, Df = fec_var.shape
      fec_id_emb = self.fec_id(fec_var[:,:,0].long())
      fec_dur_emb = self.fec_dur(fec_var[:,:,1:])

      return self.combiner(torch.cat([fec_id_emb.unsqueeze(-1), fec_dur_emb.unsqueeze(-1)], dim = -1)).squeeze(-1)

class OutputRecon(nn.Module):
    """
    Output reconstruction layer (ORL).

    Starting from the coarsest patch tokens, hierarchical ``ConvTranspose1d``
    layers upsample by ``m_k = s_k / s_{k-1}`` and add (residual fusion) the
    tokens of the next finer patch size, until the original resolution is
    reached; an MLP maps each step to ``dout`` outputs.

    Args:
        patches: patch sizes ``[s_1, ..., s_K]`` (same as the encoder).
        dmodel: token dimension ``D``.
        dout: output dimension ``d_v``.

    Forward:
        x: (B*L, T_p, D) decoded patch tokens, T: target length
        returns (B*L, T, dout).
    """

    def __init__(self, patches, dmodel, dout):

        super().__init__()

        self.patches = patches
        self.dmodel = dmodel
        self.num_patches = len(patches)
        self.dout = dout

        self.kernel_sizes = [patches[0]] + [patches[i+1]//patches[i] for i in range(self.num_patches-1)]
        out_expansion = []

        for n in range(self.num_patches):
            out_expansion.append(nn.ConvTranspose1d(in_channels = dmodel, out_channels = dmodel, kernel_size = self.kernel_sizes[n], stride = self.kernel_sizes[n]))

        self.out_layers = nn.ModuleList(out_expansion)

        self.out_proj = nn.Sequential(nn.Linear(dmodel, int(4*dmodel)),
                                      nn.SiLU(),
                                      nn.Linear(int(4*dmodel), dmodel),
                                      nn.SiLU(),
                                      nn.Linear(dmodel, dout))


    def forward(self, x, T):

        """Reconstruct (B*L, T, dout) from decoded patch tokens ``x`` (B*L, T_p, D) and the target length ``T``."""
        B, Tp, Dm = x.shape

        ids = np.cumsum([0]+[T//i for i in self.patches]).astype(int)
        #print(ids)

        for l in range(self.num_patches, 0, -1):
          #print(l, ids[l-1], ids[l])
          if l == self.num_patches:
            out_recon = x[:,ids[l-1]:ids[l]].permute(0,2,1)

          else:
            #print(out_recon.shape)
            #print(out_recon.shape[-1] - ids[l] + ids[l-1])

            pad_len = out_recon.shape[-1] - (ids[l] - ids[l-1])
            #print(out_recon.shape[-1], pad_len, ids[l]-ids[l-1])
            if pad_len < 0:
              padded_out_recon = F.pad(out_recon, (0, abs(pad_len)))
              out_recon = padded_out_recon + x[:,ids[l-1]:ids[l]].permute(0,2,1).contiguous()
              #out_recon = F.pad(out_recon, (0, abs(pad_len))) + x[:,ids[l-1]:ids[l]].permute(0,2,1)
            else:
              out_recon = out_recon.clone() + F.pad(x[:,ids[l-1]:ids[l]].permute(0,2,1), (0, pad_len)).contiguous()

          out_recon = self.out_layers[l-1](out_recon)


          #out_recon = out_recon + self.out_layers[l-1](x[:,ids[l-1]:ids[l]].permute(0,2,1))

        out = self.out_proj(F.pad(out_recon, (0, T-out_recon.shape[-1])).permute(0,2,1))

        return out

class GTGating(nn.Module):
    """
    Optional LSTM-style gating between an old and a new global token.

    With ``gating=True``: ``out = f * g_old + i * tanh(g_new)`` with input
    gate ``i = sigmoid(W_i [g_old, g_new])`` and ``f = tanh(W_f [g_old, g_new])``.
    With ``gating=False`` it returns ``g_new``. Not used by :class:`CycleBlock`.
    """

    def __init__(self, dmodel, gating = False):

        super().__init__()
        self.gating = gating
        if gating:
          self.dmodel = dmodel
          self.igate = nn.Linear(2*self.dmodel, self.dmodel)
          self.fgate = nn.Linear(2*self.dmodel, self.dmodel)
          self.sigmoid = nn.Sigmoid()
          self.tanh = nn.Tanh()

    def forward(self, gt_old, gt_new):

        """Combine ``gt_old`` and ``gt_new`` (same shape); returns ``gt_new`` when gating is disabled."""
        if self.gating:
          gt_total = torch.cat([gt_old, gt_new], dim = -1)
          igate = self.sigmoid(self.igate(gt_total))
          fgate = self.tanh(self.fgate(gt_total))
          gt_out = fgate * gt_old + igate * self.tanh(gt_new)
          return gt_out

        else:
          return gt_new

class PatchEncoder(nn.Module):
    """
    Stack of ``enc_layers`` :class:`~noce_net.transformer.ParallelTransformerBlock`
    layers with residual connections (query-key RMS normalisation, rotary
    position embeddings and multi-query attention; the xPos scaling of the
    rotary embedding is disabled because the attention is non-causal).

    Used as the Patch Encoder and as the Patch Decoder of NOCE-Net.

    Forward:
        x: (B*L, M, D) tokens, mask: (B*L, M) boolean, True = valid
        returns (B*L, M, D) (and attention maps if ``return_attn``).
    """

    def __init__(self,
                 dmodel,
                 enc_layers,
                 num_heads,
                 ff_mult,
                 attn_dropout,
                 ff_dropout,
                 return_attn=False):

        super().__init__()

        self.dmodel = dmodel
        self.num_layers = enc_layers
        self.num_heads = num_heads
        self.ff_mult = ff_mult
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.return_attn = return_attn

        self.encoder = nn.ModuleList()

        for i in range(self.num_layers):

            self.encoder.append(ParallelTransformerBlock(dim = self.dmodel,
                                                                  dim_head = self.dmodel,
                                                                  causal = False,
                                                                  heads = self.num_heads,
                                                                  qk_rmsnorm = True,
                                                                  qk_scale = 8,
                                                                  ff_mult = self.ff_mult,
                                                                  attn_dropout = self.attn_dropout,
                                                                  ff_dropout = self.ff_dropout,
                                                                  use_xpos = True,
                                                                  xpos_scale_base = 512,
                                                                  flash_attn = False))

    def forward(self, x, mask=None):
        """Apply the residual transformer stack to ``x`` (B*L, M, D) with boolean ``mask`` (B*L, M), True = valid."""
        xx = x.clone()
        layers_ops = []

        if self.return_attn:
          attn_maps = []

        if mask is not None:
          mask = mask.clone().contiguous()
        for layer in self.encoder:

          if self.return_attn:
            layer_op, attn = layer(xx, mask = mask)
            attn_maps.append(attn.detach().clone())
          else:
            layer_op = layer(xx, mask = mask)

          layers_ops.append(layer_op.clone())
          xx = (xx.clone() + layers_ops[-1].clone())#.contiguous()
          xx = xx.contiguous()

        if self.return_attn:
          return xx, attn_maps
        else:
          return xx

class FixedEmbedding(nn.Module):
    """Non-trainable sinusoidal embedding table of size (c_in, d_model)."""
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.requires_grad_(False)

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        """Look up the (non-trainable) embeddings of the integer tensor ``x``."""
        return self.emb(x).detach()

class MultiPatchIndexing(nn.Module):
    """
    Intra-cycle positional embedding aligned with the patch tokens.

    The 1-based time index of every step is embedded (learned or fixed
    sinusoidal table) and average-pooled with each patch size, giving one
    positional vector per patch token.

    Forward:
        x: (B*L, T) integer time index -> (B*L, T_p, D)
    """
    def __init__(self, d_model, patches,  max_index = 5000, embed_type='fixed'):
        super(MultiPatchIndexing, self).__init__()
        self.dmodel = d_model
        self.patches = patches
        patch_embs = []
        for p in self.patches:
            patch_embs.append(nn.AvgPool1d(p))
        self.patch_embs = nn.ModuleList(patch_embs)

        self.embedding = FixedEmbedding(max_index, d_model) if embed_type == 'fixed' else nn.Embedding(max_index, d_model)

    def forward(self, x):

        """Embed the time index ``x`` (B*L, T) into one vector per patch token (B*L, T_p, D)."""
        B, T = x.shape[0], x.shape[1]
        x = x.reshape(-1,T)
        #if len(x.shape) > 2:
        #  x_clone = x.clone().reshape(-1, T)
        #  L = x.shape[1]
        #else:
        #  x_clone = x.clone()
        #  L = 1

        unrolled_emb = self.embedding(x)

        index_patch_emb = []

        for patch in self.patch_embs:
            index_patch_emb.append(patch(unrolled_emb.permute(0,2,1).contiguous()))

        return torch.cat(index_patch_emb, dim = -1).permute(0,2,1).reshape(B, -1, self.dmodel)

class CycleBlock(nn.Module):
  """
  NOCE-Net: nested operational context encoder with recurrent backbone.

  For every FEC ``c`` of every battery (processed in parallel):

  1. **Patch Encoder** - inputs are embedded by :class:`MultiPatchEmbedding`
     plus :class:`MultiPatchIndexing`; three aggregator tokens
     ``[e_eta, e_gamma_c, e_lambda]`` (Task-Emb, FEC-Emb and a learnable
     token) are prepended and the sequence is processed by a transformer
     encoder.
  2. **Recurrent backbone** - the three encoded aggregator tokens are
     concatenated (``3D``) and fed to a GRU across FECs whose initial state
     ``h_0`` is a task-conditioned embedding of ``eta``.
  3. **Patch Decoder** - the GRU output of FEC ``c`` (reshaped to three
     tokens) replaces the aggregator tokens, a transformer decoder
     processes the sequence, and :class:`OutputRecon` reconstructs the
     voltage at the original resolution.

  Args:
      configs: ``model_configs`` dictionary (see ``model_configs.yaml``).
  """

  def __init__(self, configs):

        super().__init__()

        self.configs = configs
        self.dmodel = configs['dmodel']
        # Input Data Embedding
        self.data_embedding = MultiPatchEmbedding(configs['din'],
                                                  configs['dmodel'],
                                                  configs['patches'])

        self.index_embedding = MultiPatchIndexing(configs['dmodel'],
                                                  configs['patches'],
                                                  max_index = configs['max_cycle_index'],
                                                  embed_type = configs['index_embed_type'])

        #self.task_emb_for_input = TaskEmbedding(configs['task_dim'],
        #                                    configs['dmodel'])


        # Hidden State Embedding
        self.task_embedding = TaskEmbeddingCyc(configs['task_dim'],
                                            configs['dmodel'])

        self.FEC_Embedding = FECEmbedding(configs['dfec'],
                                          configs['dmodel'])
        #nn.Embedding(configs['max_cycles'],
                             #             configs['dmodel'])
        self.learn_Embedding = nn.Linear(1,
                                         configs['dmodel'], bias = False)

        self.rnn_init = TaskEmbeddingCyc(configs['task_dim'],
                                         configs['dmodel']*3*configs['num_recurrent_layers'])
        # Processing Layers
        # Encoder Layer
        self.patch_encoder = PatchEncoder(configs['dmodel'],
                                          configs['enc_layers'],
                                          configs['num_heads'],
                                          configs['ff_mult'],
                                          configs['attn_dropout'],
                                          configs['ff_dropout'])


        # Recurrent Latent Process Layer
        #self.global_token_gating = GTGating(configs['dmodel'],
        #                                    configs['is_gating'])

        self.gated_rnn = nn.GRU(configs['dmodel']*3,
                                configs['dmodel']*3,
                                configs['num_recurrent_layers'],
                                batch_first = True)


        # Decoder Layer

        self.patch_decoder = PatchEncoder(configs['dmodel'],
                                          configs['dec_layers'],
                                          configs['num_heads'],
                                          configs['ff_mult'],
                                          configs['attn_dropout'],
                                          configs['ff_dropout'])

        self.out_projection = OutputRecon(configs['patches'],
                                          configs['dmodel'],
                                          configs['dout'])

        # RUL Tracker
        if configs['rulm'] == 'MLP':
          self.rul_tracking = Predictor(configs)
        else:
          raise ValueError(f"Unsupported rulm='{configs['rulm']}'. Only 'MLP' is available.")

        self.ckpt = configs['grad_ckpt']


  def forward(self, x, time_index, task, fec_index, mask, glb_tkn = None, is_norm = False):
        """
        Args:
            x: (B, L, T, d_u) intra-cycle inputs.
            time_index: (B, L, T) 1-based time index (0 = padding).
            task: (B, 6) task vector ``eta``.
            fec_index: (B, L, 2) FEC features ``gamma_c``.
            mask: (B, L, T) validity mask of the time steps.
            glb_tkn: optional GRU initial state (num_recurrent_layers, B, 3D);
                when ``None`` the task-conditioned ``h_0`` is used. Training
                passes the detached final state of the previous chunk of
                FECs here (truncated back-propagation through time).
            is_norm: forwarded to the auxiliary head (unused).

        Returns:
            V_hat: (B, L, T, d_v) predicted voltage.
            rul_pred: (B, L, 1) output of the auxiliary capacity head.
            H: (B, L+1, 3D) recurrent states ``[h_0, h_1, ..., h_L]``
               (first layer of ``h_0`` followed by the GRU outputs).
            h_last: (num_recurrent_layers, B, 3D) final GRU state.
        """

        B, L, T, Din = x.shape
        _, Dtsk = task.shape
        #print(torch.numel(x))
        # fec_index.shape = B, L
        # time_index.shape = B, L, T

        # Formation of the initial state
        gt_task_init = self.task_embedding(task).unsqueeze(1).repeat(1, L, 1) # B 1 D
        gt_cycle_init = self.FEC_Embedding(fec_index) # B L D
        gt_learn_init = self.learn_Embedding(torch.ones(B, L, 1, device = x.device)) # B L D

        gt_init = torch.stack([gt_task_init, gt_cycle_init, gt_learn_init], dim = 2) # B L 3 D

        # Formation of the data embedding
        xemb, mask_emb = self.data_embedding(x.reshape(B*L, T, -1), mask.reshape(B*L, T, -1))  # (B*L, Tp, D), (B*L, Tp, 1)
        xemb = (xemb + self.index_embedding(time_index.reshape(-1, T))).reshape(B, L, -1, self.dmodel) # B, L, Tp, D

        # Combine data embedding and initial state for processing through encoder
        gt_xemb = torch.cat([gt_init, xemb.clone()], dim = 2) # B, L, Tp+3, D

        # Encoder output
        mask_state = torch.ones(B*L, 3, 1, device = x.device)
        total_mask = torch.cat([mask_state, mask_emb], dim = 1).int().bool() # B*L, Tp+3, 1

        total_out = self.patch_encoder(gt_xemb.clone().reshape(int(B*L), -1, self.dmodel), total_mask.squeeze(2)).reshape(B, L, -1, self.dmodel) # B*L, Tp+3, D

        # First 3 along time dimension are states for recurrence, rest are processed inputs
        updated_gt, updated_inps = total_out[:,:,:3].clone(), total_out[:,:,3:].clone()
        task = task.requires_grad_()

        # Initialize GRU
        if glb_tkn is not None:
          rnn_h0 = glb_tkn
        else:
          rnn_h0 = self.rnn_init(task).reshape(B,self.configs['num_recurrent_layers'], -1).permute(1,0,2).contiguous()
        updated_gt = updated_gt.requires_grad_()
        rnn_h0 = rnn_h0.contiguous().requires_grad_()

        # GRU Operation
        if self.ckpt:
          recurrent_gt, final_states = torch.utils.checkpoint.checkpoint(self.gated_rnn, updated_gt.reshape(B, L, -1), rnn_h0) # B, L , 3*D
        else:
          recurrent_gt, final_states = self.gated_rnn(updated_gt.reshape(B, L, -1), rnn_h0) # B, L , 3*D

        # Decoder Operation
        decoder_out = self.patch_decoder(torch.cat([recurrent_gt.reshape(B, L, 3, -1), updated_inps], dim = 2).reshape(B*L, -1, self.dmodel), total_mask.squeeze(2))


        # Final output with length similar to input
        final_output = self.out_projection(decoder_out[:,3:], T)

        # RUL Prediction using recurrent states
        all_states = torch.cat([rnn_h0[0].unsqueeze(1),recurrent_gt], dim = 1)
        rul_pred = self.rul_tracking(all_states, is_norm)

        return final_output.reshape(B, L, T, -1), rul_pred.reshape(B, L, -1), all_states, final_states

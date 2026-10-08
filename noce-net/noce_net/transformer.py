"""
noce_net.transformer
====================
Transformer block used by the Patch Encoder and Patch Decoder of NOCE-Net.

The code (``Attention``, ``RotaryEmbedding`` with xPos, ``SwiGLU`` and
``ParallelTransformerBlock``) is adapted from ``palm_rlhf_pytorch``
(https://github.com/lucidrains/PaLM-rlhf-pytorch, MIT License,
Copyright (c) 2022 Phil Wang). ``ParallelTransformerBlock`` computes
attention and the SwiGLU feed-forward branch in parallel from a single
pre-normalised input, with optional query-key RMS normalisation, rotary
position embeddings (with xPos scaling only for causal attention) and
multi-query attention (a single key/value head shared by all query heads).
"""
import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from torch.nn import Module
from einops import rearrange
from collections import namedtuple
from functools import wraps
from packaging import version



Config = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])
def exists(val):
    """True if ``val`` is not None."""
    return val is not None

def once(fn):
    """Wrap a one-argument function so that it is executed only on its first call."""
    called = False
    @wraps(fn)
    def inner(x):
        """Call the wrapped function the first time only."""
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

class Attention(Module):
    """Scaled dot-product attention with multi-query keys/values (q: (b, h, n, d); k, v: (b, n, d)), optional key-padding mask, causal mask and PyTorch flash attention."""
    def __init__(
        self,
        dropout = 0.,
        causal = False,
        use_flash_attn = False
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.register_buffer("mask", None, persistent=False)

        self.use_flash_attn = use_flash_attn
        assert not (use_flash_attn and version.parse(torch.__version__) < version.parse('2.0.0')), 'flash attention requires PyTorch >= 2.0'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash_attn:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = Config(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = Config(False, True, True)

    def get_mask(self, n, device):
        """Upper-triangular causal mask of size (n, n), cached as a buffer."""
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def flash_attn(self, q, k, v, mask = None):
        """Attention through ``torch.nn.functional.scaled_dot_product_attention``."""
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)
        v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0.,
                is_causal = self.causal
            )

        return out

    def forward(self, q, k, v, mask = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device = q.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        if self.use_flash_attn:
            return self.flash_attn(q, k, v, mask = mask)

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k) * scale

        # key padding mask

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal mask

        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        return out


def default(val, d):
    """Return ``val`` if it is not None, otherwise ``d``."""
    return val if exists(val) else d

def identity(t, *args, **kwargs):
    """Return ``t`` unchanged."""
    return t

def l2norm(t):
    """Normalise ``t`` to unit L2 norm along the last dimension."""
    return F.normalize(t, dim = -1)

class LayerNorm(nn.Module):
    """Layer normalisation with a learnable gain (initialised at 1) and a fixed zero bias."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        """Normalise ``x`` over its last dimension."""
        return F.layer_norm(x, x.shape[-1:], (self.gamma + 1), self.beta)

# residual


class Residual(nn.Module):
    """Residual wrapper ``x + fn(x)``."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):

        """Return ``fn(x) + x`` (in place when no gradients are required)."""
        y = self.fn(x, **kwargs)
        n = y.shape[-2]
        if not any([t.requires_grad for t in (x, y)]):
            return x.add_(y)

        return y + x[:,:n]

# rotary positional embedding w/ xpos
# https://arxiv.org/abs/2104.09864
# https://arxiv.org/abs/2212.10554v1

class RotaryEmbedding(nn.Module):
    """Rotary position embedding with optional xPos length-extrapolation scaling."""
    def __init__(self, dim, scale_base = 512, use_xpos = True):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.use_xpos = use_xpos
        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale)

    def forward(self, seq_len, device):
        """Return the rotation frequencies (seq_len, dim) and the xPos scale for ``seq_len`` positions."""
        t = torch.arange(seq_len, device = device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        if not self.use_xpos:
            return freqs, torch.ones(1, device = device)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale

def rotate_half(x):
    """Rotate the two halves of the last dimension: ``(x1, x2) -> (-x2, x1)``."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t, scale = 1.):
    """Apply the rotary embedding ``pos`` (with scale) to ``t``."""
    return (t * pos.cos() * scale) + (rotate_half(t) * pos.sin() * scale)


# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202


class SwiGLU(Module):
    """SwiGLU activation: ``silu(g) * x`` with ``(x, g)`` the two halves of the last dimension."""
    def forward(self, x):
        """(..., 2F) -> (..., F)."""
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame


class ParallelTransformerBlock(Module):
    """Pre-norm transformer block computing multi-query attention and a SwiGLU feed-forward branch in parallel from one fused projection; the two outputs are summed. ``mask`` (b, n) marks valid tokens with True."""
    def __init__(
        self,
        dim,
        dim_head = 128,
        causal = True,
        heads = 8,
        qk_rmsnorm = False,
        qk_scale = 8,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        use_xpos = True,
        xpos_scale_base = 512,
        flash_attn = False,
    ):
        super().__init__()
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

        self.qk_rmsnorm = qk_rmsnorm

        if qk_rmsnorm:
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.attend = Attention(
            causal = causal,
            dropout = attn_dropout,
            use_flash_attn = flash_attn
        )

        self.heads = heads
        self.scale = (dim_head ** -0.5) if not qk_rmsnorm else qk_scale
        self.causal = causal

        self.rotary_emb = RotaryEmbedding(dim_head, scale_base = xpos_scale_base, use_xpos = use_xpos and causal)

        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)

        self.flash_attn = flash_attn
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.flash_attn_dropout = attn_dropout

        # parallel feedforward tail

        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

        # for caching causal mask and rotary embeddings

        self.register_buffer("pos_emb", None, persistent=False)
        self.register_buffer("pos_emb_scale", None, persistent=False)

    def get_rotary_embedding(self, n, device):
        """Rotary embeddings for ``n`` positions, cached as buffers."""
        if exists(self.pos_emb) and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n], self.pos_emb_scale[:n]

        pos_emb, scale = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        self.register_buffer("pos_emb_scale", scale, persistent=False)
        return pos_emb, scale

    def forward(
        self,
        x,
        att_split_at = None,
        mask = None,
        finetune_modules = None
    ):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device, h = x.shape[1], x.device, self.heads
        if att_split_at is None:
            att_split_at = n
        # pre layernorm

        x = self.norm(x)


        # attention queries, keys, values, and feedforward inner

        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)
        #q = x[:,:att_split_at]

        # finetune loras

        #lora_q = lora_k = lora_v = lora_o = None

        #if exists(finetune_modules):
        #    lora_q, lora_k, lora_v, lora_o = finetune_modules
        #    q = q + lora_q(x)
        #    k = k + lora_k(x)
        #    v = v + lora_v(x)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # qk rmsnorm

        if self.qk_rmsnorm:
            q, k = map(l2norm, (q, k))
            q = q * self.q_scale
            k = k * self.k_scale

        # rotary embeddings with xpos decay for better length extrapolation

        positions, scale = self.get_rotary_embedding(n, device)

        q = apply_rotary_pos_emb(positions, q, scale)
        k = apply_rotary_pos_emb(positions, k, scale ** -1)

        # attention function, either regular or flash

        out = self.attend(q[:,:,:att_split_at], k, v, mask = mask)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")

        attn_out = self.attn_out(out)

        ff_out = self.ff_out(ff[:,:att_split_at])

        ##if exists(lora_o):
        #    attn_out = attn_out + lora_o(out)

        return attn_out + ff_out
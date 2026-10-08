"""
ablation.embedding_ablation
===========================
Embedding ablation of NOCE-Net (rows "wo Task Emb.", "wo FEC Emb." and
"wo Learn Emb." of the ablation table).

:class:`CycleBlockEmbeddingAblation` is NOCE-Net with one of the three
aggregator tokens removed, selected by ``model_configs['leave_emb']``:

* ``'task'``  - no Task-Emb token; the GRU initial state is a learned
  vector that does not depend on ``eta``.
* ``'fec'``   - no FEC-Emb token.
* ``'learn'`` - no learnable aggregator token.
* ``None``    - all three tokens (same structure as
  :class:`noce_net.layers.CycleBlock`, without the auxiliary capacity head).

With one token removed the recurrent state has size ``d_h = 2 D`` instead
of ``3 D``. The forward pass returns ``(V_hat, H, h_last)``.

:func:`run_embedding_ablation` trains every variant from scratch for the
requested number of independent runs and evaluates each with
:func:`test_datawise` on the in-sample and zero-shot splits.
"""
import csv
import gc
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from noce_net.layers import MultiPatchIndexing, MultiPatchEmbedding, TaskEmbeddingCyc, FECEmbedding, PatchEncoder, OutputRecon
from noce_net.losses import compute_cyc_loss


class CycleBlockEmbeddingAblation(nn.Module):
  """
  NOCE-Net with an optional aggregator token removed.

  Args:
      configs: ``model_configs``; ``configs['leave_emb']`` in
          ``{None, 'task', 'fec', 'learn'}`` selects the removed token.

  Attributes:
      nstates_: number of aggregator tokens per FEC (3 or 2).
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

        self.leave_emb = configs['leave_emb']

        if self.leave_emb is None:
          # Hidden State Embedding
          self.task_embedding = TaskEmbeddingCyc(configs['task_dim'],
                                            configs['dmodel'])

          self.FEC_Embedding = FECEmbedding(configs['dfec'],
                                          configs['dmodel'])
          #nn.Embedding(configs['max_cycles'],
                             #             configs['dmodel'])
          self.learn_Embedding = nn.Linear(1,
                                         configs['dmodel'], bias = False)
          self.nstates_ = 3

          self.rnn_init = TaskEmbeddingCyc(configs['task_dim'],
                                         configs['dmodel']*self.nstates_*configs['num_recurrent_layers'])

        elif self.leave_emb == 'task':
          self.FEC_Embedding = FECEmbedding(configs['dfec'],
                                          configs['dmodel'])
          #nn.Embedding(configs['max_cycles'],
                             #             configs['dmodel'])
          self.learn_Embedding = nn.Linear(1,
                                         configs['dmodel'], bias = False)
          self.nstates_ = 2
          self.rnn_init = nn.Linear(1, configs['dmodel']*self.nstates_*configs['num_recurrent_layers'])

        elif self.leave_emb == 'fec':
          self.task_embedding = TaskEmbeddingCyc(configs['task_dim'],
                                            configs['dmodel'])
                             #             configs['dmodel'])
          self.learn_Embedding = nn.Linear(1,
                                         configs['dmodel'], bias = False)
          self.nstates_ = 2
          self.rnn_init = TaskEmbeddingCyc(configs['task_dim'],
                                         configs['dmodel']*self.nstates_*configs['num_recurrent_layers'])
        elif self.leave_emb == 'learn':
          self.task_embedding = TaskEmbeddingCyc(configs['task_dim'],
                                            configs['dmodel'])

          self.FEC_Embedding = FECEmbedding(configs['dfec'],
                                          configs['dmodel'])

          self.nstates_ = 2

          self.rnn_init = TaskEmbeddingCyc(configs['task_dim'],
                                         configs['dmodel']*self.nstates_*configs['num_recurrent_layers'])
        else:
          raise NotImplementedError

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

        self.gated_rnn = nn.GRU(configs['dmodel']*self.nstates_,
                                configs['dmodel']*self.nstates_,
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


        self.ckpt = configs['grad_ckpt']


  def forward(self, x, time_index, task, fec_index, mask, glb_tkn = None, is_norm = False):
        """
        Same inputs as :meth:`noce_net.layers.CycleBlock.forward`.

        Returns:
            V_hat: (B, L, T, d_v) predicted voltage.
            H: (B, L+1, nstates_ * D) recurrent states including ``h_0``.
            h_last: (num_recurrent_layers, B, nstates_ * D) final GRU state.
        """

        B, L, T, Din = x.shape
        _, Dtsk = task.shape
        #print(torch.numel(x))
        # fec_index.shape = B, L
        # time_index.shape = B, L, T

        # Formation of the initial state
        if self.leave_emb == 'task':
          #gt_task_init = self.task_embedding(task).unsqueeze(1).repeat(1, L, 1) # B 1 D
          gt_cycle_init = self.FEC_Embedding(fec_index) # B L D
          gt_learn_init = self.learn_Embedding(torch.ones(B, L, 1, device = x.device)) # B L D

          gt_init = torch.stack([gt_cycle_init, gt_learn_init], dim = 2) # B L 3 D
        elif self.leave_emb == 'fec':
          gt_task_init = self.task_embedding(task).unsqueeze(1).repeat(1, L, 1) # B 1 D
          gt_learn_init = self.learn_Embedding(torch.ones(B, L, 1, device = x.device)) # B L D

          gt_init = torch.stack([gt_task_init, gt_learn_init], dim = 2) # B L 3 D
        elif self.leave_emb == 'learn':

          gt_task_init = self.task_embedding(task).unsqueeze(1).repeat(1, L, 1) # B 1 D
          gt_cycle_init = self.FEC_Embedding(fec_index) # B L D

          gt_init = torch.stack([gt_task_init, gt_cycle_init], dim = 2) # B L 3 D
        elif self.leave_emb is None:

          gt_task_init = self.task_embedding(task).unsqueeze(1).repeat(1, L, 1) # B 1 D
          gt_cycle_init = self.FEC_Embedding(fec_index) # B L D
          gt_learn_init = self.learn_Embedding(torch.ones(B, L, 1, device = x.device)) # B L D
          gt_init = torch.stack([gt_task_init, gt_cycle_init, gt_learn_init], dim = 2) # B L 3 D

        # Formation of the data embedding
        xemb, mask_emb = self.data_embedding(x.reshape(B*L, T, -1), mask.reshape(B*L, T, -1))  # (B*L, Tp, D), (B*L, Tp, 1)
        xemb = (xemb + self.index_embedding(time_index.reshape(-1, T))).reshape(B, L, -1, self.dmodel) # B, L, Tp, D

        # Combine data embedding and initial state for processing through encoder
        gt_xemb = torch.cat([gt_init, xemb.clone()], dim = 2) # B, L, Tp+3, D
        #gt_xemb = xemb.clone()

        # Encoder output
        mask_state = torch.ones(B*L, self.nstates_, 1, device = x.device)
        total_mask = torch.cat([mask_state, mask_emb], dim = 1).int().bool() # B*L, Tp+3, 1
        #total_mask = mask_emb.int().nool()

        total_out = self.patch_encoder(gt_xemb.clone().reshape(int(B*L), -1, self.dmodel), total_mask.squeeze(2)).reshape(B, L, -1, self.dmodel) # B*L, Tp+3, D

        # First 3 along time dimension are states for recurrence, rest are processed inputs
        updated_gt, updated_inps = total_out[:,:,:self.nstates_].clone(), total_out[:,:,self.nstates_:].clone()
        task = task.requires_grad_()

        # Initialize GRU
        if glb_tkn is not None:
          rnn_h0 = glb_tkn
        else:
          if self.leave_emb != 'task':
            rnn_h0 = self.rnn_init(task).reshape(B,self.configs['num_recurrent_layers'], -1).permute(1,0,2).contiguous()
          else:
            rnn_h0 = self.rnn_init(torch.ones(B, 1, device = x.device)).reshape(B,self.configs['num_recurrent_layers'], -1).permute(1,0,2).contiguous()
        updated_gt = updated_gt.requires_grad_()
        rnn_h0 = rnn_h0.contiguous().requires_grad_()

        # GRU Operation
        if self.ckpt:
          recurrent_gt, final_states = torch.utils.checkpoint.checkpoint(self.gated_rnn, updated_gt.reshape(B, L, -1), rnn_h0) # B, L , 3*D
        else:
          recurrent_gt, final_states = self.gated_rnn(updated_gt.reshape(B, L, -1), rnn_h0) # B, L , 3*D

        # Decoder Operation
        decoder_out = self.patch_decoder(torch.cat([recurrent_gt.reshape(B, L, self.nstates_, -1), updated_inps], dim = 2).reshape(B*L, -1, self.dmodel), total_mask.squeeze(2))
        #decoder_out = self.patch_decoder(total_out.reshape(B*L, -1, self.dmodel), total_mask.squeeze(2))


        # Final output with length similar to input
        final_output = self.out_projection(decoder_out[:,self.nstates_:], T)
        all_states = torch.cat([rnn_h0[0].unsqueeze(1),recurrent_gt], dim = 1)
        

        return final_output.reshape(B, L, T, -1), all_states, final_states 


class BatteryModelEmbeddingAblation(nn.Module):
  """Container exposing :class:`CycleBlockEmbeddingAblation` as ``cycle_model``."""
  def __init__(self, configs):
    super().__init__()
    self.configs = configs
    self.cycle_model = CycleBlockEmbeddingAblation(configs)


def learn_n_evaluate(model,
                     dataloader,
                     mode,
                     log_path,
                     epoch,
                     max_epoch,
                     model_configs,
                     optimizer = None,
                     scheduler = None,
                     cap_optimizer = None,
                     cap_scheduler = None,
                     joint_optimizer = None,
                     joint_scheduler = None,
                     beta = 0.05,
                     beta_prop = 'constant',
                     wflag = False,
                     is_norm = False):
  """
  One training or validation epoch for :class:`BatteryModelEmbeddingAblation`.

  Identical to :func:`train.train_response.learn_n_evaluate` except that the
  model output is unpacked as ``(V_hat, H, h_last)``. See that function for
  the arguments and the files written.
  """

  device = model_configs['device']
  cyc_loss_epoch = []
  quantile_loss_epoch = []
  diff_loss_epoch = []
  spike_loss_epoch = []
  step_index = []
  align_loss_epoch = []


  if mode == 'train':
    log_path = os.path.join(log_path, 'train')
    if not os.path.exists(log_path):
      os.makedirs(log_path)

  elif mode == 'val':
    log_path = os.path.join(log_path, 'val')
    if not os.path.exists(log_path):
      os.makedirs(log_path)

    if not os.path.exists(os.path.join(log_path, 'best_model_loss.pkl')):
      best_loss_dict = {'cycle':[np.inf],
                        'joint_prod':[np.inf],
                        'joint_sum':[np.inf],
                        'capacity':[np.inf]}
      with open(os.path.join(log_path, 'best_model_loss.pkl'), 'wb') as f:
        pickle.dump(best_loss_dict, f)

  cyc_step = 0

  if mode == 'train':
        if beta_prop == 'linear':
          if epoch % 10 == 0:
            beta += (epoch//10) * (beta/(max_epoch//10 -1))
            print(f'Beta changed to {beta}')
  elif beta_prop == 'constant':
          beta = beta

  for data_id, data in enumerate(dataloader):

      batch_cyc_loss = 0
      batch_quantile_loss = 0
      batch_diff_loss = 0
      batch_spike_loss = 0
      batch_align_loss = 0

      inp_batch, out_batch, cap_batch, task_batch, index_batch, fec_batch, cyc_mask_batch, cap_mask_batch, wbatch, dname_batch = data

      #for b in range(rul_batch.shape[0]):
      #  if torch.all(rul_batch[b] == 0):
      #    cap_mask_batch[b] = 0

      inp_batch = torch.nan_to_num(inp_batch, nan = 0.0, posinf = 0.0, neginf = 0.0)
      Nmax = inp_batch.shape[1]

      print('Bats: ', dname_batch, 'Weights: ', wbatch)

      seq_len = 0
      cycle_breaks = [seq_len]
      while seq_len < Nmax:
          rint = 100
          seq_len = seq_len + rint
          cycle_breaks.append(min(seq_len, Nmax))

      cycle_breaks = torch.tensor(cycle_breaks)
      cycle_splits = (cycle_breaks[1:] - cycle_breaks[:-1]).int().numpy().tolist()

      if cycle_splits[-1] < 50:
          last_split = cycle_splits.pop(-1)
          divide_splits = last_split//len(cycle_splits)
          rem_splits = last_split % len(cycle_splits)
          for i in range(len(cycle_splits)):
            cycle_splits[i] = cycle_splits[i] + divide_splits
          cycle_splits[-1] = cycle_splits[-1] + rem_splits

      print(cycle_splits, inp_batch.shape[2])
      cycle_split_cumsum = [0] + np.cumsum(cycle_splits).tolist()

      inp_batch_split = torch.split(torch.nan_to_num(inp_batch), cycle_splits, dim=1)
      op_batch_split = torch.split(torch.nan_to_num(out_batch), cycle_splits, dim=1)
      index_batch_split = torch.split(torch.nan_to_num(index_batch), cycle_splits, dim=1)
      fec_batch_split = torch.split(torch.nan_to_num(fec_batch), cycle_splits, dim=1)
      cyc_mask_split = torch.split(torch.nan_to_num(cyc_mask_batch), cycle_splits, dim=1)



      step_loss = 0
      gtoken = None


      for cid, c in enumerate(cycle_splits):

          if torch.numel(inp_batch_split[cid]) > 0:

            cyc_pred, states, final_states = model.cycle_model(inp_batch_split[cid].to(device),
                                                                         index_batch_split[cid].to(device),
                                                                         task_batch.to(device),
                                                                         fec_batch_split[cid].to(device),
                                                                         cyc_mask_split[cid].to(device),
                                                                         gtoken,
                                                                         is_norm
                                                                         )

            Qc_inp = inp_batch_split[cid][:,:,:,2:3]
            Qd_inp = inp_batch_split[cid][:,:,:,3:4]

            #cap_pred = model.cap_model(states, is_norm)
            if final_states is not None:
                gtoken = final_states.clone().detach().to(device)  # Ensuring out-of-place operation
            else:
                gtoken = None

            cyc_loss, quantile_cyc_loss, diff_loss, spike_loss, align_loss = compute_cyc_loss(cyc_pred.clone(),
                                                                                              op_batch_split[cid].to(device),
                                                                                              Qc_inp.to(device),
                                                                                              Qd_inp.to(device),
                                                                                              cyc_mask_split[cid].to(device),
                                                                                              model_configs['align_loss_type'],
                                                                                              n_proj = 64)

            
            if mode == 'train':
              if model_configs['training_mode'] == 'cycle':
                optimizer.zero_grad()

                cyc_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.cycle_model.parameters(), max_norm= 1.0)
                optimizer.step()

              else:
                pass

            else:
              pass


            batch_cyc_loss += cyc_loss.detach()/len(cycle_splits)
            batch_quantile_loss += quantile_cyc_loss/len(cycle_splits)
            batch_diff_loss += diff_loss/len(cycle_splits)
            batch_spike_loss += spike_loss/len(cycle_splits)
            batch_align_loss += align_loss/len(cycle_splits)

            del cyc_pred, states, cyc_loss, quantile_cyc_loss, diff_loss, spike_loss
            torch.cuda.empty_cache()
            gc.collect()

            cyc_step += 1

      cyc_loss_epoch.append(batch_cyc_loss.item())
      quantile_loss_epoch.append(batch_quantile_loss.item())
      diff_loss_epoch.append(batch_diff_loss.item())
      spike_loss_epoch.append(batch_spike_loss.item())
      align_loss_epoch.append(batch_align_loss.item())

      step_index.append(data_id)

      print('Dataset name:', dname_batch,
            'Cycle Loss:', batch_cyc_loss.item(),
            'Quantile Loss:', batch_quantile_loss.item(),
            'Diff Loss:', batch_diff_loss.item(),
            'Spike Loss:', batch_spike_loss.item(),
            'Align Loss:', batch_align_loss.item())

      print(f'Complete step {data_id}/{len(dataloader)} of epoch {epoch}')

  # Schduler Update
  if mode == 'train':
      if scheduler is not None:
        if model_configs['training_mode'] == 'cycle':
          scheduler.step()
          print(f'Cycle mode learning rate changed to: {scheduler.get_last_lr()}')


  # Model saving and best loss checkpointing
  if mode == 'val':

    with open(os.path.join(log_path, 'best_model_loss.pkl'), 'rb') as f:
        curr_model_loss = pickle.load(f)

    if np.mean(cyc_loss_epoch) <= curr_model_loss['cycle'][-1]:
        curr_model_loss['cycle'].append(np.mean(cyc_loss_epoch))
        torch.save(model.state_dict(), os.path.join(log_path, 'best_cyc_model.pth'))
        print('Cyc Best Model saved')
        with open(os.path.join(log_path, 'best_model_loss.pkl'), 'wb') as f:
          pickle.dump(curr_model_loss, f)
    #else:


  # Training logs
  print(f'Epoch {epoch} complete')
  print(f'Cycle Loss: {np.mean(cyc_loss_epoch)}')
  print(f'Quantile Loss: {np.mean(quantile_loss_epoch)}')
  print(f'Diff Loss: {np.mean(diff_loss_epoch)}')
  print(f'Spike Loss: {np.mean(spike_loss_epoch)}')
  print(f'Align Loss: {np.mean(align_loss_epoch)}')

  epoch_index = [epoch]*len(step_index)
  data_cyc = np.column_stack([epoch_index, step_index, cyc_loss_epoch, quantile_loss_epoch, diff_loss_epoch, spike_loss_epoch, align_loss_epoch])
  columns = ['Epoch', 'Step', 'Cycle Loss', 'Quantile Loss', 'Diff Loss', 'Spike Loss', 'Align Loss']

  cycle_log_file = mode + '_cycle.csv'

  if not os.path.exists(os.path.join(log_path , cycle_log_file)):
      df = pd.DataFrame(data_cyc, columns = columns)
      df.to_csv(os.path.join(log_path , cycle_log_file), index = False)

  else:
      data_list = [d.tolist() for d in data_cyc]

      with open(os.path.join(log_path , cycle_log_file), "a") as stem_data:
              w = csv.writer(stem_data, delimiter=',')
              w.writerows(data_list)
              stem_data.close()
      del data_list
  del data_cyc


def test_datawise(model,
         datapath,
         log_path,
         test_dataloader,
         test_log_file = 'test_res_log.csv',
         task_type = 'class',
         beta = 0.1,
         is_norm = False,
         ):
  """
  Per-dataset evaluation of an ablated model (batch size 1).

  Loads ``<log_path>/val/best_cyc_model.pth`` if present and computes per
  source dataset ``[MAE, MAPE, sMAPE, MAPE(V > 1e-4)]`` (in %). The ``'Total'``
  entry is computed over all batteries and holds
  ``[MAE, MAPE(V > 1e-4), sMAPE, MAPE(V > 1e-4)]``; unlike the per-dataset
  entries, its second element is the thresholded MAPE and its sMAPE is a
  fraction (not multiplied by 100). Voltages ``V`` are the normalised targets.

  The result is written to ``res_datawise.csv`` if that file does not exist
  yet, otherwise to ``res_datawise_test.csv`` (first call: in-sample split,
  second call: zero-shot split).

  Args:
      model: :class:`BatteryModelEmbeddingAblation`.
      datapath, test_log_file, task_type, beta: unused.
      log_path: run directory.
      test_dataloader: dataloader with batch size 1.
      is_norm: forwarded to the model.

  Returns:
      dict mapping dataset names and ``'Total'`` to the metric lists.
  """

  device = next(model.parameters()).device
  if os.path.exists(os.path.join(log_path, 'best_test_loss.pkl')):
    with open(os.path.join(log_path, 'best_test_loss.pkl'), 'rb') as f:
      best_test_losses = pickle.load(f)
    #best_test_loss = best_test_losses[-1]
  else:
    best_test_losses = [np.inf]
  #best_test_loss = best_test_losses[-1]

  model_paths = [os.path.join(log_path, 'val', 'best_cyc_model.pth'),
                 ]

  all_model_tests = np.inf
  best_test_model_state_dict = None

  for mid, model_path in enumerate(model_paths):

              if os.path.exists(model_path):
                    loaded_state_dict = torch.load(model_path, map_location = device)
                    model.load_state_dict(loaded_state_dict)
                    print('Loaded saved model for testing!!')

              cyc_preds = []
              cyc_trues = []
              cyc_masks = []
              cyc_abs = []
              dataset_names = []


              model.eval()
              with torch.no_grad():
                  for data_id, data in enumerate(test_dataloader):

                      print('Starting idx:', data_id,'  of ', len(test_dataloader))


                      inp_batch, out_batch, cap_batch, task_batch, index_batch, fec_batch, cyc_mask_batch, cap_mask_batch, wbatch, dname_batch = data
                      inp_batch = torch.nan_to_num(inp_batch, nan = 0.0, posinf = 0.0, neginf = 0.0)
                      Nmax = inp_batch.shape[1]


                      cyc_pred, states, final_states = model.cycle_model(inp_batch.to(device),
                                                                                   index_batch.to(device),
                                                                                   task_batch.to(device),
                                                                                   fec_batch.to(device),
                                                                                   cyc_mask_batch.to(device),
                                                                                   None,
                                                                                   is_norm
                                                                                   )


                      cyc_preds.append(cyc_pred.detach().reshape(1,-1).cpu().numpy())
                      cyc_trues.append(out_batch.reshape(1,-1).cpu().numpy())
                      cyc_masks.append(cyc_mask_batch.reshape(1,-1).cpu().numpy())
                      dataset_names.append(dname_batch[0])


                      #cyc_loss, quantile_cyc_loss, diff_loss, spike_loss = compute_cyc_loss(cyc_pred.clone(),
                      #                                                                      out_batch.to(device),
                      #                                                                      cyc_mask_batch.to(device))


                  cyc_ae_list = [np.abs(cyc_preds[i] - cyc_trues[i]) * cyc_masks[i] for i in range(len(cyc_preds))]
                  cyc_ape_list = [cyc_ae_list[i]/(np.abs(cyc_trues[i])*cyc_masks[i] + 1e-3) for i in range(len(cyc_preds))]
                  cyc_sape_list = [2*cyc_ae_list[i]/((np.abs(cyc_trues[i])+np.abs(cyc_preds[i]))*cyc_masks[i] + 1e-3) for i in range(len(cyc_preds))]

                  data_names_new_list = []
                  for u in dataset_names:
                    if u.startswith('Stanford'):
                      data_names_new_list.append('Stanford')
                    else:
                      data_names_new_list.append(u)
                  unique_datasets = np.unique(data_names_new_list)
                  datanames_np = np.array(data_names_new_list)

                  error_datasets = {}
                  for u in unique_datasets:
                    uidxs = np.where(datanames_np == u)[0]
                    cyc_umask = np.concatenate([cyc_masks[i].reshape(-1,1) for i in uidxs], axis = 0).reshape(-1)
                    cyc_umae = np.concatenate([cyc_ae_list[i].reshape(-1,1) for i in uidxs], axis = 0).reshape(-1)
                    cyc_uape = np.concatenate([cyc_ape_list[i].reshape(-1,1) for i in uidxs], axis = 0).reshape(-1)
                    cyc_usape = np.concatenate([cyc_sape_list[i].reshape(-1,1) for i in uidxs], axis = 0).reshape(-1)
                    cyc_utrues =  np.concatenate([cyc_trues[i].reshape(-1,1) for i in uidxs], axis = 0).reshape(-1)

                    mape_corr_cyc = np.sum(cyc_uape[cyc_utrues > 1e-4])/np.sum(cyc_umask[cyc_utrues > 1e-4]) * 100
                    mape_cyc = (np.sum(cyc_uape)/np.sum(cyc_umask)) * 100
                    mae_cyc = np.sum(cyc_umae)/np.sum(cyc_umask)
                    smapes_cyc = (np.sum(cyc_usape)/np.sum(cyc_umask)) * 100
                    error_datasets[u] = [ mae_cyc, mape_cyc, smapes_cyc, mape_corr_cyc]
                    print(u, mape_corr_cyc)


                  cyc_true_np = np.concatenate(cyc_trues, axis = -1)
                  cyc_pred_np = np.concatenate(cyc_preds, axis = -1)
                  cyc_masks_np = np.concatenate(cyc_masks, axis = -1)
                  cyc_abs_errs_np = np.abs(cyc_true_np - cyc_pred_np)*cyc_masks_np
                  cyc_ape_np = cyc_abs_errs_np/(np.abs(cyc_true_np)*cyc_masks_np + 1e-3)
                  cyc_sape_np = (2*cyc_abs_errs_np) / ((np.abs(cyc_true_np)*cyc_masks_np + np.abs(cyc_pred_np)*cyc_masks_np) + 1e-3)

                  #cyc_sape_np_batwise =
                  #all_cycs_test_data = np.column_stack((cyc_pred_np, cyc_true_np, cyc_masks_np, cyc_abs_errs_np, cyc_ape_np))

                  mape_cyc_total = (np.sum(cyc_ape_np[cyc_true_np > 1e-4])/np.sum(cyc_masks_np[cyc_true_np > 1e-4])) * 100
                  mae_cyc_total = np.sum(cyc_abs_errs_np)/np.sum(cyc_masks_np)
                  smape_cyc_total = np.sum(cyc_sape_np)/np.sum(cyc_masks_np)

                  error_datasets['Total'] = [mae_cyc_total, mape_cyc_total, smape_cyc_total, mape_cyc_total]

                  #############################################################
                  df = pd.DataFrame.from_dict(error_datasets)
                  if not os.path.exists(os.path.join(log_path , 'res_datawise.csv')):
                      df.to_csv(os.path.join(log_path , 'res_datawise.csv'), index = False)
                  else:
                      df.to_csv(os.path.join(log_path , 'res_datawise_test.csv'), index = False)

  return error_datasets


def run_embedding_ablation(model_configs,
                           train_dataloader,
                           val_dataloader,
                           val_test_dataloader,
                           test_dataloader,
                           runs_dir,
                           leave_embs = ('task', 'fec', 'learn'),
                           runs = (1, 2, 3),
                           max_epoch = 20,
                           lr_init = 1e-2,
                           step_size = 5,
                           gamma = 0.33,
                           beta = 0.05,
                           init_checkpoint = None):
  """
  Train and evaluate the embedding-ablation variants.

  For every ``run`` and every ``leave_emb`` a fresh model is trained in
  ``<runs_dir>/ablation_<leave_emb>_run<run>``. Training resumes from
  ``latest_model.pth`` (saved after every epoch) if it exists. After the
  last epoch the best checkpoint is evaluated on the in-sample split
  (``val_test_dataloader`` -> ``res_datawise.csv``) and on the zero-shot
  split (``test_dataloader`` -> ``res_datawise_test.csv``).

  Args:
      model_configs: configuration dictionary (a copy is modified per run).
      train_dataloader, val_dataloader: training / validation loaders.
      val_test_dataloader: in-sample evaluation loader (batch size 1).
      test_dataloader: zero-shot evaluation loader (batch size 1).
      runs_dir: parent directory of the run folders.
      leave_embs: ablated embeddings to sweep.
      runs: run identifiers (independent repetitions).
      max_epoch: training epochs.
      lr_init: initial Adam learning rate.
      step_size, gamma: StepLR schedule (stepped once per epoch); when
          training resumes at epoch ``e`` the learning rate restarts at
          ``lr_init * gamma ** (e // step_size)``.
      beta: forwarded to :func:`learn_n_evaluate` (unused by the loss).
      init_checkpoint: optional checkpoint used to initialise
          ``cycle_model`` (non-strict) when no ``latest_model.pth`` exists.
  """
  device = model_configs['device']

  for run in runs:
    for leave_emb in leave_embs:

          model_configs = dict(model_configs)
          model_configs['leave_emb'] = leave_emb
          model_configs['training_mode'] = 'cycle'

          beta_prop = 'constant'
          wflag = False
          log_path = os.path.join(runs_dir, f'ablation_{leave_emb}_run{run}')
          print(log_path)

          model = BatteryModelEmbeddingAblation(model_configs).to(device)

          nump = 0
          for p in model.parameters():
            nump += p.numel()
          print(f'Number of model parameters: {nump}')

          if not os.path.exists(log_path):
              os.makedirs(log_path)

          if os.path.exists(os.path.join(log_path, 'latest_model.pth')):
            loaded_state_dict = torch.load(os.path.join(log_path, 'latest_model.pth'), map_location = device)
            model.load_state_dict(loaded_state_dict)
            print('Loaded Latest Model')
          elif init_checkpoint is not None and os.path.exists(init_checkpoint):
            loaded_state_dict = torch.load(init_checkpoint, map_location = device)
            model.cycle_model.load_state_dict(loaded_state_dict, strict = False)
            print('Loaded saved model')

          for p in model.parameters():
            p.requires_grad = True

          if os.path.exists(os.path.join(log_path, 'val', 'val_cycle.csv')):
            val_saved = pd.read_csv(os.path.join(log_path, 'val', 'val_cycle.csv'))
            start_epoch = val_saved['Epoch'].max() + 1
          else:
            start_epoch = 0

          lr = lr_init*(gamma**(start_epoch//step_size))

          optimizer = torch.optim.Adam(model.cycle_model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
          scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)

          for epoch in range(int(start_epoch), max_epoch):

                model.train()
                print(f'Start Training for {log_path}')
                learn_n_evaluate(model = model,
                                 model_configs = model_configs,
                                 dataloader = train_dataloader,
                                 mode = 'train',
                                 log_path = log_path,
                                 epoch = epoch,
                                 max_epoch = max_epoch,
                                 optimizer=optimizer,
                                 scheduler=scheduler,
                                 beta = beta,
                                 beta_prop = beta_prop,
                                 wflag = wflag,
                                 is_norm = False)

                torch.save(model.state_dict(), os.path.join(log_path, 'latest_model.pth'))

                print('Start Validating:')
                model.eval()
                with torch.no_grad():
                    learn_n_evaluate(model = model,
                                     model_configs = model_configs,
                                     dataloader = val_dataloader,
                                     mode = 'val',
                                     log_path = log_path,
                                     epoch = epoch,
                                     max_epoch = max_epoch,
                                     is_norm = False)

                if (epoch+1) == max_epoch:
                  test_datawise(model, None, log_path, val_test_dataloader, beta = beta, is_norm = False)
                  test_datawise(model, None, log_path, test_dataloader, beta = beta, is_norm = False)

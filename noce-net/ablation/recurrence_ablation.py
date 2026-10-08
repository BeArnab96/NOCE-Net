"""
ablation.recurrence_ablation
============================
Recurrence ablation of NOCE-Net (row "wo Rec. Backbone" of the ablation
table and the figure "MAE across FEC%").

:class:`CycleBlockNoRecurrence` keeps the Patch Encoder, the three
aggregator tokens and the Patch Decoder of NOCE-Net but has no GRU: the
decoder receives the encoder output of each FEC directly, so no information
is propagated between FECs. The forward pass returns only ``V_hat``.

:func:`run_recurrence_ablation` trains this model from scratch for the
requested number of independent runs and evaluates it with
:func:`eval.voltage_eval.test`.
"""
import csv
import gc
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from eval.voltage_eval import test
from noce_net.layers import MultiPatchIndexing, MultiPatchEmbedding, TaskEmbeddingCyc, FECEmbedding, PatchEncoder, OutputRecon
from noce_net.losses import compute_cyc_loss


class CycleBlockNoRecurrence(nn.Module):
  """
  NOCE-Net without the recurrent backbone.

  Args:
      configs: ``model_configs`` dictionary.
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


        # Hidden State Embedding
        self.task_embedding = TaskEmbeddingCyc(configs['task_dim'],
                                            configs['dmodel'])

        self.FEC_Embedding = FECEmbedding(configs['dfec'],
                                          configs['dmodel'])
        #nn.Embedding(configs['max_cycles'],
                             #             configs['dmodel'])
        self.learn_Embedding = nn.Linear(1,
                                         configs['dmodel'], bias = False)

        # Processing Layers
        # Encoder Layer
        self.patch_encoder = PatchEncoder(configs['dmodel'],
                                          configs['enc_layers'],
                                          configs['num_heads'],
                                          configs['ff_mult'],
                                          configs['attn_dropout'],
                                          configs['ff_dropout'])


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
        Same inputs as :meth:`noce_net.layers.CycleBlock.forward`;
        ``glb_tkn`` and ``is_norm`` are ignored.

        Returns:
            V_hat: (B, L, T, d_v) predicted voltage.
        """

        B, L, T, Din = x.shape
        _, Dtsk = task.shape
        

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
        #gt_xemb = xemb.clone()

        # Encoder output
        mask_state = torch.ones(B*L, 3, 1, device = x.device)
        total_mask = torch.cat([mask_state, mask_emb], dim = 1).int().bool() # B*L, Tp+3, 1
        #total_mask = mask_emb.int().nool()

        total_out = self.patch_encoder(gt_xemb.clone().reshape(int(B*L), -1, self.dmodel), total_mask.squeeze(2)).reshape(B, L, -1, self.dmodel) # B*L, Tp+3, D

        # Decoder Operation
        #decoder_out = self.patch_decoder(torch.cat([recurrent_gt.reshape(B, L, 3, -1), updated_inps], dim = 2).reshape(B*L, -1, self.dmodel), total_mask.squeeze(2))
        decoder_out = self.patch_decoder(total_out.reshape(B*L, -1, self.dmodel), total_mask.squeeze(2))


        # Final output with length similar to input
        final_output = self.out_projection(decoder_out[:,3:], T)


        return final_output.reshape(B, L, T, -1)


class BatteryModelNoRecurrence(nn.Module):
  """Container exposing :class:`CycleBlockNoRecurrence` as ``cycle_model``."""
  def __init__(self, configs):
    super().__init__()
    self.configs = configs
    self.cycle_model = CycleBlockNoRecurrence(configs)

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
  One training or validation epoch for :class:`BatteryModelNoRecurrence`.

  Identical to :func:`train.train_response.learn_n_evaluate` except that the
  model output is the voltage only and no state is carried between chunks
  of FECs. See that function for the arguments and the files written.
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

            cyc_pred =  model.cycle_model(inp_batch_split[cid].to(device),
                                          index_batch_split[cid].to(device),
                                          task_batch.to(device),
                                          fec_batch_split[cid].to(device),
                                          cyc_mask_split[cid].to(device),
                                          gtoken,
                                          is_norm
                                          )

            Qc_inp = inp_batch_split[cid][:,:,:,2:3]
            Qd_inp = inp_batch_split[cid][:,:,:,3:4]

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

            del cyc_pred, cyc_loss, quantile_cyc_loss, diff_loss, spike_loss
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


def run_recurrence_ablation(model_configs,
                            train_dataloader,
                            val_dataloader,
                            val_test_dataloader,
                            test_dataloader,
                            runs_dir,
                            runs = (1, 2, 3),
                            max_epoch = 20,
                            lr_init = 1e-2,
                            step_size = 5,
                            gamma = 0.33,
                            beta = 0.05,
                            init_checkpoint = None):
  """
  Train and evaluate NOCE-Net without recurrence.

  Each run is trained in ``<runs_dir>/ablation_norec_run<run>``, resuming
  from ``latest_model.pth`` if present. After the last epoch the best
  checkpoint is evaluated with :func:`eval.voltage_eval.test` on the
  in-sample split (``full_res_datawise.csv``) and the zero-shot split
  (``full_res_datawise_test.csv``).

  Args:
      model_configs: configuration dictionary.
      train_dataloader, val_dataloader: training / validation loaders.
      val_test_dataloader: in-sample evaluation loader (batch size 1).
      test_dataloader: zero-shot evaluation loader (batch size 1).
      runs_dir: parent directory of the run folders.
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
  model_configs = dict(model_configs)
  model_configs['training_mode'] = 'cycle'

  for run in runs:

          beta_prop = 'constant'
          wflag = False
          log_path = os.path.join(runs_dir, f'ablation_norec_run{run}')
          print(log_path)

          model = BatteryModelNoRecurrence(model_configs).to(device)

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
                  test(model, None, log_path, val_test_dataloader, results_file = 'full_res_datawise.csv')
                  test(model, None, log_path, test_dataloader, results_file = 'full_res_datawise_test.csv')

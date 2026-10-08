"""
train.train_response
====================
One epoch of NOCE-Net training or validation on voltage response.

Each battery in a batch is processed as consecutive chunks of about 100
FECs (Appendix "Training Configurations"). The GRU state at the end of a
chunk is detached and used as the initial state of the next chunk
(truncated back-propagation through time); in training mode an optimiser
step is taken after every chunk.

Files written under ``<run_dir>/train`` or ``<run_dir>/val``:

* ``<mode>_cycle.csv``      - per-batch losses, appended every epoch.
* ``val/best_model_loss.pkl`` - history of the best mean validation loss.
* ``val/best_cyc_model.pth``  - checkpoint with the lowest mean validation
  loss (model selection uses the validation split only).
"""
import csv
import gc
import os
import pickle

import numpy as np
import pandas as pd
import torch

from noce_net.losses import compute_cyc_loss

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
                     beta = 0.01,
                     beta_prop = 'constant',
                     wflag = False,
                     is_norm = False):
  """
  Run one epoch of training (``mode='train'``) or evaluation (``'val'``).

  Args:
      model: :class:`~noce_net.model.BatteryModel` (only ``model.cycle_model``
          is used).
      dataloader: ``DataLoader`` over :class:`~noce_net.dataset.Battery_Dataset`
          with :func:`~noce_net.dataset.collate_batch`.
      mode: ``'train'`` or ``'val'``.
      log_path: run directory; logs go to ``<log_path>/<mode>``.
      epoch, max_epoch: current epoch and total epochs (used for logging
          and the optional linear ``beta`` schedule).
      model_configs: configuration dictionary (uses ``device``,
          ``align_loss_type`` and ``training_mode``).
      optimizer, scheduler: optimiser and LR scheduler (training only); the
          scheduler is stepped once per epoch.
      cap_optimizer, cap_scheduler, joint_optimizer, joint_scheduler: unused.
      beta, beta_prop, wflag: unused by the loss (kept for interface
          compatibility with the ablation scripts).
      is_norm: forwarded to the model (unused).

  Returns:
      None. Losses are printed and written to CSV; in ``'val'`` mode the
      best checkpoint is saved.
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

            cyc_pred, rul_pred, states, final_states = model.cycle_model(inp_batch_split[cid].to(device),
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

            #rul_loss = compute_cap_loss(rul_pred, rul_batch_split[cid].to(device), cap_mask_split[cid].to(device), wbatch, dname_batch)

            #cap_loss = cap_criterion(cap_pred, cap_batch_split[cid].to(device), cap_mask_split[cid].to(device), wbatch, dname_batch)

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

            del cyc_pred, states, final_states, cyc_loss, quantile_cyc_loss, diff_loss, spike_loss
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

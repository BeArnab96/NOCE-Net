"""
eval.voltage_eval
=================
Evaluation of voltage-response prediction (in-sample and zero-shot
results) and export of predictions for analysis.

* :func:`test` - loads ``<run_dir>/val/best_cyc_model.pth`` and computes
  MAE, MAPE, sMAPE and MAPE on V > 5e-3 per source dataset and in total.
* :func:`save_dVdQ_and_Vres` - stores predicted and true voltages, inputs
  and differential voltage curves of every battery in
  ``<run_dir>/dVdQ_and_Vres.pkl``.
* :func:`cycle_wise_errors` / :func:`plot_cycle_wise_errors` - MAE per
  decile of battery life computed from that file (Figure "MAE across FEC%"
  of the ablation study).
* :func:`aggregate_runs` - mean and standard deviation of the per-dataset
  metrics over several independent runs.

All evaluation dataloaders must use a batch size of 1 (one battery per
batch), as in the paper.
"""
import os
import pickle

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import median_filter


def _predict_voltage(model, inp_batch, index_batch, task_batch, fec_batch, cyc_mask_batch, is_norm=False):
  """
  Run ``model.cycle_model`` on one batch and return only the voltage prediction.

  ``CycleBlock`` returns a 4-tuple ``(V_hat, rul_pred, H, h_last)`` while the
  model without recurrence (``ablation.recurrence_ablation``) returns ``V_hat``
  alone; both are supported.

  Returns:
      (B, L, T, 1) predicted voltage.
  """
  device = next(model.parameters()).device
  out = model.cycle_model(inp_batch.to(device),
                          index_batch.to(device),
                          task_batch.to(device),
                          fec_batch.to(device),
                          cyc_mask_batch.to(device),
                          None,
                          is_norm)
  return out[0] if isinstance(out, (tuple, list)) else out


def test(model,
         datapath,
         log_path,
         test_dataloader,
         test_log_file = 'test_res_log.csv',
         task_type = 'class',
         beta = 0.1,
         is_norm = False,
         results_file = None,
         ):
  """
  Evaluate the best checkpoint of a run on one dataloader.

  The checkpoint ``<log_path>/val/best_cyc_model.pth`` is loaded if it
  exists. Metrics are computed per source dataset (all ``Stanford_*``
  batteries are grouped as ``'Stanford'``) and over all batteries:

  * MAE  = sum |V_hat - V| / number of valid steps
  * MAPE = mean of |V_hat - V| / (|V| + 1e-3) over valid steps (in %)
  * sMAPE= mean of 2 |V_hat - V| / (|V| + |V_hat| + 1e-3) (in %)
  * MAPE restricted to steps with V > 5e-3 (in %)

  Args:
      model: :class:`~noce_net.model.BatteryModel`, or a model whose
          ``cycle_model`` returns only the voltage.
      datapath, test_log_file, task_type, beta: unused (kept for interface
          compatibility).
      log_path: run directory.
      test_dataloader: dataloader with batch size 1.
      is_norm: forwarded to the model.
      results_file: name of the CSV written in ``log_path``. When ``None``
          the result is written to ``full_res_datawise.csv`` if that file
          does not exist yet and to ``full_res_datawise_test.csv`` otherwise
          (i.e. the first call, on the in-sample split, and the second call,
          on the zero-shot split).

  Returns:
      dict mapping dataset name (and ``'Total'``) to
      ``[MAE, MAPE, sMAPE, MAPE(V > 5e-3)]``. The same values are written as
      columns of the CSV file (rows in that order).
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
              else:
                    print(f'Checkpoint {model_path} not found; evaluating the model as passed.')

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


                      cyc_pred = _predict_voltage(model, inp_batch, index_batch, task_batch,
                                                  fec_batch, cyc_mask_batch, is_norm)


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

                    mape_corr_cyc = np.sum(cyc_uape[cyc_utrues > 5e-3])/np.sum(cyc_umask[cyc_utrues > 5e-3]) * 100
                    mape_cyc = (np.sum(cyc_uape)/np.sum(cyc_umask)) * 100
                    mae_cyc = np.sum(cyc_umae)/np.sum(cyc_umask)
                    smapes_cyc = (np.sum(cyc_usape)/np.sum(cyc_umask)) * 100
                    error_datasets[u] = [ mae_cyc, mape_cyc, smapes_cyc, mape_corr_cyc]
                    print(u, mape_corr_cyc)

                  cyc_true_np = np.concatenate(cyc_trues, axis = -1).reshape(-1)
                  cyc_masks_np = np.concatenate(cyc_masks, axis = -1).reshape(-1)
                  cyc_abs_errs_np = np.concatenate(cyc_ae_list, axis = -1).reshape(-1)#np.abs(cyc_true_np - cyc_pred_np)*cyc_masks_np
                  cyc_ape_np = np.concatenate(cyc_ape_list, axis = -1).reshape(-1)
                  cyc_sape_np = np.concatenate(cyc_sape_list, axis = -1).reshape(-1)

                  mape_np = np.sum(cyc_ape_np)/np.sum(cyc_masks_np) * 100
                  mae_np = np.sum(cyc_abs_errs_np)/np.sum(cyc_masks_np)
                  smapes_np = np.sum(cyc_sape_np)/np.sum(cyc_masks_np) * 100
                  mape_corr_np = np.sum(cyc_ape_np[cyc_true_np > 5e-3])/np.sum(cyc_masks_np[cyc_true_np > 5e-3]) * 100

                  error_datasets['Total'] = [mae_np, mape_np, smapes_np, mape_corr_np]
                  #cyc_true_np = np.concatenate(cyc_trues, axis = 0)
                  #cyc_pred_np = np.concatenate(cyc_preds, axis = 0)
                  #cyc_masks_np = np.concatenate(cyc_masks, axis = 0)
                  #cyc_abs_errs_np = np.abs(cyc_true_np - cyc_pred_np)*cyc_masks_np
                  #cyc_ape_np = cyc_abs_errs_np/(np.abs(cyc_true_np)*cyc_masks_np + 1e-3)
                  #cyc_sape_np = (2*cyc_abs_errs_np) / ((np.abs(cyc_true_np)*cyc_masks_np + np.abs(cyc_pred_np)*cyc_masks_np) + 1e-3)

                  #cyc_sape_np_batwise =
                  #all_cycs_test_data = np.column_stack((cyc_pred_np, cyc_true_np, cyc_masks_np, cyc_abs_errs_np, cyc_ape_np))

                  #mape_cyc = (np.sum(cyc_ape_np)/np.sum(cyc_masks_np)) * 100
                  #mae_cyc = np.sum(cyc_abs_errs_np)/np.sum(cyc_masks_np)
                  #############################################################
                  df = pd.DataFrame.from_dict(error_datasets)
                  if results_file is not None:
                      df.to_csv(os.path.join(log_path , results_file), index = False)
                  elif not os.path.exists(os.path.join(log_path , 'full_res_datawise.csv')):

                      df.to_csv(os.path.join(log_path , 'full_res_datawise.csv'), index = False)
                  else:

                      df.to_csv(os.path.join(log_path , 'full_res_datawise_test.csv'), index = False)
                  #res_test = np.asarray([beta, int(wflag), mid, mape_cyc, mae_cyc]).reshape(1,-1)#epoch_index, step_index, cyc_loss_epoch, quantile_loss_epoch, diff_loss_epoch, spike_loss_epoch])
                  #res_columns = ['Beta', 'Weight_Flag','Model ID', 'MAPE_Cyc', 'MAE_Cyc']

                  #if not os.path.exists(os.path.join(log_path , test_log_file)):
                  #    df = pd.DataFrame(res_test, columns = res_columns)
                  #    df.to_csv(os.path.join(log_path , test_log_file), index = False)

                  #else:
                  #    res_list = [d.tolist() for d in res_test]

                  #    with open(os.path.join(log_path , test_log_file), "a") as stem_data:
                  #            w = csv.writer(stem_data, delimiter=',')
                  #            w.writerows(res_list)
                  #            stem_data.close()


                  ############################################################

  return error_datasets


def return_delv_delQ(V, Qc, Qd):
  """
  Differential voltage ``|dV/dQ|`` of every FEC of one battery.

  Args:
      V, Qc, Qd: (L, T) voltage, charge and discharge throughput.

  Returns:
      ``|dV/dQ|`` of shape (L, T-1) (with ``Q = Qc + Qd`` and a 1e-3
      stabiliser) and a mask that is 0 where ``Qc`` or ``Qd`` decreases.
  """

  delV = V[:,1:] - V[:,:-1]
  delQc = Qc[:,1:] - Qc[:,:-1]
  delQd = Qd[:,1:] - Qd[:,:-1]
  Q = Qc + Qd
  delQ = Q[:,1:] - Q[:,:-1]

  dVdQ_masks = np.ones_like(delQc)
  dVdQ_masks[delQc<0] = 0
  dVdQ_masks[delQd<0] = 0

  delV_delQ = delV/(delQ + 1e-3)
  return np.abs(delV_delQ), dVdQ_masks


def save_dVdQ_and_Vres(model,
                       log_path,
                       test_dataloader):
              """
              Store per-battery predictions of the best checkpoint of a run.

              Loads ``<log_path>/val/best_cyc_model.pth`` (if present), runs
              the model on every battery of ``test_dataloader`` (batch size 1)
              and writes ``<log_path>/dVdQ_and_Vres.pkl`` with the lists
              ``Vhat``, ``V``, ``Vmask`` (each (1, L, T[, 1])), ``Qc``, ``Qd``,
              ``Ic``, ``Id``, ``Dnames`` (dataset names) and the differential
              voltage curves ``dvdq``, ``dvdq_hat`` with their masks.
              """

              device = next(model.parameters()).device
              model_path = os.path.join(log_path, 'val', 'best_cyc_model.pth')

              if os.path.exists(model_path):
                                loaded_state_dict = torch.load(model_path, map_location = device)
                                model.load_state_dict(loaded_state_dict)

              cyc_preds = []
              cyc_trues = []
              cyc_masks = []
              cyc_abs = []
              dataset_names = []
              Qc_list = []
              Qd_list = []
              Ic_list = []
              Id_list = []
              dvdq_list = []
              dvdq_hat_list = []
              dvdq_masks_list = []
              dvdq_hat_masks_list = []

              model.eval()
              with torch.no_grad():
                  for data_id, data in enumerate(test_dataloader):

                      print('Starting idx:', data_id,'  of ', len(test_dataloader))


                      inp_batch, out_batch, cap_batch, task_batch, index_batch, fec_batch, cyc_mask_batch, cap_mask_batch, wbatch, dname_batch = data
                      inp_batch = torch.nan_to_num(inp_batch, nan = 0.0, posinf = 0.0, neginf = 0.0)
                      Nmax = inp_batch.shape[1]

                      Qc = inp_batch[:,:,:,2:3]
                      Qd = inp_batch[:,:,:,3:4]
                      Ic = inp_batch[:,:,:,0:1]
                      Id = inp_batch[:,:,:,1:2]

                      cyc_pred = _predict_voltage(model, inp_batch, index_batch, task_batch,
                                                  fec_batch, cyc_mask_batch, False)

                      cyc_pred_mf = median_filter(cyc_pred.detach().cpu().numpy(), 10, axes = 2)

                      cyc_preds.append(cyc_pred.detach().cpu().numpy())
                      cyc_trues.append(out_batch.cpu().numpy())
                      cyc_masks.append(cyc_mask_batch.cpu().numpy())
                      dataset_names.append(dname_batch[0])
                      Qc_list.append(Qc.cpu().numpy())
                      Qd_list.append(Qd.cpu().numpy())
                      Ic_list.append(Ic.cpu().numpy())
                      Id_list.append(Id.cpu().numpy())

                      delV_delQ_hat, dVdQ_hat_masks = return_delv_delQ(cyc_preds[-1][0,:,:,0], Qc_list[-1][0,:,:,0], Qd_list[-1][0,:,:,0])
                      delV_delQ, dVdQ_masks = return_delv_delQ(cyc_trues[-1][0,:,:,0], Qc_list[-1][0,:,:,0], Qd_list[-1][0,:,:,0])
                      dvdq_list.append(delV_delQ)
                      dvdq_hat_list.append(delV_delQ_hat)
                      dvdq_masks_list.append(dVdQ_masks)
                      dvdq_hat_masks_list.append(dVdQ_hat_masks)

                  res_file = {'Vhat':cyc_preds,
                              'V':cyc_trues,
                              'Vmask':cyc_masks,
                              'Dnames':dataset_names,
                              'Qc':Qc_list,
                              'Qd':Qd_list,
                              'Ic':Ic_list,
                              'Id':Id_list,
                              'dvdq':dvdq_list,
                              'dvdq_hat':dvdq_hat_list,
                              'dvdq_mask':dvdq_masks_list,
                              'dvdq_hat_mask':dvdq_hat_masks_list}

                  with open(os.path.join(log_path, 'dVdQ_and_Vres.pkl'), 'wb') as f:
                    pickle.dump(res_file, f)


def cycle_wise_errors(res_data, n_bins = 10):
  """
  Mean absolute voltage error per decile of battery life.

  For every battery the masked MAE of each FEC is computed; the FECs are
  split into ``n_bins`` consecutive groups of equal size and the MAE is
  averaged inside each group.

  Args:
      res_data: dictionary loaded from ``dVdQ_and_Vres.pkl``
          (see :func:`save_dVdQ_and_Vres`).
      n_bins: number of life-fraction bins (10 = deciles).

  Returns:
      np.ndarray of shape (n_batteries, n_bins).
  """
  Vres, Vhat, Vmask = res_data['V'], res_data['Vhat'], res_data['Vmask']
  cycle_wise_diffs = np.zeros((len(Vres), n_bins))

  for i in range(len(Vres)):

    vdiff = np.abs(Vres[i][0,:,:,0] - Vhat[i][0,:,:,0])
    vdiff_cyc = np.sum(vdiff * Vmask[i][0], axis = -1)/(1e-5 + np.sum(Vmask[i][0], axis = -1, keepdims = True))
    idx10s = np.linspace(0, len(vdiff_cyc), n_bins + 1).astype(int)
    for u in range(n_bins):
      cycle_wise_diffs[i,u] = np.nan_to_num(np.mean(vdiff_cyc[idx10s[u]:idx10s[u+1]]))

  return cycle_wise_diffs


def plot_cycle_wise_errors(cycle_wise_diffs, save_path = None):
  """
  Plot the per-decile MAE of :func:`cycle_wise_errors`.

  Batteries whose 25th percentile of decile-to-decile MAE change is
  negative are left out; each remaining battery is drawn as a thin line and
  their mean as a thick black line.

  Args:
      cycle_wise_diffs: (n_batteries, n_bins) array.
      save_path: optional output path of the figure.

  Returns:
      Tuple ``(fig, cycle_wise_diffs_kept)``.
  """
  import matplotlib.pyplot as plt

  n_bins = cycle_wise_diffs.shape[1]
  cycle_wise_diffs_del = cycle_wise_diffs[:,1:] - cycle_wise_diffs[:,:-1]
  needed_ids = np.percentile(cycle_wise_diffs_del, 25, axis = -1) >= 0
  cycle_wise_diffs_val = cycle_wise_diffs[needed_ids,:]

  pct = np.arange(1, n_bins + 1) * (100 // n_bins)
  fig = plt.figure()
  for i in range(cycle_wise_diffs_val.shape[0]):
    plt.plot(pct, cycle_wise_diffs_val[i,:], alpha = 0.3)
  plt.plot(pct, np.mean(cycle_wise_diffs_val, axis = 0), color='black', linewidth=4)
  plt.xticks(pct)
  plt.xlabel('Percentage Cycles Covered')
  plt.ylabel('Absolute Errors')
  if save_path:
    fig.savefig(save_path, bbox_inches = 'tight')
  return fig, cycle_wise_diffs_val


def aggregate_runs(run_dirs, results_file):
  """
  Mean and standard deviation of evaluation results over several runs.

  Args:
      run_dirs: list of run directories, each containing ``results_file``
          written by :func:`test`.
      results_file: CSV file name, e.g. ``'full_res_datawise.csv'``
          (in-sample) or ``'full_res_datawise_test.csv'`` (zero-shot).

  Returns:
      ``(mean_df, std_df)``: DataFrames with the same layout as the
      per-run CSV (columns = datasets and ``'Total'``; rows = MAE, MAPE,
      sMAPE and the thresholded MAPE of the function that wrote the file:
      V > 5e-3 for :func:`test`, V > 1e-4 for
      :func:`ablation.embedding_ablation.test_datawise`, see
      :mod:`eval.fewshot` for the few-shot results).
  """
  frames = [pd.read_csv(os.path.join(r, results_file)) for r in run_dirs]
  stacked = np.stack([f.values for f in frames])
  columns = frames[0].columns
  index = ['MAE', 'MAPE', 'sMAPE', 'MAPE_thresholded']
  mean_df = pd.DataFrame(np.mean(stacked, axis = 0), columns = columns, index = index)
  std_df = pd.DataFrame(np.std(stacked, axis = 0), columns = columns, index = index)
  return mean_df, std_df

"""
eval.fewshot
============
Few-shot voltage forecasting on the zero-shot (``Test``) datasets.

:func:`test_fewshot` loads ``<run_dir>/val/best_cyc_model.pth``, freezes all
parameters except those whose name starts with ``unfreeze`` (by default the
MLP of the output reconstruction layer, ``cycle_model.out_projection.out_proj``)
and, for every test battery, fine-tunes these parameters on the first
``few_shot_cycles`` FECs. The remaining FECs of the battery are then
predicted, starting from the recurrent state ``h`` reached after the
fine-tuning FECs, and the errors are computed on these remaining FECs only.

The procedure follows the few-shot notebook used for the paper:

* one Adam optimizer (lr ``5e-4``, betas (0.9, 0.98), eps 1e-9) is created
  once; the fine-tuned weights and the optimizer state are carried over
  from one test battery to the next (the weights are not reset per battery);
* each battery is fine-tuned for ``fewshot_epochs`` full-batch steps on its
  first ``few_shot_cycles`` FECs with the training loss
  :func:`noce_net.losses.compute_cyc_loss` and gradient clipping (max norm 1);
* ``model.train()`` is called once before the first battery and
  ``model.eval()`` before every prediction, so dropout is only active while
  fine-tuning the first battery.

Metrics per source dataset are ``[MAE, MAPE, sMAPE, MAPE(V > thr)]`` (in %,
except MAE). For the thresholded MAPE the per-dataset numerator uses
``V > 5e-3`` and the denominator ``V > 1e-2``, as in the notebook. The
``'Total'`` entry holds ``[MAE, MAPE(V > 5e-3), sMAPE, MAPE(V > 5e-3)]``
with the total sMAPE as a fraction (not multiplied by 100). Voltages ``V``
are the normalised targets. Results are written to
``<run_dir>/res_datawise_fshot_<unfreeze>_<few_shot_cycles>.csv`` if that
file does not exist yet.

The evaluation dataloader must use a batch size of 1.
"""
import os

import numpy as np
import pandas as pd
import torch

from noce_net.losses import compute_cyc_loss


def test_fewshot(model,
                 log_path,
                 test_dataloader,
                 model_configs,
                 few_shot_cycles=50,
                 unfreeze='cycle_model.out_projection.out_proj',
                 fewshot_epochs=5,
                 lr=5e-4):
  """
  Few-shot fine-tuning and evaluation (see the module docstring).

  Args:
      model: :class:`~noce_net.model.BatteryModel`.
      log_path: run directory containing ``val/best_cyc_model.pth``; the
          results CSV is written here.
      test_dataloader: zero-shot loader with batch size 1
          (``scripts.common.build_eval_loader(data_root, 'test')``).
      model_configs: configuration dictionary (``align_loss_type`` is
          passed to the loss).
      few_shot_cycles: number of leading FECs per battery used for
          fine-tuning (50 in the paper).
      unfreeze: name prefix of the parameters that are fine-tuned.
          ``'cycle_model.out_projection.out_proj'`` is the ORL MLP (paper
          setting), ``'cycle_model.out_projection'`` the complete ORL.
      fewshot_epochs: fine-tuning steps per battery.
      lr: Adam learning rate.

  Returns:
      dict mapping dataset name (and ``'Total'``) to the metric list.
  """
  device = model_configs['device']

  model_path = os.path.join(log_path, 'val', 'best_cyc_model.pth')
  if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print('Saved model loaded')

  for name, p in model.named_parameters():
    p.requires_grad = name.startswith(unfreeze)

  for name, p in model.named_parameters():
    if p.requires_grad:
      print(name, p.numel())
  print('Model weights frozen except the parameters listed above')

  optimizer = torch.optim.Adam(model.cycle_model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

  model.train()

  cyc_preds = []
  cyc_trues = []
  cyc_masks = []
  dataset_names = []

  for data_id, data in enumerate(test_dataloader):
    print('Starting idx:', data_id, '  of ', len(test_dataloader))

    inp_batch, out_batch, cap_batch, task_batch, index_batch, fec_batch, cyc_mask_batch, cap_mask_batch, wbatch, dname_batch = data
    inp_batch = torch.nan_to_num(inp_batch, nan=0.0, posinf=0.0, neginf=0.0)
    Qc_inp = inp_batch[:, :, :, 2:3]
    Qd_inp = inp_batch[:, :, :, 3:4]

    # Fine-tuning on the first few_shot_cycles FECs
    for epoch in range(fewshot_epochs):
        cyc_pred, rul_pred, states, final_states = model.cycle_model(inp_batch[:, :few_shot_cycles].to(device),
                                                                     index_batch[:, :few_shot_cycles].to(device),
                                                                     task_batch.to(device),
                                                                     fec_batch[:, :few_shot_cycles].to(device),
                                                                     cyc_mask_batch[:, :few_shot_cycles].to(device),
                                                                     None,
                                                                     is_norm=False)
        cyc_loss, _, _, _, _ = compute_cyc_loss(cyc_pred.clone(),
                                                out_batch[:, :few_shot_cycles].to(device),
                                                Qc_inp[:, :few_shot_cycles].to(device),
                                                Qd_inp[:, :few_shot_cycles].to(device),
                                                cyc_mask_batch[:, :few_shot_cycles].to(device),
                                                model_configs['align_loss_type'],
                                                n_proj=64)
        optimizer.zero_grad()
        cyc_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.cycle_model.parameters(), max_norm=1.0)
        optimizer.step()

    # Prediction of the remaining FECs from the state after fine-tuning
    model.eval()
    with torch.no_grad():
        cyc_pred, rul_pred, states, _ = model.cycle_model(inp_batch[:, few_shot_cycles:].to(device),
                                                          index_batch[:, few_shot_cycles:].to(device),
                                                          task_batch.to(device),
                                                          fec_batch[:, few_shot_cycles:].to(device),
                                                          cyc_mask_batch[:, few_shot_cycles:].to(device),
                                                          final_states.clone().detach().to(device),
                                                          is_norm=False)

        cyc_preds.append(cyc_pred.detach().reshape(1, -1).cpu().numpy())
        cyc_trues.append(out_batch[:, few_shot_cycles:].reshape(1, -1).cpu().numpy())
        cyc_masks.append(cyc_mask_batch[:, few_shot_cycles:].reshape(1, -1).cpu().numpy())
        dataset_names.append(dname_batch[0])

  cyc_ae_list = [np.abs(cyc_preds[i] - cyc_trues[i]) * cyc_masks[i] for i in range(len(cyc_preds))]
  cyc_ape_list = [cyc_ae_list[i] / (np.abs(cyc_trues[i]) * cyc_masks[i] + 1e-3) for i in range(len(cyc_preds))]
  cyc_sape_list = [2 * cyc_ae_list[i] / ((np.abs(cyc_trues[i]) + np.abs(cyc_preds[i])) * cyc_masks[i] + 1e-3)
                   for i in range(len(cyc_preds))]

  # All Stanford batteries are reported as one dataset
  data_names_new_list = ['Stanford' if u.startswith('Stanford') else u for u in dataset_names]
  unique_datasets = np.unique(data_names_new_list)
  datanames_np = np.array(data_names_new_list)

  error_datasets = {}
  for u in unique_datasets:
    uidxs = np.where(datanames_np == u)[0]
    cyc_umask = np.concatenate([cyc_masks[i].reshape(-1, 1) for i in uidxs], axis=0).reshape(-1)
    cyc_umae = np.concatenate([cyc_ae_list[i].reshape(-1, 1) for i in uidxs], axis=0).reshape(-1)
    cyc_uape = np.concatenate([cyc_ape_list[i].reshape(-1, 1) for i in uidxs], axis=0).reshape(-1)
    cyc_usape = np.concatenate([cyc_sape_list[i].reshape(-1, 1) for i in uidxs], axis=0).reshape(-1)
    cyc_utrues = np.concatenate([cyc_trues[i].reshape(-1, 1) for i in uidxs], axis=0).reshape(-1)

    mape_corr_cyc = np.sum(cyc_uape[cyc_utrues > 5e-3]) / np.sum(cyc_umask[cyc_utrues > 1e-2]) * 100
    mape_cyc = (np.sum(cyc_uape) / np.sum(cyc_umask)) * 100
    mae_cyc = np.sum(cyc_umae) / np.sum(cyc_umask)
    smapes_cyc = (np.sum(cyc_usape) / np.sum(cyc_umask)) * 100
    error_datasets[u] = [mae_cyc, mape_cyc, smapes_cyc, mape_corr_cyc]
    print(u, mape_corr_cyc)

  cyc_true_np = np.concatenate(cyc_trues, axis=-1)
  cyc_pred_np = np.concatenate(cyc_preds, axis=-1)
  cyc_masks_np = np.concatenate(cyc_masks, axis=-1)
  cyc_abs_errs_np = np.abs(cyc_true_np - cyc_pred_np) * cyc_masks_np
  cyc_ape_np = cyc_abs_errs_np / (np.abs(cyc_true_np) * cyc_masks_np + 1e-3)
  cyc_sape_np = (2 * cyc_abs_errs_np) / ((np.abs(cyc_true_np) * cyc_masks_np + np.abs(cyc_pred_np) * cyc_masks_np) + 1e-3)

  mape_cyc_total = (np.sum(cyc_ape_np[cyc_true_np > 5e-3]) / np.sum(cyc_masks_np[cyc_true_np > 5e-3])) * 100
  mae_cyc_total = np.sum(cyc_abs_errs_np) / np.sum(cyc_masks_np)
  smape_cyc_total = np.sum(cyc_sape_np) / np.sum(cyc_masks_np)

  error_datasets['Total'] = [mae_cyc_total, mape_cyc_total, smape_cyc_total, mape_cyc_total]

  df = pd.DataFrame.from_dict(error_datasets)
  results_file = os.path.join(log_path, f'res_datawise_fshot_{unfreeze}_{few_shot_cycles}.csv')
  if not os.path.exists(results_file):
    df.to_csv(results_file, index=False)
  return error_datasets

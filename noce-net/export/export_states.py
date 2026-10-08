"""
export.export_states
====================
Export of NOCE-Net recurrent states for the SOH analyses.

:func:`save_model_outputs` runs a trained model over every battery of a
split (batch size 1, full lifetime in a single forward pass) and writes
``<out_dir>/<split>_save_model_outputs.pkl``. All modules in ``soh/`` read
these files through :class:`soh.data_pipeline.BatteryDataModule`.
"""
import os
import pickle

import numpy as np
import torch
from scipy.ndimage import median_filter

from eval.voltage_eval import return_delv_delQ


def save_model_outputs(model,
                       log_path,
                       test_dataloader,
                       save = {'rec_states', 'voltage', 'capacity', 'dV_dQ', 'inputs'},
                       loader_mode = 'test'):
              """
              Run a trained NOCE-Net over one split and pickle its outputs.

              The model must already hold the trained weights (e.g. loaded
              from ``<run_dir>/val/best_cyc_model.pth``).

              Args:
                  model: :class:`~noce_net.model.BatteryModel`.
                  log_path: output directory.
                  test_dataloader: dataloader over the split, batch size 1.
                  save: subset of ``{'rec_states', 'voltage', 'capacity',
                      'dV_dQ', 'inputs'}`` selecting what is stored.
                  loader_mode: split name used in the file name
                      (``'train'``, ``'val'`` or ``'test'``).

              Output file ``<log_path>/<loader_mode>_save_model_outputs.pkl``
              is a dictionary of per-battery lists:

              * ``rec_states['pred']`` : (1, L+1, d_h) recurrent states
                ``[h_0, h_1, ..., h_L]``; ``rec_states['mask']`` (1, L)
              * ``capacity['true']``   : (1, L, 1) normalised SOH; ``['mask']``
              * ``voltage['true'|'pred'|'mask']`` : (1, L, T[, 1])
              * ``dV_dQ['true'|'pred']`` : median-filtered |dV/dQ| curves and masks
              * ``inputs['Qc'|'Qd'|'Ic'|'Id'|'mask']``
              * ``dataset_names``     : ``Bat_dataset`` value of each battery
              """

              device = next(model.parameters()).device
              for p in model.parameters():
                p.requires_grad = False


              cyc_preds = []
              cyc_trues = []
              cyc_masks = []

              dataset_names = []

              Qc_list = []
              Qd_list = []
              Ic_list = []
              Id_list = []
              inp_mask_list = []

              dvdq_list = []
              dvdq_hat_list = []
              dvdq_masks_list = []
              dvdq_hat_masks_list = []

              states_list = []
              states_masks_list = []

              capacity_list = []
              cap_masks_list = []

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

                      cyc_pred, rul_pred, states, final_states = model.cycle_model(inp_batch.to(device),
                                                                                   index_batch.to(device),
                                                                                   task_batch.to(device),
                                                                                   fec_batch.to(device),
                                                                                   cyc_mask_batch.to(device),
                                                                                   None,
                                                                                   False
                                                                                   )

                      if 'dV_dQ' in save:
                        cyc_trues_mf = median_filter(out_batch.detach().cpu().numpy(), 10, axes = 2)
                        cyc_pred_mf = median_filter(cyc_pred.detach().cpu().numpy(), 10, axes = 2)
                        QC_mf = median_filter(Qc.detach().cpu().numpy(), 10, axes = 2)
                        QD_mf = median_filter(Qd.detach().cpu().numpy(), 10, axes = 2)

                        delV_delQ_hat, dVdQ_hat_masks = return_delv_delQ(cyc_pred_mf[0,:,:,0], QC_mf[0,:,:,0], QD_mf[0,:,:,0])
                        delV_delQ, dVdQ_masks = return_delv_delQ(cyc_trues_mf[0,:,:,0], QC_mf[0,:,:,0], QD_mf[0,:,:,0])
                        dvdq_list.append(delV_delQ)
                        dvdq_hat_list.append(delV_delQ_hat)
                        dvdq_masks_list.append(dVdQ_masks)
                        dvdq_hat_masks_list.append(dVdQ_hat_masks)

                      if 'rec_states' in save:
                        states_list.append(states.detach().cpu().numpy())
                        states_masks_list.append(cap_mask_batch.detach().cpu().numpy())

                      if 'capacity' in save:
                        capacity_list.append(cap_batch.detach().cpu().numpy())
                        cap_masks_list.append(cap_mask_batch.detach().cpu().numpy())

                      if 'inputs' in save:
                        Qc_list.append(Qc.cpu().numpy())
                        Qd_list.append(Qd.cpu().numpy())
                        Ic_list.append(Ic.cpu().numpy())
                        Id_list.append(Id.cpu().numpy())
                        inp_mask_list.append(cyc_mask_batch.detach().cpu().numpy())

                      if 'voltage' in save:
                        cyc_preds.append(cyc_pred.detach().cpu().numpy())
                        cyc_trues.append(out_batch.cpu().numpy())
                        cyc_masks.append(cyc_mask_batch.cpu().numpy())

                      dataset_names.append(dname_batch[0])

                  res_file = {'voltage':{'true':cyc_trues,
                                         'pred':cyc_preds,
                                         'mask':cyc_masks},
                              'rec_states':{'pred':states_list,
                                            'mask':states_masks_list},
                              'capacity':{'true':capacity_list,
                                          'mask':cap_masks_list},
                              'dV_dQ':{'true':dvdq_list,
                                       'pred':dvdq_hat_list,
                                       'mask':{'true':dvdq_masks_list,
                                               'pred':dvdq_hat_masks_list}},
                              'inputs':{'Qc':Qc_list,
                                        'Qd':Qd_list,
                                        'Ic':Ic_list,
                                        'Id':Id_list,
                                        'mask':inp_mask_list},
                              'dataset_names':dataset_names
                              }

                  with open(os.path.join(log_path, loader_mode+'_save_model_outputs.pkl'), 'wb') as f:
                    pickle.dump(res_file, f)

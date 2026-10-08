"""
noce_net.losses
===============
Training loss of NOCE-Net (Appendix "Loss Function" of the paper).

:func:`compute_cyc_loss` implements, for each chunk of FECs,

    L = L_MSE + 0.2 * L_Q + 0.05 * L_Diff + 5e-4 * L_align

* ``L_MSE``  - masked mean squared error of the voltage.
* ``L_Q``    - mean of the largest 25 % symmetric absolute percentage errors
  along each FEC (tail-penalising term, weight beta = 0.2).
* ``L_Diff`` - masked L1 error of first differences (differential
  preserving term, weight alpha = 0.05).
* ``L_align``- optional dV/dQ distribution-alignment term. It is selected by
  ``model_configs['align_loss_type']``; with ``None`` (the setting of all
  reported results) :func:`dVdQ_loss` returns 0, so the term is inactive.

Shapes: predictions and targets are (B, L, T, 1); masks are (B, L, T).
"""
import torch
import numpy as np


# The functions below implement the optional dV/dQ alignment term. They are
# kept for completeness; the reported experiments use align_loss_type=None.

def dVdQ_comp(Qc, Qd, V):
  """
  Finite-difference differential voltage ``|dV/dQ|`` along each FEC.

  Args:
      Qc, Qd: (B, L, T, 1) charge and discharge throughput.
      V: (B, L, T, 1) voltage.

  Returns:
      ``|dV/dQ|`` of shape (B, L, T-1, 1) and a mask that is 0 where the
      throughput decreases or does not change.
  """

  delV = V[:,:,1:] - V[:,:,:-1]
  delQc = Qc[:,:,1:] - Qc[:,:,:-1]
  delQd = Qd[:,:,1:] - Qd[:,:,:-1]
  Q = Qc - Qd
  delQ = Q[:,:,1:] - Q[:,:,:-1]

  dVdQ_masks = torch.ones_like(delQc)
  dVdQ_masks[delQc<0] = 0
  dVdQ_masks[delQd<0] = 0
  dVdQ_masks[delQ==0] = 0

  delV_delQ = delV/(delQ + 1e-4)
  return torch.abs(delV_delQ), dVdQ_masks

def pairwise_sq_dists(X, Y):
    """
    X: (B, L, T, d)
    Y: (B, L, T, d)
    returns: (T, T) matrix of squared Euclidean distances
    """
    dev = X-Y # B L T d
    return torch.einsum('blnd, blmd -> blnm', dev, dev)
    #XX = (X**2).sum(dim=1, keepdim=True)      # (N, 1)
    #YY = (Y**2).sum(dim=1, keepdim=True).T    # (1, M)
    #return XX + YY - 2.0 * X @ Y.T

def multi_rbf_kernel(X, Y, sigmas):
    """
    Computes multi-scale RBF kernel matrix.

    X: (B, L, T, d)
    Y: (B, L, T, d)
    sigmas: list or tensor of bandwidths

    returns: (B, L, N, M)
    """
    D2 = pairwise_sq_dists(X, Y) # B L T T

    K = 0.0
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * sigma**2)
        K = K + torch.exp(-gamma * D2)

    return K / len(sigmas)

def median_heuristic(Z):
    """
    Z: (B, L, N, d)
    returns: scalar sigma
    """
    B, L, T, _ = Z.shape
    with torch.no_grad():
        D2 = pairwise_sq_dists(Z, Z) # B L T T
        # Remove diagonal
        D2 = D2[~torch.eye(T, dtype=torch.bool, device=Z.device).reshape(1,1,-1, T).repeat(B, L, 1, 1)]
        median = torch.median(torch.abs(D2))
        sigma = torch.sqrt(0.5 * median + 1e-8)
    return sigma

def get_multiscale_sigmas(Z, scales=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """Kernel bandwidths ``sigma_0 * s`` for each scale ``s``, with ``sigma_0`` from :func:`median_heuristic`."""
    base_sigma = median_heuristic(Z)
    sigmas = [base_sigma * s for s in scales]
    return sigmas

def mmd_loss(Z_pred, Z_true):
    """
    Multi-scale RBF kernel matrices for a maximum mean discrepancy (MMD).

    Args:
        Z_pred: (B, L, T, 2) predicted (dV/dQ, V) samples.
        Z_true: (B, L, T, 2) reference (dV/dQ, V) samples.

    Returns:
        Kernel matrices ``(K_pp, K_tt, K_pt)``; :func:`dVdQ_loss` combines them
        into the masked MMD.
    """
    # Combine for adaptive bandwidth selection
    Zp_mean, Zp_std = Z_pred.mean(-2, keepdim = True), Z_pred.std(-2, keepdim = True)
    Zt_mean, Zt_std = Z_true.mean(-2, keepdim = True), Z_true.std(-2, keepdim = True)

    Z_pred = (Z_pred - Zp_mean)/(Zp_std + 1e-8)
    Z_true = (Z_true - Zt_mean)/(Zt_std + 1e-8)

    Z_all = torch.cat([Z_pred, Z_true], dim=0)
    sigmas = get_multiscale_sigmas(Z_all)

    K_pp = multi_rbf_kernel(Z_pred, Z_pred, sigmas)
    K_tt = multi_rbf_kernel(Z_true, Z_true, sigmas)
    K_pt = multi_rbf_kernel(Z_pred, Z_true, sigmas)

    return K_pp, K_tt, K_pt
    #loss = K_pp.mean() + K_tt.mean() - 2.0 * K_pt.mean()
    #return loss



def dVdQ_loss(cyc_pred, cyc_out, Qc_inp, Qd_inp, mask, loss_type = 'SW', n_proj = 64):
    """
    Distribution distance between predicted and true (dV/dQ, V) samples.

    Args:
        cyc_pred, cyc_out: (B, L, T, 1) predicted and true voltage.
        Qc_inp, Qd_inp: (B, L, T, 1) charge / discharge throughput.
        mask: (B, L, T, 1) validity mask.
        loss_type: ``'SW'`` (sliced Wasserstein with ``n_proj`` random
            projections), ``'MMD'`` or ``None`` (returns 0).
        n_proj: number of projections for ``'SW'``.

    Returns:
        Tensor with the loss value (zeros of shape (1,) for ``None``).
    """
    dVdQ, dVdQ_masks = dVdQ_comp(Qc_inp, Qd_inp, cyc_out)
    dVhat_dQ, _ = dVdQ_comp(Qc_inp, Qd_inp, cyc_pred)

    Zt = torch.concat((dVdQ, cyc_out[:,:,:-1]), dim = -1)      # B L T 2
    Zp = torch.concat((dVhat_dQ, cyc_pred[:,:,:-1]), dim = -1) # B L T 2

    if loss_type == 'SW':

      theta = torch.randn(2, n_proj, device = cyc_pred.device)
      theta = theta/theta.norm(dim = 0, keepdim = True)

      p_proj = torch.einsum('bltz, zn -> bltn', Zp, theta).sort(dim = 2).values
      t_proj = torch.einsum('bltz, zn -> bltn', Zt, theta).sort(dim = 2).values

      loss = torch.sum(((p_proj - t_proj)**2) * mask[:,:,:-1])/ (n_proj*mask[:,:,:-1].sum() + 1e-8)

    elif loss_type == 'MMD':
      K_pp, K_tt, K_pt = mmd_loss(Zp, Zt)
      mask_2D = torch.einsum('bltd, blnd -> bltn',mask[:,:,:-1], mask[:,:,:-1])
      loss = ((K_pp*mask_2D).sum(dim =(2,3))/(1e-8+mask_2D.sum(dim = (2,3)))) + ((K_tt*mask_2D).sum(dim =(2,3))/(1e-8+mask_2D.sum(dim = (2,3)))) - 2.0 * ((K_pt*mask_2D).sum(dim =(2,3))/(1e-8+mask_2D.sum(dim = (2,3))))
      loss = torch.mean(loss)

    else:
      loss = torch.zeros((1,),device=cyc_pred.device)

    return loss

def compute_cyc_loss(cyc_pred, cyc_out, Qc_inp, Qd_inp, mask, loss_type, n_proj):
    """
    Composite voltage-response loss (see the module docstring).

    Args:
        cyc_pred: (B, L, T, 1) predicted voltage.
        cyc_out: (B, L, T, 1) true voltage.
        Qc_inp, Qd_inp: (B, L, T, 1) throughput inputs (only used by the
            alignment term).
        mask: (B, L, T) validity mask.
        loss_type: alignment type passed to :func:`dVdQ_loss`.
        n_proj: projections for the sliced-Wasserstein alignment.

    Returns:
        ``(loss, L_Q, L_Diff, L_spike, L_align)``; only ``loss`` carries
        gradients. ``L_spike`` (error on steps with |dV| > 0.1) is logged
        but not part of ``loss``.
    """

    mask = mask.unsqueeze(-1)

    # 1. base loss
    mse = torch.sum(torch.square(cyc_pred*mask - cyc_out*mask)) / (torch.sum(mask) + 1e-8)

    # 2. ape 3rd quantile loss
    ape = (2*torch.abs(cyc_pred - cyc_out)/(torch.abs(cyc_out) + torch.abs(cyc_pred) + 1e-8)) * mask
    k = int(0.25*ape.shape[2])
    topk_ape, _ = torch.topk(ape, k, dim = 2)
    quantile_loss = torch.mean(topk_ape)
    #sorted_ape, _ = torch.sort(ape, dim=2)
    #quantile_75_index = int(0.75 * sorted_ape.shape[2])
    #quantile_75 = sorted_ape[:,:,quantile_75_index:]
    #quantile_loss =  torch.mean(quantile_75)

    #quantile_loss = quantile_75
    #quantile_75 = torch.quantile(ape, 0.75, dim = 2, keepdim = True)
    #ape_over_75 = ape * (ape >= quantile_75)

    # 3. first derivative loss (In versions 2 and 3) (was MSE in version 2. Changed to MAE in version 3)

    pred_diff = cyc_pred[:,:,1:] * mask[:,:,1:] - cyc_pred[:,:,:-1] * mask[:,:,:-1]
    true_diff = cyc_out[:,:,1:] * mask[:,:,1:] - cyc_out[:,:,:-1] * mask[:,:,:-1]
    diff_loss = 2 * torch.sum(torch.abs(pred_diff - true_diff)) / (torch.sum(mask[:,:,1:]) + 1e-8 + torch.sum(mask[:,:,:-1]))

    # 4. Spiky change loss (Stays in version 2. Removed from backprop in version 3. Not in version 1.)

    spike_mask = (torch.abs(true_diff) > 0.1).float()
    spike_loss = torch.sum(torch.abs(pred_diff - true_diff) * spike_mask)/(torch.sum(spike_mask) + 1e-5)

    # 5. dVdQ loss

    dVdQ_loss_ = dVdQ_loss(cyc_pred, cyc_out, Qc_inp, Qd_inp, mask, loss_type, n_proj)

    #if torch.sum(mask) < 0.5*torch.numel(mask):
    #  lambd = 0.5
    #else:
    #  lambd = 0.1

    loss =  mse + 0.2 * quantile_loss + 0.05 * diff_loss + 5e-4 * dVdQ_loss_ #+ 0.2 * spike_loss (Spike Loss removed in version 3. Stays in Version 2)

    return torch.nan_to_num(loss), quantile_loss.detach(), diff_loss.detach() , spike_loss.detach(), dVdQ_loss_.detach()

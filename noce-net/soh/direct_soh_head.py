"""
soh.direct_soh_head
===================
Direct SOH prediction from frozen NOCE-Net states (Appendix "Direct SOH
Prediction Head" and "Shape Diagnostic").

* :class:`DirectSOHHead` - PID-R features followed by an MLP that outputs
  one SOH value per FEC.
* :class:`SOHLoss`       - MSE + MAPE + soft monotonicity + BOL anchor.
* :class:`SOHTrainer`    - joint training on all training datasets and
  per-dataset evaluation on the in-sample (``val``) and zero-shot
  (``test``) splits.
* :func:`shape_diagnostic` - Pearson shape correlation, mean offset and
  scale ratio between predicted and true SOH trajectories.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from soh.data_pipeline import BatteryDataModule
from soh.pidr_features import PIDFeatureExtractor

# ─────────────────────────────────────────────────────────────────────────────
# Head
# ─────────────────────────────────────────────────────────────────────────────

class DirectSOHHead(nn.Module):
    """
    Predicts the SOH trajectory SOH(c) directly from PID-R features.

    Architecture:
        PIDFeatureExtractor(H) → φ_c ∈ R^{4·d_h}
        MLP(φ_c)               → SOH_pred(c) ∈ R

    The MLP has ``len(hidden_dims)`` hidden layers (Linear, LayerNorm, GELU,
    Dropout) and a linear output; no output activation is applied, so
    negative SOH values (batteries cycled beyond the 80 % threshold) can be
    predicted. Monotonicity is encouraged by a soft penalty in
    :class:`SOHLoss` rather than by the architecture.

    Args:
        d_h         : GRU hidden state dimension
        hidden_dims : MLP hidden layer sizes
        lambda_init : initial EMA decay for PID extractor
        dropout     : dropout rate between MLP layers (default 0.1)
    """

    def __init__(
        self,
        d_h:         int,
        hidden_dims: Tuple[int, ...] = (128, 64),
        lambda_init: float = 0.95,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.pid = PIDFeatureExtractor(d_h, lambda_init=lambda_init)

        in_dim = 4 * d_h
        layers = []
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        H_batch:  torch.Tensor,     # (B, N_max, d_h)
        lengths:  torch.LongTensor, # (B,)
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """
        Returns
        -------
        SOH_pred : (B, N_max, 1)  predicted SOH (0 at padded positions)
        mask     : (B, N_max)     True = valid position
        """
        phi, mask = self.pid(H_batch, lengths)          # (B, N_max, 4*d_h)
        logits    = self.mlp(phi)                        # (B, N_max, 1)
        SOH_pred  = logits                # (B, N_max, 1)
        # Zero out padding
        SOH_pred  = SOH_pred * mask.unsqueeze(-1).float()
        return SOH_pred, mask


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class SOHLoss(nn.Module):
    """
    Composite loss for direct SOH prediction.

    Components
    ----------
    L_mse   : MSE between predicted and true SOH
    L_mape  : mean absolute % error (denominator = 1 since SOH ∈ [0,1],
              clipped to avoid explosion near SOH=0 at EOL)
    L_mono  : soft monotone penalty — penalises SOH(c) > SOH(c-1) + margin
              (encourages decreasing trajectory without hard constraint)
    L_bol   : squared error of the first predicted SOH value of each
              battery (beginning-of-life anchor)

    L_mse, L_mape and L_mono are computed over valid (non-padded) positions.

    Args:
        mse_weight   : weight on MSE term
        mape_weight  : weight on MAPE term
        mono_weight  : weight on monotone penalty
        mono_margin  : allowed upward deviation before penalising (default 0.01)
        mape_eps     : floor for MAPE denominator (default 0.05)
        bol_weight   : weight on the BOL anchor term

    Returns (forward):
        dict with ``loss`` (weighted sum), ``L_mse``, ``L_mape`` and ``L_mono``.
    """

    def __init__(
        self,
        mse_weight:  float = 1.0,
        mape_weight: float = 0.05,
        mono_weight: float = 0.3,
        mono_margin: float = 0.01,
        mape_eps:    float = 0.05,
        bol_weight  : float = 1.0,
    ):
        super().__init__()
        self.mse_w  = mse_weight
        self.mape_w = mape_weight
        self.mono_w = mono_weight
        self.bol_w  = bol_weight
        self.margin = mono_margin
        self.eps    = mape_eps

    def forward(
        self,
        SOH_pred: torch.Tensor,    # (B, N_max, 1)
        SOH_true: torch.Tensor,    # (B, N_max, 1)
        mask:     torch.BoolTensor, # (B, N_max)
    ) -> Dict[str, torch.Tensor]:

        """Weighted loss on valid positions; ``SOH_pred`` and ``SOH_true`` are (B, N_max, 1), ``mask`` (B, N_max)."""
        m       = mask.unsqueeze(-1).float()   # (B, N_max, 1)
        n_valid = m.sum().clamp(min=1)

        # ── MSE ──────────────────────────────────────────────────────────────
        L_mse = ((SOH_pred - SOH_true) ** 2 * m).sum() / n_valid

        # ── MAPE  (denom clipped to avoid explosion at SOH ≈ 0) ─────────────
        denom  = SOH_true.abs().clamp(min=self.eps)
        L_mape = ((SOH_pred - SOH_true).abs() / denom * m).sum() / n_valid

        # ── Soft monotone penalty ─────────────────────────────────────────────
        # Penalise positive increments: relu(SOH(c) - SOH(c-1) - margin)
        delta    = SOH_pred[:, 1:, :] - SOH_pred[:, :-1, :]  # (B, N_max-1, 1)
        m_pairs  = (m[:, 1:, :] * m[:, :-1, :])               # valid consecutive pairs
        L_mono   = (torch.relu(delta - self.margin) * m_pairs).sum() / m_pairs.sum().clamp(min=1)

        # ── BOL penalty ─────────────────────────────────────────────
        # First predicted SOH value per battery should be near 1.0
        # Only penalise the first valid position of each sequence
        first_pred = SOH_pred[:, 0, :]          # (B, 1)
        first_true = SOH_true[:, 0, :]          # (B, 1)  should all be ~1.0
        L_bol = ((first_pred - first_true) ** 2).mean()

        loss = self.mse_w * L_mse + self.mape_w * L_mape + self.mono_w * L_mono + self.bol_w * L_bol

        return dict(loss=loss, L_mse=L_mse, L_mape=L_mape, L_mono=L_mono)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class SOHTrainer:
    """
    Trains DirectSOHHead on the train split; evaluates on val / test with
    per-dataset breakdowns.

    Args
    ----
    data_module     : BatteryDataModule
    d_h             : hidden state dim (inferred from data if None)
    hidden_dims     : MLP layer sizes
    lambda_init     : EMA decay init for PID
    lr              : learning rate
    loss_kwargs     : passed to SOHLoss
    exclude_datasets: dataset names to skip during training
                      (e.g. ["evTOL", "UMich"])
    device          : "auto" | "cuda" | "cpu"
    """

    def __init__(
        self,
        data_module:      BatteryDataModule,
        d_h:              Optional[int]   = None,
        hidden_dims:      Tuple           = (128, 64),
        lambda_init:      float           = 0.95,
        lr:               float           = 1e-3,
        loss_kwargs:      dict            = None,
        exclude_datasets: List[str]       = None,
        device:           str             = "auto",
    ):
        self.dm               = data_module
        self.exclude_datasets = set(exclude_datasets or [])
        self.loss_fn          = SOHLoss(**(loss_kwargs or {}))
        self.history: List[dict] = []

        if d_h is None:
            d_h = data_module.train_ds.items[0]["H"].shape[1]

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else torch.device(device)
        )
        self.model = DirectSOHHead(
            d_h=d_h, hidden_dims=hidden_dims, lambda_init=lambda_init
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=lr * 0.01
        )

    # ── training ─────────────────────────────────────────────────────────────

    def train(
        self,
        n_epochs:  int  = 100,
        val_every: int  = 10,
        verbose:   bool = True,
    ) -> List[dict]:
        """
        Train the head on the ``train`` split (Adam, cosine LR schedule).

        Batteries of ``exclude_datasets`` are removed from every batch.

        Args:
            n_epochs: training epochs.
            val_every: validation / logging interval in epochs.
            verbose: print losses.

        Returns:
            History: one dict of averaged losses per epoch.
        """

        loader     = self.dm.train_loader(shuffle=True)
        val_loader = self.dm.val_loader()

        for epoch in range(1, n_epochs + 1):
            self.model.train()
            epoch_losses = defaultdict(float)
            n_batches = 0

            for batch in loader:
                # Skip excluded datasets
                if self.exclude_datasets and all(
                    d in self.exclude_datasets for d in batch["datasets"]
                ):
                    continue

                # Mask out excluded batteries within mixed batches
                keep = torch.tensor(
                    [d not in self.exclude_datasets for d in batch["datasets"]],
                    dtype=torch.bool,
                )

                H       = batch["H_batch"][keep].to(self.device)
                SOH     = batch["Q_batch"][keep].unsqueeze(-1).to(self.device)
                lengths = batch["lengths"][keep].to(self.device)

                if H.shape[0] == 0:
                    continue

                self.optimizer.zero_grad()
                SOH_pred, mask = self.model(H, lengths)
                losses = self.loss_fn(SOH_pred, SOH, mask)
                losses["loss"].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                for k, v in losses.items():
                    epoch_losses[k] += v.item()
                n_batches += 1

            self.scheduler.step()

            if n_batches == 0:
                continue

            avg    = {k: v / n_batches for k, v in epoch_losses.items()}
            record = {"epoch": epoch, **{f"train_{k}": v for k, v in avg.items()}}

            if val_loader is not None and epoch % val_every == 0:
                val_m = self._eval_loader(val_loader)
                record.update({f"val_{k}": v for k, v in val_m.items()})

            self.history.append(record)

            if verbose and epoch % val_every == 0:
                msg = (f"Ep {epoch:>4d}/{n_epochs}  "
                       f"loss={avg['loss']:.4f}  "
                       f"MSE={avg['L_mse']:.4f}  "
                       f"MAPE={avg['L_mape']*100:.2f}%  "
                       f"mono={avg['L_mono']:.4f}")
                if "val_L_mape" in record:
                    msg += f"  |  val_MAPE={record['val_L_mape']*100:.2f}%"
                print(msg)

        return self.history

    # ── inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        split:        str            = "test",
        dataset_name: Optional[str]  = None,
    ) -> dict:
        """
        Run inference on a split (or one named dataset within a split).

        Per battery, MAPE = mean(|SOH_pred - SOH| / max(|SOH|, 0.05)) and
        MAE = mean(|SOH_pred - SOH|); aggregates are means over batteries.

        Args:
            split: ``'val'`` (in-sample) or ``'test'`` (zero-shot). With
                ``dataset_name`` any split name accepted by the data module.
            dataset_name: optional source dataset to restrict to.

        Returns
        -------
        dict with keys:
            SOH_pred, SOH_true : lists of (N_i,) tensors
            mape, mae          : per-battery floats
            datasets, battery_indices
            aggregate          : overall + per-dataset breakdown
        """
        if dataset_name:
            loader = self.dm.loader_for_dataset(dataset_name, split=split)
        else:
            loader = (self.dm.val_loader() if split == "val"
                      else self.dm.test_loader())

        self.model.eval()
        all_pred, all_true = [], []
        all_mape, all_mae  = [], []
        all_ds,   all_bidx = [], []

        for batch in loader:
            H       = batch["H_batch"].to(self.device)
            SOH_t   = batch["Q_batch"].to(self.device)
            lengths = batch["lengths"].to(self.device)

            SOH_pred, mask = self.model(H, lengths)
            SOH_pred = SOH_pred.squeeze(-1)   # (B, N_max)

            for i in range(len(lengths)):
                n   = lengths[i].item()
                sp  = SOH_pred[i, :n].cpu()
                st  = SOH_t[i, :n].cpu()
                # MAPE with floor denominator (SOH ∈ [0,1])
                mape = ((sp - st).abs() / st.abs().clamp(min=0.05)).mean().item()
                mae  = (sp - st).abs().mean().item()
                all_pred.append(sp)
                all_true.append(st)
                all_mape.append(mape)
                all_mae.append(mae)
                all_ds.append(batch["datasets"][i])
                all_bidx.append(batch["battery_indices"][i])

        per_ds: Dict[str, dict] = defaultdict(lambda: {"mape": [], "mae": []})
        for ds, mape, mae in zip(all_ds, all_mape, all_mae):
            per_ds[ds]["mape"].append(mape)
            per_ds[ds]["mae"].append(mae)

        per_ds_summary = {
            name: {
                "mean_mape": float(np.mean(v["mape"])),
                "mean_mae":  float(np.mean(v["mae"])),
                "n":         len(v["mape"]),
            }
            for name, v in per_ds.items()
        }

        return dict(
            SOH_pred=all_pred, SOH_true=all_true,
            mape=all_mape, mae=all_mae,
            datasets=all_ds, battery_indices=all_bidx,
            aggregate=dict(
                overall_mape=float(np.mean(all_mape)),
                overall_mae=float(np.mean(all_mae)),
                per_dataset=per_ds_summary,
            ),
        )

    def evaluate_all_splits(self, verbose: bool = True) -> dict:
        """Run :meth:`predict` on ``val`` and ``test`` and print per-dataset tables."""
        out = {}
        for split in ("val", "test"):
            try:
                res = self.predict(split=split)
                out[split] = res
                if verbose:
                    self._print_table(split, res["aggregate"])
            except (ValueError, AttributeError):
                pass
        return out

    def plot_predictions(
        self,
        pred_result:    dict,
        n_per_dataset:  int  = 3,
        save_path:      Optional[str] = None,
        dpi:            int  = 150,
    ) -> plt.Figure:
        """
        Plot predicted and true SOH for up to ``n_per_dataset`` batteries
        per dataset (output of :meth:`predict`).
        """
        by_ds: Dict[str, List[int]] = defaultdict(list)
        for i, ds in enumerate(pred_result["datasets"]):
            by_ds[ds].append(i)

        ds_list = sorted(by_ds.keys())
        ncols   = min(n_per_dataset, 4)
        fig, axes = plt.subplots(
            len(ds_list), ncols,
            figsize=(4.5 * ncols, 3.5 * len(ds_list)),
            squeeze=False,
        )

        for row, ds in enumerate(ds_list):
            for col, idx in enumerate(by_ds[ds][:ncols]):
                ax   = axes[row][col]
                st   = pred_result["SOH_true"][idx].numpy()
                sp   = pred_result["SOH_pred"][idx].numpy()
                pct  = np.linspace(0, 100, len(st))
                ax.plot(pct, st, color="#2166ac", lw=1.5, label="True SOH")
                ax.plot(pct, sp, color="#d6604d", lw=1.5,
                        ls="--", label="Predicted SOH")
                ax.set_ylim(-0.05, 1.1)
                ax.set_title(
                    f"{ds} | bat={pred_result['battery_indices'][idx]}\n"
                    f"MAPE={pred_result['mape'][idx]*100:.1f}%  "
                    f"MAE={pred_result['mae'][idx]:.3f}",
                    fontsize=8,
                )
                ax.set_xlabel("Cycle %", fontsize=8)
                ax.set_ylabel("SOH", fontsize=8)
                if col == 0 and row == 0:
                    ax.legend(fontsize=7)
            for col in range(len(by_ds[ds][:ncols]), ncols):
                axes[row][col].axis("off")

        fig.suptitle("Direct SOH Predictions", fontsize=12, fontweight="bold")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved → {save_path}")
        return fig

    # ── internal helpers ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _eval_loader(self, loader: DataLoader) -> dict:
        """Average :class:`SOHLoss` terms over a loader (no exclusion applied)."""
        self.model.eval()
        totals = defaultdict(float)
        n = 0
        for batch in loader:
            H       = batch["H_batch"].to(self.device)
            SOH     = batch["Q_batch"].unsqueeze(-1).to(self.device)
            lengths = batch["lengths"].to(self.device)
            pred, mask = self.model(H, lengths)
            losses = self.loss_fn(pred, SOH, mask)
            for k, v in losses.items():
                totals[k] += v.item()
            n += 1
        return {k: v / max(n, 1) for k, v in totals.items()}

    @staticmethod
    def _print_table(split: str, agg: dict):
        """Print per-dataset and total MAPE / MAE."""
        print(f"\n{'─'*55}\nInference results — {split.upper()}\n{'─'*55}")
        print(f"{'Dataset':<22} {'N':>4}  {'MAPE':>8}  {'MAE':>8}")
        for name, v in sorted(agg["per_dataset"].items()):
            print(f"{name:<22} {v['n']:>4}  "
                  f"{v['mean_mape']*100:>7.2f}%  {v['mean_mae']:>8.4f}")
        print("═" * 55)
        print(f"{'TOTAL':<27}  "
              f"{agg['overall_mape']*100:>7.2f}%  {agg['overall_mae']:>8.4f}")


def shape_diagnostic(trainer: SOHTrainer, split: str = "val", verbose: bool = True) -> dict:
    """
    Decompose the SOH prediction error into shape, offset and scale.

    For every battery of ``split``:

    * ``shape_r``      : Pearson correlation between the centred predicted
      and true trajectories,
    * ``offset``       : mean(SOH_pred) - mean(SOH),
    * ``scale_needed`` : std(SOH) / std(SOH_pred).

    The values are averaged per source dataset (Table "Shape diagnostic";
    ``split='val'`` gives the in-sample rows, ``split='test'`` the
    zero-shot rows).

    Args:
        trainer: trained :class:`SOHTrainer`.
        split: ``'val'`` or ``'test'``.
        verbose: print the per-dataset table.

    Returns:
        dict mapping dataset name to per-battery lists and the means
        ``mean_shape_r``, ``mean_offset`` and ``mean_scale``.
    """
    pred_result = trainer.predict(split=split)
    by_ds: Dict[str, dict] = defaultdict(
        lambda: {"shape_r": [], "offset": [], "scale_needed": [],
                 "battery_indices": []}
    )
    for i, ds in enumerate(pred_result["datasets"]):
        sp = pred_result["SOH_pred"][i].numpy()
        st = pred_result["SOH_true"][i].numpy()
        sp_c = sp - sp.mean(); st_c = st - st.mean()
        denom = np.linalg.norm(sp_c) * np.linalg.norm(st_c)
        r   = float(np.dot(sp_c, st_c) / denom) if denom > 1e-8 else 0.0
        off = float(sp.mean() - st.mean())
        sc  = float(st.std() / sp.std()) if sp.std() > 1e-8 else float("nan")
        by_ds[ds]["shape_r"].append(r)
        by_ds[ds]["offset"].append(off)
        by_ds[ds]["scale_needed"].append(sc)
        by_ds[ds]["battery_indices"].append(pred_result["battery_indices"][i])

    summary = {}
    for ds, vals in by_ds.items():
        sr  = [v for v in vals["shape_r"]     if not np.isnan(v)]
        off = [v for v in vals["offset"]       if not np.isnan(v)]
        sc  = [v for v in vals["scale_needed"] if not np.isnan(v)]
        summary[ds] = dict(
            shape_r=vals["shape_r"], offset=vals["offset"],
            scale_needed=vals["scale_needed"],
            battery_indices=vals["battery_indices"],
            mean_shape_r=float(np.mean(sr))  if sr  else float("nan"),
            mean_offset =float(np.mean(off)) if off else float("nan"),
            mean_scale  =float(np.mean(sc))  if sc  else float("nan"),
        )

    if verbose:
        sep = "─" * 65
        print(f"\n{sep}\nShape Diagnostic — {split.upper()}\n{sep}")
        print(f"{'Dataset':<22}  {'shape_r':>9}  {'offset':>9}  {'scale_needed':>13}")
        print(sep)
        for ds, v in sorted(summary.items()):
            print(f"{ds:<22}  {v['mean_shape_r']:>+9.3f}  "
                  f"{v['mean_offset']:>+9.4f}  {v['mean_scale']:>13.3f}")
    return summary

"""
soh.per_dataset_mlp
===================
Per-dataset SOH regression from frozen PID-R features with a varying
fraction of in-domain training batteries.

For every source dataset a small MLP (:class:`LightweightSOHMLP`) is
trained on the PID-R features ``phi_c`` (4 d_h) of a random subset of the
dataset's batteries and evaluated on the remaining batteries. The PID-R
extractor is taken from a trained :class:`~soh.direct_soh_head.DirectSOHHead`
and kept frozen. The sweep over training fractions is repeated
``n_repeats`` times with different random battery splits.

Notation: ``N`` FECs of a battery, ``h_c`` (d_h) recurrent state of FEC
``c``, ``phi_c`` (4 d_h) its PID-R feature, ``Q_c`` the normalised SOH
label.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from soh.data_pipeline import BatteryDataModule
from soh.pidr_features import PIDFeatureExtractor

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Datasets left out of the sweep
EXCLUDE_DATASETS = {"evTOL", "UMich"}

# Minimum number of batteries for a dataset to be included
MIN_BATTERIES = 4

# Minimum number of training batteries per split; training fractions that
# leave no test battery after applying this minimum are skipped
MIN_TRAIN_BATTERIES = 2


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (frozen PID)
# ─────────────────────────────────────────────────────────────────────────────

def extract_pid_features(
    items:      List[dict],
    pid:        PIDFeatureExtractor,
    device:     torch.device,
    max_len:    Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor, List[int]]:
    """
    Compute PID-R features and SOH labels for a list of batteries.

    Args:
        items: battery dictionaries of :class:`~soh.data_pipeline.BatteryResDataset`
            (keys ``'H'`` (N, d_h) and ``'Q'`` (N,)).
        pid: PID-R extractor; evaluated without gradients.
        device: device on which the features are computed.
        max_len: padded length ``N_max`` (default: longest battery).

    Returns:
        ``(phi, soh, mask, lengths)`` on the CPU with ``phi`` (B, N_max, 4 d_h),
        ``soh`` (B, N_max, 1), ``mask`` (B, N_max) True at valid FECs and
        ``lengths`` the list of N per battery.
    """
    lengths = [it["H"].shape[0] for it in items]
    N_max   = max_len or max(lengths)
    B       = len(items)
    d_h     = items[0]["H"].shape[1]

    H_pad   = torch.zeros(B, N_max, d_h)
    soh_pad = torch.zeros(B, N_max, 1)

    for i, it in enumerate(items):
        n            = it["H"].shape[0]
        H_pad[i, :n] = it["H"]
        soh_pad[i,:n,0] = it["Q"]

    H_pad   = H_pad.to(device)
    lengths_ = torch.tensor(lengths, dtype=torch.long, device=device)

    pid.eval()
    with torch.no_grad():
        phi, mask = pid(H_pad, lengths_)   # (B, N_max, 4*d_h), (B, N_max)

    soh_pad = soh_pad.to(device)
    return phi.cpu(), soh_pad.cpu(), mask.cpu(), lengths


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight per-dataset MLP
# ─────────────────────────────────────────────────────────────────────────────

class LightweightSOHMLP(nn.Module):
    """
    Two-hidden-layer MLP mapping a PID-R feature ``phi_c`` (4 d_h) to SOH.

    Layers: Linear(in_dim, hidden) -> LayerNorm -> GELU -> Dropout(0.1) ->
    Linear(hidden, hidden/2) -> LayerNorm -> GELU -> Linear(hidden/2, 1) ->
    Sigmoid, so predictions lie in (0, 1).

    Args:
        in_dim: input dimension (4 d_h).
        hidden: width of the first hidden layer.
    """

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map features (..., in_dim) to SOH predictions (..., 1)."""
        return self.net(x)


def train_mlp_head(
    phi_train:  torch.Tensor,   # (N_total_train_cycles, 4*d_h)
    soh_train:  torch.Tensor,   # (N_total_train_cycles,)
    in_dim:     int,
    hidden:     int   = 64,
    n_epochs:   int   = 80,
    lr:         float = 1e-3,
    batch_size: int   = 256,
    device:     torch.device = torch.device("cpu"),
) -> LightweightSOHMLP:
    """
    Train a :class:`LightweightSOHMLP` on FEC-level ``(phi_c, Q_c)`` pairs.

    Adam with cosine annealing to ``0.01 * lr``, MSE loss, gradient
    clipping at 1.0 and shuffled mini-batches of FECs.

    Args:
        phi_train: (M, 4 d_h) features of all training FECs.
        soh_train: (M,) SOH labels.
        in_dim: input dimension.
        hidden: hidden width.
        n_epochs, lr, batch_size: optimisation settings.
        device: training device.

    Returns:
        The trained model.
    """
    dataset    = TensorDataset(phi_train, soh_train)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model      = LightweightSOHMLP(in_dim, hidden=hidden).to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01
    )
    loss_fn    = nn.MSELoss()

    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb).squeeze(), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    return model


@torch.no_grad()
def eval_mlp_head(
    model:     LightweightSOHMLP,
    phi_test:  torch.Tensor,     # (N_total_test_cycles, 4*d_h)
    soh_test:  torch.Tensor,     # (N_total_test_cycles,)
    device:    torch.device,
) -> Tuple[float, float]:
    """
    Evaluate a trained head on FEC-level test pairs.

    MAPE uses the denominator ``max(|Q_c|, 0.05)``.

    Returns:
        ``(MAPE, MAE)`` averaged over all test FECs (MAPE as a fraction).
    """
    model.eval()
    phi_test = phi_test.to(device)
    pred     = model(phi_test).squeeze().cpu().numpy()
    true     = soh_test.numpy()
    mape     = float(np.mean(
        np.abs(pred - true) / np.clip(np.abs(true), 0.05, None)
    ))
    mae      = float(np.mean(np.abs(pred - true)))
    return mape, mae


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset analysis runner
# ─────────────────────────────────────────────────────────────────────────────

class PerDatasetMLPAnalysis:
    """
    Per-dataset sweep over the fraction of training batteries.

    The batteries of the ``train`` and ``test`` splits of the data module
    are pooled per source dataset (datasets in ``EXCLUDE_DATASETS`` and
    datasets with fewer than ``MIN_BATTERIES`` batteries are skipped). For
    every dataset and training fraction ``p``:

    1. ``n_train = max(MIN_TRAIN_BATTERIES, ceil(p * n_batteries))``.
    2. For each repeat ``r`` the batteries are randomly permuted (seed
       ``1000 r + 100 p``); the first ``n_train`` train the head, the rest
       are used for testing.
    3. A :class:`LightweightSOHMLP` is trained on the PID-R features of the
       training FECs and evaluated on the test FECs (MAPE, MAE).

    Mean and standard deviation over repeats are reported.

    Args:
        data_module: :class:`~soh.data_pipeline.BatteryDataModule`.
        pid_extractor: trained PID-R extractor (e.g. ``DirectSOHHead.pid``), kept frozen.
        d_h: dimension of the recurrent states.
        sweep_pcts: training fractions (default ``[0.1, 0.2, 0.3, 0.4, 0.5]``).
        mlp_hidden: hidden width of the MLP.
        n_epochs: training epochs per split.
        n_repeats: number of random splits per fraction.
        device: ``'auto'`` or a device string.
    """

    DEFAULT_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5]

    def __init__(
        self,
        data_module:   BatteryDataModule,
        pid_extractor: PIDFeatureExtractor,
        d_h:           int,
        sweep_pcts:    List[float] = None,
        mlp_hidden:    int   = 64,
        n_epochs:      int   = 80,
        n_repeats:     int   = 3,
        device:        str   = "auto",
    ):
        self.dm         = data_module
        self.pid        = pid_extractor
        self.d_h        = d_h
        self.sweep      = sweep_pcts or self.DEFAULT_SWEEP
        self.hidden     = mlp_hidden
        self.n_epochs   = n_epochs
        self.n_repeats  = n_repeats
        self.in_dim     = 4 * d_h
        self.device     = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else torch.device(device)
        )

    def _collect_dataset_batteries(self) -> Dict[str, List[dict]]:
        """
        Pool the batteries of the ``train`` and ``test`` splits by source
        dataset, excluding ``EXCLUDE_DATASETS`` and datasets with fewer than
        ``MIN_BATTERIES`` batteries.
        """
        pool: Dict[str, List[dict]] = defaultdict(list)

        for split_name in ("train", "test"):
            try:
                ds_obj = self.dm._split_ds(split_name)
            except (ValueError, AttributeError):
                continue
            for item in ds_obj.items:
                if item["dataset"] in EXCLUDE_DATASETS:
                    continue
                pool[item["dataset"]].append(item)

        return {k: v for k, v in pool.items() if len(v) >= MIN_BATTERIES}

    def _resolve_sweep(self, n_bats: int) -> List[float]:
        """
        Keep the training fractions that leave at least one test battery
        after enforcing ``MIN_TRAIN_BATTERIES``; fall back to ``[0.5]``.
        """
        out = []
        for p in self.sweep:
            n_train = max(MIN_TRAIN_BATTERIES, int(np.ceil(p * n_bats)))
            if n_train < n_bats:  # need at least 1 test battery
                out.append(p)
        if not out:
            # fallback: use 50% only
            out = [0.5]
        return out

    def run(self, verbose: bool = True) -> Dict[str, dict]:
        """
        Run the sweep for every eligible dataset.

        Args:
            verbose: print one table per dataset.

        Returns:
            ``{dataset: {pct: {'mean_mape', 'std_mape', 'mean_mae', 'std_mae',
            'n_train', 'n_test'}}}`` with MAPE as a fraction.
        """
        pool    = self._collect_dataset_batteries()
        results = {}

        for ds_name, items in sorted(pool.items()):
            n_bats = len(items)
            sweep  = self._resolve_sweep(n_bats)
            ds_results = {}

            if verbose:
                print(f"\n{'─'*55}")
                print(f"Dataset: {ds_name}  ({n_bats} batteries)")
                print(f"{'pct':>6}  {'n_train':>7}  {'n_test':>6}  "
                      f"{'MAPE mean':>10}  {'MAPE std':>9}  "
                      f"{'MAE mean':>9}  {'MAE std':>8}")

            # Extract PID features for all batteries in this dataset once
            phi_all, soh_all, mask_all, lens = extract_pid_features(
                items, self.pid, self.device
            )
            # Flatten to cycle-level: (total_valid_cycles, 4*d_h) and (total,)
            # Keep per-battery indices for train/test splitting
            bat_phi_list = []
            bat_soh_list = []
            for i, n in enumerate(lens):
                bat_phi_list.append(phi_all[i, :n])   # (n, 4*d_h)
                bat_soh_list.append(soh_all[i, :n, 0]) # (n,)

            for pct in sweep:
                n_train = max(MIN_TRAIN_BATTERIES, int(np.ceil(pct * n_bats)))
                n_test  = n_bats - n_train
                if n_test < 1:
                    continue

                mapes, maes = [], []

                for rep in range(self.n_repeats):
                    # Random shuffle of battery indices for this repeat
                    rng     = np.random.default_rng(seed=rep * 1000 + int(pct * 100))
                    bat_idx = rng.permutation(n_bats)
                    train_idx = bat_idx[:n_train]
                    test_idx  = bat_idx[n_train:]

                    # Concatenate train cycles
                    phi_tr = torch.cat([bat_phi_list[i] for i in train_idx], dim=0)
                    soh_tr = torch.cat([bat_soh_list[i] for i in train_idx], dim=0)

                    # Concatenate test cycles
                    phi_te = torch.cat([bat_phi_list[i] for i in test_idx], dim=0)
                    soh_te = torch.cat([bat_soh_list[i] for i in test_idx], dim=0)

                    model = train_mlp_head(
                        phi_tr, soh_tr,
                        in_dim=self.in_dim,
                        hidden=self.hidden,
                        n_epochs=self.n_epochs,
                        device=self.device,
                    )
                    mape, mae = eval_mlp_head(model, phi_te, soh_te, self.device)
                    mapes.append(mape)
                    maes.append(mae)

                ds_results[pct] = dict(
                    mean_mape=float(np.mean(mapes)),
                    std_mape=float(np.std(mapes)),
                    mean_mae=float(np.mean(maes)),
                    std_mae=float(np.std(maes)),
                    n_train=n_train,
                    n_test=n_test,
                )

                if verbose:
                    r = ds_results[pct]
                    print(
                        f"{pct*100:>5.0f}%  {r['n_train']:>7d}  {r['n_test']:>6d}  "
                        f"{r['mean_mape']*100:>9.2f}%  {r['std_mape']*100:>8.2f}%  "
                        f"{r['mean_mae']:>9.4f}  {r['std_mae']:>8.4f}"
                    )

            results[ds_name] = ds_results

        return results

    # ── plotting ──────────────────────────────────────────────────────────────

    def plot_sweep(
        self,
        results:   Dict[str, dict],
        save_path: Optional[str] = None,
        dpi:       int = 150,
    ) -> plt.Figure:
        """
        Plot MAPE (mean ± std over repeats) against the training fraction,
        one subplot per dataset.

        Args:
            results: output of :meth:`run`.
            save_path: optional output image path.
            dpi: resolution of the saved figure.

        Returns:
            The matplotlib figure.
        """
        ds_list = sorted(results.keys())
        ncols   = min(len(ds_list), 4)
        nrows   = int(np.ceil(len(ds_list) / ncols))

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.5 * ncols, 3.5 * nrows),
                                 squeeze=False)

        for idx, ds_name in enumerate(ds_list):
            row, col = divmod(idx, ncols)
            ax       = axes[row][col]
            ds_res   = results[ds_name]

            pcts     = sorted(ds_res.keys())
            means    = [ds_res[p]["mean_mape"] * 100 for p in pcts]
            stds     = [ds_res[p]["std_mape"]  * 100 for p in pcts]
            n_trains = [ds_res[p]["n_train"]         for p in pcts]
            pct_lbl  = [f"{p*100:.0f}%"              for p in pcts]

            ax.errorbar(pct_lbl, means, yerr=stds, fmt="o-", color="#1f77b4",
                        capsize=4, lw=2, elinewidth=1.5, label="MAPE ± std")
            ax.set_xlabel("Train % (batteries)", fontsize=9)
            ax.set_ylabel("MAPE (%)", fontsize=9)
            n_total = ds_res[pcts[0]]["n_train"] + ds_res[pcts[0]]["n_test"]
            ax.set_title(f"{ds_name}  (total={n_total} batteries)", fontsize=9)
            ax.grid(alpha=0.3)

            # Annotate with n_train
            for i, (pl, m, n) in enumerate(zip(pct_lbl, means, n_trains)):
                ax.annotate(f"n={n}", (pl, m),
                            textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=7, color="#555")

        # Hide unused axes
        for idx in range(len(ds_list), nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].axis("off")

        fig.suptitle(
            "Per-Dataset MLP Head: MAPE vs Training Battery Fraction",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved → {save_path}")
        return fig

    def plot_mae_sweep(
        self,
        results:   Dict[str, dict],
        save_path: Optional[str] = None,
        dpi:       int = 150,
    ) -> plt.Figure:
        """Same as :meth:`plot_sweep` for the MAE."""
        ds_list = sorted(results.keys())
        ncols   = min(len(ds_list), 4)
        nrows   = int(np.ceil(len(ds_list) / ncols))

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.5 * ncols, 3.5 * nrows),
                                 squeeze=False)

        for idx, ds_name in enumerate(ds_list):
            row, col = divmod(idx, ncols)
            ax       = axes[row][col]
            ds_res   = results[ds_name]
            pcts     = sorted(ds_res.keys())
            means    = [ds_res[p]["mean_mae"] for p in pcts]
            stds     = [ds_res[p]["std_mae"]  for p in pcts]
            pct_lbl  = [f"{p*100:.0f}%"       for p in pcts]

            ax.errorbar(pct_lbl, means, yerr=stds, fmt="s-", color="#d62728",
                        capsize=4, lw=2, elinewidth=1.5, label="MAE ± std")
            ax.set_xlabel("Train % (batteries)", fontsize=9)
            ax.set_ylabel("MAE (SOH units)", fontsize=9)
            n_total = ds_res[pcts[0]]["n_train"] + ds_res[pcts[0]]["n_test"]
            ax.set_title(f"{ds_name}  (total={n_total} batteries)", fontsize=9)
            ax.grid(alpha=0.3)

        for idx in range(len(ds_list), nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].axis("off")

        fig.suptitle(
            "Per-Dataset MLP Head: MAE vs Training Battery Fraction",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved → {save_path}")
        return fig

    def results_table(self, results: Dict[str, dict]) -> str:
        """Format the output of :meth:`run` as a text table (MAPE in %)."""
        lines = [
            f"{'═'*75}",
            "Per-Dataset MLP Head Analysis",
            f"{'─'*75}",
            f"{'Dataset':<20}  {'%':>4}  {'n_tr':>4}  {'n_te':>4}  "
            f"{'MAPE mean':>10}  {'MAPE std':>9}  {'MAE mean':>9}  {'MAE std':>8}",
            f"{'─'*75}",
        ]
        for ds_name in sorted(results.keys()):
            for pct in sorted(results[ds_name].keys()):
                r = results[ds_name][pct]
                lines.append(
                    f"{ds_name:<20}  {pct*100:>3.0f}%  "
                    f"{r['n_train']:>4d}  {r['n_test']:>4d}  "
                    f"{r['mean_mape']*100:>9.2f}%  {r['std_mape']*100:>8.2f}%  "
                    f"{r['mean_mae']:>9.4f}  {r['std_mae']:>8.4f}"
                )
            lines.append(f"{'─'*75}")
        return "\n".join(lines)

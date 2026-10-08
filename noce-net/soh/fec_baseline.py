"""
soh.fec_baseline
================
FEC-only SOH baselines.

These models receive only the normalised FEC index
``c_norm = linspace(0, 1, N)`` of each battery (no recurrent state) and
regress the SOH ``Q_c``. Comparing them with the PID-R heads in
``soh/direct_soh_head.py`` and ``soh/per_dataset_mlp.py`` tests whether
the SOH information extracted from the NOCE-Net states goes beyond the
position of a FEC within the battery life.

Two protocols are provided:

* :func:`run_fec_only_analysis` -- a linear fit and an MLP trained on the
  ``train`` split, evaluated on randomly held-out batteries of the training
  datasets (in-sample) and on the ``test`` split (zero-shot).
* :class:`FECPerDatasetMLPAnalysis` -- the per-dataset sweep over training
  fractions of :class:`soh.per_dataset_mlp.PerDatasetMLPAnalysis` with
  ``c_norm`` as the only input.

Metrics: ``MAPE = mean(|Q_hat - Q| / max(|Q|, 0.05))`` and
``MAE = mean(|Q_hat - Q|)`` over all FECs.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Datasets left out of all FEC-only analyses
EXCLUDE_DATASETS    = {"evTOL", "UMich"}
# Minimum number of batteries for a dataset to enter the per-dataset sweep
MIN_BATTERIES       = 4
# Minimum number of training batteries per split in the per-dataset sweep
MIN_TRAIN_BATTERIES = 2


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction: normalised FEC index only
# ─────────────────────────────────────────────────────────────────────────────

def extract_fec_features(items: List[dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Flatten batteries into FEC-level arrays with ``c_norm`` as the only feature.

    Batteries of ``EXCLUDE_DATASETS`` are skipped.

    Args:
        items: battery dictionaries of :class:`~soh.data_pipeline.BatteryResDataset`
            (``'H'`` is used only for its length N, ``'Q'`` (N,), ``'dataset'``).

    Returns:
        ``(X, y, ds_labels)`` with ``X`` (M, 1) the normalised FEC index in
        [0, 1], ``y`` (M,) the SOH and ``ds_labels`` the dataset name of each
        of the M FECs.
    """
    X_all, y_all, ds_all = [], [], []
    for it in items:
        if it["dataset"] in EXCLUDE_DATASETS:
            continue
        H = it["H"]
        Q = it["Q"]
        N = H.shape[0]
        c_norm = np.linspace(0.0, 1.0, N, dtype=np.float32).reshape(-1, 1)
        Q_np   = Q.detach().cpu().numpy() if isinstance(Q, torch.Tensor) else np.asarray(Q, dtype=np.float32)
        Q_np   = Q_np[:N].astype(np.float32)
        X_all.append(c_norm)
        y_all.append(Q_np)
        ds_all.extend([it["dataset"]] * N)
    if not X_all:
        return (np.zeros((0, 1), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                [])
    return np.vstack(X_all), np.concatenate(y_all), ds_all

# ─────────────────────────────────────────────────────────────────────────────
# Model 1: linear regression
# ─────────────────────────────────────────────────────────────────────────────

def fit_linear_regression(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Least-squares fit ``y = a * x + b`` of a single feature.

    Returns:
        ``(a, b)``.
    """
    coeffs = np.polyfit(X.squeeze(), y, deg=1)
    return float(coeffs[0]), float(coeffs[1])


def predict_linear(X: np.ndarray, a: float, b: float) -> np.ndarray:
    """Evaluate ``a * X + b`` for ``X`` (M, 1); returns (M,)."""
    return a * X.squeeze() + b

# ─────────────────────────────────────────────────────────────────────────────
# Model 2: MLP head (architecture of LightweightSOHMLP without the sigmoid)
# ─────────────────────────────────────────────────────────────────────────────

class FECOnlyMLP(nn.Module):
    """
    Two-hidden-layer MLP mapping ``c_norm`` (1 input) to SOH, used by
    :func:`run_fec_only_analysis`.

    Same layers as :class:`soh.per_dataset_mlp.LightweightSOHMLP` with
    ``in_dim = 1`` and a linear output (no sigmoid).

    Args:
        hidden: width of the first hidden layer.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``c_norm`` (..., 1) to SOH (..., 1)."""
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset sweep with c_norm as the only input
# ─────────────────────────────────────────────────────────────────────────────

def extract_fec_features_per_battery(
    items: List[dict],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Per-battery ``c_norm`` features and SOH labels.

    Args:
        items: battery dictionaries (``'H'`` is used only for its length N,
            ``'Q'`` (N,)).

    Returns:
        Two parallel lists with one entry per battery: ``c_norm`` tensors
        (N, 1) and SOH tensors (N,), so that train/test splits can be made
        at battery level.
    """
    bat_fec_list: List[torch.Tensor] = []
    bat_soh_list: List[torch.Tensor] = []

    for it in items:
        H = it["H"]
        N = H.shape[0]

        # Normalised FEC index: 0 at BOL, 1 at EOL for every battery
        c_norm = torch.linspace(0.0, 1.0, N, dtype=torch.float32).unsqueeze(1)  # (N, 1)

        # SOH labels — handle both tensor and numpy
        Q = it["Q"]
        if isinstance(Q, torch.Tensor):
            soh = Q[:N].float()
        else:
            soh = torch.from_numpy(np.asarray(Q[:N], dtype=np.float32))

        bat_fec_list.append(c_norm)
        bat_soh_list.append(soh)

    return bat_fec_list, bat_soh_list



class FECOnlySOHMLP(nn.Module):
    """
    :class:`soh.per_dataset_mlp.LightweightSOHMLP` with ``in_dim = 1``
    (sigmoid output in (0, 1)), used by :class:`FECPerDatasetMLPAnalysis`.

    Args:
        hidden: width of the first hidden layer.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
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
        """Map ``c_norm`` (..., 1) to SOH (..., 1)."""
        return self.net(x)


def _train_fec_mlp(
    fec_train:  torch.Tensor,   # (N_cycles, 1)
    soh_train:  torch.Tensor,   # (N_cycles,)
    hidden:     int   = 64,
    n_epochs:   int   = 80,
    lr:         float = 1e-3,
    batch_size: int   = 256,
    device:     torch.device = torch.device("cpu"),
) -> FECOnlySOHMLP:
    """
    Train a :class:`FECOnlySOHMLP` with the optimisation settings of
    :func:`soh.per_dataset_mlp.train_mlp_head`; returns the model on the CPU.
    """
    dataset   = TensorDataset(fec_train, soh_train)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model     = FECOnlySOHMLP(hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01
    )
    loss_fn   = nn.MSELoss()

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

    return model.cpu()


@torch.no_grad()
def _eval_fec_mlp(
    model:     FECOnlySOHMLP,
    fec_test:  torch.Tensor,   # (N_cycles, 1)
    soh_test:  torch.Tensor,   # (N_cycles,)
) -> Tuple[float, float]:
    """Return ``(MAPE, MAE)`` over all test FECs (MAPE denominator clipped at 0.05)."""
    model.eval()
    pred = model(fec_test).squeeze().numpy()
    true = soh_test.numpy()
    mape = float(np.mean(np.abs(pred - true) / np.clip(np.abs(true), 0.05, None)))
    mae  = float(np.mean(np.abs(pred - true)))
    return mape, mae


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis class
# ─────────────────────────────────────────────────────────────────────────────

class FECPerDatasetMLPAnalysis:
    """
    Per-dataset sweep over training fractions with ``c_norm`` as the only input.

    Uses the same battery pooling, split sizes and random seeds as
    :class:`soh.per_dataset_mlp.PerDatasetMLPAnalysis`, so that the two
    result dictionaries can be compared entry by entry.

    Args:
        data_module: :class:`~soh.data_pipeline.BatteryDataModule`.
        sweep_pcts: training fractions (default ``[0.1, 0.2, 0.3, 0.4, 0.5]``).
        mlp_hidden: hidden width of the MLP.
        n_epochs: training epochs per split.
        n_repeats: number of random splits per fraction.
        device: ``'auto'`` or a device string.
    """

    DEFAULT_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5]

    def __init__(
        self,
        data_module: object,
        sweep_pcts:  List[float] = None,
        mlp_hidden:  int   = 64,
        n_epochs:    int   = 80,
        n_repeats:   int   = 3,
        device:      str   = "auto",
    ):
        self.dm        = data_module
        self.sweep     = sweep_pcts or self.DEFAULT_SWEEP
        self.hidden    = mlp_hidden
        self.n_epochs  = n_epochs
        self.n_repeats = n_repeats
        self.device    = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else torch.device(device)
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _collect_dataset_batteries(self) -> Dict[str, List[dict]]:
        """Pool the ``train`` and ``test`` batteries by dataset (see :class:`PerDatasetMLPAnalysis`)."""
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
        """Keep fractions that leave at least one test battery; fall back to ``[0.5]``."""
        out = []
        for p in self.sweep:
            n_train = max(MIN_TRAIN_BATTERIES, int(np.ceil(p * n_bats)))
            if n_train < n_bats:
                out.append(p)
        return out or [0.5]

    # ── main runner ───────────────────────────────────────────────────────────

    def run(self, verbose: bool = True) -> Dict[str, dict]:
        """
        Run the sweep for every eligible dataset.

        Returns:
            ``{dataset: {pct: {'mean_mape', 'std_mape', 'mean_mae', 'std_mae',
            'n_train', 'n_test'}}}`` with MAPE as a fraction.
        """
        pool    = self._collect_dataset_batteries()
        results = {}

        for ds_name, items in sorted(pool.items()):
            n_bats = len(items)
            sweep  = self._resolve_sweep(n_bats)
            ds_results: dict = {}

            if verbose:
                print(f"\n{'─'*60}")
                print(f"Dataset: {ds_name}  ({n_bats} batteries)  [FEC-only input]")
                print(f"{'pct':>6}  {'n_train':>7}  {'n_test':>6}  "
                      f"{'MAPE mean':>10}  {'MAPE std':>9}  "
                      f"{'MAE mean':>9}  {'MAE std':>8}")

            # Extract per-battery FEC features once for the whole dataset
            bat_fec_list, bat_soh_list = extract_fec_features_per_battery(items)

            for pct in sweep:
                n_train = max(MIN_TRAIN_BATTERIES, int(np.ceil(pct * n_bats)))
                n_test  = n_bats - n_train
                if n_test < 1:
                    continue

                mapes, maes = [], []

                for rep in range(self.n_repeats):
                    rng       = np.random.default_rng(seed=rep * 1000 + int(pct * 100))
                    bat_idx   = rng.permutation(n_bats)
                    train_idx = bat_idx[:n_train]
                    test_idx  = bat_idx[n_train:]

                    fec_tr = torch.cat([bat_fec_list[i] for i in train_idx], dim=0)
                    soh_tr = torch.cat([bat_soh_list[i] for i in train_idx], dim=0)
                    fec_te = torch.cat([bat_fec_list[i] for i in test_idx],  dim=0)
                    soh_te = torch.cat([bat_soh_list[i] for i in test_idx],  dim=0)

                    model = _train_fec_mlp(
                        fec_tr, soh_tr,
                        hidden=self.hidden,
                        n_epochs=self.n_epochs,
                        lr=1e-3,
                        device=self.device,
                    )
                    mape, mae = _eval_fec_mlp(model, fec_te, soh_te)
                    mapes.append(mape)
                    maes.append(mae)

                ds_results[pct] = dict(
                    mean_mape=float(np.mean(mapes)),
                    std_mape =float(np.std(mapes)),
                    mean_mae =float(np.mean(maes)),
                    std_mae  =float(np.std(maes)),
                    n_train  =n_train,
                    n_test   =n_test,
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

    # ── comparison table ──────────────────────────────────────────────────────

    def print_comparison_table(
        self,
        fec_results:       Dict[str, dict],
        pid_results:       Optional[Dict[str, dict]] = None,
        target_pct:        float = 0.3,
    ) -> None:
        """
        Print the FEC-only sweep at one training fraction, optionally next to
        the PID-R sweep.

        Args:
            fec_results: output of :meth:`run`.
            pid_results: output of :meth:`soh.per_dataset_mlp.PerDatasetMLPAnalysis.run`
                (optional). When given, the MAPE difference
                ``FEC-only - PID-R`` is printed per dataset.
            target_pct: training fraction to report.
        """
        sep = "─" * 88
        pct_label = f"{target_pct*100:.0f}%"
        print(f"\n{'═'*88}")
        print(f"Per-Dataset MLP at {pct_label} training fraction | FEC-only vs PID features")
        print(f"{'═'*88}")

        if pid_results is not None:
            print(f"{'Dataset':<16}  "
                  f"{'FEC-MLP MAPE (±std)':>21}  "
                  f"{'PID MAPE':>10}  {'Δ FEC-MLP':>10}")
            print(sep)
        else:
            print(f"{'Dataset':<16}  {'FEC-MLP MAPE':>13}  {'FEC-MLP MAE':>12}  "
                  f"{'n_train':>8}  {'n_test':>7}")
            print(sep)

        for ds_name in sorted(fec_results.keys()):
            fec_r = fec_results[ds_name].get(target_pct)
            if fec_r is None:
                continue

            fec_mape = fec_r["mean_mape"] * 100
            fec_std  = fec_r["std_mape"]  * 100
            fec_mae  = fec_r["mean_mae"]

            if pid_results is not None:
                pid_r    = (pid_results.get(ds_name) or {}).get(target_pct)
                pid_mape = pid_r["mean_mape"] * 100 if pid_r else float("nan")
                delta    = fec_mape - pid_mape
                print(f"{ds_name:<16}  "
                      f"{fec_mape:>10.2f}% (±{fec_std:.1f})  "
                      f"{pid_mape:>9.2f}%  "
                      f"{delta:>+10.1f}%")
            else:
                print(f"{ds_name:<16}  "
                      f"{fec_mape:>10.2f}% (±{fec_std:.1f})  "
                      f"{fec_mae:>12.4f}  "
                      f"{fec_r['n_train']:>8d}  {fec_r['n_test']:>7d}")

        print(sep)
        print("\nPositive Δ = FEC-MLP is worse than PID head → hidden states add information")
        print("Negative Δ = FEC-MLP is better → PID head underperforms on this dataset")

    # ── plotting ──────────────────────────────────────────────────────────────

    def plot_sweep(
        self,
        results:        Dict[str, dict],
        pid_results:    Optional[Dict[str, dict]] = None,
        save_path:      Optional[str] = None,
        dpi:            int = 150,
    ) -> plt.Figure:
        """
        Plot MAPE (mean ± std) against the training fraction, one subplot per
        dataset, optionally overlaying the PID-R sweep.

        Args:
            results: output of :meth:`run`.
            pid_results: optional output of ``PerDatasetMLPAnalysis.run``.
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

            pcts   = sorted(ds_res.keys())
            means  = [ds_res[p]["mean_mape"] * 100 for p in pcts]
            stds   = [ds_res[p]["std_mape"]  * 100 for p in pcts]
            labels = [f"{p*100:.0f}%"              for p in pcts]

            ax.errorbar(labels, means, yerr=stds, fmt="s--", color="#d62728",
                        capsize=4, lw=2, elinewidth=1.5, label="FEC-only MLP")

            if pid_results and ds_name in pid_results:
                pid_ds  = pid_results[ds_name]
                p_pcts  = sorted(pid_ds.keys())
                p_means = [pid_ds[p]["mean_mape"] * 100 for p in p_pcts]
                p_stds  = [pid_ds[p]["std_mape"]  * 100 for p in p_pcts]
                p_labels= [f"{p*100:.0f}%"              for p in p_pcts]
                ax.errorbar(p_labels, p_means, yerr=p_stds, fmt="o-", color="#1f77b4",
                            capsize=4, lw=2, elinewidth=1.5, label="PID MLP")
                ax.legend(fontsize=7)

            n_total = ds_res[pcts[0]]["n_train"] + ds_res[pcts[0]]["n_test"]
            ax.set_title(f"{ds_name}  (n={n_total})", fontsize=9)
            ax.set_xlabel("Train % (batteries)", fontsize=9)
            ax.set_ylabel("MAPE (%)", fontsize=9)
            ax.grid(alpha=0.3)

        for idx in range(len(ds_list), nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].axis("off")

        fig.suptitle(
            "Per-Dataset SOH MLP: FEC-only (red) vs PID features (blue)",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved → {save_path}")
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# Single model trained on the training datasets
# ─────────────────────────────────────────────────────────────────────────────

def train_fec_mlp(
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    hidden:     int   = 64,
    n_epochs:   int   = 200,
    lr:         float = 1e-3,
    batch_size: int   = 256,
    device:     str   = "auto",
    verbose:    bool  = False,
) -> FECOnlyMLP:
    """
    Train a :class:`FECOnlyMLP` on FEC-level ``(c_norm, Q)`` pairs.

    Adam with cosine annealing to ``0.01 * lr``, MSE loss and gradient
    clipping at 1.0.

    Args:
        X_train: (M, 1) normalised FEC indices.
        y_train: (M,) SOH labels.
        hidden, n_epochs, lr, batch_size: model and optimisation settings.
        device: ``'auto'`` or a device string.
        verbose: print the loss every 50 epochs.

    Returns:
        The trained model on the CPU.
    """
    dev = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
           if device == "auto" else torch.device(device))

    X_t = torch.from_numpy(X_train.astype(np.float32))
    y_t = torch.from_numpy(y_train.astype(np.float32))
    ds  = TensorDataset(X_t, y_t)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model     = FECOnlyMLP(hidden=hidden).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01
    )
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(n_epochs):
        ep_loss = 0.0
        n_batches = 0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            n_batches += 1
        scheduler.step()
        if verbose and (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1:>3d}  loss={ep_loss/max(n_batches,1):.4f}")

    return model.cpu()


@torch.no_grad()
def predict_fec_mlp(model: FECOnlyMLP, X: np.ndarray) -> np.ndarray:
    """Predict SOH (M,) for ``X`` (M, 1) with a CPU model."""
    model.eval()
    X_t = torch.from_numpy(X.astype(np.float32))
    return model(X_t).squeeze(-1).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics per dataset
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 0.05) -> Tuple[float, float]:
    """Return ``(MAPE, MAE)``; the MAPE denominator is ``max(|y_true|, eps)``. NaNs for empty input."""
    if len(y_true) == 0:
        return float("nan"), float("nan")
    mape = float(np.mean(np.abs(y_pred - y_true) / np.clip(np.abs(y_true), eps, None)))
    mae  = float(np.mean(np.abs(y_pred - y_true)))
    return mape, mae


def metrics_by_dataset(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    ds_labels: List[str],
) -> Dict[str, Tuple[float, float]]:
    """Return ``{dataset: (MAPE, MAE)}`` plus the pooled entry ``'Total'``."""
    ds_arr = np.asarray(ds_labels)
    out: Dict[str, Tuple[float, float]] = {}
    for ds in sorted(set(ds_labels)):
        mask = ds_arr == ds
        out[ds] = _metrics(y_true[mask], y_pred[mask])
    out["Total"] = _metrics(y_true, y_pred)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_fec_only_analysis(
    data_module,
    in_sample_test_frac: float = 0.2,
    n_repeats:           int   = 3,
    mlp_hidden:          int   = 64,
    mlp_epochs:          int   = 200,
    mlp_lr:              float = 1e-3,
    seed:                int   = 42,
    verbose:             bool  = True,
) -> Dict[str, Dict]:
    """
    Linear and MLP FEC-only baselines trained on the training datasets.

    In every repeat, ``in_sample_test_frac`` of the batteries of each
    dataset in the ``train`` split (at least one) are held out at random.
    Both models are fitted on the remaining batteries and evaluated on the
    held-out batteries (in-sample) and on the ``test`` split (zero-shot).
    Datasets in ``EXCLUDE_DATASETS`` are ignored.

    Args:
        data_module: :class:`~soh.data_pipeline.BatteryDataModule`.
        in_sample_test_frac: fraction of batteries held out per dataset.
        n_repeats: number of random splits.
        mlp_hidden, mlp_epochs, mlp_lr: MLP settings.
        seed: seed of the random generator used for the splits.
        verbose: print per-repeat results and the final table.

    Returns:
        ``{'linear': {'in_sample', 'zero_shot', 'params_mean'},
        'mlp': {'in_sample', 'zero_shot'}, 'in_sample_frac'}``, where every
        ``in_sample``/``zero_shot`` entry maps a dataset name (and
        ``'Total'``) to ``{'mape_mean', 'mape_std', 'mae_mean', 'mae_std'}``
        over repeats and ``params_mean`` is the mean ``(a, b)`` of the linear fit.
    """
    rng = np.random.default_rng(seed)

    # ── Collect train and test items ──────────────────────────────────────
    train_ds = data_module._split_ds("train")
    test_ds  = data_module._split_ds("test")

    train_items = [it for it in train_ds.items
                   if it["dataset"] not in EXCLUDE_DATASETS]
    test_items  = [it for it in test_ds.items
                   if it["dataset"] not in EXCLUDE_DATASETS]

    # ── Repeat over random splits of the training set ─────────────────────
    linear_in_sample_runs: List[Dict[str, Tuple[float, float]]] = []
    linear_zero_shot_runs: List[Dict[str, Tuple[float, float]]] = []
    linear_params_runs:    List[Tuple[float, float]]            = []

    mlp_in_sample_runs:    List[Dict[str, Tuple[float, float]]] = []
    mlp_zero_shot_runs:    List[Dict[str, Tuple[float, float]]] = []

    for rep in range(n_repeats):
        # Split training items per dataset into train/in-sample-test
        by_ds: Dict[str, List[dict]] = defaultdict(list)
        for it in train_items:
            by_ds[it["dataset"]].append(it)

        train_fit_items: List[dict]     = []
        in_sample_items: List[dict]     = []
        for ds, its in by_ds.items():
            idx = rng.permutation(len(its))
            n_test = max(1, int(round(in_sample_test_frac * len(its))))
            in_sample_items.extend([its[i] for i in idx[:n_test]])
            train_fit_items.extend( [its[i] for i in idx[n_test:]])

        X_tr, y_tr, ds_tr = extract_fec_features(train_fit_items)
        X_is, y_is, ds_is = extract_fec_features(in_sample_items)
        X_te, y_te, ds_te = extract_fec_features(test_items)

        # ── Linear regression ─────────────────────────────────────────────
        a, b = fit_linear_regression(X_tr, y_tr)
        linear_params_runs.append((a, b))

        y_is_pred = predict_linear(X_is, a, b)
        y_te_pred = predict_linear(X_te, a, b)
        linear_in_sample_runs.append(metrics_by_dataset(y_is, y_is_pred, ds_is))
        linear_zero_shot_runs.append(metrics_by_dataset(y_te, y_te_pred, ds_te))

        # ── MLP head ──────────────────────────────────────────────────────
        model = train_fec_mlp(
            X_tr, y_tr,
            hidden=mlp_hidden,
            n_epochs=mlp_epochs,
            lr=mlp_lr,
            verbose=False,
        )
        y_is_pred = predict_fec_mlp(model, X_is)
        y_te_pred = predict_fec_mlp(model, X_te)
        mlp_in_sample_runs.append(metrics_by_dataset(y_is, y_is_pred, ds_is))
        mlp_zero_shot_runs.append(metrics_by_dataset(y_te, y_te_pred, ds_te))

        if verbose:
            print(f"  [rep {rep+1}/{n_repeats}]  linear a={a:+.4f} b={b:+.4f}  "
                  f"| linear Total zero-shot: MAPE={linear_zero_shot_runs[-1]['Total'][0]*100:.2f}%  "
                  f"| MLP Total zero-shot: MAPE={mlp_zero_shot_runs[-1]['Total'][0]*100:.2f}%")

    # ── Aggregate across repetitions ──────────────────────────────────────
    def _agg(runs: List[Dict[str, Tuple[float, float]]]) -> Dict[str, Dict[str, float]]:
        """Mean and std of (MAPE, MAE) over repeats for every dataset key."""
        keys = sorted(set().union(*[r.keys() for r in runs]))
        agg: Dict[str, Dict[str, float]] = {}
        for k in keys:
            mapes = np.array([r[k][0] for r in runs if k in r])
            maes  = np.array([r[k][1] for r in runs if k in r])
            agg[k] = dict(
                mape_mean=float(np.nanmean(mapes)),
                mape_std =float(np.nanstd(mapes)),
                mae_mean =float(np.nanmean(maes)),
                mae_std  =float(np.nanstd(maes)),
            )
        return agg

    results = {
        "linear": {
            "in_sample": _agg(linear_in_sample_runs),
            "zero_shot": _agg(linear_zero_shot_runs),
            "params_mean": (
                float(np.mean([p[0] for p in linear_params_runs])),
                float(np.mean([p[1] for p in linear_params_runs])),
            ),
        },
        "mlp": {
            "in_sample": _agg(mlp_in_sample_runs),
            "zero_shot": _agg(mlp_zero_shot_runs),
        },
        "in_sample_frac": in_sample_test_frac,
    }

    if verbose:
        print_results_table(results)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results: Dict[str, Dict]) -> None:
    """Print the in-sample and zero-shot MAPE/MAE of both models (output of :func:`run_fec_only_analysis`)."""
    print("\n" + "=" * 90)
    print("FEC-ONLY SOH BASELINE  |  Linear & MLP models trained on c_norm ∈ [0,1] only")
    print("=" * 90)

    # ── IN-SAMPLE ─────────────────────────────────────────────────────────
    print("\n─── IN-SAMPLE (held-out batteries from training datasets) ───")
    ds_list = sorted(k for k in results["linear"]["in_sample"].keys() if k != "Total")
    header  = f"{'Model':<20}  " + "  ".join([f"{ds:>7}" for ds in ds_list + ["Total"]])
    print(header)
    print("─" * len(header))

    for name, key in [("Linear (MAPE)", "linear"),
                      ("MLP (MAPE)",    "mlp")]:
        parts = [f"{name:<20}"]
        for ds in ds_list + ["Total"]:
            r = results[key]["in_sample"][ds]
            parts.append(f"{r['mape_mean']*100:>6.2f}%")
        print("  ".join(parts))

    for name, key in [("Linear (MAE)",  "linear"),
                      ("MLP (MAE)",     "mlp")]:
        parts = [f"{name:<20}"]
        for ds in ds_list + ["Total"]:
            r = results[key]["in_sample"][ds]
            parts.append(f"{r['mae_mean']:>7.4f}")
        print("  ".join(parts))

    # ── ZERO-SHOT ─────────────────────────────────────────────────────────
    print("\n─── ZERO-SHOT (held-out test datasets, never seen in training) ───")
    ds_list_zs = sorted(k for k in results["linear"]["zero_shot"].keys() if k != "Total")
    header = f"{'Model':<20}  " + "  ".join([f"{ds:>7}" for ds in ds_list_zs + ["Total"]])
    print(header)
    print("─" * len(header))

    for name, key in [("Linear (MAPE)", "linear"),
                      ("MLP (MAPE)",    "mlp")]:
        parts = [f"{name:<20}"]
        for ds in ds_list_zs + ["Total"]:
            r = results[key]["zero_shot"][ds]
            parts.append(f"{r['mape_mean']*100:>6.2f}%")
        print("  ".join(parts))

    for name, key in [("Linear (MAE)",  "linear"),
                      ("MLP (MAE)",     "mlp")]:
        parts = [f"{name:<20}"]
        for ds in ds_list_zs + ["Total"]:
            r = results[key]["zero_shot"][ds]
            parts.append(f"{r['mae_mean']:>7.4f}")
        print("  ".join(parts))

    a, b = results["linear"]["params_mean"]
    print(f"\nLinear regression fit (mean across repeats): SOH(c) = {a:+.4f} * c_norm + {b:+.4f}")
    print(f"In-sample held-out fraction: {results['in_sample_frac']*100:.0f}%")
    print("=" * 90)
    print("\nCompare with the DirectSOHHead results (PID-R features of the recurrent states).")

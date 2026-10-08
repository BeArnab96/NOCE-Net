"""
soh.diagnostics
===============
Rank-correlation diagnostics of the NOCE-Net recurrent states (rank-correlation
analysis and Spearman diagnostics of the paper).

For every battery, Spearman ``|rho|`` is computed between the FEC index
``c`` (or the SOH trajectory) and four scalar summaries of the states:
``||h_c||_2``, ``PC1(H)``, and their relativised versions computed on
``(H - H[0])[1:]``. :class:`DiagnosticsRunner` aggregates the values per
source dataset and split; :func:`plot_relative_signals` and
:func:`plot_pca_trajectories` draw the two panels of the representation
diagnostics figure.
"""
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from soh.data_pipeline import BatteryDataModule, _to_numpy

#: Mapping from ``Bat_dataset`` values to the dataset names used in the paper.
EXP_TO_PAPER_NAME = {'Severson': 'MATR',
                     'Unknown': 'HUST',
                     'NatComm': 'TJU',
                     'Stanford': 'Stanford-EV',
                     'NASA PCoE': 'NASA-PCoE',
                     'XJTU_Full': 'XJTU-Full',
                     'evTOL': 'evTOL',
                     'Iowa_Uni': 'ISU_ILCC',
                     'XJTU_Partial': 'XJTU-Partial',
                     'NASA Rand': 'NASA-Rand'}


def spearman_monotonicity_check(
    H: np.ndarray,
    return_pc1: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Checks whether the GRU hidden states encode a monotone degradation signal
    by computing the Spearman rank correlation between the cycle index c and
    either (a) ||h_c||_2  or (b) the first principal component of H.

    A |rho| > 0.8 across test batteries suggests the representation encodes
    degradation direction even if the absolute capacity scale is off.

    Args:
        H           : (N, d_h) array of hidden states for one battery
        return_pc1  : if True, also correlate against PC1 of H (recommended)
        verbose     : print a summary

    Returns:
        dict with keys:
            'rho_norm'   : Spearman rho between cycle index and ||h_c||_2
            'p_norm'     : p-value for rho_norm
            'rho_pc1'    : Spearman rho between cycle index and PC1 score
                           (None if return_pc1=False)
            'p_pc1'      : p-value for rho_pc1
            'pc1_scores' : (N,) PC1 projection of H (useful for plotting)
            'norms'      : (N,) L2 norms of each h_c
    """
    if isinstance(H, torch.Tensor):
        H = H.detach().cpu().numpy()

    N = H.shape[0]
    cycle_idx = np.arange(N)

    # --- (a) L2 norm correlation -------------------------------------------
    norms = np.linalg.norm(H, axis=1)          # (N,)
    rho_norm, p_norm = spearmanr(cycle_idx, norms)

    # --- (b) PC1 correlation --------------------------------------------------
    rho_pc1, p_pc1, pc1_scores = None, None, None
    if return_pc1:
        pca = PCA(n_components=1)
        pc1_scores = pca.fit_transform(H).squeeze()   # (N,)
        rho_pc1, p_pc1 = spearmanr(cycle_idx, pc1_scores)

    if verbose:
        print(f"[Monotonicity Check] N={N} cycles")
        print(f"  ||h_c||₂  vs cycle index : rho={rho_norm:+.3f}  p={p_norm:.2e}")
        if return_pc1:
            print(f"  PC1(H)    vs cycle index : rho={rho_pc1:+.3f}  p={p_pc1:.2e}")
        if abs(rho_norm) > 0.8 or (rho_pc1 is not None and abs(rho_pc1) > 0.8):
            print("  ✓ Strong monotone signal — representation encodes degradation.")
        else:
            print("  ✗ Weak monotone signal — states may not track degradation well.")

    return dict(
        rho_norm=rho_norm, p_norm=p_norm,
        rho_pc1=rho_pc1,   p_pc1=p_pc1,
        pc1_scores=pc1_scores, norms=norms,
    )


def plot_monotonicity(
    H_list: list,
    labels: Optional[list] = None,
    use_pc1: bool = True,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """
    Visualises the degradation trajectory encoded in H for multiple batteries.

    Produces two subplots:
        Left  : PC1 (or ||h_c||) trajectory per battery, coloured by dataset
        Right : 2-D PCA of all hidden states, coloured by normalised cycle %

    Args:
        H_list  : list of (N_i, d_h) arrays, one per battery
        labels  : optional list of string labels (e.g. dataset name per battery)
        use_pc1 : use PC1 projection; if False, use L2 norm
        figsize : figure size

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Left: trajectory per battery ----------------------------------------
    ax = axes[0]
    cmap = matplotlib.colormaps["tab10"]
    all_H, all_pct = [], []

    for i, H in enumerate(H_list):
        if isinstance(H, torch.Tensor):
            H = H.detach().cpu().numpy()
        N = H.shape[0]
        pct = np.linspace(0, 100, N)

        if use_pc1:
            pca = PCA(n_components=1)
            signal = pca.fit_transform(H).squeeze()
            ylabel = "PC1 of h_c"
        else:
            signal = np.linalg.norm(H, axis=1)
            ylabel = "||h_c||₂"

        color = cmap(i % 10)
        lbl = labels[i] if labels else f"Battery {i}"
        ax.plot(pct, signal, alpha=0.7, color=color, label=lbl, linewidth=1.2)

        all_H.append(H)
        all_pct.append(pct)

    ax.set_xlabel("Cycle % (0=BOL, 100=EOL)")
    ax.set_ylabel(ylabel)
    ax.set_title("Degradation signal in hidden states")
    if labels:
        ax.legend(fontsize=7, ncol=2)

    # --- Right: 2-D PCA coloured by cycle % ----------------------------------
    ax2 = axes[1]
    H_all = np.vstack(all_H)
    pct_all = np.concatenate(all_pct)

    pca2 = PCA(n_components=2)
    coords = pca2.fit_transform(H_all)

    sc = ax2.scatter(coords[:, 0], coords[:, 1], c=pct_all,
                     cmap="plasma", s=4, alpha=0.6)
    plt.colorbar(sc, ax=ax2, label="Cycle %")
    ax2.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
    ax2.set_title("2-D PCA of all hidden states\n(colour = cycle %)")

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Step-1 Diagnostics runner
# ─────────────────────────────────────────────────────────────────────────────

class DiagnosticsRunner:
    """
    Runs the Spearman monotonicity diagnostics over any combination of
    splits and source datasets.

    Usage
    -----
    runner = DiagnosticsRunner(data_module)
    results = runner.run(splits=["train","val","test"])
    fig     = runner.plot(results, save_path="diagnostics.png")
    """

    def __init__(self, data_module: BatteryDataModule):
        self.dm = data_module

    # ── core check for a single battery ─────────────────────────────────────

    @staticmethod
    def _check_one(H: np.ndarray, Q: np.ndarray) -> dict:
        """
        Returns Spearman ρ for:
            norm    : ‖h_c‖₂ vs cycle index
            pc1     : PC1(H)  vs cycle index
            Q_raw   : Q       vs cycle index
            Q_rel   : Q-Q[0]  vs cycle index  (for raw H case)
        and for the relative version:
            norm_rel, pc1_rel computed on (H-h0)[1:]
        """
        N         = H.shape[0]
        c_idx     = np.arange(N)
        c_idx_rel = np.arange(N - 1)

        # --- raw H -----------------------------------------------------------
        norms        = np.linalg.norm(H, axis=1)
        rho_norm, _  = spearmanr(c_idx, norms)

        pca          = PCA(n_components=1)
        pc1          = pca.fit_transform(H).squeeze()
        if np.corrcoef(pc1, np.arange(len(pc1)))[0,1] < 0:
          pc1 = -pc1
        rho_pc1, _   = spearmanr(c_idx, pc1)

        rho_q,   _   = spearmanr(c_idx, Q)
        rho_qrel,_   = spearmanr(c_idx, Q - Q[0])

        # --- relative H = (H - h0)[1:] ---------------------------------------
        H_rel        = (H - H[0:1])[1:]
        norms_rel    = np.linalg.norm(H_rel, axis=1)
        rho_norm_rel,_ = spearmanr(c_idx_rel, norms_rel)

        pc1_rel      = pca.fit_transform(H_rel).squeeze() if len(H_rel) > 1 else np.zeros(1)
        if np.corrcoef(pc1_rel, np.arange(len(pc1_rel)))[0,1] < 0:
          pc1_rel = -pc1_rel
        rho_pc1_rel, _ = spearmanr(c_idx_rel, pc1_rel)

        rho_q_hnorm, _ = spearmanr(Q, norms)
        rho_q_hpc1, _  = spearmanr(Q, pc1)
        rho_q_hnorm_rel, _ = spearmanr(Q[1:], norms_rel)
        rho_q_hpc1_rel, _  = spearmanr(Q[1:], pc1_rel)

        rho_qrel_hnorm,_ = spearmanr(Q - Q[0], norms)
        rho_qrel_hpc1,_ = spearmanr(Q - Q[0], pc1)
        rho_qrel_hnorm_rel,_ = spearmanr((Q - Q[0])[1:], norms_rel)
        rho_qrel_hpc1_rel,_ = spearmanr((Q - Q[0])[1:], pc1_rel)

        return dict(
            rho_norm=rho_norm, rho_pc1=rho_pc1,
            rho_q=rho_q,       rho_qrel=rho_qrel,
            rho_norm_rel=rho_norm_rel, rho_pc1_rel=rho_pc1_rel,
            pc1=pc1, norms=norms, Q=Q,
            pc1_rel=pc1_rel, norms_rel=norms_rel, Q_rel=Q - Q[0],

            rho_q_hnorm=rho_q_hnorm, rho_q_hpc1=rho_q_hpc1,
            rho_q_hnorm_rel=rho_q_hnorm_rel, rho_q_hpc1_rel=rho_q_hpc1_rel,

            rho_qrel_hnorm=rho_qrel_hnorm, rho_qrel_hpc1=rho_qrel_hpc1,
            rho_qrel_hnorm_rel=rho_qrel_hnorm_rel, rho_qrel_hpc1_rel=rho_qrel_hpc1_rel,
            N=N,
        )

    # ── aggregate over a dataset or a full split ─────────────────────────────

    def run(
        self,
        splits: List[str] = ("train", "val", "test"),
        verbose: bool = True,
    ) -> dict:
        """
        Run diagnostics for every split × every source dataset, plus
        aggregated results over each full split.

        Returns a nested dict:
            results[split][dataset_name] = {
                'batteries' : list of per-battery check dicts,
                'mean_rho_norm':    float,
                'mean_rho_pc1':     float,
                'mean_rho_norm_rel':float,
                'mean_rho_pc1_rel': float,
                'mean_rho_q':       float,
                'mean_rho_qrel':    float,

                'mean_rho_q_hnorm': float,
                'mean_rho_q_hpc1':  float,
                'mean_rho_q_hnorm_rel': float,
                'mean_rho_q_hpc1_rel':  float,
                'mean_rho_qrel_hnorm':  float,
                'mean_rho_qrel_hpc1':   float,
                'mean_rho_qrel_hnorm_rel': float,
                'mean_rho_qrel_hpc1_rel': float,
            }
            results[split]['__all__'] = same structure over entire split
        """
        results = {}

        for split in splits:
            try:
                ds = self.dm._split_ds(split)
            except (ValueError, AttributeError):
                continue

            results[split] = {}

            # Collect per-battery checks grouped by source dataset
            all_checks: List[dict] = []
            per_ds_checks: Dict[str, List[dict]] = defaultdict(list)

            for item in ds.items:
                H = _to_numpy(item["H"])
                Q = _to_numpy(item["Q"])
                chk = self._check_one(H, Q)
                chk["dataset"] = item["dataset"]
                chk["battery_idx"] = item["battery_idx"]
                all_checks.append(chk)
                per_ds_checks[item["dataset"]].append(chk)

            # Summarise per source dataset
            for ds_name, checks in per_ds_checks.items():
                results[split][ds_name] = self._summarise(checks)

            # Summarise over entire split
            results[split]["__all__"] = self._summarise(all_checks)

            if verbose:
                self._print_table(split, results[split])

        return results

    @staticmethod
    def _summarise(checks: List[dict]) -> dict:
        """Mean, median and minimum of ``|rho|`` over a list of per-battery checks."""
        keys = ["rho_norm", "rho_pc1", "rho_q", "rho_qrel",
                "rho_norm_rel", "rho_pc1_rel",
                'rho_q_hnorm', 'rho_q_hpc1', 'rho_q_hnorm_rel', 'rho_q_hpc1_rel',
                'rho_qrel_hnorm', 'rho_qrel_hpc1', 'rho_qrel_hnorm_rel', 'rho_qrel_hpc1_rel']
        summary = {"batteries": checks}
        for k in keys:
            vals = [abs(c[k]) for c in checks if not np.isnan(c[k])]
            summary[f"mean_{k}"]   = float(np.mean(vals))   if vals else float("nan")
            summary[f"median_{k}"] = float(np.median(vals)) if vals else float("nan")
            summary[f"min_{k}"]    = float(np.min(vals))    if vals else float("nan")
        return summary

    @staticmethod
    def _print_table(split: str, split_results: dict):
        """Print the mean ``|rho|`` values of one split, one row per dataset."""
        header = (f"\n{'─'*80}\n"
                  f"Diagnostics — split: {split.upper()}\n"
                  f"{'─'*80}\n"
                  f"{'Dataset':<22} {'N':>4}  "
                  f"{'|ρ| ‖h‖':>8}  {'|ρ| PC1':>8}  "
                  f"{'|ρ|‖h-h₀‖':>10}  {'|ρ|PC1rel':>10}  "
                  f"{'|ρ| Q':>8}  {'|ρ| ΔQ':>8}"
                  f"{'|ρ| Q-‖h‖':>8}  {'|ρ| ΔQ-‖h‖':>8}"
                  f"{'|ρ| Q-PC1':>8}  {'|ρ| ΔQ-PC1':>8}"
                  f"{'|ρ| Q-‖h-h₀‖':>8}  {'|ρ| ΔQ-‖h-h₀‖':>8}"
                  f"{'|ρ| Q-PC1rel':>8}  {'|ρ| ΔQ-PC1rel':>8}"

                  )
        print(header)
        order = sorted(k for k in split_results if k != "__all__")
        order.append("__all__")
        for name in order:
            r = split_results[name]
            n = len(r["batteries"])
            sep = "═" * 80 if name == "__all__" else ""
            if sep:
                print(sep)
            print(
                f"{name:<22} {n:>4}  "
                f"{r['mean_rho_norm']:>8.3f}  {r['mean_rho_pc1']:>8.3f}  "
                f"{r['mean_rho_norm_rel']:>10.3f}  {r['mean_rho_pc1_rel']:>10.3f}  "
                f"{r['mean_rho_q']:>8.3f}  {r['mean_rho_qrel']:>8.3f}"
                f"{r['mean_rho_q_hnorm']:>8.3f}  {r['mean_rho_qrel_hnorm']:>8.3f}"
                f"{r['mean_rho_q_hpc1']:>8.3f}  {r['mean_rho_qrel_hpc1']:>8.3f}"
                f"{r['mean_rho_q_hnorm_rel']:>8.3f}  {r['mean_rho_qrel_hnorm_rel']:>8.3f}"
                f"{r['mean_rho_q_hpc1_rel']:>8.3f}  {r['mean_rho_qrel_hpc1_rel']:>8.3f}"

            )

    # ── plotting ─────────────────────────────────────────────────────────────

    def plot(
        self,
        results: dict,
        splits_to_plot: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> plt.Figure:
        """
        Produces a figure with one row per split.
        Each row: [PC1 traj | ‖h‖ traj | Q traj | 2-D PCA scatter]
        for both raw H and relative (H-h₀)[1:] versions side by side.

        Parameters
        ----------
        results       : output of self.run()
        splits_to_plot: which splits to include (default: all in results)
        save_path     : if given, save figure here
        dpi           : figure DPI

        Returns
        -------
        matplotlib Figure
        """
        exp_to_paper_name = EXP_TO_PAPER_NAME

        if splits_to_plot is None:
            splits_to_plot = [s for s in ("train", "val", "test") if s in results]

        n_splits = len(splits_to_plot)
        if n_splits == 0:
            raise ValueError("No splits to plot.")

        # 2 versions (raw / relative) × 4 columns × n_splits rows
        n_rows = n_splits * 2
        fig    = plt.figure(figsize=(20, 4 * n_rows))
        gs     = gridspec.GridSpec(n_rows, 4, figure=fig,
                                   hspace=0.50, wspace=0.35)

        cmap_ds  = matplotlib.colormaps["tab20"]
        cmap_pct = "plasma"

        col_titles = [
            "PC1(H) vs cycle %",
            "‖h_c‖₂ vs cycle %",
            "Capacity vs cycle %",
            "2-D PCA (colour=cycle%)",
        ]

        for s_idx, split in enumerate(splits_to_plot):
            split_res  = results[split]
            all_checks = split_res["__all__"]["batteries"]

            # Consistent colour per source dataset
            ds_names = sorted({c["dataset"] for c in all_checks})
            ds_color = {n: cmap_ds(i / max(len(ds_names) - 1, 1))
                        for i, n in enumerate(ds_names)}

            # Pre-fetch H arrays from the dataset (avoids re-searching in inner loop)
            ds_obj   = self.dm._split_ds(split)
            # Build lookup: (battery_idx, dataset) -> item
            item_lookup = {
                (it["battery_idx"], it["dataset"]): it
                for it in ds_obj.items
            }

            for v_idx, version in enumerate(("raw", "relative")):
                row = s_idx * 2 + v_idx
                v_label = "H" if version == "raw" else "H−h₀"

                # Pre-create the four axes for this row so that they can be annotated
                axs = [fig.add_subplot(gs[row, c]) for c in range(4)]

                H_all_list, pct_all_list = [], []

                for chk in all_checks:
                    N_full = chk["N"]
                    color  = ds_color[chk["dataset"]]

                    if version == "raw":
                        pc1_sig  = chk["pc1"]          # (N,)
                        norm_sig = chk["norms"]         # (N,)
                        q_sig    = chk["Q"]             # (N,)
                        pct      = np.linspace(0, 100, N_full)
                        # H for PCA scatter
                        H_np = _to_numpy(
                            item_lookup[(chk["battery_idx"], chk["dataset"])]["H"]
                        )
                    else:
                        # (H-h0)[1:] is length N-1; trim all signals to match
                        pc1_sig  = chk["pc1_rel"]       # (N-1,)
                        norm_sig = chk["norms_rel"]      # (N-1,)
                        q_sig    = chk["Q_rel"][1:]      # (N-1,)  drop the zero at 0
                        pct      = np.linspace(0, 100, N_full - 1)
                        H_raw = _to_numpy(
                            item_lookup[(chk["battery_idx"], chk["dataset"])]["H"]
                        )
                        H_np = (H_raw - H_raw[0:1])[1:]

                    H_all_list.append(H_np)
                    pct_all_list.append(pct)

                    axs[0].plot(pct, pc1_sig,  color=color, lw=1.0, alpha=0.6,
                                label=chk["dataset"])
                    axs[1].plot(pct, norm_sig, color=color, lw=1.0, alpha=0.6)
                    axs[2].plot(pct, q_sig,    color=color, lw=1.0, alpha=0.6)

                # ── col 3: 2-D PCA scatter ───────────────────────────────────
                H_stack   = np.vstack(H_all_list)
                pct_stack = np.concatenate(pct_all_list)
                pca2      = PCA(n_components=2)
                coords    = pca2.fit_transform(H_stack)
                ev        = pca2.explained_variance_ratio_

                sc = axs[3].scatter(coords[:, 0], coords[:, 1],
                                    c=pct_stack, cmap=cmap_pct, s=3, alpha=0.5)
                plt.colorbar(sc, ax=axs[3], label="Cycle %", pad=0.02)
                axs[3].set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=8)
                axs[3].set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=8)

                # ── Axis labels, titles, legend ──────────────────────────────
                rho_map_raw = ["mean_rho_pc1", "mean_rho_norm",
                               "mean_rho_q",   "mean_rho_pc1"]
                rho_map_rel = ["mean_rho_pc1_rel", "mean_rho_norm_rel",
                               "mean_rho_qrel",    "mean_rho_pc1_rel"]
                rho_map     = rho_map_raw if version == "raw" else rho_map_rel
                ylabels     = ["PC1", "‖h_c‖₂",
                               "Q (Ah)" if version == "raw" else "Q−Q₀ (Ah)",
                               f"PC2 ({ev[1]*100:.1f}%)"]
                xlabels     = ["Cycle %", "Cycle %", "Cycle %", f"PC1 ({ev[0]*100:.1f}%)"]

                for col_i in range(4):
                    mean_rho = split_res["__all__"][rho_map[col_i]]
                    axs[col_i].set_title(
                        f"{col_titles[col_i]}\n[{v_label}]  mean|ρ|={mean_rho:.3f}",
                        fontsize=8.5,
                    )
                    axs[col_i].set_xlabel(xlabels[col_i], fontsize=8)
                    axs[col_i].set_ylabel(ylabels[col_i], fontsize=8)

                # Legend on PC1 panel
                handles = [plt.Line2D([0], [0], color=ds_color[n], lw=2, label=exp_to_paper_name[n])
                           for n in ds_names]
                axs[0].legend(handles=handles, fontsize=6.5,
                              loc="best", ncol=2)

                # Row label in figure margin
                fig.text(
                    0.002,
                    1 - (row + 0.5) / n_rows,
                    f"{split.upper()}\n{v_label}",
                    va="center", ha="left",
                    fontsize=9, fontweight="bold",
                    rotation=90, color="#333",
                )

        fig.suptitle(
            "Monotonicity Diagnostics — Raw H vs Relative (H−h₀, Q−Q₀)",
            fontsize=13, fontweight="bold", y=1.005,
        )

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Diagnostics plot saved → {save_path}")

        return fig

    def plot_pca(
        self,
        results: dict,
        splits_to_plot: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> plt.Figure:
        """
        Same figure as :meth:`plot` with a taller layout and legends that
        show the ``Bat_dataset`` names instead of the paper names.

        Parameters
        ----------
        results       : output of self.run()
        splits_to_plot: which splits to include (default: all in results)
        save_path     : if given, save figure here
        dpi           : figure DPI

        Returns
        -------
        matplotlib Figure
        """
        if splits_to_plot is None:
            splits_to_plot = [s for s in ("train", "val", "test") if s in results]

        n_splits = len(splits_to_plot)
        if n_splits == 0:
            raise ValueError("No splits to plot.")

        # 2 versions (raw / relative) × 4 columns × n_splits rows
        n_rows = n_splits * 2
        fig    = plt.figure(figsize=(20, 4.5 * n_rows))
        gs     = gridspec.GridSpec(n_rows, 4, figure=fig,
                                   hspace=0.50, wspace=0.35)

        cmap_ds  = matplotlib.colormaps["tab20"]
        cmap_pct = "plasma"

        col_titles = [
            "PC1(H) vs cycle %",
            "‖h_c‖₂ vs cycle %",
            "Capacity vs cycle %",
            "2-D PCA (colour=cycle%)",
        ]

        for s_idx, split in enumerate(splits_to_plot):
            split_res  = results[split]
            all_checks = split_res["__all__"]["batteries"]

            # Consistent colour per source dataset
            ds_names = sorted({c["dataset"] for c in all_checks})
            ds_color = {n: cmap_ds(i / max(len(ds_names) - 1, 1))
                        for i, n in enumerate(ds_names)}

            # Pre-fetch H arrays from the dataset (avoids re-searching in inner loop)
            ds_obj   = self.dm._split_ds(split)
            # Build lookup: (battery_idx, dataset) -> item
            item_lookup = {
                (it["battery_idx"], it["dataset"]): it
                for it in ds_obj.items
            }

            for v_idx, version in enumerate(("raw", "relative")):
                row = s_idx * 2 + v_idx
                v_label = "H" if version == "raw" else "H−h₀"

                # Pre-create the four axes for this row so that they can be annotated
                axs = [fig.add_subplot(gs[row, c]) for c in range(4)]

                H_all_list, pct_all_list = [], []

                for chk in all_checks:
                    N_full = chk["N"]
                    color  = ds_color[chk["dataset"]]

                    if version == "raw":
                        pc1_sig  = chk["pc1"]          # (N,)
                        norm_sig = chk["norms"]         # (N,)
                        q_sig    = chk["Q"]             # (N,)
                        pct      = np.linspace(0, 100, N_full)
                        # H for PCA scatter
                        H_np = _to_numpy(
                            item_lookup[(chk["battery_idx"], chk["dataset"])]["H"]
                        )
                    else:
                        # (H-h0)[1:] is length N-1; trim all signals to match
                        pc1_sig  = chk["pc1_rel"]       # (N-1,)
                        norm_sig = chk["norms_rel"]      # (N-1,)
                        q_sig    = chk["Q_rel"][1:]      # (N-1,)  drop the zero at 0
                        pct      = np.linspace(0, 100, N_full - 1)
                        H_raw = _to_numpy(
                            item_lookup[(chk["battery_idx"], chk["dataset"])]["H"]
                        )
                        H_np = (H_raw - H_raw[0:1])[1:]

                    H_all_list.append(H_np)
                    pct_all_list.append(pct)

                    axs[0].plot(pct, pc1_sig,  color=color, lw=1.0, alpha=0.6,
                                label=chk["dataset"])
                    axs[1].plot(pct, norm_sig, color=color, lw=1.0, alpha=0.6)
                    axs[2].plot(pct, q_sig,    color=color, lw=1.0, alpha=0.6)

                # ── col 3: 2-D PCA scatter ───────────────────────────────────
                H_stack   = np.vstack(H_all_list)
                pct_stack = np.concatenate(pct_all_list)
                pca2      = PCA(n_components=2)
                coords    = pca2.fit_transform(H_stack)
                ev        = pca2.explained_variance_ratio_

                sc = axs[3].scatter(coords[:, 0], coords[:, 1],
                                    c=pct_stack, cmap=cmap_pct, s=3, alpha=0.5)
                plt.colorbar(sc, ax=axs[3], label="Cycle %", pad=0.02)
                axs[3].set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=8)
                axs[3].set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=8)

                # ── Axis labels, titles, legend ──────────────────────────────
                rho_map_raw = ["mean_rho_pc1", "mean_rho_norm",
                               "mean_rho_q",   "mean_rho_pc1"]
                rho_map_rel = ["mean_rho_pc1_rel", "mean_rho_norm_rel",
                               "mean_rho_qrel",    "mean_rho_pc1_rel"]
                rho_map     = rho_map_raw if version == "raw" else rho_map_rel
                ylabels     = ["PC1", "‖h_c‖₂",
                               "Q (Ah)" if version == "raw" else "Q−Q₀ (Ah)",
                               f"PC2 ({ev[1]*100:.1f}%)"]
                xlabels     = ["Cycle %", "Cycle %", "Cycle %", f"PC1 ({ev[0]*100:.1f}%)"]

                for col_i in range(4):
                    mean_rho = split_res["__all__"][rho_map[col_i]]
                    axs[col_i].set_title(
                        f"{col_titles[col_i]}\n[{v_label}]  mean|ρ|={mean_rho:.3f}",
                        fontsize=8.5,
                    )
                    axs[col_i].set_xlabel(xlabels[col_i], fontsize=8)
                    axs[col_i].set_ylabel(ylabels[col_i], fontsize=8)

                # Legend on PC1 panel
                handles = [plt.Line2D([0], [0], color=ds_color[n], lw=2, label=n)
                           for n in ds_names]
                axs[0].legend(handles=handles, fontsize=6.5,
                              loc="best", ncol=2)

                # Row label in figure margin
                fig.text(
                    0.002,
                    1 - (row + 0.5) / n_rows,
                    f"{split.upper()}\n{v_label}",
                    va="center", ha="left",
                    fontsize=9, fontweight="bold",
                    rotation=90, color="#333",
                )

        fig.suptitle(
            "Monotonicity Diagnostics — Raw H vs Relative (H−h₀, Q−Q₀)",
            fontsize=13, fontweight="bold", y=1.005,
        )

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Diagnostics plot saved → {save_path}")

        return fig


def plot_relative_signals(dm: BatteryDataModule, diag_res: dict, split: str = "test",
                          save_path: Optional[str] = None, dpi: int = 200) -> plt.Figure:
    """
    Plot ``PC1(H - h_0)``, ``||h_c - h_0||_2`` and SOH against cycle % for one split.

    This is the top panel of the representation-diagnostics figure of
    the paper (one line per battery, coloured by source dataset; titles
    report the mean ``|rho|`` of the split).

    Args:
        dm: data module used to compute ``diag_res``.
        diag_res: output of :meth:`DiagnosticsRunner.run`.
        split: split to plot.
        save_path: optional output path.
        dpi: resolution of the saved figure.

    Returns:
        matplotlib Figure.
    """
    fig, axs = plt.subplots(1, 3, figsize=(28, 6))
    cmap_ds = matplotlib.colormaps["tab20"]

    split_res  = diag_res[split]
    all_checks = split_res["__all__"]["batteries"]
    ds_names = sorted({c["dataset"] for c in all_checks})
    ds_color = {n: cmap_ds(i / max(len(ds_names) - 1, 1)) for i, n in enumerate(ds_names)}

    for chk in all_checks:
        N_full = chk["N"]
        color  = ds_color[chk["dataset"]]
        pc1_sig  = chk["pc1_rel"]       # (N-1,)
        norm_sig = chk["norms_rel"]     # (N-1,)
        q_sig    = chk["Q"][1:]         # (N-1,)
        pct      = np.linspace(0, 100, N_full - 1)
        axs[0].plot(pct, pc1_sig,  color=color, lw=1.0, alpha=0.6, label=chk["dataset"])
        axs[1].plot(pct, norm_sig, color=color, lw=1.0, alpha=0.6)
        axs[2].plot(pct, q_sig,    color=color, lw=1.0, alpha=0.6)

    rho_map   = ["mean_rho_pc1_rel", "mean_rho_norm_rel", "mean_rho_q"]
    title_map = ["PC1(H-ho)", r"$‖H-ho‖₂$", "Q"]
    for i in range(3):
        mean_rho = split_res["__all__"][rho_map[i]]
        axs[i].set_title(f"{split.upper()}\n[{title_map[i]}]  mean|ρ|={mean_rho:.3f}", fontsize=12)
        axs[i].set_xlabel("Cycle %", fontsize=12)
    axs[0].set_ylabel("PC1", fontsize=12)
    axs[1].set_ylabel('‖h_c‖₂', fontsize=12)
    axs[2].set_ylabel('Q normalized', fontsize=12)

    handles = [plt.Line2D([0], [0], color=ds_color[n], lw=2, label=EXP_TO_PAPER_NAME.get(n, n))
               for n in ds_names]
    axs[0].legend(handles=handles, fontsize=10, loc="best", ncol=2)
    fig.suptitle(f"Relativized State signals and Capacity vs cycle % — {split.upper()}",
                 fontsize=13, fontweight="bold", y=1.005)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_pca_trajectories(dm: BatteryDataModule, diag_res: dict,
                          splits=("train", "test"),
                          save_path: Optional[str] = None, dpi: int = 200) -> plt.Figure:
    """
    2-D PCA of the recurrent states, raw ``H`` and relativised ``(H - h_0)[1:]``.

    One panel per (split, version); points are coloured by cycle %. This is
    the bottom panel of the representation-diagnostics figure of the paper.

    Args:
        dm: data module used to compute ``diag_res``.
        diag_res: output of :meth:`DiagnosticsRunner.run`.
        splits: two splits to plot.
        save_path: optional output path.
        dpi: resolution of the saved figure.

    Returns:
        matplotlib Figure.
    """
    splits_to_plot = [s for s in splits if s in diag_res]
    fig, axs = plt.subplots(1, 2 * len(splits_to_plot), figsize=(28, 6), squeeze=False)
    axs = axs[0]
    cmap_pct = "plasma"

    for s_idx, split in enumerate(splits_to_plot):
        split_res  = diag_res[split]
        all_checks = split_res["__all__"]["batteries"]
        ds_obj   = dm._split_ds(split)
        item_lookup = {(it["battery_idx"], it["dataset"]): it for it in ds_obj.items}

        for v_idx, version in enumerate(("raw", "relative")):
            pidx = s_idx * 2 + v_idx
            v_label = "H" if version == "raw" else "H−h₀"
            H_all_list, pct_all_list = [], []
            for chk in all_checks:
                N_full = chk["N"]
                H_raw = _to_numpy(item_lookup[(chk["battery_idx"], chk["dataset"])]["H"])
                if version == "raw":
                    pct  = np.linspace(0, 100, N_full)
                    H_np = H_raw
                else:
                    pct  = np.linspace(0, 100, N_full - 1)
                    H_np = (H_raw - H_raw[0:1])[1:]
                H_all_list.append(H_np)
                pct_all_list.append(pct)

            coords = PCA(n_components=2).fit(np.vstack(H_all_list))
            ev     = coords.explained_variance_ratio_
            xy     = coords.transform(np.vstack(H_all_list))
            sc = axs[pidx].scatter(xy[:, 0], xy[:, 1], c=np.concatenate(pct_all_list),
                                   cmap=cmap_pct, s=3, alpha=0.5)
            plt.colorbar(sc, ax=axs[pidx], label="Cycle %", pad=0.02)
            axs[pidx].set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=12)
            axs[pidx].set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=12)
            rho_key  = "mean_rho_pc1" if version == "raw" else "mean_rho_pc1_rel"
            mean_rho = split_res["__all__"][rho_key]
            axs[pidx].set_title(f"{split.upper()}\n[{v_label}]  mean|ρ|={mean_rho:.3f}", fontsize=12)

    fig.suptitle("2-D PCA (colour=cycle%) — Raw H vs Relative H−h₀",
                 fontsize=13, fontweight="bold", y=1.005)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig

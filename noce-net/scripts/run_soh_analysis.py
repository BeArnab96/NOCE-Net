"""
scripts.run_soh_analysis
========================
Analyses of the frozen NOCE-Net recurrent states with respect to SOH.

The script reads the exported states of one trained run (written by
``scripts/run_export.py`` to ``<runs_dir>/noce_net_run<k>/exports``) and
writes all outputs to ``--out-dir`` (default
``<runs_dir>/noce_net_run<k>/soh_analysis``).

Analyses (``--analysis``):

* ``diagnostics``     : Spearman |rho| between the PC1 / norm of the
  relative states and the SOH trajectory, per split and dataset
  (:class:`soh.diagnostics.DiagnosticsRunner`), plus the relative-signal
  and PCA-trajectory figures.
* ``direct-soh``      : :class:`soh.direct_soh_head.SOHTrainer` trained on
  the ``train`` split, evaluated on ``val`` (in-sample) and ``test``
  (zero-shot), followed by the shape diagnostic on both splits.
* ``per-dataset-mlp`` : per-dataset PID-R MLP sweep over training
  fractions (:class:`soh.per_dataset_mlp.PerDatasetMLPAnalysis`) and the
  same sweep with the FEC index as the only input
  (:class:`soh.fec_baseline.FECPerDatasetMLPAnalysis`).
* ``fec-baseline``    : linear and MLP FEC-only baselines trained on the
  training datasets (:func:`soh.fec_baseline.run_fec_only_analysis`).
* ``all``             : every analysis above.

Results are stored as pickled dictionaries (``<analysis>_results.pkl``),
text tables and PNG figures.

``--run-dir`` replaces ``<runs_dir>/noce_net_run<k>`` (e.g. ``--run-dir
model_ckpts`` after exporting the states of the released checkpoint).

Usage (from the ``noce-net`` folder)::

    python scripts/run_soh_analysis.py --run 1 --analysis all
    python scripts/run_soh_analysis.py --run-dir model_ckpts --analysis all
"""
import argparse
import contextlib
import io
import os
import pickle
import random
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import run_dir  # noqa: E402  (also adds the package root to sys.path)
from soh.data_pipeline import BatteryDataModule, load_exports  # noqa: E402

ANALYSES = ("diagnostics", "direct-soh", "per-dataset-mlp", "fec-baseline")


def _plain(obj):
    """Recursively convert ``defaultdict`` objects (which may hold lambdas) to ``dict`` for pickling."""
    if isinstance(obj, defaultdict):
        obj = dict(obj)
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def _save(obj, out_dir, name):
    """Pickle ``obj`` to ``<out_dir>/<name>``."""
    path = os.path.join(out_dir, name)
    with open(path, "wb") as f:
        pickle.dump(_plain(obj), f)
    print(f"Wrote {path}")


@contextlib.contextmanager
def _tee(path):
    """Print to the console and copy everything printed to ``path``."""
    buffer = io.StringIO()

    class _Tee(io.TextIOBase):
        """Text stream writing to the console and to an in-memory buffer."""
        def write(self, s):
            """Write ``s`` to the console and to the buffer."""
            sys.__stdout__.write(s)
            buffer.write(s)
            return len(s)

    old = sys.stdout
    sys.stdout = _Tee()
    try:
        yield
    finally:
        sys.stdout = old
        with open(path, "w", encoding="utf-8") as f:
            f.write(buffer.getvalue())
        print(f"Wrote {path}")


def set_seed(seed):
    """Seed ``random``, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Analyses
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostics(dm, out_dir):
    """Spearman diagnostics and the corresponding figures."""
    from soh.diagnostics import DiagnosticsRunner, plot_pca_trajectories, plot_relative_signals

    runner = DiagnosticsRunner(dm)
    with _tee(os.path.join(out_dir, "diagnostics_table.txt")):
        diag_res = runner.run(splits=["train", "val", "test"], verbose=True)
    _save(diag_res, out_dir, "diagnostics_results.pkl")

    for split in ("train", "val", "test"):
        if split in diag_res:
            fig = plot_relative_signals(dm, diag_res, split=split,
                                        save_path=os.path.join(out_dir, f"relative_signals_{split}.png"))
            plt.close(fig)
    fig = plot_pca_trajectories(dm, diag_res, splits=("train", "test"),
                                save_path=os.path.join(out_dir, "pca_trajectories.png"))
    plt.close(fig)
    return diag_res


def run_direct_soh(dm, out_dir, device, d_h):
    """DirectSOHHead training, evaluation and shape diagnostic."""
    from soh.direct_soh_head import SOHTrainer, shape_diagnostic

    trainer = SOHTrainer(dm, d_h=d_h, hidden_dims=(256, 64), lr=3e-4,
                         exclude_datasets=["evTOL", "UMich"], device=device)
    with _tee(os.path.join(out_dir, "direct_soh_table.txt")):
        trainer.train(n_epochs=30, val_every=1, verbose=True)
        results = trainer.evaluate_all_splits(verbose=True)
        shape = {split: shape_diagnostic(trainer, split=split, verbose=True)
                 for split in ("val", "test")}
    _save({"evaluation": results, "shape_diagnostic": shape, "history": trainer.history},
          out_dir, "direct_soh_results.pkl")
    torch.save(trainer.model.state_dict(), os.path.join(out_dir, "direct_soh_head.pth"))

    fig = trainer.plot_predictions(trainer.predict(split="test"), n_per_dataset=2,
                                   save_path=os.path.join(out_dir, "direct_soh_predictions_test.png"))
    plt.close(fig)
    return results, shape


def run_per_dataset_mlp(dm, out_dir, device, d_h):
    """PID-R and FEC-only per-dataset MLP sweeps."""
    from soh.direct_soh_head import SOHTrainer
    from soh.fec_baseline import FECPerDatasetMLPAnalysis
    from soh.per_dataset_mlp import PerDatasetMLPAnalysis

    sweep = [0.1, 0.2, 0.3, 0.4, 0.5]
    with _tee(os.path.join(out_dir, "per_dataset_mlp_table.txt")):
        # A DirectSOHHead is trained first; its PID-R extractor is then frozen.
        trainer = SOHTrainer(dm, d_h=d_h, hidden_dims=(64, 32), exclude_datasets=["evTOL"],
                             loss_kwargs={"mse_weight": 1.0, "mape_weight": 0.5,
                                          "mono_weight": 0.3, "bol_weight": 1.0},
                             device=device)
        trainer.train(n_epochs=8, val_every=4, verbose=True)
        pid_extractor = trainer.model.pid

        pid_analysis = PerDatasetMLPAnalysis(data_module=dm, pid_extractor=pid_extractor, d_h=d_h,
                                             sweep_pcts=sweep, mlp_hidden=64, n_epochs=60,
                                             n_repeats=3, device=device)
        pid_results = pid_analysis.run(verbose=True)
        print(pid_analysis.results_table(pid_results))

        fec_analysis = FECPerDatasetMLPAnalysis(data_module=dm, sweep_pcts=sweep, n_repeats=3,
                                                n_epochs=80, device=device)
        fec_results = fec_analysis.run(verbose=True)
        fec_analysis.print_comparison_table(fec_results, pid_results, target_pct=0.3)

    _save({"pid": pid_results, "fec_only": fec_results}, out_dir, "per_dataset_mlp_results.pkl")
    fig = pid_analysis.plot_sweep(pid_results, save_path=os.path.join(out_dir, "per_dataset_mlp_mape.png"))
    plt.close(fig)
    fig = pid_analysis.plot_mae_sweep(pid_results, save_path=os.path.join(out_dir, "per_dataset_mlp_mae.png"))
    plt.close(fig)
    fig = fec_analysis.plot_sweep(fec_results, pid_results,
                                  save_path=os.path.join(out_dir, "per_dataset_fec_vs_pid.png"))
    plt.close(fig)
    return pid_results, fec_results


def run_fec_baseline(dm, out_dir):
    """Linear and MLP FEC-only baselines trained on the training datasets."""
    from soh.fec_baseline import run_fec_only_analysis

    with _tee(os.path.join(out_dir, "fec_baseline_table.txt")):
        results = run_fec_only_analysis(dm, in_sample_test_frac=0.2, n_repeats=3, mlp_hidden=64,
                                        mlp_epochs=100, mlp_lr=1e-3, seed=42, verbose=True)
    _save(results, out_dir, "fec_baseline_results.pkl")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="./runs", help="Parent folder of all run directories.")
    ap.add_argument("--run", type=int, default=1, help="Run identifier of the trained NOCE-Net.")
    ap.add_argument("--run-dir", default=None, help="Explicit run directory (overrides --runs-dir/--run).")
    ap.add_argument("--exports-dir", default=None,
                    help="Folder with <split>_save_model_outputs.pkl (default: <run_dir>/exports).")
    ap.add_argument("--out-dir", default=None, help="Output folder (default: <run_dir>/soh_analysis).")
    ap.add_argument("--analysis", nargs="+", default=["all"], choices=list(ANALYSES) + ["all"])
    ap.add_argument("--device", default="auto", help="'auto', 'cuda' or 'cpu'.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    log_path = args.run_dir or run_dir(args.runs_dir, "noce_net", args.run)
    exports_dir = args.exports_dir or os.path.join(log_path, "exports")
    out_dir = args.out_dir or os.path.join(log_path, "soh_analysis")
    os.makedirs(out_dir, exist_ok=True)
    analyses = ANALYSES if "all" in args.analysis else args.analysis

    exports = load_exports(exports_dir, splits=("train", "val", "test"))
    dm = BatteryDataModule(train_res=exports["train"], test_res=exports["test"],
                           val_res=exports["val"], batch_size=4)
    print(dm.summary())
    d_h = dm.train_ds.items[0]["H"].shape[1]

    for name in analyses:
        set_seed(args.seed)
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        if name == "diagnostics":
            run_diagnostics(dm, out_dir)
        elif name == "direct-soh":
            run_direct_soh(dm, out_dir, args.device, d_h)
        elif name == "per-dataset-mlp":
            run_per_dataset_mlp(dm, out_dir, args.device, d_h)
        elif name == "fec-baseline":
            run_fec_baseline(dm, out_dir)


if __name__ == "__main__":
    main()

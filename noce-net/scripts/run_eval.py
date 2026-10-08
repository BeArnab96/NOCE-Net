"""
scripts.run_eval
================
Evaluate trained runs and aggregate results over independent runs.

For every run directory given with ``--experiment``/``--runs`` the best
checkpoint is evaluated on the in-sample split (``full_res_datawise.csv``)
and the zero-shot split (``full_res_datawise_test.csv``), unless
``--aggregate-only`` is set. The mean and standard deviation over runs are
then written to ``<runs_dir>/<experiment>_summary_{in_sample,zero_shot}_{mean,std}.csv``.

With ``--cycle-wise`` the per-battery predictions of the first run on the
zero-shot split are stored in ``dVdQ_and_Vres.pkl`` and the MAE per decile
of battery life is plotted to ``cycle_wise_mae.png`` (figure "MAE across
FEC%" of the ablation study; use ``--experiment ablation_norec`` for the
model without recurrence).

``--run-dirs`` replaces the ``<runs_dir>/<experiment>_run<k>`` naming by
explicit run directories, e.g. the released checkpoint folder
``model_ckpts`` (which contains ``val/best_cyc_model.pth``); the summary
files are then written to the first of these directories.

Usage (from the ``noce-net`` folder)::

    python scripts/run_eval.py --data-root "/path/to/Battery Data Files" --experiment noce_net --runs 1 2 3 4 5
    python scripts/run_eval.py --data-root "/path/to/Battery Data Files" --run-dirs model_ckpts
"""
import argparse
import os
import pickle

import numpy as np

from common import add_common_args, build_eval_loader, load_configs_from_args, run_dir
from eval.voltage_eval import (aggregate_runs, cycle_wise_errors, plot_cycle_wise_errors,
                               save_dVdQ_and_Vres, test)


def build_model(experiment, model_configs):
    """
    Instantiate the model class that matches an experiment name.

    Args:
        experiment: ``'noce_net'``, ``'ablation_norec'`` or
            ``'ablation_<task|fec|learn>'``.
        model_configs: configuration dictionary.

    Returns:
        The model on ``model_configs['device']``.
    """
    if experiment == "noce_net":
        from noce_net.model import BatteryModel
        model = BatteryModel(model_configs)
    elif experiment == "ablation_norec":
        from ablation.recurrence_ablation import BatteryModelNoRecurrence
        model = BatteryModelNoRecurrence(model_configs)
    elif experiment.startswith("ablation_"):
        from ablation.embedding_ablation import BatteryModelEmbeddingAblation
        model_configs = dict(model_configs, leave_emb=experiment.split("_", 1)[1])
        model = BatteryModelEmbeddingAblation(model_configs)
    else:
        raise ValueError(f"Unknown experiment '{experiment}'.")
    return model.to(model_configs["device"])


def main():
    """Command-line entry point (see the module docstring)."""
    ap = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--experiment", default="noce_net",
                    help="noce_net | ablation_task | ablation_fec | ablation_learn | ablation_norec")
    ap.add_argument("--runs", type=int, nargs="+", default=[1])
    ap.add_argument("--run-dirs", nargs="+", default=None,
                    help="Explicit run directories (override --runs-dir/--experiment/--runs naming).")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Only aggregate existing result CSV files.")
    ap.add_argument("--cycle-wise", action="store_true",
                    help="Also compute and plot the MAE per decile of battery life (first run).")
    args = ap.parse_args()

    if args.run_dirs:
        run_dirs = args.run_dirs
        summary_dir = run_dirs[0]
    else:
        run_dirs = [run_dir(args.runs_dir, args.experiment, r) for r in args.runs]
        summary_dir = args.runs_dir
    model_configs = load_configs_from_args(args)
    is_embedding_ablation = args.experiment.startswith("ablation_") and args.experiment != "ablation_norec"

    if not args.aggregate_only or args.cycle_wise:
        in_sample_loader = build_eval_loader(args.data_root, "val")
        zero_shot_loader = build_eval_loader(args.data_root, "test")

    if not args.aggregate_only:
        for log_path in run_dirs:
            model = build_model(args.experiment, model_configs)
            if is_embedding_ablation:
                from ablation.embedding_ablation import test_datawise
                for name in ("res_datawise.csv", "res_datawise_test.csv"):
                    if os.path.exists(os.path.join(log_path, name)):
                        os.remove(os.path.join(log_path, name))
                test_datawise(model, None, log_path, in_sample_loader)
                test_datawise(model, None, log_path, zero_shot_loader)
            else:
                test(model, None, log_path, in_sample_loader, results_file="full_res_datawise.csv")
                test(model, None, log_path, zero_shot_loader, results_file="full_res_datawise_test.csv")

    files = (("in_sample", "res_datawise.csv"), ("zero_shot", "res_datawise_test.csv")) if is_embedding_ablation \
        else (("in_sample", "full_res_datawise.csv"), ("zero_shot", "full_res_datawise_test.csv"))
    for label, fname in files:
        mean_df, std_df = aggregate_runs(run_dirs, fname)
        mean_df.to_csv(os.path.join(summary_dir, f"{args.experiment}_summary_{label}_mean.csv"))
        std_df.to_csv(os.path.join(summary_dir, f"{args.experiment}_summary_{label}_std.csv"))
        print(f"\n=== {args.experiment} | {label} | mean over {len(run_dirs)} run(s) ===")
        print(mean_df.to_string())

    if args.cycle_wise:
        log_path = run_dirs[0]
        model = build_model(args.experiment, model_configs)
        save_dVdQ_and_Vres(model, log_path, zero_shot_loader)
        with open(os.path.join(log_path, "dVdQ_and_Vres.pkl"), "rb") as f:
            res_data = pickle.load(f)
        diffs = cycle_wise_errors(res_data)
        np.save(os.path.join(log_path, "cycle_wise_diffs.npy"), diffs)
        plot_cycle_wise_errors(diffs, save_path=os.path.join(log_path, "cycle_wise_mae.png"))


if __name__ == "__main__":
    main()

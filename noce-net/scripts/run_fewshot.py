"""
scripts.run_fewshot
===================
Few-shot voltage forecasting on the zero-shot (``Test``) datasets.

For every run directory the best checkpoint is fine-tuned per test battery
on its first ``--few-shot-cycles`` FECs and evaluated on the remaining
FECs (:func:`eval.fewshot.test_fewshot`). By default only the MLP of the
output reconstruction layer (``cycle_model.out_projection.out_proj``) is
fine-tuned on the first 50 FECs, the setting reported in the paper.
Results are written to
``<run_dir>/res_datawise_fshot_<unfreeze>_<few_shot_cycles>.csv``; the mean
and standard deviation over runs are written to
``<runs_dir>/noce_net_summary_fewshot_<unfreeze>_<few_shot_cycles>_{mean,std}.csv``
(or to the first ``--run-dirs`` entry).

Usage (from the ``noce-net`` folder)::

    python scripts/run_fewshot.py --data-root "/path/to/Battery Data Files" --runs 1 2 3 4 5
    python scripts/run_fewshot.py --data-root "/path/to/Battery Data Files" --run-dirs model_ckpts
"""
import argparse
import os

from common import add_common_args, build_eval_loader, load_configs_from_args, run_dir
from eval.fewshot import test_fewshot
from eval.voltage_eval import aggregate_runs
from noce_net.model import BatteryModel


def main():
    """Command-line entry point (see the module docstring)."""
    ap = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--runs", type=int, nargs="+", default=[1])
    ap.add_argument("--run-dirs", nargs="+", default=None,
                    help="Explicit run directories (override --runs-dir/--runs naming).")
    ap.add_argument("--few-shot-cycles", type=int, default=50,
                    help="Leading FECs per test battery used for fine-tuning.")
    ap.add_argument("--unfreeze", default="cycle_model.out_projection.out_proj",
                    help="Name prefix of the fine-tuned parameters "
                         "(ORL MLP by default; 'cycle_model.out_projection' for the complete ORL).")
    ap.add_argument("--fewshot-epochs", type=int, default=5, help="Fine-tuning steps per battery.")
    ap.add_argument("--lr", type=float, default=5e-4, help="Adam learning rate for fine-tuning.")
    args = ap.parse_args()

    if args.run_dirs:
        run_dirs = args.run_dirs
        summary_dir = run_dirs[0]
    else:
        run_dirs = [run_dir(args.runs_dir, "noce_net", r) for r in args.runs]
        summary_dir = args.runs_dir
    model_configs = load_configs_from_args(args)
    zero_shot_loader = build_eval_loader(args.data_root, "test")

    tag = f"{args.unfreeze}_{args.few_shot_cycles}"
    for log_path in run_dirs:
        model = BatteryModel(model_configs).to(model_configs["device"])
        test_fewshot(model, log_path, zero_shot_loader, model_configs,
                     few_shot_cycles=args.few_shot_cycles, unfreeze=args.unfreeze,
                     fewshot_epochs=args.fewshot_epochs, lr=args.lr)

    mean_df, std_df = aggregate_runs(run_dirs, f"res_datawise_fshot_{tag}.csv")
    mean_df.to_csv(os.path.join(summary_dir, f"noce_net_summary_fewshot_{tag}_mean.csv"))
    std_df.to_csv(os.path.join(summary_dir, f"noce_net_summary_fewshot_{tag}_std.csv"))
    print(f"\n=== noce_net | few-shot ({tag}) | mean over {len(run_dirs)} run(s) ===")
    print(mean_df.to_string())


if __name__ == "__main__":
    main()

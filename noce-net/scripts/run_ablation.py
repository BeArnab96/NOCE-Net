"""
scripts.run_ablation
====================
Train and evaluate the NOCE-Net ablations.

* ``--type embedding``  : removes the task, FEC or learnable aggregator
  token (``--leave-embs``); runs are written to
  ``<runs_dir>/ablation_<task|fec|learn>_run<k>``.
* ``--type recurrence`` : removes the GRU backbone; runs are written to
  ``<runs_dir>/ablation_norec_run<k>``.

Each run is evaluated after its last epoch on the in-sample and zero-shot
splits. Use ``scripts/run_eval.py --aggregate-only`` afterwards to average
the runs.

Usage (from the ``noce-net`` folder)::

    python scripts/run_ablation.py --data-root "/path/to/Battery Data Files" --type embedding --runs 1 2 3
    python scripts/run_ablation.py --data-root "/path/to/Battery Data Files" --type recurrence --runs 1 2 3
"""
import argparse

from common import add_common_args, build_eval_loader, build_train_val_loaders, load_configs_from_args


def main():
    """Command-line entry point (see the module docstring)."""
    ap = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--type", choices=["embedding", "recurrence"], required=True)
    ap.add_argument("--leave-embs", nargs="+", default=["task", "fec", "learn"],
                    choices=["task", "fec", "learn"], help="Embeddings to ablate (embedding ablation only).")
    ap.add_argument("--runs", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--max-epoch", type=int, default=20)
    ap.add_argument("--lr-init", type=float, default=1e-2)
    ap.add_argument("--step-size", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.33)
    args = ap.parse_args()

    model_configs = load_configs_from_args(args)
    train_dataloader, val_dataloader = build_train_val_loaders(args.data_root, model_configs)
    in_sample_loader = build_eval_loader(args.data_root, "val")
    zero_shot_loader = build_eval_loader(args.data_root, "test")

    if args.type == "embedding":
        from ablation.embedding_ablation import run_embedding_ablation
        run_embedding_ablation(model_configs, train_dataloader, val_dataloader,
                               in_sample_loader, zero_shot_loader, args.runs_dir,
                               leave_embs=args.leave_embs, runs=args.runs,
                               max_epoch=args.max_epoch, lr_init=args.lr_init,
                               step_size=args.step_size, gamma=args.gamma)
    else:
        from ablation.recurrence_ablation import run_recurrence_ablation
        run_recurrence_ablation(model_configs, train_dataloader, val_dataloader,
                                in_sample_loader, zero_shot_loader, args.runs_dir,
                                runs=args.runs, max_epoch=args.max_epoch, lr_init=args.lr_init,
                               step_size=args.step_size, gamma=args.gamma)


if __name__ == "__main__":
    main()

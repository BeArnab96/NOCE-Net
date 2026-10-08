"""
scripts.run_training
====================
Train NOCE-Net and evaluate its best checkpoint (main voltage-forecasting
results).

For every ``--runs`` identifier a model is trained from scratch in
``<runs_dir>/noce_net_run<k>``. Training resumes automatically from
``latest_model.pth`` when a run directory already contains one. After the
last epoch the best checkpoint (lowest validation loss) is evaluated on

* the in-sample split (held-out 20 % of the training datasets)
  -> ``full_res_datawise.csv``
* the zero-shot split (all ``Test`` datasets) -> ``full_res_datawise_test.csv``

Usage (from the ``noce-net`` folder)::

    python scripts/run_training.py --data-root "/path/to/Battery Data Files" --runs 1 2 3 4 5
"""
import argparse
import os

import pandas as pd
import torch

from common import (add_common_args, build_eval_loader, build_train_val_loaders,
                    load_configs_from_args, run_dir)
from eval.voltage_eval import test
from noce_net.model import BatteryModel
from train.train_response import learn_n_evaluate


def train_noce_net(model_configs, train_dataloader, val_dataloader, log_path,
                   max_epoch=20, lr_init=1e-2, step_size=5, gamma=0.33):
    """
    Train one NOCE-Net model in ``log_path``.

    Args:
        model_configs: configuration dictionary.
        train_dataloader, val_dataloader: loaders from
            :func:`common.build_train_val_loaders`.
        log_path: run directory.
        max_epoch: number of epochs.
        lr_init: initial learning rate of Adam (betas (0.9, 0.98), eps 1e-9).
        step_size, gamma: StepLR schedule (stepped once per epoch); the
            defaults are the values of the paper (step size 5, gamma 0.33).

    Returns:
        The trained :class:`~noce_net.model.BatteryModel` (last epoch weights;
        the best weights are in ``<log_path>/val/best_cyc_model.pth``).
    """
    device = model_configs["device"]
    model = BatteryModel(model_configs).to(device)
    os.makedirs(log_path, exist_ok=True)

    start_epoch = 0
    latest = os.path.join(log_path, "latest_model.pth")
    if os.path.exists(latest):
        model.load_state_dict(torch.load(latest, map_location=device))
        val_log = os.path.join(log_path, "val", "val_cycle.csv")
        if os.path.exists(val_log):
            start_epoch = int(pd.read_csv(val_log)["Epoch"].max()) + 1
        print(f"Resuming {log_path} from epoch {start_epoch}")

    optimizer = torch.optim.Adam(model.cycle_model.parameters(), lr=lr_init, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    for _ in range(start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, max_epoch):
        model.train()
        learn_n_evaluate(model=model, dataloader=train_dataloader, mode="train",
                         log_path=log_path, epoch=epoch, max_epoch=max_epoch,
                         model_configs=model_configs, optimizer=optimizer, scheduler=scheduler)
        torch.save(model.state_dict(), latest)

        model.eval()
        with torch.no_grad():
            learn_n_evaluate(model=model, dataloader=val_dataloader, mode="val",
                             log_path=log_path, epoch=epoch, max_epoch=max_epoch,
                             model_configs=model_configs)
    return model


def main():
    """Command-line entry point (see the module docstring)."""
    ap = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--runs", type=int, nargs="+", default=[1],
                    help="Run identifiers; one independent model is trained per identifier.")
    ap.add_argument("--max-epoch", type=int, default=20)
    ap.add_argument("--lr-init", type=float, default=1e-2)
    ap.add_argument("--step-size", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.33)
    ap.add_argument("--skip-eval", action="store_true", help="Only train; do not evaluate at the end.")
    args = ap.parse_args()

    model_configs = load_configs_from_args(args)
    train_dataloader, val_dataloader = build_train_val_loaders(args.data_root, model_configs)
    if not args.skip_eval:
        in_sample_loader = build_eval_loader(args.data_root, "val")
        zero_shot_loader = build_eval_loader(args.data_root, "test")

    for run in args.runs:
        log_path = run_dir(args.runs_dir, "noce_net", run)
        model = train_noce_net(model_configs, train_dataloader, val_dataloader, log_path,
                               max_epoch=args.max_epoch, lr_init=args.lr_init,
                               step_size=args.step_size, gamma=args.gamma)
        if not args.skip_eval:
            test(model, None, log_path, in_sample_loader, results_file="full_res_datawise.csv")
            test(model, None, log_path, zero_shot_loader, results_file="full_res_datawise_test.csv")


if __name__ == "__main__":
    main()

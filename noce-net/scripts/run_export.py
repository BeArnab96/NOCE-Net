"""
scripts.run_export
==================
Export recurrent states of a trained NOCE-Net run for the SOH analyses.

Loads ``<runs_dir>/noce_net_run<k>/val/best_cyc_model.pth`` and writes
``train_save_model_outputs.pkl``, ``val_save_model_outputs.pkl`` and
``test_save_model_outputs.pkl`` to ``--out-dir`` (default:
``<runs_dir>/noce_net_run<k>/exports``).

``--run-dir`` replaces ``<runs_dir>/noce_net_run<k>``, e.g. ``--run-dir
model_ckpts`` for the released checkpoint.

Usage (from the ``noce-net`` folder)::

    python scripts/run_export.py --data-root "/path/to/Battery Data Files" --run 1
    python scripts/run_export.py --data-root "/path/to/Battery Data Files" --run-dir model_ckpts
"""
import argparse
import os

import torch

from common import add_common_args, build_eval_loader, load_configs_from_args, run_dir
from export.export_states import save_model_outputs
from noce_net.model import BatteryModel


def main():
    """Command-line entry point (see the module docstring)."""
    ap = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--run", type=int, default=1, help="Run identifier of the trained NOCE-Net.")
    ap.add_argument("--run-dir", default=None, help="Explicit run directory (overrides --runs-dir/--run).")
    ap.add_argument("--checkpoint", default=None,
                    help="Checkpoint to load (default: <run_dir>/val/best_cyc_model.pth).")
    ap.add_argument("--out-dir", default=None, help="Output folder (default: <run_dir>/exports).")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    args = ap.parse_args()

    model_configs = load_configs_from_args(args)
    log_path = args.run_dir or run_dir(args.runs_dir, "noce_net", args.run)
    checkpoint = args.checkpoint or os.path.join(log_path, "val", "best_cyc_model.pth")
    out_dir = args.out_dir or os.path.join(log_path, "exports")
    os.makedirs(out_dir, exist_ok=True)

    model = BatteryModel(model_configs).to(model_configs["device"])
    model.load_state_dict(torch.load(checkpoint, map_location=model_configs["device"]))
    print(f"Loaded {checkpoint}")

    for split in args.splits:
        loader = build_eval_loader(args.data_root, split)
        save_model_outputs(model, out_dir, loader, loader_mode=split)
        print(f"Wrote {os.path.join(out_dir, split + '_save_model_outputs.pkl')}")


if __name__ == "__main__":
    main()

"""
scripts.common
==============
Helpers shared by the command-line entry points in ``scripts/``.

Directory conventions used by every script:

* ``--data-root`` : folder containing ``Train/`` and ``Test/`` in the
  ``re-BatteryData`` format (see ``noce_net.dataset``).
* ``--runs-dir``  : parent folder of all experiment outputs. Every training
  run writes to ``<runs_dir>/<experiment>_run<k>/`` with sub-folders
  ``train/`` and ``val/`` (the best checkpoint is
  ``val/best_cyc_model.pth``). Experiment names are ``noce_net``,
  ``ablation_task``, ``ablation_fec``, ``ablation_learn`` and
  ``ablation_norec``.
"""
import argparse
import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from torch.utils.data import DataLoader  # noqa: E402

from noce_net.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from noce_net.dataset import Battery_Dataset, Battery_Batch_Sampler, collate_batch  # noqa: E402


def run_dir(runs_dir, experiment, run):
    """Return ``<runs_dir>/<experiment>_run<run>``."""
    return os.path.join(runs_dir, f"{experiment}_run{run}")


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ``--data-root``, ``--runs-dir``, ``--config`` and ``--device`` to a parser."""
    parser.add_argument("--data-root", required=True,
                        help="Folder containing Train/ and Test/ in the re-BatteryData format.")
    parser.add_argument("--runs-dir", default="./runs",
                        help="Parent folder of all run directories (default: ./runs).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help="Model configuration YAML (default: model_configs.yaml).")
    parser.add_argument("--device", default="auto",
                        help="'auto', 'cuda', 'cpu' or a device string such as 'cuda:1'.")
    return parser


def load_configs_from_args(args) -> dict:
    """Load ``model_configs`` using ``args.config`` and ``args.device``."""
    return load_config(args.config, args.device)


def build_train_val_loaders(data_root, model_configs):
    """
    Training and validation dataloaders on ``<data_root>/Train``.

    The batteries of every training dataset are split 80/20 into training
    and validation (in-sample) sets. The training loader uses
    ``cyc_batch_size`` batteries per batch and the validation loader
    ``4 * cyc_batch_size``.

    Returns:
        ``(train_dataloader, val_dataloader)``
    """
    train_dir = os.path.join(data_root, "Train")
    numel_path = os.path.join(train_dir, "train_cycle_numel.npy")

    train_dataset = Battery_Dataset(train_dir, mode="train")
    val_dataset = Battery_Dataset(train_dir, mode="val")

    train_sampler = Battery_Batch_Sampler(train_dir, numel_path,
                                          indices=train_dataset.train_ids,
                                          batch_size=model_configs["cyc_batch_size"])
    val_sampler = Battery_Batch_Sampler(train_dir, numel_path,
                                        indices=train_dataset.val_ids,
                                        batch_size=model_configs["cyc_batch_size"] * 4)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, collate_fn=collate_batch)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, collate_fn=collate_batch)
    return train_dataloader, val_dataloader


def build_eval_loader(data_root, split):
    """
    Evaluation dataloader with one battery per batch.

    Args:
        data_root: folder containing ``Train/`` and ``Test/``.
        split: ``'train'`` or ``'val'`` (both from ``Train/``, using the same
            80/20 split as training) or ``'test'`` (all batteries of ``Test/``,
            the zero-shot datasets).

    Returns:
        ``DataLoader`` yielding batches produced by :func:`collate_batch`.
    """
    if split in ("train", "val"):
        split_dir = os.path.join(data_root, "Train")
        numel_path = os.path.join(split_dir, "train_cycle_numel.npy")
        dataset = Battery_Dataset(split_dir, mode=split)
        indices = dataset.train_ids if split == "train" else dataset.val_ids
    elif split == "test":
        split_dir = os.path.join(data_root, "Test")
        numel_path = os.path.join(split_dir, "test_cycle_numel.npy")
        dataset = Battery_Dataset(split_dir, mode="test")
        indices = dataset.data_ids
    else:
        raise ValueError(f"Unknown split '{split}'. Use 'train', 'val' or 'test'.")

    sampler = Battery_Batch_Sampler(split_dir, numel_path, indices=indices, batch_size=1)
    return DataLoader(dataset, sampler=sampler, collate_fn=collate_batch)

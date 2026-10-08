"""
soh.data_pipeline
=================
Data layer shared by all SOH analyses.

The inputs are the files ``<split>_save_model_outputs.pkl`` written by
``export/export_states.py``. For every battery the following are used:

* ``H``     : (N, d_h) recurrent states of NOCE-Net (``d_h = 3 D = 192``)
* ``Q``     : (N,) normalised SOH trajectory,
  ``SOH = (Q / Q_nom - 0.8) / 0.2`` divided by its maximum
* dataset name (``Bat_dataset``; all ``Stanford_*`` cells are grouped as
  ``'Stanford'``)

Notation used in this package: ``N`` FECs per battery, ``c`` FEC index,
``h_c`` recurrent state of FEC ``c``, ``B`` batteries in a batch,
``N_max`` longest battery in a batch.
"""
import os
import pickle
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_numpy(x) -> np.ndarray:
    """Coerce a list / tensor / ndarray to a float32 numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _to_tensor(x) -> torch.Tensor:
    """Coerce a list / tensor / ndarray to a float32 torch tensor."""
    return torch.from_numpy(_to_numpy(x))


def _extract_battery(res_file: dict, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Extract (H, Q, Q_nom) for battery at position `idx` in a res_file.

    The exported state sequence contains ``h_0`` followed by one state per
    FEC (N+1 entries) while ``Q`` has N entries; the last ``N`` states are
    kept so that ``H[c]`` and ``Q[c]`` refer to the same FEC.

    Returns
    -------
    H     : (N, d_h)  float32 tensor of GRU hidden states
    Q     : (N,)      float32 tensor of normalised SOH values
    Q_nom : float     SOH value at the first FEC
    """
    H_raw = res_file["rec_states"]["pred"][idx]
    Q_raw = res_file["capacity"]["true"][idx]

    H = _to_tensor(H_raw).squeeze(0)   # (N, d_h)
    Q = _to_tensor(Q_raw).squeeze(0)   # (N,) or (N,1)
    if Q.dim() == 2:
        Q = Q.squeeze(-1)   # → (N,)

    # Align lengths in case of rare off-by-one between states and capacity
    N = min(H.shape[0], Q.shape[0])
    H, Q = H[-N:], Q[:N]

    Q_nom = float(Q[0].item())
    return H, Q, Q_nom


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BatteryResDataset(Dataset):
    """
    PyTorch Dataset wrapping one res_file dict.

    Each item is a dict:
        H        : (N_i, d_h)  GRU hidden states
        Q        : (N_i,)      normalised SOH trajectory
        Q_nom    : float       SOH value at the first FEC
        dataset  : str         source dataset name (``Bat_dataset`` value,
                               e.g. ``'Unknown'`` for HUST)
        battery_idx : int      original index in the res_file lists

    Batteries with constant or NaN SOH, or with fewer than ``min_cycles``
    FECs, are skipped with a warning.

    Parameters
    ----------
    res_file   : the res_file dict described in module docstring
    split_name : human label ("train" / "val" / "test") used in logs / plots
    min_cycles : drop batteries with fewer than this many FECs (default 10)
    """

    def __init__(
        self,
        res_file: dict,
        split_name: str = "test",
        min_cycles: int = 10,
    ):
        self.split_name = split_name

        self.items: List[dict] = []
        self._dataset_to_indices: Dict[str, List[int]] = defaultdict(list)

        dataset_names = res_file["dataset_names"]
        n_batteries = len(dataset_names)

        all_stanford_bats = [id for id in range(n_batteries) if dataset_names[id].startswith('Stanford')]
        if len(all_stanford_bats) != 0:
          for id in all_stanford_bats:
            dataset_names[id] = 'Stanford'


        for i in range(n_batteries):

              H, Q, Q_nom = _extract_battery(res_file, i)
              Q_range = float(Q.max() - Q.min())
              if Q_range < 1e-4 or torch.isnan(Q).any():
                  warnings.warn(f"Battery {i} ({dataset_names[i]}): Q is constant or NaN — skipped.")
                  continue

              if H.shape[0] < min_cycles:
                  warnings.warn(
                      f"[{split_name}] Battery {i} ({dataset_names[i]}) has only "
                      f"{H.shape[0]} cycles — skipped (min_cycles={min_cycles})."
                  )
                  continue

              item = dict(
                  H=H, Q=Q, Q_nom=Q_nom,
                  dataset=dataset_names[i],
                  battery_idx=i,
              )
              local_idx = len(self.items)
              self.items.append(item)
              self._dataset_to_indices[dataset_names[i]].append(local_idx)


    # ── standard Dataset interface ──────────────────────────────────────────
    def __len__(self) -> int:
        """Number of batteries."""
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        """Battery dictionary ``idx``."""
        return self.items[idx]

    # ── convenience accessors ───────────────────────────────────────────────
    @property
    def dataset_names(self) -> List[str]:
        """Sorted list of unique dataset names present in this split."""
        return sorted(self._dataset_to_indices.keys())

    def indices_for_dataset(self, name: str) -> List[int]:
        """Local item indices belonging to a specific source dataset."""
        if name not in self._dataset_to_indices:
            raise KeyError(f"Dataset '{name}' not found. "
                           f"Available: {self.dataset_names}")
        return self._dataset_to_indices[name]

    def subset_for_dataset(self, name: str) -> Subset:
        """Return a torch Subset containing only batteries from `name`."""
        return Subset(self, self.indices_for_dataset(name))

    def summary(self) -> str:
        """Text summary: number of batteries and FEC-count statistics per dataset."""
        lines = [f"BatteryResDataset  split={self.split_name}  "
                 f"n_batteries={len(self)}"]
        for ds in self.dataset_names:
            idxs  = self.indices_for_dataset(ds)
            lens  = [self.items[i]["H"].shape[0] for i in idxs]
            lines.append(
                f"  {ds:<20s}  n={len(idxs):>3d}  "
                f"N_cycles  min={min(lens)}  max={max(lens)}  "
                f"mean={np.mean(lens):.0f}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Collator — pads variable-length batteries into a batch
# ─────────────────────────────────────────────────────────────────────────────

def collate_batteries(batch: List[dict]) -> dict:
    """
    Collate function for DataLoader.

    Pads H and Q along the FEC dimension to the longest battery in the batch.

    Returns a dict (``Q_batch`` holds the SOH targets):
        H_batch  : (B, N_max, d_h)   zero-padded hidden states
        Q_batch  : (B, N_max)        zero-padded capacity
        Q_noms   : (B,)              nominal capacities
        lengths  : (B,)  LongTensor  actual FEC counts per battery
        datasets : List[str]         source dataset per battery
        battery_indices : List[int]  original res_file indices
        mask     : (B, N_max)  bool  True = valid (not padding)
    """
    lengths = torch.tensor([item["H"].shape[0] for item in batch], dtype=torch.long)
    N_max   = int(lengths.max().item())
    B       = len(batch)
    d_h     = batch[0]["H"].shape[1]

    H_batch = torch.zeros(B, N_max, d_h)
    Q_batch = torch.zeros(B, N_max)
    Q_noms  = torch.zeros(B)

    for i, item in enumerate(batch):
        n = item["H"].shape[0]
        H_batch[i, :n] = item["H"]
        Q_batch[i, :n] = item["Q"]
        Q_noms[i]      = item["Q_nom"]

    # Build boolean mask: True where position is real data
    idx  = torch.arange(N_max).unsqueeze(0)   # (1, N_max)
    mask = idx < lengths.unsqueeze(1)          # (B, N_max)

    return dict(
        H_batch=H_batch,
        Q_batch=Q_batch,
        Q_noms=Q_noms,
        lengths=lengths,
        mask=mask,
        datasets=[item["dataset"]       for item in batch],
        battery_indices=[item["battery_idx"] for item in batch],
    )


# ─────────────────────────────────────────────────────────────────────────────
# DataModule — single entry point for all three splits
# ─────────────────────────────────────────────────────────────────────────────

class BatteryDataModule:
    """
    Wraps train / val / test res_file dicts and exposes DataLoaders plus
    per-dataset subsets for diagnostics and targeted inference.

    ``train`` holds the training batteries of the training datasets,
    ``val`` the held-out (in-sample) batteries of the same datasets and
    ``test`` the zero-shot datasets.

    Parameters
    ----------
    train_res, val_res, test_res : res_file dicts (val may be None)
    batch_size  : DataLoader batch size for training
    num_workers : DataLoader workers (set 0 to avoid multiprocessing issues
                  in interactive sessions)
    min_cycles  : minimum FEC count to include a battery
    """

    def __init__(
        self,
        train_res:   dict,
        test_res:    dict,
        val_res:     Optional[dict]  = None,
        batch_size:  int  = 8,
        num_workers: int  = 0,
        min_cycles:  int  = 10,
    ):
        self.batch_size  = batch_size
        self.num_workers = num_workers

        self.train_ds = BatteryResDataset(train_res, "train", min_cycles)
        self.test_ds  = BatteryResDataset(test_res,  "test",  min_cycles)
        self.val_ds   = (BatteryResDataset(val_res, "val", min_cycles)
                         if val_res is not None else None)

    # ── DataLoaders ──────────────────────────────────────────────────────────

    def train_loader(self, shuffle: bool = True) -> DataLoader:
        """DataLoader over the training batteries."""
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_batteries,
            num_workers=self.num_workers,
        )

    def val_loader(self) -> Optional[DataLoader]:
        """DataLoader over the in-sample (validation) batteries, or None."""
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_batteries,
            num_workers=self.num_workers,
        )

    def test_loader(self) -> DataLoader:
        """DataLoader over the zero-shot (test) batteries."""
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_batteries,
            num_workers=self.num_workers,
        )

    def loader_for_dataset(
        self,
        dataset_name: str,
        split: str = "test",
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        """
        Returns a DataLoader restricted to one source dataset within a split.

        Parameters
        ----------
        dataset_name : e.g. "HUST", "Stanford"
        split        : "train" | "val" | "test"
        batch_size   : override global batch_size (default: all batteries
                       in one batch for easy per-dataset inference)
        """
        ds = self._split_ds(split)
        subset = ds.subset_for_dataset(dataset_name)
        bs     = batch_size or len(subset)
        return DataLoader(
            subset,
            batch_size=bs,
            shuffle=False,
            collate_fn=collate_batteries,
            num_workers=self.num_workers,
        )

    def dataset_names(self, split: str = "test") -> List[str]:
        """Sorted source-dataset names present in ``split``."""
        return self._split_ds(split).dataset_names

    def summary(self, split: str = "all") -> str:
        """Concatenated :meth:`BatteryResDataset.summary` of the requested split(s)."""
        parts = []
        for s in (["train", "val", "test"] if split == "all" else [split]):
            try:
                parts.append(self._split_ds(s).summary())
            except AttributeError:
                pass
        return "\n\n".join(parts)

    def _split_ds(self, split: str) -> BatteryResDataset:
        """Return the :class:`BatteryResDataset` of ``'train'``, ``'val'`` or ``'test'``."""
        mapping = {"train": self.train_ds, "val": self.val_ds, "test": self.test_ds}
        ds = mapping.get(split)
        if ds is None:
            raise ValueError(f"Split '{split}' not available "
                             f"(val_res was not provided).")
        return ds


def load_exports(exports_dir: str, splits=("train", "val", "test")) -> Dict[str, dict]:
    """
    Load ``<split>_save_model_outputs.pkl`` files written by ``export/export_states.py``.

    Args:
        exports_dir: folder containing the exported files.
        splits: splits to load.

    Returns:
        dict mapping split name to the loaded res_file dictionary.
    """
    out = {}
    for split in splits:
        with open(os.path.join(exports_dir, f"{split}_save_model_outputs.pkl"), "rb") as f:
            out[split] = pickle.load(f)
    return out

"""
data_prep.assemble_dataset
==========================
Assemble the outputs of the preparation scripts in ``data_prep/src`` into
the ``re-BatteryData`` layout read by NOCE-Net::

    <data_root>/
        Train/
            train_cycle_numel.npy
            <DatasetFolder>/metadata.csv
            <DatasetFolder>/60secs/<id>.pkl
        Test/
            test_cycle_numel.npy
            <DatasetFolder>/metadata.csv
            <DatasetFolder>/60secs/<id>.pkl

For every dataset folder of ``--prepared-root`` (the ``--out-root`` of the
preparation scripts) the script

1. reads ``metadata.csv`` (in the dataset folder or in its ``60secs*``
   sub-folder) and the battery pickles of the ``60secs_wunc`` or ``60secs``
   sub-folder,
2. applies optional metadata overrides (``--metadata-overrides``),
3. checks that ``Form Factor``, ``Anode`` and ``Cathode`` take values known
   to the task-vector encoding of the data loaders,
4. keeps the first eight columns ``[V, Ichg, Idchg, Qc, Qd, Temp, t, dt]``
   of every FEC array (the uncertainty columns are dropped),
5. drops batteries with ``--min-fec`` FECs or fewer,
6. writes the pickles to ``<data_root>/<Split>/<DatasetFolder>/60secs/``
   and a ``metadata.csv`` with the additional column ``Cycle count`` and
   ``Bat_path = <DatasetFolder>/60secs/<id>.pkl``.

Finally ``<split>_cycle_numel.npy`` (total number of time steps of every
battery, in the order in which the loaders concatenate the metadata, i.e.
``os.listdir`` order of the dataset folders) is written for both splits.

The dataset folder ``Unknown`` (HUST) must keep this name: the loaders
convert its capacities from mAh to Ah based on the folder name.

Usage::

    python data_prep/assemble_dataset.py --prepared-root ./Data_Files --data-root "./Battery Data Files"

``--metadata-overrides`` takes a JSON file mapping ``Bat_dataset`` values to
column values, e.g. ``{"Unknown": {"Form Factor": "Cylindrical"}}``. HUST
(folder ``Unknown``) is written with the form factor ``Unknown``, which is not
in ``ALLOWED`` and therefore requires such an override.
"""
import argparse
import json
import os
import pickle
import shutil

import numpy as np
import pandas as pd

# Dataset folders of the two splits (folder names written by data_prep/src)
TRAIN_FOLDERS = ["Iowa_Uni", "NASA_Rand", "NatComm", "Severson", "UMich_Pouch", "XJTU_Partial"]
TEST_FOLDERS = ["evTOL", "NASA_PCoE", "Stanford", "Unknown", "XJTU_Full"]

# Values accepted by the task-vector encoding (Battery_Dataset.get_task_vector)
ALLOWED = {
    "Form Factor": {"Cylindrical", "Prismatic", "Pouch", "Polymer"},
    "Anode": {"Graphite", "Graphite/Silicon", "Other", "Unknown"},
    "Cathode": {"LFP", "NMC", "NCM", "NCA", "LCO", "NCM_NCA", "Unknown"},
}
REQUIRED_COLUMNS = ["Bat_id", "Bat_name", "Bat_path", "Bat_dataset", "Form Factor",
                    "Cathode", "Anode", "Qnom", "Vmax", "Vmin"]


def find_inputs(folder):
    """
    Locate the metadata file and the pickle folder of one prepared dataset.

    Returns:
        ``(metadata_csv, pickle_dir)``.
    """
    pickle_dir = None
    for sub in ("60secs_wunc", "60secs"):
        if os.path.isdir(os.path.join(folder, sub)):
            pickle_dir = os.path.join(folder, sub)
            break
    if pickle_dir is None:
        raise FileNotFoundError(f"No 60secs_wunc/ or 60secs/ folder in {folder}")
    for candidate in (os.path.join(folder, "metadata.csv"), os.path.join(pickle_dir, "metadata.csv")):
        if os.path.exists(candidate):
            return candidate, pickle_dir
    raise FileNotFoundError(f"No metadata.csv in {folder} or {pickle_dir}")


def file_name(bat_path):
    """Base name of a ``Bat_path`` written with either path separator."""
    return str(bat_path).replace("\\", "/").split("/")[-1]


def check_task_columns(meta, folder):
    """Raise ``ValueError`` if a task column holds a value unknown to the loaders."""
    problems = []
    for col, allowed in ALLOWED.items():
        bad = sorted(set(meta[col].astype(str)) - allowed)
        if bad:
            problems.append(f"'{col}' has {bad} (allowed: {sorted(allowed)})")
    if problems:
        raise ValueError(f"{folder}: " + "; ".join(problems) +
                         ". Use --metadata-overrides to set these columns.")


def assemble_folder(src_folder, dst_folder, overrides, min_fec):
    """
    Copy one prepared dataset into the ``re-BatteryData`` layout.

    Args:
        src_folder: prepared dataset folder (``<prepared_root>/<DatasetFolder>``).
        dst_folder: target folder (``<data_root>/<Split>/<DatasetFolder>``).
        overrides: ``{Bat_dataset: {column: value}}``.
        min_fec: batteries with ``min_fec`` FECs or fewer are dropped.

    Returns:
        list with the total number of time steps of every kept battery
        (metadata order).
    """
    name = os.path.basename(src_folder)
    meta_csv, pickle_dir = find_inputs(src_folder)
    meta = pd.read_csv(meta_csv)
    meta = meta.loc[:, [c for c in meta.columns if not c.startswith("Unnamed")]]
    missing = [c for c in REQUIRED_COLUMNS if c not in meta.columns]
    if missing:
        raise ValueError(f"{name}: metadata.csv lacks the columns {missing}")

    for ds_name, values in overrides.items():
        rows = meta["Bat_dataset"].astype(str) == ds_name
        for col, val in values.items():
            meta.loc[rows, col] = val
    check_task_columns(meta, name)

    out_dir = os.path.join(dst_folder, "60secs")
    os.makedirs(out_dir, exist_ok=True)

    keep_rows, cycle_counts, numels = [], [], []
    for i, row in meta.iterrows():
        fname = file_name(row["Bat_path"])
        with open(os.path.join(pickle_dir, fname), "rb") as f:
            data = pickle.load(f)
        cycles = [np.asarray(c)[..., :8] for c in data["cycle"]]
        n_fec = len(cycles)
        if n_fec <= min_fec:
            print(f"  {name}/{fname}: {n_fec} FECs -> dropped")
            continue
        with open(os.path.join(out_dir, fname), "wb") as f:
            pickle.dump({"cycle": cycles, "capacity": data["capacity"]}, f)
        meta.at[i, "Bat_path"] = f"{name}/60secs/{fname}"
        keep_rows.append(i)
        cycle_counts.append(n_fec)
        numels.append(int(sum(c.reshape(-1, c.shape[-1]).shape[0] for c in cycles)))

    meta = meta.loc[keep_rows].copy()
    meta["Cycle count"] = cycle_counts
    meta.to_csv(os.path.join(dst_folder, "metadata.csv"), index=False)
    print(f"  {name}: {len(meta)} batteries")
    return numels


def main():
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared-root", default="./Data_Files",
                    help="Output folder of the preparation scripts (default: ./Data_Files).")
    ap.add_argument("--data-root", required=True, help="Target folder (receives Train/ and Test/).")
    ap.add_argument("--train-folders", nargs="+", default=TRAIN_FOLDERS)
    ap.add_argument("--test-folders", nargs="+", default=TEST_FOLDERS)
    ap.add_argument("--metadata-overrides", default=None, help="JSON file with metadata overrides.")
    ap.add_argument("--min-fec", type=int, default=100)
    ap.add_argument("--overwrite", action="store_true", help="Delete existing Train/ and Test/ folders first.")
    args = ap.parse_args()

    overrides = {}
    if args.metadata_overrides:
        with open(args.metadata_overrides) as f:
            overrides = json.load(f)

    for split, folders, numel_name in (("Train", args.train_folders, "train_cycle_numel.npy"),
                                       ("Test", args.test_folders, "test_cycle_numel.npy")):
        split_dir = os.path.join(args.data_root, split)
        if os.path.exists(split_dir) and args.overwrite:
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)
        print(f"{split}:")
        numel_per_folder = {}
        for folder in folders:
            src = os.path.join(args.prepared_root, folder)
            if not os.path.isdir(src):
                print(f"  {folder}: not found in {args.prepared_root}, skipped")
                continue
            numel_per_folder[folder] = assemble_folder(src, os.path.join(split_dir, folder),
                                                       overrides, args.min_fec)

        # The loaders concatenate the metadata in os.listdir order of the dataset folders.
        numel = []
        for folder in os.listdir(split_dir):
            if not os.path.isdir(os.path.join(split_dir, folder)):
                continue
            if folder not in numel_per_folder:
                raise RuntimeError(f"{split_dir}/{folder} was not written by this run; "
                                   "use --overwrite to rebuild the split.")
            numel.extend(numel_per_folder[folder])
        np.save(os.path.join(split_dir, numel_name), np.asarray(numel))
        print(f"  wrote {numel_name} ({len(numel)} batteries)")


if __name__ == "__main__":
    main()

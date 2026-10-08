"""
Command-line handling shared by the dataset preparation scripts.

Every script in ``data_prep/src`` is run as::

    python data_prep/src/<script>.py --raw-root <RAW_ROOT> --out-root <OUT_ROOT>

``RAW_ROOT`` is the folder that holds the downloaded raw data in the
sub-folder expected by the script (see ``data_prep/README.md``); the
scripts change the working directory to it. ``OUT_ROOT`` (default
``./Data_Files``) receives one sub-folder per dataset with the processed
FEC pickles and a ``metadata.csv``.
"""
import argparse
import os
import sys

DATA_PREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DATA_PREP_ROOT not in sys.path:
    # makes ``data_prep/utils.py`` importable as ``utils``
    sys.path.insert(0, DATA_PREP_ROOT)


def parse_paths(description=None):
    """
    Parse ``--raw-root`` and ``--out-root``, create the output folder and
    change the working directory to the raw-data folder.

    Args:
        description: help text of the calling script.

    Returns:
        ``(RAW_ROOT, OUT_ROOT)`` as absolute paths.
    """
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", default=".",
                    help="Folder containing the downloaded raw data (default: current folder).")
    ap.add_argument("--out-root", default="./Data_Files",
                    help="Folder receiving the processed data (default: ./Data_Files).")
    args = ap.parse_args()
    raw_root = os.path.abspath(args.raw_root)
    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)
    os.chdir(raw_root)
    return raw_root, out_root

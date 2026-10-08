# NOCE-Net: Representation Learning for Battery Operational Context via Nested Sequence Modelling

Code release of the NeurIPS 2026 paper *NOCE-Net: Representation Learning
for Battery Operational Context via Nested Sequence Modelling* (Arnab
Bhattacharjee, Wayes Tushar, Tapan Kumar Saha).

## Abstract

Learning effective representations from battery operational history is
essential for predictive health management and control-oriented
applications. Existing approaches rely on response-dependent engineered
features or short early-life windows, limiting generalisability and
precluding use in prospective settings where future responses are
unavailable. We propose NOCE-Net, a deep nested sequence model that learns
control-enabling representations directly from raw cyclic operational or
telemetric data. Our data pipeline, re-BatteryData, standardises
heterogeneous aging datasets into semantically equivalent Full Equivalent
Cycles (FECs) defined by absolute charge throughput, enabling end-to-end
learning across diverse operational regimes. NOCE-Net combines multi-patch
transformer encoders for intra-cycle dynamics with a GRU backbone for
inter-cycle state evolution, operating on approximately 700 batteries
across ten public datasets spanning over 300 operational profiles. In
zero-shot evaluation on unseen battery chemistries and usage patterns,
NOCE-Net achieves 13.7% MAPE in voltage response prediction, outperforming
strong long-sequence baselines by over 43% while using 40x fewer
parameters. Shape and component diagnostics of the learned hidden states
reveal that the GRU's principal direction of evolution achieves Spearman
|rho| >= 0.95 with the capacity fade trajectory on average across all
datasets with monotone degradation profiles, including zero-shot test
batteries, providing evidence that the representations encode degradation
dynamics transferable to downstream tasks beyond battery response
modelling.

## Repository structure

| Folder | Content | Documentation |
|---|---|---|
| `data_prep/` | re-BatteryData: per-dataset preparation scripts and assembly of the `Train`/`Test` layout | `data_prep/README.md` |
| `noce-net/` | NOCE-Net model, training, evaluation, ablations, state export, SOH analyses and a trained checkpoint (`model_ckpts/`) | `noce-net/README.md` |

## 1. Download the battery datasets

The datasets are not redistributed with this repository and have to be
downloaded from their original sources:

| Dataset (paper name) | Chemistry (cathode / anode) | Batteries | Split | Source |
|---|---|---|---|---|
| ISU-ILCC | NCM / Graphite | 155 | Train | https://doi.org/10.25380/iastate.22582234 |
| NASA-Rand (randomized battery usage) | NMC / Graphite | 13 | Train | https://data.nasa.gov/dataset/randomized-battery-usage-1-random-walk (NASA PCoE data repository) |
| TJU (Zhu et al., 2022) | NCA / Graphite | 100 | Train | https://zenodo.org/records/6405084 |
| MATR (Severson et al., 2019) | LFP / Graphite | 124 | Train | https://data.matr.io/1/ |
| UMICH | NMC / Graphite | 21 | Train | https://doi.org/10.7302/7tw1-kc35 |
| XJTU-Partial (Wang et al., 2024) | NCM / Graphite | 16 | Train | https://zenodo.org/records/10963339 |
| evTOL (Bills et al.) | Unknown | 14 | Test | https://kilthub.cmu.edu/articles/dataset/eVTOL_Battery_Dataset/14226830 |
| NASA-PCoE (Saha and Goebel, 2007) | NMC / Graphite | 10 | Test | https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ |
| Stanford-EV (Pozzato et al., 2022) | NMC / Graphite-Si | 6 | Test | https://onorilab.stanford.edu/products/code-and-data-set |
| HUST (Ma et al., 2022) | LFP / Graphite | 77 | Test | https://doi.org/10.17632/nsc7hnsg4s.2 |
| XJTU-Full (Wang et al., 2024) | NCM / Graphite | 39 | Test | https://zenodo.org/records/10963339 |

XJTU-Partial and XJTU-Full are the partial-cycling and full-cycling
batteries of the same XJTU release. Batteries of the `Train` datasets are
used for training, validation and in-sample testing; the `Test` datasets
are never seen during training and form the zero-shot test set.

## 2. Prepare the data (re-BatteryData)

1. Place the raw files of each dataset below one folder `RAW_ROOT` with the
   sub-folders listed in `data_prep/README.md` (Step 1).
2. Run the preparation script of every dataset:

   ```bash
   python data_prep/src/<script>.py --raw-root <RAW_ROOT> --out-root ./Data_Files
   ```

   Each script segments the records into FECs (`2 * Qnom` of absolute
   charge throughput), resamples them to 60 s resolution and writes one
   pickle per battery plus a `metadata.csv`. Scripts are provided for
   ISU-ILCC, MATR, UMICH, XJTU-Partial, XJTU-Full, evTOL, NASA-PCoE,
   Stanford-EV and HUST. The preparation scripts of TJU and NASA-Rand are
   not part of this release; `assemble_dataset.py` picks up these datasets
   from the folders `NatComm` (TJU) and `NASA_Rand` when they are present
   in the same format.
3. Assemble the `Train`/`Test` layout read by all models:

   ```bash
   python data_prep/assemble_dataset.py --prepared-root ./Data_Files --data-root "./Battery Data Files"
   ```

   HUST is prepared with the form factor `Unknown`, which is not one of the
   form-factor classes of the task vector; its `Form Factor` column has to
   be set with `--metadata-overrides` (see `data_prep/README.md`).

The result has the layout

```
Battery Data Files/
    Train/train_cycle_numel.npy
    Train/<DatasetFolder>/metadata.csv
    Train/<DatasetFolder>/60secs/<id>.pkl     {'cycle': [X_c (T_c, 8)], 'capacity': Q}
    Test/test_cycle_numel.npy
    Test/<DatasetFolder>/...
```

with `X_c` holding `[V, Ichg, Idchg, Qc, Qd, Temp, t, dt]` for every time
step of FEC `c`.

## 3. Run the experiments

All commands are run from `noce-net/`; every script prints its options
with `--help`.

| Experiment | Command(s) |
|---|---|
| NOCE-Net training and in-sample / zero-shot voltage forecasting (5 runs) | `python scripts/run_training.py --data-root <DATA> --runs 1 2 3 4 5`, then `python scripts/run_eval.py --data-root <DATA> --experiment noce_net --runs 1 2 3 4 5 --aggregate-only` |
| Evaluation of the released checkpoint | `python scripts/run_eval.py --data-root <DATA> --run-dirs model_ckpts` |
| Embedding ablations (task / FEC / learnable token, 3 runs) | `python scripts/run_ablation.py --data-root <DATA> --type embedding --runs 1 2 3` |
| Recurrence ablation (no GRU backbone, 3 runs) | `python scripts/run_ablation.py --data-root <DATA> --type recurrence --runs 1 2 3` |
| Few-shot voltage forecasting (ORL MLP fine-tuned on the first 50 FECs of every zero-shot battery) | `python scripts/run_fewshot.py --data-root <DATA> --runs 1 2 3 4 5` |
| MAE across battery life | `python scripts/run_eval.py --data-root <DATA> --experiment <noce_net or ablation_norec> --runs 1 --aggregate-only --cycle-wise` |
| SOH analyses of the recurrent states (Spearman diagnostics, direct SOH head and shape diagnostic, per-dataset MLP, FEC-only baselines) | `python scripts/run_export.py --data-root <DATA> --run 1`, then `python scripts/run_soh_analysis.py --run 1 --analysis all` |

Outputs are written to `./runs/<experiment>_run<k>/` by default
(`--runs-dir` changes the parent folder). Details on the outputs of each
script are given in `noce-net/README.md`. The code of the long-sequence
baselines (TimerXL, AutoTimes, TTM and Mamba) is not part of this release.

## Note on the code

The experiments of the paper were run in Google Colab notebooks. The
functional structure of this repository (packages, command-line scripts
and run-directory conventions) was created for the code release. Should
any part of the code fail, the authors can be reached at
be.arnab96@gmail.com.

## Citation

```bibtex
@inproceedings{bhattacharjee2026nocenet,
  title     = {{NOCE-Net}: Representation Learning for Battery Operational Context via Nested Sequence Modelling},
  author    = {Bhattacharjee, Arnab and Tushar, Wayes and Saha, Tapan Kumar},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

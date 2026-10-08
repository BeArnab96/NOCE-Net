# re-BatteryData preparation

Scripts that convert the public battery aging datasets into the
`re-BatteryData` format used by NOCE-Net: every battery
is segmented into full equivalent cycles (FECs) and resampled to 60 s
resolution, and the batteries are arranged into a `Train/` and a `Test/`
split.

```bash
pip install -r requirements.txt
```

## Step 1: download the raw data

Download each dataset (links in the top-level README) and place it below a
common folder `RAW_ROOT` with the following layout:

| Script | Dataset | Expected raw data below `RAW_ROOT` |
|---|---|---|
| `src/isu_ilcc.py` | ISU-ILCC | `valid_cells.csv`, `Cycling_json/Release 1.0/`, `Cycling_json/Release 2.0/`, `RPT_json/Release 1.0/`, `RPT_json/Release 2.0/` |
| `src/matr.py` | MATR | `Cycling_full_regular/Severson/` with `2017-05-12_batchdata_updated_struct_errorcorrect-004.mat`, `2017-06-30_batchdata_updated_struct_errorcorrect.mat`, `2018-04-12_batchdata_updated_struct_errorcorrect-006.mat` |
| `src/umich.py` | UMICH | `Partial_Regular/UMich_Pouch/data/<cell>/cycling_wExpansion.csv` and `OCV_wExpansion.csv` |
| `src/xjtu_partial.py` | XJTU-Partial | `Partial_Regular/XJTU/<batch>/*.mat` |
| `src/xjtu_full.py` | XJTU-Full | `Cycling_full_regular/XJTU/<batch>/*.mat` |
| `src/evtol.py` | evTOL | `evTOL/VAH*.csv` |
| `src/nasa_pcoe.py` | NASA-PCoE | `Cycling_full_regular/NASA_PCoE/<N. BatteryAgingARC...>/B00xx.mat` |
| `src/stanford_ev.py` | Stanford-EV | `Full_Irregular/Stanford/cycling_tests/`, `Full_Irregular/Stanford/diagnostic_tests/capacity_test.mat` |
| `src/hust.py` | HUST | `Cycling_full_regular/our_data/*.pkl` |

The XJTU scripts read the same XJTU release from two locations.
Preparation scripts for TJU and NASA-Rand are not included; prepared
folders `NatComm` (TJU, `Bat_dataset = NatComm`) and `NASA_Rand`
(`Bat_dataset = NASA Rand`) in the output format below are picked up by
the assembly step.

## Step 2: run the preparation scripts

```bash
python data_prep/src/<script>.py --raw-root <RAW_ROOT> --out-root ./Data_Files
```

Each script writes one folder below `--out-root` with one pickle per
battery, `{'cycle': [X_c], 'capacity': Q}`, and a `metadata.csv`
(columns `Bat_id, Bat_name, Bat_path, Bat_dataset, Form Factor, Cathode,
Anode, Qnom, Vmax, Vmin`). The arrays `X_c` have the columns
`[V, Ichg, Idchg, Qc, Qd, Temp, t, dt]`; scripts that write to a
`60secs_wunc/` folder append four uncertainty columns. The scripts also
draw diagnostic figures with matplotlib.

| Script | Output folder | `Bat_dataset` |
|---|---|---|
| `isu_ilcc.py` | `Iowa_Uni/60secs_wunc/` | `Iowa_Uni` |
| `matr.py` | `Severson/60secs_wunc/` | `Severson` |
| `umich.py` | `UMich_Pouch/60secs_wunc/` | `UMich` |
| `xjtu_partial.py` | `XJTU_Partial/60secs_wunc/` | `XJTU_Partial` |
| `xjtu_full.py` | `XJTU_Full/60secs_wunc/` | `XJTU_Full` |
| `evtol.py` | `evTOL/60secs_wunc/` | `evTOL` |
| `nasa_pcoe.py` | `NASA_PCoE/60secs/` | `NASA PCoE` |
| `stanford_ev.py` | `Stanford/60secs_wunc/` | `Stanford_<cell>` |
| `hust.py` | `Unknown/60secs/` | `Unknown` |

The FEC segmentation follows the paper: a new FEC starts once the absolute
charge throughput since the start of the current FEC reaches `2 * Qnom`
(Stanford-EV is split into the cycles of its drive-cycle tests). Shared
helpers (`chunk_sampling`, `stat_downsampling`, `capacity_interpolation`,
...) are in `utils.py`.

## Step 3: assemble the Train/Test layout

```bash
python data_prep/assemble_dataset.py --prepared-root ./Data_Files --data-root "./Battery Data Files" \
    [--metadata-overrides overrides.json]
```

The script copies every prepared dataset into
`<data_root>/{Train,Test}/<DatasetFolder>/60secs/`, keeps the first
eight columns of every FEC, drops batteries with 100 FECs or fewer, adds
the `Cycle count` column, rewrites `Bat_path` as
`<DatasetFolder>/60secs/<id>.pkl` and writes `train_cycle_numel.npy` /
`test_cycle_numel.npy`. The default split is

* `Train`: `Iowa_Uni`, `NASA_Rand`, `NatComm`, `Severson`, `UMich_Pouch`, `XJTU_Partial`
* `Test`: `evTOL`, `NASA_PCoE`, `Stanford`, `Unknown`, `XJTU_Full`

(folders that are not present are skipped; `--train-folders` and
`--test-folders` change the split). The `Unknown` folder name of HUST must
be kept, because the data loaders convert its capacities from mAh to Ah
based on this name.

The task vector of every battery is built from the `Form Factor`, `Anode`
and `Cathode` columns, which must take one of the values
`Cylindrical/Prismatic/Pouch/Polymer`,
`Graphite/Graphite/Silicon/Other/Unknown` and
`LFP/NMC/NCM/NCA/LCO/NCM_NCA/Unknown`. The assembly stops with an error
listing any other value; such columns are set with a JSON file passed to
`--metadata-overrides`. `hust.py` writes the form factor `Unknown`
(Table 1 of the paper), which is not one of the four form-factor classes of
the task-vector encoding, so the HUST folder (`Unknown`) needs an override
of this column with one of these classes:

```json
{"Unknown": {"Form Factor": "<Cylindrical|Prismatic|Pouch|Polymer>"}}
```

The resulting folder is the `--data-root` of all NOCE-Net
scripts.

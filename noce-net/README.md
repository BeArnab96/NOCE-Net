# NOCE-Net

Model, training, evaluation, ablation and state-of-health (SOH) analysis
code of NOCE-Net. All commands below are run from this folder
(`noce-net/`). Data preparation is described in `../data_prep/README.md`.

## Installation

```bash
pip install -r requirements.txt     # or: pip install -e .
```

Python >= 3.9, PyTorch >= 2.0 and SciPy >= 1.11 are required. A GPU is
used automatically when available (`--device` overrides the choice).

## Layout

| Path | Content |
|---|---|
| `model_configs.yaml` | Model hyperparameters (read by `noce_net.config.load_config`) |
| `model_ckpts/` | Trained NOCE-Net weights: `val/best_cyc_model.pth` (lowest validation loss) and `latest_model.pth` (last epoch) |
| `noce_net/model.py` | `BatteryModel` (`cycle_model` = NOCE-Net) |
| `noce_net/layers.py` | Patch encoder/decoder, embeddings, GRU backbone (`CycleBlock`) and output reconstruction layer |
| `noce_net/transformer.py` | Transformer block adapted from [PaLM-rlhf-pytorch](https://github.com/lucidrains/PaLM-rlhf-pytorch) (MIT licence) |
| `noce_net/dataset.py` | `Battery_Dataset`, `collate_batch` and the length-aware `Battery_Batch_Sampler` |
| `noce_net/losses.py` | Training loss (MSE + top-quartile sAPE + first-difference term, optional dV/dQ alignment) |
| `train/train_response.py` | One training/validation epoch with truncated back-propagation through time |
| `eval/voltage_eval.py` | Per-dataset MAE / MAPE / sMAPE evaluation, run aggregation and error-vs-life analysis |
| `eval/fewshot.py` | Few-shot fine-tuning of the output reconstruction layer on the first FECs of each zero-shot battery |
| `export/export_states.py` | Export of recurrent states, predictions and capacities for the SOH analyses |
| `ablation/embedding_ablation.py` | Variants without the task, FEC or learnable aggregator token |
| `ablation/recurrence_ablation.py` | Variant without the GRU backbone |
| `soh/` | SOH analyses of the frozen states: Spearman diagnostics, PID-R features, direct SOH head and shape diagnostic, per-dataset MLP sweep, FEC-only baselines |
| `scripts/` | Command-line entry points (below) |

## Data

`--data-root` points to the folder produced by
`data_prep/assemble_dataset.py`:

```
<data_root>/Train/<DatasetFolder>/{metadata.csv, 60secs/<id>.pkl}, train_cycle_numel.npy
<data_root>/Test/<DatasetFolder>/{metadata.csv, 60secs/<id>.pkl},  test_cycle_numel.npy
```

The batteries of every training dataset are split 80/20 (in metadata
order) into training and in-sample test batteries; all batteries in
`Test/` form the zero-shot set.

## Experiments

Every run is written to `<runs_dir>/<experiment>_run<k>/` (default
`--runs-dir ./runs`) with sub-folders `train/`, `val/` (best checkpoint
`val/best_cyc_model.pth`) and the result files listed below. Experiment
names: `noce_net`, `ablation_task`, `ablation_fec`, `ablation_learn`,
`ablation_norec`.

### 1. Training and voltage forecasting (in-sample and zero-shot)

```bash
python scripts/run_training.py --data-root "/path/to/Battery Data Files" --runs 1 2 3 4 5
python scripts/run_eval.py     --data-root "/path/to/Battery Data Files" --experiment noce_net --runs 1 2 3 4 5 --aggregate-only
```

`run_training.py` trains one model per run identifier (Adam with initial
learning rate `--lr-init 1e-2` and a StepLR schedule with `--step-size 5`
and `--gamma 0.33`, as in the paper; `--max-epoch 20`; training resumes
from `latest_model.pth` if present) and evaluates the best checkpoint on both splits:
`full_res_datawise.csv` (in-sample) and `full_res_datawise_test.csv`
(zero-shot), with rows MAE, MAPE, sMAPE and MAPE on `V > 5e-3`
(normalised voltage) and one column per dataset plus `Total`.
`run_eval.py --aggregate-only` writes the mean and standard deviation over
runs to `<runs_dir>/noce_net_summary_{in_sample,zero_shot}_{mean,std}.csv`;
without `--aggregate-only` the checkpoints are evaluated again first.

The released checkpoint is evaluated with

```bash
python scripts/run_eval.py --data-root "/path/to/Battery Data Files" --run-dirs model_ckpts
```

The parameter count of each model is printed when it is built
(`sum(p.numel() for p in model.parameters())`).

### 2. Ablations

```bash
python scripts/run_ablation.py --data-root "/path/to/Battery Data Files" --type embedding  --runs 1 2 3
python scripts/run_ablation.py --data-root "/path/to/Battery Data Files" --type recurrence --runs 1 2 3
python scripts/run_eval.py --data-root "/path/to/Battery Data Files" --experiment ablation_task  --runs 1 2 3 --aggregate-only
python scripts/run_eval.py --data-root "/path/to/Battery Data Files" --experiment ablation_norec --runs 1 2 3 --aggregate-only
```

Both ablations use the optimiser and learning-rate schedule of the full
model (`--lr-init 1e-2`, `--step-size 5`, `--gamma 0.33`, `--max-epoch 20`).
The embedding ablation writes `res_datawise.csv` / `res_datawise_test.csv`
(thresholded MAPE on `V > 1e-4`) and the recurrence ablation
`full_res_datawise.csv` / `full_res_datawise_test.csv` (thresholded MAPE on
`V > 5e-3`) in each run directory.

MAE as a function of battery life (deciles of the FEC index) on the
zero-shot split, for the first run of an experiment:

```bash
python scripts/run_eval.py --data-root "/path/to/Battery Data Files" --experiment noce_net      --runs 1 --aggregate-only --cycle-wise
python scripts/run_eval.py --data-root "/path/to/Battery Data Files" --experiment ablation_norec --runs 1 --aggregate-only --cycle-wise
```

This writes `dVdQ_and_Vres.pkl`, `cycle_wise_diffs.npy` and
`cycle_wise_mae.png` to the run directory.

### 3. Few-shot voltage forecasting

```bash
python scripts/run_fewshot.py --data-root "/path/to/Battery Data Files" --runs 1 2 3 4 5
python scripts/run_fewshot.py --data-root "/path/to/Battery Data Files" --run-dirs model_ckpts
```

For every zero-shot battery, only the MLP of the output reconstruction
layer (`--unfreeze cycle_model.out_projection.out_proj`) is fine-tuned on
the first `--few-shot-cycles 50` FECs (`--fewshot-epochs 5` Adam steps,
`--lr 5e-4`), and the remaining FECs are predicted starting from the
recurrent state reached after these FECs. The fine-tuned weights are
carried over from one battery to the next. Results are written to
`res_datawise_fshot_<unfreeze>_<few_shot_cycles>.csv` in each run directory
and averaged over runs in
`noce_net_summary_fewshot_<unfreeze>_<few_shot_cycles>_{mean,std}.csv`.
The details of the procedure and of the reported metrics are given in the
docstring of `eval/fewshot.py`.

### 4. SOH analyses of the recurrent states

Export the states of a trained run (or of the released checkpoint with
`--run-dir model_ckpts`), then run the analyses:

```bash
python scripts/run_export.py --data-root "/path/to/Battery Data Files" --run 1
python scripts/run_soh_analysis.py --run 1 --analysis all
```

`run_export.py` writes `{train,val,test}_save_model_outputs.pkl` to
`<run_dir>/exports/`. `run_soh_analysis.py` writes to
`<run_dir>/soh_analysis/`:

| `--analysis` | Content | Outputs |
|---|---|---|
| `diagnostics` | Spearman \|rho\| between the PC1 / norm of the relative states `h_c - h_1` and SOH, per split and dataset | `diagnostics_table.txt`, `diagnostics_results.pkl`, `relative_signals_<split>.png`, `pca_trajectories.png` |
| `direct-soh` | PID-R features + MLP head trained on the training datasets (evTOL and UMich excluded), evaluated in-sample and zero-shot; shape / offset / scale diagnostic | `direct_soh_table.txt`, `direct_soh_results.pkl`, `direct_soh_head.pth`, `direct_soh_predictions_test.png` |
| `per-dataset-mlp` | Per-dataset MLP heads on frozen PID-R features for 10-50 % training batteries (3 random splits), and the same sweep with the FEC index as the only input | `per_dataset_mlp_table.txt`, `per_dataset_mlp_results.pkl`, `per_dataset_*.png` |
| `fec-baseline` | Linear and MLP models using only the normalised FEC index, trained on the training datasets | `fec_baseline_table.txt`, `fec_baseline_results.pkl` |

The hyperparameters of each analysis are set in the corresponding
`run_*` function of `scripts/run_soh_analysis.py`.

## Configuration

`model_configs.yaml` holds the architecture used for the released
checkpoint (D = `dmodel` = 64, patch sizes 4/16/32, 6 encoder and 3
decoder layers, 8 attention heads, FFN width multiplier 2, attention and
FFN dropout 0.1, 2 GRU layers with hidden size 3D, 2 batteries per training
batch), i.e. the values of the hyperparameter table of the paper. A
different file can be passed to every script with `--config`. The loss
weights (0.2 for the top-quartile term, 0.05 for the first-difference
term) are set in `noce_net/losses.py`, and the optimiser settings
(Adam, initial learning rate 1e-2, StepLR with step size 5 and gamma 0.33)
are command-line options of the training and ablation scripts.

## Notation used in the docstrings

B batteries per batch, L FECs per battery, T time steps per FEC, d_u = 7
inputs per time step `[Ichg, Idchg, Qc, Qd, Temp, t, dt]`, D = `dmodel`,
T_p patch tokens per FEC, `eta` the 6-dim task vector
`[form factor, anode, cathode, Qnom, Vmax, Vmin]`, `gamma_c = [c, Delta_c]`
the FEC index and duration (h), `h_c` the recurrent state (d_h = 3D) of
FEC `c`, and N the number of FECs of a battery in the SOH analyses.

# Reproducing the paper

Step-by-step commands to regenerate every result from a fresh checkout.
Pretrained checkpoints are included so training from scratch is optional.

---

## 0. Data

```bash
wget -O btcusdt_lob_oct2023.tar.gz https://zenodo.org/records/20046390/files/btcusdt_lob_oct2023.tar.gz
tar -xzf btcusdt_lob_oct2023.tar.gz -C data/raw/
```

DOI: https://doi.org/10.5281/zenodo.20046390 — see [`data/README.md`](data/README.md) for full details and schema.

---

## 1. Prerequisites

```bash
pip install -e .
```

Place the 21 daily JSON files at `data/raw/BTCUSDT_10-DD.json` for DD ∈ 01..21
(see `data/README.md` for the schema and acquisition notes).

**Smoke test:**

```bash
python -c "import lobgrad; from lobgrad.config import BenchmarkConfig; print(BenchmarkConfig().data_dir)"
```

Should print an absolute path ending in `data/raw`.

---

## 2. Train models (optional — skip if using pretrained checkpoints)

All scripts write their checkpoint to `checkpoints/<name>/best_model.pt` and
are resume-safe (re-running picks up from the last saved epoch).

### 2a. LOBGrad + CGAN — full HPO sweep

```bash
python scripts/run_hpo.py
```

Runs TimeGrad HPO then CGAN HPO sequentially via Optuna (~24h on 1× A100).
Toggle `RUN_TIMEGRAD` / `RUN_CGAN` at the top of the script to skip either half.
Canonical trials: TimeGrad trial 88, CGAN trial 42. Copy the winners:

```bash
cp checkpoints/optuna_deep_v2/trial_88/best_model.pt  checkpoints/final/joint_timegrad/best_model.pt
cp checkpoints/cgan_optuna_deep_v2/trial_42/cgan.pt   checkpoints/final/cgan/cgan.pt
```

### 2b. Ablation models

Each script uses the trial-88 hyperparameters and takes ~2–4h on 1× A100.

```bash
python scripts/train_timegrad_queue_only.py        # → checkpoints/timegrad_queue_only/best_model.pt
python scripts/train_timegrad_return_only.py       # → checkpoints/timegrad_return_only/best_model.pt
python scripts/train_timegrad_joint_flat_window.py # → checkpoints/timegrad_joint_flat_window/best_model.pt
```

Then move the results into `checkpoints/final/`:

```bash
cp checkpoints/timegrad_queue_only/best_model.pt        checkpoints/final/timegrad_queue_only/best_model.pt
cp checkpoints/timegrad_return_only/best_model.pt       checkpoints/final/timegrad_return_only/best_model.pt
cp checkpoints/timegrad_joint_flat_window/best_model.pt checkpoints/final/timegrad_joint_flat_window/best_model.pt
```

---

## 3. LOB grid sizing (Figure A6 / appendix)

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/calibrate_price_level.ipynb
```

Writes `figures/appendix/fig_level_coverage.pdf`.

---

## 4. Main paper figures and tables

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/paper_figures.ipynb
```

Generates (all written to `figures/`):

| Output | Paper location |
|---|---|
| `figures/main/fig_crps_horizon.pdf` | Figure 1 |
| `figures/main/fig_price_stylized.pdf` | Figure 2 |
| `figures/main/fig_trajectory_fan_crps.pdf` | Figure 3 |
| `figures/appendix/fig_queue_dist.pdf` | Figure A1 |
| `figures/appendix/fig_corr_heatmap.pdf` | Figure A2 |
| `figures/appendix/fig_queue_marginals.pdf` | Figure A3 |
| `figures/tables/table_headline.tex` | Table 1 |
| `figures/tables/table_headline_by_regime.tex` | Table A1 |
| `figures/tables/table_hyperparams.tex` | Table A2 |

---

## 4b. Headline table with bootstrap CIs (no-Jupyter path)

`scripts/bootstrap_tables.py` mirrors §4's load + generate logic so it can run
on a server without a Jupyter kernel, and adds window-level percentile
bootstrap CIs to every metric in Table 1.

```bash
python scripts/bootstrap_tables.py                      # default: 1000 resamples, all CPUs
python scripts/bootstrap_tables.py --n-boot 500 --n-jobs 8
python scripts/bootstrap_tables.py --no-cache           # ignore cache/generations/
```

Outputs:

| Output | Description |
|---|---|
| `figures/bootstrapped_tables/table_headline.tex` | Table 1 with sub-row 95% CIs |
| `figures/bootstrapped_tables/table_headline.csv` | Long-form numbers (`mean`, `display`, `ci_lo`, `ci_hi`) |
| `figures/bootstrapped_tables/run_manifest.json` | Run params, model list, timings |

Generations are cached at `cache/generations/{model}_seed{S}_W{N_W}_P{N_P}_H{H}.npz`.
**Note:** the cache key does not fingerprint checkpoint paths or hyperparameters,
so retraining a model requires deleting the affected `.npz` (or passing
`--no-cache`) before rerunning.

---

## 5. Limit-order execution benchmark (Table 2)

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 \
  notebooks/downstream_limit_order.ipynb
```

The timeout is set to 1h; the benchmark loops over 500 windows × 4 sources and
takes ~20–40 min on CPU. Generates:

| Output | Paper location |
|---|---|
| `figures/tables/table_execution_limit_order.csv` | Table 2 |
| `figures/tables/table_execution_limit_order.tex` | Table 2 |

To run fewer windows during development, set `N_WINDOWS = 50` in §0 of the notebook.

---

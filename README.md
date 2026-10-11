# LOBGrad

> Joint diffusion modeling for limit order book forecasting.
> Anonymous submission.

## Repository structure

```
TimeGrad/
├── lobgrad/             # Main Python package
│   ├── config.py        # BenchmarkConfig (all hyperparameters)
│   ├── paths.py         # REPO_ROOT, DATA_DIR, CHECKPOINTS_DIR, FIGURES_DIR
│   ├── data/            # Data loading, normalization, splits
│   ├── models/          # TimeGrad, CGAN, Hawkes, Poisson generators
│   ├── eval/            # Distributional + microstructure metrics
│   └── execution/       # MPC execution benchmark
├── src/                 # TimeGrad core (model, train, encoder)
│                          imported by lobgrad/ via sys.path
├── scripts/
│   ├── run_hpo.py                          # Optuna HPO (TimeGrad + CGAN)
│   ├── train_timegrad_queue_only.py        # Ablation: queue head only
│   ├── train_timegrad_return_only.py       # Ablation: return head only
│   ├── train_timegrad_joint_flat_window.py # Ablation: no recurrent encoder
│   └── bootstrap_tables.py                 # Headline Table 1 with bootstrap CIs (no-Jupyter)
├── notebooks/
│   ├── calibrate_price_level.ipynb   # LOB grid sizing (Fig. A6)
│   ├── paper_figures.ipynb           # All main + appendix figures
│   └── downstream_execution.ipynb    # MPC execution benchmark
├── checkpoints/
│   ├── final/                        # Canonical paper models
├── data/raw/            # BTCUSDT_10-DD.json files (place here)
├── figures/
│   ├── main/                  # Paper-final main figures
│   ├── appendix/              # Paper-final appendix figures
│   ├── tables/                # CSV + LaTeX results tables
│   └── bootstrapped_tables/   # Bootstrap-CI version of Table 1 (CSV + LaTeX + manifest)
├── REPRODUCE.md
├── pyproject.toml
└── LICENSE
```

## Installation

```bash
pip install -e .
```

Python ≥ 3.10. CUDA optional — CPU inference works. All dependencies are in `pyproject.toml`.

## Data

LOBGrad is trained on BTCUSDT order-book snapshots (Oct 1–21, 2023, 5-second resolution).
Place files at `data/raw/BTCUSDT_10-DD.json` for DD ∈ 01..21.

See `data/README.md` for the schema and acquisition notes.

## Reproducing results

See [REPRODUCE.md](REPRODUCE.md) for the full step-by-step guide.

## Training from scratch

```bash
# Full HPO sweep — TimeGrad + CGAN (~24h on 1× A100)
python scripts/run_hpo.py

# Individual ablation models (~2–4h each)
python scripts/train_timegrad_queue_only.py
python scripts/train_timegrad_return_only.py
python scripts/train_timegrad_joint_flat_window.py
```

## Citation

Citation will be added upon acceptance.

## License

MIT — see [LICENSE](LICENSE).

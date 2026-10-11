# US-equity Table 1 (XNAS.ITCH) — reproduction

Regenerates the paper's **Table 1 (distributional metrics)** on Databento
**XNAS.ITCH `mbp-10`** deep-book data for **MSFT** and **GOOGL** (in `data2/`),
reusing the existing `lobgrad/` benchmark pipeline. Produces one table per symbol.

## What's in `data2/`
- `xnas-itch-YYYYMMDD.mbp-10.csv.zst` — 16 trading days, 2026-07-01 → 07-23,
  10 levels/side, CSV (`pretty_px`/`pretty_ts`/`map_symbols` on), symbols MSFT & GOOGL.
- `databento` is **not** needed (CSV, not DBN) — only `zstandard`.

## Pipeline (4 new scripts)

```bash
# 1. Decode CSV.zst -> per-day JSONL snapshots in integer CENTS, RTH-only (13:30-20:00 UTC),
#    sampled every 5 s -> data2/processed/{SYMBOL}_MM-DD.json   (~4,680 snapshots/day)
python scripts/decode_xnas_csv.py

# 2. Calibrate per-symbol grid (delta = price_level, delta_mid_clip) on the train split,
#    so d*delta covers ~98% of 5-s |dmid| (paper Fig. 4 criterion).
#    Writes data2/processed/calibration_{SYMBOL}.json
python scripts/calibrate_equity_grid.py --all
#    -> MSFT: price_level=6c, delta_mid_clip=65c ; GOOGL: price_level=5c, delta_mid_clip=60c

# 3. Train + generate + bootstrap Table 1 (per symbol).
#    Diffusion trains on MPS; CGAN + all generation on CPU (fastest for K-step sampling).
python scripts/run_equity_table.py --symbol MSFT  --device mps --gen-device cpu --cgan-epochs 20 --cgan-max-batches 50
python scripts/run_equity_table.py --symbol GOOGL --device mps --gen-device cpu --cgan-epochs 20 --cgan-max-batches 50
```

Output per symbol: `figures/equity_tables/{SYMBOL}/table_headline.{csv,tex}` + `run_manifest.json`.
Checkpoints: `checkpoints/equity/{SYMBOL}/{joint_timegrad,timegrad_joint_flat_window,timegrad_queue_only,timegrad_return_only,cgan}/`.

## Key adaptations vs the BTC pipeline
- **Prices in integer cents** (`round($ * 100)`) so the loader's floor/ceil rounding is a
  no-op and `price_level` is a clean cent grid. No change to `lobgrad/data/loader.py`.
- **Regular trading hours** only (13:30–20:00 UTC = 09:30–16:00 ET, EDT).
- **Splits** (chronological, `scripts/equity_profiles.py`): train 07-01→07-15 (10 days),
  val 07-16→07-17 (2), test 07-20→07-23 (4).
- **d = 5 levels** (dy = 11), depth 10, 5-s snapshots — same protocol as the paper
  (context 100, horizon 20, 100 paths, 500 windows, 1000-resample bootstrap).

## Config / hyperparameters
`scripts/equity_profiles.py` is the single source of truth (data dir, splits, per-symbol
calibrated grid, TimeGrad trial-88 architecture, CGAN trial-42 architecture). Training and
evaluation build identical configs from it.

## Options
- `--smoke` : tiny budget (2 epochs, 20 windows, 5 paths) to validate wiring.
- `--skip-train` : evaluate existing checkpoints only.
- `--models LOBGrad,LG-Q,...` : subset of columns.
- `--device` / `--gen-device` / `--cgan-device` : cpu | mps | cuda.

## Caveats
- **No new HPO** — paper-selected hyperparameters (TimeGrad trial 88, CGAN trial 42
  architecture) are reused; only weights are retrained on equity data. Exact trial-42 CGAN
  training knobs (lr/n_critic/lambda_gp/stride) weren't recoverable (no Optuna DB), so
  standard WGAN-GP values are used.
- **CGAN budget trimmed** to `epochs=20, max_batches=50` (the deep 80×300 budget is ~37 h
  on CPU; GP double-backward is the bottleneck).
- Absolute metric values differ from the paper (different instrument, period, cent grid);
  the **relative column ordering** is the comparison of interest.
- On this machine CUDA is unavailable; MPS trains ~3.7× faster than CPU, but CPU generates
  ~4× faster than MPS (MPS op-launch overhead on sequential diffusion sampling).

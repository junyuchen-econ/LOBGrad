"""
equity_profiles.py
------------------
Single source of truth for the US-equity (XNAS.ITCH) Table-1 reproduction.

Maps a symbol (MSFT / GOOGL) to its dataset profile — processed-data dir,
chronological train/val/test split, grid resolution (levels, price_level in
CENTS, delta_mid_clip in CENTS) — and bundles the paper's Table-1 architecture
(TimeGrad trial 88) and CGAN architecture (trial 42) so that TRAINING and
EVALUATION always build byte-identical configs.

Grid parameters (price_level, delta_mid_clip) are per-symbol and read from
``data2/processed/calibration_{SYMBOL}.json`` written by
``scripts/calibrate_equity_grid.py``. Run the calibrator once before training.

Prices are integer cents (the decoder multiplies dollars by 100), so the
loader's floor/ceil rounding is a no-op and ``price_level`` is a cent grid.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = REPO_ROOT / "data2" / "processed"

# Make the lobgrad package (repo root) and legacy src/ importable regardless of CWD.
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lobgrad.config import BenchmarkConfig

SYMBOLS = ["MSFT", "GOOGL"]

# Chronological split over the 16 available XNAS.ITCH trading days
# (2026-07-01 .. 2026-07-23; weekends/holidays absent files are skipped by the
# loader). Ranges are inclusive. -> ~10 train / 2 val / 4 test days.
SPLITS = {
    "train": ("2026-07-01", "2026-07-15"),
    "val":   ("2026-07-16", "2026-07-17"),
    "test":  ("2026-07-20", "2026-07-23"),
}

# Shared data settings (queue representation + snapshot cadence + depth)
DATA = dict(
    levels       = 5,        # d=5 -> dy = 2d+1 = 11 (paper faithful; 10 levels available)
    depth        = 10,       # raw levels available in mbp-10
    queue_repr   = "signed",
    queue_mode   = "absolute",
    time_diff_ms = 5000,
    context_len  = 100,
    horizon      = 20,
)

# TimeGrad Table-1 architecture + training recipe (HPO trial 88), shared by
# LOBGrad, LG-Q, LG-R, LG-NE. Overridden per-model where noted in the driver
# (LG-NE uses cell_type="flat_window", use_cond_projection=False).
ARCH = dict(
    timegrad_cell_type            = "LSTM",
    timegrad_hidden_enc           = 151,
    timegrad_hidden_noise         = 255,
    timegrad_num_enc_layers       = 1,
    timegrad_emb_dim              = 32,
    timegrad_k                    = 102,
    timegrad_schedule             = "linear",
    timegrad_noise_type           = "mlp",
    timegrad_use_cond_projection  = True,
    timegrad_loss_type            = "huber",
    timegrad_dropout              = 0.1781322278935634,
    timegrad_beta_start           = 0.0001,
    timegrad_beta_end             = 0.1,
    timegrad_history_length       = 100,
    timegrad_prediction_length    = 20,
    timegrad_batch_size           = 256,
    timegrad_lr                   = 0.004038458201855693,
    timegrad_epochs               = 50,
    timegrad_max_batches          = 1000,
)

# CGAN architecture from HPO trial 42 (latent/hidden/layers) + standard WGAN-GP
# training knobs (the exact trial-42 lr/n_critic/lambda_gp/stride are not
# recoverable from the repo — the Optuna DB is absent — so we use config
# defaults, matching the deep-train budget epochs=80/max_batches=300).
CGAN_ARCH = dict(
    cgan_latent_dim       = 74,
    cgan_hidden_dim       = 367,
    cgan_n_encoder_layers = 1,
    cgan_n_decoder_layers = 2,
    cgan_lr_g             = 1e-4,
    cgan_lr_d             = 1e-4,
    cgan_n_critic         = 3,
    cgan_lambda_gp        = 10.0,
    cgan_batch_size       = 64,
    cgan_epochs           = 80,
    cgan_max_batches      = 300,
    cgan_stride           = 10,
)


def calibration_path(symbol: str) -> Path:
    return PROC_DIR / f"calibration_{symbol}.json"


def load_calibration(symbol: str) -> dict:
    p = calibration_path(symbol)
    if not p.exists():
        raise FileNotFoundError(
            f"No calibration for {symbol} at {p}. "
            f"Run: python scripts/calibrate_equity_grid.py --symbol {symbol}"
        )
    return json.loads(p.read_text())


def make_config(symbol: str, price_level: float | None = None,
                delta_mid_clip: float | None = None, **overrides) -> BenchmarkConfig:
    """Build a BenchmarkConfig for `symbol` (data + splits + arch). Per-model
    checkpoint/save_dir/train flags are applied by the driver via **overrides."""
    if symbol not in SYMBOLS:
        raise ValueError(f"unknown symbol {symbol!r}; expected one of {SYMBOLS}")
    if price_level is None or delta_mid_clip is None:
        cal = load_calibration(symbol)
        price_level    = cal["price_level"]    if price_level    is None else price_level
        delta_mid_clip = cal["delta_mid_clip"] if delta_mid_clip is None else delta_mid_clip

    kw = dict(
        data_dir       = str(PROC_DIR),
        symbol         = symbol,
        price_level    = float(price_level),
        delta_mid_clip = float(delta_mid_clip),
        train_start    = SPLITS["train"][0], train_end = SPLITS["train"][1],
        val_start      = SPLITS["val"][0],   val_end   = SPLITS["val"][1],
        test_start     = SPLITS["test"][0],  test_end  = SPLITS["test"][1],
        **DATA, **ARCH, **CGAN_ARCH,
    )
    kw.update(overrides)
    return BenchmarkConfig(**kw)


def ckpt_dir(symbol: str, model: str) -> Path:
    return REPO_ROOT / "checkpoints" / "equity" / symbol / model


def out_dir(symbol: str) -> Path:
    return REPO_ROOT / "figures" / "equity_tables" / symbol


def cache_dir(symbol: str) -> Path:
    return REPO_ROOT / "cache" / "equity_generations" / symbol

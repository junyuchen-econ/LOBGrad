"""
train_timegrad_queue_only.py
----------------------------
Train a *queue-only* TimeGrad diffusion model with the **same architecture and
training recipe** as the final joint model (`checkpoints/final/joint_timegrad/
best_model.pt`, HPO trial 88 in `optuna_deep_v2/`), but without the mid-price
head. This is the queue-only baseline for the JointTimeGrad ablation in the
NeurIPS paper (Table A2 / "is joint modelling actually helping the queue?").

Hyperparameters pulled from the trial_88 Optuna record (see `checkpoints/
optuna_v2/hpo_optuna.db`):

    Architecture                     Training recipe
    ----------------------------     -------------------------------
    cell_type            = LSTM      lr               = 0.004038
    hidden_enc           = 151       loss_type        = huber
    hidden_noise         = 255       dropout          = 0.178
    num_enc_layers       = 1         prediction_length = 20  (= HORIZON)
    emb_dim              = 32        batch_size       = 256
    k (diffusion steps)  = 102       epochs           = 50
    schedule             = linear    max_batches      = 1000
    noise_type           = mlp
    use_cond_projection  = True

Stripping the +1 price dim turns input/output from 2K+1=11 to 2K=10. Every
other knob is held at the trial_88 value so the two checkpoints differ only
in the price-head dimension — making the queue-only vs joint comparison a
clean ablation of "does joint modelling help the queue?"

Output: checkpoints/timegrad_queue_only/{best_model.pt, final_model.pt, training_loss.png}

Usage:
    python scripts/train_timegrad_queue_only.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")


def _find_root() -> Path:
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "lobgrad" / "__init__.py").is_file():
            return p
        nested = p / "LobBench_comparison"
        if (nested / "lobgrad" / "__init__.py").is_file():
            return nested
    raise RuntimeError("Could not find lobgrad package")


ROOT = _find_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from lobgrad.config import BenchmarkConfig
from lobgrad.data.splits import date_splits
from lobgrad.models.timegrad import TimeGradGenerator


SAVE_DIR   = "checkpoints/timegrad_queue_only"
CKPT_PATH  = f"{SAVE_DIR}/best_model.pt"


def build_config() -> BenchmarkConfig:
    """Match trial_88 architecture *and* training recipe exactly; queue-only data path."""
    return BenchmarkConfig(
        # Data — must match the joint training run
        levels                       = 5,
        price_level                  = 2.0,
        # Architecture — from joint trial_88 Optuna record
        timegrad_cell_type           = "LSTM",
        timegrad_hidden_enc          = 151,
        timegrad_num_enc_layers      = 1,
        timegrad_emb_dim             = 32,
        timegrad_hidden_noise        = 255,
        timegrad_k                   = 102,
        timegrad_schedule            = "linear",
        timegrad_noise_type          = "mlp",
        timegrad_use_cond_projection = True,
        # Training recipe — from joint trial_88 Optuna record (run_hpo.py Stage 3)
        timegrad_loss_type           = "huber",
        timegrad_dropout             = 0.1781322278935634,
        timegrad_train               = True,
        timegrad_history_length      = 100,
        timegrad_prediction_length   = 20,     # = HORIZON in run_hpo.py (joint trained on 20-step targets)
        timegrad_batch_size          = 256,    # match run_hpo.py tg_trial_to_cfg line 240
        timegrad_lr                  = 0.004038458201855693,
        timegrad_epochs              = 50,     # match run_hpo.py Stage 3 deep pass
        timegrad_max_batches         = 1000,   # match run_hpo.py Stage 3 deep pass
        # Output — disjoint from any existing checkpoint dir
        timegrad_save_dir            = SAVE_DIR,
        timegrad_checkpoint          = CKPT_PATH,
    )


def main() -> None:
    cfg = build_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[queue-only ablation] device={device}")
    print(f"[queue-only ablation] save_dir={cfg.timegrad_save_dir}")
    print(f"[queue-only ablation] arch: K={cfg.levels}, cell={cfg.timegrad_cell_type}, "
          f"hidden_enc={cfg.timegrad_hidden_enc}, hidden_noise={cfg.timegrad_hidden_noise}, "
          f"diffusion_K={cfg.timegrad_k}, emb={cfg.timegrad_emb_dim}, "
          f"cond_proj={cfg.timegrad_use_cond_projection}, loss={cfg.timegrad_loss_type}")

    if os.path.exists(CKPT_PATH):
        print(f"[queue-only ablation] WARNING: {CKPT_PATH} already exists. "
              f"Training will overwrite as new best epochs improve val loss.")

    os.makedirs(cfg.timegrad_save_dir, exist_ok=True)

    # ── Load queue-only data (no price anchor needed for training) ────────────
    print("[queue-only ablation] loading data splits…")
    t0 = time.time()
    (ts_train, X_train, _anchors_train), \
    (ts_val,   X_val,   _anchors_val), \
    (ts_test,  X_test,  _anchors_test), \
    normalizer = date_splits(cfg)
    print(f"[queue-only ablation] data loaded in {time.time()-t0:.1f}s. "
          f"train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    assert X_train.shape[1] == 2 * cfg.levels, \
        f"expected feature dim {2*cfg.levels}, got {X_train.shape[1]}"

    # ── Build generator + train ──────────────────────────────────────────────
    print("[queue-only ablation] building TimeGradGenerator (random init)…")
    generator = TimeGradGenerator(cfg, normalizer=normalizer, device=device)

    n_params = sum(p.numel() for p in generator.model.parameters())
    print(f"[queue-only ablation] model params = {n_params:,}")

    print("[queue-only ablation] starting training…")
    t0 = time.time()
    generator.fit(X_train, ts_train)
    print(f"[queue-only ablation] training finished in {(time.time()-t0)/60:.1f} min")

    # ── Verify the checkpoint landed and shapes are correct ──────────────────
    if not os.path.exists(CKPT_PATH):
        raise RuntimeError(f"expected checkpoint at {CKPT_PATH}, none found")

    state = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    sd = state["model_state_dict"]
    print(f"[queue-only ablation] saved best_model.pt: epoch={state.get('epoch','?')}, "
          f"val_loss={state.get('val_loss','?'):.4f}")

    in_shape  = tuple(sd["encoder.rnn.weight_ih_l0"].shape)
    out_shape = tuple(sd["noise_net.net.4.weight"].shape)
    print(f"[queue-only ablation] encoder input → {in_shape} (expect (604, 10))")
    print(f"[queue-only ablation] noise_net out → {out_shape} (expect (10, 255))")
    assert in_shape  == (604, 10), f"unexpected encoder input shape: {in_shape}"
    assert out_shape == (10, 255), f"unexpected noise_net output shape: {out_shape}"

    # ── Sanity-generate one batch from a held-out test context ────────────────
    print("[queue-only ablation] sanity generation from test context…")
    ctx = X_test[:cfg.timegrad_history_length]                # (100, 2K)
    samples = generator.generate(
        context=ctx, steps=20, n_paths=4, rng=np.random.default_rng(0)
    )
    print(f"[queue-only ablation] sample shape = {samples.shape} (expect (4, 20, 10))")
    assert samples.shape == (4, 20, 2 * cfg.levels), \
        f"unexpected sample shape: {samples.shape}"
    assert samples.dtype == np.float32, f"expected float32, got {samples.dtype}"

    # Sign convention check (signed repr: bids ≤ 0, asks ≥ 0)
    if cfg.queue_repr == "signed":
        bid_max = float(samples[..., :cfg.levels].max())
        ask_min = float(samples[..., cfg.levels:].min())
        print(f"[queue-only ablation] sign check: bid_max={bid_max:.3f} (≤0?), "
              f"ask_min={ask_min:.3f} (≥0?)")

    print(f"[queue-only ablation] DONE. Checkpoint at {CKPT_PATH}")


if __name__ == "__main__":
    main()

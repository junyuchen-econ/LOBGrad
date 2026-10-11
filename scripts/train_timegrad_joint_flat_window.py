"""
train_timegrad_joint_flat_window.py
-----------------------------------
Train a *joint* TimeGrad diffusion model that is **identical to the joint
trial_88 baseline (`checkpoints/final/joint_timegrad/best_model.pt`) except
the LSTM encoder is replaced by a flat-window pass-through**. Per-position
conditioning becomes the concatenation of the previous ``window_size``
observations (no recurrent state, no learned summarizer).

This is the encoderless-but-information-matched companion to LG-NE.
LG-NE conditions on the previous *single* observation (Markov-1);
LG-NE-Window conditions on the previous ``window_size`` observations
flattened into a vector of dim ``window_size * (2K+1)``. With
``window_size = cfg.timegrad_history_length`` (=100), the encoderless
baseline has the same information content as LOBGrad's recurrent encoder
but does not summarize it — the noise predictor sees the raw flattened
1100-dim history vector. The gap to LOBGrad isolates the contribution of
recurrent summarization (Cor 4.4: encoder reduces effective conditioning
cost from s*d_y to m).

How "flat window" is achieved:
    The encoder is selected by ``cfg.timegrad_cell_type``. Setting it to
    ``"flat_window"`` makes ``_TimeGradBase`` build a ``FlatWindowEncoder``
    (defined in ``src/model/encoder.py``). Per-position output is the
    rolling concatenation of the last ``window_size`` observations
    (zero-padded for early positions). With ``use_cond_projection=False``
    the conditioning is the raw flattened window of dim
    ``window_size * input_dim``.

Net effect:
    • Training:  cond_t = concat(full_joint[t-s], ..., full_joint[t-1])
                 (raw window of length s, zero-padded on the left)
    • Sampling:  window slides by one each step; sampled y_t enters the
                 right end of the window for the next step.

Output: checkpoints/timegrad_joint_flat_window/{best_model.pt, final_model.pt, training_loss.png}

Usage:
    python scripts/train_timegrad_joint_flat_window.py
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
import torch.nn as nn

from lobgrad.config import BenchmarkConfig
from lobgrad.data.splits import date_splits
from lobgrad.execution.simulate import build_joint_features

# TimeGradMLP and the training utilities live in the legacy `src/` tree.
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from model import TimeGradMLP, FlatWindowEncoder   # type: ignore
from train import TimeSeriesDataset, train_model   # type: ignore


SAVE_DIR  = "checkpoints/timegrad_joint_flat_window"
CKPT_PATH = f"{SAVE_DIR}/best_model.pt"


# ============================== Config ==============================
def build_config() -> BenchmarkConfig:
    """Match the joint trial_88 architecture and training recipe exactly,
    except cell_type is set to ``"flat_window"`` to disable the recurrent
    encoder and instead feed the flattened ``window_size``-step history."""
    return BenchmarkConfig(
        # Data — joint feature path
        levels                       = 5,
        price_level                  = 2.0,
        delta_mid_clip               = 12.0,
        # Architecture — from joint trial_88 Optuna record
        timegrad_cell_type           = "flat_window",  # ← flat-window encoder
        timegrad_hidden_enc          = 151,            # ignored when cell_type='flat_window'
        timegrad_num_enc_layers      = 1,              # ignored when cell_type='flat_window'
        timegrad_emb_dim             = 32,
        timegrad_hidden_noise        = 255,
        timegrad_k                   = 102,
        timegrad_schedule            = "linear",
        timegrad_noise_type          = "mlp",
        timegrad_use_cond_projection = False,          # ← cond_dim = INPUT_DIM * window_size (raw passthrough)
        # Training recipe
        timegrad_loss_type           = "l1",
        timegrad_dropout             = 0.1781322278935634,
        timegrad_train               = True,
        timegrad_history_length      = 100,
        timegrad_prediction_length   = 20,
        timegrad_batch_size          = 256,
        timegrad_lr                  = 0.004038458201855693,
        timegrad_epochs              = 50,
        timegrad_max_batches         = 1000,
        # Output
        timegrad_save_dir            = SAVE_DIR,
        timegrad_checkpoint          = CKPT_PATH,
    )


# ============================== Train ==============================
def main() -> None:
    cfg = build_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    INPUT_DIM   = 2 * cfg.levels + 1   # 2K queue + 1 Δmid
    WINDOW_SIZE = cfg.timegrad_history_length
    COND_DIM    = INPUT_DIM * WINDOW_SIZE

    print(f"[joint flat-window] device={device}")
    print(f"[joint flat-window] save_dir={cfg.timegrad_save_dir}")
    print(f"[joint flat-window] arch: FLAT-WINDOW ENCODER (FlatWindowEncoder), "
          f"input_dim={INPUT_DIM}, window_size={WINDOW_SIZE}, cond_dim={COND_DIM} (flattened window), "
          f"hidden_noise={cfg.timegrad_hidden_noise}, diffusion_K={cfg.timegrad_k}, "
          f"emb={cfg.timegrad_emb_dim}, loss={cfg.timegrad_loss_type}")
    print(f"[joint flat-window] training: lr={cfg.timegrad_lr:.5g}, "
          f"bs={cfg.timegrad_batch_size}, epochs={cfg.timegrad_epochs}, "
          f"max_batches={cfg.timegrad_max_batches}, "
          f"pred_len={cfg.timegrad_prediction_length}, hist_len={cfg.timegrad_history_length}")

    if os.path.exists(CKPT_PATH):
        print(f"[joint flat-window] WARNING: {CKPT_PATH} already exists. "
              f"Training will overwrite as new best epochs improve val loss.")

    os.makedirs(cfg.timegrad_save_dir, exist_ok=True)

    # ── Load splits — keep both queue features and anchor (mid-price) series ─
    print("[joint flat-window] loading data splits…")
    t0 = time.time()
    (ts_train, X_train, anchors_train), \
    (ts_val,   X_val,   anchors_val), \
    (ts_test,  X_test,  anchors_test), \
    _normalizer = date_splits(cfg)
    print(f"[joint flat-window] data loaded in {time.time()-t0:.1f}s. "
          f"X_train={X_train.shape}, anchors_train={anchors_train.shape}")

    # ── Build joint feature stream (T, 2K+1) — matches JointTimeGradGenerator.fit ─
    X_joint = build_joint_features(anchors_train, X_train, cfg.delta_mid_clip)
    X_joint[:, 0] = np.clip(X_joint[:, 0], -cfg.delta_mid_clip, cfg.delta_mid_clip)
    queue_clip_lo = 0.0 if cfg.queue_repr == "unsigned" else -5.0
    X_joint[:, 1:] = np.clip(X_joint[:, 1:], queue_clip_lo, 5.0)
    print(f"[joint flat-window] joint features: shape={X_joint.shape}, "
          f"Δmid std={float(X_joint[:, 0].std()):.4f}, "
          f"Δmid frac_clipped={float(np.mean(np.abs(X_joint[:, 0]) >= cfg.delta_mid_clip - 1e-6)):.4f}, "
          f"queue range=[{float(X_joint[:, 1:].min()):+.2f}, {float(X_joint[:, 1:].max()):+.2f}]")
    assert X_joint.shape[1] == INPUT_DIM, f"expected feature dim {INPUT_DIM}, got {X_joint.shape[1]}"

    # ── Build TimeGradMLP — cell_type='flat_window' selects FlatWindowEncoder ──
    print(f"[joint flat-window] building TimeGradMLP with cell_type='flat_window', window_size={WINDOW_SIZE}…")
    model = TimeGradMLP(
        input_dim           = INPUT_DIM,
        output_dim          = INPUT_DIM,
        hidden_encoder_dim  = cfg.timegrad_hidden_enc,    # ignored when cell_type='flat_window'
        hidden_noise_dim    = cfg.timegrad_hidden_noise,
        num_encoder_layers  = cfg.timegrad_num_enc_layers,
        emb_dim             = cfg.timegrad_emb_dim,
        K                   = cfg.timegrad_k,
        beta_start          = cfg.timegrad_beta_start,
        beta_end            = cfg.timegrad_beta_end,
        schedule            = cfg.timegrad_schedule,
        loss_type           = cfg.timegrad_loss_type,
        dropout_rate        = cfg.timegrad_dropout,
        cell_type           = cfg.timegrad_cell_type,
        time_embedding      = cfg.timegrad_time_embedding,
        use_cond_projection = cfg.timegrad_use_cond_projection,
        flat_window_size    = WINDOW_SIZE,
    )
    assert isinstance(model.encoder, FlatWindowEncoder), \
        f"expected FlatWindowEncoder, got {type(model.encoder).__name__}"
    assert model.encoder.window_size == WINDOW_SIZE, \
        f"expected window_size={WINDOW_SIZE}, got {model.encoder.window_size}"
    assert isinstance(model.cond_projection, nn.Identity), \
        "use_cond_projection=False should leave cond_projection as nn.Identity"

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[joint flat-window] model params = {n_params:,} "
          f"(no encoder.* params; only diffusion + noise_net)")

    # ── Dataloader (90/10 train/val split inside the train slice) ────────────
    dataset = TimeSeriesDataset(
        targets           = X_joint,
        context_length    = cfg.timegrad_history_length,
        prediction_length = cfg.timegrad_prediction_length,
    )
    indices = dataset.valid_indices
    split   = int(0.9 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    print(f"[joint flat-window] dataset windows: train={len(train_idx)}, val={len(val_idx)}")

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=cfg.timegrad_batch_size, shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        batch_size=cfg.timegrad_batch_size, shuffle=False,
    )

    # ── Optimiser + scheduler (matches JointTimeGradGenerator.fit) ──────────
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.timegrad_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.timegrad_epochs, eta_min=1e-5
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print("[joint flat-window] starting training…")
    t0 = time.time()
    train_model(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        optimizer    = optimizer,
        num_epochs   = cfg.timegrad_epochs,
        device       = torch.device(device),
        verbose      = True,
        save_dir     = cfg.timegrad_save_dir,
        max_batches  = cfg.timegrad_max_batches,
        scheduler    = scheduler,
    )
    print(f"[joint flat-window] training finished in {(time.time()-t0)/60:.1f} min")

    # ── Verify the checkpoint landed and shapes are correct ──────────────────
    if not os.path.exists(CKPT_PATH):
        raise RuntimeError(f"expected checkpoint at {CKPT_PATH}, none found")

    state = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    sd    = state["model_state_dict"]
    print(f"[joint flat-window] saved best_model.pt: epoch={state.get('epoch','?')}, "
          f"val_loss={state.get('val_loss','?'):.4f}")

    assert not any(k.startswith("encoder.") for k in sd), \
        "flat-window model should not contain any encoder.* parameters"
    assert not any(k.startswith("cond_projection.") for k in sd), \
        "use_cond_projection=False should leave no cond_projection params"

    in_shape  = tuple(sd["noise_net.net.0.weight"].shape)
    out_shape = tuple(sd["noise_net.net.4.weight"].shape)
    expected_in = (cfg.timegrad_hidden_noise, INPUT_DIM + COND_DIM + cfg.timegrad_emb_dim)
    print(f"[joint flat-window] noise_net in  → {in_shape} (expect {expected_in})")
    print(f"[joint flat-window] noise_net out → {out_shape} (expect ({INPUT_DIM}, {cfg.timegrad_hidden_noise}))")
    assert in_shape  == expected_in,                         f"unexpected noise_net input shape: {in_shape}"
    assert out_shape == (INPUT_DIM, cfg.timegrad_hidden_noise), f"unexpected noise_net output shape: {out_shape}"

    # ── Sanity-generate one batch from a held-out test context ──────────────
    print("[joint flat-window] sanity generation from test context…")
    ctx_anchors = anchors_test[: cfg.timegrad_history_length + 1]   # +1 for first diff
    ctx_queue   = X_test[: cfg.timegrad_history_length + 1]
    ctx_joint   = build_joint_features(ctx_anchors, ctx_queue, cfg.delta_mid_clip)[1:]   # drop leading 0
    assert ctx_joint.shape == (cfg.timegrad_history_length, INPUT_DIM)

    model.eval()
    with torch.no_grad():
        x_hist  = torch.from_numpy(ctx_joint.astype(np.float32)).unsqueeze(0).to(device)
        x_hist  = x_hist.repeat(4, 1, 1)
        samples = model.sample(x_hist, prediction_steps=20).cpu().numpy()

    print(f"[joint flat-window] sample shape = {samples.shape} (expect (4, 20, {INPUT_DIM}))")
    assert samples.shape == (4, 20, INPUT_DIM), f"unexpected sample shape: {samples.shape}"
    print(f"[joint flat-window] sample stats — Δmid: "
          f"min={float(samples[..., 0].min()):+.3f}, max={float(samples[..., 0].max()):+.3f}, "
          f"std={float(samples[..., 0].std()):.4f}")
    print(f"[joint flat-window] sample stats — queue: "
          f"min={float(samples[..., 1:].min()):+.3f}, max={float(samples[..., 1:].max()):+.3f}, "
          f"std={float(samples[..., 1:].std()):.4f}")

    print(f"[joint flat-window] DONE. Checkpoint at {CKPT_PATH}")


if __name__ == "__main__":
    main()

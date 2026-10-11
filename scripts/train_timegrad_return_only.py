"""
train_timegrad_return_only.py
-----------------------------
Train a *return-only* TimeGrad diffusion model with the **same architecture and
training recipe** as the final joint model (`checkpoints/final/joint_timegrad/
best_model.pt`, HPO trial 88 in `optuna_deep_v2/`), but with no queue head.
This is the symmetric companion to `train_timegrad_queue_only.py`: queue-only
ablates the price head, return-only ablates the queue head. Together they
isolate the contribution of joint modelling on each modality.

The single feature is Δmid (raw USDT, clipped to ±cfg.delta_mid_clip), built
identically to `lobgrad.execution.simulate.build_joint_features` so the joint
and return-only models train on the *exact same Δmid stream*.

Hyperparameters pulled from the trial_88 Optuna record:

    Architecture                     Training recipe
    ----------------------------     -------------------------------
    cell_type            = LSTM      lr               = 0.004038
    hidden_enc           = 151       loss_type        = huber
    hidden_noise         = 255       dropout          = 0.178
    num_enc_layers       = 1         prediction_length = 20  (= HORIZON)
    emb_dim              = 32        batch_size       = 256
    k (diffusion steps)  = 102       epochs           = 50
    schedule             = linear    max_batches      = 1000
    noise_type           = mlp       delta_mid_clip   = 12.0  (USDT)
    use_cond_projection  = True

Stripping the 2K queue dims turns input/output from 2K+1=11 to 1. Every other
knob is held at the trial_88 value so the two checkpoints differ only in the
feature dimension — making the return-only vs joint comparison a clean ablation
of "does joint modelling help the price?"

Output: checkpoints/timegrad_return_only/{best_model.pt, final_model.pt, training_loss.png}

Usage:
    python scripts/train_timegrad_return_only.py
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

# TimeGradMLP and the training utilities live in the legacy `src/` tree.
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from model import TimeGradMLP                      # type: ignore
from train import TimeSeriesDataset, train_model   # type: ignore


SAVE_DIR  = "checkpoints/timegrad_return_only"
CKPT_PATH = f"{SAVE_DIR}/best_model.pt"


def build_config() -> BenchmarkConfig:
    """Match trial_88 architecture *and* training recipe exactly; return-only data path."""
    return BenchmarkConfig(
        # Data — must match the joint training run
        levels                       = 5,
        price_level                  = 2.0,
        delta_mid_clip               = 12.0,    # joint default; clip applied in raw USDT
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
        timegrad_prediction_length   = 20,    # = HORIZON in run_hpo.py
        timegrad_batch_size          = 256,
        timegrad_lr                  = 0.004038458201855693,
        timegrad_epochs              = 50,
        timegrad_max_batches         = 1000,
        # Output — disjoint from any existing checkpoint dir
        timegrad_save_dir            = SAVE_DIR,
        timegrad_checkpoint          = CKPT_PATH,
    )


def build_delta_mid(anchors: np.ndarray, clip: float) -> np.ndarray:
    """Mirror lobgrad.execution.simulate.build_joint_features for the Δmid column.

    Returns an (T, 1) float32 array. Row 0 is zero (no previous mid). Subsequent
    rows are anchors[t] - anchors[t-1] in raw USDT, clipped to ±clip. The
    clipping matches the joint training pipeline, so the joint and return-only
    models see an identical Δmid sequence.
    """
    T = len(anchors)
    delta = np.zeros(T, dtype=np.float32)
    delta[1:] = (anchors[1:] - anchors[:-1]).astype(np.float32)
    delta = np.clip(delta, -clip, clip).reshape(-1, 1)
    return delta


def main() -> None:
    cfg = build_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[return-only ablation] device={device}")
    print(f"[return-only ablation] save_dir={cfg.timegrad_save_dir}")
    print(f"[return-only ablation] arch: cell={cfg.timegrad_cell_type}, "
          f"hidden_enc={cfg.timegrad_hidden_enc}, hidden_noise={cfg.timegrad_hidden_noise}, "
          f"diffusion_K={cfg.timegrad_k}, emb={cfg.timegrad_emb_dim}, "
          f"cond_proj={cfg.timegrad_use_cond_projection}, loss={cfg.timegrad_loss_type}, "
          f"dropout={cfg.timegrad_dropout:.3f}")
    print(f"[return-only ablation] training: lr={cfg.timegrad_lr:.5g}, "
          f"bs={cfg.timegrad_batch_size}, epochs={cfg.timegrad_epochs}, "
          f"max_batches={cfg.timegrad_max_batches}, "
          f"pred_len={cfg.timegrad_prediction_length}, hist_len={cfg.timegrad_history_length}")

    if os.path.exists(CKPT_PATH):
        print(f"[return-only ablation] WARNING: {CKPT_PATH} already exists. "
              f"Training will overwrite as new best epochs improve val loss.")

    os.makedirs(cfg.timegrad_save_dir, exist_ok=True)

    # ── Load splits — keep only the anchor (mid-price) series ────────────────
    print("[return-only ablation] loading data splits…")
    t0 = time.time()
    (ts_train, _X_train, anchors_train), \
    (ts_val,   _X_val,   anchors_val), \
    (ts_test,  _X_test,  anchors_test), \
    _normalizer = date_splits(cfg)
    print(f"[return-only ablation] data loaded in {time.time()-t0:.1f}s. "
          f"anchors_train={anchors_train.shape}, "
          f"anchors_val={anchors_val.shape}, anchors_test={anchors_test.shape}")

    # ── Build Δmid feature stream (T, 1) ─────────────────────────────────────
    delta_train = build_delta_mid(anchors_train, cfg.delta_mid_clip)
    print(f"[return-only ablation] Δmid stream: shape={delta_train.shape}, "
          f"min={float(delta_train.min()):+.3f}, max={float(delta_train.max()):+.3f}, "
          f"std={float(delta_train.std()):.4f}, "
          f"frac_clipped={float(np.mean(np.abs(delta_train) >= cfg.delta_mid_clip - 1e-6)):.4f}")
    assert delta_train.shape[1] == 1, f"expected feature dim 1, got {delta_train.shape[1]}"

    # ── Build the model directly (TimeGradGenerator hardcodes 2K input_dim) ─
    print("[return-only ablation] building TimeGradMLP (random init, input_dim=output_dim=1)…")
    model = TimeGradMLP(
        input_dim           = 1,
        output_dim          = 1,
        hidden_encoder_dim  = cfg.timegrad_hidden_enc,
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
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[return-only ablation] model params = {n_params:,}")

    # ── Dataloader (90/10 train/val split inside the train slice) ────────────
    dataset = TimeSeriesDataset(
        targets           = delta_train,
        context_length    = cfg.timegrad_history_length,
        prediction_length = cfg.timegrad_prediction_length,
    )
    indices = dataset.valid_indices
    split   = int(0.9 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    print(f"[return-only ablation] dataset windows: train={len(train_idx)}, val={len(val_idx)}")

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
    print("[return-only ablation] starting training…")
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
    print(f"[return-only ablation] training finished in {(time.time()-t0)/60:.1f} min")

    # ── Verify the checkpoint landed and shapes are correct ──────────────────
    if not os.path.exists(CKPT_PATH):
        raise RuntimeError(f"expected checkpoint at {CKPT_PATH}, none found")

    state = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    sd    = state["model_state_dict"]
    print(f"[return-only ablation] saved best_model.pt: epoch={state.get('epoch','?')}, "
          f"val_loss={state.get('val_loss','?'):.4f}")

    in_shape  = tuple(sd["encoder.rnn.weight_ih_l0"].shape)
    out_shape = tuple(sd["noise_net.net.4.weight"].shape)
    print(f"[return-only ablation] encoder input → {in_shape} (expect (604, 1))")
    print(f"[return-only ablation] noise_net out → {out_shape} (expect (1, 255))")
    assert in_shape  == (604, 1),   f"unexpected encoder input shape: {in_shape}"
    assert out_shape == (1,   255), f"unexpected noise_net output shape: {out_shape}"

    # ── Sanity-generate one batch from a held-out test context ──────────────
    print("[return-only ablation] sanity generation from test context…")
    ctx_anchors  = anchors_test[: cfg.timegrad_history_length + 1]   # need +1 for first diff
    ctx_delta    = build_delta_mid(ctx_anchors, cfg.delta_mid_clip)[1:]   # drop the leading 0
    assert ctx_delta.shape == (cfg.timegrad_history_length, 1)

    model.eval()
    with torch.no_grad():
        x_hist = torch.from_numpy(ctx_delta).unsqueeze(0).to(device)   # (1, ctx_len, 1)
        x_hist = x_hist.repeat(4, 1, 1)                                # 4 paths share the context
        samples = model.sample(x_hist, prediction_steps=20).cpu().numpy()

    print(f"[return-only ablation] sample shape = {samples.shape} (expect (4, 20, 1))")
    assert samples.shape == (4, 20, 1), f"unexpected sample shape: {samples.shape}"
    print(f"[return-only ablation] sample stats: "
          f"min={float(samples.min()):+.3f}, max={float(samples.max()):+.3f}, "
          f"mean={float(samples.mean()):+.4f}, std={float(samples.std()):.4f}")

    print(f"[return-only ablation] DONE. Checkpoint at {CKPT_PATH}")


if __name__ == "__main__":
    main()

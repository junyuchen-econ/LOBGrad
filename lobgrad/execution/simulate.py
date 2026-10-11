"""
simulate.py
-----------
JointTimeGradGenerator — extends TimeGrad to model (Δmid, queue) jointly.

Feature vector per step (2k+1 dims):
  index 0      : raw Δmid in USDT, clipped to ±cfg.delta_mid_clip
  indices 1..2k: normalised queue features (signed log transform via Normalizer)

The price column is intentionally *not* log-compressed — at this tick
resolution raw USDT deltas are directly interpretable and the compression
adds no measurable benefit. See `build_joint_features` below.

Training:
  - Compute Δanchor_t = anchor_t - anchor_{t-1} from the raw anchor series
    (Δanchor_0 = 0 by convention)
  - Clip to ±cfg.delta_mid_clip and prepend to the normalised queue features
  - Train TimeGrad on the resulting (T, 2k+1) series

Generation:
  - model.sample() returns (n_paths, steps, 2k+1)
  - First dim → raw Δmid (clipped); reconstruct mid_t = mid_{t-1} + Δmid_t,
    seeded from the last context anchor (cumulative sum)
  - Remaining dims → normalised queue features (same as standard TimeGrad)
  - Queue features inverse-transformed to raw absolute sizes for cost computation
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch

# Make src/ importable
_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from model import TimeGradMLP, TimeGradWaveNet   # type: ignore
from train import train_model, train_epoch, validate, TimeSeriesDataset  # type: ignore

from ..config import BenchmarkConfig
from ..data.normalizer import Normalizer


# Default Δmid clip (raw USDT). Callers should pass cfg.delta_mid_clip; the
# constant is kept for callers that build joint features without a config.
# Log-signed transform is intentionally omitted — at this tick resolution the
# compression adds no benefit and raw USDT deltas are directly interpretable.
DELTA_MID_CLIP = 100.0


def build_joint_features(
    anchors:    np.ndarray,   # (T,) raw mid-prices (anchor prices)
    queue_norm: np.ndarray,   # (T, 2k) normalised queue features
    clip:       float = DELTA_MID_CLIP,
) -> np.ndarray:
    """
    Build (T, 2k+1) joint feature matrix.

    Index 0 : raw Δmid (USDT), clipped to ±clip. Row 0 = 0.
    Index 1…: normalised queue features (unchanged).
    """
    T = len(anchors)
    delta_mid     = np.zeros(T, dtype=np.float32)
    delta_mid[1:] = (anchors[1:] - anchors[:-1]).astype(np.float32)
    delta_mid     = np.clip(delta_mid, -clip, clip).reshape(-1, 1)
    return np.concatenate([delta_mid, queue_norm], axis=1).astype(np.float32)


class JointTimeGradGenerator:
    """
    TimeGrad trained on joint (Δmid, ΔQ) features.

    .fit(queue_norm, anchors, ts)  — builds joint features and trains TimeGrad
    .generate(context_queue, context_anchors, steps, n_paths)
             — returns (n_paths, steps, 2k) queue features
               and (n_paths, steps) mid-price trajectories

    Parameters
    ----------
    cfg        : BenchmarkConfig — reuses timegrad_* hyperparameters.
                 cfg.timegrad_input_dim must equal 2k+1.
    normalizer : Normalizer — used to inverse-transform queue outputs.
    device     : torch device string ("cpu" or "cuda").
    """

    def __init__(
        self,
        cfg:        BenchmarkConfig,
        normalizer: Normalizer,
        device:     Optional[str] = None,
    ):
        self.cfg            = cfg
        self.normalizer     = normalizer
        self.device         = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.delta_mid_clip = float(cfg.delta_mid_clip)
        self.model          = self._build_model()

    # ------------------------------------------------------------------
    def _build_model(self):
        dim        = self.cfg.joint_input_dim   # 2k+1: Δmid + 2k queue features
        noise_type = self.cfg.timegrad_noise_type
        loss_type  = self.cfg.timegrad_loss_type
        dropout    = self.cfg.timegrad_dropout

        if noise_type == "wavenet":
            model = TimeGradWaveNet(
                input_dim             = dim,
                output_dim            = dim,
                hidden_encoder_dim    = self.cfg.timegrad_hidden_enc,
                num_encoder_layers    = self.cfg.timegrad_num_enc_layers,
                emb_dim               = self.cfg.timegrad_emb_dim,
                K                     = self.cfg.timegrad_k,
                beta_start            = self.cfg.timegrad_beta_start,
                beta_end              = self.cfg.timegrad_beta_end,
                schedule              = self.cfg.timegrad_schedule,
                loss_type             = loss_type,
                dropout_rate          = dropout,
                cell_type             = self.cfg.timegrad_cell_type,
                residual_layers       = self.cfg.timegrad_wavenet_res_layers,
                residual_channels     = self.cfg.timegrad_wavenet_res_channels,
                dilation_cycle_length = self.cfg.timegrad_wavenet_dilation_cycle,
            )
        else:
            model = TimeGradMLP(
                input_dim           = dim,
                output_dim          = dim,
                hidden_encoder_dim  = self.cfg.timegrad_hidden_enc,
                hidden_noise_dim    = self.cfg.timegrad_hidden_noise,
                num_encoder_layers  = self.cfg.timegrad_num_enc_layers,
                emb_dim             = self.cfg.timegrad_emb_dim,
                K                   = self.cfg.timegrad_k,
                beta_start          = self.cfg.timegrad_beta_start,
                beta_end            = self.cfg.timegrad_beta_end,
                schedule            = self.cfg.timegrad_schedule,
                loss_type           = loss_type,
                dropout_rate        = dropout,
                cell_type           = self.cfg.timegrad_cell_type,
                time_embedding      = self.cfg.timegrad_time_embedding,
                use_cond_projection = self.cfg.timegrad_use_cond_projection,
                flat_window_size    = self.cfg.timegrad_flat_window_size,
            )

        ckpt = self.cfg.timegrad_checkpoint
        if os.path.exists(ckpt):
            state = torch.load(ckpt, map_location=self.device, weights_only=True)
            model.load_state_dict(state["model_state_dict"])
            print(f"[JointTimeGrad] Loaded checkpoint from {ckpt}")
        elif not self.cfg.timegrad_train:
            raise FileNotFoundError(
                f"[JointTimeGrad] No checkpoint at {ckpt} and timegrad_train=False. "
                "Either provide a trained checkpoint or set timegrad_train=True."
            )
        else:
            print(f"[JointTimeGrad] No checkpoint at {ckpt}. Will train from scratch.")

        model.eval()
        model.to(self.device)
        return model

    # ------------------------------------------------------------------
    def fit(
        self,
        queue_norm:    np.ndarray,        # (T, 2k)  normalised queue features
        anchors:       np.ndarray,        # (T,)     raw anchor (mid) prices
        timestamps:    np.ndarray,        # (T,)     unused but kept for API consistency
        optuna_trial=None,                # optuna.Trial or None; enables per-epoch pruning
    ) -> None:
        """Train on joint (Δmid, ΔQ) features."""
        if not self.cfg.timegrad_train:
            return

        from torch.utils.data import DataLoader, Subset

        X_joint = build_joint_features(anchors, queue_norm, self.delta_mid_clip)

        # Clip Δmid (col 0) and queue features (cols 1..) separately.
        X_joint[:, 0] = np.clip(X_joint[:, 0], -self.delta_mid_clip, self.delta_mid_clip)
        clip_lo = 0.0 if self.cfg.queue_repr == "unsigned" else -5.0
        X_joint[:, 1:] = np.clip(X_joint[:, 1:], clip_lo, 5.0)

        dataset = TimeSeriesDataset(
            targets           = X_joint,
            context_length    = self.cfg.timegrad_history_length,
            prediction_length = self.cfg.timegrad_prediction_length,
        )
        indices    = dataset.valid_indices
        split      = int(0.9 * len(indices))
        train_idx, val_idx = indices[:split], indices[split:]

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, train_idx),
            batch_size=self.cfg.timegrad_batch_size, shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, val_idx),
            batch_size=self.cfg.timegrad_batch_size, shuffle=False,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.timegrad_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.timegrad_epochs, eta_min=1e-5
        )
        self.model.to(self.device)

        if optuna_trial is None:
            train_model(
                model        = self.model,
                train_loader = train_loader,
                val_loader   = val_loader,
                optimizer    = optimizer,
                num_epochs   = self.cfg.timegrad_epochs,
                device       = torch.device(self.device),
                verbose      = True,
                save_dir     = self.cfg.timegrad_save_dir,
                max_batches  = self.cfg.timegrad_max_batches,
                scheduler    = scheduler,
            )
        else:
            # Epoch-by-epoch training with Optuna pruning callbacks.
            import optuna as _optuna
            os.makedirs(self.cfg.timegrad_save_dir, exist_ok=True)
            best_val = float("inf")
            device   = torch.device(self.device)

            for epoch in range(self.cfg.timegrad_epochs):
                train_epoch(self.model, train_loader, optimizer, device,
                            self.cfg.timegrad_max_batches)
                val_loss = validate(self.model, val_loader, device)
                scheduler.step()

                optuna_trial.report(val_loss, epoch)
                if optuna_trial.should_prune():
                    raise _optuna.TrialPruned()

                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({
                        "epoch":                epoch,
                        "model_state_dict":     self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss":             val_loss,
                    }, os.path.join(self.cfg.timegrad_save_dir, "best_model.pt"))

        best_ckpt = os.path.join(self.cfg.timegrad_save_dir, "best_model.pt")
        if os.path.exists(best_ckpt):
            state = torch.load(best_ckpt, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state["model_state_dict"])
            print(f"[JointTimeGrad] Loaded best checkpoint from {best_ckpt}")

        self.model.eval()

    # ------------------------------------------------------------------
    def generate(
        self,
        context_queue:    np.ndarray,   # (context_len, 2k) normalised queue
        context_anchors:  np.ndarray,   # (context_len,)   raw anchor prices
        steps:            int,
        n_paths:          int,
        seed:             int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate joint (queue, mid-price) paths.

        If ``seed`` is provided, PyTorch's global RNG (CPU + CUDA) is reseeded
        before sampling, making outputs reproducible on the same machine.

        Returns
        -------
        queue_paths : (n_paths, steps, 2k)  normalised queue features
        mid_paths   : (n_paths, steps)      reconstructed mid-prices (raw USDT)
        """
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        k = self.cfg.levels

        # Build joint context features
        ctx_joint = build_joint_features(context_anchors, context_queue, self.delta_mid_clip)

        ctx_joint[:, 0] = np.clip(ctx_joint[:, 0], -self.delta_mid_clip, self.delta_mid_clip)
        clip_lo = 0.0 if self.cfg.queue_repr == "unsigned" else -5.0
        ctx_joint[:, 1:] = np.clip(ctx_joint[:, 1:], clip_lo, 5.0)
        ctx_joint = ctx_joint.astype(np.float32)

        ctx_tensor = torch.tensor(
            np.tile(ctx_joint[np.newaxis], (n_paths, 1, 1)),
            dtype=torch.float32, device=self.device,
        )   # (n_paths, context_len, 2k+1)

        with torch.no_grad():
            raw_out = self.model.sample(ctx_tensor, prediction_steps=steps).cpu().numpy()
        # raw_out: (n_paths, steps, 2k+1)

        # Split into Δmid and queue features
        delta_mid_raw  = np.clip(raw_out[:, :, 0], -self.delta_mid_clip, self.delta_mid_clip)
        queue_norm_out = raw_out[:, :, 1:]     # (n_paths, steps, 2k)

        # Enforce sign convention for queue features
        if self.cfg.queue_repr != "unsigned":
            queue_norm_out[:, :, :k]   = queue_norm_out[:, :, :k].clip(max=0.0)
            queue_norm_out[:, :, k:]   = queue_norm_out[:, :, k:].clip(min=0.0)
        else:
            queue_norm_out = np.maximum(queue_norm_out, 0.0)

        # Reconstruct mid-price trajectories via cumsum of raw Δmid
        last_anchor = float(context_anchors[-1])
        mid_paths   = last_anchor + np.cumsum(delta_mid_raw, axis=1)  # (n_paths, steps)

        return queue_norm_out.astype(np.float32), mid_paths

    def generate_raw(
        self,
        context_queue:   np.ndarray,   # (context_len, 2k)
        context_anchors: np.ndarray,   # (context_len,)
        steps:           int,
        n_paths:         int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Like generate() but returns raw (inverse-transformed) queue sizes.

        Returns
        -------
        raw_queue_paths : (n_paths, steps, 2k)  raw absolute queue sizes (BTC)
        mid_paths       : (n_paths, steps)       mid-prices (USDT)
        """
        queue_norm, mid_paths = self.generate(
            context_queue, context_anchors, steps, n_paths
        )
        raw_queue = np.stack([
            self.normalizer.inverse_transform(queue_norm[p])
            for p in range(n_paths)
        ])  # (n_paths, steps, 2k)
        raw_queue = np.abs(raw_queue)   # ensure non-negative for cost function
        return raw_queue, mid_paths

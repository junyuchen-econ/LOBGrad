"""
timegrad.py
-----------
Wraps the existing TimeGrad diffusion model as a LOBGenerator.

Models only the 2k-dim queue representation — anchor prices are NOT modelled
here.  The initial anchor is returned flat (constant) across all generated
steps so that volume/imbalance metrics are unaffected by price drift.

Two queue content modes (controlled via cfg.queue_mode):
  "absolute"  — train on absolute normalised queue sizes directly.
  "delta"     — train on step-wise changes ΔQ (via normalizer.to_delta());
                generation converts delta output back to absolute via
                normalizer.from_delta(initial_raw).

Feature vector per step: queue features (2k), either absolute or delta.

Training: sliding-window TimeSeriesDataset on the 2k-dim series.
  - Queue features clipped to ±6 before training to prevent expm1 explosion
    during inverse transform at evaluation time.

Inference: model.sample() generates 2k-dim steps autoregressively.
  - Returns (n_paths, steps, 2k) absolute normalised features.
"""

from __future__ import annotations

import sys
import os
from typing import Optional

import numpy as np
import torch

from .base import LOBGenerator
from ..config import BenchmarkConfig
from ..data.normalizer import Normalizer

# Make sure src/ is importable regardless of working directory
_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from model import TimeGradMLP  # type: ignore
from train import train_model, TimeSeriesDataset  # type: ignore


class TimeGradGenerator(LOBGenerator):
    """
    Wraps the TimeGrad diffusion model (queue-only).

    .fit()      trains on the 2k-dim normalised queue features.
    .generate() samples 2k-dim output and returns (features, flat_anchors).
    """

    def __init__(
        self,
        cfg:        BenchmarkConfig,
        normalizer: Optional[Normalizer] = None,
        device:     Optional[str] = None,
    ):
        self.cfg        = cfg
        self.normalizer = normalizer  # needed for delta mode
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model      = self._load_model()

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def _load_model(self) -> TimeGradMLP:
        dim   = self.cfg.timegrad_input_dim   # 2k
        model = TimeGradMLP(
            input_dim          = dim,
            output_dim         = dim,
            hidden_encoder_dim = self.cfg.timegrad_hidden_enc,
            hidden_noise_dim   = self.cfg.timegrad_hidden_noise,
            num_encoder_layers = self.cfg.timegrad_num_enc_layers,
            emb_dim            = self.cfg.timegrad_emb_dim,
            K                  = self.cfg.timegrad_k,
            beta_start         = self.cfg.timegrad_beta_start,
            beta_end           = self.cfg.timegrad_beta_end,
            schedule           = self.cfg.timegrad_schedule,
            cell_type          = self.cfg.timegrad_cell_type,
        )
        ckpt_path = self.cfg.timegrad_checkpoint
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state["model_state_dict"])
            print(f"[TimeGrad] Loaded checkpoint from {ckpt_path}")
        elif not self.cfg.timegrad_train:
            print(f"[TimeGrad] WARNING: checkpoint not found at {ckpt_path}. Using random weights.")
        else:
            print(f"[TimeGrad] No checkpoint found at {ckpt_path}. Training from scratch.")
        model.eval()
        model.to(self.device)
        return model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        snapshots:  np.ndarray,   # (T, 2k) absolute normalised
        timestamps: np.ndarray,   # (T,)
    ) -> None:
        """Train TimeGrad on the 2k-dim queue features (absolute or delta)."""
        if not self.cfg.timegrad_train:
            return

        from torch.utils.data import DataLoader, Subset

        # Clip queue features: p99.9 of training data is ±3.6, max is ±5.5.
        # Unsigned features are log(Q+1) ≥ 0, so lower bound is 0 not -5.
        clip_lo = 0.0 if self.cfg.queue_repr == "unsigned" else -5.0
        X = np.clip(snapshots.astype(np.float32), clip_lo, 5.0)

        # Delta mode: convert absolute normalised features → step-wise changes.
        # to_delta() inverse-transforms to raw sizes, diffs, then applies signed log.
        if self.cfg.queue_mode == "delta":
            if self.normalizer is None:
                raise ValueError("normalizer must be provided when queue_mode='delta'")
            X = self.normalizer.to_delta(X)

        dataset = TimeSeriesDataset(
            targets           = X,
            context_length    = self.cfg.timegrad_history_length,
            prediction_length = self.cfg.timegrad_prediction_length,
        )
        indices    = dataset.valid_indices
        split      = int(0.9 * len(indices))
        train_idx, val_idx = indices[:split], indices[split:]

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size = self.cfg.timegrad_batch_size,
            shuffle    = True,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size = self.cfg.timegrad_batch_size,
            shuffle    = False,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.timegrad_lr)
        self.model.to(self.device)

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
        )

        best_ckpt = os.path.join(self.cfg.timegrad_save_dir, "best_model.pt")
        if os.path.exists(best_ckpt):
            state = torch.load(best_ckpt, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state["model_state_dict"])
            print(f"[TimeGrad] Loaded best checkpoint from {best_ckpt}")

        self.model.eval()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        context: np.ndarray,            # (context_len, 2k) absolute normalised
        steps:   int,
        n_paths: int,
        rng:     Optional[np.random.Generator] = None,
        seed:    Optional[int] = None,
    ) -> np.ndarray:
        """
        Returns
        -------
        features : (n_paths, steps, 2k)  normalised queue features

        If ``seed`` is provided, PyTorch's global RNG (CPU + CUDA) is reseeded
        before sampling, making outputs reproducible on the same machine.
        """
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        k = self.cfg.levels

        lo = 0.0 if self.cfg.queue_repr == "unsigned" else -5.0
        ctx_abs = np.clip(context.astype(np.float32), lo, 5.0)

        if self.cfg.queue_mode == "delta":
            if self.normalizer is None:
                raise ValueError("normalizer must be provided when queue_mode='delta'")
            ctx_input = self.normalizer.to_delta(ctx_abs)
            # Last raw absolute snapshot — used to reconstruct absolute from deltas.
            initial_raw = self.normalizer.inverse_transform(ctx_abs[-1])
        else:
            ctx_input  = ctx_abs
            initial_raw = None  # unused in absolute mode

        ctx_tensor = torch.tensor(
            np.tile(ctx_input[np.newaxis], (n_paths, 1, 1)),
            dtype=torch.float32,
            device=self.device,
        )  # (n_paths, context_len, 2k)

        with torch.no_grad():
            raw_output = self.model.sample(
                ctx_tensor, prediction_steps=steps
            ).cpu().numpy()
        # raw_output: (n_paths, steps, 2k)

        # Enforce sign convention (bids ≤ 0, asks ≥ 0) when using signed representation.
        if self.cfg.queue_repr != "unsigned":
            raw_output[:, :, :k]    = raw_output[:, :, :k].clip(max=0.0)
            raw_output[:, :, k:2*k] = raw_output[:, :, k:2*k].clip(min=0.0)

        if self.cfg.queue_mode == "delta":
            # Convert each path's delta features back to absolute normalised.
            features_norm = np.stack([
                self.normalizer.from_delta(raw_output[p], initial_raw)
                for p in range(n_paths)
            ])
        else:
            features_norm = raw_output

        # In unsigned mode all features are log(Q+1) ≥ 0; clamp out any small negatives
        # that the diffusion sampler can produce near zero.
        if self.cfg.queue_repr == "unsigned":
            features_norm = np.maximum(features_norm, 0.0)

        return features_norm.astype(np.float32)

    @property
    def name(self) -> str:
        return "TimeGrad"

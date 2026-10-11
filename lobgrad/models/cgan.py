"""
cgan.py
-------
Conditional Wasserstein GAN with Gradient Penalty (WGAN-GP) for LOB generation.

Architecture
------------
Generator G(z, context):
  1. GRU encoder over context     → h : (batch, hidden_dim)
  2. c = concat(z, h)             → condition vector
  3. Linear(c) → LSTM init state
  4. AR LSTM decoder for M steps:
       input_t = concat(prev_output, c)
       LSTM cell → hidden state
       Separate heads → delta_mid (unconstrained) + queue (Softplus constraints)

     Output layout (2k+1 dims, same as TimeGrad joint model):
       index 0      : delta_mid  — unconstrained price change
       signed mode  — bids (idx 1..k)    : -Softplus(raw)  (≤ 0)
                      asks (idx k+1..2k) :  Softplus(raw)  (≥ 0)
       unsigned mode — queue (idx 1..2k) :  Softplus(raw)  (≥ 0)

Discriminator D(context, future):
  concat([context, future], dim=1) → GRU → Linear → scalar  [no sigmoid, WGAN]

Training: WGAN-GP
  n_critic discriminator steps per generator step
  gradient penalty weight lambda_gp
"""

from __future__ import annotations

from typing import Optional

import time

import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .base import LOBGenerator
from ..config import BenchmarkConfig
from ..execution.simulate import build_joint_features


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class _Generator(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int,
                 n_encoder_layers: int, n_decoder_layers: int,
                 queue_repr: str = "signed"):
        super().__init__()
        self.input_dim    = input_dim   # 2k+1
        self.latent_dim   = latent_dim
        self.hidden_dim   = hidden_dim
        self.queue_repr   = queue_repr
        self.k            = (input_dim - 1) // 2  # levels per side

        # GRU context encoder — processes joint (2k+1) context
        self.encoder = nn.GRU(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = n_encoder_layers,
            batch_first = True,
        )

        cond_dim = latent_dim + hidden_dim
        self.n_decoder_layers = n_decoder_layers

        # Initialise LSTM hidden/cell state from condition vector
        self.init_h = nn.Linear(cond_dim, hidden_dim)
        self.init_c = nn.Linear(cond_dim, hidden_dim)

        # AR LSTM decoder — input is prev joint output (2k+1) + condition
        self.decoder = nn.LSTM(
            input_size  = input_dim + cond_dim,
            hidden_size = hidden_dim,
            num_layers  = n_decoder_layers,
            batch_first = True,
        )

        # Separate heads — same format as TimeGrad joint model
        self.price_head = nn.Linear(hidden_dim, 1)             # delta_mid, unconstrained
        self.queue_head = nn.Linear(hidden_dim, 2 * self.k)    # queue features, constrained

    def forward(
        self,
        z: torch.Tensor,          # (batch, latent_dim)
        context: torch.Tensor,    # (batch, context_len, 2k+1)
        steps: int,
        k: int,                   # levels per side
    ) -> torch.Tensor:            # (batch, steps, 2k+1)
        batch = z.size(0)

        _, h_n = self.encoder(context)
        h_ctx  = h_n[-1]                        # (batch, hidden_dim)

        c = torch.cat([z, h_ctx], dim=-1)       # (batch, cond_dim)

        h0 = torch.tanh(self.init_h(c)).unsqueeze(0).repeat(self.n_decoder_layers, 1, 1)
        c0 = torch.tanh(self.init_c(c)).unsqueeze(0).repeat(self.n_decoder_layers, 1, 1)
        lstm_state = (h0, c0)

        prev = context[:, -1, :]                # (batch, 2k+1)

        outputs = []
        for _ in range(steps):
            inp = torch.cat([prev, c], dim=-1).unsqueeze(1)
            out, lstm_state = self.decoder(inp, lstm_state)
            h = out.squeeze(1)
            snap = self._apply_constraints(h, k)
            outputs.append(snap)
            prev = snap

        return torch.stack(outputs, dim=1)      # (batch, steps, 2k+1)

    def _apply_constraints(self, h: torch.Tensor, k: int) -> torch.Tensor:
        delta_mid = self.price_head(h)                  # (batch, 1)
        queue_raw = self.queue_head(h)                  # (batch, 2k)
        if self.queue_repr == "unsigned":
            queue = F.softplus(queue_raw)
        else:
            bids  = -F.softplus(queue_raw[:, :k])
            asks  =  F.softplus(queue_raw[:, k:])
            queue = torch.cat([bids, asks], dim=-1)
        return torch.cat([delta_mid, queue], dim=-1)    # (batch, 2k+1)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class _Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.rnn = nn.GRU(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = 2,
            batch_first = True,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        context: torch.Tensor,
        future:  torch.Tensor,
    ) -> torch.Tensor:
        seq = torch.cat([context, future], dim=1)
        _, h_n = self.rnn(seq)
        return self.head(h_n[-1])


# ---------------------------------------------------------------------------
# Gradient penalty
# ---------------------------------------------------------------------------

def _gradient_penalty(
    D: _Discriminator,
    context: torch.Tensor,
    real: torch.Tensor,
    fake: torch.Tensor,
    lambda_gp: float,
    device: str,
) -> torch.Tensor:
    batch = real.size(0)
    alpha = torch.rand(batch, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    with torch.backends.cudnn.flags(enabled=False):
        d_interp = D(context, interpolated)
    grad = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_norm = grad.reshape(batch, -1).norm(2, dim=1)
    return lambda_gp * ((grad_norm - 1) ** 2).mean()


# ---------------------------------------------------------------------------
# LOBGenerator wrapper
# ---------------------------------------------------------------------------

class CGANGenerator(LOBGenerator):
    """
    Conditional AR-LSTM WGAN-GP for joint (delta_mid, queue) LOB generation.
    Generates (2k+1)-dim sequences: index 0 = delta_mid, indices 1..2k = queue.
    Same joint feature format as JointTimeGradGenerator.
    """

    def __init__(self, cfg: BenchmarkConfig, device: Optional[str] = None):
        self.cfg    = cfg
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            print("Using MPS device")
            self.device = "mps"
        else:
            print("Using CPU device")
            self.device = "cpu"
        dim = cfg.joint_input_dim   # 2k+1

        self.G = _Generator(
            input_dim       = dim,
            latent_dim      = cfg.cgan_latent_dim,
            hidden_dim      = cfg.cgan_hidden_dim,
            n_encoder_layers= cfg.cgan_n_encoder_layers,
            n_decoder_layers= cfg.cgan_n_decoder_layers,
            queue_repr      = cfg.queue_repr,
        ).to(self.device)

        self.D = _Discriminator(
            input_dim  = dim,
            hidden_dim = cfg.cgan_hidden_dim,
        ).to(self.device)

        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=cfg.cgan_lr_g, betas=(0.5, 0.9))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=cfg.cgan_lr_d, betas=(0.5, 0.9))

    def fit(
        self,
        snapshots:  np.ndarray,
        timestamps: np.ndarray,
        anchors:    np.ndarray | None = None,
    ) -> None:
        """
        snapshots : (T, 2k) normalised queue features
        anchors   : (T,) raw anchor prices — builds joint (2k+1) features
        """
        cfg = self.cfg
        torch.multiprocessing.set_sharing_strategy("file_system")

        clip_lo = 0.0 if cfg.queue_repr == "unsigned" else -5.0
        snapshots = np.clip(snapshots.astype(np.float32), clip_lo, 5.0)

        if anchors is not None:
            snapshots = build_joint_features(anchors, snapshots, cfg.delta_mid_clip)
        else:
            snapshots = np.concatenate(
                [np.zeros((len(snapshots), 1), dtype=np.float32), snapshots], axis=1
            )
        # Re-clip queue cols only (col 0 is raw Δmid, already clipped in build_joint_features)
        snapshots[:, 1:] = np.clip(snapshots[:, 1:], clip_lo, 5.0)

        N, M, stride = cfg.context_len, cfg.horizon, cfg.cgan_stride
        T = len(snapshots)
        ctx_list, fut_list = [], []
        for start in range(0, T - N - M + 1, stride):
            ctx_list.append(snapshots[start : start + N])
            fut_list.append(snapshots[start + N : start + N + M])
        contexts = np.stack(ctx_list)
        futures  = np.stack(fut_list)

        X_ctx = torch.tensor(contexts, dtype=torch.float32)
        X_fut = torch.tensor(futures,  dtype=torch.float32)
        dataset = TensorDataset(X_ctx, X_fut)
        pin = self.device == "cuda"
        loader = DataLoader(
            dataset, batch_size=cfg.cgan_batch_size, shuffle=True, drop_last=True,
            num_workers=0, pin_memory=pin,
        )

        k = cfg.levels
        print(f"[CGAN] Device: {self.device}")
        n_params_G = sum(p.numel() for p in self.G.parameters())
        n_params_D = sum(p.numel() for p in self.D.parameters())
        n_batches  = min(len(loader), cfg.cgan_max_batches) if cfg.cgan_max_batches else len(loader)
        print(f"[CGAN] Generator params: {n_params_G:,} | Discriminator params: {n_params_D:,}")
        print(f"[CGAN] Training on {len(dataset)} windows | "
              f"{len(loader)} batches/epoch (capped at {n_batches}) | "
              f"{cfg.cgan_epochs} epochs | n_critic={cfg.cgan_n_critic}")

        use_amp = self.device == "cuda"
        amp_dev = "cuda" if use_amp else "cpu"
        scaler  = torch.amp.GradScaler(amp_dev, enabled=use_amp)

        train_start = time.time()
        epoch_bar = tqdm(range(1, cfg.cgan_epochs + 1), desc="[CGAN] Epoch", unit="epoch")
        for epoch in epoch_bar:
            g_losses, d_losses = [], []
            epoch_start = time.time()

            batch_bar = tqdm(
                enumerate(loader),
                total=n_batches,
                desc=f"  Epoch {epoch:>3}/{cfg.cgan_epochs}",
                unit="batch",
                leave=False,
            )
            for batch_idx, (ctx_batch, fut_batch) in batch_bar:
                if cfg.cgan_max_batches is not None and batch_idx >= cfg.cgan_max_batches:
                    break
                ctx_batch = ctx_batch.to(self.device, non_blocking=pin)
                fut_batch = fut_batch.to(self.device, non_blocking=pin)
                batch_sz  = ctx_batch.size(0)

                for _ in range(cfg.cgan_n_critic):
                    z = torch.randn(batch_sz, cfg.cgan_latent_dim, device=self.device)
                    with torch.no_grad():
                        fake = self.G(z, ctx_batch, cfg.horizon, k)
                    with torch.amp.autocast(amp_dev, enabled=use_amp):
                        d_real = self.D(ctx_batch, fut_batch)
                        d_fake = self.D(ctx_batch, fake)
                        gp     = _gradient_penalty(
                            self.D, ctx_batch, fut_batch, fake,
                            cfg.cgan_lambda_gp, self.device
                        )
                        loss_D = d_fake.mean() - d_real.mean() + gp
                    self.opt_D.zero_grad()
                    scaler.scale(loss_D).backward()
                    scaler.step(self.opt_D)
                    scaler.update()
                    d_losses.append(loss_D.item())

                z = torch.randn(batch_sz, cfg.cgan_latent_dim, device=self.device)
                with torch.amp.autocast(amp_dev, enabled=use_amp):
                    fake   = self.G(z, ctx_batch, cfg.horizon, k)
                    loss_G = -self.D(ctx_batch, fake).mean()
                self.opt_G.zero_grad()
                scaler.scale(loss_G).backward()
                scaler.step(self.opt_G)
                scaler.update()
                g_losses.append(loss_G.item())
                batch_bar.set_postfix(D=f"{np.mean(d_losses):.4f}", G=f"{np.mean(g_losses):.4f}")

            epoch_elapsed = time.time() - epoch_start
            mean_d, mean_g = np.mean(d_losses), np.mean(g_losses)
            epoch_bar.set_postfix(D=f"{mean_d:.4f}", G=f"{mean_g:.4f}", s=f"{epoch_elapsed:.1f}s")
            tqdm.write(
                f"[CGAN] Epoch {epoch:>3}/{cfg.cgan_epochs} | "
                f"D: {mean_d:+.4f}  G: {mean_g:+.4f} | "
                f"{epoch_elapsed:.1f}s  (total {time.time()-train_start:.0f}s)"
            )

    def generate(
        self,
        context:         np.ndarray,
        steps:           int,
        n_paths:         int,
        context_anchors: np.ndarray | None = None,
        rng:             np.random.Generator | None = None,
        seed:            int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        queue_paths : (n_paths, steps, 2k)
        price_paths : (n_paths, steps)

        If ``seed`` is provided, PyTorch's global RNG (CPU + CUDA) is reseeded
        before sampling the latent ``z``, making outputs reproducible on the
        same machine.
        """
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.G.eval()
        k = self.cfg.levels

        if context_anchors is not None:
            ctx_joint = build_joint_features(context_anchors, context, self.cfg.delta_mid_clip)
        else:
            ctx_joint = np.concatenate(
                [np.zeros((len(context), 1), dtype=np.float32), context], axis=1
            )

        clip_lo = 0.0 if self.cfg.queue_repr == "unsigned" else -5.0
        ctx_joint = np.clip(ctx_joint, clip_lo, 5.0).astype(np.float32)

        ctx = torch.tensor(
            np.tile(ctx_joint[np.newaxis], (n_paths, 1, 1)),
            dtype=torch.float32,
            device=self.device,
        )
        z = torch.randn(n_paths, self.cfg.cgan_latent_dim, device=self.device)

        with torch.no_grad():
            out = self.G(z, ctx, steps, k)

        self.G.train()
        out = out.cpu().numpy().astype(np.float32)   # (n_paths, steps, 2k+1)

        delta_mid_raw = np.clip(out[:, :, 0], -self.cfg.delta_mid_clip, self.cfg.delta_mid_clip)
        queue         = out[:, :, 1:]

        if self.cfg.queue_repr == "unsigned":
            queue = np.maximum(queue, 0.0)
        else:
            queue[:, :, :k] = np.minimum(queue[:, :, :k], 0.0)
            queue[:, :, k:] = np.maximum(queue[:, :, k:], 0.0)

        last_anchor = float(context_anchors[-1]) if context_anchors is not None else 0.0
        price_paths = last_anchor + np.cumsum(delta_mid_raw, axis=1)

        return queue, price_paths

    def save(self, path: str) -> None:
        torch.save({"G": self.G.state_dict(), "D": self.D.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.G.load_state_dict(ckpt["G"])
        self.D.load_state_dict(ckpt["D"])

    @property
    def name(self) -> str:
        return "CGAN"

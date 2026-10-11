"""
paint.py
--------
"Painting the Market" (Backhouse et al., arXiv:2509.05107, 2025) reimplemented
as a benchmark generator for the LOBGrad pipeline.

Method (faithful): a window of LOB states is recast as a 2-D image
(features x time); the past columns are the *unnoised known* region and the
future columns are generated **in one shot** by a 2-D convolutional DDPM doing
conditional inpainting (the network sees noised image + unnoised-history +
binary mask as input channels and predicts the injected noise). No
autoregression.

Scaled-faithful adaptation to this repo (see plan / paper differences):
  * Image content = the joint (2k+1) feature `build_joint_features` produces —
    row 0 = Δmid, rows 1..2k = signed-log queues — transposed to (F x T), per-row
    standardized, F padded to a multiple of 8. This replaces the paper's
    2-channel (price,size) raw-L2 image because our representation is a fixed
    δ-grid (price is implicit; the mid move is the Δmid row).
  * Rectangular image (F_pad x (context+horizon)) instead of square 128x128
    padding — far cheaper and natural for F≈11.
  * Compact pure-torch U-Net (no `diffusers` dependency); DDPM ε-prediction,
    T_train diffusion steps, DDIM sampling.

Interface matches `JointTimeGradGenerator`:
    .fit(queue_norm (T,2k), anchors (T,), timestamps)
    .generate(context_queue (C,2k), context_anchors (C,), steps, n_paths, seed)
        -> (queue (n_paths,steps,2k), mid (n_paths,steps))
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ..config import BenchmarkConfig
from ..data.normalizer import Normalizer
from ..execution.simulate import build_joint_features


# ─────────────────────────── U-Net building blocks ──────────────────────────
def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def _gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, ch), ch)


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_dim: int):
        super().__init__()
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb = nn.Linear(temb_dim, out_ch)
        self.norm2 = _gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class _Attn(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.norm = _gn(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.heads = heads

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]           # (b, heads, cd, hw)
        attn = torch.softmax((q.transpose(-1, -2) @ k) / math.sqrt(q.shape[-2]), dim=-1)
        out = (v @ attn.transpose(-1, -2)).reshape(b, c, h, w)
        return x + self.proj(out)


class _UNet2D(nn.Module):
    """Compact U-Net: len(mults) resolution levels, 2 res-blocks each, attention
    at the bottleneck. in_ch=3 (noised, history, mask) -> out_ch=1 (predicted ε)."""

    def __init__(self, in_ch=3, out_ch=1, base=64, mults=(1, 2, 4), n_res=2):
        super().__init__()
        temb_dim = base * 4
        self.temb_mlp = nn.Sequential(nn.Linear(base, temb_dim), nn.SiLU(),
                                      nn.Linear(temb_dim, temb_dim))
        self.base = base
        chs = [base * m for m in mults]
        self.init = nn.Conv2d(in_ch, base, 3, padding=1)

        # Down path
        self.downs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev = base
        self.skip_chs = [base]
        for i, ch in enumerate(chs):
            blocks = nn.ModuleList()
            for _ in range(n_res):
                blocks.append(_ResBlock(prev, ch, temb_dim)); prev = ch
                self.skip_chs.append(ch)
            self.downs.append(blocks)
            if i < len(chs) - 1:
                self.downsamples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                self.skip_chs.append(ch)
            else:
                self.downsamples.append(None)

        # Middle
        self.mid1 = _ResBlock(prev, prev, temb_dim)
        self.mid_attn = _Attn(prev)
        self.mid2 = _ResBlock(prev, prev, temb_dim)

        # Up path (mirror)
        self.ups = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, ch in reversed(list(enumerate(chs))):
            if i < len(chs) - 1:
                self.upsamples.append(nn.Conv2d(prev, prev, 3, padding=1))  # after nearest-upsample
            else:
                self.upsamples.append(None)
            blocks = nn.ModuleList()
            for _ in range(n_res + 1):
                blocks.append(_ResBlock(prev + self.skip_chs.pop(), ch, temb_dim)); prev = ch
            self.ups.append(blocks)

        self.out = nn.Sequential(_gn(prev), nn.SiLU(), nn.Conv2d(prev, out_ch, 3, padding=1))

    def forward(self, x, t):
        temb = self.temb_mlp(_sinusoidal_embedding(t, self.base))
        h = self.init(x)
        skips = [h]
        for blocks, ds in zip(self.downs, self.downsamples):
            for blk in blocks:
                h = blk(h, temb); skips.append(h)
            if ds is not None:
                h = ds(h); skips.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)
        for blocks, us in zip(self.ups, self.upsamples):
            if us is not None:
                h = us(F.interpolate(h, scale_factor=2, mode="nearest"))
            for blk in blocks:
                h = blk(torch.cat([h, skips.pop()], dim=1), temb)
        return self.out(h)


# ─────────────────────────── diffusion schedule ─────────────────────────────
class _Schedule:
    def __init__(self, T: int, device, beta0=1e-4, beta1=2e-2):
        self.T = T
        betas = torch.linspace(beta0, beta1, T, device=device)
        self.alphas = 1.0 - betas
        self.abar = torch.cumprod(self.alphas, dim=0)


# ─────────────────────────── generator ──────────────────────────────────────
class PaintDiffusionGenerator:
    """Image-inpainting DDPM (Painting the Market) over the joint δ-grid feature."""

    def __init__(
        self,
        cfg: BenchmarkConfig,
        normalizer: Normalizer,
        device: Optional[str] = None,
        t_train: int = 100,
        sample_steps: int = 50,
        base_channels: int = 64,
        mults: Tuple[int, ...] = (1, 2, 4),
        epochs: int = 25,
        lr: float = 2e-4,
        batch_size: int = 64,
        stride: int = 5,
        max_train_windows: int = 6000,
        save_dir: str = "checkpoints/paint",
        train: bool = True,
    ):
        self.cfg = cfg
        self.normalizer = normalizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.k = cfg.levels
        self.F_real = 2 * cfg.levels + 1                 # Δmid + 2k queues
        self.F_pad = int(math.ceil(self.F_real / 8) * 8)  # e.g. 11 -> 16
        self.context_len = cfg.timegrad_history_length
        self.delta_mid_clip = float(cfg.delta_mid_clip)
        self.t_train = t_train
        self.sample_steps = sample_steps
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.stride = stride
        self.max_train_windows = max_train_windows
        self.save_dir = save_dir
        self.train_flag = train

        self.model = _UNet2D(in_ch=3, out_ch=1, base=base_channels, mults=mults).to(self.device)
        self.sched = _Schedule(t_train, self.device)
        self.row_mean = None   # (F_real,)
        self.row_std = None

    # ---- helpers ----
    def _content(self, queue_norm, anchors):
        """(T, F_real) joint content: row0 Δmid, rows1.. signed-log queues."""
        return build_joint_features(anchors, queue_norm, self.delta_mid_clip)  # (T, F_real)

    def _standardize(self, content):
        return (content - self.row_mean) / self.row_std

    def _to_image(self, content_std):
        """(win, F_real) standardized -> (1, F_pad, win) image (rows=features)."""
        win = content_std.shape[0]
        img = np.zeros((self.F_pad, win), dtype=np.float32)
        img[: self.F_real] = content_std.T
        return img[None]  # (1, F_pad, win)

    # ---- training ----
    def fit(self, queue_norm: np.ndarray, anchors: np.ndarray, timestamps: np.ndarray) -> None:
        if not self.train_flag:
            return
        content = self._content(queue_norm, anchors).astype(np.float32)  # (T, F_real)
        self.row_mean = content.mean(0, keepdims=True)
        self.row_std = content.std(0, keepdims=True) + 1e-6
        content_std = self._standardize(content)                        # (T, F_real)

        win = self.context_len + self.cfg.horizon
        W = win
        assert W % 4 == 0, f"window {W} must be divisible by 4"
        starts = list(range(0, len(content_std) - win + 1, self.stride))
        if len(starts) > self.max_train_windows:
            rng = np.random.default_rng(0)
            starts = sorted(rng.choice(starts, self.max_train_windows, replace=False).tolist())
        imgs = np.stack([self._to_image(content_std[s:s + win]) for s in starts])  # (N,1,F_pad,W)
        imgs = torch.tensor(imgs, dtype=torch.float32)

        # future-region mask: rows 0..F_real-1, cols context..W-1
        mask = torch.zeros(1, self.F_pad, W)
        mask[:, : self.F_real, self.context_len:] = 1.0
        mask = mask.to(self.device)

        ds = torch.utils.data.TensorDataset(imgs)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        os.makedirs(self.save_dir, exist_ok=True)

        self.model.train()
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[Paint] training: {len(starts)} windows, img={self.F_pad}x{W}, "
              f"params={n_params:,}, T_train={self.t_train}, epochs={self.epochs}, device={self.device}")
        for ep in range(self.epochs):
            tot, nb = 0.0, 0
            for (x0,) in loader:
                x0 = x0.to(self.device)                          # (B,1,F_pad,W)
                b = x0.shape[0]
                t = torch.randint(0, self.t_train, (b,), device=self.device)
                noise = torch.randn_like(x0)
                ab = self.sched.abar[t][:, None, None, None]
                x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
                hist = x0 * (1 - mask)                            # unnoised history, future zeroed
                m = mask.expand(b, -1, -1, -1)
                inp = torch.cat([x_t, hist, m], dim=1)           # (B,3,F_pad,W)
                pred = self.model(inp, t)
                loss = ((pred - noise) ** 2 * m).sum() / (m.sum() + 1e-8)  # masked ε-MSE (future)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); nb += 1
            print(f"[Paint]   epoch {ep+1}/{self.epochs}  loss={tot/max(nb,1):.4f}")

        torch.save({"model": self.model.state_dict(),
                    "row_mean": self.row_mean, "row_std": self.row_std,
                    "F_real": self.F_real, "F_pad": self.F_pad}, os.path.join(self.save_dir, "paint.pt"))
        self.model.eval()

    def load(self, path: str) -> None:
        st = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(st["model"])
        self.row_mean, self.row_std = st["row_mean"], st["row_std"]
        self.model.eval().to(self.device)
        print(f"[Paint] loaded checkpoint from {path}")

    # ---- sampling (DDIM inpainting) ----
    @torch.no_grad()
    def generate(self, context_queue, context_anchors, steps, n_paths, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        assert steps == self.cfg.horizon, "PaintDiffusion generates a fixed horizon block"
        W = self.context_len + self.cfg.horizon

        # standardized history content -> image; future cols = 0 (unknown)
        ctx_content = self._content(context_queue, context_anchors).astype(np.float32)  # (C, F_real)
        ctx_std = self._standardize(ctx_content)
        x0_hist = np.zeros((1, self.F_pad, W), dtype=np.float32)
        x0_hist[0, : self.F_real, : self.context_len] = ctx_std[-self.context_len:].T
        x0_hist = torch.tensor(x0_hist, device=self.device).expand(n_paths, -1, -1, -1).contiguous()

        mask = torch.zeros(1, self.F_pad, W, device=self.device)
        mask[:, : self.F_real, self.context_len:] = 1.0
        m = mask.expand(n_paths, -1, -1, -1)

        abar = self.sched.abar
        ts = torch.linspace(self.t_train - 1, 0, self.sample_steps, device=self.device).long()
        x_fut = torch.randn(n_paths, 1, self.F_pad, W, device=self.device) * m   # future noise

        x0_pred = None
        for i, t in enumerate(ts):
            ab_t = abar[t]
            # re-noise known history to level t (RePaint), keep evolving future
            hist_noised = torch.sqrt(ab_t) * x0_hist + torch.sqrt(1 - ab_t) * torch.randn_like(x0_hist)
            x_in = m * x_fut + (1 - m) * hist_noised
            hist_cond = x0_hist * (1 - m)
            tb = torch.full((n_paths,), int(t), device=self.device, dtype=torch.long)
            eps = self.model(torch.cat([x_in, hist_cond, m], dim=1), tb)
            x0_pred = (x_in - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)
            if i < len(ts) - 1:
                ab_prev = abar[ts[i + 1]]
                x_prev = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1 - ab_prev) * eps  # DDIM eta=0
                x_fut = m * x_prev

        # extract future region, de-standardize
        fut = x0_pred[:, 0, : self.F_real, self.context_len:]          # (n_paths, F_real, horizon)
        fut = fut.permute(0, 2, 1).cpu().numpy()                        # (n_paths, horizon, F_real)
        rm = self.row_mean.reshape(-1)[: self.F_real]
        rs = self.row_std.reshape(-1)[: self.F_real]
        content = fut * rs + rm                                         # de-standardize

        delta_mid = np.clip(content[:, :, 0], -self.delta_mid_clip, self.delta_mid_clip)
        queue = content[:, :, 1:]                                       # (n_paths, horizon, 2k)
        # signed convention + clip (mirror JointTimeGradGenerator.generate)
        queue[:, :, :self.k] = np.clip(queue[:, :, :self.k], -5.0, 0.0)
        queue[:, :, self.k:] = np.clip(queue[:, :, self.k:], 0.0, 5.0)

        last_anchor = float(context_anchors[-1])
        mid = last_anchor + np.cumsum(delta_mid, axis=1)               # (n_paths, horizon)
        return queue.astype(np.float32), mid.astype(np.float32)

    # ---- batched sampling over many windows (fast path for eval) ----
    @torch.no_grad()
    def generate_many(self, ctx_queues, ctx_anchors, steps, n_paths, chunk=512, seed=None):
        """
        ctx_queues  : (W, C, 2k) ; ctx_anchors : (W, C)
        Returns queue (W, n_paths, steps, 2k), mid (W, n_paths, steps).
        Runs the DDIM inpainting reverse process over all W*n_paths images at
        once, in minibatches of `chunk`, to amortize the U-Net cost.
        """
        assert steps == self.cfg.horizon
        Wt = self.context_len + self.cfg.horizon
        Wn = len(ctx_queues)
        dev = self.device

        hist = np.zeros((Wn, 1, self.F_pad, Wt), dtype=np.float32)
        last_anchor = np.empty(Wn, dtype=np.float32)
        for i in range(Wn):
            cs = self._standardize(self._content(ctx_queues[i], ctx_anchors[i]).astype(np.float32))
            hist[i, 0, : self.F_real, : self.context_len] = cs[-self.context_len:].T
            last_anchor[i] = float(ctx_anchors[i][-1])
        x0_hist = torch.tensor(hist, device=dev).repeat_interleave(n_paths, dim=0)  # (Wn*P,1,F,Wt)
        total = x0_hist.shape[0]

        m = torch.zeros(1, 1, self.F_pad, Wt, device=dev)
        m[:, :, : self.F_real, self.context_len:] = 1.0

        if seed is not None:
            torch.manual_seed(seed)
        x_fut = torch.randn(total, 1, self.F_pad, Wt, device=dev) * m

        abar = self.sched.abar
        ts = torch.linspace(self.t_train - 1, 0, self.sample_steps, device=dev).long()
        for i, t in enumerate(ts):
            ab_t = abar[t]
            ab_prev = abar[ts[i + 1]] if i < len(ts) - 1 else None
            for c0 in range(0, total, chunk):
                sl = slice(c0, min(c0 + chunk, total))
                xh = x0_hist[sl]
                b = xh.shape[0]
                hn = torch.sqrt(ab_t) * xh + torch.sqrt(1 - ab_t) * torch.randn_like(xh)
                x_in = m * x_fut[sl] + (1 - m) * hn
                inp = torch.cat([x_in, xh * (1 - m), m.expand(b, -1, -1, -1)], dim=1)
                tb = torch.full((b,), int(t), device=dev, dtype=torch.long)
                eps = self.model(inp, tb)
                x0p = (x_in - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)
                if ab_prev is not None:
                    x_fut[sl] = m * (torch.sqrt(ab_prev) * x0p + torch.sqrt(1 - ab_prev) * eps)
                else:
                    x_fut[sl] = m * x0p     # final: keep x0 prediction

        fut = x_fut[:, 0, : self.F_real, self.context_len:].permute(0, 2, 1).cpu().numpy()
        fut = fut.reshape(Wn, n_paths, self.cfg.horizon, self.F_real)
        rm = self.row_mean.reshape(-1)[: self.F_real]
        rs = self.row_std.reshape(-1)[: self.F_real]
        content = fut * rs + rm
        delta_mid = np.clip(content[..., 0], -self.delta_mid_clip, self.delta_mid_clip)
        queue = content[..., 1:]
        queue[..., :self.k] = np.clip(queue[..., :self.k], -5.0, 0.0)
        queue[..., self.k:] = np.clip(queue[..., self.k:], 0.0, 5.0)
        mid = last_anchor[:, None, None] + np.cumsum(delta_mid, axis=2)
        return queue.astype(np.float32), mid.astype(np.float32)

    @property
    def name(self) -> str:
        return "PTM"

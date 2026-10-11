"""
supervised.py
-------------
Three published LOB deep-learning models — **DeepLOB** (arXiv:1808.03668),
**TransLOB** (arXiv:2003.00130), **LOBS5** (arXiv:2309.00638) — adapted from
their original classification / L3-token-generation setting into our
regression-forecasting benchmark.

Each is used as the **backbone** of a shared `GaussianARForecaster`:
  * backbone: (B, 100, 11) context  ->  (B, 2*11) Gaussian params (mean, log-var)
    of the NEXT joint step (row0 = Δmid, rows1..2k = signed-log queues).
  * training: one-step-ahead Gaussian NLL on sliding windows.
  * generation: autoregressive rollout — sample μ+σ·ε each step, slide the
    window, repeat `horizon` steps; `n_paths` independent draws.  Output matches
    JointTimeGradGenerator / PaintDiffusionGenerator:
        queue (n_paths, steps, 2k), mid (n_paths, steps).

Adaptations vs the originals (documented): the classification head is replaced
by a Gaussian regression head; inputs are our 11-dim δ-grid joint feature (not
FI-2010's 40-wide price/size layout), so DeepLOB's width-specific 1×2 convs
become a single feature-collapse conv. LOBS5 is the S5 backbone in spirit (the
diagonal SSM stack); its L3 message tokenizer / JAX pipeline are dropped.
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


# ───────────────────────────── DeepLOB ──────────────────────────────────────
class _DeepLOBNet(nn.Module):
    """CNN + Inception + LSTM (repo channel counts 32/64), feature width adapted
    from 40 -> 11 by a single (1,F) collapse conv."""

    def __init__(self, in_feat: int, out_dim: int):
        super().__init__()
        L = 0.01
        self.collapse = nn.Sequential(
            nn.Conv2d(1, 32, (1, in_feat)), nn.LeakyReLU(L), nn.BatchNorm2d(32))
        self.temporal = nn.Sequential(
            nn.Conv2d(32, 32, (4, 1)), nn.LeakyReLU(L), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)), nn.LeakyReLU(L), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)), nn.LeakyReLU(L), nn.BatchNorm2d(32),
        )
        self.inc1 = nn.Sequential(nn.Conv2d(32, 64, (1, 1)), nn.LeakyReLU(L), nn.BatchNorm2d(64),
                                  nn.Conv2d(64, 64, (3, 1), padding=(1, 0)), nn.LeakyReLU(L), nn.BatchNorm2d(64))
        self.inc2 = nn.Sequential(nn.Conv2d(32, 64, (1, 1)), nn.LeakyReLU(L), nn.BatchNorm2d(64),
                                  nn.Conv2d(64, 64, (5, 1), padding=(2, 0)), nn.LeakyReLU(L), nn.BatchNorm2d(64))
        self.inc3 = nn.Sequential(nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
                                  nn.Conv2d(32, 64, (1, 1)), nn.LeakyReLU(L), nn.BatchNorm2d(64))
        self.lstm = nn.LSTM(192, 64, batch_first=True)
        self.head = nn.Linear(64, out_dim)

    def forward(self, x):                    # x: (B, T, F)
        x = x.unsqueeze(1)                   # (B,1,T,F)
        x = self.collapse(x)                 # (B,32,T,1)
        x = self.temporal(x)                 # (B,32,T',1)
        x = torch.cat([self.inc1(x), self.inc2(x), self.inc3(x)], dim=1)  # (B,192,T',1)
        x = x.squeeze(-1).permute(0, 2, 1)   # (B,T',192)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])      # (B,out_dim)


# ───────────────────────────── TransLOB ─────────────────────────────────────
class _CausalConv1d(nn.Module):
    def __init__(self, cin, cout, k, dilation):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, k, dilation=dilation)

    def forward(self, x):                     # x: (B,C,T)
        return self.conv(F.pad(x, (self.pad, 0)))


class _TransformerBlock(nn.Module):
    def __init__(self, d_model, heads, ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff), nn.ReLU(), nn.Linear(ff, d_model))

    def forward(self, x, mask):
        a, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.ln1(x + a)
        return self.ln2(x + self.ff(x))


class _TransLOBNet(nn.Module):
    def __init__(self, in_feat: int, out_dim: int, conv_ch=14, heads=3, blocks=2):
        super().__init__()
        dils = [1, 2, 4, 8, 16]
        convs = []
        cin = in_feat
        for d in dils:
            convs += [_CausalConv1d(cin, conv_ch, 2, d), nn.ReLU()]
            cin = conv_ch
        self.convs = nn.ModuleList(convs)
        self.ln = nn.LayerNorm(conv_ch)
        self.d_model = conv_ch + 1            # + positional channel
        self.block = _TransformerBlock(self.d_model, heads, self.d_model * 4)
        self.n_blocks = blocks                # weight-shared (applied n_blocks times)
        self.head = nn.Sequential(nn.Linear(self.d_model, 64), nn.ReLU(), nn.Linear(64, out_dim))

    def forward(self, x):                     # x: (B,T,F)
        B, T, _ = x.shape
        h = x.transpose(1, 2)                 # (B,F,T)
        for layer in self.convs:
            h = layer(h)
        h = h.transpose(1, 2)                 # (B,T,conv_ch)
        h = self.ln(h)
        pos = torch.linspace(-1, 1, T, device=x.device).view(1, T, 1).expand(B, T, 1)
        h = torch.cat([h, pos], dim=-1)       # (B,T,d_model)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        for _ in range(self.n_blocks):
            h = self.block(h, mask)
        return self.head(h[:, -1, :])         # (B,out_dim)


# ───────────────────────────── LOBS5 (S5) ───────────────────────────────────
class _S5Layer(nn.Module):
    """Diagonal causal S5 SSM (ZOH) + prenorm + half_glu1 + residual."""

    def __init__(self, H: int, P: int):
        super().__init__()
        self.H, self.P = H, P
        self.norm = nn.LayerNorm(H)
        self.lam_re_raw = nn.Parameter(torch.full((P,), math.log(math.expm1(0.5))))  # softplus->0.5
        self.lam_im = nn.Parameter(math.pi * torch.arange(P, dtype=torch.float32))
        self.log_step = nn.Parameter(torch.empty(P).uniform_(math.log(1e-3), math.log(1e-1)))
        self.B_re = nn.Parameter(torch.randn(P, H) / math.sqrt(H))
        self.B_im = nn.Parameter(torch.randn(P, H) / math.sqrt(H))
        self.C_re = nn.Parameter(torch.randn(H, P) / math.sqrt(P))
        self.C_im = nn.Parameter(torch.randn(H, P) / math.sqrt(P))
        self.D = nn.Parameter(torch.ones(H))
        self.glu = nn.Linear(H, H)

    def _ssm(self, u):                        # u: (B,T,H) real -> (B,T,H) real
        B, T, H = u.shape
        lam = torch.complex(-F.softplus(self.lam_re_raw), self.lam_im)   # Re<0
        step = torch.exp(self.log_step)
        Lbar = torch.exp(lam * step)                                     # (P,)
        Bbar = ((Lbar - 1.0) / lam)[:, None] * torch.complex(self.B_re, self.B_im)  # (P,H)
        Bu = torch.einsum("bth,ph->btp", u.to(Bbar.dtype), Bbar)        # (B,T,P) complex
        C = torch.complex(self.C_re, self.C_im)                          # (H,P)
        x = torch.zeros(B, self.P, dtype=Bbar.dtype, device=u.device)
        ys = []
        for t in range(T):
            x = Lbar[None] * x + Bu[:, t]
            ys.append(2.0 * torch.real(torch.einsum("hp,bp->bh", C, x)))  # conj-sym => 2 Re
        return torch.stack(ys, dim=1) + u * self.D

    def forward(self, x):
        h = self._ssm(self.norm(x))
        h = F.gelu(h)
        h = h * torch.sigmoid(self.glu(h))    # half_glu1
        return x + h


class _S5Net(nn.Module):
    def __init__(self, in_feat: int, out_dim: int, H=32, P=16, layers=6):
        super().__init__()
        self.enc = nn.Linear(in_feat, H)
        self.layers = nn.ModuleList([_S5Layer(H, P) for _ in range(layers)])
        self.head = nn.Linear(H, out_dim)

    def forward(self, x):                     # (B,T,F)
        h = self.enc(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h[:, -1, :])         # (B,out_dim)


BACKBONES = {"DeepLOB": _DeepLOBNet, "TransLOB": _TransLOBNet, "LOBS5": _S5Net}


# ───────────────────────── Gaussian AR forecaster ───────────────────────────
class GaussianARForecaster:
    """Wraps a backbone as a probabilistic autoregressive joint forecaster."""

    def __init__(self, cfg: BenchmarkConfig, normalizer: Normalizer, model_name: str,
                 device: Optional[str] = None, epochs: int = 15, lr: float = 2e-4,
                 batch_size: int = 128, stride: int = 3, max_train_windows: int = 20000,
                 save_dir: str = "checkpoints/supervised", train: bool = True):
        self.cfg = cfg
        self.normalizer = normalizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.k = cfg.levels
        self.F_real = 2 * cfg.levels + 1
        self.context_len = cfg.timegrad_history_length
        self.delta_mid_clip = float(cfg.delta_mid_clip)
        self.model_name = model_name
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size
        self.stride, self.max_train_windows = stride, max_train_windows
        self.save_dir, self.train_flag = save_dir, train
        self.net = BACKBONES[model_name](self.F_real, 2 * self.F_real).to(self.device)
        self.row_mean = self.row_std = None

    def _content(self, q, a):
        return build_joint_features(a, q, self.delta_mid_clip).astype(np.float32)

    def _std(self, c):
        return (c - self.row_mean) / self.row_std

    # ---- train ----
    def fit(self, queue_norm, anchors, timestamps):
        if not self.train_flag:
            return
        c = self._content(queue_norm, anchors)
        self.row_mean = c.mean(0, keepdims=True)
        self.row_std = c.std(0, keepdims=True) + 1e-6
        cs = self._std(c)                                 # (T, F_real)
        C = self.context_len
        starts = list(range(0, len(cs) - C - 1, self.stride))
        if len(starts) > self.max_train_windows:
            rng = np.random.default_rng(0)
            starts = rng.choice(starts, self.max_train_windows, replace=False).tolist()
        X = np.stack([cs[s:s + C] for s in starts])       # (N,C,F)
        Y = np.stack([cs[s + C] for s in starts])         # (N,F)
        X = torch.tensor(X); Y = torch.tensor(Y)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y), batch_size=self.batch_size, shuffle=True, drop_last=True)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        os.makedirs(self.save_dir, exist_ok=True)
        self.net.train()
        npar = sum(p.numel() for p in self.net.parameters())
        print(f"[{self.model_name}] train: {len(starts)} windows, params={npar:,}, "
              f"epochs={self.epochs}, device={self.device}")
        for ep in range(self.epochs):
            tot = nb = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                out = self.net(xb)
                mu, logv = out[:, :self.F_real], out[:, self.F_real:]
                logv = torch.clamp(logv, -8, 4)
                loss = (0.5 * (logv + (yb - mu) ** 2 / torch.exp(logv))).mean()  # Gaussian NLL
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); nb += 1
            print(f"[{self.model_name}]   epoch {ep+1}/{self.epochs}  nll={tot/max(nb,1):.4f}")
        torch.save({"model": self.net.state_dict(), "row_mean": self.row_mean, "row_std": self.row_std},
                   os.path.join(self.save_dir, f"{self.model_name}.pt"))
        self.net.eval()

    def load(self, path):
        st = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(st["model"]); self.row_mean = st["row_mean"]; self.row_std = st["row_std"]
        self.net.eval().to(self.device)
        print(f"[{self.model_name}] loaded {path}")

    # ---- chunked autoregressive rollout (bounded memory) ----
    @torch.no_grad()
    def generate_many(self, ctx_queues, ctx_anchors, steps, n_paths, chunk=1024, seed=None):
        C = self.context_len
        Wn = len(ctx_queues)
        if seed is not None:
            torch.manual_seed(seed)
        # per-window standardized context (small); samples index into it (no 50k replication)
        ctx_w = np.stack([self._std(self._content(ctx_queues[i], ctx_anchors[i]))[-C:]
                          for i in range(Wn)]).astype(np.float32)   # (Wn,C,F)
        last_anchor_w = np.array([float(a[-1]) for a in ctx_anchors], dtype=np.float32)
        total = Wn * n_paths
        out = np.empty((total, steps, self.F_real), dtype=np.float32)
        for c0 in range(0, total, chunk):
            c1 = min(c0 + chunk, total)
            win = np.arange(c0, c1) // n_paths                     # window index per sample
            ctx = torch.tensor(ctx_w[win], device=self.device)     # (b,C,F) — only this chunk on device
            step_out = []
            for _ in range(steps):
                o = self.net(ctx)
                mu, logv = o[:, :self.F_real], torch.clamp(o[:, self.F_real:], -8, 4)
                nxt = mu + torch.exp(0.5 * logv) * torch.randn_like(mu)
                ctx = torch.cat([ctx[:, 1:], nxt[:, None]], dim=1)
                step_out.append(nxt)
            out[c0:c1] = torch.stack(step_out, dim=1).cpu().numpy()
        last_anchor = np.repeat(last_anchor_w, n_paths)            # (total,)
        content = out * self.row_std.reshape(-1)[:self.F_real] + self.row_mean.reshape(-1)[:self.F_real]
        content = content.reshape(Wn, n_paths, steps, self.F_real)
        delta = np.clip(content[..., 0], -self.delta_mid_clip, self.delta_mid_clip)
        queue = content[..., 1:]
        queue[..., :self.k] = np.clip(queue[..., :self.k], -5.0, 0.0)
        queue[..., self.k:] = np.clip(queue[..., self.k:], 0.0, 5.0)
        mid = last_anchor.reshape(Wn, n_paths)[..., None] + np.cumsum(delta, axis=2)
        return queue.astype(np.float32), mid.astype(np.float32)

    @property
    def name(self):
        return self.model_name

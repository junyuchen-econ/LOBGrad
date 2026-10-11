"""
poisson.py
----------
Poisson order flow model for LOB snapshot generation.

Three independent Poisson processes drive queue dynamics per price level:

  1. Limit submissions : constant rate λ_L_i.  Increases |Q_i|.
  2. Cancellations     : queue-proportional rate λ_C_i * |Q_i|.  Decreases |Q_i|.
  3. Market orders     : constant rate λ_M.  Acts only at best bid / best ask.

Simulation via superposition:
  Total rate Λ = Σ λ_k.  N ~ Poisson(Λ).  Events assigned via multinomial(N, λ_k/Λ).

Price grid is fixed (no price changes modelled).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .base import LOBGenerator
from ..config import BenchmarkConfig
from ..data.normalizer import Normalizer

_MARKET_FRACTION = 0.30


class PoissonGenerator(LOBGenerator):
    """
    Poisson order flow model for 2k-dim signed-queue LOB snapshots.
    """

    def __init__(self, cfg: BenchmarkConfig, normalizer: Normalizer):
        self.cfg        = cfg
        self.normalizer = normalizer
        self.signed     = cfg.queue_repr == "signed"
        k               = cfg.levels

        self.lam_L_bid_: np.ndarray  = np.ones(k)
        self.lam_L_ask_: np.ndarray  = np.ones(k)
        self.lam_C_bid_: np.ndarray  = np.ones(k) * 0.1
        self.lam_C_ask_: np.ndarray  = np.ones(k) * 0.1
        self.lam_M_bid_: float       = 0.5
        self.lam_M_ask_: float       = 0.5

    # ------------------------------------------------------------------
    def fit(
        self,
        snapshots:  np.ndarray,
        timestamps: np.ndarray,
    ) -> None:
        """
        snapshots : (T, 2k) normalised
        """
        k = self.cfg.levels

        raw_q    = self.normalizer.inverse_transform(snapshots)  # (T, 2k)
        raw_bids = raw_q[:, :k].astype(float)                    # (T, k), ≤ 0
        raw_asks = raw_q[:, k:].astype(float)                    # (T, k), ≥ 0

        # ---- Queue dynamics (raw step-to-step changes, no price shift) -
        T         = len(raw_bids)
        d_bid_pos = np.zeros((T - 1, k))
        d_bid_neg = np.zeros((T - 1, k))
        d_ask_pos = np.zeros((T - 1, k))
        d_ask_neg = np.zeros((T - 1, k))

        for t in range(T - 1):
            d_bid = raw_bids[t + 1] - raw_bids[t]
            d_ask = raw_asks[t + 1] - raw_asks[t]
            if self.signed:
                # bids ≤ 0: more negative = submission, less negative = removal
                d_bid_pos[t] = np.maximum(-d_bid, 0)
                d_bid_neg[t] = np.maximum( d_bid, 0)
            else:
                # bids ≥ 0: larger = submission, smaller = removal
                d_bid_pos[t] = np.maximum( d_bid, 0)
                d_bid_neg[t] = np.maximum(-d_bid, 0)
            d_ask_pos[t] = np.maximum( d_ask, 0)
            d_ask_neg[t] = np.maximum(-d_ask, 0)

        self.lam_L_bid_ = d_bid_pos.mean(axis=0)
        self.lam_L_ask_ = d_ask_pos.mean(axis=0)

        mean_abs_bid  = np.abs(raw_bids[:-1]).mean(axis=0)  # same regardless of sign convention
        mean_abs_ask  = raw_asks[:-1].mean(axis=0)
        total_rem_bid = d_bid_neg.mean(axis=0)
        total_rem_ask = d_ask_neg.mean(axis=0)

        mo_bid = np.zeros(k); mo_bid[0] = _MARKET_FRACTION * total_rem_bid[0]
        mo_ask = np.zeros(k); mo_ask[0] = _MARKET_FRACTION * total_rem_ask[0]
        self.lam_M_bid_ = float(mo_bid[0])
        self.lam_M_ask_ = float(mo_ask[0])

        self.lam_C_bid_ = (total_rem_bid - mo_bid) / (mean_abs_bid + 1e-8)
        self.lam_C_ask_ = (total_rem_ask - mo_ask) / (mean_abs_ask + 1e-8)

        print(
            f"[Poisson] λ_L_bid={np.round(self.lam_L_bid_, 3)}\n"
            f"  λ_L_ask={np.round(self.lam_L_ask_, 3)}\n"
            f"  λ_C_bid={np.round(self.lam_C_bid_, 4)}\n"
            f"  λ_C_ask={np.round(self.lam_C_ask_, 4)}\n"
            f"  λ_M_bid={self.lam_M_bid_:.4f}  λ_M_ask={self.lam_M_ask_:.4f}"
        )

    # ------------------------------------------------------------------
    def generate(
        self,
        context: np.ndarray,
        steps:   int,
        n_paths: int,
        rng:     Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        context : (context_len, 2k) normalised

        Returns
        -------
        features : (n_paths, steps, 2k)  normalised
        """
        if rng is None:
            rng = np.random.default_rng()

        k        = self.cfg.levels
        features = np.zeros((n_paths, steps, 2 * k), dtype=np.float32)

        last_raw = self.normalizer.inverse_transform(context[-1])

        for p in range(n_paths):
            bids = last_raw[:k].copy().astype(float)
            asks = last_raw[k:].copy().astype(float)

            for t in range(steps):
                # ---- Queue dynamics via superposition ------------------
                all_rates = np.concatenate([
                    self.lam_L_bid_,
                    self.lam_L_ask_,
                    self.lam_C_bid_ * np.abs(bids),
                    self.lam_C_ask_ * np.abs(asks),
                    [self.lam_M_bid_, self.lam_M_ask_],
                ])
                all_rates = np.maximum(all_rates, 0.0)
                Lambda    = float(all_rates.sum())
                N         = int(rng.poisson(Lambda))

                if N > 0 and Lambda > 1e-12:
                    ev  = rng.multinomial(N, all_rates / Lambda)
                    idx = 0
                    sub_bid = ev[idx:idx+k]; idx += k
                    sub_ask = ev[idx:idx+k]; idx += k
                    can_bid = ev[idx:idx+k]; idx += k
                    can_ask = ev[idx:idx+k]; idx += k
                    mo_bid  = int(ev[idx]);  idx += 1
                    mo_ask  = int(ev[idx])

                    if self.signed:
                        bids -= sub_bid.astype(float)   # more negative = more volume
                        bids += can_bid.astype(float)   # less negative = less volume
                        bids[0] += mo_bid               # less negative = consumed
                    else:
                        bids += sub_bid.astype(float)   # larger = more volume
                        bids -= can_bid.astype(float)   # smaller = less volume
                        bids[0] -= mo_bid               # smaller = consumed

                    asks += sub_ask.astype(float)
                    asks -= can_ask.astype(float)
                    asks[0] -= mo_ask

                bids = np.minimum(bids, 0.0) if self.signed else np.maximum(bids, 0.0)
                asks = np.maximum(asks, 0.0)

                raw_snap = np.concatenate([bids, asks]).astype(np.float32)
                features[p, t] = self.normalizer.transform(raw_snap)

        return features

    @property
    def name(self) -> str:
        return "Poisson"

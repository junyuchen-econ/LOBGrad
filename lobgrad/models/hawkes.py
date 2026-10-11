"""
hawkes.py
---------
Hawkes order flow model for LOB snapshot generation.

Identical to PoissonGenerator except market and cancel intensities are
self-exciting:

  λ_M_*(t) = μ_M_* + exp(-β_M_*)*(λ_M_*(t-1) + α_M_* * mo_*_{t-1})
  λ_C_*(t) = μ_C_* + exp(-β_C_*)*(λ_C_*(t-1) + α_C_* * cancel_*_{t-1})

Market buys cluster with market buys; market sells cluster with market sells.
Cancels cluster per side (aggregated across depth levels) — captures cancel
storms when prices move or makers retreat.
Excitation is driven by realised events (not price ticks) — price is not modelled.
Limit rates remain constant (Poisson). The Hawkes cancel intensity acts as a
unit-mean multiplier on the existing per-level cancel rates λ_C_*[l] * |Q[l]|.
Four (μ, α, β) triples are estimated via Poisson MLE — one per (side, channel).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize

from .base import LOBGenerator
from .poisson import _MARKET_FRACTION
from ..config import BenchmarkConfig
from ..data.normalizer import Normalizer


class HawkesGenerator(LOBGenerator):

    def __init__(self, cfg: BenchmarkConfig, normalizer: Normalizer):
        self.cfg        = cfg
        self.normalizer = normalizer
        self.signed     = cfg.queue_repr == "signed"
        k               = cfg.levels

        self.lam_L_bid_: np.ndarray = np.ones(k)
        self.lam_L_ask_: np.ndarray = np.ones(k)
        self.lam_C_bid_: np.ndarray = np.ones(k) * 0.1
        self.lam_C_ask_: np.ndarray = np.ones(k) * 0.1

        self.mu_M_ask_:   float = 0.5
        self.alpha_ask_:  float = 0.5
        self.beta_ask_:   float = 1.0
        self.mu_M_bid_:   float = 0.5
        self.alpha_bid_:  float = 0.5
        self.beta_bid_:   float = 1.0

        # Aggregated per-side cancel Hawkes
        self.mu_C_ask_:    float = 1.0
        self.alpha_C_ask_: float = 0.0
        self.beta_C_ask_:  float = 1.0
        self.mu_C_bid_:    float = 1.0
        self.alpha_C_bid_: float = 0.0
        self.beta_C_bid_:  float = 1.0
        # Mean per-step aggregated cancel count, used to normalise the
        # Hawkes intensity to a unit-mean multiplier in generation.
        self.mean_cancel_bid_: float = 1.0
        self.mean_cancel_ask_: float = 1.0

    # ------------------------------------------------------------------
    def fit(
        self,
        snapshots:  np.ndarray,
        timestamps: np.ndarray,
    ) -> None:
        """
        Hawkes excitation is driven by market order activity inferred from
        best-level queue shrinkage.
        """
        k = self.cfg.levels

        raw_q    = self.normalizer.inverse_transform(snapshots)
        raw_bids = raw_q[:, :k].astype(float)
        raw_asks = raw_q[:, k:].astype(float)

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
                d_bid_pos[t] = np.maximum(-d_bid, 0)
                d_bid_neg[t] = np.maximum( d_bid, 0)
            else:
                d_bid_pos[t] = np.maximum( d_bid, 0)
                d_bid_neg[t] = np.maximum(-d_bid, 0)
            d_ask_pos[t] = np.maximum( d_ask, 0)
            d_ask_neg[t] = np.maximum(-d_ask, 0)

        self.lam_L_bid_ = d_bid_pos.mean(axis=0)
        self.lam_L_ask_ = d_ask_pos.mean(axis=0)

        mean_abs_bid  = np.abs(raw_bids[:-1]).mean(axis=0)
        mean_abs_ask  = raw_asks[:-1].mean(axis=0)
        total_rem_bid = d_bid_neg.mean(axis=0)
        total_rem_ask = d_ask_neg.mean(axis=0)

        # Estimate per-step MO counts from best-level queue shrinkage.
        # Market sells shrink the best bid; market buys shrink the best ask.
        mo_bid_series = d_bid_neg[:, 0] * _MARKET_FRACTION  # market sells hitting best bid
        mo_ask_series = d_ask_neg[:, 0] * _MARKET_FRACTION  # market buys  hitting best ask
        self.mu_M_bid_, self.alpha_bid_, self.beta_bid_ = self._fit_hawkes_1d(mo_bid_series)
        self.mu_M_ask_, self.alpha_ask_, self.beta_ask_ = self._fit_hawkes_1d(mo_ask_series)

        mo_bid = np.zeros(k); mo_bid[0] = _MARKET_FRACTION * total_rem_bid[0]
        mo_ask = np.zeros(k); mo_ask[0] = _MARKET_FRACTION * total_rem_ask[0]
        self.lam_C_bid_ = (total_rem_bid - mo_bid) / (mean_abs_bid + 1e-8)
        self.lam_C_ask_ = (total_rem_ask - mo_ask) / (mean_abs_ask + 1e-8)

        # Aggregated per-side cancel event series. Cancels = total queue
        # shrinkage minus the MO portion (only at L1).
        cancel_bid_series = (
            d_bid_neg[:, 1:].sum(axis=1) + (1.0 - _MARKET_FRACTION) * d_bid_neg[:, 0]
        )
        cancel_ask_series = (
            d_ask_neg[:, 1:].sum(axis=1) + (1.0 - _MARKET_FRACTION) * d_ask_neg[:, 0]
        )
        self.mu_C_bid_, self.alpha_C_bid_, self.beta_C_bid_ = self._fit_hawkes_1d(cancel_bid_series)
        self.mu_C_ask_, self.alpha_C_ask_, self.beta_C_ask_ = self._fit_hawkes_1d(cancel_ask_series)
        self.mean_cancel_bid_ = max(float(cancel_bid_series.mean()), 1e-8)
        self.mean_cancel_ask_ = max(float(cancel_ask_series.mean()), 1e-8)

        print(
            f"[Hawkes] MO ask: μ={self.mu_M_ask_:.4f}  α={self.alpha_ask_:.4f}  β={self.beta_ask_:.4f}\n"
            f"  MO bid: μ={self.mu_M_bid_:.4f}  α={self.alpha_bid_:.4f}  β={self.beta_bid_:.4f}\n"
            f"  C  ask: μ={self.mu_C_ask_:.4f}  α={self.alpha_C_ask_:.4f}  β={self.beta_C_ask_:.4f}\n"
            f"  C  bid: μ={self.mu_C_bid_:.4f}  α={self.alpha_C_bid_:.4f}  β={self.beta_C_bid_:.4f}\n"
            f"  λ_L_bid={np.round(self.lam_L_bid_, 3)}\n"
            f"  λ_L_ask={np.round(self.lam_L_ask_, 3)}\n"
            f"  λ_C_bid={np.round(self.lam_C_bid_, 4)}\n"
            f"  λ_C_ask={np.round(self.lam_C_ask_, 4)}"
        )

    def _fit_hawkes_1d(self, events: np.ndarray) -> tuple:
        """
        Fit a discrete Hawkes intensity model via Poisson MLE.

        Model: λ_t = μ + exp(-β) * (λ_{t-1} + α * events_{t-1})
        Likelihood: Poisson(events_t | λ_t)  →  NLL = Σ (λ_t - events_t * log λ_t)
        """
        beta0 = self.cfg.hawkes_beta
        mu0   = max(float(events.mean()), 1e-4)

        def neg_ll(params):
            mu, alpha, beta = params
            if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= beta:
                return 1e9
            lam = np.empty(len(events))
            lam[0] = mu
            for t in range(1, len(events)):
                lam[t] = mu + np.exp(-beta) * (lam[t - 1] + alpha * events[t - 1])
            lam = np.maximum(lam, 1e-8)
            return float(np.sum(lam - events * np.log(lam)))

        res = minimize(neg_ll, x0=[mu0, 0.3, beta0],
                       method="Nelder-Mead",
                       options={"maxiter": self.cfg.hawkes_max_iter, "xatol": 1e-5})
        mu, alpha, beta = res.x
        return max(float(mu), 1e-6), max(float(alpha), 0.0), max(float(beta), 1e-3)

    # ------------------------------------------------------------------
    def generate(
        self,
        context: np.ndarray,
        steps:   int,
        n_paths: int,
        rng:     Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        k        = self.cfg.levels
        features = np.zeros((n_paths, steps, 2 * k), dtype=np.float32)

        last_raw = self.normalizer.inverse_transform(context[-1])

        # Warm up Hawkes intensities by running through the context window.
        # MO and cancel events are inferred from queue shrinkage, consistent
        # with the fitting step.  The result is a single conditioned starting
        # intensity shared by all paths (deterministic given the context).
        raw_ctx     = self.normalizer.inverse_transform(context)   # (context_len, 2k)
        raw_bid_ctx = np.abs(raw_ctx[:, :k])                       # (context_len, k)
        raw_ask_ctx = raw_ctx[:, k:]                               # (context_len, k)
        lam_M_ask_0 = self.mu_M_ask_
        lam_M_bid_0 = self.mu_M_bid_
        lam_C_ask_0 = self.mu_C_ask_
        lam_C_bid_0 = self.mu_C_bid_
        for t in range(len(context) - 1):
            d_bid_neg = np.maximum(raw_bid_ctx[t] - raw_bid_ctx[t + 1], 0.0)   # (k,)
            d_ask_neg = np.maximum(raw_ask_ctx[t] - raw_ask_ctx[t + 1], 0.0)   # (k,)
            mo_bid_ev = float(d_bid_neg[0]) * _MARKET_FRACTION
            mo_ask_ev = float(d_ask_neg[0]) * _MARKET_FRACTION
            cancel_bid_ev = float(
                d_bid_neg[1:].sum() + (1.0 - _MARKET_FRACTION) * d_bid_neg[0]
            )
            cancel_ask_ev = float(
                d_ask_neg[1:].sum() + (1.0 - _MARKET_FRACTION) * d_ask_neg[0]
            )
            lam_M_ask_0 = (self.mu_M_ask_
                           + np.exp(-self.beta_ask_) * (lam_M_ask_0 + self.alpha_ask_ * mo_ask_ev))
            lam_M_bid_0 = (self.mu_M_bid_
                           + np.exp(-self.beta_bid_) * (lam_M_bid_0 + self.alpha_bid_ * mo_bid_ev))
            lam_C_ask_0 = (self.mu_C_ask_
                           + np.exp(-self.beta_C_ask_) * (lam_C_ask_0 + self.alpha_C_ask_ * cancel_ask_ev))
            lam_C_bid_0 = (self.mu_C_bid_
                           + np.exp(-self.beta_C_bid_) * (lam_C_bid_0 + self.alpha_C_bid_ * cancel_bid_ev))

        for p in range(n_paths):
            bids      = last_raw[:k].copy().astype(float)
            asks      = last_raw[k:].copy().astype(float)
            # Start from context-conditioned intensity, not the unconditional baseline
            lam_M_ask = lam_M_ask_0
            lam_M_bid = lam_M_bid_0
            lam_C_ask = lam_C_ask_0
            lam_C_bid = lam_C_bid_0

            for t in range(steps):
                # Unit-mean multipliers: μ_C / μ̄_C ≈ 1 at steady state, but
                # excursions of the Hawkes intensity scale per-level cancels.
                m_C_bid = lam_C_bid / self.mean_cancel_bid_
                m_C_ask = lam_C_ask / self.mean_cancel_ask_
                all_rates = np.concatenate([
                    self.lam_L_bid_,
                    self.lam_L_ask_,
                    self.lam_C_bid_ * np.abs(bids) * m_C_bid,
                    self.lam_C_ask_ * np.abs(asks) * m_C_ask,
                    [lam_M_bid, lam_M_ask],
                ])
                all_rates = np.maximum(all_rates, 0.0)
                Lambda    = float(all_rates.sum())
                N         = int(rng.poisson(Lambda))

                mo_bid_ev = 0
                mo_ask_ev = 0
                cancel_bid_ev = 0
                cancel_ask_ev = 0
                if N > 0 and Lambda > 1e-12:
                    ev  = rng.multinomial(N, all_rates / Lambda)
                    idx = 0
                    sub_bid = ev[idx:idx+k]; idx += k
                    sub_ask = ev[idx:idx+k]; idx += k
                    can_bid = ev[idx:idx+k]; idx += k
                    can_ask = ev[idx:idx+k]; idx += k
                    mo_bid_ev = int(ev[idx]);  idx += 1
                    mo_ask_ev = int(ev[idx])
                    cancel_bid_ev = int(can_bid.sum())
                    cancel_ask_ev = int(can_ask.sum())

                    if self.signed:
                        bids -= sub_bid.astype(float)
                        bids += can_bid.astype(float)
                        bids[0] += mo_bid_ev
                    else:
                        bids += sub_bid.astype(float)
                        bids -= can_bid.astype(float)
                        bids[0] -= mo_bid_ev

                    asks += sub_ask.astype(float)
                    asks -= can_ask.astype(float)
                    asks[0] -= mo_ask_ev

                bids = np.minimum(bids, 0.0) if self.signed else np.maximum(bids, 0.0)
                asks = np.maximum(asks, 0.0)

                raw_snap = np.concatenate([bids, asks]).astype(np.float32)
                features[p, t] = self.normalizer.transform(raw_snap)

                # Update Hawkes intensities using this step's realised counts
                lam_M_ask = (self.mu_M_ask_
                             + np.exp(-self.beta_ask_) * (lam_M_ask + self.alpha_ask_ * mo_ask_ev))
                lam_M_bid = (self.mu_M_bid_
                             + np.exp(-self.beta_bid_) * (lam_M_bid + self.alpha_bid_ * mo_bid_ev))
                lam_C_ask = (self.mu_C_ask_
                             + np.exp(-self.beta_C_ask_) * (lam_C_ask + self.alpha_C_ask_ * cancel_ask_ev))
                lam_C_bid = (self.mu_C_bid_
                             + np.exp(-self.beta_C_bid_) * (lam_C_bid + self.alpha_C_bid_ * cancel_bid_ev))

        return features

    @property
    def name(self) -> str:
        return "Hawkes"

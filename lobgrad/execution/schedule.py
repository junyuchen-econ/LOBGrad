"""
schedule.py
-----------
Trading schedule constructors used by the open-loop downstream comparison
notebooks (paper_figures*.ipynb) and as default-policy completions inside the
MPC rollouts (see ``mpc.py``).

A schedule is a (horizon,) array where schedule[t] >= 0 and sum = order_size.

Schedules
---------
  twap_schedule           : uniform execution
  almgren_chriss_schedule : closed-form AC optimal schedule
  parametric_schedule     : flexible front/back-loaded via a decay parameter
  cost_weighted_schedule  : inverse-cost weighted schedule built from simulated paths

Note: VWAP is intentionally not included: this codebase has no trade-volume
data, so any "volume profile" would be fabricated and VWAP would collapse to
TWAP.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Schedule constructors
# ---------------------------------------------------------------------------

def twap_schedule(order_size: float, horizon: int) -> np.ndarray:
    """Uniform execution: order_size / horizon shares at each step."""
    s = np.full(horizon, order_size / horizon, dtype=np.float64)
    return s


def almgren_chriss_schedule(
    order_size: float,
    horizon:    int,
    sigma:      float,    # price volatility per step (USDT)
    eta:        float,    # temporary impact coefficient (USDT / share)
    gamma:      float = 0.0,  # permanent impact coefficient (USDT / share)
    risk_aversion: float = 1e-5,
) -> np.ndarray:
    """
    Almgren-Chriss (2001) optimal liquidation schedule.

    Closed-form solution that minimises E[cost] + λ * Var[cost] subject to
    full execution by step `horizon`.

    The discrete-time schedule is:

        x_t = X * sinh(κ * (T - t)) / sinh(κ * T)

    where κ = sqrt(risk_aversion * sigma² / eta), T = horizon, X = order_size.

    Parameters
    ----------
    order_size    : total shares to execute
    horizon       : number of steps
    sigma         : price volatility per step
    eta           : temporary linear impact (price impact per share per step)
    gamma         : permanent linear impact (default 0)
    risk_aversion : λ in E[cost] + λ*Var[cost]

    References
    ----------
    Almgren & Chriss (2001), "Optimal execution of portfolio transactions."
    """
    if eta < 1e-12:
        return twap_schedule(order_size, horizon)

    kappa = np.sqrt(risk_aversion * sigma**2 / eta)

    if kappa < 1e-8:
        # κ → 0 ⟹ TWAP
        return twap_schedule(order_size, horizon)

    T = horizon
    t = np.arange(T, dtype=np.float64)  # steps 0, 1, ..., T-1

    # Shares remaining at start of each step
    remaining = order_size * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
    # Shares traded at each step = drop in remaining
    traded = remaining - np.append(remaining[1:], 0.0)
    traded = np.maximum(traded, 0.0)

    # Renormalise to ensure exactly order_size shares traded
    total = traded.sum()
    if total > 1e-12:
        traded *= order_size / total
    return traded


def parametric_schedule(
    order_size: float,
    horizon:    int,
    decay:      float,
) -> np.ndarray:
    """
    Flexible parametric schedule via exponential decay.

      decay > 0 : front-loaded (trade more early)
      decay = 0 : TWAP
      decay < 0 : back-loaded  (trade more late)

    schedule[t] ∝ exp(-decay * t / horizon)
    """
    t = np.arange(horizon, dtype=np.float64)
    weights = np.exp(-decay * t / max(horizon - 1, 1))
    weights /= weights.sum()
    return order_size * weights


def cost_weighted_schedule(
    order_size:  float,
    mid_paths:   np.ndarray,   # (n_paths, horizon)
    queue_paths: np.ndarray,   # (n_paths, horizon, 2k) raw absolute sizes
    side:        str,
    price_level: float,
    levels:      int,
) -> np.ndarray:
    """
    Inverse-cost weighted schedule: allocate more shares to steps where
    predicted per-unit execution cost is lower.

    Per step t: estimate mean book_walk_cost(1 unit) across simulated paths.
    Schedule:   proportional to 1 / (mean_cost_t + eps).
    """
    from .cost import book_walk_cost
    H = mid_paths.shape[1]
    N = mid_paths.shape[0]
    unit_cost = np.array([
        np.mean([
            book_walk_cost(1.0, mid_paths[p, t], queue_paths[p, t],
                           side, price_level, levels)
            for p in range(N)
        ])
        for t in range(H)
    ])
    inv = 1.0 / (np.abs(unit_cost) + 1e-4)
    return order_size * inv / inv.sum()

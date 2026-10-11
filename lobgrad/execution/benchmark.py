"""
benchmark.py
------------
Multi-source × multi-window orchestration for the MPC execution downstream
task. Runs each (window, source) pair, collects per-window costs, and
computes regret vs the oracle (RealFutureRollouts) along with bootstrap CIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .mpc import run_mpc_episode_with_norm, run_twap_episode
from .rollouts import RolloutSource, RealFutureRollouts


# =====================================================================
# Window data container
# =====================================================================
@dataclass
class ExecutionWindow:
    """One evaluation window: realized future + the prefix history used as
    initial context for the model.
    """
    real_mid:         np.ndarray   # (T,) raw USDT — realized future window
    real_queue_raw:   np.ndarray   # (T, 2k) raw absolute sizes — realized future
    real_queue_norm:  np.ndarray   # (T, 2k) normalized — for evolving history
    prefix_anchors:   np.ndarray   # (ctx,) raw USDT — pre-window history
    prefix_queue_norm: np.ndarray  # (ctx, 2k) normalized — pre-window history


# =====================================================================
# Calibration helpers
# =====================================================================
def calibrate_q_from_depth(
    train_queue_raw:  np.ndarray,    # (T, 2k) raw absolute sizes
    levels:           int,
    side:             str,
    levels_to_avg:    int = 5,
    multiplier:       float = 1.0,
) -> float:
    """Return ``multiplier × avg-of-top-`levels_to_avg`-depth`` on `side`.

    Standard calibration: pick Q so the order is meaningful relative to
    typical liquidity at the top of the book. Defaults: 1.0 × top-5 depth.
    """
    if levels_to_avg > levels:
        raise ValueError(f"levels_to_avg={levels_to_avg} > levels={levels}")
    if side == "buy":
        # Walk ask side: queue indices [k .. k+levels_to_avg-1]
        side_slice = train_queue_raw[:, levels : levels + levels_to_avg]
    else:
        side_slice = train_queue_raw[:, :levels_to_avg]
    return float(np.mean(np.abs(side_slice))) * multiplier * levels_to_avg


# =====================================================================
# Single-window run across all sources
# =====================================================================
def run_window_for_sources(
    *,
    window:          ExecutionWindow,
    sources:         Dict[str, RolloutSource],
    Q_total:         float,
    n_rollouts:      int,
    n_candidates:    int,
    side:            str,
    price_level:     float,
    levels:          int,
) -> Dict[str, float]:
    """Run a single window for each rollout source plus TWAP.

    Returns
    -------
    {source_name: total_cost} for all sources, plus "TWAP".

    Note: a fresh ``RealFutureRollouts(window)`` is constructed inside this
    function and registered as the oracle source ``"Oracle"`` — callers do
    NOT need to provide it in ``sources``. Including it via ``sources`` is
    fine but redundant.
    """
    results: Dict[str, float] = {}

    # TWAP — closed form, no MPC.
    results["TWAP"] = run_twap_episode(
        real_mid       = window.real_mid,
        real_queue_raw = window.real_queue_raw,
        Q_total        = Q_total,
        side           = side,
        price_level    = price_level,
        levels         = levels,
    )["total_cost"]

    # Oracle — bound to this window.
    oracle = RealFutureRollouts(
        window_anchors   = window.real_mid,
        window_queue_raw = window.real_queue_raw,
    )
    results[oracle.name] = run_mpc_episode_with_norm(
        real_mid          = window.real_mid,
        real_queue_raw    = window.real_queue_raw,
        real_queue_norm   = window.real_queue_norm,
        history_init_q    = window.prefix_queue_norm,
        history_init_a    = window.prefix_anchors,
        Q_total           = Q_total,
        source            = oracle,
        n_rollouts        = 1,                         # deterministic
        n_candidates      = n_candidates,
        side              = side,
        price_level       = price_level,
        levels            = levels,
    )["total_cost"]

    # Each remaining source — same MPC, different rollouts.
    for name, src in sources.items():
        if name in (oracle.name, "TWAP"):
            continue
        results[name] = run_mpc_episode_with_norm(
            real_mid          = window.real_mid,
            real_queue_raw    = window.real_queue_raw,
            real_queue_norm   = window.real_queue_norm,
            history_init_q    = window.prefix_queue_norm,
            history_init_a    = window.prefix_anchors,
            Q_total           = Q_total,
            source            = src,
            n_rollouts        = n_rollouts,
            n_candidates      = n_candidates,
            side              = side,
            price_level       = price_level,
            levels            = levels,
        )["total_cost"]

    return results


# =====================================================================
# Full benchmark
# =====================================================================
def run_benchmark(
    *,
    windows:        List[ExecutionWindow],
    sources:        Dict[str, RolloutSource],
    Q_total:        float,
    n_rollouts:     int,
    n_candidates:   int,
    side:           str,
    price_level:    float,
    levels:         int,
    progress:       bool = True,
) -> Dict[str, np.ndarray]:
    """Run all windows × sources. Returns per-source per-window cost arrays.

    Returns
    -------
    {source_name: (n_windows,) total_cost array}, including "TWAP" and
    "Oracle" rows.
    """
    n_windows = len(windows)
    out: Dict[str, List[float]] = {}

    iterator = range(n_windows)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="MPC benchmark")
        except ImportError:
            pass

    for i in iterator:
        per_window = run_window_for_sources(
            window         = windows[i],
            sources        = sources,
            Q_total        = Q_total,
            n_rollouts     = n_rollouts,
            n_candidates   = n_candidates,
            side           = side,
            price_level    = price_level,
            levels         = levels,
        )
        for name, cost in per_window.items():
            out.setdefault(name, []).append(cost)

    return {name: np.asarray(vals, dtype=np.float64) for name, vals in out.items()}


# =====================================================================
# Regret + bootstrap CI
# =====================================================================
def regret_per_window(
    costs_by_source:  Dict[str, np.ndarray],
    oracle_name:      str = "Oracle",
    side:             str = "buy",
) -> Dict[str, np.ndarray]:
    """Convert per-source per-window costs to per-source per-window regret
    (relative to the oracle).

    Regret_i = (cost_i - oracle_cost_i) / oracle_cost_i   for buy side
    Regret_i = (oracle_cost_i - cost_i) / oracle_cost_i   for sell side
              (sell side: more proceeds = better, so flip the sign)

    Both directions: regret >= 0 with 0 = optimal.
    """
    if oracle_name not in costs_by_source:
        raise ValueError(f"oracle row {oracle_name!r} missing from costs_by_source")
    oracle_costs = costs_by_source[oracle_name]
    eps = 1e-12

    regret: Dict[str, np.ndarray] = {}
    for name, costs in costs_by_source.items():
        if side == "buy":
            r = (costs - oracle_costs) / (np.abs(oracle_costs) + eps)
        else:
            r = (oracle_costs - costs) / (np.abs(oracle_costs) + eps)
        regret[name] = r
    return regret


def summarize_regret(
    regret_by_source: Dict[str, np.ndarray],
    n_bootstrap:      int = 1000,
    ci_alpha:         float = 0.05,
    rng:              Optional[np.random.Generator] = None,
) -> "pd.DataFrame":
    """Mean regret per source with bootstrap CIs (paired across windows).

    Returns a DataFrame indexed by source name with columns:
        mean_regret, ci_lo, ci_hi, std_regret, n_windows
    Sorted by mean regret ascending.
    """
    import pandas as pd
    rng = rng if rng is not None else np.random.default_rng(0)

    rows = []
    for name, r in regret_by_source.items():
        n = len(r)
        # Bootstrap: resample (with replacement) the WINDOW indices and
        # take the mean over the resampled window's regret.
        boot_means = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_means[b] = r[idx].mean()
        lo, hi = np.quantile(boot_means, [ci_alpha / 2, 1 - ci_alpha / 2])
        rows.append({
            "source":      name,
            "mean_regret": float(r.mean()),
            "ci_lo":       float(lo),
            "ci_hi":       float(hi),
            "std_regret":  float(r.std()),
            "n_windows":   n,
        })
    df = pd.DataFrame(rows).set_index("source").sort_values("mean_regret")
    return df

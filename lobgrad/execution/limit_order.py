"""
limit_order.py
--------------
Limit-order CVaR benchmark for the optimal-execution downstream task.

Task: sell a fixed quantity Q via a single limit-sell order placed at
arrival. The agent chooses an offset d (number of ask-side tick levels
above mid), waits up to H steps for a passive fill, and is forced out
at market if the order doesn't fill by the deadline.

Implementation Shortfall (IS) per share:
    IS = arrival_mid - avg_fill_price
    - IS < 0: filled above arrival mid (passive fill earned the spread)
    - IS > 0: filled below arrival mid (forced-out / walked through bids)

Objective: minimize CVaR_{alpha}(IS) — expected IS in the worst alpha-tail.

Fill model (three triggers, checked in order each step):
1. Trade-through: if future_mid[t] >= limit_price, the market has traded
   through the limit — immediate passive fill at limit_price.
2. Depth-cleared: if the ask depth at our level has fallen to
   <= (1 - fill_ratio) * arrival_depth, we are likely at the front of
   the queue — passive fill at limit_price.
   fill_ratio (default 0.8) = fraction of arrival depth that must be
   consumed to trigger this signal.
3. Force-out: if still unfilled after H steps, sell at market on the
   last snapshot.

Level-index tracking: LOB levels are quoted relative to current mid.
As mid drifts, the same absolute limit price maps to a different level
index — we remap each step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cost import book_walk_cost
from .rollouts import RolloutSource


@dataclass
class LimitSellOutcome:
    is_per_share: float   # IS = arrival_mid - avg_fill_price (lower = better)
    status:       str     # "market" | "passive" | "forced"
    fill_step:    int     # step at which fill occurred (0 = immediate market)


def simulate_limit_sell(
    *,
    Q:              float,
    offset_level:   int,          # d >= 0: levels above mid; -1 = market
    arrival_mid:    float,
    arrival_queue:  np.ndarray,   # (2k,) raw absolute sizes at arrival
    future_mid:     np.ndarray,   # (H,) raw USDT mid-prices
    future_queue:   np.ndarray,   # (H, 2k) raw absolute sizes
    price_level:    float,
    levels:         int,
    fill_ratio:     float = 0.8,  # fraction of arrival depth consumed → "at front"
) -> LimitSellOutcome:
    """Simulate one limit-sell order on a single forward path.

    Returns LimitSellOutcome with IS per share and status.
    """
    H = int(future_mid.shape[0])
    k = int(levels)

    # Market order: immediate fill by walking bids.
    if offset_level < 0:
        proceeds = book_walk_cost(Q, arrival_mid, arrival_queue, "sell", price_level, levels)
        return LimitSellOutcome(arrival_mid - proceeds / Q, "market", 0)

    d = int(offset_level)
    limit_price = float(arrival_mid) + (d + 0.5) * float(price_level)

    # Depth-cleared trigger: arrival depth at level d.
    if d < k:
        D0 = float(arrival_queue[k + d])
        depth_threshold = (1.0 - float(fill_ratio)) * D0
    else:
        D0 = 0.0
        depth_threshold = 0.0  # level outside book — no depletion signal

    prev_j = d

    for t in range(H):
        cur_mid = float(future_mid[t])

        # 1. Trade-through: mid has risen to/past our limit price.
        if cur_mid >= limit_price:
            return LimitSellOutcome(arrival_mid - limit_price, "passive", t + 1)

        # 2. Level remapping: find which ask-level index corresponds to limit_price.
        j_t = int(round((limit_price - cur_mid) / float(price_level) - 0.5))
        if j_t < 0:
            # Limit price is now at or below mid — treat as filled.
            return LimitSellOutcome(arrival_mid - limit_price, "passive", t + 1)
        if j_t >= k:
            # Limit price drifted outside tracked book — no depletion info.
            prev_j = j_t
            continue

        # 3. Depth-cleared: check if arrival depth has been largely consumed.
        if j_t == prev_j and D0 > 0.0:
            cur_q = float(future_queue[t, k + j_t])
            if cur_q <= depth_threshold:
                return LimitSellOutcome(arrival_mid - limit_price, "passive", t + 1)

        prev_j = j_t

    # Force-out: sell at market on last snapshot.
    proceeds = book_walk_cost(Q, float(future_mid[-1]), future_queue[-1], "sell", price_level, levels)
    return LimitSellOutcome(arrival_mid - proceeds / Q, "forced", H)


def cvar_upper(values: np.ndarray, alpha: float = 0.95) -> float:
    """Mean of the top (1-alpha) fraction of ``values`` (upper-tail CVaR).

    For IS where lower is better, CVaR_alpha measures the expected worst-case IS.
    """
    values = np.asarray(values, dtype=np.float64)
    cutoff = np.quantile(values, alpha)
    tail = values[values >= cutoff]
    return float(tail.mean()) if len(tail) > 0 else float(cutoff)


def select_offset(
    *,
    Q:              float,
    offset_grid:    List[int],
    arrival_mid:    float,
    arrival_queue:  np.ndarray,     # (2k,)
    rollouts_mid:   np.ndarray,     # (K, H) raw USDT
    rollouts_queue: np.ndarray,     # (K, H, 2k) raw absolute
    price_level:    float,
    levels:         int,
    objective:      str   = "cvar",
    alpha:          float = 0.95,
    fill_ratio:     float = 0.8,
) -> Tuple[int, Dict[int, np.ndarray]]:
    """Choose the offset from offset_grid that minimizes the objective.

    Returns
    -------
    best_offset : int
    is_by_offset : {offset: (K,) IS array}
    """
    K = rollouts_mid.shape[0]
    is_by_offset: Dict[int, np.ndarray] = {}

    for d in offset_grid:
        is_vals = np.empty(K, dtype=np.float64)
        for ki in range(K):
            outcome = simulate_limit_sell(
                Q             = Q,
                offset_level  = d,
                arrival_mid   = arrival_mid,
                arrival_queue = arrival_queue,
                future_mid    = rollouts_mid[ki],
                future_queue  = rollouts_queue[ki],
                price_level   = price_level,
                levels        = levels,
                fill_ratio    = fill_ratio,
            )
            is_vals[ki] = outcome.is_per_share
        is_by_offset[d] = is_vals

    def score(d):
        if objective == "cvar":
            return cvar_upper(is_by_offset[d], alpha)
        elif objective == "mean":
            return float(is_by_offset[d].mean())
        else:
            raise ValueError(f"Unknown objective: {objective!r}")

    best_offset = min(offset_grid, key=score)
    return best_offset, is_by_offset


def run_limit_order_benchmark(
    *,
    windows:      list,                    # List[ExecutionWindow]
    sources:      Dict[str, RolloutSource],
    Q:            float,
    offset_grid:  List[int],
    n_rollouts:   int,
    price_level:  float,
    levels:       int,
    objective:    str   = "cvar",
    alpha:        float = 0.95,
    fill_ratio:   float = 0.8,
    add_static:   bool  = True,
    add_oracle:   bool  = True,
    progress:     bool  = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run the limit-order CVaR benchmark across all windows and sources.

    For each source, at each window:
      1. Draw K rollouts from the source (conditioned on prefix history).
      2. Select the best offset via ``select_offset``.
      3. Evaluate the CHOSEN offset on the REAL future (single path).
      4. Record: selected_offset, realized IS, status.

    Static baselines (always added if add_static=True):
        "Market"    : offset = -1 (immediate market sell)
        "Touch"     : offset = 0 (best bid = touch)
        "Passive"   : offset = max(offset_grid) (deepest limit)

    Oracle baseline (add_oracle=True):
        Uses the realized future as the single rollout — defines best possible IS.

    Returns
    -------
    {source_name: {"selected_offset": (W,), "realized_is": (W,), "status": (W, str)}}
    """
    from .benchmark import ExecutionWindow

    W = len(windows)
    results: Dict[str, Dict[str, list]] = {}

    def _init(name):
        results[name] = {"selected_offset": [], "realized_is": [], "status": []}

    if add_static:
        for label in ("Market", "Touch", "Passive"):
            _init(label)
    if add_oracle:
        _init("Oracle")
    for name in sources:
        _init(name)

    iterator = range(W)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Limit-order benchmark")
        except ImportError:
            pass

    for i in iterator:
        win = windows[i]
        T = len(win.real_mid)
        arrival_mid   = float(win.real_mid[0])
        arrival_queue = win.real_queue_raw[0]

        history0 = {
            "queue_norm": win.prefix_queue_norm,
            "anchors":    win.prefix_anchors,
        }

        def _realized_is(chosen_offset: int) -> Tuple[float, str]:
            outcome = simulate_limit_sell(
                Q             = Q,
                offset_level  = chosen_offset,
                arrival_mid   = arrival_mid,
                arrival_queue = arrival_queue,
                future_mid    = win.real_mid,
                future_queue  = win.real_queue_raw,
                price_level   = price_level,
                levels        = levels,
                fill_ratio    = fill_ratio,
            )
            return outcome.is_per_share, outcome.status

        if add_static:
            for label, d in [("Market", -1), ("Touch", 0), ("Passive", max(offset_grid))]:
                is_val, status = _realized_is(d)
                results[label]["selected_offset"].append(d)
                results[label]["realized_is"].append(is_val)
                results[label]["status"].append(status)

        if add_oracle:
            oracle_mid   = win.real_mid[None, :]          # (1, T)
            oracle_queue = win.real_queue_raw[None, :, :]  # (1, T, 2k)
            best_d, _ = select_offset(
                Q              = Q,
                offset_grid    = offset_grid,
                arrival_mid    = arrival_mid,
                arrival_queue  = arrival_queue,
                rollouts_mid   = oracle_mid,
                rollouts_queue = oracle_queue,
                price_level    = price_level,
                levels         = levels,
                objective      = objective,
                alpha          = alpha,
                fill_ratio     = fill_ratio,
            )
            is_val, status = _realized_is(best_d)
            results["Oracle"]["selected_offset"].append(best_d)
            results["Oracle"]["realized_is"].append(is_val)
            results["Oracle"]["status"].append(status)

        for name, src in sources.items():
            mid_paths, queue_paths = src.sample(
                t       = 0,
                history = history0,
                K       = n_rollouts,
                H       = T,
            )
            best_d, _ = select_offset(
                Q              = Q,
                offset_grid    = offset_grid,
                arrival_mid    = arrival_mid,
                arrival_queue  = arrival_queue,
                rollouts_mid   = mid_paths,
                rollouts_queue = queue_paths,
                price_level    = price_level,
                levels         = levels,
                objective      = objective,
                alpha          = alpha,
                fill_ratio     = fill_ratio,
            )
            is_val, status = _realized_is(best_d)
            results[name]["selected_offset"].append(best_d)
            results[name]["realized_is"].append(is_val)
            results[name]["status"].append(status)

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for name, d in results.items():
        out[name] = {
            "selected_offset": np.asarray(d["selected_offset"], dtype=np.int32),
            "realized_is":     np.asarray(d["realized_is"],     dtype=np.float64),
            "status":          np.asarray(d["status"],          dtype=object),
        }
    return out


def summarize_is(
    results:     Dict[str, Dict[str, np.ndarray]],
    alpha:       float = 0.95,
    n_bootstrap: int   = 1000,
    rng:         Optional[np.random.Generator] = None,
) -> "pd.DataFrame":
    """Summarize realized IS across sources with bootstrap CIs.

    Returns a DataFrame indexed by source with columns:
        mean_is, cvar_is, ci_lo_mean, ci_hi_mean, ci_lo_cvar, ci_hi_cvar,
        pct_passive, pct_forced, pct_market, n_windows
    Sorted by cvar_is ascending.
    """
    import pandas as pd
    rng = rng if rng is not None else np.random.default_rng(0)

    rows = []
    for name, d in results.items():
        is_arr = d["realized_is"]
        status = d["status"]
        n = len(is_arr)

        cvar_val = cvar_upper(is_arr, alpha)
        mean_val = float(is_arr.mean())

        boot_mean = np.empty(n_bootstrap)
        boot_cvar = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_mean[b] = is_arr[idx].mean()
            boot_cvar[b] = cvar_upper(is_arr[idx], alpha)

        lo_mean, hi_mean = np.quantile(boot_mean, [0.025, 0.975])
        lo_cvar, hi_cvar = np.quantile(boot_cvar, [0.025, 0.975])

        rows.append({
            "source":       name,
            "mean_is":      mean_val,
            "cvar_is":      cvar_val,
            "ci_lo_mean":   float(lo_mean),
            "ci_hi_mean":   float(hi_mean),
            "ci_lo_cvar":   float(lo_cvar),
            "ci_hi_cvar":   float(hi_cvar),
            "pct_passive":  float((status == "passive").mean()),
            "pct_forced":   float((status == "forced").mean()),
            "pct_market":   float((status == "market").mean()),
            "n_windows":    n,
        })

    df = pd.DataFrame(rows).set_index("source").sort_values("cvar_is")
    return df

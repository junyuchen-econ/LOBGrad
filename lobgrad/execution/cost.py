"""
cost.py
-------
Book-walking execution cost function.

Given a simulated (mid_price, queue_sizes) path and a trading schedule
{x_t}, computes the total execution cost by walking the simulated order book
at each time step.

Layout convention (matches BenchmarkConfig)
--------------------------------------------
  queue[0  .. k-1] : bid side (raw absolute sizes, ≥ 0), shallow → deep
  queue[k  .. 2k-1]: ask side (raw absolute sizes, ≥ 0), shallow → deep

  ask level j (j=0..k-1): price = mid_t + (j+0.5) * price_level
  bid level j (j=0..k-1): price = mid_t - (j+0.5) * price_level

Note: mid_t here is the continuous mid-price, not the integer anchor.
"""

from __future__ import annotations

import numpy as np


def book_walk_cost(
    shares:      float,
    mid:         float,
    queue:       np.ndarray,   # (2k,) raw absolute queue sizes
    side:        str,          # "sell" or "buy"
    price_level: float,
    levels:      int,
) -> float:
    """
    Compute the execution cost of filling `shares` against the simulated book.

    For a sell order we walk the bid side (best bid first).
    For a buy  order we walk the ask side (best ask first).

    Parameters
    ----------
    shares      : number of shares to fill at this step (may be fractional)
    mid         : mid-price at this step (USDT)
    queue       : (2k,) raw absolute queue sizes
    side        : "sell" (hit bids) or "buy" (lift asks)
    price_level : tick size δ (USDT)
    levels      : k

    Returns
    -------
    cost : total USDT cost of executing `shares`
           For a sell: cost is negative (proceeds received) — returned as positive
           For a buy : cost is positive (cash spent)
    """
    if shares <= 0:
        return 0.0

    k = levels
    remaining = float(shares)
    total_cost = 0.0

    if side == "sell":
        # Walk bid side: bid level j has price mid - (j+0.5)*δ
        side_queue = queue[:k]
        for j in range(k):
            level_price = mid - (j + 0.5) * price_level
            available   = float(side_queue[j])
            filled      = min(remaining, available)
            total_cost += filled * level_price
            remaining  -= filled
            if remaining <= 1e-8:
                break
    else:
        # Walk ask side: ask level j has price mid + (j+0.5)*δ
        side_queue = queue[k:]
        for j in range(k):
            level_price = mid + (j + 0.5) * price_level
            available   = float(side_queue[j])
            filled      = min(remaining, available)
            total_cost += filled * level_price
            remaining  -= filled
            if remaining <= 1e-8:
                break

    # Any remaining shares that couldn't be filled are executed at the
    # worst observed level price (conservative assumption).
    if remaining > 1e-8:
        worst_price = (
            mid - (k - 0.5) * price_level if side == "sell"
            else mid + (k - 0.5) * price_level
        )
        total_cost += remaining * worst_price

    return float(total_cost)


def schedule_cost(
    schedule:    np.ndarray,   # (horizon,) shares to trade at each step
    mid_path:    np.ndarray,   # (horizon,) mid-prices
    queue_path:  np.ndarray,   # (horizon, 2k) raw absolute queue sizes
    side:        str,
    price_level: float,
    levels:      int,
) -> float:
    """
    Total execution cost of a trading schedule on a single simulated path.

    Parameters
    ----------
    schedule   : (horizon,) array of shares to trade at each step
    mid_path   : (horizon,) mid-prices along the path
    queue_path : (horizon, 2k) raw absolute queue sizes along the path
    side       : "sell" or "buy"
    price_level: tick size δ
    levels     : k

    Returns
    -------
    total_cost : scalar USDT cost (always positive; for sell = total proceeds)
    """
    horizon = len(schedule)
    total   = 0.0
    for t in range(horizon):
        total += book_walk_cost(
            shares      = schedule[t],
            mid         = mid_path[t],
            queue       = queue_path[t],
            side        = side,
            price_level = price_level,
            levels      = levels,
        )
    return total


def implementation_shortfall(
    schedule:    np.ndarray,   # (horizon,)
    mid_path:    np.ndarray,   # (horizon,)
    queue_path:  np.ndarray,   # (horizon, 2k)
    arrival_mid: float,        # mid-price at order arrival (t=0)
    side:        str,
    price_level: float,
    levels:      int,
) -> float:
    """
    Implementation shortfall = (ideal cost) - (actual cost).

    For a sell order:
      ideal cost  = total_shares * arrival_mid
      actual cost = sum of fills at walked-book prices

    IS measures how much worse the execution was vs. trading everything
    at the arrival price.  Higher IS = worse execution.

    Returns
    -------
    is_usdt : implementation shortfall in USDT (positive = worse than ideal)
    """
    total_shares = float(schedule.sum())
    ideal = total_shares * arrival_mid
    actual = schedule_cost(schedule, mid_path, queue_path, side, price_level, levels)

    if side == "sell":
        return float(ideal - actual)   # ideal proceeds - actual proceeds
    else:
        return float(actual - ideal)   # actual spend - ideal spend

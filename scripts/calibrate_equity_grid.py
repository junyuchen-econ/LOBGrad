"""
calibrate_equity_grid.py
------------------------
Pick per-symbol grid parameters for the equity Table-1 run:

  * price_level (delta)  — cent width of each queue bin. Chosen so d*delta
    (d = levels = 5) covers ~98% of intraday 5-second |Δmid| moves, mirroring
    the paper's Figure-4 criterion (the grid should track the mid so new best
    levels rarely fall outside it).
  * delta_mid_clip       — the return clip, ~ the 99.9th percentile of intraday
    |Δmid| (cents).

Stats are computed on the TRAIN split, per day (overnight gaps excluded), by
reading the decoded JSONL directly (prices already in integer cents). Writes
``data2/processed/calibration_{SYMBOL}.json`` consumed by equity_profiles.py.

Usage:
    python scripts/calibrate_equity_grid.py --symbol MSFT
    python scripts/calibrate_equity_grid.py --all
    python scripts/calibrate_equity_grid.py --symbol GOOGL --price-level 2 --delta-mid-clip 40
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from equity_profiles import PROC_DIR, SPLITS, SYMBOLS, DATA, calibration_path


def _date_range(start: str, end: str):
    d0 = datetime.strptime(start, "%Y-%m-%d")
    d1 = datetime.strptime(end, "%Y-%m-%d")
    out, cur = [], d0
    while cur <= d1:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def _collect(symbol: str):
    """Return (intraday |Δanchor| cents, per-side book half-span cents,
    per-snapshot #levels) over the TRAIN split."""
    d_abs, halfspan, nlev = [], [], []
    files = 0
    for dt in _date_range(*SPLITS["train"]):
        fp = PROC_DIR / f"{symbol}_{dt.month:02d}-{dt.day:02d}.json"
        if not fp.exists():
            continue
        files += 1
        anchors = []
        with open(fp) as fh:
            for line in fh:
                d = json.loads(line)
                bids, asks = d["bids"], d["asks"]
                bb, ba = bids[0][0], asks[0][0]
                mid = (bb + ba) / 2.0
                anchors.append(math.ceil(mid))
                halfspan.append(max(mid - bids[-1][0], asks[-1][0] - mid))
                nlev.append(min(len(bids), len(asks)))
        a = np.asarray(anchors, dtype=np.float64)
        if len(a) > 1:
            d_abs.append(np.abs(np.diff(a)))   # intraday only (no cross-day diff)
    if not d_abs:
        raise FileNotFoundError(f"No train files for {symbol} in {PROC_DIR}")
    return (np.concatenate(d_abs), np.asarray(halfspan), np.asarray(nlev), files)


def calibrate(symbol: str, price_level=None, delta_mid_clip=None) -> dict:
    dabs, halfspan, nlev, files = _collect(symbol)
    d = int(DATA["levels"])

    qs = {q: float(np.quantile(dabs, q)) for q in (0.5, 0.9, 0.95, 0.98, 0.99, 0.999)}
    p98 = qs[0.98]

    if price_level is None:
        price_level = max(1, int(math.ceil(p98 / d)))       # d*delta >= p98(|Δmid|)
    if delta_mid_clip is None:
        delta_mid_clip = max(price_level * d, int(math.ceil(qs[0.999])))

    half = d * price_level  # grid half-width in cents
    move_cov = float(np.mean(dabs <= half))
    book_cov = float(np.mean(halfspan <= half))    # fraction of snapshots whose full book fits the grid

    stats = {
        "symbol":          symbol,
        "price_level":     int(price_level),
        "delta_mid_clip":  int(delta_mid_clip),
        "train_files":     files,
        "n_delta":         int(len(dabs)),
        "dmid_abs_quantiles_cents": {str(k): round(v, 3) for k, v in qs.items()},
        "book_halfspan_p50_cents":  round(float(np.median(halfspan)), 2),
        "book_halfspan_p90_cents":  round(float(np.quantile(halfspan, 0.9)), 2),
        "mean_levels_per_side":     round(float(np.mean(nlev)), 2),
        "grid_halfwidth_cents":     int(half),
        "move_coverage_at_grid":    round(move_cov, 4),
        "book_coverage_at_grid":    round(book_cov, 4),
    }
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", choices=SYMBOLS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--price-level", type=int, default=None, help="override delta (cents)")
    ap.add_argument("--delta-mid-clip", type=int, default=None, help="override rmax (cents)")
    args = ap.parse_args()

    targets = SYMBOLS if args.all else ([args.symbol] if args.symbol else None)
    if not targets:
        ap.error("pass --symbol SYM or --all")

    for sym in targets:
        stats = calibrate(sym, args.price_level, args.delta_mid_clip)
        calibration_path(sym).write_text(json.dumps(stats, indent=2))
        print(f"\n=== {sym} ===")
        for k, v in stats.items():
            print(f"  {k:26s}: {v}")
        print(f"  -> wrote {calibration_path(sym)}")


if __name__ == "__main__":
    main()

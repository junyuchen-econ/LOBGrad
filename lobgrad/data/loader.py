"""
loader.py
---------
Reads BTCUSDT JSON snapshot files and converts them into the (2k+1)-dimensional
signed-queue LOB representation used throughout the benchmark.

Layout (per snapshot):
  index 0      : anchor price  (raw USDT, integer = ceil(mid_price))
  indices 1..k : bid signed queues, shallow→deep  (raw, ≤ 0)
                 index 1 = best bid at anchor - 1*δ
                 index k = deepest bid at anchor - k*δ
  indices k+1..2k: ask signed queues, shallow→deep (raw, ≥ 0)
                 index k+1 = best ask at anchor
                 index 2k  = deepest ask at anchor + (k-1)*δ

The Normalizer (normalizer.py) applies Z-score to the anchor and
sign(Q)*log(|Q|+1) to the queue dimensions.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Core centering logic  (mirrors TimeGrad's LOB_Dataset.center())
# ---------------------------------------------------------------------------

def _center_signed_snapshot(
    bids: List[List[float]],
    asks: List[List[float]],
    levels: int,
    price_level: float,
) -> Tuple[float, np.ndarray]:
    """
    Convert raw bids/asks into a signed queue vector + anchor price.

    Parameters
    ----------
    bids, asks  : list of [price, qty] pairs, best price first
    levels      : number of levels per side (k)
    price_level : tick size δ in price units

    Returns
    -------
    anchor  : float  — integer ceil(mid_price), used as grid origin
    signed  : (2k,)  — bid queues (≤ 0) then ask queues (≥ 0), shallow→deep

    Grid:
      bid level j (j=1..k): price = anchor - j * δ   → signed[j-1]
      ask level j (j=1..k): price = anchor + (j-1)*δ → signed[k+j-1]
    """
    bids_arr = np.array(bids, dtype=float)
    asks_arr = np.array(asks, dtype=float)

    best_bid  = np.floor(bids_arr[0, 0])
    best_ask  = np.ceil(asks_arr[0, 0])
    mid_price = (best_bid + best_ask) / 2.0
    anchor    = float(np.ceil(mid_price))

    bid_prices = np.floor(bids_arr[:, 0])
    ask_prices = np.ceil(asks_arr[:, 0])

    signed = np.zeros(2 * levels, dtype=np.float32)

    for j in range(levels):
        # Bid bin j: aggregate all volume with floor(price) in (anchor-(j+1)*δ, anchor-j*δ]
        lo = anchor - (j + 1) * price_level
        hi = anchor - j * price_level
        signed[j] = -float(np.sum(bids_arr[:, 1][(bid_prices > lo) & (bid_prices <= hi)]))

        # Ask bin j: aggregate all volume with ceil(price) in [anchor+j*δ, anchor+(j+1)*δ)
        lo = anchor + j * price_level
        hi = anchor + (j + 1) * price_level
        signed[levels + j] = float(np.sum(asks_arr[:, 1][(ask_prices >= lo) & (ask_prices < hi)]))

    return anchor, signed


def _center_snapshot(
    bids: List[List[float]],
    asks: List[List[float]],
    levels: int,
    price_level: float,
) -> Tuple[float, np.ndarray]:
    """
    Convert raw bids/asks into a centered volume vector + mid-price.

    Parameters
    ----------
    bids, asks : list of [price, qty] pairs, best price first
    levels     : number of levels on each side
    price_level: bin width in price units

    Returns
    -------
    mid_price   : float
    volumes     : (2*levels,)  bid vols first (non-negative), then ask vols
    """
    bids_arr = np.array(bids, dtype=float)
    asks_arr = np.array(asks, dtype=float)

    best_bid = np.floor(bids_arr[0, 0])
    best_ask = np.ceil(asks_arr[0, 0])
    mid_price = (best_bid + best_ask) / 2.0
    centered_mid = np.ceil(mid_price)

    volumes = np.zeros(2 * levels, dtype=np.float32)

    bid_prices = np.floor(bids_arr[:, 0])
    ask_prices = np.ceil(asks_arr[:, 0])

    for i in range(levels):
        # Bid bin i: floor(price) in (centered_mid-(i+1)*δ, centered_mid-i*δ]
        lo = centered_mid - (i + 1) * price_level
        hi = centered_mid - i * price_level
        volumes[i] = np.sum(bids_arr[:, 1][(bid_prices > lo) & (bid_prices <= hi)])

        # Ask bin i: ceil(price) in [centered_mid+i*δ, centered_mid+(i+1)*δ)
        lo = centered_mid + i * price_level
        hi = centered_mid + (i + 1) * price_level
        volumes[levels + i] = np.sum(asks_arr[:, 1][(ask_prices >= lo) & (ask_prices < hi)])

    return float(mid_price), volumes


# ---------------------------------------------------------------------------
# Signed queue loader
# ---------------------------------------------------------------------------

def load_day_signed(
    filepath: str,
    levels: int,
    price_level: float,
    time_diff_ms: int,
    depth: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one JSON snapshot file using the signed queue representation.

    Returns
    -------
    timestamps : (T,)      Unix timestamps in milliseconds
    snapshots  : (T, 2k)   signed queue features — bid queues (≤0) then ask queues (≥0)
    anchors    : (T,)      raw anchor prices (ceil(mid_price)) in USDT
    """
    timestamps = []
    snapshots  = []
    anchors    = []
    last_ts    = -1

    with open(filepath, "r") as fh:
        for line in fh:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = entry["timestamp"]

            if ts - last_ts < time_diff_ms:
                continue
            last_ts = ts

            bids = entry["bids"][:depth]
            asks = entry["asks"][:depth]
            if len(bids) == 0 or len(asks) == 0:
                continue

            anchor, signed = _center_signed_snapshot(bids, asks, levels, price_level)

            timestamps.append(ts)
            snapshots.append(signed)
            anchors.append(anchor)

    return (
        np.array(timestamps, dtype=np.int64),
        np.array(snapshots,  dtype=np.float32),
        np.array(anchors,    dtype=np.float32),
    )


def load_days_signed(
    data_dir: str,
    symbol: str,
    dates: List[str],
    levels: int,
    price_level: float,
    time_diff_ms: int,
    depth: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and concatenate multiple days using the signed queue representation."""
    all_ts      = []
    all_snaps   = []
    all_anchors = []

    for date_str in dates:
        dt    = datetime.strptime(date_str, "%Y-%m-%d")
        fname = f"{symbol}_{dt.month:02d}-{dt.day:02d}.json"
        fpath = os.path.join(data_dir, fname)

        if not os.path.exists(fpath):
            continue

        ts, snaps, anchs = load_day_signed(fpath, levels, price_level, time_diff_ms, depth)
        if len(ts) == 0:
            continue

        all_ts.append(ts)
        all_snaps.append(snaps)
        all_anchors.append(anchs)

    if len(all_ts) == 0:
        raise FileNotFoundError(
            f"No data found in {data_dir} for dates {dates}"
        )

    return (
        np.concatenate(all_ts),
        np.concatenate(all_snaps,   axis=0),
        np.concatenate(all_anchors, axis=0),
    )


def load_split_signed(
    data_dir: str,
    symbol: str,
    start: str,
    end: str,
    levels: int,
    price_level: float,
    time_diff_ms: int,
    depth: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all days in [start, end] inclusive using the signed queue representation."""
    dates = _date_range(start, end)
    return load_days_signed(data_dir, symbol, dates, levels, price_level, time_diff_ms, depth)


def load_day_unsigned(
    filepath: str,
    levels: int,
    price_level: float,
    time_diff_ms: int,
    depth: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one JSON snapshot file using the unsigned queue representation.

    Both bid and ask volumes are stored as non-negative values.
    Direction is encoded by position (indices 0..k-1 = bids, k..2k-1 = asks).

    Returns
    -------
    timestamps : (T,)     Unix timestamps in milliseconds
    snapshots  : (T, 2k)  unsigned queue volumes (all ≥ 0)
    anchors    : (T,)     mid-price in USDT
    """
    timestamps = []
    snapshots  = []
    anchors    = []
    last_ts    = -1

    with open(filepath, "r") as fh:
        for line in fh:
            entry = json.loads(line)
            ts = entry["timestamp"]

            if ts - last_ts < time_diff_ms:
                continue
            last_ts = ts

            bids = entry["bids"][:depth]
            asks = entry["asks"][:depth]
            if len(bids) == 0 or len(asks) == 0:
                continue

            mid, vols = _center_snapshot(bids, asks, levels, price_level)

            timestamps.append(ts)
            snapshots.append(vols)   # (2k,) all non-negative
            anchors.append(mid)

    return (
        np.array(timestamps, dtype=np.int64),
        np.array(snapshots,  dtype=np.float32),
        np.array(anchors,    dtype=np.float32),
    )


def load_days_unsigned(
    data_dir: str,
    symbol: str,
    dates: List[str],
    levels: int,
    price_level: float,
    time_diff_ms: int,
    depth: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and concatenate multiple days using the unsigned queue representation."""
    all_ts      = []
    all_snaps   = []
    all_anchors = []

    for date_str in dates:
        dt    = datetime.strptime(date_str, "%Y-%m-%d")
        fname = f"{symbol}_{dt.month:02d}-{dt.day:02d}.json"
        fpath = os.path.join(data_dir, fname)

        if not os.path.exists(fpath):
            continue

        ts, snaps, anchs = load_day_unsigned(fpath, levels, price_level, time_diff_ms, depth)
        if len(ts) == 0:
            continue

        all_ts.append(ts)
        all_snaps.append(snaps)
        all_anchors.append(anchs)

    if len(all_ts) == 0:
        raise FileNotFoundError(
            f"No data found in {data_dir} for dates {dates}"
        )

    return (
        np.concatenate(all_ts),
        np.concatenate(all_snaps,   axis=0),
        np.concatenate(all_anchors, axis=0),
    )


def load_split_unsigned(
    data_dir: str,
    symbol: str,
    start: str,
    end: str,
    levels: int,
    price_level: float,
    time_diff_ms: int,
    depth: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all days in [start, end] inclusive using the unsigned queue representation."""
    dates = _date_range(start, end)
    return load_days_unsigned(data_dir, symbol, dates, levels, price_level, time_diff_ms, depth)


def _date_range(start: str, end: str) -> List[str]:
    d0 = datetime.strptime(start, "%Y-%m-%d")
    d1 = datetime.strptime(end,   "%Y-%m-%d")
    out = []
    current = d0
    while current <= d1:
        out.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return out

"""
splits.py
---------
Utilities for extracting sliding context/horizon windows from a snapshot array,
and for building the train/val/test splits from a BenchmarkConfig.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..config import BenchmarkConfig
from .loader import load_split_signed, load_split_unsigned
from .normalizer import Normalizer


def date_splits(cfg: BenchmarkConfig) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Normalizer,
]:
    """
    Load raw data for each split, normalise features, and return splits + Normalizer.

    The loader and normaliser are selected based on cfg.queue_repr:
      "signed"   – bids stored as negative values, sign(Q)*log(|Q|+1) normalisation
      "unsigned" – all volumes non-negative,       log(Q+1) normalisation

    Returns
    -------
    (ts_train, X_train, anchors_train),
    (ts_val,   X_val,   anchors_val),
    (ts_test,  X_test,  anchors_test),
    normalizer

    Each X is shape (T, 2k) normalised float32.
    Each anchors is shape (T,) raw USDT float32.
    """
    kwargs = dict(
        data_dir     = cfg.data_dir,
        symbol       = cfg.symbol,
        levels       = cfg.levels,
        price_level  = cfg.price_level,
        time_diff_ms = cfg.time_diff_ms,
        depth        = cfg.depth,
    )

    loader = load_split_signed if cfg.queue_repr == "signed" else load_split_unsigned

    ts_train, X_train_raw, anchors_train = loader(
        start=cfg.train_start, end=cfg.train_end, **kwargs)
    ts_val, X_val_raw, anchors_val = loader(
        start=cfg.val_start, end=cfg.val_end, **kwargs)
    ts_test, X_test_raw, anchors_test = loader(
        start=cfg.test_start, end=cfg.test_end, **kwargs)

    norm = Normalizer(levels=cfg.levels, mode=cfg.queue_repr).fit(X_train_raw)

    X_train = norm.transform(X_train_raw)
    X_val   = norm.transform(X_val_raw)
    X_test  = norm.transform(X_test_raw)

    return (
        (ts_train, X_train, anchors_train),
        (ts_val,   X_val,   anchors_val),
        (ts_test,  X_test,  anchors_test),
        norm,
    )


def date_splits_joint(cfg: BenchmarkConfig) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Normalizer,
]:
    """
    Like date_splits() but retains anchor (mid-price) arrays for joint modelling.

    Returns
    -------
    (ts_train, X_train, anchors_train),
    (ts_val,   X_val,   anchors_val),
    (ts_test,  X_test,  anchors_test),
    normalizer

    anchors_* : (T,) raw USDT anchor prices (ceil(mid_price) at each snapshot).
    X_*       : (T, 2k) normalised queue features (same as date_splits).
    """
    return date_splits(cfg)   # date_splits already returns anchors as third element


def sample_eval_windows_joint(
    snapshots:   np.ndarray,   # (T, 2k)
    anchors:     np.ndarray,   # (T,)
    context_len: int,
    horizon:     int,
    n_windows:   int,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Like sample_eval_windows() but also returns anchor sub-arrays.

    Returns
    -------
    ctx_queue    : (n_windows, context_len, 2k)
    fut_queue    : (n_windows, horizon, 2k)
    ctx_anchors  : (n_windows, context_len)
    fut_anchors  : (n_windows, horizon)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    T         = len(snapshots)
    window    = context_len + horizon
    max_start = T - window

    if max_start < n_windows:
        raise ValueError(
            f"Not enough snapshots ({T}) for {n_windows} windows of length {window}."
        )

    starts = np.sort(rng.choice(max_start + 1, size=n_windows, replace=False))

    ctx_q, fut_q, ctx_a, fut_a = [], [], [], []
    for s in starts:
        ctx_q.append(snapshots[s          : s + context_len])
        fut_q.append(snapshots[s + context_len : s + window])
        ctx_a.append(anchors  [s          : s + context_len])
        fut_a.append(anchors  [s + context_len : s + window])

    return (
        np.stack(ctx_q), np.stack(fut_q),
        np.stack(ctx_a), np.stack(fut_a),
    )


def sample_eval_windows(
    snapshots:   np.ndarray,
    context_len: int,
    horizon:     int,
    n_windows:   int,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly sample n_windows (context, future) pairs.

    Returns
    -------
    contexts : (n_windows, context_len, 2k)
    futures  : (n_windows, horizon, 2k)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    T         = len(snapshots)
    window    = context_len + horizon
    max_start = T - window

    if max_start < n_windows:
        raise ValueError(
            f"Not enough test snapshots ({T}) for {n_windows} windows "
            f"of length {window}."
        )

    starts = np.sort(rng.choice(max_start + 1, size=n_windows, replace=False))

    contexts_list = []
    futures_list  = []

    for s in starts:
        full_snaps = snapshots[s : s + window]
        contexts_list.append(full_snaps[:context_len])
        futures_list .append(full_snaps[context_len:])

    return np.stack(contexts_list), np.stack(futures_list)

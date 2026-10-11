"""
evaluate.py
-----------
Cross-window pooled evaluation of generative LOB models on normalised (T, 2k)
feature arrays.

Distributional metrics pool n_windows samples at each horizon step, giving
statistically robust estimates (n_windows real samples vs n_windows*n_paths
generated). Autocorrelation metrics pool per-(window, path) ACF estimates
across all windows. Price metrics require real_prices / gen_prices.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from scipy.stats import wasserstein_distance


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _safe_wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    """Wasserstein-1 between two 1-D samples, normalised by real data scale.

    Returns NaN when the real distribution is near-degenerate (e.g. an
    empty order-book level) so that nanmean silently skips it rather than
    producing artificially inflated values.
    """
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    mu    = a.mean()
    sigma = a.std()
    if sigma < 1e-6:          # near-constant / empty level — skip
        return np.nan
    return float(wasserstein_distance((a - mu) / sigma, (b - mu) / sigma))


def _split_sides(features: np.ndarray, levels: int):
    """
    Split (T, 2k) normalised features into bid and ask magnitudes.
    Works for both signed (bids negative) and unsigned modes.
    """
    bids = np.abs(features[:, :levels])   # (T, k)
    asks = np.abs(features[:, levels:])   # (T, k)
    return bids, asks


def _autocorr_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = x - x.mean()
    var = np.var(x) + 1e-10
    return np.array([np.mean(x[: len(x) - l] * x[l:]) / var for l in range(1, max_lag + 1)])


# ---------------------------------------------------------------------------
# Distributional queue metrics  (real, gen: (T, 2k), levels: int) -> float
# ---------------------------------------------------------------------------

def _l1_volume_metric(real: np.ndarray, gen: np.ndarray, levels: int) -> float:
    """Wasserstein on best-bid and best-ask queue size distributions."""
    b_r, a_r = _split_sides(real, levels)
    b_g, a_g = _split_sides(gen,  levels)
    return float(np.nanmean([
        _safe_wasserstein(b_r[:, 0], b_g[:, 0]),
        _safe_wasserstein(a_r[:, 0], a_g[:, 0]),
    ]))


def _imbalance_metric(real: np.ndarray, gen: np.ndarray, levels: int) -> float:
    """Wasserstein on OFI imbalance = (ask_total - bid_total) / (ask_total + bid_total)."""
    def imbalance(features):
        b, a = _split_sides(features, levels)
        return (a.sum(1) - b.sum(1)) / (a.sum(1) + b.sum(1) + 1e-10)
    return _safe_wasserstein(imbalance(real), imbalance(gen))


def _queue_profile_metric(real: np.ndarray, gen: np.ndarray, levels: int) -> float:
    """
    Normalised MAE between mean volume-vs-depth profiles.
    Captures whether the characteristic decay of queue size with depth is reproduced.
    """
    b_r, a_r = _split_sides(real, levels)
    b_g, a_g = _split_sides(gen,  levels)
    profile_r = np.concatenate([b_r.mean(0), a_r.mean(0)])  # (2k,)
    profile_g = np.concatenate([b_g.mean(0), a_g.mean(0)])
    scale = np.abs(profile_r).mean() + 1e-10
    return float(np.mean(np.abs(profile_r - profile_g)) / scale)


# ---------------------------------------------------------------------------
# CRPS helpers
# ---------------------------------------------------------------------------

def crps_1d(samples: np.ndarray, y: float) -> float:
    """CRPS for a scalar: E|X-y| - 0.5*E|X-X'|.  O(n_paths²) — fine for n≤100."""
    term1 = np.mean(np.abs(samples - y))
    term2 = np.mean(np.abs(samples[:, None] - samples[None, :]))
    return float(term1 - 0.5 * term2)


def crps_snapshot(real_snap: np.ndarray, gen_paths_snap: np.ndarray) -> float:
    """Mean CRPS over all 2k dimensions for one (real, generated-paths) pair.

    Parameters
    ----------
    real_snap      : (2k,)        — one real snapshot
    gen_paths_snap : (n_paths, 2k) — generated samples at the same step
    """
    dim = real_snap.shape[0]
    return float(np.mean([crps_1d(gen_paths_snap[:, d], real_snap[d]) for d in range(dim)]))


# ---------------------------------------------------------------------------
# Cross-window pooled evaluation
# ---------------------------------------------------------------------------

def evaluate_model_pooled(
    real_futures: np.ndarray,
    gen_features: np.ndarray,
    levels:       int = 10,
    real_prices:  np.ndarray | None = None,
    gen_prices:   np.ndarray | None = None,
    max_lag:      int = 5,
) -> Dict[str, Dict]:
    """
    Statistically robust evaluation via cross-window pooling.

    Distributional metrics (Wasserstein / MAE):
      At each horizon step t, pool real_futures[:, t, :] (n_windows samples)
      against gen_features[:, :, t, :] (n_windows * n_paths samples), then
      average the metric over all horizon steps.  This gives n_windows real
      samples per step rather than horizon samples, yielding far lower-variance
      Wasserstein estimates.

    Autocorrelation metrics:
      Compute the ACF independently for each (window, path) pair, then average
      the ACF vectors before taking the MAE.  This pools 200+ estimates per lag
      instead of fitting one noisy ACF from H data points.

    Metrics computed
    ----------------
    Queue:
      crps              — proper scoring rule (per-step, cross-path)
      l1_volume         — Wasserstein on best-bid/ask sizes
      ob_imbalance      — Wasserstein on order-book imbalance
      queue_profile     — normalised MAE of mean depth-vs-level profile
      queue_autocorr    — MAE of pooled L1 ACF at lags 1..max_lag

    Price (only when real_prices / gen_prices are provided):
      price_crps        — CRPS for mid-price (per-step, cross-path)
      return_dist       — Wasserstein on pooled price returns
      abs_return_autocorr — MAE of pooled ACF of |returns| at lags 1..max_lag

    Parameters
    ----------
    real_futures : (n_windows, horizon, 2k)
    gen_features : (n_windows, n_paths, horizon, 2k)
    levels       : k — price levels per side
    real_prices  : (n_windows, horizon) — raw mid-prices (optional)
    gen_prices   : (n_windows, n_paths, horizon) — generated mid-prices (optional)
    max_lag      : lags for ACF metrics
    """
    n_windows, horizon, _ = real_futures.shape
    n_paths = gen_features.shape[1]
    include_price = (real_prices is not None) and (gen_prices is not None)

    results: Dict[str, Dict] = {}

    # ------------------------------------------------------------------ #
    # Distributional metrics: pool across windows at each horizon step    #
    # ------------------------------------------------------------------ #
    DIST_FNS = {
        "l1_volume":    _l1_volume_metric,
        "ob_imbalance": _imbalance_metric,
        "queue_profile": _queue_profile_metric,
    }
    dist_per_step: Dict[str, list] = {k: [] for k in DIST_FNS}

    for t in range(horizon):
        real_t = real_futures[:, t, :]                              # (n_windows, 2k)
        gen_t  = gen_features[:, :, t, :].reshape(-1, 2 * levels)  # (n_windows*n_paths, 2k)
        for name, fn in DIST_FNS.items():
            try:
                dist_per_step[name].append(fn(real_t, gen_t, levels))
            except Exception:
                dist_per_step[name].append(np.nan)

    for name, vals in dist_per_step.items():
        results[name] = {
            "mean":     float(np.nanmean(vals)),
            "std":      float(np.nanstd(vals)),
            "per_step": vals,
        }

    # ------------------------------------------------------------------ #
    # Queue CRPS: per-step, averaged over windows                         #
    # ------------------------------------------------------------------ #
    crps_per_step = []
    for t in range(horizon):
        step_scores = [
            crps_snapshot(real_futures[w, t], gen_features[w, :, t])
            for w in range(n_windows)
        ]
        crps_per_step.append(float(np.nanmean(step_scores)))
    results["crps"] = {
        "mean":     float(np.nanmean(crps_per_step)),
        "std":      float(np.nanstd(crps_per_step)),
        "per_step": crps_per_step,
    }

    # ------------------------------------------------------------------ #
    # Queue autocorrelation: pool ACF estimates across windows & paths    #
    # ------------------------------------------------------------------ #
    real_q_acfs: list = []
    gen_q_acfs:  list = []
    for w in range(n_windows):
        b, a = _split_sides(real_futures[w], levels)
        l1 = b[:, 0] + a[:, 0]
        if len(l1) >= max_lag + 2:
            real_q_acfs.append(_autocorr_1d(l1, max_lag))
        for p in range(n_paths):
            b, a = _split_sides(gen_features[w, p], levels)
            l1 = b[:, 0] + a[:, 0]
            if len(l1) >= max_lag + 2:
                gen_q_acfs.append(_autocorr_1d(l1, max_lag))

    if real_q_acfs and gen_q_acfs:
        real_q_acf = np.mean(real_q_acfs, axis=0)
        gen_q_acf  = np.mean(gen_q_acfs,  axis=0)
        queue_autocorr_val = float(np.mean(np.abs(real_q_acf - gen_q_acf)))
    else:
        queue_autocorr_val = np.nan
    results["queue_autocorr"] = {"mean": queue_autocorr_val, "std": 0.0, "per_step": []}

    # ------------------------------------------------------------------ #
    # Price metrics (optional)                                            #
    # ------------------------------------------------------------------ #
    if include_price:
        # Mid-price CRPS on normalised step-to-step Δmid.
        #
        # Computing CRPS on absolute prices (~30,000 USDT) produces values of
        # ~5–20 USDT, while queue CRPS is ~0.07–0.09 (normalised log units).
        # The ~100–200× scale difference would make price_crps dominate the
        # composite score.  Instead, compute CRPS on Δmid_t normalised by the
        # empirical std of real Δmid across all windows and steps — identical
        # to the _safe_wasserstein normalisation used for distributional metrics.
        #
        # result is dimensionless: ~0.3–1.5, comparable to Wasserstein metrics.
        real_deltas = np.diff(real_prices, axis=1)          # (n_windows, horizon-1)
        delta_sigma = float(real_deltas.std()) + 1e-6        # scalar scale factor

        price_crps_per_step = []
        price_crps_raw_per_step = []
        for t in range(1, horizon):                          # step-to-step Δmid
            real_d_raw = real_deltas[:, t - 1]              # (n_windows,) — USDT
            gen_d_raw  = np.diff(gen_prices[:, :, max(0, t-1):t+1], axis=2)[:, :, 0]
                                                             # (n_windows, n_paths) — USDT
            step_scores_raw = [
                crps_1d(gen_d_raw[w], float(real_d_raw[w]))
                for w in range(n_windows)
            ]
            price_crps_raw_per_step.append(float(np.nanmean(step_scores_raw)))
            price_crps_per_step.append(float(np.nanmean(step_scores_raw)) / delta_sigma)
        results["price_crps"] = {
            "mean":     float(np.nanmean(price_crps_per_step)),
            "std":      float(np.nanstd(price_crps_per_step)),
            "per_step": price_crps_per_step,
        }
        results["price_crps_usdt"] = {
            "mean":     float(np.nanmean(price_crps_raw_per_step)),
            "std":      float(np.nanstd(price_crps_raw_per_step)),
            "per_step": price_crps_raw_per_step,
        }

        # Return distribution: pool all returns across windows and steps
        real_returns = np.diff(real_prices, axis=1).flatten()
        gen_returns  = np.diff(gen_prices,  axis=2).reshape(-1)
        real_returns = real_returns[np.isfinite(real_returns)]
        gen_returns  = gen_returns[np.isfinite(gen_returns)]
        ret_dist_val = (
            _safe_wasserstein(real_returns, gen_returns)
            if len(real_returns) > 1 and len(gen_returns) > 1
            else np.nan
        )
        results["return_dist"] = {"mean": float(ret_dist_val), "std": 0.0, "per_step": []}

        # ACF of |returns|: pool estimates across windows and paths
        real_ar_acfs: list = []
        gen_ar_acfs:  list = []
        for w in range(n_windows):
            abs_ret = np.abs(np.diff(real_prices[w]))
            if len(abs_ret) >= max_lag + 2:
                real_ar_acfs.append(_autocorr_1d(abs_ret, max_lag))
            for p in range(n_paths):
                abs_ret_g = np.abs(np.diff(gen_prices[w, p]))
                if len(abs_ret_g) >= max_lag + 2:
                    gen_ar_acfs.append(_autocorr_1d(abs_ret_g, max_lag))

        if real_ar_acfs and gen_ar_acfs:
            real_ar_acf = np.mean(real_ar_acfs, axis=0)
            gen_ar_acf  = np.mean(gen_ar_acfs,  axis=0)
            abs_ret_autocorr_val = float(np.mean(np.abs(real_ar_acf - gen_ar_acf)))
        else:
            abs_ret_autocorr_val = np.nan
        results["abs_return_autocorr"] = {
            "mean": abs_ret_autocorr_val, "std": 0.0, "per_step": [],
        }

        # Directional accuracy: fraction of windows where the sign of the
        # mean generated end-to-end return matches the real end-to-end return.
        # Range [0, 1]; baseline ≈ 0.5 for random directions. Higher is better.
        real_dir = np.sign(real_prices[:, -1] - real_prices[:, 0])
        gen_dir  = np.sign(
            gen_prices[:, :, -1].mean(axis=1) - gen_prices[:, :, 0].mean(axis=1)
        )
        valid    = real_dir != 0
        if valid.sum() > 0:
            dir_acc_val = float((real_dir[valid] == gen_dir[valid]).mean())
        else:
            dir_acc_val = float("nan")
        results["direction_accuracy"] = {
            "mean": dir_acc_val, "std": 0.0, "per_step": [],
        }

    return results


# ---------------------------------------------------------------------------
# Window-level bootstrap CIs
# ---------------------------------------------------------------------------

def bootstrap_metrics(
    real_futures: np.ndarray,
    gen_features: np.ndarray,
    levels:       int = 10,
    real_prices:  np.ndarray | None = None,
    gen_prices:   np.ndarray | None = None,
    max_lag:      int = 5,
    n_boot:       int = 1000,
    alpha:        float = 0.05,
    metrics:      Iterable[str] | None = None,
    seed:         int = 1234,
    progress:     bool = False,
    n_jobs:       int = 1,
) -> Dict[str, Dict]:
    """
    Window-level percentile bootstrap on top of `evaluate_model_pooled`.

    For each of `n_boot` replicates, resample window indices with replacement
    and recompute the pooled metrics on the slice. Returns the point estimate
    (from the full sample), the (alpha/2, 1-alpha/2) percentile CI, and the
    raw bootstrap samples for every metric.

    Bootstrapping along the window axis quantifies sampling variance from the
    finite eval set (n_windows). It does NOT reflect generator stochasticity
    — paths are reused across replicates.

    Parameters
    ----------
    real_futures, gen_features, levels, real_prices, gen_prices, max_lag
        Same as `evaluate_model_pooled`.
    n_boot
        Number of bootstrap replicates.
    alpha
        Two-sided coverage. e.g. 0.05 → 95% CI.
    metrics
        Restrict bootstrap to these metric names. None → all metrics returned
        by the point evaluation.
    seed
        RNG seed for the resampling. Independent of any path-generation seed.
    progress
        Show a tqdm bar if available.
    n_jobs
        Parallel workers for replicate evaluation. 1 = sequential (default).
        -1 = all cores. Uses joblib's loky backend (process pool) — safe with
        scipy/numpy and avoids the GIL. Each replicate is independent so the
        speedup is near-linear in n_jobs.

    Returns
    -------
    dict[metric_name -> {"mean": float, "ci_lo": float, "ci_hi": float,
                         "samples": list[float]}]
        `mean` is the point estimate on the full eval set, NOT the bootstrap
        mean — this avoids the small-sample bias of resampled means.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")

    point = evaluate_model_pooled(
        real_futures, gen_features, levels=levels,
        real_prices=real_prices, gen_prices=gen_prices, max_lag=max_lag,
    )
    keys = list(metrics) if metrics is not None else list(point.keys())
    keys = [k for k in keys if k in point]

    n_windows = real_futures.shape[0]
    include_price = (real_prices is not None) and (gen_prices is not None)

    # Pre-draw all replicate indices with the master RNG so results are
    # deterministic regardless of n_jobs.
    master = np.random.default_rng(seed)
    all_idx = master.integers(0, n_windows, size=(n_boot, n_windows))

    def _one_rep(idx: np.ndarray) -> Dict[str, float]:
        kw = {}
        if include_price:
            kw["real_prices"] = real_prices[idx]
            kw["gen_prices"]  = gen_prices[idx]
        rep = evaluate_model_pooled(
            real_futures[idx], gen_features[idx], levels=levels,
            max_lag=max_lag, **kw,
        )
        out = {}
        for k in keys:
            v = rep.get(k, {}).get("mean") if isinstance(rep.get(k), dict) else rep.get(k)
            if v is not None and np.isfinite(v):
                out[k] = float(v)
        return out

    if n_jobs == 1:
        iterator = range(n_boot)
        if progress:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(iterator, desc=f"bootstrap (B={n_boot})")
            except ImportError:
                pass
        rep_results = [_one_rep(all_idx[b]) for b in iterator]
    else:
        from joblib import Parallel, delayed
        verbose = 10 if progress else 0
        rep_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(
            delayed(_one_rep)(all_idx[b]) for b in range(n_boot)
        )

    samples: Dict[str, list] = {k: [] for k in keys}
    for rep in rep_results:
        for k, v in rep.items():
            samples[k].append(v)

    out: Dict[str, Dict] = {}
    for k in keys:
        vs = samples[k]
        point_mean = point[k].get("mean") if isinstance(point[k], dict) else point[k]
        if vs:
            lo = float(np.quantile(vs, alpha / 2))
            hi = float(np.quantile(vs, 1 - alpha / 2))
        else:
            lo = hi = float("nan")
        out[k] = {
            "mean":    float(point_mean) if point_mean is not None else float("nan"),
            "ci_lo":   lo,
            "ci_hi":   hi,
            "samples": vs,
        }
    return out

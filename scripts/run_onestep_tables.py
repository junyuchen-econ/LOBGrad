"""
run_onestep_tables.py
---------------------
Inference-only Table 1 for the **one-step-ahead** case (horizon = 1), as a
counterpart to the 20-step autoregressive rollout table. Runs for BTC (original
paper data + committed checkpoints) and/or the US equities MSFT / GOOGL (their
trained checkpoints). No training happens here.

One-step semantics: for each of 500 held-out windows the model predicts a single
next step from the real 100-step context (100 paths). Metrics:
  * Queue rows (L1 volume / OB imbalance Wasserstein, Queue profile MAE, CRPS)
    are the one-step distributional metrics — pooled across windows.
  * Return rows (Return Wasserstein, CRPS returns) use the genuine one-step
    return = predicted_mid - last_context_anchor (the existing evaluator diffs
    *within* the horizon, which is empty at H=1, so these are computed here).
  * ACF rows (Queue / |returns| ACF) need a multi-step trajectory and are
    therefore not applicable at one step -> shown as "---".

Outputs (kept separate from the 20-step tables):
    figures/onestep_tables/{BTC,MSFT,GOOGL}/table_headline.{csv,tex} + manifest

Usage:
    python scripts/run_onestep_tables.py --datasets BTC,MSFT,GOOGL
    python scripts/run_onestep_tables.py --datasets MSFT --n-windows 200
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO_ROOT)

import numpy as np
import pandas as pd

from lobgrad.config import BenchmarkConfig
from lobgrad.data.splits import date_splits_joint, sample_eval_windows_joint
from lobgrad.eval.evaluate import bootstrap_metrics, crps_1d, _safe_wasserstein

import bootstrap_tables as BT           # BTC models + render/generate helpers
import run_equity_table as RET          # equity model builder + config
from equity_profiles import make_config

HORIZON = 1
QUEUE_KEYS = ("l1_volume", "ob_imbalance", "queue_profile", "crps")


# ─── one-step return metrics (Wasserstein + CRPS) with window bootstrap ──────
def onestep_return_ci(real_mid0, gen_mid0, ctx_last, n_boot, seed=1234):
    """
    real_mid0 : (n_windows,)          realized next-step mid
    gen_mid0  : (n_windows, n_paths)  generated next-step mid
    ctx_last  : (n_windows,)          last observed context anchor (mid)
    Return one-step return = mid_{t+1} - mid_t.
    """
    real_ret = np.asarray(real_mid0, float) - np.asarray(ctx_last, float)          # (n,)
    gen_ret  = np.asarray(gen_mid0, float) - np.asarray(ctx_last, float)[:, None]  # (n, P)
    n = len(real_ret)

    # Per-window CRPS is precomputed once; bootstrapping the mean is then O(n).
    pw_crps = np.array([crps_1d(gen_ret[i], float(real_ret[i])) for i in range(n)])

    def _dist(idx):
        return _safe_wasserstein(real_ret[idx], gen_ret[idx].reshape(-1))

    all_idx = np.arange(n)
    pt_d, pt_c = _dist(all_idx), float(np.nanmean(pw_crps))
    rng = np.random.default_rng(seed)
    ds, cs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ds.append(_dist(idx)); cs.append(float(np.nanmean(pw_crps[idx])))
    ds, cs = np.asarray(ds), np.asarray(cs)

    def band(point, samples):
        s = samples[np.isfinite(samples)]
        return {"mean": float(point),
                "ci_lo": float(np.percentile(s, 2.5)) if len(s) else float("nan"),
                "ci_hi": float(np.percentile(s, 97.5)) if len(s) else float("nan"),
                "samples": samples.tolist()}
    return {"return_dist": band(pt_d, ds), "price_crps_usdt": band(pt_c, cs)}


def eval_onestep(name, NO_QUEUE, HAS_PRICE, fut_q, fut_a, ctx_a,
                 gen_q, gen_mid, levels, n_boot, n_jobs):
    res = {}
    if name not in NO_QUEUE and gen_q is not None:
        qci = bootstrap_metrics(fut_q, gen_q, levels=levels, n_boot=n_boot,
                                alpha=0.05, n_jobs=n_jobs, seed=1234, progress=False)
        for k in QUEUE_KEYS:
            if k in qci:
                res[k] = qci[k]
        # queue_autocorr / abs_return_autocorr: undefined at H=1 -> leave out ("---")
    if name in HAS_PRICE and gen_mid is not None:
        ret = onestep_return_ci(fut_a[:, 0], gen_mid[:, :, 0], ctx_a[:, -1], n_boot)
        res.update(ret)
    return res


# ─── dataset assembly ───────────────────────────────────────────────────────
def build_btc():
    cfg = BenchmarkConfig(
        data_dir=str(REPO_ROOT / "data"), symbol="BTCUSDT",
        levels=BT.K, price_level=BT.PRICE_LEVEL,
        queue_repr="signed", queue_mode="absolute",
    )
    (ts_tr, X_tr, a_tr), _, (ts_te, X_te, a_te), norm = date_splits_joint(cfg)
    MODELS, HAS_PRICE, NO_QUEUE = BT.build_models(cfg, norm, X_tr, ts_tr)
    return dict(cfg=cfg, models=MODELS, has_price=HAS_PRICE, no_queue=NO_QUEUE,
                X_test=X_te, anc_test=a_te, levels=cfg.levels, label="BTC")


def build_equity(symbol, gen_device):
    cfg = make_config(symbol)
    (ts_tr, X_tr, a_tr), _, (ts_te, X_te, a_te), norm = date_splits_joint(cfg)
    MODELS, HAS_PRICE, NO_QUEUE = RET.build_eval_models(
        symbol, norm, X_tr, ts_tr, gen_device, gen_device, RET.MODEL_ORDER)
    return dict(cfg=cfg, models=MODELS, has_price=HAS_PRICE, no_queue=NO_QUEUE,
                X_test=X_te, anc_test=a_te, levels=cfg.levels, label=symbol)


def run_dataset(ds, n_windows, n_paths, n_boot, n_jobs, seed):
    label = ds["label"]
    MODELS, HAS_PRICE, NO_QUEUE = ds["models"], ds["has_price"], ds["no_queue"]
    levels = ds["levels"]
    print(f"\n=== {label}: one-step (H=1) over {n_windows} windows x {n_paths} paths ===")
    print(f"    models: {list(MODELS)}")

    rng = np.random.default_rng(seed)
    ctx_q, fut_q, ctx_a, fut_a = sample_eval_windows_joint(
        ds["X_test"], ds["anc_test"], 100, HORIZON, n_windows, rng=rng)

    gen_q_all, gen_mid_all = {}, {}
    for name, mdl in MODELS.items():
        q, mid = BT.generate_paths(name, mdl, ctx_q, ctx_a, n_windows, n_paths,
                                   HORIZON, seed, cache_dir=None)
        if q is not None:
            gen_q_all[name] = q
        if mid is not None:
            gen_mid_all[name] = mid

    results_ci = {}
    for name in MODELS:
        results_ci[name] = eval_onestep(
            name, NO_QUEUE, HAS_PRICE, fut_q, fut_a, ctx_a,
            gen_q_all.get(name), gen_mid_all.get(name), levels, n_boot, n_jobs)

    # ── write ──
    od = REPO_ROOT / "figures" / "onestep_tables" / label
    od.mkdir(parents=True, exist_ok=True)
    tex = BT.render_table_1(results_ci, list(MODELS), HORIZON, n_boot=n_boot, n_windows=n_windows)
    (od / "table_headline.tex").write_text(tex)

    rows = []
    for name in MODELS:
        for key, _ in BT.QUEUE_BLOCK + BT.PRICE_BLOCK:
            band = results_ci.get(name, {}).get(key) or {}
            full_mean = band.get("mean")
            samples = band.get("samples") or []
            median = float(np.median(samples)) if len(samples) else None
            mean, lo, hi = BT._ci_triple(results_ci, name, key)
            rows.append({"model": name, "metric": key,
                         "mean": float(full_mean) if full_mean is not None and np.isfinite(full_mean) else None,
                         "median": median, "display": mean, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(rows).to_csv(od / "table_headline.csv", index=False)
    (od / "run_manifest.json").write_text(json.dumps({
        "dataset": label, "horizon": HORIZON, "generated_at": datetime.now().isoformat(),
        "models": list(MODELS), "n_windows": n_windows, "n_paths": n_paths,
        "n_boot": n_boot, "context_len": 100, "seed": seed,
        "note": "one-step-ahead; ACF rows N/A at H=1; return rows use predicted_mid - last_context_anchor",
    }, indent=2))
    print(f"    wrote {od}/table_headline.{{csv,tex}}")

    if "LOBGrad" in results_ci:
        print("    LOBGrad one-step headline:")
        for key, _ in BT.QUEUE_BLOCK + BT.PRICE_BLOCK:
            m, lo, hi = BT._ci_triple(results_ci, "LOBGrad", key)
            print(f"      {key:>22s}: " + ("---" if m is None else f"{m:.4f} [{lo:.4f}, {hi:.4f}]"))
    return results_ci


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="BTC,MSFT,GOOGL")
    ap.add_argument("--n-windows", type=int, default=500)
    ap.add_argument("--n-paths", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--gen-device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gen_device = RET.pick_device(args.gen_device)
    targets = [d.strip().upper() for d in args.datasets.split(",") if d.strip()]
    t0 = time.time()

    for label in targets:
        if label == "BTC":
            ds = build_btc()
        elif label in ("MSFT", "GOOGL"):
            ds = build_equity(label, gen_device)
        else:
            print(f"[skip] unknown dataset {label}")
            continue
        run_dataset(ds, args.n_windows, args.n_paths, args.n_boot, args.n_jobs, args.seed)

    print(f"\n=== all one-step tables done in {(time.time()-t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()

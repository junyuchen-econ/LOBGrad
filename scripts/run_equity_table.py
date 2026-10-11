"""
run_equity_table.py
-------------------
End-to-end driver that reproduces the paper's Table 1 (distributional metrics)
on the decoded XNAS.ITCH US-equity data, for a single symbol (MSFT or GOOGL).

Pipeline (per symbol):
  1. Load train/val/test splits via lobgrad (data in cents; see equity_profiles).
  2. Train the 5 neural columns on the equity train split, saving to
     checkpoints/equity/{SYMBOL}/{model}/:
        LOBGrad  = JointTimeGradGenerator (LSTM encoder, joint Δmid+queue)
        LG-NE    = JointTimeGradGenerator (flat_window encoder, no RNN)
        LG-Q     = TimeGradGenerator (queue-only)
        LG-R     = return-only TimeGradMLP (input_dim=1 Δmid)
        CGAN     = CGANGenerator (WGAN-GP, trial-42 architecture)
  3. Fit Hawkes + Poisson live.
  4. Generate 20-step, 100-path forecasts over 500 test windows (seed 42).
  5. Bootstrap Table 1 (1000 resamples) -> figures/equity_tables/{SYMBOL}/
        table_headline.{csv,tex} + run_manifest.json

Reuses the paper's eval/render helpers from scripts/bootstrap_tables.py.

Usage:
    python scripts/run_equity_table.py --symbol MSFT
    python scripts/run_equity_table.py --symbol GOOGL --device mps
    python scripts/run_equity_table.py --symbol MSFT --smoke        # fast wiring test
    python scripts/run_equity_table.py --symbol MSFT --skip-train   # eval existing ckpts
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
import torch

from equity_profiles import make_config, ckpt_dir, out_dir, cache_dir, load_calibration
from lobgrad.data.splits import date_splits_joint, sample_eval_windows_joint
from lobgrad.execution.simulate import JointTimeGradGenerator, build_joint_features
from lobgrad.models.timegrad import TimeGradGenerator
from lobgrad.models.cgan import CGANGenerator
from lobgrad.models.hawkes import HawkesGenerator
from lobgrad.models.poisson import PoissonGenerator
from lobgrad.eval.evaluate import bootstrap_metrics

from model import TimeGradMLP                       # src/
from train import TimeSeriesDataset, train_model    # src/

# Reuse the paper's generation + rendering helpers (importing runs top-level
# setup only; main() is guarded).
import bootstrap_tables as BT

MODEL_ORDER = ["LOBGrad", "LG-NE", "LG-Q", "LG-R", "CGAN", "Hawkes", "Poisson"]
QUEUE_KEYS = ("l1_volume", "ob_imbalance", "queue_profile", "queue_autocorr", "crps")


# ─── device ─────────────────────────────────────────────────────────────────
def pick_device(arg: str) -> str:
    if arg and arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _fresh(save_dir: Path):
    """Remove old checkpoints so generators train from scratch (not warm-start)."""
    for f in ("best_model.pt", "final_model.pt"):
        p = save_dir / f
        if p.exists():
            p.unlink()


# ─── training helpers ───────────────────────────────────────────────────────
def train_joint(symbol, X_train, anc_train, ts_train, normalizer, device,
                flat_window: bool, epochs, max_batches):
    """LOBGrad (flat_window=False) or LG-NE (flat_window=True)."""
    name = "LG-NE" if flat_window else "LOBGrad"
    sd = ckpt_dir(symbol, "timegrad_joint_flat_window" if flat_window else "joint_timegrad")
    sd.mkdir(parents=True, exist_ok=True)
    _fresh(sd)
    over = dict(
        timegrad_train=True, timegrad_save_dir=str(sd),
        timegrad_checkpoint=str(sd / "best_model.pt"),
        timegrad_epochs=epochs, timegrad_max_batches=max_batches,
    )
    if flat_window:
        over.update(timegrad_cell_type="flat_window", timegrad_use_cond_projection=False,
                    timegrad_flat_window_size=100, timegrad_loss_type="l1")
    cfg = make_config(symbol, **over)
    print(f"\n[{name}] training -> {sd}")
    gen = JointTimeGradGenerator(cfg, normalizer, device=device)
    gen.fit(X_train, anc_train, ts_train)
    return sd


def train_queue(symbol, X_train, ts_train, normalizer, device, epochs, max_batches):
    sd = ckpt_dir(symbol, "timegrad_queue_only")
    sd.mkdir(parents=True, exist_ok=True)
    _fresh(sd)
    cfg = make_config(symbol, timegrad_train=True, timegrad_save_dir=str(sd),
                      timegrad_checkpoint=str(sd / "best_model.pt"),
                      timegrad_epochs=epochs, timegrad_max_batches=max_batches)
    print(f"\n[LG-Q] training -> {sd}")
    gen = TimeGradGenerator(cfg, normalizer, device=device)
    gen.fit(X_train, ts_train)
    return sd


def train_return(symbol, anc_train, device, epochs, max_batches):
    """LG-R: return-only TimeGradMLP (input_dim=1 on the Δmid stream)."""
    sd = ckpt_dir(symbol, "timegrad_return_only")
    sd.mkdir(parents=True, exist_ok=True)
    _fresh(sd)
    cfg = make_config(symbol)   # arch + delta_mid_clip
    clip = cfg.delta_mid_clip
    T = len(anc_train)
    delta = np.zeros((T, 1), dtype=np.float32)
    delta[1:, 0] = np.clip(anc_train[1:] - anc_train[:-1], -clip, clip)

    model = TimeGradMLP(
        input_dim=1, output_dim=1,
        hidden_encoder_dim=cfg.timegrad_hidden_enc, hidden_noise_dim=cfg.timegrad_hidden_noise,
        num_encoder_layers=cfg.timegrad_num_enc_layers, emb_dim=cfg.timegrad_emb_dim,
        K=cfg.timegrad_k, beta_start=cfg.timegrad_beta_start, beta_end=cfg.timegrad_beta_end,
        schedule=cfg.timegrad_schedule, loss_type=cfg.timegrad_loss_type,
        dropout_rate=cfg.timegrad_dropout, cell_type=cfg.timegrad_cell_type,
        time_embedding=cfg.timegrad_time_embedding,
        use_cond_projection=cfg.timegrad_use_cond_projection,
    ).to(device)

    ds = TimeSeriesDataset(delta, cfg.timegrad_history_length, cfg.timegrad_prediction_length)
    idx = ds.valid_indices
    split = int(0.9 * len(idx))
    tl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx[:split]),
                                     batch_size=cfg.timegrad_batch_size, shuffle=True)
    vl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx[split:]),
                                     batch_size=cfg.timegrad_batch_size, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.timegrad_lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    print(f"\n[LG-R] training -> {sd}")
    train_model(model=model, train_loader=tl, val_loader=vl, optimizer=opt,
                num_epochs=epochs, device=torch.device(device), verbose=True,
                save_dir=str(sd), max_batches=max_batches, scheduler=sch)
    return sd


def train_cgan(symbol, X_train, anc_train, ts_train, device, epochs, max_batches):
    sd = ckpt_dir(symbol, "cgan")
    sd.mkdir(parents=True, exist_ok=True)
    cfg = make_config(symbol, cgan_epochs=epochs, cgan_max_batches=max_batches)
    print(f"\n[CGAN] training (device={device}) -> {sd}")
    gen = CGANGenerator(cfg, device=device)
    gen.fit(X_train, ts_train, anchors=anc_train)
    gen.save(str(sd / "cgan.pt"))
    return sd


# ─── model building for eval ────────────────────────────────────────────────
def build_eval_models(symbol, normalizer, X_train, ts_train, device, cgan_device, want):
    MODELS, HAS_PRICE, NO_QUEUE = {}, set(), set()

    def ck(model, fname="best_model.pt"):
        return ckpt_dir(symbol, model) / fname

    if "LOBGrad" in want and ck("joint_timegrad").exists():
        cfg = make_config(symbol, timegrad_train=False,
                          timegrad_checkpoint=str(ck("joint_timegrad")))
        MODELS["LOBGrad"] = JointTimeGradGenerator(cfg, normalizer, device=device)
        HAS_PRICE.add("LOBGrad")

    if "LG-NE" in want and ck("timegrad_joint_flat_window").exists():
        cfg = make_config(symbol, timegrad_train=False,
                          timegrad_checkpoint=str(ck("timegrad_joint_flat_window")),
                          timegrad_cell_type="flat_window", timegrad_use_cond_projection=False,
                          timegrad_flat_window_size=100, timegrad_loss_type="l1")
        MODELS["LG-NE"] = JointTimeGradGenerator(cfg, normalizer, device=device)
        HAS_PRICE.add("LG-NE")

    if "LG-Q" in want and ck("timegrad_queue_only").exists():
        cfg = make_config(symbol, timegrad_train=False,
                          timegrad_checkpoint=str(ck("timegrad_queue_only")))
        MODELS["LG-Q"] = TimeGradGenerator(cfg, normalizer, device=device)

    if "LG-R" in want and ck("timegrad_return_only").exists():
        clip = make_config(symbol).delta_mid_clip
        MODELS["LG-R"] = BT.ReturnOnlyTimeGradGenerator(
            str(ck("timegrad_return_only")), delta_mid_clip=clip, device=device)
        HAS_PRICE.add("LG-R")
        NO_QUEUE.add("LG-R")

    if "CGAN" in want and ck("cgan", "cgan.pt").exists():
        cfg = make_config(symbol)
        cgan = CGANGenerator(cfg, device=cgan_device)
        cgan.load(str(ck("cgan", "cgan.pt")))
        MODELS["CGAN"] = cgan
        HAS_PRICE.add("CGAN")

    if "Hawkes" in want:
        cfg = make_config(symbol)
        h = HawkesGenerator(cfg, normalizer); h.fit(X_train, ts_train)
        MODELS["Hawkes"] = h
    if "Poisson" in want:
        cfg = make_config(symbol)
        p = PoissonGenerator(cfg, normalizer); p.fit(X_train, ts_train)
        MODELS["Poisson"] = p

    ordered = {n: MODELS[n] for n in MODEL_ORDER if n in MODELS}
    return ordered, HAS_PRICE, NO_QUEUE


# ─── bootstrap dispatch (mirrors bootstrap_tables._bootstrap_one, shapes explicit) ─
def bootstrap_one(name, NO_QUEUE, HAS_PRICE, real_q, real_a, gen_q, gen_mid,
                  levels, n_paths, horizon, n_boot, n_jobs):
    if name in NO_QUEUE:
        if gen_mid is None:
            return None
        dummy = np.zeros((real_q.shape[0], n_paths, horizon, 2 * levels), dtype=np.float32)
        ci = bootstrap_metrics(real_q, dummy, levels=levels,
                               real_prices=real_a, gen_prices=gen_mid,
                               n_boot=n_boot, alpha=0.05, n_jobs=n_jobs, seed=1234, progress=False)
        for qk in QUEUE_KEYS:
            ci[qk] = {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "samples": []}
        return ci
    kw = {}
    if name in HAS_PRICE and gen_mid is not None:
        kw["real_prices"], kw["gen_prices"] = real_a, gen_mid
    return bootstrap_metrics(real_q, gen_q, levels=levels,
                             n_boot=n_boot, alpha=0.05, n_jobs=n_jobs,
                             seed=1234, progress=False, **kw)


def write_outputs(symbol, results_ci, models, horizon, n_boot, n_windows, cal, args, elapsed):
    od = out_dir(symbol); od.mkdir(parents=True, exist_ok=True)
    tex = BT.render_table_1(results_ci, list(models), horizon, n_boot=n_boot, n_windows=n_windows)
    (od / "table_headline.tex").write_text(tex)

    import pandas as pd
    rows = []
    for name in models:
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
        "symbol": symbol, "generated_at": datetime.now().isoformat(),
        "elapsed_sec": elapsed, "models": list(models),
        "n_windows": n_windows, "n_paths": args.n_paths, "horizon": horizon,
        "context_len": 100, "n_boot": n_boot, "device": args.device,
        "calibration": cal, "seed": args.seed,
    }, indent=2))
    print(f"\nWrote {od}/table_headline.{{csv,tex}} + run_manifest.json")


# ─── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", required=True, choices=["MSFT", "GOOGL"])
    ap.add_argument("--device", default="auto", help="training device: auto|cpu|mps|cuda")
    ap.add_argument("--gen-device", default="cpu",
                    help="device for eval/generation (CPU is fastest for K-step diffusion sampling)")
    ap.add_argument("--cgan-device", default=None, help="override CGAN training device (default cpu if device=mps)")
    ap.add_argument("--epochs", type=int, default=None, help="override TimeGrad epochs")
    ap.add_argument("--max-batches", type=int, default=None, help="override TimeGrad max_batches")
    ap.add_argument("--cgan-epochs", type=int, default=None)
    ap.add_argument("--cgan-max-batches", type=int, default=None)
    ap.add_argument("--n-windows", type=int, default=500)
    ap.add_argument("--n-paths", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", default=",".join(MODEL_ORDER),
                    help="comma list subset of " + ",".join(MODEL_ORDER))
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--cache", action="store_true", help="cache generations (per symbol)")
    ap.add_argument("--smoke", action="store_true", help="tiny budget to validate wiring")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = args.epochs or 2
        args.max_batches = args.max_batches or 20
        args.cgan_epochs = args.cgan_epochs or 2
        args.cgan_max_batches = args.cgan_max_batches or 20
        args.n_windows = 20 if args.n_windows == 500 else args.n_windows
        args.n_paths = 5 if args.n_paths == 100 else args.n_paths
        args.n_boot = 100 if args.n_boot == 1000 else args.n_boot

    device = pick_device(args.device)
    args.device = device
    gen_device = pick_device(args.gen_device)
    cgan_device = args.cgan_device or ("cpu" if device == "mps" else device)
    want = [m.strip() for m in args.models.split(",") if m.strip()]
    tg_epochs = args.epochs
    tg_maxb = args.max_batches
    cg_epochs = args.cgan_epochs
    cg_maxb = args.cgan_max_batches

    cal = load_calibration(args.symbol)
    print(f"=== Equity Table 1 :: {args.symbol} ===")
    print(f"  train_device={device}  cgan_device={cgan_device}  gen_device={gen_device}  models={want}")
    print(f"  price_level={cal['price_level']}c  delta_mid_clip={cal['delta_mid_clip']}c")
    t0 = time.time()

    # ── data ──
    cfg = make_config(args.symbol)
    (ts_train, X_train, anc_train), (ts_val, X_val, anc_val), \
        (ts_test, X_test, anc_test), normalizer = date_splits_joint(cfg)
    print(f"  splits: train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}  "
          f"(feat dim {X_train.shape[1]})")

    # ── train ──
    if not args.skip_train:
        e = tg_epochs if tg_epochs is not None else cfg.timegrad_epochs
        mb = tg_maxb if tg_maxb is not None else cfg.timegrad_max_batches
        ce = cg_epochs if cg_epochs is not None else cfg.cgan_epochs
        cmb = cg_maxb if cg_maxb is not None else cfg.cgan_max_batches
        if "LOBGrad" in want:
            train_joint(args.symbol, X_train, anc_train, ts_train, normalizer, device, False, e, mb)
        if "LG-NE" in want:
            train_joint(args.symbol, X_train, anc_train, ts_train, normalizer, device, True, e, mb)
        if "LG-Q" in want:
            train_queue(args.symbol, X_train, ts_train, normalizer, device, e, mb)
        if "LG-R" in want:
            train_return(args.symbol, anc_train, device, e, mb)
        if "CGAN" in want:
            train_cgan(args.symbol, X_train, anc_train, ts_train, cgan_device, ce, cmb)
        print(f"\n[train] done ({(time.time()-t0)/60:.1f} min)")

    # ── build eval models (generation on gen_device; CPU is fastest for sampling) ──
    MODELS, HAS_PRICE, NO_QUEUE = build_eval_models(
        args.symbol, normalizer, X_train, ts_train, gen_device, gen_device, want)
    print(f"\n[eval] active models: {list(MODELS)}")

    # ── eval windows ──
    rng = np.random.default_rng(args.seed)
    test_ctx_q, test_fut_q, test_ctx_a, test_fut_a = sample_eval_windows_joint(
        X_test, anc_test, 100, args.horizon, args.n_windows, rng=rng)

    # ── generate ──
    cdir = str(cache_dir(args.symbol)) if args.cache else None
    gen_q_all, gen_mid_all = {}, {}
    for name, mdl in MODELS.items():
        q, mid = BT.generate_paths(name, mdl, test_ctx_q, test_ctx_a,
                                   args.n_windows, args.n_paths, args.horizon, args.seed,
                                   cache_dir=cdir)
        if q is not None:
            gen_q_all[name] = q
        if mid is not None:
            gen_mid_all[name] = mid

    # ── bootstrap Table 1 ──
    print("\n[bootstrap] Table 1 ...")
    results_ci = {}
    for name in MODELS:
        ci = bootstrap_one(name, NO_QUEUE, HAS_PRICE, test_fut_q, test_fut_a,
                           gen_q_all.get(name), gen_mid_all.get(name),
                           cfg.levels, args.n_paths, args.horizon, args.n_boot, args.n_jobs)
        if ci is not None:
            results_ci[name] = ci

    if "LOBGrad" in results_ci:
        print("  LOBGrad headline:")
        for key, _ in BT.QUEUE_BLOCK + BT.PRICE_BLOCK:
            m, lo, hi = BT._ci_triple(results_ci, "LOBGrad", key)
            print(f"    {key:>22s}: " + ("---" if m is None else f"{m:.4f} [{lo:.4f}, {hi:.4f}]"))

    write_outputs(args.symbol, results_ci, list(MODELS), args.horizon,
                  args.n_boot, args.n_windows, cal, args, time.time() - t0)
    print(f"\n=== Done {args.symbol} in {(time.time()-t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()

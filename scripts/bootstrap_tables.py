"""
Compute bootstrap-CI versions of Table 1 (headline distributional metrics).

Mirrors the load + generate logic of notebooks/paper_figures.ipynb so it can
run on a server without a Jupyter kernel. Writes both CSV (long-form) and
LaTeX (paper-ready) for each table.

    Output: figures/bootstrapped_tables/
        table_headline.{csv,tex}
        run_manifest.json

    Cache:  cache/generations/{name}_seed{S}_W{N_W}_P{N_P}_H{H}.npz

Usage:
    python scripts/bootstrap_tables.py
    python scripts/bootstrap_tables.py --n-boot 500 --n-jobs 8
    python scripts/bootstrap_tables.py --no-cache
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

# ─── Path setup ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_SRC = str(REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
os.chdir(REPO_ROOT)

from lobgrad.config import BenchmarkConfig
from lobgrad.data.splits import date_splits_joint, sample_eval_windows_joint
from lobgrad.eval.evaluate import bootstrap_metrics
from lobgrad.execution.simulate import JointTimeGradGenerator
from lobgrad.models.timegrad import TimeGradGenerator
from lobgrad.models.cgan import CGANGenerator
from lobgrad.models.hawkes import HawkesGenerator
from lobgrad.models.poisson import PoissonGenerator
from model import TimeGradMLP  # type: ignore  # legacy src/

# ─── Hyperparameters (mirror paper_figures.ipynb config cell) ───────────
TG_CKPT   = "checkpoints/final/joint_timegrad/best_model.pt"
TGQ_CKPT  = "checkpoints/final/timegrad_queue_only/best_model.pt"
TGR_CKPT  = "checkpoints/final/timegrad_return_only/best_model.pt"
TGNW_CKPT = "checkpoints/final/timegrad_joint_flat_window/final_model.pt"
CGAN_CKPT = "checkpoints/final/cgan/cgan.pt"

TG_LABEL   = "LOBGrad"
TGQ_LABEL  = "LG-Q"
TGR_LABEL  = "LG-R"
TGNW_LABEL = "LG-NE"

K            = 5
PRICE_LEVEL  = 2.0
CONTEXT_LEN  = 100
HORIZON      = 20
N_PATHS      = 100
N_WINDOWS    = 500

EMB_DIM        = 32
HIDDEN_ENC     = 151
HIDDEN_NOISE   = 255
DIFFUSION_K    = 102
NUM_ENC_LAYERS = 1
CELL_TYPE      = "LSTM"
USE_COND_PROJ  = True
LOSS_TYPE      = "l2"

CGAN_HP = dict(
    queue_mode       = "absolute",
    latent_dim       = 74,
    hidden_dim       = 367,
    n_encoder_layers = 1,
    n_decoder_layers = 2,
)

QUEUE_KEYS = ("l1_volume", "ob_imbalance", "queue_profile",
              "queue_autocorr", "crps")

# Wasserstein-family metrics: percentile bootstrap is upward-biased under
# cross-window pooling (resampled windows have smaller effective support, so
# the empirical Wasserstein increases). For these we report the bootstrap
# median as the point — by construction it lies inside the percentile CI.
# All other metrics keep the full-data point estimate.
WASS_KEYS = frozenset(("l1_volume", "ob_imbalance", "return_dist"))

# Per-label conventions used to populate HAS_PRICE / NO_QUEUE when we skip
# Phase 2 (model build) on a cache hit. Must match what build_models would set.
HAS_PRICE_BASE = frozenset({TG_LABEL, TGNW_LABEL, TGR_LABEL, "CGAN"})
NO_QUEUE_BASE  = frozenset({TGR_LABEL})


# ─── ReturnOnlyTimeGradGenerator (ported from notebook) ─────────────────
class ReturnOnlyTimeGradGenerator:
    def __init__(self, ckpt_path, delta_mid_clip=12.0, device=None):
        self.ckpt_path      = ckpt_path
        self.delta_mid_clip = float(delta_mid_clip)
        self.device         = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TimeGradMLP(
            input_dim=1, output_dim=1,
            hidden_encoder_dim=HIDDEN_ENC, hidden_noise_dim=HIDDEN_NOISE,
            num_encoder_layers=NUM_ENC_LAYERS, emb_dim=EMB_DIM,
            K=DIFFUSION_K, beta_start=0.0001, beta_end=0.1,
            schedule="linear", cell_type=CELL_TYPE,
            use_cond_projection=USE_COND_PROJ,
        )
        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval().to(self.device)
        print(f"[TimeGrad-R] Loaded checkpoint from {ckpt_path}")

    def generate(self, context_anchors, steps, n_paths):
        ctx = np.asarray(context_anchors, dtype=np.float32).reshape(-1)
        delta = np.zeros(len(ctx), dtype=np.float32)
        delta[1:] = ctx[1:] - ctx[:-1]
        delta = np.clip(delta, -self.delta_mid_clip, self.delta_mid_clip)
        delta = delta[1:].reshape(-1, 1)
        ctx_tensor = torch.tensor(
            np.tile(delta[np.newaxis], (n_paths, 1, 1)),
            dtype=torch.float32, device=self.device,
        )
        with torch.no_grad():
            raw = self.model.sample(ctx_tensor, prediction_steps=steps).cpu().numpy()
        delta_out = np.clip(raw[:, :, 0], -self.delta_mid_clip, self.delta_mid_clip)
        last_anchor = float(ctx[-1])
        return last_anchor + np.cumsum(delta_out, axis=1)


# ─── Model construction (mirror of notebook 'models' cell) ──────────────
def build_models(cfg, normalizer, X_train, ts_train):
    MODELS    = {}
    HAS_PRICE = set()
    NO_QUEUE  = set()

    cfg_tg = BenchmarkConfig(
        levels=K, price_level=PRICE_LEVEL,
        queue_repr="signed", queue_mode="absolute",
        timegrad_checkpoint=TG_CKPT, timegrad_train=False,
        timegrad_k=DIFFUSION_K, timegrad_hidden_enc=HIDDEN_ENC,
        timegrad_hidden_noise=HIDDEN_NOISE, timegrad_num_enc_layers=NUM_ENC_LAYERS,
        timegrad_emb_dim=EMB_DIM, timegrad_cell_type=CELL_TYPE,
        timegrad_use_cond_projection=USE_COND_PROJ,
        timegrad_loss_type=LOSS_TYPE, timegrad_schedule="linear",
        timegrad_beta_start=0.0001, timegrad_beta_end=0.1,
    )
    print(f"Loading {TG_LABEL}")
    MODELS[TG_LABEL] = JointTimeGradGenerator(cfg_tg, normalizer)
    HAS_PRICE.add(TG_LABEL)

    if os.path.exists(TGNW_CKPT):
        cfg_nw = BenchmarkConfig(
            levels=K, price_level=PRICE_LEVEL,
            queue_repr="signed", queue_mode="absolute",
            timegrad_checkpoint=TGNW_CKPT, timegrad_train=False,
            timegrad_k=DIFFUSION_K, timegrad_hidden_enc=HIDDEN_ENC,
            timegrad_hidden_noise=HIDDEN_NOISE, timegrad_num_enc_layers=NUM_ENC_LAYERS,
            timegrad_emb_dim=EMB_DIM,
            timegrad_cell_type="flat_window",
            timegrad_flat_window_size=CONTEXT_LEN,
            timegrad_use_cond_projection=False,
            timegrad_loss_type="huber", timegrad_schedule="linear",
            timegrad_beta_start=0.0001, timegrad_beta_end=0.1,
        )
        print(f"Loading {TGNW_LABEL}")
        MODELS[TGNW_LABEL] = JointTimeGradGenerator(cfg_nw, normalizer)
        HAS_PRICE.add(TGNW_LABEL)
    else:
        print(f"⚠  {TGNW_LABEL} checkpoint not found — skipping")

    if os.path.exists(TGQ_CKPT):
        cfg_q = BenchmarkConfig(
            levels=K, price_level=PRICE_LEVEL,
            queue_repr="signed", queue_mode="absolute",
            timegrad_checkpoint=TGQ_CKPT, timegrad_train=False,
            timegrad_k=DIFFUSION_K, timegrad_hidden_enc=HIDDEN_ENC,
            timegrad_hidden_noise=HIDDEN_NOISE, timegrad_num_enc_layers=NUM_ENC_LAYERS,
            timegrad_emb_dim=EMB_DIM, timegrad_cell_type=CELL_TYPE,
            timegrad_use_cond_projection=USE_COND_PROJ,
            timegrad_loss_type=LOSS_TYPE, timegrad_schedule="linear",
            timegrad_beta_start=0.0001, timegrad_beta_end=0.1,
        )
        print(f"Loading {TGQ_LABEL}")
        MODELS[TGQ_LABEL] = TimeGradGenerator(cfg_q, normalizer)
    else:
        print(f"⚠  {TGQ_LABEL} checkpoint not found — skipping")

    if os.path.exists(TGR_CKPT):
        print(f"Loading {TGR_LABEL}")
        MODELS[TGR_LABEL] = ReturnOnlyTimeGradGenerator(TGR_CKPT)
        HAS_PRICE.add(TGR_LABEL)
        NO_QUEUE.add(TGR_LABEL)
    else:
        print(f"⚠  {TGR_LABEL} checkpoint not found — skipping")

    if os.path.exists(CGAN_CKPT):
        cfg_cgan = BenchmarkConfig(
            levels=K, price_level=PRICE_LEVEL, queue_repr="signed",
            queue_mode=CGAN_HP["queue_mode"],
            cgan_latent_dim=CGAN_HP["latent_dim"],
            cgan_hidden_dim=CGAN_HP["hidden_dim"],
            cgan_n_encoder_layers=CGAN_HP["n_encoder_layers"],
            cgan_n_decoder_layers=CGAN_HP["n_decoder_layers"],
        )
        print("Loading CGAN")
        cgan = CGANGenerator(cfg_cgan)
        cgan.load(CGAN_CKPT)
        MODELS["CGAN"] = cgan
        HAS_PRICE.add("CGAN")
    else:
        print(f"⚠  CGAN checkpoint not found — skipping")

    print("\nFitting Hawkes...")
    hawkes = HawkesGenerator(cfg, normalizer)
    hawkes.fit(X_train, ts_train)
    MODELS["Hawkes"] = hawkes

    print("Fitting Poisson...")
    poisson = PoissonGenerator(cfg, normalizer)
    poisson.fit(X_train, ts_train)
    MODELS["Poisson"] = poisson

    print(f"\n=> {len(MODELS)} models active: {list(MODELS)}")
    print(f"   HAS_PRICE : {sorted(HAS_PRICE)}")
    print(f"   NO_QUEUE  : {sorted(NO_QUEUE)}")
    return MODELS, HAS_PRICE, NO_QUEUE


# ─── Path generation (mirror of notebook 'generate' cell) ───────────────
def _seed_all(s):
    import random as _r
    _r.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _call_generate(name, mdl, ctx_q, ctx_a, steps, n_paths):
    if name in (TG_LABEL, TGNW_LABEL):
        q, mid = mdl.generate(ctx_q, ctx_a, steps, n_paths)
        return q, mid
    if name == "CGAN":
        q, mid = mdl.generate(ctx_q, steps=steps, n_paths=n_paths,
                              context_anchors=ctx_a)
        return q, mid
    if name == TGR_LABEL:
        mid = mdl.generate(ctx_a, steps=steps, n_paths=n_paths)
        return None, mid
    q = mdl.generate(ctx_q, steps=steps, n_paths=n_paths)
    return q, None


def _scan_gen_cache(cache_dir, seed, n_w, n_p, h):
    """Return labels of cached generations matching the run params."""
    if cache_dir is None:
        return []
    p = Path(cache_dir)
    if not p.exists():
        return []
    suffix = f"_seed{seed}_W{n_w}_P{n_p}_H{h}.npz"
    return sorted(f.name[:-len(suffix)] for f in p.glob(f"*{suffix}"))


def _eval_targets_cache(cache_dir, seed, n_w, h):
    if cache_dir is None:
        return None
    return Path(cache_dir) / f"eval_targets_seed{seed}_W{n_w}_H{h}.npz"


def _try_load_eval_targets(cache_dir, seed, n_w, h):
    p = _eval_targets_cache(cache_dir, seed, n_w, h)
    if p is None or not p.exists():
        return None
    blob = np.load(p)
    return blob["test_fut_q"], blob["test_fut_a"], blob.get("test_ctx_a")


def _save_eval_targets(cache_dir, seed, n_w, h, test_fut_q, test_fut_a, test_ctx_a):
    p = _eval_targets_cache(cache_dir, seed, n_w, h)
    if p is None:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, test_fut_q=test_fut_q, test_fut_a=test_fut_a,
                        test_ctx_a=test_ctx_a)
    print(f"  cached eval targets → {p}")


def generate_paths(name, mdl, test_ctx_q, test_ctx_a,
                   n_windows, n_paths, horizon, seed, cache_dir=None):
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / (
            f"{name.replace('/', '_')}_seed{seed}"
            f"_W{n_windows}_P{n_paths}_H{horizon}.npz"
        )
        if cache_path.exists():
            blob = np.load(cache_path, allow_pickle=False)
            q   = blob["q"]   if "q"   in blob.files else None
            mid = blob["mid"] if "mid" in blob.files else None
            print(f"  ({name}) cached  q={None if q is None else q.shape}  "
                  f"mid={None if mid is None else mid.shape}")
            return q, mid

    _seed_all(seed)
    paths_q, paths_mid = [], []
    for i in tqdm(range(n_windows), desc=name):
        q, mid = _call_generate(name, mdl, test_ctx_q[i], test_ctx_a[i],
                                horizon, n_paths)
        if q is not None:
            paths_q.append(q)
        if mid is not None:
            paths_mid.append(mid)

    out_q   = np.stack(paths_q)   if paths_q   else None
    out_mid = np.stack(paths_mid) if paths_mid else None

    if cache_path is not None:
        save_kw = {}
        if out_q   is not None: save_kw["q"]   = out_q
        if out_mid is not None: save_kw["mid"] = out_mid
        np.savez_compressed(cache_path, **save_kw)
        print(f"  cached → {cache_path}")

    return out_q, out_mid


# ─── LaTeX rendering (stacked CIs, mirrors paper_figures notebook) ──────
BS, EOL = chr(92), " " + chr(92) + chr(92)

QUEUE_BLOCK = [
    ("l1_volume",      "L1 volume Wasserstein"),
    ("ob_imbalance",   "OB imbalance Wasserstein"),
    ("queue_profile",  "Queue profile MAE"),
    ("queue_autocorr", "Queue ACF MAE"),
    ("crps",           "CRPS (queue)"),
]
PRICE_BLOCK = [
    ("return_dist",         "Return Wasserstein"),
    ("abs_return_autocorr", "Absolute returns ACF MAE"),
    ("price_crps_usdt",     "CRPS (returns)"),
]


def _stack_cell(main, lo, hi, precision=4):
    fmt = "{:." + str(precision) + "f}"
    ci_line = ("{" + BS + "scriptsize $["
               + fmt.format(lo) + "," + BS + "," + fmt.format(hi) + "]$}")
    return BS + "shortstack[r]{" + main + " " + BS + BS + " " + ci_line + "}"


def _ci_triple(results_ci, name, key):
    band = results_ci.get(name, {}).get(key)
    if band is None:
        return None, None, None
    if key in WASS_KEYS:
        samples = band.get("samples") or []
        mean = float(np.median(samples)) if len(samples) else band.get("mean")
    else:
        mean = band.get("mean")
    lo, hi = band.get("ci_lo"), band.get("ci_hi")
    if mean is None or not np.isfinite(mean):
        return None, None, None
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return float(mean), None, None
    return float(mean), float(lo), float(hi)


def render_table_1(results_ci, models, horizon, n_boot=1000, n_windows=N_WINDOWS):
    def _row(label, key):
        vals = {n: _ci_triple(results_ci, n, key) for n in models}
        finite = {n: v[0] for n, v in vals.items() if v[0] is not None}
        best = min(finite, key=finite.get) if finite else None
        cells = []
        for n in models:
            mean, lo, hi = vals[n]
            if mean is None:
                cells.append("---")
                continue
            p = f"{mean:.3f}"
            main = (BS + "textbf{" + p + "}") if n == best else p
            cells.append(main if lo is None else _stack_cell(main, lo, hi, precision=3))
        return [label, *cells]

    cols    = "l" + "r" * len(models)
    n_cols  = len(models) + 1
    multi_q = BS + "multicolumn{" + str(n_cols) + "}{l}{" + BS + "emph{Queue}}" + EOL
    multi_r = BS + "multicolumn{" + str(n_cols) + "}{l}{" + BS + "emph{Returns}}" + EOL
    head    = " & ".join(["Metric"] + list(models)) + EOL

    cap = (
        "Distributional metrics on the $" + BS + "tau{=}" + str(horizon)
        + "$-step horizon across " + str(n_windows) + " held-out windows. "
        + "Lower is better. " + BS + "textbf{Bold} marks the best value per row. "
        + "Sub-rows give 95\\% bootstrap CIs over the evaluation windows ("
        + str(n_boot) + " resamples). "
        + "Ablations: " + BS + "textsc{ne} = no RNN encoder; "
        + BS + "textsc{q} = queue only; " + BS + "textsc{r} = return only, "
        + "so metric not applicable."
    )

    lines = [
        BS + "begin{table}[t]",
        BS + "caption{" + cap + "}",
        BS + "label{tab:headline}",
        BS + "centering", BS + "small",
        BS + "renewcommand{" + BS + "arraystretch}{1.25}",
        BS + "setlength{" + BS + "tabcolsep}{4pt}",
        BS + "resizebox{" + BS + "linewidth}{!}{%",
        BS + "begin{tabular}{" + cols + "}",
        BS + "toprule", head, BS + "midrule",
        multi_q,
    ]
    for key, label in QUEUE_BLOCK:
        lines.append(" & ".join(_row(label, key)) + EOL)
    lines.append(BS + "midrule")
    lines.append(multi_r)
    for key, label in PRICE_BLOCK:
        lines.append(" & ".join(_row(label, key)) + EOL)
    lines += [
        BS + "bottomrule", BS + "end{tabular}}",
        BS + "end{table}",
    ]
    return "\n".join(lines)


# ─── Coverage helpers (imported by notebooks/coverage_lobgrad.ipynb) ────
def _coverage(real, gen_paths, level):
    """Marginal coverage: fraction of cells where real ∈ [q_α/2, q_1-α/2].

    real:       shape (..., D)        — path axis absent
    gen_paths:  shape (..., N_PATHS, D)
    """
    alpha = 1 - level
    lo = np.quantile(gen_paths, alpha / 2,     axis=1)
    hi = np.quantile(gen_paths, 1 - alpha / 2, axis=1)
    return float(((real >= lo) & (real <= hi)).mean())


def coverage_with_bootstrap(real, gen_paths, levels, n_boot=1000, seed=1234):
    n_windows = real.shape[0]
    rng     = np.random.default_rng(seed)
    all_idx = rng.integers(0, n_windows, size=(n_boot, n_windows))
    rows = []
    for lvl in levels:
        point = _coverage(real, gen_paths, lvl)
        boot  = np.fromiter(
            (_coverage(real[idx], gen_paths[idx], lvl) for idx in all_idx),
            dtype=np.float64, count=n_boot,
        )
        rows.append({
            "nominal":   lvl,
            "empirical": point,
            "ci_lo":     float(np.percentile(boot, 2.5)),
            "ci_hi":     float(np.percentile(boot, 97.5)),
        })
    return rows


def render_coverage_table(cov_q, cov_p):
    body = []
    for cq, cp in zip(cov_q, cov_p):
        nominal = f"${int(cq['nominal']*100)}" + BS + "%$"
        q_main  = f"{cq['empirical']:.4f}"
        p_main  = f"{cp['empirical']:.4f}"
        q_cell  = _stack_cell(q_main, cq["ci_lo"], cq["ci_hi"])
        p_cell  = _stack_cell(p_main, cp["ci_lo"], cp["ci_hi"])
        body.append(" & ".join([nominal, q_cell, p_cell]) + EOL)

    cap = (
        "Empirical marginal coverage of JointTimeGrad's predictive intervals "
        "on the queue and price targets. Closer to the nominal level (left "
        "column) indicates better calibration. "
        "Sub-rows give 95\\% bootstrap CIs over windows."
    )
    lines = [
        BS + "begin{table}[t]", BS + "centering", BS + "small",
        BS + "renewcommand{" + BS + "arraystretch}{1.25}",
        BS + "setlength{" + BS + "tabcolsep}{6pt}",
        BS + "begin{tabular}{lrr}",
        BS + "toprule",
        "Nominal & Queue empirical & Price empirical " + EOL,
        BS + "midrule",
        *body,
        BS + "bottomrule", BS + "end{tabular}",
        BS + "caption{" + cap + "}",
        BS + "label{tab:coverage_jointtimegrad}", BS + "end{table}",
    ]
    return "\n".join(lines)


# ─── Bootstrap dispatcher (handles NO_QUEUE / HAS_PRICE branching) ──────
def _bootstrap_one(name, NO_QUEUE, HAS_PRICE,
                   real_q, real_a, gen_q, gen_mid,
                   n_boot, n_jobs, progress):
    if name in NO_QUEUE:
        if gen_mid is None:
            return None
        n = real_q.shape[0]
        gen_q_dummy = np.zeros((n, N_PATHS, HORIZON, 2 * K), dtype=np.float32)
        ci = bootstrap_metrics(
            real_q, gen_q_dummy, levels=K,
            real_prices=real_a, gen_prices=gen_mid,
            n_boot=n_boot, alpha=0.05, n_jobs=n_jobs,
            seed=1234, progress=progress,
        )
        for qk in QUEUE_KEYS:
            ci[qk] = {"mean": float("nan"),
                      "ci_lo": float("nan"), "ci_hi": float("nan"),
                      "samples": []}
        return ci
    kw = {}
    if name in HAS_PRICE and gen_mid is not None:
        kw["real_prices"] = real_a
        kw["gen_prices"]  = gen_mid
    return bootstrap_metrics(
        real_q, gen_q, levels=K,
        n_boot=n_boot, alpha=0.05, n_jobs=n_jobs,
        seed=1234, progress=progress, **kw,
    )


# ─── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot",    type=int, default=1000)
    parser.add_argument("--n-jobs",    type=int, default=-1)
    parser.add_argument("--out-dir",   type=str, default="figures/bootstrapped_tables")
    parser.add_argument("--cache-dir", type=str, default="cache/generations")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--no-cache",  action="store_true")
    args = parser.parse_args()

    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = None if args.no_cache else args.cache_dir
    t_start   = datetime.now()
    phase_t   = {}

    def _t(label):
        phase_t[label] = (datetime.now() - t_start).total_seconds()
        print(f"  [{label}] cumulative t = {phase_t[label] / 60:.1f} min")

    # ─── Phase 0: cache scan (data + gen short-circuits are independent) ─
    cached_labels  = _scan_gen_cache(cache_dir, args.seed,
                                     N_WINDOWS, N_PATHS, HORIZON)
    cached_targets = _try_load_eval_targets(cache_dir, args.seed,
                                            N_WINDOWS, HORIZON)
    skip_data    = cached_targets is not None
    skip_models  = bool(cached_labels)
    print(f"\n=== Phase 0: cache scan ===")
    print(f"  eval-targets cached : {skip_data}")
    print(f"  gen-paths cached    : {cached_labels if skip_models else '(none)'}")

    cfg = normalizer = X_train = ts_train = None

    # ─── Phase 1: data load ─────────────────────────────────────────────
    if skip_data:
        print("\n=== Phase 1: skipped (eval targets loaded from cache) ===")
        test_fut_q, test_fut_a, _ = cached_targets
        # ctx stubs — only consumed by _call_generate, which is bypassed on cache hit
        test_ctx_q = np.empty((N_WINDOWS, CONTEXT_LEN, 2 * K), dtype=np.float32)
        test_ctx_a = np.empty((N_WINDOWS, CONTEXT_LEN), dtype=np.float64)
    else:
        print("\n=== Phase 1: data load ===")
        cfg = BenchmarkConfig(
            levels=K, price_level=PRICE_LEVEL,
            queue_repr="signed", queue_mode="absolute",
            timegrad_checkpoint=TG_CKPT, timegrad_train=False,
            timegrad_k=DIFFUSION_K, timegrad_hidden_enc=HIDDEN_ENC,
            timegrad_hidden_noise=HIDDEN_NOISE, timegrad_num_enc_layers=NUM_ENC_LAYERS,
            timegrad_emb_dim=EMB_DIM, timegrad_cell_type=CELL_TYPE,
            timegrad_use_cond_projection=USE_COND_PROJ,
            timegrad_loss_type=LOSS_TYPE, timegrad_schedule="linear",
            timegrad_beta_start=0.0001, timegrad_beta_end=0.1,
        )
        (ts_train, X_train, anchors_train), \
        (ts_val,   X_val,   anchors_val),   \
        (ts_test,  X_test,  anchors_test),  \
        normalizer = date_splits_joint(cfg)

        rng_windows = np.random.default_rng(42)
        test_ctx_q, test_fut_q, test_ctx_a, test_fut_a = sample_eval_windows_joint(
            X_test, anchors_test, CONTEXT_LEN, HORIZON, N_WINDOWS, rng=rng_windows,
        )
        print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}  Windows: {N_WINDOWS}")
        _save_eval_targets(cache_dir, args.seed, N_WINDOWS, HORIZON,
                           test_fut_q, test_fut_a, test_ctx_a)
    _t("data")

    # ─── Phase 2: build models ──────────────────────────────────────────
    if skip_models:
        print("\n=== Phase 2: skipped (using cached generations) ===")
        MODELS    = {n: None for n in cached_labels}
        HAS_PRICE = HAS_PRICE_BASE & set(cached_labels)
        NO_QUEUE  = NO_QUEUE_BASE  & set(cached_labels)
    else:
        if cfg is None:
            raise RuntimeError(
                "Cannot build models: eval-targets cached but no gen-paths "
                "cache to fall back on. Delete the eval-targets cache or "
                "regenerate generations."
            )
        print("\n=== Phase 2: build models ===")
        MODELS, HAS_PRICE, NO_QUEUE = build_models(cfg, normalizer, X_train, ts_train)
    _t("build_models")

    # ─── Phase 3: generate paths ────────────────────────────────────────
    print("\n=== Phase 3: generate paths ===")
    gen_long, gen_mid_long = {}, {}
    for name, mdl in MODELS.items():
        q, mid = generate_paths(
            name, mdl, test_ctx_q, test_ctx_a,
            N_WINDOWS, N_PATHS, HORIZON, args.seed,
            cache_dir=cache_dir,
        )
        if q   is not None: gen_long[name]     = q
        if mid is not None: gen_mid_long[name] = mid
    _t("generate")

    # ─── Phase 4: bootstrap Table 1 ─────────────────────────────────────
    print("\n=== Phase 4: bootstrap Table 1 ===")
    results_ci = {}
    for name in MODELS:
        print(f"  bootstrapping {name} ...")
        ci = _bootstrap_one(
            name, NO_QUEUE, HAS_PRICE,
            real_q=test_fut_q, real_a=test_fut_a,
            gen_q=gen_long.get(name),
            gen_mid=gen_mid_long.get(name),
            n_boot=args.n_boot, n_jobs=args.n_jobs, progress=True,
        )
        if ci is not None:
            results_ci[name] = ci

    if TG_LABEL in results_ci:
        print(f"\n--- {TG_LABEL} headline bootstrap (sanity check) ---")
        for key, label in QUEUE_BLOCK + PRICE_BLOCK:
            mean, lo, hi = _ci_triple(results_ci, TG_LABEL, key)
            if mean is None:
                print(f"  {key:>22s}:  ---")
            else:
                print(f"  {key:>22s}:  {mean:.4f}  [{lo:.4f}, {hi:.4f}]")

    print("\nWriting Table 1 ...")
    table1_tex = render_table_1(results_ci, list(MODELS), HORIZON,
                                n_boot=args.n_boot, n_windows=N_WINDOWS)
    (out_dir / "table_headline.tex").write_text(table1_tex)

    rows = []
    for name in MODELS:
        for key, _label in QUEUE_BLOCK + PRICE_BLOCK:
            band = results_ci.get(name, {}).get(key) or {}
            full_mean = band.get("mean")
            samples   = band.get("samples") or []
            median    = float(np.median(samples)) if len(samples) else None
            mean, lo, hi = _ci_triple(results_ci, name, key)  # display point
            rows.append({
                "model":   name,
                "metric":  key,
                "mean":    float(full_mean) if full_mean is not None
                           and np.isfinite(full_mean) else None,
                "median":  median,
                "display": mean,    # what the LaTeX table prints
                "ci_lo":   lo,
                "ci_hi":   hi,
            })
    pd.DataFrame(rows).to_csv(out_dir / "table_headline.csv", index=False)
    print(f"  saved {out_dir}/table_headline.{{csv,tex}}")
    _t("table_1_bootstrap")

    # ─── Manifest ───────────────────────────────────────────────────────
    elapsed = (datetime.now() - t_start).total_seconds()
    manifest = {
        "started_at":       t_start.isoformat(),
        "elapsed_sec":      elapsed,
        "phase_cumulative": phase_t,
        "args":             vars(args),
        "models":           list(MODELS),
        "n_windows":        N_WINDOWS,
        "n_paths":          N_PATHS,
        "horizon":          HORIZON,
        "context_len":      CONTEXT_LEN,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n=== Done in {elapsed/60:.1f} min ===")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()

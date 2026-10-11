"""
run_hpo.py
----------
Runs TimeGrad HPO then CGAN HPO sequentially.
Data is loaded once and shared between both runs.

Usage:
    python run_hpo.py

Both studies are resume-safe via SQLite:
    checkpoints/optuna_v2/hpo_optuna.db
    checkpoints/cgan_optuna_v2/hpo_cgan.db
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

# ── Package root ──────────────────────────────────────────────────────────────

def _find_root() -> Path:
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "lobgrad" / "__init__.py").is_file():
            return p
        nested = p / "LobBench_comparison"
        if (nested / "lobgrad" / "__init__.py").is_file():
            return nested
    raise RuntimeError("Could not find lobgrad package")

root = _find_root()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import numpy as np
import torch
import optuna
from optuna.pruners import MedianPruner

optuna.logging.set_verbosity(optuna.logging.WARNING)

from lobgrad import BenchmarkConfig
from lobgrad.data.splits import date_splits, sample_eval_windows_joint
from lobgrad.execution.simulate import JointTimeGradGenerator
from lobgrad.models.cgan import CGANGenerator
from lobgrad.eval.evaluate import evaluate_model_pooled

# ── Shared constants ──────────────────────────────────────────────────────────

RUN_CGAN = True   # set False to skip CGAN and save ~11h
RUN_TIMEGRAD = False   # set False to skip TimeGrad and save ~11h

SEED            = 42
CONTEXT_LEN     = 100
HORIZON         = 20     # 20 steps × 5 s = 100 s ≈ 1.7 min
K               = 5
PRICE_LEVEL     = 2.0
EMB_DIM         = 32

HPO_N_WINDOWS    = 20    # Stage 1: fast CRPS eval per trial
HPO_N_PATHS      = 20   # bumped 10→20 for lower-variance CRPS estimates
STAGE2_N_WINDOWS = 100   # Stage 2: re-rank candidates (non-overlapping with Stage 1)
STAGE2_N_PATHS   = 30
FINAL_N_WINDOWS  = 200   # Stage 3: final test evaluation
FINAL_N_PATHS    = 50

# Balanced 50 % queue / 50 % price — all metrics from evaluate_model_pooled
SCORE_WEIGHTS = {
    # Queue (50 %)
    "crps":                0.15,
    "l1_volume":           0.10,
    "queue_profile":       0.10,
    "ob_imbalance":        0.05,
    "queue_autocorr":      0.10,
    # Price / strategy (50 %)
    "price_crps":          0.20,
    "return_dist":         0.15,
    "abs_return_autocorr": 0.15,
}

# ── Shared helpers ────────────────────────────────────────────────────────────

def composite_score(metrics: dict) -> float:
    total, weight_sum = 0.0, 0.0
    for m, w in SCORE_WEIGHTS.items():
        if m not in metrics:
            continue
        v = metrics[m].get("mean", float("nan"))
        if not np.isfinite(v):
            continue
        total      += w * v
        weight_sum += w
    if weight_sum == 0.0:
        return float("inf")
    return total / weight_sum


def rank_select(scored: dict) -> str:
    """
    Select the key with the lowest mean rank across SCORE_WEIGHTS metrics.
    Scale-invariant: each metric gets one vote regardless of its magnitude.

    scored : {key: {metric_name: float}}  — one float per metric per candidate
    Returns the key (str) with the best (lowest) mean rank.
    """
    keys = list(scored.keys())
    if len(keys) == 1:
        return keys[0]

    cumulative_ranks: dict = {k: 0 for k in keys}
    n_metrics = 0

    for metric in SCORE_WEIGHTS:
        vals = [(k, scored[k].get(metric, float("nan"))) for k in keys]
        finite = [(k, v) for k, v in vals if np.isfinite(v)]
        if len(finite) < 2:
            continue
        finite.sort(key=lambda x: x[1])           # lower value = better rank
        for rank, (k, _) in enumerate(finite, 1):
            cumulative_ranks[k] += rank
        present = {k for k, _ in finite}
        for k in keys:
            if k not in present:
                cumulative_ranks[k] += len(keys) + 1   # NaN → worst rank
        n_metrics += 1

    if n_metrics == 0:
        return keys[0]
    return min(keys, key=lambda k: cumulative_ranks[k] / n_metrics)


def print_section(title: str) -> None:
    width = 72
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def print_param_importances(study: optuna.Study) -> None:
    try:
        importances = optuna.importance.get_param_importances(study)
        print("Parameter importances:")
        for param, imp in sorted(importances.items(), key=lambda x: -x[1]):
            bar = "█" * int(imp * 40)
            print(f"  {param:25s}: {imp:.3f}  {bar}")
    except Exception as e:
        print(f"  (importance analysis skipped: {e})")


# ── Data loading (shared) ─────────────────────────────────────────────────────

def load_data():
    print_section("Loading data")
    base_cfg = BenchmarkConfig(
        levels      = K,
        price_level = PRICE_LEVEL,
        queue_repr  = "signed",
        queue_mode  = "absolute",
        context_len = CONTEXT_LEN,
        horizon     = HORIZON,
    )

    (ts_train, X_train, anc_train), \
    (ts_val,   X_val,   anc_val),   \
    (ts_test,  X_test,  anc_test),  \
    normalizer = date_splits(base_cfg)

    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,} snapshots")

    rng_val = np.random.default_rng(SEED)
    val_ctx_q, val_fut_q, val_ctx_a, val_fut_a = sample_eval_windows_joint(
        X_val, anc_val, CONTEXT_LEN, HORIZON,
        n_windows=HPO_N_WINDOWS + STAGE2_N_WINDOWS, rng=rng_val,   # non-overlapping slices
    )
    print(f"Val windows:  ctx={val_ctx_q.shape}  fut={val_fut_q.shape}")

    rng_test = np.random.default_rng(SEED)
    test_ctx_q, test_fut_q, test_ctx_a, test_fut_a = sample_eval_windows_joint(
        X_test, anc_test, CONTEXT_LEN, HORIZON,
        n_windows=FINAL_N_WINDOWS, rng=rng_test,
    )
    print(f"Test windows: ctx={test_ctx_q.shape}  fut={test_fut_q.shape}")

    return (
        ts_train, X_train, anc_train,
        val_ctx_q, val_fut_q, val_ctx_a, val_fut_a,
        test_ctx_q, test_fut_q, test_ctx_a, test_fut_a,
        normalizer,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TimeGrad HPO
# ─────────────────────────────────────────────────────────────────────────────

def tg_generate_paths(model, ctx_q, ctx_a, horizon, n_paths):
    queue_paths, price_paths = [], []
    for i in range(len(ctx_q)):
        q, p = model.generate(ctx_q[i], ctx_a[i], steps=horizon, n_paths=n_paths)
        queue_paths.append(q)
        price_paths.append(p)
    return np.stack(queue_paths), np.stack(price_paths)


def tg_trial_to_cfg(trial, epochs, max_batches,
                    save_root="checkpoints/optuna_v2", train=True):
    p          = trial.params
    noise_type = p["noise_type"]
    loss_type  = p["loss_type"]
    save_dir   = os.path.join(save_root, f"trial_{trial.number}")
    ckpt       = os.path.join(save_dir, "best_model.pt")
    return BenchmarkConfig(
        levels                        = K,
        price_level                   = PRICE_LEVEL,
        queue_repr                    = "signed",
        queue_mode                    = "absolute",
        context_len                   = CONTEXT_LEN,
        horizon                       = HORIZON,
        timegrad_train                = train,
        timegrad_k                    = p["k"],
        timegrad_schedule             = p["schedule"],
        timegrad_hidden_enc           = p["hidden_enc"],
        timegrad_hidden_noise         = (p["hidden_noise"] if noise_type == "mlp" else 64),
        timegrad_num_enc_layers       = p["num_layers"],
        timegrad_emb_dim              = EMB_DIM,
        timegrad_cell_type            = p["cell_type"],
        timegrad_loss_type            = loss_type,
        timegrad_dropout              = p["dropout"],
        timegrad_noise_type           = noise_type,
        timegrad_wavenet_res_layers   = (p["wavenet_layers"]   if noise_type == "wavenet" else 8),
        timegrad_wavenet_res_channels = (p["wavenet_channels"] if noise_type == "wavenet" else 8),
        timegrad_history_length       = CONTEXT_LEN,
        timegrad_prediction_length    = HORIZON,
        timegrad_batch_size           = 256,
        timegrad_lr                   = p["lr"],
        timegrad_epochs               = epochs,
        timegrad_max_batches          = max_batches,
        timegrad_checkpoint           = ckpt,
        timegrad_save_dir             = save_dir,
    )


def run_timegrad_hpo(data):
    ts_train, X_train, anc_train, \
    val_ctx_q, val_fut_q, val_ctx_a, val_fut_a, \
    test_ctx_q, test_fut_q, test_ctx_a, test_fut_a, \
    normalizer = data

    # ── Stage 1: single study, loss_type as hyperparameter (90 trials) ──────────
    print_section("TimeGrad — Stage 1: HPO screening (90 trials, loss_type as hyperparameter)")

    os.makedirs("checkpoints/optuna_v2", exist_ok=True)
    N_TRIALS = 90

    study = optuna.create_study(
        study_name     = "timegrad_hpo_v2",
        storage        = "sqlite:///checkpoints/optuna_v2/hpo_optuna.db",
        direction      = "minimize",
        pruner         = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1),
        load_if_exists = True,
    )
    n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Completed trials: {n_done} / {N_TRIALS}")

    def tg_objective(trial: optuna.Trial) -> float:
        noise_type = trial.suggest_categorical("noise_type", ["mlp", "wavenet"])
        loss_type  = trial.suggest_categorical("loss_type",  ["l1", "l2", "huber"])
        save_dir   = os.path.join("checkpoints", "optuna_v2", f"trial_{trial.number}")
        ckpt       = os.path.join(save_dir, "best_model.pt")

        # Always suggest all parameters so TPE can model their distributions
        # regardless of noise_type — parameters are applied conditionally below.
        _hidden_noise     = trial.suggest_int("hidden_noise",     32, 256, log=True)
        _wavenet_layers   = trial.suggest_int("wavenet_layers",    4,  16)
        _wavenet_channels = trial.suggest_int("wavenet_channels",  4,  16)

        cfg = BenchmarkConfig(
            levels                        = K,
            price_level                   = PRICE_LEVEL,
            queue_repr                    = "signed",
            queue_mode                    = "absolute",
            context_len                   = CONTEXT_LEN,
            horizon                       = HORIZON,
            timegrad_train                = True,
            timegrad_k                    = trial.suggest_int("k", 50, 200, log=True),
            timegrad_schedule             = trial.suggest_categorical(
                                               "schedule", ["linear", "quad", "sigmoid", "cosine"]),
            timegrad_hidden_enc           = trial.suggest_int("hidden_enc", 32, 256, log=True),
            timegrad_hidden_noise         = (_hidden_noise   if noise_type == "mlp"     else 64),
            timegrad_num_enc_layers       = trial.suggest_int("num_layers", 1, 3),
            timegrad_emb_dim              = EMB_DIM,
            timegrad_cell_type            = trial.suggest_categorical("cell_type", ["GRU", "LSTM"]),
            timegrad_loss_type            = loss_type,
            timegrad_dropout              = trial.suggest_float("dropout", 0.0, 0.2),
            timegrad_noise_type           = noise_type,
            timegrad_wavenet_res_layers   = (_wavenet_layers   if noise_type == "wavenet" else 8),
            timegrad_wavenet_res_channels = (_wavenet_channels if noise_type == "wavenet" else 8),
            timegrad_history_length       = CONTEXT_LEN,
            timegrad_prediction_length    = HORIZON,
            timegrad_batch_size           = 64,
            timegrad_lr                   = trial.suggest_float("lr", 5e-4, 5e-3, log=True),
            timegrad_epochs               = 25,
            timegrad_max_batches          = 300,
            timegrad_checkpoint           = ckpt,
            timegrad_save_dir             = save_dir,
        )

        model = JointTimeGradGenerator(cfg, normalizer=normalizer)
        model.fit(X_train, anc_train, ts_train, optuna_trial=trial)

        # Objective: composite CRPS on a small validation slice.
        # Returns penalty score (10.0) if generation fails (e.g. OOM).
        try:
            gen_q, gen_p = tg_generate_paths(
                model,
                val_ctx_q[:HPO_N_WINDOWS],
                val_ctx_a[:HPO_N_WINDOWS],
                HORIZON,
                HPO_N_PATHS,
            )
            res    = evaluate_model_pooled(
                val_fut_q[:HPO_N_WINDOWS], gen_q, levels=K,
                real_prices=val_fut_a[:HPO_N_WINDOWS], gen_prices=gen_p,
            )
            crps_q = res["crps"]["mean"]
            crps_p = res.get("price_crps", {}).get("mean", crps_q)
            return 0.5 * crps_q + 0.5 * crps_p
        except Exception as e:
            print(f"  [warn] CRPS eval failed ({e}), returning penalty score")
            return 10.0   # clearly worse than any valid CRPS (~0.07–0.09)

    n_remaining = max(0, N_TRIALS - n_done)
    if n_remaining > 0:
        study.optimize(tg_objective, n_trials=n_remaining, show_progress_bar=True)
    else:
        print("Stage 1 already complete, skipping.")

    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if n_complete > 0:
        print(f"Best Stage-1 score: {study.best_value:.6f}  params: {study.best_params}")
    else:
        print("Warning: no completed trials")

    # ── Stage 2: re-rank top-8 by rank-based selection ───────────────────────
    print_section("TimeGrad — Stage 2: re-rank top-8 by mean rank (rank-based)")

    STAGE2_CACHE = "checkpoints/optuna_v2/stage2_scores.json"
    # cache format: {key: {"composite": float, "metrics_means": {metric: float}}}
    composite_scores: dict = {}
    if os.path.exists(STAGE2_CACHE):
        with open(STAGE2_CACHE) as f:
            raw = json.load(f)
        # migrate old flat-float format if needed
        composite_scores = {
            k: (v if isinstance(v, dict) else {"composite": v, "metrics_means": {}})
            for k, v in raw.items()
        }
        print(f"Loaded {len(composite_scores)} cached Stage-2 entries from {STAGE2_CACHE}")

    completed      = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top8_trials    = sorted(completed, key=lambda t: t.value)[:8]
    print(f"Candidates: {len(top8_trials)} trials")

    # Non-overlapping slice: Stage 1 used val[:HPO_N_WINDOWS], Stage 2 uses the next block
    s2_lo, s2_hi = HPO_N_WINDOWS, HPO_N_WINDOWS + STAGE2_N_WINDOWS
    s2_ctx_q = val_ctx_q[s2_lo:s2_hi]
    s2_fut_q = val_fut_q[s2_lo:s2_hi]
    s2_ctx_a = val_ctx_a[s2_lo:s2_hi]
    s2_fut_a = val_fut_a[s2_lo:s2_hi]

    for t in top8_trials:
        cache_key = f"trial_{t.number}"
        if cache_key in composite_scores:
            score = composite_scores[cache_key].get("composite", float("nan"))
            print(f"Trial {t.number:3d}: cached composite={score:.4f}")
            continue
        save_dir = os.path.join("checkpoints", "optuna_v2", f"trial_{t.number}")
        ckpt     = os.path.join(save_dir, "best_model.pt")
        if not os.path.exists(ckpt):
            print(f"Trial {t.number}: checkpoint missing, skipping")
            continue
        cfg = tg_trial_to_cfg(t, epochs=25, max_batches=300,
                              save_root="checkpoints/optuna_v2", train=False)
        cfg.timegrad_checkpoint = ckpt
        cfg.timegrad_save_dir   = save_dir
        t0  = time.time()
        mdl = JointTimeGradGenerator(cfg, normalizer=normalizer)
        gen_q, gen_p = tg_generate_paths(mdl, s2_ctx_q, s2_ctx_a, HORIZON, STAGE2_N_PATHS)
        metrics = evaluate_model_pooled(
            s2_fut_q, gen_q, levels=K,
            real_prices=s2_fut_a, gen_prices=gen_p,
        )
        score = composite_score(metrics)
        composite_scores[cache_key] = {
            "composite":     score,
            "metrics_means": {m: float(metrics[m]["mean"]) for m in SCORE_WEIGHTS
                              if m in metrics and np.isfinite(metrics[m]["mean"])},
        }
        with open(STAGE2_CACHE, "w") as f:
            json.dump(composite_scores, f)
        loss_type = t.params.get("loss_type", "?")
        print(f"Trial {t.number:3d} [{loss_type:5s}]: stage1={t.value:.5f}  composite={score:.4f}  ({time.time()-t0:.0f}s)")

    if not composite_scores:
        print("No Stage-2 scores computed (all checkpoints missing?). Aborting.")
        return

    # Rank-based selection: each metric gets one vote, scale-invariant
    metric_means = {k: v["metrics_means"] for k, v in composite_scores.items()}
    best_key     = rank_select(metric_means)
    trial_num    = int(best_key.split("_")[1])
    best_rank_score = composite_scores[best_key].get("composite", float("nan"))
    print(f"\nBest by mean rank: trial_{trial_num}  composite={best_rank_score:.4f}")

    # ── Stage 3: deep train top-1 ─────────────────────────────────────────────
    print_section("TimeGrad — Stage 3: deep training top-1")

    STAGE3_CACHE = "checkpoints/optuna_deep_v2/stage3_results.json"
    stage3_results = {}
    if os.path.exists(STAGE3_CACHE):
        with open(STAGE3_CACHE) as f:
            stage3_results = json.load(f)
        print(f"Loaded {len(stage3_results)} cached Stage-3 results from {STAGE3_CACHE}")

    cache_key = f"trial_{trial_num}"
    if cache_key in stage3_results:
        print(f"Trial {trial_num}: cached composite={stage3_results[cache_key]['composite']:.4f}")
    else:
        t         = next(tr for tr in study.trials if tr.number == trial_num)
        deep_dir  = os.path.join("checkpoints/optuna_deep_v2", f"trial_{trial_num}")
        deep_ckpt = os.path.join(deep_dir, "best_model.pt")

        # Warm-start from Stage 1 checkpoint when deep checkpoint doesn't exist yet
        stage1_ckpt = os.path.join("checkpoints", "optuna_v2", f"trial_{trial_num}", "best_model.pt")
        init_ckpt   = deep_ckpt if os.path.exists(deep_ckpt) else stage1_ckpt

        cfg = tg_trial_to_cfg(
            t, epochs=50, max_batches=1000,
            save_root="checkpoints/optuna_deep_v2", train=not os.path.exists(deep_ckpt),
        )
        cfg.timegrad_checkpoint = init_ckpt
        cfg.timegrad_save_dir   = deep_dir

        t0  = time.time()
        mdl = JointTimeGradGenerator(cfg, normalizer=normalizer)
        if os.path.exists(deep_ckpt):
            print(f"\n=== Trial {trial_num}: loading saved deep checkpoint ===")
        else:
            print(f"\n=== Deep training trial {trial_num} ===")
            mdl.fit(X_train, anc_train, ts_train)

        gen_q, gen_p = tg_generate_paths(mdl, test_ctx_q, test_ctx_a, HORIZON, FINAL_N_PATHS)
        metrics = evaluate_model_pooled(
            test_fut_q, gen_q, levels=K,
            real_prices=test_fut_a, gen_prices=gen_p,
        )
        score   = composite_score(metrics)
        dir_acc = metrics.get("direction_accuracy", {}).get("mean", float("nan"))

        stage3_results[cache_key] = {
            "composite":          score,
            "direction_accuracy": dir_acc,
            "metrics":            metrics,
        }
        os.makedirs("checkpoints/optuna_deep_v2", exist_ok=True)
        with open(STAGE3_CACHE, "w") as f:
            json.dump(stage3_results, f)
        print(f"Trial {trial_num}: composite={score:.4f}  dir_acc={dir_acc:.3f}  ({time.time()-t0:.0f}s)")

    print("\n── TimeGrad final results ──")
    r      = stage3_results[f"trial_{trial_num}"]
    best_t = next(tr for tr in study.trials if tr.number == trial_num)
    print(f"\nBest: trial_{trial_num}  composite={r['composite']:.4f}  direction_accuracy={r['direction_accuracy']:.3f}")
    print("Hyperparameters:")
    for k, v in best_t.params.items():
        print(f"  {k:30s}: {v}")

    print_param_importances(study)

    print("\nPer-metric scores (lower = better):")
    for metric in SCORE_WEIGHTS:
        if metric in r["metrics"]:
            print(f"  {metric:22s}: {r['metrics'][metric]['mean']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CGAN HPO
# ─────────────────────────────────────────────────────────────────────────────

def cgan_generate_paths(model, ctx_q, ctx_a, horizon, n_paths):
    queue_paths, price_paths = [], []
    for i in range(len(ctx_q)):
        q, p = model.generate(ctx_q[i], steps=horizon, n_paths=n_paths,
                              context_anchors=ctx_a[i])
        queue_paths.append(q)
        price_paths.append(p)
    return np.stack(queue_paths), np.stack(price_paths)


def cgan_trial_to_cfg(trial, epochs, max_batches, save_root="checkpoints/cgan_optuna_v2"):
    p        = trial.params
    save_dir = os.path.join(save_root, f"trial_{trial.number}")
    ckpt     = os.path.join(save_dir, "cgan.pt")
    cfg = BenchmarkConfig(
        levels                = K,
        price_level           = PRICE_LEVEL,
        queue_repr            = "signed",
        queue_mode            = p["queue_mode"],
        context_len           = CONTEXT_LEN,
        horizon               = HORIZON,
        cgan_latent_dim       = p["latent_dim"],
        cgan_hidden_dim       = p["hidden_dim"],
        cgan_n_encoder_layers = p["n_encoder_layers"],
        cgan_n_decoder_layers = p["n_decoder_layers"],
        cgan_lr_g             = p["lr_g"],
        cgan_lr_d             = p["lr_d"],
        cgan_n_critic         = p["n_critic"],
        cgan_lambda_gp        = p["lambda_gp"],
        cgan_batch_size       = 64,
        cgan_epochs           = epochs,
        cgan_max_batches      = max_batches,
        cgan_stride           = p["stride"],
    )
    return cfg, ckpt, save_dir


def run_cgan_hpo(data):
    ts_train, X_train, anc_train, \
    val_ctx_q, val_fut_q, val_ctx_a, val_fut_a, \
    test_ctx_q, test_fut_q, test_ctx_a, test_fut_a, \
    normalizer = data

    # ── Stage 1: screening ────────────────────────────────────────────────────
    print_section("CGAN — Stage 1: HPO screening (50 trials)")

    os.makedirs("checkpoints/cgan_optuna_v2", exist_ok=True)
    study = optuna.create_study(
        study_name     = "cgan_hpo_v2",
        storage        = "sqlite:///checkpoints/cgan_optuna_v2/hpo_cgan.db",
        direction      = "minimize",
        load_if_exists = True,
    )
    print(f"Completed trials so far: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

    def cgan_objective(trial: optuna.Trial) -> float:
        save_dir = os.path.join("checkpoints", "cgan_optuna_v2", f"trial_{trial.number}")
        ckpt     = os.path.join(save_dir, "cgan.pt")
        os.makedirs(save_dir, exist_ok=True)

        cfg = BenchmarkConfig(
            levels                = K,
            price_level           = PRICE_LEVEL,
            queue_repr            = "signed",
            queue_mode            = trial.suggest_categorical("queue_mode", ["absolute", "delta"]),
            context_len           = CONTEXT_LEN,
            horizon               = HORIZON,
            cgan_latent_dim       = trial.suggest_int("latent_dim",       32,  256, log=True),
            cgan_hidden_dim       = trial.suggest_int("hidden_dim",       64,  512, log=True),
            cgan_n_encoder_layers = trial.suggest_int("n_encoder_layers",  1,    3),
            cgan_n_decoder_layers = trial.suggest_int("n_decoder_layers",  1,    3),
            cgan_lr_g             = trial.suggest_float("lr_g",       1e-5, 1e-3, log=True),
            cgan_lr_d             = trial.suggest_float("lr_d",       1e-5, 1e-3, log=True),
            cgan_n_critic         = trial.suggest_int("n_critic",          1,    3),
            cgan_lambda_gp        = trial.suggest_float("lambda_gp",  1.0, 50.0, log=True),
            cgan_batch_size       = 64,
            cgan_epochs           = 15,
            cgan_max_batches      = 100,
            cgan_stride           = trial.suggest_categorical("stride", [5, 10, 20]),
        )

        model = CGANGenerator(cfg)
        model.fit(X_train, ts_train, anchors=anc_train)
        model.save(ckpt)

        gen_q, gen_p = cgan_generate_paths(
            model,
            val_ctx_q[:HPO_N_WINDOWS],
            val_ctx_a[:HPO_N_WINDOWS],
            HORIZON,
            HPO_N_PATHS,
        )
        metrics = evaluate_model_pooled(
            val_fut_q[:HPO_N_WINDOWS], gen_q, levels=K,
            real_prices=val_fut_a[:HPO_N_WINDOWS], gen_prices=gen_p,
        )
        return composite_score(metrics)

    N_CGAN_TRIALS    = 50
    n_cgan_done      = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_cgan_remaining = max(0, N_CGAN_TRIALS - n_cgan_done)
    if n_cgan_remaining > 0:
        study.optimize(cgan_objective, n_trials=n_cgan_remaining, show_progress_bar=True)
    else:
        print("Stage 1 already complete, skipping training.")
    n_cgan_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if n_cgan_complete > 0:
        print(f"\nBest composite score : {study.best_value:.6f}")
        print(f"Best params          : {study.best_params}")
    else:
        print("\nNo completed CGAN trials.")

    # ── Stage 2: deep train top-3, select on val (non-overlapping) ───────────
    print_section("CGAN — Stage 2: deep training top-3, selection on val set")

    os.makedirs("checkpoints/cgan_optuna_deep_v2", exist_ok=True)

    # Non-overlapping val slice: Stage 1 used val[:HPO_N_WINDOWS]
    cs2_lo, cs2_hi = HPO_N_WINDOWS, HPO_N_WINDOWS + STAGE2_N_WINDOWS
    cs2_ctx_q = val_ctx_q[cs2_lo:cs2_hi]
    cs2_fut_q = val_fut_q[cs2_lo:cs2_hi]
    cs2_ctx_a = val_ctx_a[cs2_lo:cs2_hi]
    cs2_fut_a = val_fut_a[cs2_lo:cs2_hi]

    CGAN_S2_CACHE = "checkpoints/cgan_optuna_deep_v2/stage2_val_results.json"
    stage2_val: dict = {}
    if os.path.exists(CGAN_S2_CACHE):
        with open(CGAN_S2_CACHE) as f:
            stage2_val = {int(k): v for k, v in json.load(f).items()}
        print(f"Loaded {len(stage2_val)} cached Stage-2 val results")

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top3_trials      = sorted(completed_trials, key=lambda t: t.value)[:3]

    for t in top3_trials:
        if t.number in stage2_val:
            print(f"Trial {t.number}: cached val composite={stage2_val[t.number]['composite']:.4f}")
            continue

        cfg, _, save_dir = cgan_trial_to_cfg(
            t, epochs=80, max_batches=300, save_root="checkpoints/cgan_optuna_deep_v2"
        )
        ckpt = os.path.join(save_dir, "cgan.pt")
        os.makedirs(save_dir, exist_ok=True)

        t0  = time.time()
        mdl = CGANGenerator(cfg)
        if os.path.exists(ckpt):
            print(f"\n=== Trial {t.number}: loading saved deep checkpoint ===")
            mdl.load(ckpt)
        else:
            print(f"\n=== Deep training trial {t.number} (screening={t.value:.4f}) ===")
            mdl.fit(X_train, ts_train, anchors=anc_train)
            mdl.save(ckpt)

        # Evaluate on val — used for selection only, not reporting
        gen_q, gen_p = cgan_generate_paths(mdl, cs2_ctx_q, cs2_ctx_a, HORIZON, STAGE2_N_PATHS)
        metrics = evaluate_model_pooled(
            cs2_fut_q, gen_q, levels=K,
            real_prices=cs2_fut_a, gen_prices=gen_p,
        )
        score = composite_score(metrics)
        stage2_val[t.number] = {
            "composite":     score,
            "metrics_means": {m: float(metrics[m]["mean"]) for m in SCORE_WEIGHTS
                              if m in metrics and np.isfinite(metrics[m]["mean"])},
        }
        with open(CGAN_S2_CACHE, "w") as f:
            json.dump(stage2_val, f)
        print(f"Trial {t.number}: val composite={score:.4f}  ({time.time()-t0:.0f}s)")

    # Rank-based selection on val metrics
    cgan_metric_means = {k: v["metrics_means"] for k, v in stage2_val.items()}
    best_num          = rank_select(cgan_metric_means)
    print(f"\nBest by val mean rank: trial {best_num}")

    # ── Stage 3: evaluate best CGAN on test (reporting only, no selection) ────
    print_section("CGAN — Stage 3: final evaluation on test set")

    CGAN_S3_CACHE = "checkpoints/cgan_optuna_deep_v2/stage3_test_results.json"
    stage3_test: dict = {}
    if os.path.exists(CGAN_S3_CACHE):
        with open(CGAN_S3_CACHE) as f:
            stage3_test = {int(k): v for k, v in json.load(f).items()}
        print(f"Loaded {len(stage3_test)} cached Stage-3 test results")

    if best_num not in stage3_test:
        cfg, _, save_dir = cgan_trial_to_cfg(
            next(tr for tr in study.trials if tr.number == best_num),
            epochs=80, max_batches=300, save_root="checkpoints/cgan_optuna_deep_v2"
        )
        ckpt = os.path.join(save_dir, "cgan.pt")
        mdl  = CGANGenerator(cfg)
        mdl.load(ckpt)

        t0 = time.time()
        gen_q, gen_p = cgan_generate_paths(mdl, test_ctx_q, test_ctx_a, HORIZON, FINAL_N_PATHS)
        metrics  = evaluate_model_pooled(
            test_fut_q, gen_q, levels=K,
            real_prices=test_fut_a, gen_prices=gen_p,
        )
        score   = composite_score(metrics)
        dir_acc = metrics.get("direction_accuracy", {}).get("mean", float("nan"))
        stage3_test[best_num] = {
            "composite":          score,
            "direction_accuracy": dir_acc,
            "metrics":            metrics,
        }
        with open(CGAN_S3_CACHE, "w") as f:
            json.dump(stage3_test, f)
        print(f"Trial {best_num}: test composite={score:.4f}  dir_acc={dir_acc:.3f}  ({time.time()-t0:.0f}s)")
    else:
        r = stage3_test[best_num]
        print(f"Trial {best_num}: cached test composite={r['composite']:.4f}  dir_acc={r.get('direction_accuracy', float('nan')):.3f}")

    r      = stage3_test[best_num]
    best_t = next(tr for tr in study.trials if tr.number == best_num)
    print(f"\nBest CGAN: trial {best_num}  composite={r['composite']:.4f}  direction_accuracy={r.get('direction_accuracy', float('nan')):.3f}")
    print("Hyperparameters:")
    for k, v in best_t.params.items():
        print(f"  {k:25s}: {v}")

    print_param_importances(study)

    print("\nPer-metric scores (lower = better):")
    for metric in SCORE_WEIGHTS:
        if metric in r["metrics"]:
            print(f"  {metric:22s}: {r['metrics'][metric]['mean']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_total = time.time()

    data = load_data()

    if RUN_TIMEGRAD:
        run_timegrad_hpo(data)
    else:
        print("\n[TimeGrad skipped — set RUN_TIMEGRAD = True to enable]")

    if RUN_CGAN:
        run_cgan_hpo(data)
    else:
        print("\n[CGAN skipped — set RUN_CGAN = True to enable]")

    elapsed = time.time() - t_total
    print_section(f"All done — total time: {elapsed/3600:.1f}h")

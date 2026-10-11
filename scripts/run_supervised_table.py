"""
run_supervised_table.py
-----------------------
Train the supervised-model benchmarks (DeepLOB, TransLOB, LOBS5) adapted to our
regression-forecasting framework (Gaussian probabilistic head, 20-step
autoregressive rollout), score them with the SAME distributional metrics as the
existing tables — but **point estimates, no bootstrap/CIs** (per request) — and
merge the 3 columns into the existing 8-column PTM tables to produce an
11-column comparison table per dataset.

Datasets: BTC, MSFT, GOOGL. Same seed-42 / 500-window / 100-path / horizon-20
protocol as every other column.

Outputs: figures/all_models_tables/{DATASET}/table_all.{csv,tex}, table_supervised.csv

Usage:
    python scripts/run_supervised_table.py --datasets BTC,MSFT,GOOGL
    python scripts/run_supervised_table.py --datasets BTC --smoke
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import warnings
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
from lobgrad.eval.evaluate import evaluate_model_pooled
from lobgrad.models.supervised import GaussianARForecaster, BACKBONES

import bootstrap_tables as BT
from equity_profiles import make_config
import run_equity_table as RET

NEW_MODELS = ["DeepLOB", "TransLOB", "LOBS5"]
FULL_ORDER = ["LOBGrad", "LG-NE", "LG-Q", "LG-R", "CGAN", "Hawkes", "Poisson", "PTM"] + NEW_MODELS
PTM_CSV = {ds: REPO_ROOT / "figures" / "paint_tables" / ds / "table_with_ptm.csv"
           for ds in ("BTC", "MSFT", "GOOGL")}
METRICS = [k for k, _ in BT.QUEUE_BLOCK + BT.PRICE_BLOCK]


def build_cfg(label):
    if label == "BTC":
        return BenchmarkConfig(data_dir=str(REPO_ROOT / "data"), symbol="BTCUSDT",
                               levels=5, price_level=2.0, queue_repr="signed", queue_mode="absolute",
                               horizon=20, delta_mid_clip=24.0, timegrad_history_length=100)
    return make_config(label, horizon=20, timegrad_history_length=100)


def render_all(cells, models, horizon, n_windows):
    BS, EOL = chr(92), " " + chr(92) + chr(92)

    def row(label, key):
        vals = {m: cells.get(m, {}).get(key) for m in models}
        finite = {m: v[0] for m, v in vals.items() if v and v[0] is not None and np.isfinite(v[0])}
        best = min(finite, key=finite.get) if finite else None
        out = [label]
        for m in models:
            v = vals.get(m)
            if not v or v[0] is None or not np.isfinite(v[0]):
                out.append("---"); continue
            disp, lo, hi = v
            main = f"{disp:.3f}"
            if m == best:
                main = BS + "textbf{" + main + "}"
            if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
                out.append(BS + "shortstack[r]{" + main + " " + BS + BS +
                           " {" + BS + "scriptsize $[" + f"{lo:.3f}" + "," + BS + "," + f"{hi:.3f}" + "]$}}")
            else:
                out.append(main)
        return out

    ncol = len(models) + 1
    lines = [
        BS + "begin{table}[t]",
        BS + "caption{20-step distributional metrics across " + str(n_windows) +
        " held-out windows (lower is better; " + BS + "textbf{bold}=best per row). "
        "DeepLOB/TransLOB/LOBS5 are supervised models adapted to probabilistic regression "
        "(Gaussian head, autoregressive rollout); shown as point estimates (no CI).}",
        BS + "centering", BS + "scriptsize", BS + "renewcommand{" + BS + "arraystretch}{1.25}",
        BS + "setlength{" + BS + "tabcolsep}{3pt}", BS + "resizebox{" + BS + "linewidth}{!}{%",
        BS + "begin{tabular}{l" + "r" * len(models) + "}", BS + "toprule",
        " & ".join(["Metric"] + list(models)) + EOL, BS + "midrule",
        BS + "multicolumn{" + str(ncol) + "}{l}{" + BS + "emph{Queue}}" + EOL,
    ]
    for key, lab in BT.QUEUE_BLOCK:
        lines.append(" & ".join(row(lab, key)) + EOL)
    lines += [BS + "midrule", BS + "multicolumn{" + str(ncol) + "}{l}{" + BS + "emph{Returns}}" + EOL]
    for key, lab in BT.PRICE_BLOCK:
        lines.append(" & ".join(row(lab, key)) + EOL)
    lines += [BS + "bottomrule", BS + "end{tabular}}", BS + "end{table}"]
    return "\n".join(lines)


def run(label, args, device):
    print(f"\n########## {label} :: supervised benchmarks ##########")
    cfg = build_cfg(label)
    (ts_tr, X_tr, a_tr), _, (ts_te, X_te, a_te), norm = date_splits_joint(cfg)
    print(f"  splits: train={len(X_tr):,} test={len(X_te):,}")

    rng = np.random.default_rng(args.seed)
    ctx_q, fut_q, ctx_a, fut_a = sample_eval_windows_joint(X_te, a_te, 100, args.horizon, args.n_windows, rng=rng)

    new_cells = {}
    for name in args.models:
        sd = REPO_ROOT / "checkpoints" / "supervised" / label
        ckpt = sd / f"{name}.pt"
        g = GaussianARForecaster(cfg, norm, name, device=device, epochs=args.epochs, lr=args.lr,
                                 batch_size=args.batch_size, stride=args.stride,
                                 max_train_windows=args.max_train_windows, save_dir=str(sd),
                                 train=not args.skip_train)
        if args.skip_train:
            g.load(str(ckpt))
        else:
            t0 = time.time(); g.fit(X_tr, a_tr, ts_tr); print(f"  [{name} train] {(time.time()-t0)/60:.1f} min")
        t0 = time.time()
        gen_q, gen_mid = g.generate_many(ctx_q, ctx_a, args.horizon, args.n_paths, chunk=args.chunk, seed=args.seed)
        res = evaluate_model_pooled(fut_q, gen_q, levels=cfg.levels, real_prices=fut_a, gen_prices=gen_mid)
        new_cells[name] = {k: (float(res[k]["mean"]) if k in res and np.isfinite(res[k]["mean"]) else None, None, None)
                           for k in METRICS}
        print(f"  [{name}] gen+eval {(time.time()-t0)/60:.1f} min  "
              f"crps={new_cells[name].get('crps')[0]}  return_dist={new_cells[name].get('return_dist')[0]}")

    # ── merge with the existing 8-column PTM table ──
    cells = {}
    ex = PTM_CSV[label]
    if ex.exists():
        edf = pd.read_csv(ex)
        for m in edf["model"].unique():
            sub = edf[edf["model"] == m].set_index("metric")
            cells[m] = {}
            for key in METRICS:
                if key in sub.index:
                    r = sub.loc[key]
                    cells[m][key] = (float(r["display"]) if pd.notna(r["display"]) else None,
                                     float(r["ci_lo"]) if pd.notna(r["ci_lo"]) else None,
                                     float(r["ci_hi"]) if pd.notna(r["ci_hi"]) else None)
    else:
        print(f"  ⚠ {ex} not found — supervised-only table")
    cells.update(new_cells)

    models = [m for m in FULL_ORDER if m in cells]
    od = REPO_ROOT / "figures" / "all_models_tables" / label
    od.mkdir(parents=True, exist_ok=True)
    (od / "table_all.tex").write_text(render_all(cells, models, args.horizon, args.n_windows))
    rows = [{"model": m, "metric": k, "display": cells[m][k][0],
             "ci_lo": cells[m][k][1], "ci_hi": cells[m][k][2]}
            for m in models for k in METRICS if k in cells[m]]
    pd.DataFrame(rows).to_csv(od / "table_all.csv", index=False)
    pd.DataFrame([{"model": m, "metric": k, "value": new_cells[m][k][0]}
                  for m in args.models for k in METRICS]).to_csv(od / "table_supervised.csv", index=False)
    (od / "run_manifest.json").write_text(json.dumps({
        "dataset": label, "horizon": args.horizon, "generated_at": datetime.now().isoformat(),
        "new_models": args.models, "n_windows": args.n_windows, "n_paths": args.n_paths,
        "epochs": args.epochs, "device": device, "seed": args.seed, "bootstrap": False}, indent=2))
    print(f"  wrote {od}/table_all.{{csv,tex}}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="BTC,MSFT,GOOGL")
    ap.add_argument("--models", default=",".join(NEW_MODELS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-windows", type=int, default=500)
    ap.add_argument("--n-paths", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-train-windows", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = min(args.epochs, 3)
        args.n_windows = 20 if args.n_windows == 500 else args.n_windows
        args.n_paths = 5 if args.n_paths == 100 else args.n_paths
        args.stride = 20

    args.models = [m.strip() for m in args.models.split(",") if m.strip() in BACKBONES]
    device = RET.pick_device(args.device)
    print(f"device={device}  models={args.models}")
    t0 = time.time()
    for label in [d.strip().upper() for d in args.datasets.split(",") if d.strip()]:
        if label not in PTM_CSV:
            print(f"[skip] unknown dataset {label}"); continue
        run(label, args, device)
    print(f"\n=== supervised tables done in {(time.time()-t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()

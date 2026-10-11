"""
run_paint_table.py
------------------
Train the "Painting the Market" (PTM) image-inpainting diffusion benchmark on
each dataset, generate the 20-step future block (one shot), bootstrap the same
Table-1 distributional metrics, and merge a **PTM** column into the existing
20-step tables (the other 7 columns are read from their saved CSVs — not
re-run).

Datasets: BTC (committed data), MSFT / GOOGL (equity data). PTM is evaluated on
the identical seed-42 / 500-window / 100-path / horizon-20 protocol as the
existing columns, so the comparison is apples-to-apples.

Outputs: figures/paint_tables/{DATASET}/table_with_ptm.{csv,tex}, table_ptm_only.csv

Usage:
    python scripts/run_paint_table.py --datasets BTC,MSFT,GOOGL
    python scripts/run_paint_table.py --datasets BTC --smoke
    python scripts/run_paint_table.py --datasets MSFT --skip-train
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
import torch
from tqdm.auto import tqdm

from lobgrad.config import BenchmarkConfig
from lobgrad.data.splits import date_splits_joint, sample_eval_windows_joint
from lobgrad.eval.evaluate import bootstrap_metrics
from lobgrad.models.paint import PaintDiffusionGenerator

import bootstrap_tables as BT
from equity_profiles import make_config
import run_equity_table as RET

MODEL_ORDER = ["LOBGrad", "LG-NE", "LG-Q", "LG-R", "CGAN", "Hawkes", "Poisson", "PTM"]
EXISTING_CSV = {
    "BTC":   REPO_ROOT / "figures" / "bootstrapped_tables" / "table_headline.csv",
    "MSFT":  REPO_ROOT / "figures" / "equity_tables" / "MSFT" / "table_headline.csv",
    "GOOGL": REPO_ROOT / "figures" / "equity_tables" / "GOOGL" / "table_headline.csv",
}


def build_cfg(label):
    if label == "BTC":
        return BenchmarkConfig(
            data_dir=str(REPO_ROOT / "data"), symbol="BTCUSDT",
            levels=5, price_level=2.0, queue_repr="signed", queue_mode="absolute",
            horizon=20, delta_mid_clip=24.0, timegrad_history_length=100)
    return make_config(label, horizon=20, timegrad_history_length=100)


# ─── combined table renderer (reads existing cols from CSV; PTM computed) ────
def render_combined(cells, models, horizon, n_windows, n_boot):
    """cells[model][metric] = (display, lo, hi) or (nan,..) for N/A."""
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
        BS + "caption{One-shot " + str(horizon) + "-step distributional metrics across " +
        str(n_windows) + " held-out windows (lower is better; " + BS + "textbf{bold}=best per row; " +
        str(n_boot) + "-resample 95\\% CIs). PTM = Painting-the-Market image-diffusion benchmark.}",
        BS + "centering", BS + "small", BS + "renewcommand{" + BS + "arraystretch}{1.25}",
        BS + "setlength{" + BS + "tabcolsep}{4pt}", BS + "resizebox{" + BS + "linewidth}{!}{%",
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
    print(f"\n########## {label} :: PTM benchmark ##########")
    cfg = build_cfg(label)
    (ts_tr, X_tr, a_tr), _, (ts_te, X_te, a_te), norm = date_splits_joint(cfg)
    print(f"  splits: train={len(X_tr):,} test={len(X_te):,} feat_dim={X_tr.shape[1]}")

    sd = REPO_ROOT / "checkpoints" / "paint" / label
    ckpt = sd / "paint.pt"
    gen = PaintDiffusionGenerator(
        cfg, normalizer=norm, device=device,
        t_train=args.t_train, sample_steps=args.sample_steps,
        base_channels=args.base_channels, epochs=args.epochs, stride=args.stride,
        batch_size=args.batch_size, max_train_windows=args.max_train_windows,
        save_dir=str(sd), train=not args.skip_train)

    if args.skip_train:
        gen.load(str(ckpt))
    else:
        t0 = time.time()
        gen.fit(X_tr, a_tr, ts_tr)
        print(f"  [train] {(time.time()-t0)/60:.1f} min")

    # same eval windows as the existing columns
    rng = np.random.default_rng(args.seed)
    ctx_q, fut_q, ctx_a, fut_a = sample_eval_windows_joint(
        X_te, a_te, 100, args.horizon, args.n_windows, rng=rng)

    # generate PTM 20-step block for all windows (batched)
    t0 = time.time()
    gen_q, gen_mid = gen.generate_many(ctx_q, ctx_a, args.horizon, args.n_paths,
                                       chunk=args.chunk, seed=args.seed)
    print(f"  [generate] {args.n_windows}x{args.n_paths} blocks in {(time.time()-t0)/60:.1f} min "
          f"-> q{gen_q.shape} mid{gen_mid.shape}")

    print("  bootstrapping PTM ...")
    ptm_ci = bootstrap_metrics(fut_q, gen_q, levels=cfg.levels,
                               real_prices=fut_a, gen_prices=gen_mid,
                               n_boot=args.n_boot, alpha=0.05, n_jobs=args.n_jobs,
                               seed=1234, progress=False)

    # ── assemble merged cells: existing 7 from CSV, PTM computed ──
    cells = {}
    ex_path = EXISTING_CSV[label]
    metrics = [k for k, _ in BT.QUEUE_BLOCK + BT.PRICE_BLOCK]
    if ex_path.exists():
        ex = pd.read_csv(ex_path)
        for m in ex["model"].unique():
            sub = ex[ex["model"] == m].set_index("metric")
            cells[m] = {}
            for key in metrics:
                if key in sub.index:
                    r = sub.loc[key]
                    d = r["display"]
                    cells[m][key] = (float(d) if pd.notna(d) else None,
                                     float(r["ci_lo"]) if pd.notna(r["ci_lo"]) else None,
                                     float(r["ci_hi"]) if pd.notna(r["ci_hi"]) else None)
    else:
        print(f"  ⚠ existing table {ex_path} not found — PTM-only table")
    cells["PTM"] = {key: BT._ci_triple({"PTM": ptm_ci}, "PTM", key) for key in metrics}

    models = [m for m in MODEL_ORDER if m in cells]
    od = REPO_ROOT / "figures" / "paint_tables" / label
    od.mkdir(parents=True, exist_ok=True)
    (od / "table_with_ptm.tex").write_text(
        render_combined(cells, models, args.horizon, args.n_windows, args.n_boot))

    rows = []
    for m in models:
        for key in metrics:
            v = cells[m].get(key)
            rows.append({"model": m, "metric": key,
                         "display": (v[0] if v else None),
                         "ci_lo": (v[1] if v else None), "ci_hi": (v[2] if v else None)})
    pd.DataFrame(rows).to_csv(od / "table_with_ptm.csv", index=False)
    pd.DataFrame([{"metric": k, "display": cells["PTM"][k][0],
                   "ci_lo": cells["PTM"][k][1], "ci_hi": cells["PTM"][k][2]}
                  for k in metrics]).to_csv(od / "table_ptm_only.csv", index=False)
    (od / "run_manifest.json").write_text(json.dumps({
        "dataset": label, "horizon": args.horizon, "generated_at": datetime.now().isoformat(),
        "n_windows": args.n_windows, "n_paths": args.n_paths, "n_boot": args.n_boot,
        "t_train": args.t_train, "sample_steps": args.sample_steps, "epochs": args.epochs,
        "device": device, "seed": args.seed}, indent=2))

    print(f"  wrote {od}/table_with_ptm.{{csv,tex}}")
    print("  PTM headline:")
    for key in metrics:
        d, lo, hi = cells["PTM"][key]
        print(f"    {key:>22s}: " + ("---" if d is None else f"{d:.4f} [{lo:.4f}, {hi:.4f}]"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="BTC,MSFT,GOOGL")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-windows", type=int, default=500)
    ap.add_argument("--n-paths", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    # PTM knobs (sized for MPS feasibility; see benchmark in the plan)
    ap.add_argument("--t-train", type=int, default=100)
    ap.add_argument("--sample-steps", type=int, default=20)
    ap.add_argument("--base-channels", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-train-windows", type=int, default=6000)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = min(args.epochs, 3); args.t_train = 50; args.sample_steps = 10
        args.n_windows = 20 if args.n_windows == 500 else args.n_windows
        args.n_paths = 5 if args.n_paths == 100 else args.n_paths
        args.n_boot = 100 if args.n_boot == 1000 else args.n_boot
        args.stride = 20

    device = RET.pick_device(args.device)
    print(f"device={device}")
    t0 = time.time()
    for label in [d.strip().upper() for d in args.datasets.split(",") if d.strip()]:
        if label not in EXISTING_CSV:
            print(f"[skip] unknown dataset {label}"); continue
        run(label, args, device)
    print(f"\n=== all PTM tables done in {(time.time()-t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()

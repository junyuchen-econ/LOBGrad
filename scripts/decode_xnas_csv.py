"""
decode_xnas_csv.py
------------------
Decode Databento XNAS.ITCH ``mbp-10`` CSV files (``data2/*.csv.zst``) into the
per-day JSONL snapshot format consumed by ``lobgrad/data/loader.py``.

Each output line is one 5-second snapshot:

    {"timestamp": <unix_ms>, "bids": [[px_cents, size], ...], "asks": [...]}

with **prices in integer cents** (``round(price_dollars * 100)``) and best price
first. Emitting cents makes the loader's ``floor``/``ceil`` price rounding a
no-op and turns ``price_level`` (delta) into a clean 1-cent grid.

Key choices (see the approved plan):
  * Regular trading hours only: 13:30-20:00 UTC == 09:30-16:00 ET (EDT, summer).
  * One snapshot per ``--time-diff-ms`` (default 5000 ms), on a grid aligned to
    the RTH open, using the most recent event's top-10 book.
  * Both MSFT and GOOGL are split into separate ``{SYMBOL}_{MM-DD}.json`` files.

Usage:
    python scripts/decode_xnas_csv.py                       # all files, both symbols
    python scripts/decode_xnas_csv.py --symbols MSFT
    python scripts/decode_xnas_csv.py --limit 500000        # quick test (rows/file)
    python scripts/decode_xnas_csv.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import zstandard

REPO_ROOT = Path(__file__).resolve().parent.parent
FNAME_RE = re.compile(r"xnas-itch-(\d{4})(\d{2})(\d{2})\.mbp-10\.csv\.zst$")
LEVELS = 10  # mbp-10


def _tod_seconds(hhmm: str) -> int:
    """'13:30' -> 48600 seconds since midnight."""
    h, m = hhmm.split(":")
    return int(h) * 3600 + int(m) * 60


def _extract_book(row, idx):
    """Return (bids, asks) in cents (best first) or None if empty/crossed.

    mbp-10 top levels are contiguous from the top, so we stop at the first empty
    price on each side.
    """
    bids, asks = [], []
    for lvl in range(LEVELS):
        bp = row[idx["bid_px"][lvl]]
        if not bp:
            break
        bs = row[idx["bid_sz"][lvl]]
        px = round(float(bp) * 100.0)
        sz = int(bs) if bs else 0
        if sz > 0:
            bids.append([px, sz])
    for lvl in range(LEVELS):
        ap = row[idx["ask_px"][lvl]]
        if not ap:
            break
        asz = row[idx["ask_sz"][lvl]]
        px = round(float(ap) * 100.0)
        sz = int(asz) if asz else 0
        if sz > 0:
            asks.append([px, sz])
    if not bids or not asks:
        return None
    if bids[0][0] >= asks[0][0]:      # crossed / locked book — skip
        return None
    return bids, asks


def _column_index(header):
    """Map logical names to CSV column positions (robust to layout changes)."""
    idx = {
        "ts_event": header.index("ts_event"),
        "symbol": header.index("symbol"),
        "bid_px": [header.index(f"bid_px_{i:02d}") for i in range(LEVELS)],
        "ask_px": [header.index(f"ask_px_{i:02d}") for i in range(LEVELS)],
        "bid_sz": [header.index(f"bid_sz_{i:02d}") for i in range(LEVELS)],
        "ask_sz": [header.index(f"ask_sz_{i:02d}") for i in range(LEVELS)],
    }
    return idx


def decode_file(
    path: Path,
    out_dir: Path,
    symbols: set[str],
    time_diff_ms: int,
    rth_start_s: int,
    rth_end_s: int,
    limit: int,
    overwrite: bool,
) -> dict[str, int]:
    m = FNAME_RE.search(path.name)
    if not m:
        return {}
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    date_base_ms = int(datetime(y, mo, d, tzinfo=timezone.utc).timestamp() * 1000)
    mmdd = f"{mo:02d}-{d:02d}"

    out_paths = {s: out_dir / f"{s}_{mmdd}.json" for s in symbols}
    if not overwrite and all(p.exists() for p in out_paths.values()):
        print(f"  [skip] {path.name} — outputs already exist")
        return {s: -1 for s in symbols}

    rth_start_ms = date_base_ms + rth_start_s * 1000
    rth_end_ms = date_base_ms + rth_end_s * 1000

    # per-symbol streaming state. We keep the most recent *row* (cheap reference)
    # and only parse its book at emit points (~4,680/day) rather than every row.
    last_row = {s: None for s in symbols}
    next_emit = {s: rth_start_ms for s in symbols}
    counts = {s: 0 for s in symbols}
    handles = {s: open(out_paths[s], "w") for s in symbols}

    dctx = zstandard.ZstdDecompressor()
    try:
        with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
            tw = io.TextIOWrapper(reader, encoding="utf-8")
            rd = csv.reader(tw)
            header = next(rd)
            idx = _column_index(header)
            i_ts, i_sym = idx["ts_event"], idx["symbol"]
            i_bb, i_ba = idx["bid_px"][0], idx["ask_px"][0]

            for n, row in enumerate(rd):
                if limit and n >= limit:
                    break
                sym = row[i_sym]
                if sym not in symbols:
                    continue

                ts = row[i_ts]  # 'YYYY-MM-DDTHH:MM:SS.fffffffffZ'
                tod_ms = (
                    int(ts[11:13]) * 3600_000
                    + int(ts[14:16]) * 60_000
                    + int(ts[17:19]) * 1000
                    + int(ts[20:23])          # first 3 fractional digits = ms
                )
                row_ms = date_base_ms + tod_ms

                # Emit snapshots for every grid point passed by this row, using the
                # book from the most recent event before it (parsed once here).
                ne = next_emit[sym]
                if ne <= row_ms and ne < rth_end_ms:
                    lr = last_row[sym]
                    book = _extract_book(lr, idx) if lr is not None else None
                    while ne <= row_ms and ne < rth_end_ms:
                        if book is not None:
                            bids, asks = book
                            handles[sym].write(
                                json.dumps({"timestamp": ne, "bids": bids, "asks": asks})
                                + "\n"
                            )
                            counts[sym] += 1
                        ne += time_diff_ms
                    next_emit[sym] = ne

                # Cheap update: remember this row if it has a two-sided top level.
                if row[i_bb] and row[i_ba]:
                    last_row[sym] = row
    finally:
        for h in handles.values():
            h.close()

    print(f"  [ok]   {path.name} -> " + ", ".join(f"{s}:{counts[s]}" for s in symbols))
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data2"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data2" / "processed"))
    ap.add_argument("--symbols", default="MSFT,GOOGL")
    ap.add_argument("--time-diff-ms", type=int, default=5000)
    ap.add_argument("--rth-start", default="13:30", help="UTC HH:MM (EDT 09:30)")
    ap.add_argument("--rth-end", default="20:00", help="UTC HH:MM (EDT 16:00)")
    ap.add_argument("--limit", type=int, default=0, help="max rows/file (0 = all; for testing)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = {s.strip() for s in args.symbols.split(",") if s.strip()}
    rth_start_s = _tod_seconds(args.rth_start)
    rth_end_s = _tod_seconds(args.rth_end)

    files = sorted(data_dir.glob("xnas-itch-*.mbp-10.csv.zst"))
    if not files:
        print(f"No CSV files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Decoding {len(files)} file(s) -> {out_dir}")
    print(f"  symbols={sorted(symbols)}  RTH={args.rth_start}-{args.rth_end} UTC  "
          f"step={args.time_diff_ms}ms")
    totals = {s: 0 for s in symbols}
    for path in files:
        counts = decode_file(path, out_dir, symbols, args.time_diff_ms,
                             rth_start_s, rth_end_s, args.limit, args.overwrite)
        for s, c in counts.items():
            if c > 0:
                totals[s] += c

    print("\nDone. Total snapshots written:")
    for s in sorted(symbols):
        print(f"  {s}: {totals[s]:,}")


if __name__ == "__main__":
    main()

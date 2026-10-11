# Data

LOBGrad is trained and evaluated on BTCUSDT limit order book snapshots at
5-second resolution, Oct 1–21 2023, 100 levels of depth per side (~5.6 GB).

---

## Obtaining the dataset

The raw snapshots were collected via the Binance public REST depth endpoint
at the time of the study. **Historical order book data at this resolution is
not available through Binance's public API retroactively**, so the exact
paper dataset cannot be re-collected from scratch.

### Download from Zenodo (required for exact reproduction)

```
DOI: https://doi.org/10.5281/zenodo.20046390
```

```bash
wget -O btcusdt_lob_oct2023.tar.gz https://zenodo.org/records/20046390/files/btcusdt_lob_oct2023.tar.gz
tar -xzf btcusdt_lob_oct2023.tar.gz -C data/raw/
```

After extraction `data/raw/` should contain 21 files named `BTCUSDT_10-DD.json`.

### Collecting new data (different time period)

To run the pipeline on a different date range, poll the live Binance endpoint
every 5 seconds and write one JSON line per snapshot:

```python
import json, time, requests, pathlib

SYMBOL  = "BTCUSDT"
DEPTH   = 100
OUT     = pathlib.Path("data/raw/BTCUSDT_<MM-DD>.json")
URL     = f"https://api.binance.com/api/v3/depth?symbol={SYMBOL}&limit={DEPTH}"

with open(OUT, "a") as f:
    while True:
        t0 = time.time()
        snap = requests.get(URL, timeout=5).json()
        record = {
            "timestamp": int(time.time() * 1000),
            "bids": [[float(p), float(q)] for p, q in snap["bids"]],
            "asks": [[float(p), float(q)] for p, q in snap["asks"]],
        }
        f.write(json.dumps(record) + "\n")
        f.flush()
        time.sleep(max(0.0, 5.0 - (time.time() - t0)))
```

No API key required. Results will reflect current market conditions, not the
Oct 2023 data used in the paper.

---

## File layout

```
data/
└── raw/
    ├── BTCUSDT_10-01.json   # one JSON line per 5-second snapshot
    ├── ...
    └── BTCUSDT_10-21.json
```

Each line:

```json
{"timestamp": 1696118400123, "bids": [[26950.0, 1.42], ...], "asks": [[26951.0, 0.87], ...]}
```

- `timestamp` — Unix time in milliseconds
- `bids` — up to 100 `[price, qty]` pairs, best bid first (descending price)
- `asks` — up to 100 `[price, qty]` pairs, best ask first (ascending price)

Parsed by `lobgrad/data/loader.py`.

---

## Train / val / test splits

| Split | Dates | Files |
|-------|-------|-------|
| Train | 2023-10-01 – 2023-10-15 | `10-01` … `10-15` |
| Val   | 2023-10-16 – 2023-10-17 | `10-16`, `10-17` |
| Test  | 2023-10-18 – 2023-10-21 | `10-18` … `10-21` |

Splits are applied in code (`lobgrad/data/splits.py`) — no pre-split files needed.

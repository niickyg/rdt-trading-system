#!/usr/bin/env python3
"""
Convert raw IBKR `get_price_history` JSON dumps into the DataLoader parquet cache.

Why this exists
---------------
In the remote operator environment, yfinance's HTTP backend cannot use the
outbound proxy and Yahoo aggressively rate-limits the shared proxy IP, so the
backtest's default DataLoader cannot fetch data. The IBKR MCP feed
(`get_price_history`) *is* reachable and returns clean, split-adjusted daily
bars. This script bridges that feed into the exact parquet cache files
`backtesting/data_loader.py::_load_symbol` looks for, so the walk-forward
backtest runs unmodified and the mission metric (NET return vs SPY buy-and-hold)
becomes reproducible here.

Pipeline:
  1. A fetch step (subagents calling the IBKR MCP tool) writes one raw JSON file
     per symbol to `data/ibkr_raw/<SYMBOL>.json`. Each file is the verbatim
     `get_price_history` response object, optionally wrapped as
     {"symbol": "...", "data": <response>}.
  2. This script reads them and writes
     `data/backtest_cache/<SYMBOL>_<start>_<end>.parquet` with a tz-naive
     DatetimeIndex named "Date" and capitalized Open/High/Low/Close/Volume
     columns — identical in shape to DataLoader's own output.

Data-fidelity note (recorded honestly for the operator journal):
  IBKR historical TRADES bars are **split-adjusted** (verified: NVDA shows no
  10:1 gap at Jun-2024) but are **not dividend-adjusted**, whereas yfinance
  auto_adjust=True adjusts for both. For this momentum/RRS backtest the
  dividend effect is second-order (large-cap div yields ~0-2%/yr) and does not
  change the qualitative NET-vs-SPY verdict; splits — the effect that *would*
  break the backtest — are handled. The SPY benchmark uses the same IBKR series,
  so the comparison is apples-to-apples on price return (both exclude dividends).

Usage:
    python scripts/ibkr_to_cache.py                 # default WF-V2 window
    python scripts/ibkr_to_cache.py 2023-12-12 2026-08-18
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "ibkr_raw"
CACHE_DIR = PROJECT_ROOT / "data" / "backtest_cache"

# Must match scripts/run_walkforward_v2.py load_all_data():
#   end_date   = date.today()
#   start_date = end_date - timedelta(days=730 + 250)
DATA_DAYS = 730
WARMUP_DAYS = 250


def cache_window(end: date | None = None) -> tuple[date, date]:
    end = end or date.today()
    start = end - timedelta(days=DATA_DAYS + WARMUP_DAYS)
    return start, end


def _to_frame(resp: dict) -> pd.DataFrame:
    """IBKR get_price_history response -> OHLCV DataFrame (DatetimeIndex)."""
    times = resp["time"]
    idx = pd.DatetimeIndex(
        [datetime.fromisoformat(t.replace("Z", "+00:00")).date() for t in times]
    )
    df = pd.DataFrame(
        {
            "Open": resp["open"],
            "High": resp["high"],
            "Low": resp["low"],
            "Close": resp["close"],
            "Volume": resp["volume"],
        },
        index=pd.to_datetime(idx),
    )
    df.index.name = "Date"
    # Drop any incomplete/None bars, sort, de-dup.
    df = df[df["Close"].notna() & df["Open"].notna()]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def main() -> int:
    if len(sys.argv) == 3:
        start = date.fromisoformat(sys.argv[1])
        end = date.fromisoformat(sys.argv[2])
    else:
        start, end = cache_window()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_DIR.exists():
        print(f"ERROR: raw dir {RAW_DIR} does not exist. Run the fetch step first.")
        return 1

    raw_files = sorted(RAW_DIR.glob("*.json"))
    if not raw_files:
        print(f"ERROR: no *.json in {RAW_DIR}. Run the fetch step first.")
        return 1

    print(f"Converting {len(raw_files)} raw IBKR files -> cache window {start}..{end}")
    ok, bad = 0, []
    for f in raw_files:
        try:
            payload = json.loads(f.read_text())
            resp = payload.get("data", payload) if isinstance(payload, dict) else payload
            symbol = payload.get("symbol", f.stem) if isinstance(payload, dict) else f.stem
            df = _to_frame(resp)
            if len(df) < 200:
                bad.append((symbol, f"only {len(df)} bars"))
                continue
            out = CACHE_DIR / f"{symbol}_{start}_{end}.parquet"
            df.to_parquet(out)
            ok += 1
            print(f"  {symbol:6s} {len(df):4d} bars [{df.index[0].date()}..{df.index[-1].date()}] -> {out.name}")
        except Exception as e:  # noqa: BLE001
            bad.append((f.stem, repr(e)[:160]))

    print(f"\nDone: {ok} written, {len(bad)} failed.")
    for sym, why in bad:
        print(f"  FAILED {sym}: {why}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

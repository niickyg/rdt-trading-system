#!/usr/bin/env python3
"""
Proxy-friendly Yahoo Finance prefetch for backtest data.

Why this exists
---------------
`backtesting/data_loader.py` downloads via the `yfinance` library, whose modern
`curl_cffi` HTTP backend does not honor this environment's outbound HTTPS proxy
(``HTTPS_PROXY`` + ``REQUESTS_CA_BUNDLE``), so it fails with SSL/connection-reset
errors. The plain ``requests`` library *does* honor the proxy and CA bundle.

This module fetches daily OHLCV bars straight from Yahoo's public chart endpoint
(``query2.finance.yahoo.com/v8/finance/chart``) using ``requests`` and writes them
into the exact parquet cache files that ``DataLoader._load_symbol`` looks for
*before* it would ever call yfinance. Result: the walk-forward backtest runs
unmodified and fully offline-from-yfinance, and the mission metric (NET return
vs SPY buy-and-hold) becomes reproducible in this environment.

Adjustment: prices are back-adjusted for splits and dividends using Yahoo's
``adjclose`` series (factor = adjclose/close applied to O/H/L/C), matching
``yfinance``'s ``auto_adjust=True``. This is essential — e.g. NVDA's 10:1 split
(Jun 2024) falls inside the backtest window and unadjusted data would inject a
fake 90% gap.

Usage:
    python -m backtesting.yahoo_prefetch            # prefetch default WF-V2 set
    python -m backtesting.yahoo_prefetch AAPL MSFT  # prefetch specific symbols
"""

from __future__ import annotations

import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "backtest_cache"

# Mirrors scripts/run_walkforward_v2.py: DATA_DAYS(730) + 250-day SMA warmup.
DATA_DAYS = 730
WARMUP_DAYS = 250

_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": _BROWSER_UA, "Accept": "application/json"})
    return s


def fetch_daily(
    symbol: str,
    start: date,
    end: date,
    session: Optional[requests.Session] = None,
    retries: int = 4,
) -> Optional[pd.DataFrame]:
    """Fetch split/dividend-adjusted daily OHLCV for one symbol.

    Returns a DataFrame indexed by tz-naive DatetimeIndex with capitalized
    columns Open/High/Low/Close/Volume — identical in shape to what
    DataLoader._load_symbol produces — or None on failure.
    """
    session = session or _session()
    period1 = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp())
    # +1 day so the end date's bar is inclusive.
    period2 = int(datetime(end.year, end.month, end.day, tzinfo=timezone.utc).timestamp()) + 86400

    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }

    last_err = None
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 429:
                last_err = "HTTP 429 (rate limited)"
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            payload = r.json()
            return _parse_chart(payload)
        except Exception as e:  # noqa: BLE001 - network/parse errors are all retryable
            last_err = repr(e)[:200]
            time.sleep(1.5 * (attempt + 1))
    print(f"    ! {symbol}: failed after {retries} attempts ({last_err})")
    return None


def _parse_chart(payload: dict) -> Optional[pd.DataFrame]:
    chart = payload.get("chart", {})
    if chart.get("error"):
        return None
    results = chart.get("result")
    if not results:
        return None
    res = results[0]
    timestamps = res.get("timestamp")
    if not timestamps:
        return None

    quote = res["indicators"]["quote"][0]
    opens = quote.get("open", [])
    highs = quote.get("high", [])
    lows = quote.get("low", [])
    closes = quote.get("close", [])
    volumes = quote.get("volume", [])

    adj = None
    adjblock = res["indicators"].get("adjclose")
    if adjblock and adjblock[0].get("adjclose"):
        adj = adjblock[0]["adjclose"]

    rows = []
    idx = []
    for i, ts in enumerate(timestamps):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        v = volumes[i]
        # Skip incomplete bars (Yahoo often includes a null-close in-progress bar).
        if c is None or o is None or h is None or l is None:
            continue
        # Back-adjust O/H/L/C by adjclose/close to match yfinance auto_adjust=True.
        factor = 1.0
        if adj is not None and i < len(adj) and adj[i] is not None and c:
            factor = adj[i] / c
        idx.append(datetime.utcfromtimestamp(ts).date())
        rows.append(
            {
                "Open": o * factor,
                "High": h * factor,
                "Low": l * factor,
                "Close": c * factor,
                "Volume": int(v) if v is not None else 0,
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows, index=pd.DatetimeIndex(pd.to_datetime(idx)))
    df.index.name = "Date"
    return df


def prefetch(
    symbols: List[str],
    data_days: int = DATA_DAYS,
    warmup_days: int = WARMUP_DAYS,
    end: Optional[date] = None,
    pause: float = 0.4,
) -> dict:
    """Populate the DataLoader parquet cache for `symbols`.

    Writes files named `{symbol}_{start}_{end}.parquet` — the exact names
    DataLoader._load_symbol checks — so a subsequent backtest run uses them and
    never touches yfinance.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    end = end or date.today()
    start = end - timedelta(days=data_days + warmup_days)
    session = _session()

    print(f"Prefetch window: {start} -> {end}  ({len(symbols)} symbols) -> {CACHE_DIR}")
    ok, fail = {}, []
    for i, sym in enumerate(symbols, 1):
        df = fetch_daily(sym, start, end, session=session)
        if df is None or len(df) < 20:
            fail.append(sym)
            print(f"  [{i}/{len(symbols)}] {sym}: FAILED / insufficient")
        else:
            cache_file = CACHE_DIR / f"{sym}_{start}_{end}.parquet"
            df.to_parquet(cache_file)
            ok[sym] = len(df)
            print(f"  [{i}/{len(symbols)}] {sym}: {len(df)} bars "
                  f"[{df.index[0].date()}..{df.index[-1].date()}] -> {cache_file.name}")
        time.sleep(pause)

    print(f"\nDone. {len(ok)} ok, {len(fail)} failed.")
    if fail:
        print(f"Failed symbols: {fail}")
    return {"ok": ok, "failed": fail, "start": start, "end": end}


# Default set = walk-forward V2 universe (watchlist + SPY + VIX + sector ETFs).
DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "V", "JNJ", "UNH", "HD", "PG", "MA", "DIS",
    "PYPL", "ADBE", "CRM", "NFLX", "INTC", "AMD", "CSCO",
    "PEP", "KO", "MRK", "ABT", "TMO", "COST", "AVGO", "TXN",
]
DEFAULT_SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLB", "XLRE", "XLC", "XLU"]
DEFAULT_EXTRA = ["SPY", "^VIX"]


def default_symbols() -> List[str]:
    return DEFAULT_EXTRA + DEFAULT_WATCHLIST + DEFAULT_SECTOR_ETFS


if __name__ == "__main__":
    syms = sys.argv[1:] or default_symbols()
    prefetch(syms)

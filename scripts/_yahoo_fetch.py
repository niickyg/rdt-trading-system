"""
Minimal, proxy-aware Yahoo daily-bar fetcher.

Why this exists: yfinance's curl_cffi backend ignores HTTPS_PROXY / the agent CA
bundle in the remote operator environment, so it fails with connection resets.
This uses plain `requests`, which honors HTTPS_PROXY and REQUESTS_CA_BUNDLE, and
hits the public Yahoo chart endpoint directly with polite backoff + on-disk cache.

Daily bars only. Not for production trading — this is a research/validation tool
for the operator's honest-backtest work.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

_CACHE = Path(__file__).resolve().parent.parent / "data" / "operator_journal" / "_datacache"
_CACHE.mkdir(parents=True, exist_ok=True)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    )
}


def _cache_path(symbol: str, start: str, end: str) -> Path:
    return _CACHE / f"{symbol}_{start}_{end}.json"


def fetch_daily(symbol: str, rng: str = "2y", *, start: str | None = None,
                end: str | None = None, max_retries: int = 6,
                pause: float = 2.0) -> list[dict]:
    """Return a list of {date, open, high, low, close, volume} dicts (auto-adjusted).

    Uses the query2 host with a `range` param (e.g. '2y', '5y') — the query1
    period1/period2 path is rate-limited (429) from the operator environment, but
    query2+range works. If start/end ('YYYY-MM-DD') are given, the range result is
    sliced to that window client-side. Results are cached on disk so re-runs are
    free and do not re-hit Yahoo.
    """
    cp = _cache_path(symbol, rng, f"{start or ''}_{end or ''}")
    if cp.exists():
        return json.loads(cp.read_text())

    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "range": rng,
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }

    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=30)
            if r.status_code == 429:
                last_err = "429 Too Many Requests"
                time.sleep(pause * (2 ** attempt))
                continue
            r.raise_for_status()
            payload = r.json()
            bars = _parse(payload)
            if start:
                bars = [b for b in bars if b["date"] >= start]
            if end:
                bars = [b for b in bars if b["date"] <= end]
            cp.write_text(json.dumps(bars))
            return bars
        except Exception as e:  # noqa: BLE001 - research tool, surface + retry
            last_err = str(e)
            time.sleep(pause * (2 ** attempt))
    raise RuntimeError(f"fetch_daily({symbol}) failed after {max_retries} tries: {last_err}")


def _parse(payload: dict) -> list[dict]:
    result = payload["chart"]["result"][0]
    ts = result["timestamp"]
    q = result["indicators"]["quote"][0]
    # Prefer adjusted close for total-return fidelity; fall back to raw close.
    adj = None
    if "adjclose" in result["indicators"]:
        adj = result["indicators"]["adjclose"][0]["adjclose"]
    bars = []
    for i, t in enumerate(ts):
        o, h, l, c, v = q["open"][i], q["high"][i], q["low"][i], q["close"][i], q["volume"][i]
        if None in (o, h, l, c):
            continue
        close_adj = adj[i] if adj and adj[i] is not None else c
        # Scale OHLC by the adj/close ratio so the whole bar is total-return consistent.
        ratio = close_adj / c if c else 1.0
        bars.append({
            "date": time.strftime("%Y-%m-%d", time.gmtime(t)),
            "open": o * ratio,
            "high": h * ratio,
            "low": l * ratio,
            "close": close_adj,
            "volume": v or 0,
        })
    return bars


if __name__ == "__main__":
    import sys
    syms = sys.argv[1:] or ["SPY"]
    for i, sym in enumerate(syms):
        if i:
            time.sleep(2.5)  # polite spacing between symbols
        try:
            data = fetch_daily(sym, "2y")
            print(f"{sym}: {len(data)} bars  {data[0]['date']} -> {data[-1]['date']}")
        except Exception as e:  # noqa: BLE001
            print(f"{sym}: FAILED {e}")

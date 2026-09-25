#!/usr/bin/env python3
"""Operator benchmark check: buy-and-hold return of a passive benchmark over a window.

Used by the autonomous operator to independently answer the mission question
"does the strategy beat SPY buy-and-hold?" without trusting documented numbers.

Usage:
    python scripts/operator/verify_benchmark.py [SYMBOL ...] \
        [--start YYYY-MM-DD] [--end YYYY-MM-DD]

Defaults to SPY and QQQ over the documented walk-forward window
(2024-02-01 -> 2025-11-30). Requires: pandas, yfinance (pip install if missing).
"""
from __future__ import annotations

import argparse


def bh_stats(symbol: str, start: str, end: str) -> dict | None:
    import yfinance as yf

    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    close = df["Close"]
    if hasattr(close, "columns"):  # yfinance may return MultiIndex columns
        close = close.iloc[:, 0]
    close = close.dropna()
    if len(close) < 2:
        return None
    c0, c1 = float(close.iloc[0]), float(close.iloc[-1])
    days = (close.index[-1] - close.index[0]).days or 1
    return {
        "symbol": symbol,
        "start": str(close.index[0].date()),
        "end": str(close.index[-1].date()),
        "total_pct": (c1 / c0 - 1) * 100,
        "annualized_pct": ((c1 / c0) ** (365.25 / days) - 1) * 100,
        "max_dd_pct": float((close / close.cummax() - 1).min()) * 100,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("symbols", nargs="*", default=["SPY", "QQQ"])
    p.add_argument("--start", default="2024-02-01")
    p.add_argument("--end", default="2025-11-30")
    args = p.parse_args()

    symbols = args.symbols or ["SPY", "QQQ"]
    for sym in symbols:
        s = bh_stats(sym, args.start, args.end)
        if not s:
            print(f"{sym}: NO DATA")
            continue
        print(
            f"{s['symbol']}: {s['start']}->{s['end']} "
            f"total={s['total_pct']:.1f}% annualized={s['annualized_pct']:.1f}% "
            f"maxDD={s['max_dd_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()

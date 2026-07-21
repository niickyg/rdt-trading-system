#!/usr/bin/env python3
"""
Operator reproducibility harness.

Runs the project's own walk-forward backtest (scripts/run_walkforward_v2.py)
UNMODIFIED, but routes all market-data fetches through a requests-based Yahoo
shim (yfinance's curl_cffi transport fails TLS through the agent proxy).

Then prints the SPY buy-and-hold return over the identical window — the
benchmark the operator mandate requires every result to be compared against.

Usage:
    python scripts/operator/run_walkforward.py
"""
import sys
import os
import importlib.util
from datetime import date, timedelta

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Install the Yahoo shim as `yfinance` before importing anything that needs it.
shim = _load("yfinance", os.path.join(ROOT, "scripts", "operator", "yahoo_shim.py"))
sys.modules["yfinance"] = shim


def spy_buy_and_hold(start_str, end_str):
    """Total + annualized SPY return over [start, end]."""
    spy = shim.Ticker("SPY").history(
        start=date.today() - timedelta(days=980), end=date.today()
    )

    def price_on_or_before(d):
        sub = spy[spy.index <= pd.Timestamp(d)]
        return sub["Close"].iloc[-1], sub.index[-1].date()

    p0, d0 = price_on_or_before(start_str)
    p1, d1 = price_on_or_before(end_str)
    total = (p1 / p0 - 1) * 100
    yrs = (pd.Timestamp(end_str) - pd.Timestamp(start_str)).days / 365.25
    ann = ((p1 / p0) ** (1 / yrs) - 1) * 100 if yrs > 0 else float("nan")
    return d0, p0, d1, p1, total, ann, yrs


def main():
    wf = _load("wf_v2", os.path.join(ROOT, "scripts", "run_walkforward_v2.py"))
    wf.main()

    # Benchmark. The v2 windows span ~day 91..600 of a (730+250)-day pull,
    # i.e. roughly 2024-03-27 .. 2026-04-09 when run in mid-2026. Adjust here
    # if run_walkforward_v2.py's windowing changes.
    print("\n" + "=" * 88)
    print(" SPY BUY-AND-HOLD BENCHMARK (the bar the strategy must clear)")
    print("=" * 88)
    d0, p0, d1, p1, total, ann, yrs = spy_buy_and_hold("2024-03-27", "2026-04-09")
    print(f"  SPY {d0} ${p0:.2f} -> {d1} ${p1:.2f}")
    print(f"  Total: {total:+.1f}%   Annualized: {ann:+.1f}%   ({yrs:.2f} years)")
    print("  Compare this against the aggregate strategy returns printed above.")
    print("=" * 88)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
benchmark_vs_spy.py — The mission benchmark, in one command.

The operator mandate says the bot is only "profitable" if it beats SPY
buy-and-hold over the same period, net of honest costs. Backtest scripts
report the strategy's GROSS return in isolation; none of the backtest engines
model transaction costs, and none print the buy-and-hold number the strategy
must beat. This tool closes that gap.

It:
  1. Runs the best strategy config (RDT-filtered, from run_walkforward_v2) across
     the same 6 walk-forward windows.
  2. Downloads SPY and computes buy-and-hold over BOTH the continuous span and
     the exact in-window trading days (apples-to-apples market exposure).
  3. Applies a transaction-cost sensitivity band to the strategy's gross return
     (per-round-trip $ cost x trade count), since the engines model zero costs.
  4. Prints a clear verdict: did the strategy beat buy-and-hold, net of costs?

Usage:
    cd /home/user/rdt-trading-system
    python scripts/benchmark_vs_spy.py

Dependencies: yfinance, pandas, numpy, pyarrow (pip install if missing).
No engine code is modified; this is a read-only measurement tool.
"""

import sys
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="ERROR")

# Reuse the exact walk-forward machinery so the strategy number matches.
from scripts.run_walkforward_v2 import (
    load_all_data, run_rdt_filtered, WALK_FORWARD_WINDOWS, INITIAL_CAPITAL,
)

# Cost band (round-trip $ per position). IBKR large-cap liquid names:
# commission is near-zero; the real cost is spread + slippage. This brackets
# a generous range so the verdict is robust to the assumption.
COST_PER_ROUND_TRIP = [0.0, 2.0, 5.0, 10.0, 20.0]


def spy_buy_and_hold(all_trading_dates):
    """Compute SPY b&h over the walk-forward span and in-window only."""
    start = date.today() - timedelta(days=730 + 250)
    end = date.today()
    spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
    spy = spy.dropna()
    close = spy["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    dates = list(spy.index.date)
    n = len(dates)

    # Continuous: first window start index -> last window end index
    s_idx = min(WALK_FORWARD_WINDOWS[0]["start_day"], n - 1)
    e_idx = min(WALK_FORWARD_WINDOWS[-1]["end_day"], n - 1)
    p0, p1 = float(close.iloc[s_idx]), float(close.iloc[e_idx])
    cont_ret = p1 / p0 - 1

    # In-window only: compound SPY return across exactly the traded windows
    capital = INITIAL_CAPITAL
    tdays = 0
    for w in WALK_FORWARD_WINDOWS:
        si, ei = min(w["start_day"], n - 1), min(w["end_day"], n - 1)
        capital *= (1 + (float(close.iloc[ei]) / float(close.iloc[si]) - 1))
        tdays += (ei - si)
    inwin_ret = capital / INITIAL_CAPITAL - 1
    return {
        "cont_ret": cont_ret,
        "cont_dollar": INITIAL_CAPITAL * cont_ret,
        "inwin_ret": inwin_ret,
        "inwin_dollar": capital - INITIAL_CAPITAL,
        "inwin_trading_days": tdays,
        "span": (dates[s_idx], dates[e_idx]),
    }


def main():
    print("\n" + "=" * 80)
    print("MISSION BENCHMARK: RDT strategy vs SPY buy-and-hold".center(80))
    print("=" * 80)

    stock_data, spy_data, vix_data, sector_etf_data = load_all_data()
    all_trading_dates = sorted(spy_data.index.date)

    # Run the best strategy config across all windows
    total_gross = 0.0
    total_trades = 0
    for w in WALK_FORWARD_WINDOWS:
        si = min(w["start_day"], len(all_trading_dates) - 1)
        ei = min(w["end_day"], len(all_trading_dates) - 1)
        rd = run_rdt_filtered(
            stock_data, spy_data, vix_data, sector_etf_data,
            all_trading_dates[si], all_trading_dates[ei], w["name"],
        )
        total_gross += rd.result.total_return
        total_trades += rd.result.total_trades

    spy = spy_buy_and_hold(all_trading_dates)

    print(f"\nWindow span: {spy['span'][0]} -> {spy['span'][1]}")
    print(f"Strategy trades: {total_trades}")
    print("\n--- SPY BUY-AND-HOLD (the number to beat) ---")
    print(f"  Continuous: {spy['cont_ret']*100:+.2f}%  (${spy['cont_dollar']:+,.0f} on ${INITIAL_CAPITAL:,.0f})")
    print(f"  In-window : {spy['inwin_ret']*100:+.2f}%  (${spy['inwin_dollar']:+,.0f}) "
          f"over {spy['inwin_trading_days']} traded days")

    print("\n--- RDT STRATEGY (best config), net of cost band ---")
    print(f"  {'Cost/round-trip':>16} {'Net $':>12} {'Net %':>10} {'vs SPY in-window':>18}")
    for c in COST_PER_ROUND_TRIP:
        net = total_gross - c * total_trades
        net_pct = net / INITIAL_CAPITAL * 100
        gap = net - spy["inwin_dollar"]
        print(f"  {('$' + format(c, '.0f')):>16} {('$' + format(net, '+,.0f')):>12} "
              f"{(format(net_pct, '+.2f') + '%'):>10} {('$' + format(gap, '+,.0f')):>18}")

    beats = (total_gross - COST_PER_ROUND_TRIP[0] * total_trades) > spy["inwin_dollar"]
    print("\n" + "=" * 80)
    print("VERDICT".center(80))
    print("=" * 80)
    if beats:
        print("  Strategy beats SPY buy-and-hold (even before costs). Investigate further.")
    else:
        gap = spy["inwin_dollar"] - total_gross
        print(f"  Strategy LOSES to SPY buy-and-hold by ${gap:,.0f} even at ZERO cost.")
        print(f"  Gross strategy: ${total_gross:+,.0f} vs SPY in-window: ${spy['inwin_dollar']:+,.0f}.")
        print("  Costs only widen the gap. Per mandate: report honestly; do not tune to hide this.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

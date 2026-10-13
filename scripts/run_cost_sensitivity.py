#!/usr/bin/env python3
"""
Cost-Sensitivity Analysis: does the RDT strategy survive honest transaction costs?

The backtest engines (`backtesting/engine_enhanced.py`) model ZERO transaction
costs -- P&L is computed as raw (exit_price - entry_price) * shares with perfect
fills at the computed stop/target prices. Reported backtest returns are therefore
GROSS and frictionless.

This script:
  1. Runs the RDT-aligned filtered engine (Config C -- the documented "best")
     over a continuous ~2-year window to extract the actual executed trade list.
  2. Applies realistic round-trip costs at several severity levels.
  3. Recomputes net P&L and compares to SPY buy-and-hold over the IDENTICAL window
     (the mandate's benchmark).
  4. Reports the break-even cost -- the per-side slippage that reduces net P&L to
     zero -- i.e. the strategy's gross edge expressed in basis points of notional.

Cost model (per trade, round trip = entry fill + exit fill):
  - Commission:  IBKR fixed-tier $0.005/share, $1.00 minimum per order, charged on
                 both the entry and the exit.
  - Slippage + half-spread:  `slippage_bps` of notional PER SIDE. Backtests assume
                 fills exactly at close (entry) and exactly at the stop/target
                 (exit); real stop-market fills slip, targets require trade-through,
                 and every fill crosses the bid/ask. 5-20 bps/side is a realistic
                 range for a stop-based swing strategy even in liquid mega-caps.
  - Short borrow is NOT modeled: the watchlist is easy-to-borrow mega-caps held for
                 days, so borrow (~0.25-0.5%/yr) is negligible here. It would matter
                 for a real, broader short book -- noted, not simulated.

Usage:
    cd /home/user/rdt-trading-system
    python3.13 scripts/run_cost_sensitivity.py
"""

import sys
import time
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

# Reuse the exact engines + data loader the documented walk-forward uses.
from scripts.run_walkforward_v2 import (
    RDTFilteredEngine,
    load_all_data,
    make_engine_kwargs,
    BASE_ENGINE_CONFIG,
    INITIAL_CAPITAL,
)
from copy import deepcopy


# --- Cost model -------------------------------------------------------------

COMMISSION_PER_SHARE = 0.005   # IBKR fixed tier
COMMISSION_MIN_ORDER = 1.00    # per order (entry and exit each count)
SLIPPAGE_SCENARIOS_BPS = [0.0, 5.0, 10.0, 20.0]  # per side, in bps of notional


def trade_commission(shares: int) -> float:
    """Round-trip commission: one entry order + one exit order."""
    per_order = max(COMMISSION_MIN_ORDER, COMMISSION_PER_SHARE * abs(shares))
    return 2.0 * per_order


def trade_slippage(entry_price: float, exit_price: float, shares: int, bps_per_side: float) -> float:
    """Round-trip slippage/spread: bps of notional on entry notional + exit notional."""
    frac = bps_per_side / 10000.0
    entry_notional = abs(entry_price * shares)
    exit_notional = abs((exit_price or entry_price) * shares)
    return frac * (entry_notional + exit_notional)


def net_pnl_for_trade(t, bps_per_side: float) -> float:
    comm = trade_commission(t.shares)
    slip = trade_slippage(t.entry_price, t.exit_price, t.shares, bps_per_side)
    return t.pnl - comm - slip


# --- SPY buy-and-hold benchmark --------------------------------------------

def spy_buy_and_hold(spy_data: pd.DataFrame, start_date: date, end_date: date) -> dict:
    close_col = 'Close' if 'Close' in spy_data.columns else 'close'
    window = spy_data[(spy_data.index.date >= start_date) & (spy_data.index.date <= end_date)]
    if len(window) < 2:
        return {"return_pct": 0.0, "return_dollar": 0.0}
    first = float(window[close_col].iloc[0])
    last = float(window[close_col].iloc[-1])
    ret_pct = (last / first - 1.0) * 100.0
    return {
        "return_pct": ret_pct,
        "return_dollar": INITIAL_CAPITAL * ret_pct / 100.0,
        "first_price": first,
        "last_price": last,
    }


# --- Main -------------------------------------------------------------------

def main():
    print("\n" + "=" * 100)
    print("COST-SENSITIVITY ANALYSIS: RDT STRATEGY (Config C) NET OF HONEST COSTS".center(100))
    print("=" * 100 + "\n")

    t0 = time.time()
    stock_data, spy_data, vix_data, sector_etf_data = load_all_data()
    print(f"Data loaded in {time.time() - t0:.1f}s\n")

    if len(stock_data) < 5:
        print("ERROR: too few symbols loaded (yfinance connectivity). Aborting.")
        sys.exit(1)

    all_dates = sorted(spy_data.index.date)
    # Skip the first 250 days (reserved for 200-SMA warmup), trade the rest continuously.
    start_idx = min(250, len(all_dates) - 2)
    w_start = all_dates[start_idx]
    w_end = all_dates[-1]
    print(f"Continuous trade window: {w_start} to {w_end} ({len(all_dates) - start_idx} trading days)\n")

    # Run the RDT-aligned engine (the documented best configuration).
    config = deepcopy(BASE_ENGINE_CONFIG)
    kwargs = make_engine_kwargs(config)
    kwargs["vix_data"] = vix_data
    kwargs["sector_etf_data"] = sector_etf_data
    engine = RDTFilteredEngine(**kwargs)
    print("Running RDT-filtered engine (Config C)...")
    t1 = time.time()
    result = engine.run(stock_data, spy_data, start_date=w_start, end_date=w_end)
    print(f"  done in {time.time() - t1:.1f}s -- {result.total_trades} trades\n")

    trades = result.trades
    if not trades:
        print("No trades executed; nothing to analyze.")
        sys.exit(0)

    gross_total = sum(t.pnl for t in trades)
    n = len(trades)
    avg_notional = np.mean([abs(t.entry_price * t.shares) for t in trades])
    total_notional = sum(abs(t.entry_price * t.shares) for t in trades)

    # Benchmark
    bh = spy_buy_and_hold(spy_data, w_start, w_end)

    # Per-scenario net results
    print("-" * 100)
    print(f"{'Slippage/side':>14} {'Commission':>14} {'Slippage':>14} {'Net P&L':>14} "
          f"{'Net Return':>12} {'Win Rate':>10} {'PF':>7}")
    print("-" * 100)

    rows = []
    for bps in SLIPPAGE_SCENARIOS_BPS:
        total_comm = sum(trade_commission(t.shares) for t in trades)
        total_slip = sum(trade_slippage(t.entry_price, t.exit_price, t.shares, bps) for t in trades)
        net_trades = [net_pnl_for_trade(t, bps) for t in trades]
        net_total = sum(net_trades)
        winners = [p for p in net_trades if p > 0]
        losers = [p for p in net_trades if p <= 0]
        wr = len(winners) / n * 100.0
        gross_w = sum(winners)
        gross_l = abs(sum(losers))
        pf = gross_w / gross_l if gross_l > 0 else float('inf')
        net_ret_pct = net_total / INITIAL_CAPITAL * 100.0
        rows.append((bps, net_total, net_ret_pct))
        print(f"{bps:>12.0f}b {total_comm:>13,.0f}$ {total_slip:>13,.0f}$ {net_total:>13,.0f}$ "
              f"{net_ret_pct:>11.2f}% {wr:>9.1f}% {pf:>7.2f}")

    print("-" * 100 + "\n")

    # Break-even slippage: solve net_total(bps) = 0.
    # net_total(bps) = gross_total - fixed_commission - (bps/10000) * sum(entry_notional+exit_notional)
    fixed_comm = sum(trade_commission(t.shares) for t in trades)
    notional_sum = sum(
        abs(t.entry_price * t.shares) + abs((t.exit_price or t.entry_price) * t.shares)
        for t in trades
    )
    gross_after_comm = gross_total - fixed_comm
    breakeven_bps = (gross_after_comm / notional_sum) * 10000.0 if notional_sum > 0 else 0.0

    print("SUMMARY")
    print(f"  Trades executed:                 {n}")
    print(f"  Avg position notional:           ${avg_notional:,.0f}")
    print(f"  GROSS P&L (frictionless):        ${gross_total:,.0f}  ({gross_total/INITIAL_CAPITAL*100:.2f}%)")
    print(f"  Total commission (round trip):   ${fixed_comm:,.0f}")
    print(f"  Gross edge per trade:            ${gross_total/n:,.2f}")
    print(f"  Break-even slippage per side:    {breakeven_bps:.2f} bps")
    print(f"    (if real per-side slippage exceeds this, the strategy loses money net of costs)")
    print()
    print(f"  SPY BUY-AND-HOLD (same window):  ${bh['return_dollar']:,.0f}  ({bh['return_pct']:.2f}%)")
    print(f"    SPY {bh.get('first_price', 0):.2f} -> {bh.get('last_price', 0):.2f}")
    print()

    # Verdict
    net_at_10 = next(r[1] for r in rows if r[0] == 10.0)
    print("VERDICT")
    print(f"  At a realistic 10 bps/side, net P&L = ${net_at_10:,.0f} "
          f"vs SPY buy-and-hold ${bh['return_dollar']:,.0f}.")
    if net_at_10 < bh['return_dollar']:
        print("  => The active strategy UNDERPERFORMS passive SPY buy-and-hold net of honest costs.")
    else:
        print("  => The active strategy beats SPY buy-and-hold net of honest costs.")
    print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main()

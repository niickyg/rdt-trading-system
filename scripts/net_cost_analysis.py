#!/usr/bin/env python3
"""
Net-of-Cost Analysis + SPY Benchmark for the RDT walk-forward backtest.

WHY THIS EXISTS
---------------
`backtesting/engine_enhanced.py` models ZERO transaction costs: entries fill at
the daily close, exits fill at the exact stop/target, and there is no commission,
slippage, or bid/ask spread anywhere. Every headline number the walk-forward
prints ("$2,463", "9.85%") is therefore GROSS and optimistic.

The operator mandate (data/operator_journal/MANDATE.md, constraint #6) requires
honest cost accounting and a SPY buy-and-hold comparison over the identical
window. This script provides both WITHOUT modifying the shared engine:

  1. Re-runs config C (RDT filters) across the same walk-forward windows.
  2. Extracts every trade's real notional (entry_price * shares).
  3. Applies a transparent, configurable cost model per trade and reports NET
     P&L under several realistic scenarios.
  4. Prints SPY buy-and-hold over the exact backtest span for comparison.

COST MODEL (per trade, conservative LOWER BOUND)
------------------------------------------------
Each trade is charged as ONE entry fill + ONE exit fill (2 fills). The enhanced
engine also does partial "scale-out" exits, which would add MORE fills and thus
MORE cost -- so the drag reported here understates reality.

    per_fill_cost = commission_per_share * shares
                    + slippage_bps/10000 * fill_notional
    trade_cost    = entry_fill_cost + exit_fill_cost

Defaults reflect a retail IBKR-style account trading liquid large caps:
commission $0.005/share (min $1), and a sweep over 1 / 3 / 5 bps of
one-way slippage+spread.

Usage:
    python scripts/net_cost_analysis.py
"""

import sys
import glob
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from scripts.run_walkforward_v2 import (
    WALK_FORWARD_WINDOWS,
    load_all_data,
    run_rdt_filtered,
)

COMMISSION_PER_SHARE = 0.005
COMMISSION_MIN = 1.0
SLIPPAGE_BPS_SCENARIOS = [1.0, 3.0, 5.0]  # one-way, in basis points


def fill_cost(shares: int, notional: float, slippage_bps: float) -> float:
    commission = max(COMMISSION_MIN, COMMISSION_PER_SHARE * shares)
    slippage = (slippage_bps / 10000.0) * notional
    return commission + slippage


def trade_round_trip_cost(entry_price: float, shares: int, pnl: float,
                          slippage_bps: float) -> float:
    """Conservative 2-fill (1 entry + 1 exit) round-trip cost."""
    entry_notional = entry_price * shares
    # exit_notional ~ entry_notional +/- pnl; pnl is small vs notional, and its
    # sign depends on direction, so use entry_notional as the fill base for both
    # legs (a fair approximation for large-cap intraday holds).
    exit_notional = entry_notional
    return (fill_cost(shares, entry_notional, slippage_bps)
            + fill_cost(shares, exit_notional, slippage_bps))


def main():
    print("\n" + "=" * 96)
    print("NET-OF-COST ANALYSIS — RDT config C (walk-forward), vs SPY buy & hold".center(96))
    print("=" * 96 + "\n")

    stock_data, spy_data, vix_data, sector_etf_data = load_all_data()
    all_dates = sorted(spy_data.index.date)

    all_trades = []
    span_start = None
    span_end = None
    for window in WALK_FORWARD_WINDOWS:
        s = min(window["start_day"], len(all_dates) - 1)
        e = min(window["end_day"], len(all_dates) - 1)
        w_start, w_end = all_dates[s], all_dates[e]
        span_start = w_start if span_start is None else span_start
        span_end = w_end
        rd = run_rdt_filtered(stock_data, spy_data, vix_data, sector_etf_data,
                              w_start, w_end, window["name"])
        all_trades.extend(rd.result.trades)
        print(f"  {window['name']}: {rd.result.total_trades} trades, "
              f"{rd.result.total_return_pct:+.2f}% (gross)")

    n = len(all_trades)
    gross_pnl = sum(t.pnl for t in all_trades)
    notionals = [t.entry_price * t.shares for t in all_trades]
    avg_notional = sum(notionals) / n if n else 0.0
    total_notional = sum(notionals)

    print("\n" + "-" * 96)
    print(f"Trades: {n} | Gross P&L: ${gross_pnl:,.2f} | "
          f"Avg entry notional: ${avg_notional:,.0f} | "
          f"Total entry notional (turnover): ${total_notional:,.0f}")
    print("-" * 96)

    INITIAL = 25000.0
    print(f"\n{'Scenario (one-way bps)':<28}{'Total cost':>14}{'Net P&L':>14}"
          f"{'Net %':>10}{'Net ann.%':>12}")
    print("-" * 96)

    # span length in years for annualization
    yrs = (span_end - span_start).days / 365.25

    rows = []
    for bps in SLIPPAGE_BPS_SCENARIOS:
        total_cost = sum(
            trade_round_trip_cost(t.entry_price, t.shares, t.pnl, bps)
            for t in all_trades
        )
        net = gross_pnl - total_cost
        net_pct = net / INITIAL * 100
        net_ann = ((1 + net / INITIAL) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0.0
        rows.append((bps, total_cost, net, net_pct, net_ann))
        label = f"{bps:.0f} bps + ${COMMISSION_PER_SHARE:.3f}/sh"
        print(f"{label:<28}{('$'+format(total_cost, ',.0f')):>14}"
              f"{('$'+format(net, ',.0f')):>14}"
              f"{(format(net_pct, '+.2f')+'%'):>10}"
              f"{(format(net_ann, '+.2f')+'%'):>12}")

    # SPY buy & hold over the exact span
    spy_files = glob.glob(str(PROJECT_ROOT / "data/backtest_cache/SPY_*.parquet"))
    print("\n" + "-" * 96)
    if spy_files:
        spy = pd.read_parquet(spy_files[0])["Close"].dropna()
        seg = spy.loc[str(span_start):str(span_end)]
        f, l = float(seg.iloc[0]), float(seg.iloc[-1])
        spy_tot = (l / f - 1) * 100
        spy_ann = ((l / f) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0.0
        print(f"SPY BUY & HOLD  {span_start} -> {span_end}  ({yrs:.2f} yr):")
        print(f"  {f:.2f} -> {l:.2f}   Total {spy_tot:+.1f}%   Annualized {spy_ann:+.1f}%"
              f"   |  $25,000 -> ${INITIAL*l/f:,.0f}  (+${INITIAL*(l/f-1):,.0f})")
    else:
        print("SPY cache not found — run the walk-forward first to populate cache.")

    print("\n" + "=" * 96)
    print("VERDICT".center(96))
    print("=" * 96)
    print("The gross figure ignores all costs. Even at 3 bps one-way (optimistic for")
    print("crossing the spread on intraday large-cap fills), compare Net ann.% above to")
    print("SPY's annualized return. This is the only comparison the mandate cares about.")
    print("=" * 96 + "\n")


if __name__ == "__main__":
    main()

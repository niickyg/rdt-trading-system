"""Honest-cost overlay for the RDT-filtered strategy over one full ~2yr window.
Reuses the exact engine + data from scripts/run_walkforward_v2.py.
Computes gross return, total traded notional, applies a range of per-side friction
assumptions, and benchmarks against SPY buy-and-hold over the identical window.
"""
import sys, time
from pathlib import Path
from copy import deepcopy
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from loguru import logger
logger.remove()

import scripts.run_walkforward_v2 as wf

stock_data, spy_data, vix_data, sector_etf_data = wf.load_all_data()
close_col0 = 'Close' if 'Close' in spy_data.columns else 'close'
# Drop any tail rows where SPY close is NaN (yfinance partial bars)
spy_data = spy_data[spy_data[close_col0].notna()]
for sym in list(stock_data.keys()):
    c = 'Close' if 'Close' in stock_data[sym].columns else 'close'
    stock_data[sym] = stock_data[sym][stock_data[sym][c].notna()]
dates = sorted(spy_data.index.date)
# Full window: skip 250-day warmup, run to end
start = dates[min(250, len(dates)-1)]
end = dates[-1]
print(f"\nFull-window backtest: {start} -> {end}  ({(end-start).days} calendar days)")

config = deepcopy(wf.BASE_ENGINE_CONFIG)
kwargs = wf.make_engine_kwargs(config)
kwargs["vix_data"] = vix_data
kwargs["sector_etf_data"] = sector_etf_data
engine = wf.RDTFilteredEngine(**kwargs)
result = engine.run(stock_data, spy_data, start_date=start, end_date=end)

trades = result.trades
n = len(trades)
gross = result.total_return
# Entry notional per trade
entry_notional = sum(abs(t.shares) * t.entry_price for t in trades)
avg_notional = entry_notional / n if n else 0
scale1 = getattr(result, 'scale_1_exits', 0)
scale2 = getattr(result, 'scale_2_exits', 0)
# Fills: entry + final exit per trade + partial scale fills
fills = n * 2 + scale1 + scale2

print(f"\n=== GROSS (frictionless, as the engine reports) ===")
print(f"Trades:            {n}")
print(f"Gross P&L:         ${gross:,.2f}  ({gross/wf.INITIAL_CAPITAL*100:.2f}% on ${wf.INITIAL_CAPITAL:,.0f})")
print(f"Total entry notional: ${entry_notional:,.0f}   (avg ${avg_notional:,.0f}/trade)")
print(f"Est. fills:        {fills}  (entries {n} + final exits {n} + scale1 {scale1} + scale2 {scale2})")

# Annualization
years = (end - start).days / 365.25
def annualized(total):
    if wf.INITIAL_CAPITAL <= 0 or years <= 0: return 0
    return ((1 + total/wf.INITIAL_CAPITAL) ** (1/years) - 1) * 100

print(f"\n=== NET OF HONEST COSTS (friction on notional, both sides) ===")
print(f"Window length: {years:.2f} years")
print(f"{'Round-trip friction':>22} | {'$ cost':>10} | {'Net P&L':>11} | {'Net %':>8} | {'Net annualized':>15}")
print("-"*80)
# friction expressed as bps of notional PER SIDE; round trip = 2x notional
commission_per_fill = 0.0035 * 0  # IBKR ~ negligible per share; fold into bps
for bps_side in [2.5, 5, 7.5, 10, 15]:
    frac = bps_side/10000.0
    cost = frac * entry_notional * 2  # entry + exit each on ~entry notional
    net = gross - cost
    print(f"{bps_side*2:>18.0f}bps | ${cost:>9,.0f} | ${net:>10,.0f} | {net/wf.INITIAL_CAPITAL*100:>7.2f}% | {annualized(net):>13.2f}%")

# SPY buy-and-hold over identical window
close_col = 'Close' if 'Close' in spy_data.columns else 'close'
spy_win = spy_data[(spy_data.index.date >= start) & (spy_data.index.date <= end)][close_col].dropna()
spy_ret = (float(spy_win.iloc[-1]) / float(spy_win.iloc[0]) - 1) * 100
print(f"\n=== BENCHMARK ===")
print(f"SPY buy-and-hold, same window: {spy_ret:.2f}%  ({annualized(spy_ret/100*wf.INITIAL_CAPITAL):.2f}% annualized)")
print(f"SPY $ on ${wf.INITIAL_CAPITAL:,.0f}: ${spy_ret/100*wf.INITIAL_CAPITAL:,.0f}")
print(f"\nGross strategy annualized: {annualized(gross):.2f}%   |   SPY annualized: {annualized(spy_ret/100*wf.INITIAL_CAPITAL):.2f}%")

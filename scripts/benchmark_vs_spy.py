#!/usr/bin/env python3
"""
SPY Buy-and-Hold Benchmark — the number the Operator Mandate is measured against.

The walk-forward backtest (scripts/run_walkforward_v2.py) reports strategy
returns but NEVER compares them to simply holding SPY over the same window.
That comparison is the whole point of the mandate: a strategy that returns
+4%/yr while SPY returns +18%/yr is an expensive way to underperform an index
fund.

This script computes SPY total/annualized return over the EXACT active span of
the walk-forward backtest (trading-day index 91..600 of the loaded daily
series, which is the contiguous union of walk-forward windows 1..6), so the two
numbers are directly comparable on the same capital and the same dates.

Usage:
    python scripts/benchmark_vs_spy.py

Keep this in sync with run_walkforward_v2.py if DATA_DAYS or the window
indices change there.
"""
import sys
from pathlib import Path
from datetime import date, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.data_loader import DataLoader  # noqa: E402

# Mirror run_walkforward_v2.py
DATA_DAYS = 730
WARMUP_DAYS = 250
START_IDX = 91    # first trading day of Window 1
END_IDX = 600     # last trading day of Window 6
INITIAL_CAPITAL = 25000.0


def main():
    end_date = date.today()
    start_date = end_date - timedelta(days=DATA_DAYS + WARMUP_DAYS)

    loader = DataLoader(cache_dir=str(PROJECT_ROOT / "data" / "backtest_cache"))
    spy = loader.load_spy_data(start_date, end_date, use_cache=True)
    close_col = "Close" if "Close" in spy.columns else "close"
    dates = sorted(spy.index.date)
    close = spy[close_col]

    def px_at(idx):
        d = dates[min(idx, len(dates) - 1)]
        return float(close[close.index.date == d].iloc[0]), d

    p0, d0 = px_at(START_IDX)
    p1, d1 = px_at(END_IDX)
    years = (d1 - d0).days / 365.25
    total_ret = (p1 / p0 - 1) * 100
    annualized = ((p1 / p0) ** (1 / years) - 1) * 100

    span = close[(close.index.date >= d0) & (close.index.date <= d1)]
    dd = (span / span.cummax() - 1).min() * 100

    print("=" * 72)
    print("SPY BUY-AND-HOLD BENCHMARK (same active span as walk-forward V2)")
    print("=" * 72)
    print(f"Span (idx {START_IDX}..{END_IDX}):  {d0}  ->  {d1}   (~{years:.2f} yrs)")
    print(f"SPY price:          ${p0:,.2f}  ->  ${p1:,.2f}")
    print(f"TOTAL RETURN:       {total_ret:+.2f}%")
    print(f"ANNUALIZED:         {annualized:+.2f}%")
    print(f"SPY max drawdown:   {dd:.2f}%")
    print("-" * 72)
    print(f"On ${INITIAL_CAPITAL:,.0f}:  ending ${INITIAL_CAPITAL * p1 / p0:,.0f}  "
          f"(profit ${INITIAL_CAPITAL * (p1 / p0 - 1):+,.0f})")
    print("=" * 72)
    print("Compare against the AGGREGATE strategy return from run_walkforward_v2.py.")


if __name__ == "__main__":
    main()

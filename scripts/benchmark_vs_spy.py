#!/usr/bin/env python3
"""
benchmark_vs_spy.py — The only benchmark the mandate cares about.

The RDT operator's mission is *actual positive P&L net of honest costs, beating
SPY buy-and-hold*. Strategy backtests in this repo (run_walkforward_v2.py, etc.)
report absolute returns but never compare against the zero-effort alternative of
simply holding SPY over the same window. This script computes that hurdle so
every operator session can state, in one number, whether the active strategy is
worth running at all.

Usage:
    python scripts/benchmark_vs_spy.py                     # trailing 730 days
    python scripts/benchmark_vs_spy.py --days 365
    python scripts/benchmark_vs_spy.py --start 2024-01-01 --end 2026-01-01
    python scripts/benchmark_vs_spy.py --days 730 --strategy-return-pct 8.69

If --strategy-return-pct is supplied (e.g. the aggregate % from a walk-forward
run over the same window), the script prints a verdict: does the strategy beat
buy-and-hold, and by how much per year.

No dependencies beyond yfinance + pandas (already required by the backtests).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta


def spy_buy_and_hold(start: str, end: str) -> dict:
    import yfinance as yf

    df = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        raise RuntimeError("No SPY data returned from yfinance")
    close = df["Close"].dropna()
    first = float(close.iloc[0].item())
    last = float(close.iloc[-1].item())
    d0, d1 = close.index[0].date(), close.index[-1].date()
    days = max((d1 - d0).days, 1)
    total_ret = (last / first - 1.0) * 100.0
    annualized = ((last / first) ** (365.0 / days) - 1.0) * 100.0
    return {
        "start": d0,
        "end": d1,
        "calendar_days": days,
        "trading_days": len(close),
        "first": first,
        "last": last,
        "total_return_pct": total_ret,
        "annualized_pct": annualized,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark a strategy return against SPY buy-and-hold.")
    p.add_argument("--days", type=int, default=730, help="Trailing window length in days (default 730).")
    p.add_argument("--start", type=str, default=None, help="Explicit start date YYYY-MM-DD (overrides --days).")
    p.add_argument("--end", type=str, default=None, help="Explicit end date YYYY-MM-DD (default today).")
    p.add_argument("--capital", type=float, default=25000.0, help="Account size for dollar figures.")
    p.add_argument("--strategy-return-pct", type=float, default=None,
                   help="Strategy total return %% over the SAME window, for a verdict.")
    args = p.parse_args()

    end = args.end or str(date.today())
    if args.start:
        start = args.start
    else:
        start = str(date.fromisoformat(end) - timedelta(days=args.days))

    try:
        spy = spy_buy_and_hold(start, end)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: could not compute SPY benchmark: {e}", file=sys.stderr)
        return 1

    cap = args.capital
    print("=" * 72)
    print("SPY BUY-AND-HOLD HURDLE".center(72))
    print("=" * 72)
    print(f"  Window            : {spy['start']} -> {spy['end']} "
          f"({spy['calendar_days']} cal days, {spy['trading_days']} trading days)")
    print(f"  SPY price         : {spy['first']:.2f} -> {spy['last']:.2f}")
    print(f"  Total return      : {spy['total_return_pct']:+.2f}%")
    print(f"  Annualized        : {spy['annualized_pct']:+.2f}%")
    print(f"  On ${cap:,.0f}      : {cap * spy['total_return_pct'] / 100.0:+,.0f} (zero effort, zero overnight monitoring)")

    if args.strategy_return_pct is not None:
        strat = args.strategy_return_pct
        strat_ann = ((1.0 + strat / 100.0) ** (365.0 / spy["calendar_days"]) - 1.0) * 100.0
        gap = strat - spy["total_return_pct"]
        gap_ann = strat_ann - spy["annualized_pct"]
        print("-" * 72)
        print("STRATEGY vs BUY-AND-HOLD".center(72))
        print("-" * 72)
        print(f"  Strategy total    : {strat:+.2f}%  (annualized {strat_ann:+.2f}%)")
        print(f"  Excess vs SPY     : {gap:+.2f}% total  |  {gap_ann:+.2f}%/yr")
        print(f"  On ${cap:,.0f}      : strategy {cap * strat / 100.0:+,.0f}  vs  SPY {cap * spy['total_return_pct'] / 100.0:+,.0f}")
        print("-" * 72)
        if gap_ann > 0:
            print(f"  VERDICT: Strategy BEATS buy-and-hold by {gap_ann:.2f}%/yr. "
                  f"Worth running IF this survives honest intraday costs.")
        else:
            print(f"  VERDICT: Strategy TRAILS buy-and-hold by {abs(gap_ann):.2f}%/yr. "
                  f"On this evidence, holding SPY is strictly better.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

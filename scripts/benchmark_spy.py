#!/usr/bin/env python3
"""
SPY Buy-and-Hold Benchmark
==========================

The operator mandate's benchmark for the trading bot is *SPY buy-and-hold over the
identical window, same starting capital*. The repo's walk-forward scripts report
strategy returns but never compute this benchmark, so there was no honest
apples-to-apples comparison. This script fills that gap.

It computes SPY total return (dividend-adjusted, via yfinance auto_adjust) over:
  1. The full trailing window used by run_walkforward_v2.py (DATA_DAYS = 730 calendar days)
  2. The post-warmup window (skips the first 90 trading days, matching the walk-forward
     windows which start at trading-day index 91) — this is the period the strategy's
     summed window returns actually cover.

Usage:
    python scripts/benchmark_spy.py [--days 730] [--warmup-trading-days 90] [--capital 25000]

Output is plain text: start/end dates, SPY start/end price, total return %, annualized %,
and the dollar P&L on the given starting capital. No live trading, no side effects.
"""

import argparse
from datetime import date, timedelta

import yfinance as yf


def spy_buy_and_hold(start: date, end: date, capital: float):
    """Return (start_date, end_date, start_px, end_px, total_ret_pct, dollars)."""
    df = yf.Ticker("SPY").history(start=start, end=end + timedelta(days=1), auto_adjust=True)
    if df is None or len(df) == 0:
        raise SystemExit("ERROR: could not load SPY data from yfinance")
    df = df.sort_index()
    start_px = float(df["Close"].iloc[0])
    end_px = float(df["Close"].iloc[-1])
    total_ret = (end_px / start_px) - 1.0
    dollars = capital * total_ret
    return (
        df.index[0].date(),
        df.index[-1].date(),
        start_px,
        end_px,
        total_ret * 100.0,
        dollars,
    )


def annualized(total_ret_pct: float, start: date, end: date) -> float:
    years = max((end - start).days / 365.25, 1e-9)
    return ((1.0 + total_ret_pct / 100.0) ** (1.0 / years) - 1.0) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=730,
                    help="Trailing calendar-day window (matches run_walkforward_v2 DATA_DAYS)")
    ap.add_argument("--warmup-trading-days", type=int, default=90,
                    help="Trading days to skip for warmup (walk-forward windows start at ~index 91)")
    ap.add_argument("--capital", type=float, default=25000.0)
    args = ap.parse_args()

    end = date.today()
    start = end - timedelta(days=args.days)

    # Full window
    sd, ed, sp, ep, ret, usd = spy_buy_and_hold(start, end, args.capital)
    ann = annualized(ret, sd, ed)

    print("=" * 70)
    print("SPY BUY-AND-HOLD BENCHMARK (dividend-adjusted close)")
    print("=" * 70)
    print(f"Starting capital:       ${args.capital:,.0f}")
    print()
    print(f"[Full {args.days}-day window]")
    print(f"  Dates:                {sd} -> {ed}")
    print(f"  SPY price:            ${sp:,.2f} -> ${ep:,.2f}")
    print(f"  Total return:         {ret:+.2f}%  (${usd:+,.0f})")
    print(f"  Annualized:           {ann:+.2f}%")

    # Post-warmup window: fetch full, then slice off the first N trading days
    df = yf.Ticker("SPY").history(start=start, end=end + timedelta(days=1), auto_adjust=True).sort_index()
    if len(df) > args.warmup_trading_days + 1:
        warm = df.iloc[args.warmup_trading_days:]
        w_sd, w_ed = warm.index[0].date(), warm.index[-1].date()
        w_sp, w_ep = float(warm["Close"].iloc[0]), float(warm["Close"].iloc[-1])
        w_ret = (w_ep / w_sp - 1.0) * 100.0
        w_usd = args.capital * w_ret / 100.0
        w_ann = annualized(w_ret, w_sd, w_ed)
        print()
        print(f"[Post-warmup window — skip first {args.warmup_trading_days} trading days]")
        print(f"  (This is the period the walk-forward strategy windows actually cover.)")
        print(f"  Dates:                {w_sd} -> {w_ed}")
        print(f"  SPY price:            ${w_sp:,.2f} -> ${w_ep:,.2f}")
        print(f"  Total return:         {w_ret:+.2f}%  (${w_usd:+,.0f})")
        print(f"  Annualized:           {w_ann:+.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()

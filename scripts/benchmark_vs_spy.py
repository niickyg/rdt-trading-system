#!/usr/bin/env python3
"""
Honest benchmark: strategy P&L vs SPY buy-and-hold vs risk-free, with a cost overlay.

The mandate's central question every operator session is the same:
    "Does this bot, NET OF HONEST COSTS, beat SPY buy-and-hold over a comparable window?"

The repo's backtest engine (backtesting/engine_enhanced.py) models NO commissions,
slippage, or borrow — its P&L is raw (exit - entry) * shares (see lines ~423-428).
So every headline backtest number is a *gross, frictionless* figure. This script makes
the honest comparison explicit and reproducible in one command.

Usage:
    # Compare a frictionless backtest result to SPY over the same span:
    python scripts/benchmark_vs_spy.py \
        --start 2024-07-08 --end 2026-07-20 \
        --capital 25000 --gross-pnl 2472.19 --trades 280

It prints SPY buy-and-hold over the span, the risk-free hurdle, and a cost-sensitivity
table showing the strategy's NET return under a range of per-round-trip friction
assumptions. No trading, no broker, read-only market data via yfinance.
"""
import argparse
import sys
import warnings

warnings.filterwarnings("ignore")


def spy_buy_and_hold(start: str, end: str):
    import yfinance as yf

    df = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        raise RuntimeError("No SPY data returned for the requested span.")
    close = df["Close"]
    close = close.iloc[:, 0] if hasattr(close, "columns") else close
    s, e = float(close.iloc[0]), float(close.iloc[-1])
    days = (close.index[-1] - close.index[0]).days
    years = days / 365.25
    total = e / s - 1.0
    ann = (e / s) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    return {
        "first_date": close.index[0].date(),
        "last_date": close.index[-1].date(),
        "years": years,
        "total_pct": total * 100,
        "ann_pct": ann * 100,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True, help="Span start (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="Span end (YYYY-MM-DD)")
    ap.add_argument("--capital", type=float, default=25000.0)
    ap.add_argument("--gross-pnl", type=float, required=True,
                    help="Frictionless backtest profit ($) over the span")
    ap.add_argument("--trades", type=int, required=True,
                    help="Number of trades that produced the gross P&L")
    ap.add_argument("--risk-free", type=float, default=4.5,
                    help="Annualized risk-free rate %% for the hurdle (default 4.5)")
    args = ap.parse_args()

    spy = spy_buy_and_hold(args.start, args.end)
    yrs = spy["years"]

    print("=" * 78)
    print("  HONEST BENCHMARK: strategy vs SPY buy-and-hold vs risk-free")
    print("=" * 78)
    print(f"  Span: {spy['first_date']} -> {spy['last_date']}  ({yrs:.2f} years)")
    print(f"  Capital: ${args.capital:,.0f}\n")

    spy_dollars = args.capital * spy["total_pct"] / 100.0
    print(f"  SPY buy & hold : {spy['total_pct']:+6.1f}% total  "
          f"{spy['ann_pct']:+5.1f}%/yr  (${spy_dollars:,.0f})")

    rf_total = ((1 + args.risk_free / 100.0) ** yrs - 1) * 100
    rf_dollars = args.capital * rf_total / 100.0
    print(f"  Risk-free ({args.risk_free:.1f}%): {rf_total:+6.1f}% total  "
          f"{args.risk_free:+5.1f}%/yr  (${rf_dollars:,.0f})")

    g_total = args.gross_pnl / args.capital * 100
    g_ann = ((1 + g_total / 100.0) ** (1.0 / yrs) - 1) * 100 if yrs > 0 else float("nan")
    print(f"  Strategy GROSS : {g_total:+6.1f}% total  {g_ann:+5.1f}%/yr  "
          f"(${args.gross_pnl:,.0f})  [frictionless]")

    print(f"\n  Cost sensitivity ({args.trades} trades, net of per-round-trip friction):")
    print("  " + "-" * 60)
    for rt in (3, 5, 8, 12):
        net = args.gross_pnl - args.trades * rt
        net_total = net / args.capital * 100
        net_ann = ((1 + net_total / 100.0) ** (1.0 / yrs) - 1) * 100 if yrs > 0 else float("nan")
        beats_spy = "YES" if net_total > spy["total_pct"] else "no"
        beats_rf = "YES" if net_total > rf_total else "no"
        print(f"    ${rt:>2}/round-trip: net ${net:>8,.0f}  "
              f"{net_total:+6.1f}% total  {net_ann:+5.1f}%/yr  "
              f"| beats SPY? {beats_spy:<3}  beats risk-free? {beats_rf}")

    print("\n" + "=" * 78)
    verdict_gross = "BEATS" if g_total > spy["total_pct"] else "LOSES TO"
    print(f"  VERDICT (even frictionless): strategy {verdict_gross} SPY buy-and-hold.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

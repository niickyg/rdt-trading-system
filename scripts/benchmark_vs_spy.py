#!/usr/bin/env python3
"""
Benchmark helper: report a strategy's return against SPY buy-and-hold total return.

The operator mandate (data/operator_journal/MANDATE.md) says the bar for "success"
is beating SPY buy-and-hold over the SAME period, net of honest costs. This helper
makes that comparison the headline number, using a committed, real SPY price series
so it works with no network access (the remote agent env blocks Yahoo/yfinance).

Usage:
    # Show the SPY buy-and-hold benchmark over the cached window:
    python scripts/benchmark_vs_spy.py

    # Compare a strategy result against SPY over the same window:
    python scripts/benchmark_vs_spy.py --strategy-return-pct 6.9 --months 21

SPY data: data/benchmarks/spy_monthly_2yr_20260819.csv (real, from IBKR, price-only).
Dividends add ~1.2%/yr; pass --dividend-yield to include them in the SPY bar.
"""
import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "benchmarks" / "spy_monthly_2yr_20260819.csv"


def load_spy(csv_path: Path):
    rows = []
    with open(csv_path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("date"):
                continue
            r = next(csv.reader([line]))
            rows.append((r[0], float(r[4])))  # date, close
    return rows


def annualize(total_pct: float, months: float) -> float:
    if months <= 0:
        return total_pct
    return ((1 + total_pct / 100) ** (12.0 / months) - 1) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--strategy-return-pct", type=float, default=None,
                    help="Strategy TOTAL return %% over the window (net of costs).")
    ap.add_argument("--months", type=float, default=None,
                    help="Length of the window in months (for annualization).")
    ap.add_argument("--dividend-yield", type=float, default=0.0,
                    help="Annual SPY dividend yield %% to add to price return (e.g. 1.2).")
    args = ap.parse_args()

    spy = load_spy(Path(args.csv))
    start_date, start_px = spy[0]
    end_date, end_px = spy[-1]
    window_months = args.months
    if window_months is None:
        window_months = len(spy) - 1  # monthly bars

    spy_price_total = (end_px / start_px - 1) * 100
    div_add = args.dividend_yield * (window_months / 12.0)
    spy_total = spy_price_total + div_add
    spy_ann = annualize(spy_total, window_months)

    print(f"\nSPY buy-and-hold benchmark  [{start_date} -> {end_date}, ~{window_months:.0f} mo]")
    print(f"  price return : {spy_price_total:+.1f}%  ({annualize(spy_price_total, window_months):+.1f}% annualized)")
    if args.dividend_yield:
        print(f"  + dividends  : {div_add:+.1f}%  (at {args.dividend_yield:.1f}%/yr)")
    print(f"  TOTAL return : {spy_total:+.1f}%  ({spy_ann:+.1f}% annualized)   <-- the bar to beat\n")

    if args.strategy_return_pct is not None:
        strat_ann = annualize(args.strategy_return_pct, window_months)
        gap = args.strategy_return_pct - spy_total
        print(f"Strategy      : {args.strategy_return_pct:+.1f}%  ({strat_ann:+.1f}% annualized)")
        verdict = "BEATS SPY" if gap > 0 else "LOSES TO SPY"
        print(f"Gap vs SPY    : {gap:+.1f} pts total return   ==>  {verdict}\n")


if __name__ == "__main__":
    main()

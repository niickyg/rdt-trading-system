#!/usr/bin/env python3
"""
Honest Benchmark Backtest — net-of-cost strategy return vs SPY buy-and-hold.

Why this exists
---------------
The repository's headline walk-forward results (CLAUDE.md, WEALTH_*.md) have two
material omissions that make them misleading as evidence of profitability:

  1. NO transaction costs. backtesting/engine_enhanced.py never deducts
     commission, slippage, or spread. The quoted ~6.9% / 3.4%-annualized is a
     GROSS figure for a strategy that takes hundreds of trades.

  2. NO benchmark. The reported returns are never compared to simply buying and
     holding SPY over the same period — the single most important yardstick for
     deciding whether an actively-managed bot creates or destroys value.

This script fixes both. It runs ONE contiguous backtest (compounding, like a
real account) over ~2 years ending today, subtracts realistic round-trip costs
from every trade, and prints the net result next to SPY buy-and-hold over the
identical window.

Cost model (deliberately generous to the strategy — real day-trading is worse):
  - Commission: $0.005/share each way, $1.00 minimum per order (IBKR-like)
  - Slippage + half-spread: 5 bps of notional each way (0.10% round trip)
  Scale-out fills are NOT charged extra commission, which understates cost in
  the strategy's favour.

Usage:
    cd /home/user/rdt-trading-system
    python scripts/honest_benchmark_backtest.py [--years 2] [--slippage-bps 5]
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

import scripts.run_walkforward_v2 as wf  # noqa: E402
from backtesting.engine_enhanced import EnhancedBacktestEngine  # noqa: E402

INITIAL_CAPITAL = wf.INITIAL_CAPITAL


def round_trip_cost(trade, commission_per_share=0.005, min_commission=1.0, slippage_bps=5.0):
    """Realistic round-trip cost for one position (entry + exit)."""
    shares = abs(trade.shares)
    entry_notional = trade.entry_price * shares
    exit_price = trade.exit_price if trade.exit_price else trade.entry_price
    exit_notional = exit_price * shares
    commission = max(min_commission, commission_per_share * shares) * 2.0
    slippage = (slippage_bps / 10000.0) * (entry_notional + exit_notional)
    return commission + slippage


def summarize(result, slippage_bps):
    trades = result.trades
    gross = result.total_return
    costs = sum(round_trip_cost(t, slippage_bps=slippage_bps) for t in trades)
    net = gross - costs
    wins = sum(1 for t in trades if t.pnl > 0)
    n = len(trades)
    return {
        "trades": n,
        "win_rate_pct": (wins / n * 100) if n else 0.0,
        "gross_return": gross,
        "gross_return_pct": gross / INITIAL_CAPITAL * 100,
        "total_cost": costs,
        "cost_per_trade": (costs / n) if n else 0.0,
        "net_return": net,
        "net_return_pct": net / INITIAL_CAPITAL * 100,
        "max_drawdown": result.max_drawdown,
    }


def spy_buy_hold(spy_data, start_date, end_date):
    close_col = 'Close' if 'Close' in spy_data.columns else 'close'
    mask = (spy_data.index.date >= start_date) & (spy_data.index.date <= end_date)
    s = spy_data.loc[mask, close_col].astype(float)
    if len(s) < 2:
        return None
    ret_pct = (float(s.iloc[-1]) / float(s.iloc[0]) - 1) * 100
    dollars = INITIAL_CAPITAL * ret_pct / 100
    return {"start": float(s.iloc[0]), "end": float(s.iloc[-1]),
            "return_pct": ret_pct, "return_dollars": dollars}


def avg_capital_utilization(result):
    eq = result.equity_curve
    if not eq:
        return None
    # equity_curve dicts carry 'equity' and possibly 'cash'/'positions_value'
    sample = eq[0]
    if 'cash' in sample:
        util = [1 - (e.get('cash', 0) / e['equity']) for e in eq if e.get('equity')]
        return float(np.mean(util) * 100) if util else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    args = ap.parse_args()

    print("=" * 100)
    print("HONEST BENCHMARK BACKTEST — net-of-cost strategy vs SPY buy-and-hold".center(100))
    print("=" * 100)
    print(f"Initial capital: ${INITIAL_CAPITAL:,.0f} | slippage+spread: {args.slippage_bps:.0f} bps/side | "
          f"commission: $0.005/sh (min $1)")
    print()

    stock_data, spy_data, vix_data, sector_etf_data = wf.load_all_data()
    if len(stock_data) < 5:
        print("ERROR: too few symbols loaded (yfinance connectivity).")
        sys.exit(1)

    dates = sorted(spy_data.index.date)
    # Need ~200 bars warmup for the 200-SMA/EMA gates. Then run contiguously.
    trading_days_needed = int(args.years * 252)
    warmup = 210
    start_idx = max(warmup, len(dates) - trading_days_needed - 1)
    start_date = dates[start_idx]
    end_date = dates[-1]
    print(f"Contiguous test window: {start_date} -> {end_date} "
          f"({len([d for d in dates if start_date <= d <= end_date])} trading days)")
    print()

    # --- A) Baseline (no filters) ---
    base_engine = EnhancedBacktestEngine(**wf.make_engine_kwargs(wf.BASE_ENGINE_CONFIG))
    base_result = base_engine.run(stock_data, spy_data, start_date=start_date, end_date=end_date)

    # --- C) RDT-filtered (repo's "best") ---
    rdt_kwargs = wf.make_engine_kwargs(wf.BASE_ENGINE_CONFIG)
    rdt_kwargs["vix_data"] = vix_data
    rdt_kwargs["sector_etf_data"] = sector_etf_data
    rdt_engine = wf.RDTFilteredEngine(**rdt_kwargs)
    rdt_result = rdt_engine.run(stock_data, spy_data, start_date=start_date, end_date=end_date)

    base = summarize(base_result, args.slippage_bps)
    rdt = summarize(rdt_result, args.slippage_bps)
    bench = spy_buy_hold(spy_data, start_date, end_date)

    def block(title, s):
        print(f"--- {title} ---")
        print(f"  Trades:            {s['trades']}")
        print(f"  Win rate:          {s['win_rate_pct']:.1f}%")
        print(f"  GROSS return:      ${s['gross_return']:>+12,.2f}  ({s['gross_return_pct']:+.2f}%)")
        print(f"  Transaction costs: ${s['total_cost']:>12,.2f}  (${s['cost_per_trade']:.2f}/trade)")
        print(f"  NET return:        ${s['net_return']:>+12,.2f}  ({s['net_return_pct']:+.2f}%)")
        print(f"  Max drawdown:      ${s['max_drawdown']:>12,.2f}")
        print()

    block("A) Baseline (no filters, RRS 2.0)", base)
    block("C) RDT-filtered (SPY gate + SMA + MTF + VIX + sector + regime)", rdt)

    print("--- BENCHMARK: SPY buy-and-hold (fully invested, 1 trade) ---")
    if bench:
        print(f"  SPY: ${bench['start']:.2f} -> ${bench['end']:.2f}")
        print(f"  Return:            ${bench['return_dollars']:>+12,.2f}  ({bench['return_pct']:+.2f}%)")
    print()

    years = (end_date - start_date).days / 365.25
    print("=" * 100)
    print("VERDICT".center(100))
    print("=" * 100)
    for label, s in [("Baseline (net)", base), ("RDT-filtered (net)", rdt)]:
        ann = ((1 + s['net_return_pct'] / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
        print(f"  {label:<22} {s['net_return_pct']:+7.2f}% total   {ann:+6.2f}% annualized")
    if bench:
        ann_b = ((1 + bench['return_pct'] / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
        print(f"  {'SPY buy-and-hold':<22} {bench['return_pct']:+7.2f}% total   {ann_b:+6.2f}% annualized")
        gap = rdt['net_return_pct'] - bench['return_pct']
        print()
        print(f"  RDT-filtered NET vs SPY buy-and-hold: {gap:+.2f} percentage points over {years:.2f} years")
        print(f"  => Strategy {'BEAT' if gap > 0 else 'UNDERPERFORMED'} simply holding SPY.")
    print("=" * 100)

    out = {
        "generated_utc": date.today().isoformat(),
        "window": {"start": str(start_date), "end": str(end_date), "years": round(years, 2)},
        "cost_model": {"commission_per_share": 0.005, "min_commission": 1.0,
                       "slippage_bps_per_side": args.slippage_bps},
        "symbols_loaded": len(stock_data),
        "baseline_net": base, "rdt_filtered_net": rdt, "spy_buy_hold": bench,
    }
    out_path = PROJECT_ROOT / "data" / "operator_journal" / "evidence" / f"honest_benchmark_{date.today().isoformat()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nEvidence written: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

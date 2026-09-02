#!/usr/bin/env python3
"""
Honest Benchmark Harness — is the strategy actually worth running?

The single question this answers: over a real window, does the RRS strategy beat
SPY buy-and-hold **net of honest transaction costs**? That is the only success
criterion in the operator mandate (data/operator_journal/MANDATE.md). Every
existing backtest script compares filter configs against each other; NONE of
them computes the benchmark that matters — so this exists to fill that gap.

Design notes
------------
- The cost model, SPY buy-and-hold, and PASS/FAIL verdict are PURE functions so
  they can be unit-tested without network or market data
  (tests/unit/test_honest_benchmark.py).
- The data-dependent wiring lives in main(). In the remote agent sandbox, Yahoo
  Finance egress is blocked, so DataLoader (yfinance) cannot fetch data there —
  main() will report that honestly rather than pretend. Run this on the human's
  infra, or wire in an alternate data source (e.g. the IBKR MCP tools).

Usage:
    python scripts/honest_benchmark.py --days 730
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Pure, testable core
# ============================================================================

# IBKR-tiered-style defaults. Deliberately conservative, not optimistic.
DEFAULT_COMMISSION_PER_SHARE = 0.005   # $/share
DEFAULT_MIN_COMMISSION = 1.0           # $ per order
DEFAULT_SLIPPAGE_BPS = 5.0             # basis points of notional, per leg


def round_trip_cost(
    shares: float,
    entry_price: float,
    exit_price: float,
    commission_per_share: float = DEFAULT_COMMISSION_PER_SHARE,
    min_commission: float = DEFAULT_MIN_COMMISSION,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> float:
    """Honest round-trip cost (entry + exit) for one trade, in dollars.

    Two commissions (one per leg, each floored at the per-order minimum) plus
    slippage on both the entry and exit notionals. Never negative.
    """
    shares = abs(shares)
    if shares <= 0:
        return 0.0
    commission = 2.0 * max(min_commission, commission_per_share * shares)
    slippage = (slippage_bps / 1e4) * (shares * abs(entry_price) + shares * abs(exit_price))
    return commission + slippage


def spy_buy_and_hold_pct(spy_df, start_date: date, end_date: date) -> Optional[float]:
    """Percent return of buying SPY at the first close in [start,end] and holding
    to the last close. Returns None if there isn't enough data. Expects a
    DataFrame with a DatetimeIndex and a 'Close'/'close' column.
    """
    close_col = "Close" if "Close" in spy_df.columns else "close"
    sub = spy_df[(spy_df.index.date >= start_date) & (spy_df.index.date <= end_date)]
    if len(sub) < 2:
        return None
    p0 = float(sub[close_col].iloc[0])
    p1 = float(sub[close_col].iloc[-1])
    if p0 <= 0:
        return None
    return (p1 / p0 - 1.0) * 100.0


@dataclass
class BenchmarkVerdict:
    strategy_gross_pct: float
    strategy_cost_pct: float
    strategy_net_pct: float
    spy_net_pct: float
    edge_pct: float          # strategy_net - spy_net
    passes: bool
    num_trades: int
    note: str = ""

    def render(self) -> str:
        status = "PASS ✅" if self.passes else "FAIL ❌"
        lines = [
            "=" * 60,
            "HONEST BENCHMARK — strategy vs SPY buy-and-hold (net of costs)",
            "=" * 60,
            f"  Strategy gross return : {self.strategy_gross_pct:+.2f}%",
            f"  Transaction costs     : {self.strategy_cost_pct:.2f}%  ({self.num_trades} trades)",
            f"  Strategy NET return   : {self.strategy_net_pct:+.2f}%",
            f"  SPY buy-and-hold NET  : {self.spy_net_pct:+.2f}%",
            "-" * 60,
            f"  Edge vs SPY           : {self.edge_pct:+.2f}%",
            f"  VERDICT               : {status}",
        ]
        if self.note:
            lines.append(f"  Note                  : {self.note}")
        lines.append("=" * 60)
        return "\n".join(lines)


def make_verdict(
    strategy_gross_return_dollars: float,
    total_cost_dollars: float,
    initial_capital: float,
    spy_net_pct: Optional[float],
    num_trades: int,
    spy_round_trip_cost_dollars: float = 0.0,
    min_sample: int = 30,
) -> BenchmarkVerdict:
    """Combine strategy P&L, costs, and the SPY benchmark into a PASS/FAIL.

    PASS requires the strategy's NET return to exceed SPY buy-and-hold's NET
    return AND a sample of at least `min_sample` trades (a thin sample cannot
    establish an edge, per the mandate).
    """
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")

    gross_pct = strategy_gross_return_dollars / initial_capital * 100.0
    cost_pct = total_cost_dollars / initial_capital * 100.0
    net_pct = gross_pct - cost_pct

    if spy_net_pct is None:
        return BenchmarkVerdict(
            strategy_gross_pct=gross_pct,
            strategy_cost_pct=cost_pct,
            strategy_net_pct=net_pct,
            spy_net_pct=float("nan"),
            edge_pct=float("nan"),
            passes=False,
            num_trades=num_trades,
            note="No SPY benchmark available — cannot judge. Treated as FAIL.",
        )

    spy_net = spy_net_pct - (spy_round_trip_cost_dollars / initial_capital * 100.0)
    edge = net_pct - spy_net
    enough_sample = num_trades >= min_sample
    passes = (edge > 0.0) and enough_sample

    note = ""
    if not enough_sample:
        note = f"Sample too small (n={num_trades} < {min_sample}) to establish an edge."
    return BenchmarkVerdict(
        strategy_gross_pct=gross_pct,
        strategy_cost_pct=cost_pct,
        strategy_net_pct=net_pct,
        spy_net_pct=spy_net,
        edge_pct=edge,
        passes=passes,
        num_trades=num_trades,
        note=note,
    )


def total_costs_for_trades(trades) -> float:
    """Sum round-trip costs over a list of executed trades. Each trade is
    expected to expose `.shares`, `.entry_price`, `.exit_price` (skips trades
    with no exit)."""
    total = 0.0
    for t in trades:
        exit_price = getattr(t, "exit_price", None)
        if exit_price is None:
            continue
        total += round_trip_cost(
            getattr(t, "shares", 0) or 0,
            getattr(t, "entry_price", 0.0) or 0.0,
            exit_price,
        )
    return total


# ============================================================================
# Data-dependent driver (needs market data; degrades honestly without it)
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Honest benchmark: strategy vs SPY buy-and-hold")
    parser.add_argument("--days", type=int, default=730, help="Lookback window in days")
    parser.add_argument("--rrs-threshold", type=float, default=2.0)
    parser.add_argument("--capital", type=float, default=25000.0)
    args = parser.parse_args()

    end_date = date.today()
    start_date = end_date - timedelta(days=args.days)

    try:
        from backtesting.data_loader import DataLoader
        from backtesting.engine_enhanced import EnhancedBacktestEngine
    except Exception as e:  # pragma: no cover - import guard
        print(f"ERROR: could not import backtest engine: {e}")
        return 2

    watchlist = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V",
        "JNJ", "UNH", "HD", "PG", "MA", "DIS", "ADBE", "CRM", "NFLX", "AMD", "AVGO",
    ]

    # No cache_dir: caching writes parquet, which would add a hard pyarrow
    # dependency. The harness runs fine (just re-downloads) without it.
    loader = DataLoader(cache_dir=None)
    print(f"Loading data {start_date} → {end_date} ...")
    try:
        spy_data = loader._load_symbol("SPY", start_date, end_date, use_cache=False)
    except Exception as e:
        print(f"\nDATA LOAD FAILED ({type(e).__name__}: {e}).\nReporting UNVALIDATED — no fabricated result.")
        return 3
    if spy_data is None or len(spy_data) < 2:
        print(
            "\nCOULD NOT LOAD MARKET DATA (SPY empty).\n"
            "In the remote agent sandbox, Yahoo Finance egress is blocked, so\n"
            "yfinance returns nothing. Run this on infra with data access, or\n"
            "wire an alternate source (IBKR MCP). Reporting UNVALIDATED, not a\n"
            "fabricated result."
        )
        return 3

    stock_data = loader.load_stock_data(watchlist, start_date, end_date, use_cache=True)
    if not stock_data:
        print("COULD NOT LOAD any stock data. Aborting (UNVALIDATED).")
        return 3

    engine = EnhancedBacktestEngine(
        initial_capital=args.capital, rrs_threshold=args.rrs_threshold
    )
    result = engine.run(stock_data, spy_data, start_date=start_date, end_date=end_date)

    spy_pct = spy_buy_and_hold_pct(spy_data, start_date, end_date)
    costs = total_costs_for_trades(result.trades)
    spy_rt = round_trip_cost(
        args.capital / float(spy_data["Close" if "Close" in spy_data.columns else "close"].iloc[0]),
        float(spy_data["Close" if "Close" in spy_data.columns else "close"].iloc[0]),
        float(spy_data["Close" if "Close" in spy_data.columns else "close"].iloc[-1]),
    )
    verdict = make_verdict(
        strategy_gross_return_dollars=result.total_return,
        total_cost_dollars=costs,
        initial_capital=args.capital,
        spy_net_pct=spy_pct,
        num_trades=result.total_trades,
        spy_round_trip_cost_dollars=spy_rt,
    )
    print(verdict.render())
    return 0 if verdict.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Intraday Walk-Forward Backtest using 1-minute Alpaca data.

Compares 3 configs with real intraday data:
  A) Baseline — No filters, no VWAP, no first-hour block
  B) RDT Filters — SPY gate + SMA + VWAP + first-hour block (no intermarket)
  C) RDT + Intermarket — B + intermarket analysis

This is the definitive test using real 1-minute bars.

Usage:
    cd /home/user0/rdt-trading-system
    python scripts/run_intraday_backtest.py
"""

import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from backtesting.engine_intraday import IntradayBacktestEngine
from backtesting.engine_enhanced import EnhancedBacktestResult
from backtesting.data_loader import DataLoader
from risk.models import RiskLimits

# ============================================================================
# Configuration
# ============================================================================

INITIAL_CAPITAL = 25000.0
INTRADAY_CACHE = PROJECT_ROOT / "data" / "intraday_cache"

STOCK_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    'JPM', 'V', 'JNJ', 'UNH', 'HD', 'PG', 'MA', 'DIS',
    'PYPL', 'ADBE', 'CRM', 'NFLX', 'INTC', 'AMD', 'CSCO',
    'PEP', 'KO', 'MRK', 'ABT', 'TMO', 'COST', 'AVGO', 'TXN'
]

# We have ~271 trading days. Use 4 walk-forward windows (~60 days each),
# skipping first 30 days for indicator warmup.
WALK_FORWARD_WINDOWS = [
    {"name": "W1 (Apr-Jun 2025)", "start_day": 30,  "end_day": 90},
    {"name": "W2 (Jul-Sep 2025)", "start_day": 91,  "end_day": 150},
    {"name": "W3 (Oct-Dec 2025)", "start_day": 151, "end_day": 210},
    {"name": "W4 (Jan-Feb 2026)", "start_day": 211, "end_day": 270},
]

BASE_CONFIG = {
    "initial_capital": INITIAL_CAPITAL,
    "rrs_threshold": 2.0,
    "max_positions": 8,
    "stop_atr_multiplier": 1.5,
    "target_atr_multiplier": 2.0,
    "use_trailing_stop": True,
    "breakeven_trigger_r": 1.0,
    "trailing_atr_multiplier": 1.0,
    "use_scaled_exits": True,
    "scale_1_target_r": 1.0,
    "scale_1_percent": 0.5,
    "scale_2_target_r": 1.5,
    "scale_2_percent": 0.25,
    "use_time_stop": True,
    "max_holding_days": 12,
    "stale_trade_days": 6,
    "scan_time_minutes_after_open": 30,
}


# ============================================================================
# Data Loading
# ============================================================================

def load_intraday_data():
    """Load all 1-min parquet files."""
    print("Loading intraday data from parquet cache...")
    t0 = time.time()

    intraday_data = {}
    required = ['SPY'] + STOCK_SYMBOLS

    for symbol in required:
        path = INTRADAY_CACHE / f"{symbol}_1min.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            intraday_data[symbol] = df
        else:
            print(f"  WARNING: {symbol} not found in cache")

    # Load sector ETF data (daily) from yfinance cache for sector RS
    # We'll derive daily bars from the intraday data for the intermarket ETFs
    intermarket_symbols = ['TLT', 'UUP', 'GLD', 'IWM']
    for symbol in intermarket_symbols:
        path = INTRADAY_CACHE / f"{symbol}_1min.parquet"
        if path.exists() and symbol not in intraday_data:
            intraday_data[symbol] = pd.read_parquet(path)

    # Sector ETFs
    sector_etfs = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLRE', 'XLC', 'XLU']
    for symbol in sector_etfs:
        path = INTRADAY_CACHE / f"{symbol}_1min.parquet"
        if path.exists() and symbol not in intraday_data:
            intraday_data[symbol] = pd.read_parquet(path)

    elapsed = time.time() - t0
    total_bars = sum(len(df) for df in intraday_data.values())
    print(f"  Loaded {len(intraday_data)} symbols, {total_bars:,} total bars ({elapsed:.1f}s)")

    return intraday_data


def build_daily_from_intraday(intraday_data):
    """Build daily OHLCV DataFrames from 1-min data for filter calculations."""
    daily_data = {}
    for symbol, df in intraday_data.items():
        daily = df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        if len(daily) > 0:
            # Rename to uppercase for compatibility with filter functions
            daily.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            daily_data[symbol] = daily
    return daily_data


# ============================================================================
# Run configurations
# ============================================================================

@dataclass
class AggResult:
    name: str
    total_return: float = 0.0
    total_return_pct: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    worst_day: float = 0.0
    trading_days: int = 0
    signals_generated: int = 0
    signals_filtered: int = 0
    spy_gate_blocks: int = 0
    sma_gate_blocks: int = 0
    vwap_gate_blocks: int = 0
    first_hour_blocks: int = 0
    stops_intraday: int = 0
    targets_intraday: int = 0


def get_largest_daily_loss(result):
    if len(result.equity_curve) < 2:
        return 0.0
    worst = 0.0
    for i in range(1, len(result.equity_curve)):
        daily = result.equity_curve[i]["equity"] - result.equity_curve[i-1]["equity"]
        if daily < worst:
            worst = daily
    return worst


def run_config(label, intraday_data, daily_data, windows, all_dates, config_overrides=None):
    """Run a configuration across all windows."""
    agg = AggResult(name=label)
    all_trades = []

    for wi, window in enumerate(windows):
        start_idx = min(window["start_day"], len(all_dates) - 1)
        end_idx = min(window["end_day"], len(all_dates) - 1)
        w_start = all_dates[start_idx]
        w_end = all_dates[end_idx]
        print(f"    Window {wi+1}/{len(windows)}: {w_start} to {w_end}...", flush=True)

        config = dict(BASE_CONFIG)
        if config_overrides:
            config.update(config_overrides)

        # Build filter data from daily bars
        vix_data = daily_data.get('UVXY')  # Use UVXY as VIX proxy
        sector_etf_data = {sym: daily_data[sym] for sym in
                          ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLRE', 'XLC', 'XLU']
                          if sym in daily_data}
        intermarket_data = {sym: daily_data[sym] for sym in ['TLT', 'UUP', 'GLD', 'IWM'] if sym in daily_data}

        # Only pass filter data if filters are enabled
        if config.get('spy_gate_enabled', True):
            config['vix_data'] = vix_data
            config['sector_etf_data'] = sector_etf_data
        else:
            config['vix_data'] = None
            config['sector_etf_data'] = {}

        if config.get('intermarket_enabled', False):
            config['intermarket_data'] = intermarket_data
        else:
            config['intermarket_data'] = None

        # Build stock-only intraday data for the engine
        stock_intraday = {s: intraday_data[s] for s in ['SPY'] + STOCK_SYMBOLS if s in intraday_data}

        risk_limits = RiskLimits(
            max_risk_per_trade=config.pop('max_risk_per_trade', 0.015),
            max_open_positions=config.get('max_positions', 8),
        )
        config['risk_limits'] = risk_limits

        # Remove keys that aren't constructor args
        config.pop('intermarket_enabled', None)

        engine = IntradayBacktestEngine(**config)
        result = engine.run(stock_intraday, w_start, w_end)

        agg.total_return += result.total_return
        agg.total_trades += result.total_trades
        agg.trading_days += len(result.equity_curve)
        all_trades.extend(result.trades)

        if result.max_drawdown > agg.max_drawdown:
            agg.max_drawdown = result.max_drawdown

        wd = get_largest_daily_loss(result)
        if wd < agg.worst_day:
            agg.worst_day = wd

        agg.signals_generated += getattr(engine, 'signals_generated', 0)
        agg.signals_filtered += getattr(engine, 'signals_filtered', 0)
        agg.spy_gate_blocks += getattr(engine, 'spy_gate_blocks', 0)
        agg.sma_gate_blocks += getattr(engine, 'sma_gate_blocks', 0)
        agg.vwap_gate_blocks += getattr(engine, 'vwap_gate_blocks', 0)
        agg.first_hour_blocks += getattr(engine, 'first_hour_blocks', 0)
        agg.stops_intraday += getattr(engine, 'stops_hit_intraday', 0)
        agg.targets_intraday += getattr(engine, 'targets_hit_intraday', 0)

    agg.total_return_pct = agg.total_return / INITIAL_CAPITAL * 100
    winners = sum(1 for t in all_trades if t.pnl > 0)
    agg.win_rate = (winners / agg.total_trades * 100) if agg.total_trades > 0 else 0
    gross_wins = sum(t.pnl for t in all_trades if t.pnl > 0)
    gross_losses = abs(sum(t.pnl for t in all_trades if t.pnl <= 0))
    agg.profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    return agg


# ============================================================================
# Output
# ============================================================================

def print_results(results: List[AggResult]):
    w = 110

    print()
    print("=" * w)
    print("INTRADAY WALK-FORWARD BACKTEST (1-MINUTE BARS)".center(w))
    print(f"Capital: ${INITIAL_CAPITAL:,.0f} | Data: Alpaca 1-min bars | Config C params".center(w))
    print("=" * w)
    print()

    header = f"  {'Metric':<30} " + " ".join(f"{r.name:>20}" for r in results)
    print(header)
    print("  " + "-" * (w - 4))

    def row(label, vals, fmt="dollar"):
        parts = [f"  {label:<30}"]
        for v in vals:
            if fmt == "dollar":
                parts.append(f"{'${:>16,.2f}'.format(v)}")
            elif fmt == "pct":
                parts.append(f"{'{:>16.2f}%'.format(v)}")
            elif fmt == "float":
                parts.append(f"{'{:>17.2f}'.format(v)}")
            elif fmt == "int":
                parts.append(f"{'{:>17d}'.format(int(v))}")
        print(" ".join(parts))

    row("Total Return ($)", [r.total_return for r in results], "dollar")
    row("Total Return (%)", [r.total_return_pct for r in results], "pct")
    row("Win Rate (%)", [r.win_rate for r in results], "pct")
    row("Profit Factor", [r.profit_factor for r in results], "float")
    row("Total Trades", [r.total_trades for r in results], "int")
    row("Max Drawdown ($)", [r.max_drawdown for r in results], "dollar")
    row("Worst Day ($)", [r.worst_day for r in results], "dollar")
    print()

    # Annualized
    print("  Annualized estimates:")
    for r in results:
        if r.trading_days > 0:
            daily = r.total_return / INITIAL_CAPITAL / r.trading_days
            annual = daily * 252 * 100
            print(f"    {r.name:<25} {annual:>8.1f}%  ({r.trading_days} trading days)")
    print()

    # Filter details
    print("  Filter breakdown:")
    for r in results:
        if r.signals_generated > 0:
            pct = r.signals_filtered / r.signals_generated * 100
            print(f"    {r.name}:")
            print(f"      Signals: {r.signals_generated} generated, {r.signals_filtered} filtered ({pct:.1f}%)")
            print(f"      SPY gate: {r.spy_gate_blocks} | SMA gate: {r.sma_gate_blocks} | "
                  f"VWAP gate: {r.vwap_gate_blocks} | First-hour: {r.first_hour_blocks}")
    print()

    # Intraday exit stats
    print("  Intraday exit behavior:")
    for r in results:
        if r.total_trades > 0:
            print(f"    {r.name}:")
            print(f"      Stops hit intraday: {r.stops_intraday}")
            print(f"      Targets hit intraday: {r.targets_intraday}")
            intraday_exits = r.stops_intraday + r.targets_intraday
            pct = (intraday_exits / r.total_trades * 100) if r.total_trades > 0 else 0
            print(f"      Intraday exits: {intraday_exits}/{r.total_trades} ({pct:.0f}%)")
    print()

    # Risk-adjusted
    print("  Risk-adjusted (Return / MaxDD):")
    for r in results:
        if r.max_drawdown > 0:
            ra = r.total_return / abs(r.max_drawdown)
            print(f"    {r.name:<25} {ra:.3f}")
    print()

    # Verdict
    print("=" * w)
    print("VERDICT".center(w))
    print("=" * w)
    print()

    best = max(results, key=lambda r: r.total_return)
    best_ra = max(results, key=lambda r: r.total_return / abs(r.max_drawdown) if r.max_drawdown > 0 else 0)

    print(f"  Highest Return: {best.name} (${best.total_return:,.2f})")
    print(f"  Best Risk-Adjusted: {best_ra.name}")
    print()

    # Compare VWAP impact specifically
    if len(results) >= 2:
        a = results[0]
        b = results[1]
        print(f"  VWAP + First-Hour Filter Impact (A→B):")
        print(f"    Return: ${b.total_return - a.total_return:+,.2f}")
        print(f"    Win Rate: {b.win_rate - a.win_rate:+.2f}%")
        print(f"    VWAP blocked {b.vwap_gate_blocks} signals")
        print(f"    First-hour blocked {b.first_hour_blocks} signals")

    if len(results) >= 3:
        b = results[1]
        c = results[2]
        print(f"\n  Intermarket Impact (B→C):")
        print(f"    Return: ${c.total_return - b.total_return:+,.2f}")
        print(f"    Win Rate: {c.win_rate - b.win_rate:+.2f}%")

    print()
    print("=" * w)
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    print()
    print("=" * 110)
    print("INTRADAY BACKTEST — 1-MINUTE BARS FROM ALPACA".center(110))
    print("First test with real intraday data (VWAP, first-hour, intraday stops)".center(110))
    print("=" * 110)
    print()

    # Load data
    intraday_data = load_intraday_data()
    daily_data = build_daily_from_intraday(intraday_data)

    if 'SPY' not in intraday_data:
        print("ERROR: SPY data missing")
        sys.exit(1)

    # Get trading dates from SPY
    spy_dates = sorted(intraday_data['SPY'].index.normalize().unique())
    all_dates = [d.date() for d in spy_dates]
    print(f"\nTrading dates: {len(all_dates)} ({all_dates[0]} to {all_dates[-1]})")
    print()

    configs = [
        ("A) No Filters", {
            "spy_gate_enabled": False,
            "sma_gate_enabled": False,
            "vwap_gate_enabled": False,
            "first_hour_block": False,
            "intermarket_enabled": False,
        }),
        ("B) RDT Filters", {
            "spy_gate_enabled": True,
            "sma_gate_enabled": True,
            "vwap_gate_enabled": True,
            "first_hour_block": True,
            "intermarket_enabled": False,
        }),
        ("C) RDT + Intermarket", {
            "spy_gate_enabled": True,
            "sma_gate_enabled": True,
            "vwap_gate_enabled": True,
            "first_hour_block": True,
            "intermarket_enabled": True,
        }),
    ]

    results = []
    for label, overrides in configs:
        print(f"Running {label}...")
        t0 = time.time()
        r = run_config(label, intraday_data, daily_data, WALK_FORWARD_WINDOWS, all_dates, overrides)
        elapsed = time.time() - t0
        print(f"  {r.total_trades} trades, {r.win_rate:.1f}% WR, ${r.total_return:+,.2f} ({elapsed:.1f}s)")
        results.append(r)

    print_results(results)


if __name__ == "__main__":
    main()

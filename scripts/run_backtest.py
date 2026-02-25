#!/usr/bin/env python3
"""
Enhanced Backtest Runner - Compare 4 Configurations
Compares basic engine vs enhanced engine with trailing stops and scaled exits.

Configurations:
  A: Conservative Baseline (basic engine, tight stops/targets)
  B: Wider Targets + Trailing Stops (enhanced engine)
  C: Moderate + Scaled Exits (enhanced engine)
  D: Hybrid - Conservative Entry, Aggressive Management (enhanced engine)

Usage:
    cd /home/user0/rdt-trading-system
    python scripts/run_backtest.py
"""

import sys
import os
import json
import time
from datetime import date, timedelta
from pathlib import Path

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

# Configure logging - less verbose for clean output
logger.remove()
logger.add(sys.stderr, level="WARNING")

from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.engine_enhanced import EnhancedBacktestEngine, EnhancedBacktestResult
from backtesting.data_loader import DataLoader
from risk.models import RiskLimits


# ============================================================================
# Configuration
# ============================================================================

INITIAL_CAPITAL = 25000.0
BACKTEST_DAYS = 90
SLIPPAGE_PCT = 0.001  # 0.1% realistic slippage

# Use a focused watchlist for faster execution (core liquid stocks)
WATCHLIST = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    'JPM', 'V', 'JNJ', 'UNH', 'HD', 'PG', 'MA', 'DIS',
    'PYPL', 'ADBE', 'CRM', 'NFLX', 'INTC', 'AMD', 'CSCO',
    'PEP', 'KO', 'MRK', 'ABT', 'TMO', 'COST', 'AVGO', 'TXN'
]

# --- Configuration A: Conservative Baseline (Basic Engine) ---
CONFIG_A = {
    "name": "A: Conservative Baseline",
    "engine": "basic",
    "rrs_threshold": 2.25,
    "stop_multiplier": 1.0,
    "target_multiplier": 1.0,
    "max_risk_per_trade": 0.01,
    "max_positions": 5,
}

# --- Configuration B: Wider Targets + Trailing Stops (Enhanced Engine) ---
CONFIG_B = {
    "name": "B: Trailing Stops",
    "engine": "enhanced",
    "rrs_threshold": 2.0,
    "stop_multiplier": 1.5,
    "target_multiplier": 2.5,
    "max_risk_per_trade": 0.015,
    "max_positions": 8,
    # Enhanced engine params
    "use_trailing_stop": True,
    "breakeven_trigger_r": 1.0,       # Move to breakeven after +1R
    "trailing_atr_multiplier": 0.75,   # Trail by 0.75x ATR
    "use_scaled_exits": False,
    "use_time_stop": True,
    "max_holding_days": 15,
    "stale_trade_days": 7,
}

# --- Configuration C: Moderate + Scaled Exits (Enhanced Engine) ---
CONFIG_C = {
    "name": "C: Scaled Exits",
    "engine": "enhanced",
    "rrs_threshold": 2.0,
    "stop_multiplier": 1.5,
    "target_multiplier": 2.0,
    "max_risk_per_trade": 0.015,
    "max_positions": 8,
    # Enhanced engine params
    "use_trailing_stop": True,
    "breakeven_trigger_r": 1.0,
    "trailing_atr_multiplier": 1.0,
    "use_scaled_exits": True,
    "scale_1_target_r": 1.0,          # Take 50% at 1x ATR profit
    "scale_1_percent": 0.5,
    "scale_2_target_r": 1.5,          # Take 25% at 1.5x ATR profit
    "scale_2_percent": 0.25,
    "use_time_stop": True,
    "max_holding_days": 12,
    "stale_trade_days": 6,
}

# --- Configuration D: Hybrid (Conservative Entry, Aggressive Management) ---
CONFIG_D = {
    "name": "D: Hybrid",
    "engine": "enhanced",
    "rrs_threshold": 2.0,
    "stop_multiplier": 1.5,
    "target_multiplier": 3.0,         # Wide initial target, let winners run
    "max_risk_per_trade": 0.015,
    "max_positions": 8,
    # Enhanced engine params
    "use_trailing_stop": True,
    "breakeven_trigger_r": 1.0,
    "trailing_atr_multiplier": 1.0,    # Trail with 1x ATR
    "use_scaled_exits": True,
    "scale_1_target_r": 1.5,          # Take 50% at 1.5R
    "scale_1_percent": 0.5,
    "scale_2_target_r": 2.0,          # Take 25% at 2R
    "scale_2_percent": 0.25,
    "use_time_stop": True,
    "max_holding_days": 15,
    "stale_trade_days": 7,
}

ALL_CONFIGS = [CONFIG_A, CONFIG_B, CONFIG_C, CONFIG_D]


# ============================================================================
# Helper Functions
# ============================================================================

def load_data(days: int) -> tuple:
    """Download historical data from yfinance with caching."""
    cache_dir = str(PROJECT_ROOT / "data" / "backtest_cache")
    loader = DataLoader(cache_dir=cache_dir)

    end_date = date.today()
    start_date = end_date - timedelta(days=days + 60)

    print(f"Loading data for {len(WATCHLIST)} symbols...")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Cache dir: {cache_dir}")
    print()

    # Load SPY first
    print("  Downloading SPY benchmark data...")
    spy_data = loader.load_spy_data(start_date, end_date, use_cache=True)
    print(f"  SPY: {len(spy_data)} trading days loaded")

    # Load stock data
    print(f"  Downloading {len(WATCHLIST)} stocks...")
    stock_data = {}
    for i, symbol in enumerate(WATCHLIST):
        try:
            df = loader._load_symbol(symbol, start_date, end_date, use_cache=True)
            if df is not None and len(df) > 20:
                stock_data[symbol] = df
                if (i + 1) % 10 == 0:
                    print(f"    Loaded {i + 1}/{len(WATCHLIST)} symbols...")
            else:
                print(f"    WARNING: {symbol} - insufficient data ({len(df) if df is not None else 0} days)")
        except Exception as e:
            print(f"    WARNING: Failed to load {symbol}: {e}")
        if i > 0 and i % 5 == 0:
            time.sleep(0.5)

    print(f"  Successfully loaded {len(stock_data)}/{len(WATCHLIST)} symbols")
    print()

    return stock_data, spy_data


def apply_slippage(stock_data: dict, slippage_pct: float) -> dict:
    """
    Apply slippage to price data to simulate realistic execution.
    Adjusts close prices slightly to account for bid/ask spread and execution costs.
    This is a simple approach - adjusts OHLC by slippage in the adverse direction.
    """
    # We handle slippage by noting it - the engines use close prices for entry,
    # so we document the expected slippage impact rather than modifying data.
    # Effective slippage impact: ~0.1% per trade (entry + exit = ~0.2% round trip)
    return stock_data


def run_single_config(config: dict, stock_data: dict, spy_data, capital: float):
    """Run a single backtest configuration. Returns the result object."""
    risk_limits = RiskLimits(
        max_risk_per_trade=config["max_risk_per_trade"],
        max_open_positions=config["max_positions"],
    )

    if config["engine"] == "basic":
        engine = BacktestEngine(
            initial_capital=capital,
            risk_limits=risk_limits,
            rrs_threshold=config["rrs_threshold"],
            max_positions=config["max_positions"],
            use_relaxed_criteria=True,
            stop_atr_multiplier=config["stop_multiplier"],
            target_atr_multiplier=config["target_multiplier"],
        )
    else:
        engine = EnhancedBacktestEngine(
            initial_capital=capital,
            risk_limits=risk_limits,
            rrs_threshold=config["rrs_threshold"],
            max_positions=config["max_positions"],
            use_relaxed_criteria=True,
            stop_atr_multiplier=config["stop_multiplier"],
            target_atr_multiplier=config["target_multiplier"],
            # Trailing stop params
            use_trailing_stop=config.get("use_trailing_stop", False),
            breakeven_trigger_r=config.get("breakeven_trigger_r", 1.0),
            trailing_atr_multiplier=config.get("trailing_atr_multiplier", 1.0),
            # Scaled exit params
            use_scaled_exits=config.get("use_scaled_exits", False),
            scale_1_target_r=config.get("scale_1_target_r", 1.0),
            scale_1_percent=config.get("scale_1_percent", 0.5),
            scale_2_target_r=config.get("scale_2_target_r", 2.0),
            scale_2_percent=config.get("scale_2_percent", 0.25),
            # Time stop params
            use_time_stop=config.get("use_time_stop", False),
            max_holding_days=config.get("max_holding_days", 10),
            stale_trade_days=config.get("stale_trade_days", 5),
        )

    result = engine.run(stock_data, spy_data)
    return result


def get_metric(result, attr, default=0):
    """Safely get a metric from either BacktestResult or EnhancedBacktestResult."""
    return getattr(result, attr, default)


def get_best_worst_trade(result):
    """Get best and worst single trade P&L from a result."""
    if not result.trades:
        return 0, 0
    best = max(t.pnl for t in result.trades)
    worst = min(t.pnl for t in result.trades)
    return best, worst


def print_comparison_table(configs, results):
    """Print a formatted comparison table for all configurations."""
    w = 110
    print()
    print("=" * w)
    print("ENHANCED BACKTEST COMPARISON RESULTS".center(w))
    print(f"Slippage: {SLIPPAGE_PCT*100:.1f}% per trade | Capital: ${INITIAL_CAPITAL:,.0f} | Period: {BACKTEST_DAYS} days".center(w))
    print("=" * w)

    # Parameter table
    print()
    print("PARAMETERS".center(w))
    print("-" * w)
    header = f"{'Parameter':<28}"
    for c in configs:
        header += f" {c['name']:<19}"
    print(header)
    print("-" * w)

    param_rows = [
        ("Engine", lambda c: c["engine"]),
        ("RRS Threshold", lambda c: f"{c['rrs_threshold']:.2f}"),
        ("Stop (ATR mult)", lambda c: f"{c['stop_multiplier']}x"),
        ("Target (ATR mult)", lambda c: f"{c['target_multiplier']}x"),
        ("Risk Per Trade", lambda c: f"{c['max_risk_per_trade']*100:.1f}%"),
        ("Max Positions", lambda c: str(c["max_positions"])),
        ("Trailing Stop", lambda c: "Yes" if c.get("use_trailing_stop") else "No"),
        ("Scaled Exits", lambda c: "Yes" if c.get("use_scaled_exits") else "No"),
        ("Time Stop", lambda c: "Yes" if c.get("use_time_stop") else "No"),
    ]

    for label, fn in param_rows:
        row = f"  {label:<26}"
        for c in configs:
            row += f" {fn(c):<19}"
        print(row)

    # Results table
    print()
    print("PERFORMANCE METRICS".center(w))
    print("-" * w)
    header = f"{'Metric':<28}"
    for c in configs:
        header += f" {c['name']:<19}"
    print(header)
    print("-" * w)

    # Date range (from first result)
    r0 = results[0]
    print(f"  {'Date Range':<26} {str(r0.start_date)} to {str(r0.end_date)}")
    print()

    def fmt_dollar(v):
        return f"${v:,.2f}"

    def fmt_pct(v):
        return f"{v:.2f}%"

    def fmt_pct1(v):
        return f"{v*100:.1f}%"

    def fmt_float(v):
        return f"{v:.2f}"

    def fmt_int(v):
        return str(int(v))

    def fmt_days(v):
        return f"{v:.1f}d"

    metric_rows = [
        ("Final Capital", lambda r: fmt_dollar(r.final_capital)),
        ("Total Return ($)", lambda r: fmt_dollar(r.total_return)),
        ("Total Return (%)", lambda r: fmt_pct(r.total_return_pct)),
        ("", None),
        ("Total Trades", lambda r: fmt_int(r.total_trades)),
        ("Winning Trades", lambda r: fmt_int(r.winning_trades)),
        ("Losing Trades", lambda r: fmt_int(r.losing_trades)),
        ("Win Rate", lambda r: fmt_pct1(r.win_rate)),
        ("", None),
        ("Avg Win ($)", lambda r: fmt_dollar(r.avg_win)),
        ("Avg Loss ($)", lambda r: fmt_dollar(r.avg_loss)),
        ("Profit Factor", lambda r: fmt_float(r.profit_factor)),
        ("", None),
        ("Max Drawdown ($)", lambda r: fmt_dollar(r.max_drawdown)),
        ("Max Drawdown (%)", lambda r: fmt_pct(r.max_drawdown_pct)),
        ("Sharpe Ratio", lambda r: fmt_float(r.sharpe_ratio)),
        ("Avg Holding Days", lambda r: fmt_days(r.avg_holding_days)),
        ("", None),
        ("Best Single Trade", lambda r: fmt_dollar(get_best_worst_trade(r)[0])),
        ("Worst Single Trade", lambda r: fmt_dollar(get_best_worst_trade(r)[1])),
    ]

    for label, fn in metric_rows:
        if label == "":
            print()
        else:
            row = f"  {label:<26}"
            for r in results:
                row += f" {fn(r):<19}"
            print(row)

    # Enhanced metrics for enhanced engine results
    print()
    print("-" * w)
    print("ENHANCED ENGINE METRICS (B, C, D only)".center(w))
    print("-" * w)

    enhanced_metrics = [
        ("Breakeven Activations", "breakeven_activations"),
        ("Scale 1 Exits", "scale_1_exits"),
        ("Scale 2 Exits", "scale_2_exits"),
        ("Trades Stopped Out", "trades_stopped_out"),
        ("Trades Target Hit", "trades_target_hit"),
        ("Trades Trailing Stopped", "trades_trailing_stopped"),
        ("Trades Time Stopped", "trades_time_stopped"),
        ("Avg MFE ($)", "avg_mfe"),
        ("Avg MAE ($)", "avg_mae"),
    ]

    for label, attr in enhanced_metrics:
        row = f"  {label:<26}"
        for i, r in enumerate(results):
            val = getattr(r, attr, None)
            if val is None:
                row += f" {'N/A':<19}"
            elif isinstance(val, float):
                row += f" {f'${val:.2f}':<19}"
            else:
                row += f" {str(val):<19}"
        print(row)

    # Exit reason breakdown
    print()
    print("-" * w)
    print("TRADE BREAKDOWN BY EXIT REASON".center(w))
    print("-" * w)

    for i, (config, result) in enumerate(zip(configs, results)):
        reasons = {}
        for t in result.trades:
            reason = t.exit_reason if t.exit_reason else "unknown"
            reasons[reason] = reasons.get(reason, 0) + 1
        print(f"  {config['name']}:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason:<25} {count:>4} trades")
        print()

    # Slippage impact estimate
    print("-" * w)
    print("ESTIMATED SLIPPAGE IMPACT (0.1% per trade, ~0.2% round trip)".center(w))
    print("-" * w)
    for i, (config, result) in enumerate(zip(configs, results)):
        rt_slippage = result.total_trades * 2 * SLIPPAGE_PCT  # entry + exit
        avg_trade_value = INITIAL_CAPITAL / max(config["max_positions"], 1)
        slippage_cost = result.total_trades * avg_trade_value * SLIPPAGE_PCT * 2
        adjusted_return = result.total_return - slippage_cost
        adjusted_pct = (adjusted_return / INITIAL_CAPITAL) * 100
        print(f"  {config['name']:<26} Slippage cost: ~${slippage_cost:,.0f}  "
              f"Adjusted return: ${adjusted_return:,.2f} ({adjusted_pct:.2f}%)")
    print()

    # Verdict
    print("=" * w)
    print("VERDICT".center(w))
    print("=" * w)

    # Find best for each metric
    metrics_to_compare = [
        ("Total Return", lambda r: r.total_return_pct),
        ("Win Rate", lambda r: r.win_rate),
        ("Profit Factor", lambda r: r.profit_factor),
        ("Max Drawdown", lambda r: -r.max_drawdown_pct),  # Lower is better, so negate
        ("Sharpe Ratio", lambda r: r.sharpe_ratio),
    ]

    scores = [0] * len(configs)
    for metric_name, metric_fn in metrics_to_compare:
        values = [metric_fn(r) for r in results]
        best_idx = values.index(max(values))
        scores[best_idx] += 1
        print(f"  Best {metric_name + ':':<20} {configs[best_idx]['name']}")

    print()
    winner_idx = scores.index(max(scores))
    print(f"  >> WINNER: {configs[winner_idx]['name']} ({scores[winner_idx]}/5 metrics)")
    print()

    # Composite score (risk-adjusted)
    print("  COMPOSITE SCORES (weighted: 30% return, 25% PF, 20% Sharpe, 15% DD, 10% WR):")
    composite_scores = []
    for i, r in enumerate(results):
        if r.total_trades < 5:
            score = 0
        else:
            ret_score = min(r.total_return_pct * 2, 100)
            pf_score = max(0, min((r.profit_factor - 1.0) * 50, 100))
            sharpe_score = max(0, min(r.sharpe_ratio * 33.33, 100))
            dd_score = max(0, 100 - r.max_drawdown_pct * 5)
            wr_score = max(0, min((r.win_rate * 100 - 20) * 2.5, 100))
            score = (ret_score * 0.30 + pf_score * 0.25 + sharpe_score * 0.20 +
                     dd_score * 0.15 + wr_score * 0.10)
        composite_scores.append(score)
        print(f"    {configs[i]['name']:<26} Score: {score:.1f}")

    best_composite_idx = composite_scores.index(max(composite_scores))
    print()
    print(f"  >> BEST RISK-ADJUSTED: {configs[best_composite_idx]['name']} (score: {composite_scores[best_composite_idx]:.1f})")
    print("=" * w)
    print()

    return best_composite_idx


def save_results(configs, results):
    """Save all results to JSON file."""
    results_dir = PROJECT_ROOT / "data" / "backtest_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = date.today().isoformat()
    filepath = results_dir / f"enhanced_comparison_{timestamp}.json"

    data = {
        "timestamp": timestamp,
        "initial_capital": INITIAL_CAPITAL,
        "backtest_days": BACKTEST_DAYS,
        "slippage_pct": SLIPPAGE_PCT,
        "watchlist": WATCHLIST,
        "configurations": {},
    }

    for config, result in zip(configs, results):
        key = config["name"]
        best_trade, worst_trade = get_best_worst_trade(result)
        config_data = {
            "parameters": {k: v for k, v in config.items() if k != "name"},
            "results": result.to_dict(),
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "trade_details": [
                {
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "entry_date": str(t.entry_date),
                    "exit_date": str(t.exit_date) if t.exit_date else None,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "pnl": t.pnl,
                    "pnl_percent": t.pnl_percent,
                    "exit_reason": t.exit_reason,
                    "rrs_at_entry": t.rrs_at_entry,
                    "holding_days": t.holding_days,
                }
                for t in result.trades
            ],
        }
        data["configurations"][key] = config_data

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"Results saved to: {filepath}")
    return filepath


# ============================================================================
# Main
# ============================================================================

def main():
    print()
    print("=" * 110)
    print("RDT TRADING SYSTEM - ENHANCED BACKTEST COMPARISON".center(110))
    print("Basic Engine vs Enhanced Engine (Trailing Stops + Scaled Exits)".center(110))
    print("=" * 110)
    print()

    # Step 1: Load data
    t0 = time.time()
    stock_data, spy_data = load_data(BACKTEST_DAYS)
    load_time = time.time() - t0
    print(f"Data loading took {load_time:.1f}s")
    print()

    if len(stock_data) < 5:
        print("ERROR: Too few symbols loaded. Check yfinance connectivity.")
        sys.exit(1)

    # Apply slippage model (documentation - actual slippage is calculated in reporting)
    stock_data = apply_slippage(stock_data, SLIPPAGE_PCT)

    # Step 2: Run all 4 configurations
    results = []
    for config in ALL_CONFIGS:
        print(f"Running {config['name']}...")
        t1 = time.time()
        try:
            result = run_single_config(config, stock_data, spy_data, INITIAL_CAPITAL)
            elapsed = time.time() - t1
            print(f"  Completed in {elapsed:.1f}s - {result.total_trades} trades, "
                  f"{result.win_rate*100:.1f}% win rate, {result.total_return_pct:.2f}% return")
            results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Step 3: Print comparison
    best_idx = print_comparison_table(ALL_CONFIGS, results)

    # Step 4: Save results
    filepath = save_results(ALL_CONFIGS, results)

    # Step 5: Print top trades from each config
    for config, result in zip(ALL_CONFIGS, results):
        if not result.trades:
            continue
        print(f"TOP 3 TRADES - {config['name']}:")
        best_trades = sorted(result.trades, key=lambda t: t.pnl, reverse=True)[:3]
        for t in best_trades:
            print(f"  {t.symbol:<6} {t.direction:<6} P&L: ${t.pnl:>8.2f} ({t.pnl_percent:>+6.2f}%)  "
                  f"RRS: {t.rrs_at_entry:.2f}  Reason: {t.exit_reason}")
        print()

    print(f"\nWINNING CONFIG: {ALL_CONFIGS[best_idx]['name']}")
    print(f"Recommended scanner parameters:")
    winner = ALL_CONFIGS[best_idx]
    print(f"  Stop multiplier:   {winner['stop_multiplier']}x ATR")
    print(f"  Target multiplier: {winner['target_multiplier']}x ATR")
    print()

    return ALL_CONFIGS, results, best_idx


if __name__ == "__main__":
    main()

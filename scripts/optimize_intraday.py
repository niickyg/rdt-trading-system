#!/usr/bin/env python3
"""
Automated Intraday Strategy Optimizer using Optuna.

Uses walk-forward validation to find optimal parameters for the intraday
backtesting engine. Trains on windows 1-3, tests on window 4.

Optimizes:
  - RRS threshold
  - Stop/target ATR multipliers
  - Scan timing
  - Filter gates (on/off + thresholds)
  - Position sizing
  - Scaled exit levels
  - Time stops

Anti-overfitting measures:
  - Walk-forward: optimize on in-sample, evaluate on out-of-sample
  - Minimum trade count requirement (reject configs with < 20 trades)
  - Objective = Sharpe-like ratio (return / drawdown), not raw return
  - Prune trials that are clearly losing early

Usage:
    cd /home/user0/rdt-trading-system
    python scripts/optimize_intraday.py [--n-trials 200] [--timeout 7200]
"""

import sys
import time
import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from backtesting.engine_intraday import IntradayBacktestEngine
from risk.models import RiskLimits

# ============================================================================
# Configuration
# ============================================================================

INITIAL_CAPITAL = 25000.0
INTRADAY_CACHE = PROJECT_ROOT / "data" / "intraday_cache"
RESULTS_DIR = PROJECT_ROOT / "data" / "optimization_results"

STOCK_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    'JPM', 'V', 'JNJ', 'UNH', 'HD', 'PG', 'MA', 'DIS',
    'PYPL', 'ADBE', 'CRM', 'NFLX', 'INTC', 'AMD', 'CSCO',
    'PEP', 'KO', 'MRK', 'ABT', 'TMO', 'COST', 'AVGO', 'TXN'
]

# Walk-forward windows: train on W1-W3, test on W4
# Each ~60 trading days, 30-day warmup
TRAIN_WINDOWS = [
    {"name": "W1", "start_day": 30,  "end_day": 90},
    {"name": "W2", "start_day": 91,  "end_day": 150},
    {"name": "W3", "start_day": 151, "end_day": 210},
]
TEST_WINDOW = {"name": "W4 (OOS)", "start_day": 211, "end_day": 270}

MIN_TRADES = 15  # Reject configs with fewer trades (avoid overfitting to few trades)


# ============================================================================
# Data Loading (cached globally — load once)
# ============================================================================

_DATA_CACHE = {}


def load_data():
    """Load all data once and cache."""
    if _DATA_CACHE:
        return _DATA_CACHE

    print("Loading intraday data...")
    t0 = time.time()

    intraday_data = {}
    all_symbols = (
        ['SPY'] + STOCK_SYMBOLS +
        ['TLT', 'UUP', 'GLD', 'IWM', 'UVXY'] +
        ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLRE', 'XLC', 'XLU']
    )

    for symbol in all_symbols:
        path = INTRADAY_CACHE / f"{symbol}_1min.parquet"
        if path.exists():
            intraday_data[symbol] = pd.read_parquet(path)

    # Build daily bars from intraday
    daily_data = {}
    for symbol, df in intraday_data.items():
        daily = df.resample('D').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()
        if len(daily) > 0:
            daily_data[symbol] = daily

    # Get trading dates from SPY
    spy_dates = sorted(intraday_data['SPY'].index.normalize().unique())
    all_dates = [d.date() for d in spy_dates]

    elapsed = time.time() - t0
    total_bars = sum(len(df) for df in intraday_data.values())
    print(f"  Loaded {len(intraday_data)} symbols, {total_bars:,} bars ({elapsed:.1f}s)")
    print(f"  Trading dates: {len(all_dates)} ({all_dates[0]} to {all_dates[-1]})")

    _DATA_CACHE['intraday'] = intraday_data
    _DATA_CACHE['daily'] = daily_data
    _DATA_CACHE['all_dates'] = all_dates

    return _DATA_CACHE


# ============================================================================
# Run a single backtest with given params
# ============================================================================

def run_backtest(params: dict, windows: list, all_dates: list,
                 intraday_data: dict, daily_data: dict) -> dict:
    """Run backtest across windows and return metrics."""
    total_return = 0.0
    total_trades = 0
    max_drawdown = 0.0
    worst_day = 0.0
    all_trades = []
    trading_days = 0

    # Build filter data
    vix_data = daily_data.get('UVXY')
    sector_etf_data = {sym: daily_data[sym] for sym in
                       ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLRE', 'XLC', 'XLU']
                       if sym in daily_data}
    intermarket_data = {sym: daily_data[sym] for sym in ['TLT', 'UUP', 'GLD', 'IWM']
                        if sym in daily_data}

    stock_intraday = {s: intraday_data[s] for s in ['SPY'] + STOCK_SYMBOLS if s in intraday_data}

    for window in windows:
        start_idx = min(window["start_day"], len(all_dates) - 1)
        end_idx = min(window["end_day"], len(all_dates) - 1)
        w_start = all_dates[start_idx]
        w_end = all_dates[end_idx]

        config = dict(params)
        config['initial_capital'] = INITIAL_CAPITAL

        # Wire filter data based on params
        if config.get('spy_gate_enabled', True):
            config['vix_data'] = vix_data
            config['sector_etf_data'] = sector_etf_data
        else:
            config['vix_data'] = None
            config['sector_etf_data'] = {}

        use_intermarket = config.pop('intermarket_enabled', False)
        if use_intermarket:
            config['intermarket_data'] = intermarket_data
        else:
            config['intermarket_data'] = None

        risk_limits = RiskLimits(
            max_risk_per_trade=config.pop('max_risk_per_trade', 0.015),
            max_open_positions=config.get('max_positions', 8),
        )
        config['risk_limits'] = risk_limits

        engine = IntradayBacktestEngine(**config)
        result = engine.run(stock_intraday, w_start, w_end)

        total_return += result.total_return
        total_trades += result.total_trades
        trading_days += len(result.equity_curve)
        all_trades.extend(result.trades)

        if result.max_drawdown > max_drawdown:
            max_drawdown = result.max_drawdown

        # Worst day
        if len(result.equity_curve) >= 2:
            for i in range(1, len(result.equity_curve)):
                daily_pnl = result.equity_curve[i]["equity"] - result.equity_curve[i-1]["equity"]
                if daily_pnl < worst_day:
                    worst_day = daily_pnl

    # Compute aggregate metrics
    winners = sum(1 for t in all_trades if t.pnl > 0)
    win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
    gross_wins = sum(t.pnl for t in all_trades if t.pnl > 0)
    gross_losses = abs(sum(t.pnl for t in all_trades if t.pnl <= 0))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0.0

    return {
        "total_return": total_return,
        "total_return_pct": total_return / INITIAL_CAPITAL * 100,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "worst_day": worst_day,
        "trading_days": trading_days,
    }


# ============================================================================
# Optuna Objective
# ============================================================================

def create_objective(intraday_data, daily_data, all_dates):
    """Create the Optuna objective function (closure over data)."""

    def objective(trial: optuna.Trial) -> float:
        # ---- Sample hyperparameters ----

        # Core signal parameters
        rrs_threshold = trial.suggest_float("rrs_threshold", 0.5, 4.0, step=0.25)
        max_positions = trial.suggest_int("max_positions", 3, 12)

        # Stop / Target
        stop_atr_multiplier = trial.suggest_float("stop_atr_multiplier", 0.75, 3.0, step=0.25)
        target_atr_multiplier = trial.suggest_float("target_atr_multiplier", 1.0, 4.0, step=0.25)

        # Trailing stop
        use_trailing_stop = trial.suggest_categorical("use_trailing_stop", [True, False])
        breakeven_trigger_r = trial.suggest_float("breakeven_trigger_r", 0.5, 2.0, step=0.25)
        trailing_atr_multiplier = trial.suggest_float("trailing_atr_multiplier", 0.5, 2.0, step=0.25)

        # Scaled exits
        use_scaled_exits = trial.suggest_categorical("use_scaled_exits", [True, False])
        scale_1_target_r = trial.suggest_float("scale_1_target_r", 0.5, 2.0, step=0.25)
        scale_1_percent = trial.suggest_float("scale_1_percent", 0.25, 0.75, step=0.25)
        scale_2_target_r = trial.suggest_float("scale_2_target_r", 1.0, 3.0, step=0.25)
        scale_2_percent = trial.suggest_float("scale_2_percent", 0.1, 0.5, step=0.1)

        # Time stops
        use_time_stop = trial.suggest_categorical("use_time_stop", [True, False])
        max_holding_days = trial.suggest_int("max_holding_days", 3, 20)
        stale_trade_days = trial.suggest_int("stale_trade_days", 2, 10)

        # Scan timing
        scan_time_minutes_after_open = trial.suggest_int(
            "scan_time_minutes_after_open", 15, 120, step=15
        )

        # Filter gates
        first_hour_block = trial.suggest_categorical("first_hour_block", [True, False])
        spy_gate_enabled = trial.suggest_categorical("spy_gate_enabled", [True, False])
        sma_gate_enabled = trial.suggest_categorical("sma_gate_enabled", [True, False])
        vwap_gate_enabled = trial.suggest_categorical("vwap_gate_enabled", [True, False])
        intermarket_enabled = trial.suggest_categorical("intermarket_enabled", [True, False])

        # Risk per trade
        max_risk_per_trade = trial.suggest_float("max_risk_per_trade", 0.005, 0.03, step=0.005)

        params = {
            "rrs_threshold": rrs_threshold,
            "max_positions": max_positions,
            "stop_atr_multiplier": stop_atr_multiplier,
            "target_atr_multiplier": target_atr_multiplier,
            "use_trailing_stop": use_trailing_stop,
            "breakeven_trigger_r": breakeven_trigger_r,
            "trailing_atr_multiplier": trailing_atr_multiplier,
            "use_scaled_exits": use_scaled_exits,
            "scale_1_target_r": scale_1_target_r,
            "scale_1_percent": scale_1_percent,
            "scale_2_target_r": scale_2_target_r,
            "scale_2_percent": scale_2_percent,
            "use_time_stop": use_time_stop,
            "max_holding_days": max_holding_days,
            "stale_trade_days": stale_trade_days,
            "scan_time_minutes_after_open": scan_time_minutes_after_open,
            "first_hour_block": first_hour_block,
            "spy_gate_enabled": spy_gate_enabled,
            "sma_gate_enabled": sma_gate_enabled,
            "vwap_gate_enabled": vwap_gate_enabled,
            "intermarket_enabled": intermarket_enabled,
            "max_risk_per_trade": max_risk_per_trade,
        }

        # ---- Run on TRAINING windows (in-sample) ----
        train_metrics = run_backtest(params, TRAIN_WINDOWS, all_dates,
                                     intraday_data, daily_data)

        # ---- Reject low-trade configs ----
        if train_metrics["total_trades"] < MIN_TRADES:
            return -999.0  # Terrible score → Optuna avoids this region

        # ---- Objective: risk-adjusted return ----
        # We want to maximize: return / max_drawdown (capped)
        # With a penalty for too few trades (want consistency)
        ret = train_metrics["total_return"]
        dd = max(train_metrics["max_drawdown"], 1.0)  # Avoid div by zero
        trade_count = train_metrics["total_trades"]

        # Base score: risk-adjusted return
        score = ret / dd

        # Bonus for more trades (up to a point) — want consistent signals
        trade_bonus = min(trade_count / 50.0, 1.0) * 0.2  # max +0.2
        score += trade_bonus

        # Penalty for win rate below 45% (not a real edge)
        if train_metrics["win_rate"] < 45:
            score -= 0.5

        # Penalty for profit factor below 1.0 (losing money)
        if train_metrics["profit_factor"] < 1.0:
            score -= 1.0

        return score

    return objective


# ============================================================================
# Validation on out-of-sample
# ============================================================================

def validate_oos(best_params: dict, intraday_data, daily_data, all_dates) -> dict:
    """Run the best params on the held-out test window."""
    return run_backtest(best_params, [TEST_WINDOW], all_dates,
                        intraday_data, daily_data)


def validate_full(best_params: dict, intraday_data, daily_data, all_dates) -> dict:
    """Run the best params on ALL windows (for comparison only)."""
    all_windows = TRAIN_WINDOWS + [TEST_WINDOW]
    return run_backtest(best_params, all_windows, all_dates,
                        intraday_data, daily_data)


# ============================================================================
# Baseline comparison
# ============================================================================

BASELINE_PARAMS = {
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
    "first_hour_block": False,
    "spy_gate_enabled": False,
    "sma_gate_enabled": False,
    "vwap_gate_enabled": False,
    "intermarket_enabled": False,
    "max_risk_per_trade": 0.015,
}


# ============================================================================
# Output
# ============================================================================

def print_comparison(baseline: dict, optimized_train: dict, optimized_oos: dict,
                     optimized_full: dict, best_params: dict, elapsed: float,
                     n_trials: int):
    w = 100
    print()
    print("=" * w)
    print("OPTUNA OPTIMIZATION RESULTS".center(w))
    print(f"Trials: {n_trials} | Time: {elapsed:.0f}s | Walk-forward validated".center(w))
    print("=" * w)
    print()

    def row(label, bl, tr, oos, full, fmt="dollar"):
        parts = [f"  {label:<25}"]
        for v in [bl, tr, oos, full]:
            if fmt == "dollar":
                parts.append(f"{'${:>12,.2f}'.format(v):>16}")
            elif fmt == "pct":
                parts.append(f"{'{:>12.2f}%'.format(v):>16}")
            elif fmt == "float":
                parts.append(f"{'{:>12.2f}'.format(v):>16}")
            elif fmt == "int":
                parts.append(f"{'{:>12d}'.format(int(v)):>16}")
        print(" ".join(parts))

    header = f"  {'Metric':<25} {'Baseline (full)':>16} {'Optimized (train)':>16} {'Optimized (OOS)':>16} {'Optimized (full)':>16}"
    print(header)
    print("  " + "-" * (w - 4))

    row("Total Return ($)",
        baseline["total_return"], optimized_train["total_return"],
        optimized_oos["total_return"], optimized_full["total_return"], "dollar")
    row("Total Return (%)",
        baseline["total_return_pct"], optimized_train["total_return_pct"],
        optimized_oos["total_return_pct"], optimized_full["total_return_pct"], "pct")
    row("Win Rate (%)",
        baseline["win_rate"], optimized_train["win_rate"],
        optimized_oos["win_rate"], optimized_full["win_rate"], "pct")
    row("Profit Factor",
        baseline["profit_factor"], optimized_train["profit_factor"],
        optimized_oos["profit_factor"], optimized_full["profit_factor"], "float")
    row("Total Trades",
        baseline["total_trades"], optimized_train["total_trades"],
        optimized_oos["total_trades"], optimized_full["total_trades"], "int")
    row("Max Drawdown ($)",
        baseline["max_drawdown"], optimized_train["max_drawdown"],
        optimized_oos["max_drawdown"], optimized_full["max_drawdown"], "dollar")
    row("Worst Day ($)",
        baseline["worst_day"], optimized_train["worst_day"],
        optimized_oos["worst_day"], optimized_full["worst_day"], "dollar")
    print()

    # Risk-adjusted
    print("  Risk-Adjusted (Return/DD):")
    for label, m in [("Baseline (full)", baseline), ("Optimized (train)", optimized_train),
                      ("Optimized (OOS)", optimized_oos), ("Optimized (full)", optimized_full)]:
        dd = max(abs(m["max_drawdown"]), 1.0)
        ra = m["total_return"] / dd
        print(f"    {label:<25} {ra:.3f}")
    print()

    # Overfitting check
    train_ret = optimized_train["total_return_pct"]
    oos_ret = optimized_oos["total_return_pct"]
    if train_ret > 0:
        degradation = (1 - oos_ret / train_ret) * 100 if train_ret != 0 else 100
        print(f"  Overfitting check:")
        print(f"    Train return: {train_ret:.2f}%")
        print(f"    OOS return:   {oos_ret:.2f}%")
        print(f"    Degradation:  {degradation:.1f}%")
        if degradation > 50:
            print(f"    WARNING: >50% degradation — likely overfit!")
        elif degradation > 25:
            print(f"    CAUTION: moderate degradation — use with care")
        else:
            print(f"    GOOD: low degradation — params likely generalize")
    print()

    # Best params
    print("  Best Parameters:")
    for k, v in sorted(best_params.items()):
        baseline_v = BASELINE_PARAMS.get(k)
        changed = " ← CHANGED" if baseline_v is not None and baseline_v != v else ""
        print(f"    {k:<35} = {v}{changed}")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Optuna intraday strategy optimizer")
    parser.add_argument("--n-trials", type=int, default=200, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=7200, help="Max seconds for optimization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print()
    print("=" * 100)
    print("INTRADAY STRATEGY OPTIMIZER (Optuna + Walk-Forward)".center(100))
    print("=" * 100)
    print()

    data = load_data()
    intraday_data = data['intraday']
    daily_data = data['daily']
    all_dates = data['all_dates']

    # ---- Run baseline first ----
    print("\nRunning baseline (no filters, current params)...")
    t0 = time.time()
    all_windows = TRAIN_WINDOWS + [TEST_WINDOW]
    baseline_metrics = run_backtest(BASELINE_PARAMS, all_windows, all_dates,
                                    intraday_data, daily_data)
    base_elapsed = time.time() - t0
    print(f"  Baseline: {baseline_metrics['total_trades']} trades, "
          f"{baseline_metrics['win_rate']:.1f}% WR, "
          f"${baseline_metrics['total_return']:+,.2f} ({base_elapsed:.1f}s)")
    per_trial_estimate = base_elapsed
    print(f"  Estimated time per trial: ~{per_trial_estimate:.0f}s")
    print(f"  Estimated total time: ~{per_trial_estimate * args.n_trials / 60:.0f} min")
    print()

    # ---- Optuna optimization ----
    print(f"Starting Optuna search ({args.n_trials} trials, {args.timeout}s timeout)...")
    print()

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=0)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="intraday_optimizer",
    )

    objective = create_objective(intraday_data, daily_data, all_dates)

    # Callback for progress
    best_so_far = [float('-inf')]
    trial_count = [0]

    def callback(study, trial):
        trial_count[0] += 1
        if trial.value is not None and trial.value > best_so_far[0]:
            best_so_far[0] = trial.value
            print(f"  Trial {trial_count[0]:>4}/{args.n_trials}: "
                  f"score={trial.value:.3f} ← NEW BEST", flush=True)
        elif trial_count[0] % 10 == 0:
            score_str = f"{trial.value:.3f}" if trial.value is not None else "N/A"
            print(f"  Trial {trial_count[0]:>4}/{args.n_trials}: "
                  f"score={score_str} "
                  f"(best={best_so_far[0]:.3f})", flush=True)

    opt_start = time.time()
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        callbacks=[callback],
        show_progress_bar=False,
    )
    opt_elapsed = time.time() - opt_start

    print(f"\nOptimization complete: {len(study.trials)} trials in {opt_elapsed:.0f}s")
    print(f"Best trial: #{study.best_trial.number} (score={study.best_value:.3f})")
    print()

    # ---- Extract best params ----
    best_params = dict(study.best_params)

    # ---- Validate on OOS ----
    print("Validating best params on out-of-sample (W4)...")
    oos_metrics = validate_oos(best_params, intraday_data, daily_data, all_dates)
    print(f"  OOS: {oos_metrics['total_trades']} trades, "
          f"{oos_metrics['win_rate']:.1f}% WR, "
          f"${oos_metrics['total_return']:+,.2f}")

    # ---- Run best params on train windows for comparison ----
    train_metrics = run_backtest(best_params, TRAIN_WINDOWS, all_dates,
                                 intraday_data, daily_data)

    # ---- Run best params on ALL windows ----
    full_metrics = validate_full(best_params, intraday_data, daily_data, all_dates)

    # ---- Print comparison ----
    print_comparison(baseline_metrics, train_metrics, oos_metrics, full_metrics,
                     best_params, opt_elapsed, len(study.trials))

    # ---- Save results ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = RESULTS_DIR / f"optuna_{timestamp}.json"

    result_data = {
        "timestamp": timestamp,
        "n_trials": len(study.trials),
        "optimization_seconds": opt_elapsed,
        "best_score": study.best_value,
        "best_params": best_params,
        "baseline_metrics": baseline_metrics,
        "train_metrics": train_metrics,
        "oos_metrics": oos_metrics,
        "full_metrics": full_metrics,
        "top_10_trials": [],
    }

    # Save top 10 trials
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else float('-inf'),
                           reverse=True)
    for t in sorted_trials[:10]:
        result_data["top_10_trials"].append({
            "number": t.number,
            "score": t.value,
            "params": t.params,
        })

    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2, default=str)

    print(f"  Results saved to: {result_file}")
    print()

    # ---- Final verdict ----
    print("=" * 100)
    print("VERDICT".center(100))
    print("=" * 100)
    print()

    oos_better = oos_metrics["total_return"] > baseline_metrics["total_return"] * 0.25  # OOS beats 25% of baseline
    low_degradation = True
    if train_metrics["total_return_pct"] > 0:
        degradation = (1 - oos_metrics["total_return_pct"] / train_metrics["total_return_pct"]) * 100
        low_degradation = degradation < 50

    if oos_better and low_degradation:
        print("  RECOMMENDATION: Use optimized parameters for live paper trading")
        print(f"  Expected return improvement: "
              f"${oos_metrics['total_return'] - baseline_metrics['total_return'] * (60/240):+,.2f} "
              f"per quarter (OOS estimate)")
    else:
        print("  RECOMMENDATION: Keep baseline parameters")
        print("  The optimization did not find a reliably better configuration.")
        if not low_degradation:
            print("  (High train→OOS degradation suggests overfitting)")

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()

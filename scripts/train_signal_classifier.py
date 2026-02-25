#!/usr/bin/env python3
"""
ML Signal Classifier — learns which signals become winning trades.

Instead of rule-based filters (SPY gate, SMA gate, VWAP gate), this trains
a gradient boosting classifier on ALL generated signals with known outcomes
(did price hit target or stop first in the 1-min bars?).

Pipeline:
  1. Generate signals from all trading days using the intraday engine
  2. For each signal, compute features at entry time
  3. Label each signal: did it win (hit target) or lose (hit stop)?
  4. Train sklearn GradientBoostingClassifier with walk-forward splits
  5. Save the model for use in the optimizer / live scanner

Usage:
    cd /home/user0/rdt-trading-system
    python scripts/train_signal_classifier.py
"""

import sys
import time
import json
import pickle
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from backtesting.engine_intraday import (
    IntradayBacktestEngine, _market_open_utc, _market_close_utc,
    _calculate_atr, _close_col
)
from risk.models import RiskLimits

# ============================================================================
# Configuration
# ============================================================================

INTRADAY_CACHE = PROJECT_ROOT / "data" / "intraday_cache"
MODEL_DIR = PROJECT_ROOT / "data" / "ml_models"

STOCK_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    'JPM', 'V', 'JNJ', 'UNH', 'HD', 'PG', 'MA', 'DIS',
    'PYPL', 'ADBE', 'CRM', 'NFLX', 'INTC', 'AMD', 'CSCO',
    'PEP', 'KO', 'MRK', 'ABT', 'TMO', 'COST', 'AVGO', 'TXN'
]

STOP_ATR_MULT = 1.5
TARGET_ATR_MULT = 2.0
MAX_HOLDING_BARS = 390 * 5  # 5 trading days worth of 1-min bars


# ============================================================================
# Signal Generation + Labeling
# ============================================================================

def load_data():
    """Load intraday + build daily data."""
    print("Loading data...")
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
            df = pd.read_parquet(path)
            # Ensure lowercase columns
            df.columns = [c.lower() for c in df.columns]
            intraday_data[symbol] = df

    # Build daily bars
    daily_data = {}
    for symbol, df in intraday_data.items():
        daily = df.resample('D').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()
        if len(daily) > 0:
            daily_data[symbol] = daily

    print(f"  Loaded {len(intraday_data)} symbols ({time.time()-t0:.1f}s)")
    return intraday_data, daily_data


def generate_labeled_signals(intraday_data, daily_data):
    """
    Generate all possible signals and label them by outcome.

    For each trading day, at scan times (30, 120, 210 min after open):
      - Compute RRS for each stock
      - If |RRS| > 0 → signal generated
      - Determine direction (long if RRS > 0, short if RRS < 0)
      - Simulate: walk forward through 1-min bars to see if target or stop hit first
      - Label: 1 = target hit first (win), 0 = stop hit first or time expired (loss)
      - Extract features at signal time
    """
    print("\nGenerating and labeling signals...")
    t0 = time.time()

    spy_daily = daily_data.get("SPY")
    if spy_daily is None:
        print("ERROR: No SPY daily data")
        return pd.DataFrame()

    spy_intraday = intraday_data.get("SPY")
    if spy_intraday is None:
        print("ERROR: No SPY intraday data")
        return pd.DataFrame()

    # Get all trading dates (skip first 30 for warmup)
    all_dates = sorted(set(spy_daily.index.date))
    all_dates = all_dates[30:]  # Warmup

    signals = []
    scan_offsets = [30, 120, 210]  # Minutes after open

    for day_idx, current_date in enumerate(all_dates):
        if day_idx % 50 == 0:
            print(f"  Day {day_idx}/{len(all_dates)} ({current_date})...", flush=True)

        market_open = _market_open_utc(current_date)
        market_close = _market_close_utc(current_date)

        # SPY context
        spy_up_to = spy_daily[spy_daily.index.date <= current_date]
        if len(spy_up_to) < 20:
            continue

        spy_close = float(spy_up_to["close"].iloc[-1])
        spy_prev_close = float(spy_up_to["close"].iloc[-2])
        spy_pc = ((spy_close / spy_prev_close) - 1) * 100

        # SPY SMA context
        spy_sma50 = float(spy_up_to["close"].tail(50).mean()) if len(spy_up_to) >= 50 else None
        spy_sma200 = float(spy_up_to["close"].tail(200).mean()) if len(spy_up_to) >= 200 else None
        spy_above_sma50 = spy_close > spy_sma50 if spy_sma50 else None
        spy_above_sma200 = spy_close > spy_sma200 if spy_sma200 else None

        # SPY volatility (ATR)
        spy_atr = _calculate_atr(spy_up_to)

        for scan_offset in scan_offsets:
            scan_time = market_open + timedelta(minutes=scan_offset)

            for symbol in STOCK_SYMBOLS:
                if symbol not in intraday_data or symbol not in daily_data:
                    continue

                sym_daily = daily_data[symbol]
                sym_intraday = intraday_data[symbol]

                sym_up_to = sym_daily[sym_daily.index.date <= current_date]
                if len(sym_up_to) < 20:
                    continue

                # Get price at scan time
                if scan_time not in sym_intraday.index:
                    continue

                scan_bar = sym_intraday.loc[scan_time]
                if isinstance(scan_bar, pd.DataFrame):
                    scan_bar = scan_bar.iloc[0]

                current_price = float(scan_bar["close"])
                if current_price <= 0:
                    continue

                # Stock context
                stock_prev_close = float(sym_up_to["close"].iloc[-1])
                if sym_up_to.index.date[-1] == current_date and len(sym_up_to) >= 2:
                    stock_prev_close = float(sym_up_to["close"].iloc[-2])

                atr = _calculate_atr(sym_up_to)
                if atr <= 0:
                    continue

                # RRS
                atr_pct = (atr / current_price) * 100
                if atr_pct <= 0:
                    continue
                stock_pc = ((current_price / stock_prev_close) - 1) * 100
                rrs = (stock_pc - spy_pc) / atr_pct

                if not np.isfinite(rrs) or rrs == 0:
                    continue

                direction = "long" if rrs > 0 else "short"

                # ---- Compute stop/target ----
                stop_distance = atr * STOP_ATR_MULT
                target_distance = atr * TARGET_ATR_MULT

                if direction == "long":
                    stop_price = current_price - stop_distance
                    target_price = current_price + target_distance
                else:
                    stop_price = current_price + stop_distance
                    target_price = current_price - target_distance

                # ---- Simulate outcome ----
                outcome = simulate_trade(
                    sym_intraday, scan_time, direction,
                    stop_price, target_price, current_date, intraday_data
                )

                # ---- Extract features ----
                # Stock SMAs
                sma20 = float(sym_up_to["close"].tail(20).mean())
                sma50 = float(sym_up_to["close"].tail(50).mean()) if len(sym_up_to) >= 50 else current_price
                sma200 = float(sym_up_to["close"].tail(200).mean()) if len(sym_up_to) >= 200 else current_price

                # VWAP at scan time (approximate from day's bars so far)
                day_bars = sym_intraday[(sym_intraday.index >= market_open) & (sym_intraday.index <= scan_time)]
                vwap = 0.0
                if len(day_bars) > 0:
                    tp = (day_bars["high"] + day_bars["low"] + day_bars["close"]) / 3
                    vwap = float((tp * day_bars["volume"]).sum() / day_bars["volume"].sum()) if day_bars["volume"].sum() > 0 else current_price

                # Volume context
                avg_volume_20d = float(sym_up_to["volume"].tail(20).mean()) if len(sym_up_to) >= 20 else 0
                today_volume = float(day_bars["volume"].sum()) if len(day_bars) > 0 else 0
                volume_ratio = today_volume / avg_volume_20d if avg_volume_20d > 0 else 1.0

                # Price position relative to range
                day_high = float(day_bars["high"].max()) if len(day_bars) > 0 else current_price
                day_low = float(day_bars["low"].min()) if len(day_bars) > 0 else current_price
                day_range = day_high - day_low
                price_in_range = (current_price - day_low) / day_range if day_range > 0 else 0.5

                # Sector ETF RS (use XLK as proxy for tech)
                sector_etf = _get_sector_etf(symbol)
                sector_rs = 0.0
                if sector_etf and sector_etf in daily_data:
                    etf_daily = daily_data[sector_etf]
                    etf_up_to = etf_daily[etf_daily.index.date <= current_date]
                    if len(etf_up_to) >= 2:
                        etf_pc = ((float(etf_up_to["close"].iloc[-1]) / float(etf_up_to["close"].iloc[-2])) - 1) * 100
                        sector_rs = etf_pc - spy_pc

                # Daily candle pattern
                daily_change_pct = stock_pc
                prev_close = float(sym_up_to["close"].iloc[-2]) if len(sym_up_to) >= 2 else current_price
                prev2_close = float(sym_up_to["close"].iloc[-3]) if len(sym_up_to) >= 3 else prev_close
                momentum_2d = ((prev_close / prev2_close) - 1) * 100 if prev2_close > 0 else 0

                signals.append({
                    "date": current_date,
                    "symbol": symbol,
                    "scan_offset_min": scan_offset,
                    "direction": direction,
                    "entry_price": current_price,
                    "outcome": outcome,  # 1 = win, 0 = loss

                    # Features
                    "rrs": abs(rrs),
                    "rrs_raw": rrs,
                    "atr_pct": atr_pct,
                    "price_vs_vwap_pct": ((current_price / vwap) - 1) * 100 if vwap > 0 else 0,
                    "price_vs_sma20_pct": ((current_price / sma20) - 1) * 100,
                    "price_vs_sma50_pct": ((current_price / sma50) - 1) * 100,
                    "price_vs_sma200_pct": ((current_price / sma200) - 1) * 100,
                    "volume_ratio": min(volume_ratio, 5.0),  # Cap outliers
                    "price_in_day_range": price_in_range,
                    "scan_time_minutes": scan_offset,
                    "daily_change_pct": daily_change_pct,
                    "momentum_2d_pct": momentum_2d,
                    "sector_rs": sector_rs,
                    "spy_change_pct": spy_pc,
                    "spy_above_sma50": 1 if spy_above_sma50 else 0,
                    "spy_above_sma200": 1 if spy_above_sma200 else 0,
                    "spy_atr_pct": (spy_atr / spy_close * 100) if spy_close > 0 else 0,
                    "is_long": 1 if direction == "long" else 0,
                    "stop_distance_pct": (stop_distance / current_price) * 100,
                    "target_distance_pct": (target_distance / current_price) * 100,
                })

    elapsed = time.time() - t0
    df = pd.DataFrame(signals)
    print(f"  Generated {len(df)} signals in {elapsed:.1f}s")

    if len(df) > 0:
        wins = df["outcome"].sum()
        print(f"  Win rate: {wins}/{len(df)} ({wins/len(df)*100:.1f}%)")

    return df


def simulate_trade(sym_intraday, entry_time, direction, stop_price, target_price,
                   entry_date, intraday_data):
    """
    Walk forward through 1-min bars to determine if target or stop hit first.
    Returns 1 (win) or 0 (loss/timeout).
    """
    # Get bars after entry
    future_bars = sym_intraday[sym_intraday.index > entry_time]
    if len(future_bars) == 0:
        return 0

    # Limit to MAX_HOLDING_BARS
    future_bars = future_bars.iloc[:MAX_HOLDING_BARS]

    for _, bar in future_bars.iterrows():
        high = float(bar["high"])
        low = float(bar["low"])

        if direction == "long":
            if low <= stop_price:
                return 0  # Stop hit → loss
            if high >= target_price:
                return 1  # Target hit → win
        else:
            if high >= stop_price:
                return 0  # Stop hit → loss
            if low <= target_price:
                return 1  # Target hit → win

    return 0  # Time expired → loss


def _get_sector_etf(symbol):
    """Map stock to sector ETF."""
    sector_map = {
        'AAPL': 'XLK', 'MSFT': 'XLK', 'GOOGL': 'XLC', 'AMZN': 'XLY',
        'NVDA': 'XLK', 'META': 'XLC', 'TSLA': 'XLY', 'JPM': 'XLF',
        'V': 'XLF', 'JNJ': 'XLV', 'UNH': 'XLV', 'HD': 'XLY',
        'PG': 'XLP', 'MA': 'XLF', 'DIS': 'XLC', 'PYPL': 'XLF',
        'ADBE': 'XLK', 'CRM': 'XLK', 'NFLX': 'XLC', 'INTC': 'XLK',
        'AMD': 'XLK', 'CSCO': 'XLK', 'PEP': 'XLP', 'KO': 'XLP',
        'MRK': 'XLV', 'ABT': 'XLV', 'TMO': 'XLV', 'COST': 'XLP',
        'AVGO': 'XLK', 'TXN': 'XLK',
    }
    return sector_map.get(symbol)


# ============================================================================
# Training
# ============================================================================

FEATURE_COLS = [
    "rrs", "atr_pct", "price_vs_vwap_pct", "price_vs_sma20_pct",
    "price_vs_sma50_pct", "price_vs_sma200_pct", "volume_ratio",
    "price_in_day_range", "scan_time_minutes", "daily_change_pct",
    "momentum_2d_pct", "sector_rs", "spy_change_pct",
    "spy_above_sma50", "spy_above_sma200", "spy_atr_pct",
    "is_long", "stop_distance_pct", "target_distance_pct",
]


def train_classifier(df: pd.DataFrame):
    """Train gradient boosting classifier with time-series cross-validation."""
    print("\nTraining signal classifier...")
    print(f"  Features: {len(FEATURE_COLS)}")
    print(f"  Samples: {len(df)}")
    print(f"  Class balance: {df['outcome'].value_counts().to_dict()}")

    X = df[FEATURE_COLS].fillna(0).values
    y = df["outcome"].values

    # Time-series split (respect temporal ordering)
    tscv = TimeSeriesSplit(n_splits=4)

    cv_scores = []
    cv_aucs = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob) if len(set(y_val)) > 1 else 0.5
        cv_scores.append(acc)
        cv_aucs.append(auc)
        print(f"  Fold {fold+1}: accuracy={acc:.3f}, AUC={auc:.3f}")

    print(f"  Mean accuracy: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")
    print(f"  Mean AUC: {np.mean(cv_aucs):.3f} (+/- {np.std(cv_aucs):.3f})")

    # ---- Train final model on all data ----
    print("\nTraining final model on all data...")
    final_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )
    final_model.fit(X, y)

    # Feature importance
    print("\n  Feature importance (top 10):")
    importances = list(zip(FEATURE_COLS, final_model.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)
    for feat, imp in importances[:10]:
        bar = "█" * int(imp * 100)
        print(f"    {feat:<25} {imp:.4f} {bar}")

    # ---- Walk-forward backtest simulation ----
    # Use last 25% of data as test set, train on first 75%
    split_idx = int(len(df) * 0.75)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    df_test = df.iloc[split_idx:]

    wf_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=20, random_state=42,
    )
    wf_model.fit(X_train, y_train)

    y_prob_test = wf_model.predict_proba(X_test)[:, 1]

    # Simulate: only take trades where model predicts > threshold
    print("\n  Walk-forward filter simulation (last 25% of data):")
    print(f"  {'Threshold':>10} {'Trades':>8} {'Win Rate':>10} {'vs Unfiltered':>15}")
    print(f"  {'-'*48}")

    unfiltered_wr = y_test.mean() * 100
    print(f"  {'(none)':>10} {len(y_test):>8} {unfiltered_wr:>9.1f}%  {'(baseline)':>15}")

    best_threshold = 0.5
    best_improvement = 0

    for threshold in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = y_prob_test >= threshold
        if mask.sum() < 5:
            continue
        filtered_y = y_test[mask]
        filtered_wr = filtered_y.mean() * 100
        delta = filtered_wr - unfiltered_wr
        print(f"  {threshold:>10.2f} {mask.sum():>8} {filtered_wr:>9.1f}% {delta:>+14.1f}%")

        if delta > best_improvement and mask.sum() >= 10:
            best_improvement = delta
            best_threshold = threshold

    print(f"\n  Best threshold: {best_threshold:.2f} (improves WR by {best_improvement:+.1f}%)")

    return final_model, {
        "cv_accuracy": float(np.mean(cv_scores)),
        "cv_auc": float(np.mean(cv_aucs)),
        "feature_importance": {k: float(v) for k, v in importances},
        "best_threshold": best_threshold,
        "features": FEATURE_COLS,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print()
    print("=" * 80)
    print("ML SIGNAL CLASSIFIER TRAINING".center(80))
    print("=" * 80)

    intraday_data, daily_data = load_data()

    # Generate and label signals
    df = generate_labeled_signals(intraday_data, daily_data)

    if len(df) < 100:
        print(f"\nERROR: Only {len(df)} signals generated — need at least 100")
        sys.exit(1)

    # Train classifier
    model, metadata = train_classifier(df)

    # Save model and metadata
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    model_path = MODEL_DIR / f"signal_classifier_{timestamp}.pkl"
    meta_path = MODEL_DIR / f"signal_classifier_{timestamp}_meta.json"
    data_path = MODEL_DIR / f"signal_data_{timestamp}.parquet"

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    df.to_parquet(data_path)

    print(f"\n  Model saved: {model_path}")
    print(f"  Metadata saved: {meta_path}")
    print(f"  Signal data saved: {data_path}")
    print(f"  Total signals: {len(df)}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()

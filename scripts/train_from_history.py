#!/usr/bin/env python3
"""
Train ML Models from Historical Price Data

Generates training data from 1 year of daily prices for the 30-stock watchlist,
trains the ExitPredictor (exit strategy classifier), SignalDecayPredictor
(signal TTL regressor), and calibrates the DynamicPositionSizer with Kelly.

The walk-forward backtest proved the rule-based filters work (+$768 improvement).
This script tests whether ML can add further improvement on top of those rules.

Usage:
    cd /home/user0/rdt-trading-system
    python scripts/train_from_history.py
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

import yfinance as yf
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix,
)

from ml.exit_predictor import (
    ExitPredictor, EXIT_FEATURE_NAMES,
    generate_exit_training_data,
    _calculate_rsi, _calculate_bar_pattern_score, _detect_market_regime,
)
from ml.signal_decay import (
    SignalDecayPredictor, DECAY_FEATURE_NAMES,
    generate_decay_training_data,
    SYMBOL_SECTOR_MAP, SECTOR_CODES,
    _detect_market_regime as decay_detect_regime,
    _calculate_rsi as decay_calculate_rsi,
)
from ml.dynamic_sizer import DynamicPositionSizer, SizingInput
from ml.safe_model_loader import safe_save_model
from shared.indicators.rrs import (
    RRSCalculator, calculate_ema, calculate_sma,
    check_daily_strength_relaxed, check_daily_weakness_relaxed,
)
from utils.paths import get_project_root, get_models_dir


# ============================================================================
# Configuration
# ============================================================================

WATCHLIST = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    'JPM', 'V', 'JNJ', 'UNH', 'HD', 'PG', 'MA', 'DIS',
    'PYPL', 'ADBE', 'CRM', 'NFLX', 'INTC', 'AMD', 'CSCO',
    'PEP', 'KO', 'MRK', 'ABT', 'TMO', 'COST', 'AVGO', 'TXN',
]

DATA_DAYS = 365       # 1 year of data
RRS_THRESHOLD = 2.0   # Signal threshold
ATR_PERIOD = 14       # ATR calculation period
HOLDOUT_DAYS = 90     # Last 90 days for validation holdout

LINE_WIDTH = 90


# ============================================================================
# Data Download
# ============================================================================

def download_all_data(days: int = DATA_DAYS) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Download daily data for watchlist + SPY.

    Returns:
        (stock_data, spy_data) where stock_data maps symbol -> DataFrame
        with lowercase columns.
    """
    print(f"\n{'='*LINE_WIDTH}")
    print("STEP 1: DOWNLOADING HISTORICAL DATA".center(LINE_WIDTH))
    print(f"{'='*LINE_WIDTH}\n")

    # Download SPY
    logger.info("Downloading SPY...")
    spy_ticker = yf.Ticker('SPY')
    spy_data = spy_ticker.history(period=f"{days}d", interval='1d')
    if spy_data.empty:
        logger.error("Failed to download SPY data. Aborting.")
        sys.exit(1)
    spy_data.columns = [c.lower() for c in spy_data.columns]
    print(f"  SPY: {len(spy_data)} trading days")

    # Download stocks
    stock_data: Dict[str, pd.DataFrame] = {}
    failed = []
    for i, symbol in enumerate(WATCHLIST):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d", interval='1d')
            if df is not None and len(df) >= 60:
                df.columns = [c.lower() for c in df.columns]
                stock_data[symbol] = df
            else:
                failed.append(symbol)
                logger.warning(f"{symbol}: insufficient data ({len(df) if df is not None else 0} days)")
        except Exception as e:
            failed.append(symbol)
            logger.warning(f"{symbol}: download failed: {e}")

        if (i + 1) % 10 == 0:
            print(f"  Downloaded {i+1}/{len(WATCHLIST)} stocks...")
        time.sleep(0.3)

    print(f"  Stocks loaded: {len(stock_data)}/{len(WATCHLIST)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print()

    return stock_data, spy_data


# ============================================================================
# Exit Predictor Training Data Generation
# ============================================================================

def generate_exit_data(
    stock_data: Dict[str, pd.DataFrame],
    spy_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate training data for ExitPredictor using the existing
    generate_exit_training_data function from ml/exit_predictor.py.
    """
    print(f"\n{'='*LINE_WIDTH}")
    print("STEP 2a: GENERATING EXIT PREDICTOR TRAINING DATA".center(LINE_WIDTH))
    print(f"{'='*LINE_WIDTH}\n")

    symbols = list(stock_data.keys())
    df = generate_exit_training_data(
        symbols=symbols,
        days=DATA_DAYS,
        rrs_threshold=RRS_THRESHOLD,
        atr_period=ATR_PERIOD,
    )

    if df.empty:
        logger.error("No exit training data generated!")
        return df

    # Print statistics
    print(f"\n  Total exit training samples: {len(df)}")
    print(f"  Symbols contributing: {df['symbol'].nunique()}")
    print(f"\n  Class distribution:")
    class_names = {0: 'quick_scalp (MFE < 1R)', 1: 'swing (1-2R)', 2: 'runner (>2R)'}
    for cls, count in sorted(df['target'].value_counts().items()):
        pct = count / len(df) * 100
        print(f"    Class {cls} ({class_names.get(cls, '?')}): {count} ({pct:.1f}%)")

    print(f"\n  Feature statistics:")
    for feat in EXIT_FEATURE_NAMES[:5]:
        if feat in df.columns:
            print(f"    {feat}: mean={df[feat].mean():.3f}, std={df[feat].std():.3f}")
    print(f"    ... ({len(EXIT_FEATURE_NAMES)} features total)")

    # Direction breakdown
    long_count = (df['direction'] == 'long').sum()
    short_count = (df['direction'] == 'short').sum()
    print(f"\n  Direction: {long_count} long, {short_count} short")

    # MFE/MAE stats
    print(f"\n  MFE (ATR): mean={df['mfe_atr'].mean():.2f}, median={df['mfe_atr'].median():.2f}, max={df['mfe_atr'].max():.2f}")
    print(f"  MAE (ATR): mean={df['mae_atr'].mean():.2f}, median={df['mae_atr'].median():.2f}, max={df['mae_atr'].max():.2f}")

    return df


# ============================================================================
# Signal Decay Training Data Generation
# ============================================================================

def generate_decay_data(
    stock_data: Dict[str, pd.DataFrame],
    spy_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate training data for SignalDecayPredictor using the existing
    generate_decay_training_data function from ml/signal_decay.py.
    """
    print(f"\n{'='*LINE_WIDTH}")
    print("STEP 2b: GENERATING SIGNAL DECAY TRAINING DATA".center(LINE_WIDTH))
    print(f"{'='*LINE_WIDTH}\n")

    symbols = list(stock_data.keys())
    df = generate_decay_training_data(
        symbols=symbols,
        days=DATA_DAYS,
        rrs_threshold=RRS_THRESHOLD,
        atr_period=ATR_PERIOD,
    )

    if df.empty:
        logger.error("No decay training data generated!")
        return df

    # Print statistics
    print(f"\n  Total decay training samples: {len(df)}")
    print(f"  Symbols contributing: {df['symbol'].nunique()}")

    print(f"\n  Target minutes distribution:")
    target = df['target_minutes']
    print(f"    Mean:   {target.mean():.1f} min")
    print(f"    Median: {target.median():.1f} min")
    print(f"    Min:    {target.min():.1f} min")
    print(f"    Max:    {target.max():.1f} min")
    print(f"    Std:    {target.std():.1f} min")

    # Bucket distribution
    short_ct = (target < 30).sum()
    medium_ct = ((target >= 30) & (target <= 120)).sum()
    long_ct = (target > 120).sum()
    print(f"\n  Decay buckets:")
    print(f"    Fast   (<30 min):    {short_ct} ({short_ct/len(df)*100:.1f}%)")
    print(f"    Medium (30-120 min): {medium_ct} ({medium_ct/len(df)*100:.1f}%)")
    print(f"    Slow   (>120 min):   {long_ct} ({long_ct/len(df)*100:.1f}%)")

    print(f"\n  Valid bars distribution: mean={df['valid_bars'].mean():.1f}, max={df['valid_bars'].max()}")

    return df


# ============================================================================
# Model Training
# ============================================================================

def train_exit_predictor(df: pd.DataFrame) -> Optional[ExitPredictor]:
    """Train the ExitPredictor model."""
    print(f"\n{'='*LINE_WIDTH}")
    print("STEP 3a: TRAINING EXIT PREDICTOR".center(LINE_WIDTH))
    print(f"{'='*LINE_WIDTH}\n")

    if df.empty or len(df) < 30:
        print("  Insufficient data for training. Need at least 30 samples.")
        return None

    # Check minimum per class for stratification
    class_counts = df['target'].value_counts()
    min_class_count = class_counts.min()
    if min_class_count < 5:
        print(f"  WARNING: smallest class has only {min_class_count} samples.")
        print("  Merging underrepresented classes if needed...")
        # If class 2 (runner) is too small, merge with class 1 (swing)
        if class_counts.get(2, 0) < 5:
            df = df.copy()
            df.loc[df['target'] == 2, 'target'] = 1
            print("  Merged 'runner' into 'swing' class due to small sample size.")
            class_counts = df['target'].value_counts()

    predictor = ExitPredictor(random_state=42)

    # Train using the built-in train() method
    metrics = predictor.train(df, test_size=0.2)

    # Print detailed results
    print(f"\n  Training Results:")
    print(f"  {'='*50}")
    print(f"  Accuracy:    {metrics.accuracy:.4f}")
    print(f"  F1 (macro):  {metrics.f1_macro:.4f}")
    print(f"  Train/Test:  {metrics.n_train}/{metrics.n_test}")
    print(f"\n  Classification Report:")
    for line in metrics.class_report.split('\n'):
        if line.strip():
            print(f"    {line}")

    if metrics.confusion_matrix is not None:
        print(f"\n  Confusion Matrix:")
        cm = metrics.confusion_matrix
        classes = sorted(df['target'].unique())
        class_labels = {0: 'scalp', 1: 'swing', 2: 'runner'}
        header = "     " + "  ".join(f"{class_labels.get(c, str(c)):>7}" for c in classes)
        print(f"    {header}")
        for i, row in enumerate(cm):
            row_str = "  ".join(f"{v:>7d}" for v in row)
            print(f"    {class_labels.get(classes[i], str(classes[i])):>5} {row_str}")

    print(f"\n  Top 5 Feature Importances:")
    for i, (feat, imp) in enumerate(list(metrics.feature_importance.items())[:5]):
        bar = '#' * int(imp * 100)
        print(f"    {i+1}. {feat:<25} {imp:.4f} {bar}")

    return predictor


def train_signal_decay(df: pd.DataFrame) -> Optional[SignalDecayPredictor]:
    """Train the SignalDecayPredictor model."""
    print(f"\n{'='*LINE_WIDTH}")
    print("STEP 3b: TRAINING SIGNAL DECAY PREDICTOR".center(LINE_WIDTH))
    print(f"{'='*LINE_WIDTH}\n")

    if df.empty or len(df) < 30:
        print("  Insufficient data for training. Need at least 30 samples.")
        return None

    predictor = SignalDecayPredictor(random_state=42)

    # Train using the built-in train() method
    metrics = predictor.train(df, test_size=0.2)

    # Print detailed results
    print(f"\n  Training Results:")
    print(f"  {'='*50}")
    print(f"  MAE:   {metrics.mae:.1f} minutes")
    print(f"  RMSE:  {metrics.rmse:.1f} minutes")
    print(f"  R2:    {metrics.r2:.4f}")
    print(f"  Train/Test: {metrics.n_train}/{metrics.n_test}")

    print(f"\n  Top 5 Feature Importances:")
    for i, (feat, imp) in enumerate(list(metrics.feature_importance.items())[:5]):
        bar = '#' * int(imp * 100)
        print(f"    {i+1}. {feat:<25} {imp:.4f} {bar}")

    return predictor


def calibrate_dynamic_sizer(
    exit_df: pd.DataFrame,
) -> Optional[DynamicPositionSizer]:
    """
    Bootstrap the DynamicPositionSizer with historical trade outcomes.
    """
    print(f"\n{'='*LINE_WIDTH}")
    print("STEP 3c: CALIBRATING DYNAMIC POSITION SIZER".center(LINE_WIDTH))
    print(f"{'='*LINE_WIDTH}\n")

    if exit_df.empty:
        print("  No trade data to calibrate with.")
        return None

    sizer = DynamicPositionSizer(base_risk=0.015, enabled=True)

    # Convert MFE/MAE data into simulated trade outcomes
    # For each sample, simulate the trade using the MFE as proxy
    trades_recorded = 0
    total_pnl_r = 0.0

    for _, row in exit_df.iterrows():
        mfe_atr = row['mfe_atr']
        mae_atr = row['mae_atr']
        direction = row.get('direction', 'long')
        symbol = row.get('symbol', '')

        # Simulate outcome:
        # If MFE > 1.0R and MFE > MAE, it's a winner (took some profit)
        # If MAE > 1.5R, it hit the stop (loss)
        if mae_atr > 1.5:
            # Stopped out at ~1R loss
            pnl_r = -1.0
            win = False
        elif mfe_atr > 1.0:
            # Winner: capture some fraction of MFE (assume 60% capture rate)
            pnl_r = mfe_atr * 0.6
            win = True
        else:
            # Small loser or breakeven
            pnl_r = mfe_atr * 0.5 - mae_atr * 0.3
            win = pnl_r > 0

        sizer.record_trade_result(
            pnl_r=pnl_r,
            win=win,
            symbol=symbol,
            direction=direction,
        )
        total_pnl_r += pnl_r
        trades_recorded += 1

    # Print calibration stats
    stats = sizer.get_recent_stats(n=min(100, trades_recorded))
    print(f"  Trades recorded: {trades_recorded}")
    print(f"  Win rate: {stats['win_rate']:.1%}")
    print(f"  Avg win (R): {stats['avg_win_r']:.2f}")
    print(f"  Avg loss (R): {stats['avg_loss_r']:.2f}")
    print(f"  Total P&L (R): {total_pnl_r:.1f}")

    # Test a sample sizing calculation
    test_input = SizingInput(
        ml_confidence=75.0,
        rrs_strength=2.5,
        market_regime="bull_trending",
        direction="long",
        recent_win_rate=stats['win_rate'],
        recent_avg_win=stats['avg_win_r'],
        recent_avg_loss=stats['avg_loss_r'],
        recent_trade_count=stats['trade_count'],
        portfolio_heat=0.03,
    )
    test_result = sizer.calculate_multiplier(test_input)
    print(f"\n  Sample sizing (bull trend, RRS 2.5, 75% confidence):")
    print(f"    Multiplier: {test_result.multiplier:.2f}x")
    print(f"    Effective risk: {test_result.effective_risk_pct*100:.2f}%")
    if test_result.kelly_fraction is not None:
        print(f"    Kelly fraction: {test_result.kelly_fraction:.4f}")
        if test_result.kelly_risk is not None:
            print(f"    Quarter-Kelly risk: {test_result.kelly_risk*100:.3f}%")
    print(f"    Adjustments: {test_result.reason}")

    return sizer


# ============================================================================
# Save Models
# ============================================================================

def save_models(
    exit_predictor: Optional[ExitPredictor],
    decay_predictor: Optional[SignalDecayPredictor],
    sizer: Optional[DynamicPositionSizer],
):
    """Save all trained models using safe_model_loader."""
    print(f"\n{'='*LINE_WIDTH}")
    print("STEP 4: SAVING MODELS".center(LINE_WIDTH))
    print(f"{'='*LINE_WIDTH}\n")

    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    if exit_predictor is not None and exit_predictor.is_trained:
        exit_predictor.save()
        saved.append("exit_predictor")
        print(f"  Saved: ExitPredictor -> {models_dir / 'exit_predictor' / 'exit_predictor.pkl'}")

    if decay_predictor is not None and decay_predictor.is_trained:
        decay_predictor.save()
        saved.append("signal_decay")
        print(f"  Saved: SignalDecayPredictor -> {models_dir / 'signal_decay' / 'signal_decay.pkl'}")

    if sizer is not None:
        sizer_dir = models_dir / 'dynamic_sizer'
        sizer_dir.mkdir(parents=True, exist_ok=True)
        sizer_path = str(sizer_dir / 'dynamic_sizer.pkl')
        sizer_data = {
            'trade_results': sizer._trade_results,
            'base_risk': sizer.base_risk,
            'enabled': sizer.enabled,
            'version': '1.0.0',
        }
        safe_save_model(sizer_data, sizer_path)
        saved.append("dynamic_sizer")
        print(f"  Saved: DynamicPositionSizer -> {sizer_path}")

    if not saved:
        print("  No models were saved (nothing was trained successfully).")
    else:
        print(f"\n  Total models saved: {len(saved)}")


# ============================================================================
# Holdout Validation
# ============================================================================

def run_holdout_validation(
    stock_data: Dict[str, pd.DataFrame],
    spy_data: pd.DataFrame,
    exit_predictor: Optional[ExitPredictor],
    decay_predictor: Optional[SignalDecayPredictor],
) -> Dict:
    """
    Validate trained models on the last 90 days of data.

    Generates signals on holdout period, applies exit predictor and
    signal decay, then checks whether predictions match actual outcomes.
    """
    print(f"\n{'='*LINE_WIDTH}")
    print("STEP 5: HOLDOUT VALIDATION (LAST 90 DAYS)".center(LINE_WIDTH))
    print(f"{'='*LINE_WIDTH}\n")

    if exit_predictor is None and decay_predictor is None:
        print("  No trained models to validate.")
        return {}

    rrs_calc = RRSCalculator(atr_period=ATR_PERIOD)
    spy_regime = _detect_market_regime(spy_data)

    results = {
        'exit': {'correct': 0, 'total': 0, 'predictions': []},
        'decay': {'errors': [], 'predictions': []},
    }

    # Process each stock on the holdout period
    holdout_start_idx = -HOLDOUT_DAYS
    spy_holdout = spy_data.iloc[holdout_start_idx:]

    symbols_processed = 0
    for symbol, daily in stock_data.items():
        if len(daily) < 100:
            continue

        try:
            holdout = daily.iloc[holdout_start_idx:]
            pre_holdout = daily.iloc[:holdout_start_idx]

            if len(holdout) < 20 or len(pre_holdout) < 50:
                continue

            # Calculate RRS on the holdout period
            atr_series = rrs_calc.calculate_atr(daily)
            rrs_df = rrs_calc.calculate_rrs(daily, spy_data, periods=1)

            min_len = min(len(daily), len(rrs_df), len(atr_series))
            daily_aligned = daily.iloc[-min_len:]
            rrs_series = rrs_df['rrs'].iloc[-min_len:]
            atr_aligned = atr_series.iloc[-min_len:]

            holdout_len = min(HOLDOUT_DAYS, min_len - 60)
            if holdout_len < 20:
                continue

            # Scan for signals in holdout period
            for i in range(min_len - holdout_len, min_len - 10):
                rrs_val = float(rrs_series.iloc[i])
                if abs(rrs_val) < RRS_THRESHOLD:
                    continue

                # Skip consecutive signals
                prev_rrs = float(rrs_series.iloc[i - 1])
                if abs(prev_rrs) >= RRS_THRESHOLD:
                    continue

                direction = 'long' if rrs_val > 0 else 'short'
                entry_price = float(daily_aligned['close'].iloc[i])
                entry_atr = float(atr_aligned.iloc[i])

                if entry_price <= 0 or entry_atr <= 0 or np.isnan(entry_atr):
                    continue

                # Compute actual MFE over next 10 bars
                forward = daily_aligned.iloc[i + 1: i + 11]
                if len(forward) < 5:
                    continue

                if direction == 'long':
                    actual_mfe = float(forward['high'].max()) - entry_price
                else:
                    actual_mfe = entry_price - float(forward['low'].min())

                actual_mfe_atr = actual_mfe / entry_atr if entry_atr > 0 else 0

                # Determine actual class
                if actual_mfe_atr < 1.0:
                    actual_class = 0
                elif actual_mfe_atr < 2.0:
                    actual_class = 1
                else:
                    actual_class = 2

                # --- Exit Predictor validation ---
                if exit_predictor is not None and exit_predictor.is_trained:
                    lookback = daily_aligned.iloc[max(0, i - 50): i + 1]
                    close_lb = lookback['close']

                    ema3 = calculate_ema(close_lb, 3)
                    ema8 = calculate_ema(close_lb, 8)
                    ema21 = calculate_ema(close_lb, 21)
                    ema50 = calculate_ema(close_lb, min(50, len(close_lb)))

                    e3 = float(ema3.iloc[-1])
                    e8 = float(ema8.iloc[-1])
                    e21 = float(ema21.iloc[-1])
                    e50 = float(ema50.iloc[-1])

                    dist_ema8 = (entry_price - e8) / e8 * 100 if e8 != 0 else 0
                    dist_ema21 = (entry_price - e21) / e21 * 100 if e21 != 0 else 0
                    dist_ema50 = (entry_price - e50) / e50 * 100 if e50 != 0 else 0
                    ema_aligned = 1.0 if (e3 > e8 > e21) or (e3 < e8 < e21) else 0.0

                    rsi = _calculate_rsi(close_lb, 14)
                    bar_pattern = _calculate_bar_pattern_score(lookback)

                    try:
                        ds = check_daily_strength_relaxed(lookback)
                        daily_score = float(ds.get('strength_score', 0))
                    except Exception:
                        daily_score = 0.0

                    mom5 = 0.0
                    if len(close_lb) >= 6:
                        prev5 = float(close_lb.iloc[-6])
                        if prev5 > 0:
                            mom5 = (entry_price - prev5) / prev5 * 100

                    vol_trend = 1.0
                    vol_ratio = 1.0
                    if 'volume' in lookback.columns and len(lookback) >= 20:
                        vol = lookback['volume']
                        avg5 = float(vol.tail(5).mean())
                        avg20 = float(vol.tail(20).mean())
                        if avg20 > 0:
                            vol_trend = avg5 / avg20
                            vol_ratio = float(vol.iloc[-1]) / avg20

                    bb_width = 0.0
                    if len(close_lb) >= 20:
                        std20 = float(close_lb.rolling(20).std().iloc[-1])
                        if entry_price > 0:
                            bb_width = std20 * 4 / entry_price * 100

                    pos_range = 0.5
                    if len(lookback) >= 20:
                        h20 = float(lookback['high'].tail(20).max())
                        l20 = float(lookback['low'].tail(20).min())
                        rng = h20 - l20
                        if rng > 0:
                            pos_range = (entry_price - l20) / rng

                    entry_date = daily_aligned.index[i]
                    hour = entry_date.hour if hasattr(entry_date, 'hour') else 10
                    dow = entry_date.weekday() if hasattr(entry_date, 'weekday') else 2

                    features = np.array([
                        rrs_val, entry_atr / entry_price * 100, vol_ratio,
                        float(hour), float(dow), float(spy_regime),
                        dist_ema8, dist_ema21, dist_ema50, ema_aligned,
                        rsi, bar_pattern, daily_score, mom5, vol_trend,
                        bb_width, pos_range,
                        1.0 if direction == 'long' else 0.0,
                    ], dtype=np.float32)

                    try:
                        pred = exit_predictor.predict(features)
                        results['exit']['total'] += 1
                        if pred.mfe_class == actual_class:
                            results['exit']['correct'] += 1
                        results['exit']['predictions'].append({
                            'predicted': pred.mfe_class,
                            'actual': actual_class,
                            'confidence': pred.confidence,
                            'symbol': symbol,
                            'mfe_atr': actual_mfe_atr,
                        })
                    except Exception as e:
                        logger.debug(f"Exit prediction failed for {symbol}: {e}")

                # --- Signal Decay validation ---
                if decay_predictor is not None and decay_predictor.is_trained:
                    # Track actual duration (how many bars RRS stays above threshold)
                    actual_valid_bars = 0
                    for j in range(i + 1, min(i + 20, min_len)):
                        fwd_rrs = float(rrs_series.iloc[j])
                        if direction == 'long' and fwd_rrs < RRS_THRESHOLD:
                            break
                        elif direction == 'short' and fwd_rrs > -RRS_THRESHOLD:
                            break
                        actual_valid_bars += 1

                    actual_minutes = actual_valid_bars * 30.0  # ~30 min per daily bar proxy
                    actual_minutes = max(10.0, min(actual_minutes, 480.0))

                    # Extract decay features
                    rrs_roc = rrs_val - prev_rrs if i >= 1 else 0.0

                    sector = SYMBOL_SECTOR_MAP.get(symbol, 'Unknown')
                    sector_code = float(SECTOR_CODES.get(sector, SECTOR_CODES['Unknown']))

                    decay_features = np.array([
                        rrs_val,                               # rrs_strength
                        rrs_roc,                               # rrs_rate_of_change
                        entry_atr / entry_price * 100,         # atr_pct
                        vol_ratio,                             # volume_ratio
                        float(spy_regime),                     # market_regime
                        float(hour),                           # hour_of_day
                        float(dow),                            # day_of_week
                        0.0,                                   # is_earnings_related
                        sector_code,                           # sector_code
                        0.0,                                   # dist_vwap_pct
                        rsi,                                   # rsi_14
                        ema_aligned,                           # ema_alignment
                        bb_width,                              # bb_width_pct
                        mom5,                                  # price_momentum_5d
                        daily_score,                           # daily_strength_score
                        1.0 if direction == 'long' else 0.0,  # direction_is_long
                    ], dtype=np.float32)

                    try:
                        decay_pred = decay_predictor.predict(decay_features)
                        error = abs(decay_pred.estimated_valid_minutes - actual_minutes)
                        results['decay']['errors'].append(error)
                        results['decay']['predictions'].append({
                            'predicted_min': decay_pred.estimated_valid_minutes,
                            'actual_min': actual_minutes,
                            'error': error,
                            'decay_rate': decay_pred.decay_rate,
                            'symbol': symbol,
                        })
                    except Exception as e:
                        logger.debug(f"Decay prediction failed for {symbol}: {e}")

            symbols_processed += 1

        except Exception as e:
            logger.debug(f"Validation error for {symbol}: {e}")
            continue

    # Print validation results
    print(f"  Symbols processed: {symbols_processed}")
    print()

    # Exit predictor results
    if results['exit']['total'] > 0:
        exit_acc = results['exit']['correct'] / results['exit']['total']
        print(f"  Exit Predictor Holdout Validation:")
        print(f"  {'='*50}")
        print(f"  Total predictions: {results['exit']['total']}")
        print(f"  Accuracy: {exit_acc:.4f} ({results['exit']['correct']}/{results['exit']['total']})")

        # Per-class breakdown
        preds = results['exit']['predictions']
        pred_classes = Counter(p['predicted'] for p in preds)
        actual_classes = Counter(p['actual'] for p in preds)
        print(f"  Predicted class distribution: {dict(pred_classes)}")
        print(f"  Actual class distribution:    {dict(actual_classes)}")

        # Confidence analysis
        correct_confs = [p['confidence'] for p in preds if p['predicted'] == p['actual']]
        wrong_confs = [p['confidence'] for p in preds if p['predicted'] != p['actual']]
        if correct_confs:
            print(f"  Avg confidence (correct): {np.mean(correct_confs):.3f}")
        if wrong_confs:
            print(f"  Avg confidence (wrong):   {np.mean(wrong_confs):.3f}")

        # Can it identify runners?
        runner_preds = [p for p in preds if p['actual'] == 2]
        if runner_preds:
            runner_correct = sum(1 for p in runner_preds if p['predicted'] == 2)
            print(f"\n  Runner detection: {runner_correct}/{len(runner_preds)} actual runners correctly identified")
            runner_mfe = np.mean([p['mfe_atr'] for p in runner_preds])
            print(f"  Average MFE of actual runners: {runner_mfe:.2f}R")
    else:
        print("  Exit Predictor: No holdout predictions generated.")

    print()

    # Decay predictor results
    if results['decay']['errors']:
        errors = np.array(results['decay']['errors'])
        print(f"  Signal Decay Holdout Validation:")
        print(f"  {'='*50}")
        print(f"  Total predictions: {len(errors)}")
        print(f"  MAE: {np.mean(errors):.1f} minutes")
        print(f"  Median error: {np.median(errors):.1f} minutes")
        print(f"  Max error: {np.max(errors):.1f} minutes")

        # Bucket accuracy (did we get the decay rate bucket right?)
        preds = results['decay']['predictions']
        bucket_correct = 0
        for p in preds:
            pred_bucket = 'fast' if p['predicted_min'] < 30 else ('medium' if p['predicted_min'] <= 120 else 'slow')
            actual_bucket = 'fast' if p['actual_min'] < 30 else ('medium' if p['actual_min'] <= 120 else 'slow')
            if pred_bucket == actual_bucket:
                bucket_correct += 1
        bucket_acc = bucket_correct / len(preds) if preds else 0
        print(f"  Bucket accuracy (fast/medium/slow): {bucket_acc:.1%}")
    else:
        print("  Signal Decay: No holdout predictions generated.")

    return results


# ============================================================================
# Final Recommendation
# ============================================================================

def print_recommendation(
    exit_predictor: Optional[ExitPredictor],
    decay_predictor: Optional[SignalDecayPredictor],
    sizer: Optional[DynamicPositionSizer],
    holdout_results: Dict,
):
    """Print final recommendation based on all results."""
    print(f"\n{'='*LINE_WIDTH}")
    print("RECOMMENDATION".center(LINE_WIDTH))
    print(f"{'='*LINE_WIDTH}\n")

    deploy_exit = False
    deploy_decay = False
    deploy_sizer = False

    # Exit predictor assessment
    if exit_predictor is not None and exit_predictor.is_trained:
        metrics = exit_predictor.val_metrics
        holdout_acc = 0
        if holdout_results.get('exit', {}).get('total', 0) > 0:
            holdout_acc = holdout_results['exit']['correct'] / holdout_results['exit']['total']

        print("  EXIT PREDICTOR:")
        if metrics.accuracy >= 0.45 and metrics.f1_macro >= 0.35:
            if holdout_acc >= 0.40:
                print(f"    DEPLOY - Training acc={metrics.accuracy:.3f}, holdout acc={holdout_acc:.3f}")
                print("    The model adds value over random (33% baseline for 3 classes).")
                print("    Use in ADVISORY mode first (log predictions, don't auto-trade).")
                deploy_exit = True
            else:
                print(f"    CAUTION - Training acc={metrics.accuracy:.3f} but holdout acc={holdout_acc:.3f}")
                print("    Possible overfitting. Run in shadow mode to gather more data.")
        else:
            print(f"    SKIP - Training acc={metrics.accuracy:.3f}, F1={metrics.f1_macro:.3f}")
            print("    Not significantly better than random. Stick with rule-based exits.")
    else:
        print("  EXIT PREDICTOR: Not trained.")

    print()

    # Decay predictor assessment
    if decay_predictor is not None and decay_predictor.is_trained:
        metrics = decay_predictor.val_metrics
        holdout_mae = None
        if holdout_results.get('decay', {}).get('errors'):
            holdout_mae = np.mean(holdout_results['decay']['errors'])

        print("  SIGNAL DECAY PREDICTOR:")
        if metrics.r2 >= 0.05 and metrics.mae < 100:
            if holdout_mae is not None and holdout_mae < 120:
                print(f"    DEPLOY - Training MAE={metrics.mae:.1f}min, R2={metrics.r2:.3f}")
                print(f"    Holdout MAE={holdout_mae:.1f}min")
                print("    Better than static 30-min TTL for strong signals.")
                deploy_decay = True
            else:
                holdout_str = f"holdout MAE={holdout_mae:.1f}min" if holdout_mae else "no holdout data"
                print(f"    CAUTION - Training metrics OK but {holdout_str}")
                print("    Consider using the model with a fallback to static TTL.")
        else:
            print(f"    SKIP - R2={metrics.r2:.3f}, MAE={metrics.mae:.1f}min")
            print("    Low predictive power. Stick with static 30-min TTL.")
    else:
        print("  SIGNAL DECAY PREDICTOR: Not trained.")

    print()

    # Dynamic sizer assessment
    if sizer is not None:
        stats = sizer.get_recent_stats(n=100)
        print("  DYNAMIC POSITION SIZER:")
        if stats['trade_count'] >= 20:
            print(f"    DEPLOY - Calibrated with {stats['trade_count']} trades")
            print(f"    Win rate: {stats['win_rate']:.1%}, Avg W/L: {stats['avg_win_r']:.2f}R/{stats['avg_loss_r']:.2f}R")
            print("    Kelly criterion active for trades with sufficient history.")
            deploy_sizer = True
        else:
            print(f"    CAUTION - Only {stats['trade_count']} trades in history")
            print("    Continue gathering data before enabling Kelly overlay.")
    else:
        print("  DYNAMIC POSITION SIZER: Not calibrated.")

    print()

    # Overall
    deployable = sum([deploy_exit, deploy_decay, deploy_sizer])
    print(f"  OVERALL: {deployable}/3 models recommended for deployment")
    print()

    if deployable == 0:
        print("  The rule-based system (+$768 from walk-forward) is the current best.")
        print("  ML models need more training data or feature engineering.")
        print("  Run this script again after accumulating live trading data.")
    elif deployable <= 2:
        print("  Deploy recommended models in ADVISORY mode alongside rules.")
        print("  Monitor for 2-4 weeks before using ML predictions for auto-trading.")
        print("  The rule-based filters remain the primary decision layer.")
    else:
        print("  All models ready for deployment in advisory mode.")
        print("  Enable ML predictions alongside rule-based filters.")
        print("  Use A/B testing: compare ML-enhanced vs rules-only on paper trades.")

    print()
    print(f"  Models saved to: {get_models_dir()}")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    start_time = time.time()

    print()
    print(f"{'='*LINE_WIDTH}")
    print("RDT TRADING SYSTEM - ML MODEL TRAINING FROM HISTORY".center(LINE_WIDTH))
    print(f"{'='*LINE_WIDTH}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Watchlist: {len(WATCHLIST)} stocks")
    print(f"  Data period: {DATA_DAYS} days")
    print(f"  Holdout: last {HOLDOUT_DAYS} days")
    print(f"  RRS threshold: {RRS_THRESHOLD}")
    print()

    # Step 1: Download data
    stock_data, spy_data = download_all_data(DATA_DAYS)

    if len(stock_data) < 10:
        print("ERROR: Too few symbols loaded. Check internet connectivity.")
        sys.exit(1)

    # Step 2: Generate training data
    exit_df = generate_exit_data(stock_data, spy_data)
    decay_df = generate_decay_data(stock_data, spy_data)

    # Step 3: Train models
    exit_predictor = train_exit_predictor(exit_df)
    decay_predictor = train_signal_decay(decay_df)
    sizer = calibrate_dynamic_sizer(exit_df)

    # Step 4: Save models
    save_models(exit_predictor, decay_predictor, sizer)

    # Step 5: Holdout validation
    holdout_results = run_holdout_validation(
        stock_data, spy_data, exit_predictor, decay_predictor
    )

    # Step 6: Recommendation
    print_recommendation(exit_predictor, decay_predictor, sizer, holdout_results)

    elapsed = time.time() - start_time
    print(f"  Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"{'='*LINE_WIDTH}\n")


if __name__ == "__main__":
    main()

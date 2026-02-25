#!/usr/bin/env python3
"""
Generate Training Data for ML Models

This script fetches historical data and creates labeled training data for the
trade success classifier models. It:
1. Downloads 2-3 years of daily/intraday data for watchlist symbols
2. Runs RRS signal detection on historical data
3. Simulates trade outcomes (did price reach 2R target within 10 days?)
4. Extracts features at signal time using feature_engineering.py
5. Saves labeled dataset to data/training/signals_labeled.csv

Usage:
    python scripts/generate_training_data.py [options]

Options:
    --symbols SYMBOL...     Symbols to include (default: from watchlist)
    --start-date DATE       Start date for data (default: 3 years ago)
    --end-date DATE         End date for data (default: today)
    --output PATH           Output file path
    --benchmark SYMBOL      Benchmark symbol for RRS (default: SPY)
    --min-rrs FLOAT         Minimum RRS threshold (default: 1.5)
    --target-r FLOAT        Target R-multiple (default: 2.0)
    --max-days INT          Max days for target (default: 10)
"""

# Fix curl_cffi chrome136 impersonation issue
from curl_cffi.requests import impersonate
impersonate.DEFAULT_CHROME = 'chrome110'
if hasattr(impersonate, 'REAL_TARGET_MAP'):
    impersonate.REAL_TARGET_MAP['chrome'] = 'chrome110'

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

# Import data providers
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available. Install with: pip install yfinance")

# Import RRS calculator
from shared.indicators.rrs import RRSCalculator, calculate_ema, calculate_sma


def setup_logging(verbose: bool = False):
    """Configure logging settings."""
    logger.remove()
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, format=log_format, level=level, colorize=True)

    # Also log to file
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file = log_dir / f"generate_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_file, format=log_format, level="DEBUG", rotation="10 MB")


def get_default_symbols() -> List[str]:
    """Get default symbols for training data generation."""
    # A diverse set of liquid stocks across sectors
    return [
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'CRM', 'INTC',
        # Financial
        'JPM', 'BAC', 'GS', 'V', 'MA',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV',
        # Consumer
        'WMT', 'HD', 'NKE', 'MCD', 'SBUX',
        # Industrial
        'CAT', 'BA', 'GE', 'UPS', 'HON',
        # Energy
        'XOM', 'CVX', 'COP',
        # ETFs
        'SPY', 'QQQ', 'IWM', 'DIA'
    ]


def fetch_historical_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    interval: str = '1d'
) -> Optional[pd.DataFrame]:
    """
    Fetch historical data for a symbol.

    Args:
        symbol: Stock ticker
        start_date: Start date for data
        end_date: End date for data
        interval: Data interval ('1d', '1h', etc.)

    Returns:
        DataFrame with OHLCV data or None if failed
    """
    if not YFINANCE_AVAILABLE:
        logger.error("yfinance not available")
        return None

    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True
        )

        if data.empty:
            logger.warning(f"No data retrieved for {symbol}")
            return None

        # Standardize column names
        data.columns = data.columns.str.lower()
        data = data.rename(columns={'stock splits': 'splits'})

        # Add symbol column
        data['symbol'] = symbol

        return data

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None


def calculate_rrs_signals(
    symbol_data: pd.DataFrame,
    benchmark_data: pd.DataFrame,
    rrs_calculator: RRSCalculator,
    min_rrs: float = 1.5
) -> pd.DataFrame:
    """
    Calculate RRS and identify signals.

    Args:
        symbol_data: OHLCV data for the symbol
        benchmark_data: OHLCV data for benchmark (SPY)
        rrs_calculator: RRS calculator instance
        min_rrs: Minimum RRS threshold for signals

    Returns:
        DataFrame with RRS values and signal indicators
    """
    try:
        # Calculate RRS
        symbol_close = symbol_data['close']
        benchmark_close = benchmark_data['close'].reindex(symbol_close.index, method='ffill')

        # Calculate returns
        symbol_returns = symbol_close.pct_change()
        benchmark_returns = benchmark_close.pct_change()

        # Calculate relative strength
        symbol_cumret = (1 + symbol_returns).cumprod()
        benchmark_cumret = (1 + benchmark_returns).cumprod()
        relative_strength = symbol_cumret / benchmark_cumret

        # Calculate RRS as ratio of relative strength EMAs
        rs_ema_fast = calculate_ema(relative_strength, 8)
        rs_ema_slow = calculate_ema(relative_strength, 21)
        rrs = rs_ema_fast / rs_ema_slow

        # Add to dataframe
        result = symbol_data.copy()
        result['rrs'] = rrs
        result['rrs_signal'] = (rrs > min_rrs).astype(int)

        # Determine direction based on price action
        result['ema_8'] = calculate_ema(result['close'], 8)
        result['ema_21'] = calculate_ema(result['close'], 21)
        result['direction'] = np.where(result['ema_8'] > result['ema_21'], 'long', 'short')

        # Calculate ATR for stop loss
        high_low = result['high'] - result['low']
        high_close = abs(result['high'] - result['close'].shift())
        low_close = abs(result['low'] - result['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result['atr'] = true_range.rolling(window=14).mean()
        result['atr_percent'] = result['atr'] / result['close'] * 100

        return result

    except Exception as e:
        logger.error(f"Error calculating RRS: {e}")
        return symbol_data


def simulate_trade_outcome(
    data: pd.DataFrame,
    signal_idx: int,
    direction: str,
    target_r: float = 2.0,
    max_days: int = 10
) -> Dict[str, Any]:
    """
    Simulate trade outcome from a signal.

    Args:
        data: DataFrame with OHLCV data
        signal_idx: Index of the signal
        direction: Trade direction ('long' or 'short')
        target_r: Target R-multiple
        max_days: Maximum days to hold

    Returns:
        Dict with trade outcome information
    """
    try:
        entry_row = data.iloc[signal_idx]
        entry_price = entry_row['close']
        atr = entry_row['atr']

        if pd.isna(atr) or atr <= 0:
            atr = entry_price * 0.02  # Default to 2%

        # Calculate stop loss and target
        if direction == 'long':
            stop_loss = entry_price - atr
            target = entry_price + (atr * target_r)
        else:
            stop_loss = entry_price + atr
            target = entry_price - (atr * target_r)

        # Simulate forward
        end_idx = min(signal_idx + max_days + 1, len(data))
        forward_data = data.iloc[signal_idx + 1:end_idx]

        success = False
        hit_stop = False
        exit_day = max_days
        exit_price = forward_data['close'].iloc[-1] if len(forward_data) > 0 else entry_price
        max_favorable = 0
        max_adverse = 0

        for i, (idx, row) in enumerate(forward_data.iterrows()):
            if direction == 'long':
                # Check for target hit (using high)
                if row['high'] >= target:
                    success = True
                    exit_day = i + 1
                    exit_price = target
                    break
                # Check for stop hit (using low)
                if row['low'] <= stop_loss:
                    hit_stop = True
                    exit_day = i + 1
                    exit_price = stop_loss
                    break
                # Track excursions
                max_favorable = max(max_favorable, (row['high'] - entry_price) / atr)
                max_adverse = max(max_adverse, (entry_price - row['low']) / atr)
            else:
                # Short trade
                if row['low'] <= target:
                    success = True
                    exit_day = i + 1
                    exit_price = target
                    break
                if row['high'] >= stop_loss:
                    hit_stop = True
                    exit_day = i + 1
                    exit_price = stop_loss
                    break
                max_favorable = max(max_favorable, (entry_price - row['low']) / atr)
                max_adverse = max(max_adverse, (row['high'] - entry_price) / atr)

        # Calculate P&L
        if direction == 'long':
            pnl = (exit_price - entry_price) / atr
        else:
            pnl = (entry_price - exit_price) / atr

        return {
            'success': int(success),
            'hit_stop': int(hit_stop),
            'exit_day': exit_day,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'stop_loss': stop_loss,
            'target': target,
            'atr': atr,
            'pnl_r': pnl,
            'max_favorable_r': max_favorable,
            'max_adverse_r': max_adverse
        }

    except Exception as e:
        logger.error(f"Error simulating trade: {e}")
        return {
            'success': 0, 'hit_stop': 0, 'exit_day': 0,
            'entry_price': 0, 'exit_price': 0, 'stop_loss': 0,
            'target': 0, 'atr': 0, 'pnl_r': 0,
            'max_favorable_r': 0, 'max_adverse_r': 0
        }


def extract_features(
    data: pd.DataFrame,
    signal_idx: int,
    benchmark_data: pd.DataFrame
) -> Dict[str, float]:
    """
    Extract features at signal time.

    Args:
        data: Symbol data with RRS and indicators
        signal_idx: Index of the signal
        benchmark_data: Benchmark (SPY) data

    Returns:
        Dict of features
    """
    try:
        row = data.iloc[signal_idx]

        # Get recent data for calculations
        lookback = min(signal_idx, 20)
        recent = data.iloc[signal_idx - lookback:signal_idx + 1]

        features = {
            # RRS features
            'rrs': row.get('rrs', 0),
            'rrs_3bar': data['rrs'].iloc[max(0, signal_idx-2):signal_idx+1].mean() if 'rrs' in data.columns else 0,
            'rrs_5bar': data['rrs'].iloc[max(0, signal_idx-4):signal_idx+1].mean() if 'rrs' in data.columns else 0,

            # ATR features
            'atr': row.get('atr', 0),
            'atr_percent': row.get('atr_percent', 0),

            # Price features
            'close': row['close'],
            'price_momentum_1': (row['close'] / data['close'].iloc[signal_idx - 1] - 1) * 100 if signal_idx > 0 else 0,
            'price_momentum_5': (row['close'] / data['close'].iloc[max(0, signal_idx - 5)] - 1) * 100,
            'price_momentum_20': (row['close'] / data['close'].iloc[max(0, signal_idx - 20)] - 1) * 100,

            # Volume features
            'volume': row.get('volume', 0),
            'volume_sma_20': recent['volume'].mean() if 'volume' in recent.columns else 0,
            'volume_ratio': row.get('volume', 0) / recent['volume'].mean() if recent['volume'].mean() > 0 else 1,

            # Volatility features
            'volatility_20': recent['close'].pct_change().std() * np.sqrt(252) if len(recent) > 1 else 0,
            'daily_range': (row['high'] - row['low']) / row['close'] * 100,

            # Trend features
            'ema_8': row.get('ema_8', row['close']),
            'ema_21': row.get('ema_21', row['close']),
            'ema_trend': 1 if row.get('ema_8', 0) > row.get('ema_21', 0) else 0,
            'price_vs_ema8': (row['close'] / row.get('ema_8', row['close']) - 1) * 100,
            'price_vs_ema21': (row['close'] / row.get('ema_21', row['close']) - 1) * 100,

            # RSI
            'rsi_14': calculate_rsi(recent['close'], 14),

            # Day of week
            'day_of_week': row.name.dayofweek if hasattr(row.name, 'dayofweek') else 0,
        }

        # Add benchmark features
        if benchmark_data is not None and row.name in benchmark_data.index:
            spy_row = benchmark_data.loc[row.name]
            features['spy_close'] = spy_row['close']
            features['spy_momentum_5'] = (
                (spy_row['close'] / benchmark_data['close'].iloc[
                    benchmark_data.index.get_loc(row.name) - 5
                ] - 1) * 100
            ) if benchmark_data.index.get_loc(row.name) > 5 else 0

        return features

    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return {}


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate RSI."""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    except Exception:
        return 50


def generate_training_data(
    symbols: List[str],
    benchmark: str,
    start_date: datetime,
    end_date: datetime,
    min_rrs: float,
    target_r: float,
    max_days: int
) -> pd.DataFrame:
    """
    Generate training data for all symbols.

    Returns:
        DataFrame with features and labels
    """
    rrs_calculator = RRSCalculator()
    all_data = []

    # Fetch benchmark data
    logger.info(f"Fetching benchmark data for {benchmark}")
    benchmark_data = fetch_historical_data(benchmark, start_date, end_date)
    if benchmark_data is None:
        logger.error("Failed to fetch benchmark data")
        return pd.DataFrame()

    total_signals = 0
    successes = 0

    for i, symbol in enumerate(symbols):
        logger.info(f"Processing {symbol} ({i+1}/{len(symbols)})")

        # Fetch symbol data
        data = fetch_historical_data(symbol, start_date, end_date)
        if data is None:
            continue

        # Calculate RRS and signals
        data = calculate_rrs_signals(data, benchmark_data, rrs_calculator, min_rrs)

        # Find signals
        signals = data[data['rrs_signal'] == 1].index
        logger.info(f"  Found {len(signals)} signals for {symbol}")

        for signal_date in signals:
            signal_idx = data.index.get_loc(signal_date)

            # Skip if not enough forward data
            if signal_idx >= len(data) - max_days - 1:
                continue

            row = data.iloc[signal_idx]
            direction = row['direction']

            # Simulate trade outcome
            outcome = simulate_trade_outcome(data, signal_idx, direction, target_r, max_days)

            # Extract features
            features = extract_features(data, signal_idx, benchmark_data)

            if not features:
                continue

            # Combine into record
            record = {
                'symbol': symbol,
                'date': signal_date,
                'direction': direction,
                **features,
                **outcome
            }

            all_data.append(record)
            total_signals += 1
            if outcome['success']:
                successes += 1

        # Rate limiting
        time.sleep(0.1)

    df = pd.DataFrame(all_data)

    logger.info(f"\nGenerated {len(df)} training samples")
    logger.info(f"Total signals: {total_signals}")
    logger.info(f"Successes: {successes} ({successes/total_signals*100:.1f}% win rate)" if total_signals > 0 else "")

    return df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate training data for ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        default=None,
        help='Symbols to include (default: use default list)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date (YYYY-MM-DD, default: 3 years ago)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD, default: today)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: data/training/signals_labeled.csv)'
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        default='SPY',
        help='Benchmark symbol (default: SPY)'
    )
    parser.add_argument(
        '--min-rrs',
        type=float,
        default=1.5,
        help='Minimum RRS threshold (default: 1.5)'
    )
    parser.add_argument(
        '--target-r',
        type=float,
        default=2.0,
        help='Target R-multiple (default: 2.0)'
    )
    parser.add_argument(
        '--max-days',
        type=int,
        default=10,
        help='Maximum days to reach target (default: 10)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    logger.info("=" * 80)
    logger.info("Training Data Generation")
    logger.info("=" * 80)

    # Parse dates
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start_date = datetime.now() - timedelta(days=365 * 3)

    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now()

    # Get symbols
    symbols = args.symbols or get_default_symbols()

    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Benchmark: {args.benchmark}")
    logger.info(f"Min RRS: {args.min_rrs}")
    logger.info(f"Target R: {args.target_r}")
    logger.info(f"Max days: {args.max_days}")
    logger.info("=" * 80)

    # Generate training data
    df = generate_training_data(
        symbols=symbols,
        benchmark=args.benchmark,
        start_date=start_date,
        end_date=end_date,
        min_rrs=args.min_rrs,
        target_r=args.target_r,
        max_days=args.max_days
    )

    if df.empty:
        logger.error("No training data generated")
        return 1

    # Save to file
    output_path = args.output or str(
        Path(__file__).parent.parent / "data" / "training" / "signals_labeled.csv"
    )

    # Create directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.success(f"Saved training data to {output_path}")

    # Print summary statistics
    logger.info("\nDataset Summary:")
    logger.info(f"  Total samples: {len(df)}")
    logger.info(f"  Features: {len(df.columns)}")
    logger.info(f"  Success rate: {df['success'].mean()*100:.1f}%")
    logger.info(f"  Symbols: {df['symbol'].nunique()}")
    logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    logger.info("\nClass distribution:")
    logger.info(df['success'].value_counts())

    logger.info("\nFeature columns:")
    feature_cols = [c for c in df.columns if c not in ['symbol', 'date', 'direction', 'success',
                                                        'hit_stop', 'exit_day', 'entry_price',
                                                        'exit_price', 'stop_loss', 'target',
                                                        'pnl_r', 'max_favorable_r', 'max_adverse_r']]
    for col in feature_cols:
        logger.info(f"  {col}")

    logger.info("\n" + "=" * 80)
    logger.success("Training data generation completed!")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())

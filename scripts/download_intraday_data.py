#!/usr/bin/env python3
"""
Download 1-year of 1-minute intraday data from Alpaca for backtesting.

Stores data as parquet files in data/intraday_cache/ for fast loading.

Usage:
    cd /home/user0/rdt-trading-system
    python scripts/download_intraday_data.py
"""

import sys
import time
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Config
API_KEY = "PKRCOVT5A3FMF72KQ4JFM6K5FJ"
API_SECRET = "5rbWWK2tV4KPhXCYbSYrg4WNbN3v14spptribHdVas2w"

SYMBOLS = [
    'SPY',  # Must be first — needed for RRS
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    'JPM', 'V', 'JNJ', 'UNH', 'HD', 'PG', 'MA', 'DIS',
    'PYPL', 'ADBE', 'CRM', 'NFLX', 'INTC', 'AMD', 'CSCO',
    'PEP', 'KO', 'MRK', 'ABT', 'TMO', 'COST', 'AVGO', 'TXN',
    # Intermarket ETFs
    'TLT', 'UUP', 'GLD', 'IWM',
    # VIX proxy (UVXY since Alpaca doesn't have ^VIX)
    'UVXY',
    # Sector ETFs
    'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLRE', 'XLC', 'XLU',
]

CACHE_DIR = PROJECT_ROOT / "data" / "intraday_cache"
LOOKBACK_DAYS = 365  # 1 year
CHUNK_DAYS = 7  # Download in 1-week chunks (Alpaca handles large requests but chunking is safer)


def download_symbol(client, symbol, start_date, end_date):
    """Download 1-min bars for a symbol in weekly chunks."""
    all_bars = []
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=CHUNK_DAYS), end_date)

        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=current_start,
                end=current_end,
            )
            bars = client.get_stock_bars(req)
            symbol_bars = bars.data.get(symbol, [])

            for bar in symbol_bars:
                all_bars.append({
                    'timestamp': bar.timestamp,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume),
                    'vwap': float(bar.vwap) if bar.vwap else None,
                    'trade_count': int(bar.trade_count) if bar.trade_count else None,
                })
        except Exception as e:
            logger.warning(f"  {symbol} chunk {current_start} failed: {e}")

        current_start = current_end
        time.sleep(0.2)  # Gentle rate limiting

    if not all_bars:
        return None

    df = pd.DataFrame(all_bars)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    # Remove duplicate indices
    df = df[~df.index.duplicated(keep='first')]

    return df


def main():
    print()
    print("=" * 80)
    print("ALPACA INTRADAY DATA DOWNLOADER".center(80))
    print("=" * 80)
    print()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    client = StockHistoricalDataClient(API_KEY, API_SECRET)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)

    print(f"Date range: {start_date.date()} to {end_date.date()} ({LOOKBACK_DAYS} days)")
    print(f"Symbols: {len(SYMBOLS)}")
    print(f"Cache dir: {CACHE_DIR}")
    print()

    total_bars = 0
    success = 0
    failed = []

    for i, symbol in enumerate(SYMBOLS):
        cache_file = CACHE_DIR / f"{symbol}_1min.parquet"

        # Skip if already cached and recent
        if cache_file.exists():
            mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            age_hours = (datetime.now() - mod_time).total_seconds() / 3600
            if age_hours < 12:
                existing = pd.read_parquet(cache_file)
                total_bars += len(existing)
                success += 1
                logger.info(f"[{i+1}/{len(SYMBOLS)}] {symbol}: cached ({len(existing):,} bars, {age_hours:.0f}h old)")
                continue

        logger.info(f"[{i+1}/{len(SYMBOLS)}] {symbol}: downloading...")
        t0 = time.time()

        df = download_symbol(client, symbol, start_date, end_date)

        if df is not None and len(df) > 0:
            df.to_parquet(cache_file)
            elapsed = time.time() - t0
            total_bars += len(df)
            success += 1

            # Stats
            trading_days = df.index.normalize().nunique()
            logger.info(f"  {len(df):>8,} bars | {trading_days} days | {elapsed:.1f}s")
        else:
            failed.append(symbol)
            logger.warning(f"  FAILED — no data returned")

    print()
    print("=" * 80)
    print("DOWNLOAD SUMMARY".center(80))
    print("=" * 80)
    print()
    print(f"  Symbols downloaded: {success}/{len(SYMBOLS)}")
    print(f"  Total bars: {total_bars:,}")
    print(f"  Cache directory: {CACHE_DIR}")

    if failed:
        print(f"  Failed: {', '.join(failed)}")

    # Show per-symbol stats
    print()
    print(f"  {'Symbol':<8} {'Bars':>10} {'Days':>6} {'First Date':>14} {'Last Date':>14}")
    print(f"  {'-'*56}")

    for symbol in SYMBOLS:
        cache_file = CACHE_DIR / f"{symbol}_1min.parquet"
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            days = df.index.normalize().nunique()
            print(f"  {symbol:<8} {len(df):>10,} {days:>6} {str(df.index[0].date()):>14} {str(df.index[-1].date()):>14}")

    print()

    # Estimate storage
    total_size = sum(f.stat().st_size for f in CACHE_DIR.glob("*.parquet"))
    print(f"  Total disk usage: {total_size / 1024 / 1024:.1f} MB")
    print()


if __name__ == "__main__":
    main()

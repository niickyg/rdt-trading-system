"""
Data Loader for Backtesting
Fetches and prepares historical data
"""

import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger


class DataLoader:
    """
    Load historical data for backtesting

    Supports:
    - Yahoo Finance API
    - Local CSV files
    - Cached data for repeated runs
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize data loader

        Args:
            cache_dir: Directory to cache downloaded data
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

    def load_stock_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load historical stock data

        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cached data

        Returns:
            Dict of symbol -> DataFrame with OHLCV data
        """
        data = {}

        for symbol in symbols:
            try:
                df = self._load_symbol(symbol, start_date, end_date, use_cache)
                if df is not None and len(df) > 0:
                    data[symbol] = df
                    logger.debug(f"Loaded {symbol}: {len(df)} days")
            except Exception as e:
                logger.warning(f"Failed to load {symbol}: {e}")

        logger.info(f"Loaded data for {len(data)}/{len(symbols)} symbols")
        return data

    def _load_symbol(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        use_cache: bool
    ) -> Optional[pd.DataFrame]:
        """Load data for a single symbol"""
        # Try cache first
        if use_cache and self.cache_dir:
            cache_file = self.cache_dir / f"{symbol}_{start_date}_{end_date}.parquet"
            if cache_file.exists():
                logger.debug(f"Loading {symbol} from cache")
                return pd.read_parquet(cache_file)

        # Download from Yahoo Finance
        logger.debug(f"Downloading {symbol} from Yahoo Finance")
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date,
            end=end_date + timedelta(days=1),
            auto_adjust=True
        )

        if len(df) == 0:
            return None

        # Ensure column names are consistent
        df.columns = [c.capitalize() if c.lower() in ['open', 'high', 'low', 'close', 'volume'] else c for c in df.columns]

        # Cache the data
        if self.cache_dir:
            cache_file = self.cache_dir / f"{symbol}_{start_date}_{end_date}.parquet"
            df.to_parquet(cache_file)

        return df

    def load_spy_data(
        self,
        start_date: date,
        end_date: date,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Load SPY benchmark data"""
        df = self._load_symbol("SPY", start_date, end_date, use_cache)
        if df is None:
            raise ValueError("Failed to load SPY data")
        return df

    def load_from_csv(
        self,
        file_path: str,
        date_column: str = "Date"
    ) -> pd.DataFrame:
        """
        Load data from CSV file

        Args:
            file_path: Path to CSV file
            date_column: Name of date column

        Returns:
            DataFrame with DatetimeIndex
        """
        df = pd.read_csv(file_path, parse_dates=[date_column])
        df.set_index(date_column, inplace=True)
        df.sort_index(inplace=True)
        return df

    def clear_cache(self):
        """Clear all cached data"""
        if self.cache_dir and self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cache cleared")


def load_default_watchlist() -> List[str]:
    """Load default watchlist for backtesting"""
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
        'JPM', 'V', 'JNJ', 'UNH', 'HD', 'PG', 'MA', 'DIS',
        'PYPL', 'ADBE', 'CRM', 'NFLX', 'INTC', 'AMD', 'CSCO',
        'PEP', 'KO', 'MRK', 'ABT', 'TMO', 'COST', 'AVGO', 'TXN'
    ]

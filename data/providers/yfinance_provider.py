"""
YFinance Data Provider
Wrapper around yfinance library for market data.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

# Fix curl_cffi chrome136 impersonation issue in Docker
try:
    from curl_cffi.requests import impersonate
    impersonate.DEFAULT_CHROME = 'chrome110'
    if hasattr(impersonate, 'REAL_TARGET_MAP'):
        impersonate.REAL_TARGET_MAP['chrome'] = 'chrome110'
except ImportError:
    pass

import yfinance as yf
from loguru import logger

from data.providers.base import (
    DataProvider,
    Quote,
    HistoricalData,
    ProviderError,
    RateLimitError,
    DataNotFoundError,
)


class YFinanceProvider(DataProvider):
    """
    Data provider using yfinance library.

    YFinance provides free, unlimited access to Yahoo Finance data.
    It supports both real-time quotes and historical data.
    """

    def __init__(self, priority: int = 10, retry_delay: float = 1.0, max_retries: int = 3):
        """
        Initialize YFinance provider.

        Args:
            priority: Provider priority (lower = higher priority)
            retry_delay: Base delay between retries in seconds
            max_retries: Maximum number of retry attempts
        """
        super().__init__(name="yfinance", priority=priority)
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self._requests_limit = None  # Unlimited

        logger.info(f"YFinanceProvider initialized with priority={priority}")

    def _handle_yfinance_error(self, error: Exception) -> None:
        """
        Handle yfinance-specific errors and convert to standard exceptions.

        Args:
            error: Original exception from yfinance

        Raises:
            RateLimitError: If rate limited
            DataNotFoundError: If symbol not found
            ProviderError: For other errors
        """
        error_msg = str(error).lower()

        if "401" in error_msg or "rate" in error_msg or "too many" in error_msg:
            raise RateLimitError(f"YFinance rate limited: {error}")
        elif "no data" in error_msg or "not found" in error_msg or "delisted" in error_msg:
            raise DataNotFoundError(f"Symbol not found: {error}")
        else:
            raise ProviderError(f"YFinance error: {error}")

    def _fetch_with_retry(self, fetch_func, *args, **kwargs):
        """
        Execute a fetch function with retry logic.

        Args:
            fetch_func: Function to execute
            *args: Positional arguments for fetch_func
            **kwargs: Keyword arguments for fetch_func

        Returns:
            Result from fetch_func

        Raises:
            ProviderError: If all retries fail
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                result = fetch_func(*args, **kwargs)
                self.record_success()
                return result
            except RateLimitError as e:
                last_error = e
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"YFinance rate limited, waiting {wait_time}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(wait_time)
            except (DataNotFoundError, ProviderError) as e:
                # Don't retry for data not found
                self.record_failure(str(e))
                raise
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"YFinance error: {e}, retrying in {wait_time}s")
                    time.sleep(wait_time)

        self.record_failure(str(last_error))
        self._handle_yfinance_error(last_error)

    def get_quote(self, symbol: str) -> Quote:
        """
        Get current quote for a symbol using yfinance.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Quote object with current market data

        Raises:
            ProviderError: If quote cannot be retrieved
        """
        return self._fetch_with_retry(self._fetch_quote, symbol)

    def _fetch_quote(self, symbol: str) -> Quote:
        """Internal method to fetch a single quote."""
        try:
            ticker = yf.Ticker(symbol)

            # Get intraday data for current price
            intraday = ticker.history(period="1d", interval="5m")
            if intraday.empty:
                raise DataNotFoundError(f"No intraday data for {symbol}")

            # Get daily data for previous close
            daily = ticker.history(period="5d", interval="1d")
            if daily.empty:
                raise DataNotFoundError(f"No daily data for {symbol}")

            # Normalize column names
            intraday.columns = [c.lower() for c in intraday.columns]
            daily.columns = [c.lower() for c in daily.columns]

            current_price = float(intraday["close"].iloc[-1])
            previous_close = float(daily["close"].iloc[-2]) if len(daily) >= 2 else current_price
            change = current_price - previous_close
            change_percent = (change / previous_close * 100) if previous_close != 0 else 0

            return Quote(
                symbol=symbol,
                price=current_price,
                open=float(intraday["open"].iloc[0]),
                high=float(intraday["high"].max()),
                low=float(intraday["low"].min()),
                volume=int(intraday["volume"].sum()),
                previous_close=previous_close,
                change=change,
                change_percent=change_percent,
                timestamp=datetime.now(),
                provider=self.name,
            )

        except (DataNotFoundError, ProviderError):
            raise
        except Exception as e:
            self._handle_yfinance_error(e)

    def get_historical(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> HistoricalData:
        """
        Get historical OHLCV data using yfinance.

        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            interval: Data interval (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)

        Returns:
            HistoricalData object with OHLCV DataFrame

        Raises:
            ProviderError: If historical data cannot be retrieved
        """
        return self._fetch_with_retry(self._fetch_historical, symbol, period, interval)

    def _fetch_historical(self, symbol: str, period: str, interval: str) -> HistoricalData:
        """Internal method to fetch historical data."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                raise DataNotFoundError(f"No historical data for {symbol}")

            # Keep only OHLCV columns
            ohlcv_columns = ["Open", "High", "Low", "Close", "Volume"]
            available_columns = [c for c in ohlcv_columns if c in df.columns]
            df = df[available_columns].copy()

            return HistoricalData(
                symbol=symbol,
                data=df,
                period=period,
                interval=interval,
                provider=self.name,
            )

        except (DataNotFoundError, ProviderError):
            raise
        except Exception as e:
            self._handle_yfinance_error(e)

    def get_batch_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """
        Get quotes for multiple symbols using yfinance batch download.

        This is more efficient than fetching individual quotes.

        Args:
            symbols: List of stock ticker symbols

        Returns:
            Dictionary mapping symbols to Quote objects

        Raises:
            ProviderError: If batch quotes cannot be retrieved
        """
        return self._fetch_with_retry(self._fetch_batch_quotes, symbols)

    def _fetch_batch_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """Internal method to fetch batch quotes."""
        try:
            # Batch download intraday data
            batch_intraday = yf.download(
                symbols,
                period="1d",
                interval="5m",
                group_by="ticker",
                progress=False,
                threads=True,
            )

            # Batch download daily data for previous close
            batch_daily = yf.download(
                symbols,
                period="5d",
                interval="1d",
                group_by="ticker",
                progress=False,
                threads=True,
            )

            if batch_intraday.empty:
                raise DataNotFoundError("No intraday data returned from batch download")

            quotes = {}
            now = datetime.now()

            for symbol in symbols:
                try:
                    # Handle single vs multi-symbol batch results
                    if len(symbols) == 1:
                        intraday = batch_intraday
                        daily = batch_daily
                    else:
                        if symbol not in batch_intraday.columns.get_level_values(0):
                            continue
                        intraday = batch_intraday[symbol].dropna(how="all")
                        daily = batch_daily[symbol].dropna(how="all")

                    if intraday.empty or daily.empty:
                        continue

                    # Normalize column names
                    intraday.columns = [c.lower() for c in intraday.columns]
                    daily.columns = [c.lower() for c in daily.columns]

                    current_price = float(intraday["close"].iloc[-1])
                    previous_close = float(daily["close"].iloc[-2]) if len(daily) >= 2 else current_price
                    change = current_price - previous_close
                    change_percent = (change / previous_close * 100) if previous_close != 0 else 0

                    quotes[symbol] = Quote(
                        symbol=symbol,
                        price=current_price,
                        open=float(intraday["open"].iloc[0]),
                        high=float(intraday["high"].max()),
                        low=float(intraday["low"].min()),
                        volume=int(intraday["volume"].sum()),
                        previous_close=previous_close,
                        change=change,
                        change_percent=change_percent,
                        timestamp=now,
                        provider=self.name,
                    )

                except Exception as e:
                    logger.debug(f"Error extracting {symbol} from batch: {e}")
                    continue

            if not quotes:
                raise DataNotFoundError("No valid quotes extracted from batch download")

            return quotes

        except (DataNotFoundError, ProviderError):
            raise
        except Exception as e:
            self._handle_yfinance_error(e)

    def get_batch_historical(
        self,
        symbols: List[str],
        period: str = "60d",
        interval: str = "1d"
    ) -> Dict[str, HistoricalData]:
        """
        Get historical data for multiple symbols using batch download.

        Args:
            symbols: List of stock ticker symbols
            period: Time period
            interval: Data interval

        Returns:
            Dictionary mapping symbols to HistoricalData objects
        """
        return self._fetch_with_retry(self._fetch_batch_historical, symbols, period, interval)

    def _fetch_batch_historical(
        self,
        symbols: List[str],
        period: str,
        interval: str
    ) -> Dict[str, HistoricalData]:
        """Internal method to fetch batch historical data."""
        try:
            batch_data = yf.download(
                symbols,
                period=period,
                interval=interval,
                group_by="ticker",
                progress=False,
                threads=True,
            )

            if batch_data.empty:
                raise DataNotFoundError("No historical data returned from batch download")

            result = {}
            now = datetime.now()

            for symbol in symbols:
                try:
                    if len(symbols) == 1:
                        df = batch_data
                    else:
                        if symbol not in batch_data.columns.get_level_values(0):
                            continue
                        df = batch_data[symbol].dropna(how="all")

                    if df.empty:
                        continue

                    # Keep only OHLCV columns
                    ohlcv_columns = ["Open", "High", "Low", "Close", "Volume"]
                    available_columns = [c for c in ohlcv_columns if c in df.columns]
                    df = df[available_columns].copy()

                    result[symbol] = HistoricalData(
                        symbol=symbol,
                        data=df,
                        period=period,
                        interval=interval,
                        provider=self.name,
                        fetched_at=now,
                    )

                except Exception as e:
                    logger.debug(f"Error extracting {symbol} historical from batch: {e}")
                    continue

            return result

        except (DataNotFoundError, ProviderError):
            raise
        except Exception as e:
            self._handle_yfinance_error(e)

    def is_available(self) -> bool:
        """
        Check if yfinance is available by making a simple request.

        Returns:
            True if yfinance is responding
        """
        try:
            # Quick check with a common symbol
            ticker = yf.Ticker("SPY")
            info = ticker.fast_info
            return info is not None
        except Exception as e:
            logger.debug(f"YFinance availability check failed: {e}")
            return False

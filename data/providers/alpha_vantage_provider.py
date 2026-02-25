"""
Alpha Vantage Data Provider
Implementation of Alpha Vantage API for market data.

Free tier: 25 requests/day
Premium: Higher limits based on subscription
"""

import os
import time
from datetime import datetime, date
from typing import Dict, List, Optional

import pandas as pd
import requests
from loguru import logger

from data.providers.base import (
    DataProvider,
    Quote,
    HistoricalData,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    DataNotFoundError,
)


class AlphaVantageProvider(DataProvider):
    """
    Data provider using Alpha Vantage API.

    Supports both free tier (25 requests/day) and premium subscriptions.
    API documentation: https://www.alphavantage.co/documentation/
    """

    BASE_URL = "https://www.alphavantage.co/query"

    # Mapping of standard periods to Alpha Vantage output size
    PERIOD_MAPPING = {
        "1d": "compact",
        "5d": "compact",
        "1mo": "compact",
        "60d": "compact",
        "3mo": "compact",
        "6mo": "full",
        "1y": "full",
        "2y": "full",
        "5y": "full",
        "max": "full",
    }

    # Mapping of standard intervals to Alpha Vantage functions
    INTERVAL_MAPPING = {
        "1m": ("TIME_SERIES_INTRADAY", "1min"),
        "5m": ("TIME_SERIES_INTRADAY", "5min"),
        "15m": ("TIME_SERIES_INTRADAY", "15min"),
        "30m": ("TIME_SERIES_INTRADAY", "30min"),
        "1h": ("TIME_SERIES_INTRADAY", "60min"),
        "1d": ("TIME_SERIES_DAILY", None),
        "1wk": ("TIME_SERIES_WEEKLY", None),
        "1mo": ("TIME_SERIES_MONTHLY", None),
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        priority: int = 20,
        requests_per_minute: int = 5,
        daily_limit: int = 25,
        premium: bool = False,
    ):
        """
        Initialize Alpha Vantage provider.

        Args:
            api_key: Alpha Vantage API key (or from ALPHA_VANTAGE_API_KEY env var)
            priority: Provider priority (lower = higher priority)
            requests_per_minute: Rate limit per minute
            daily_limit: Maximum requests per day (25 for free, higher for premium)
            premium: Whether using premium subscription
        """
        super().__init__(name="alpha_vantage", priority=priority)

        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        self.requests_per_minute = requests_per_minute
        self.premium = premium

        # Set limits based on subscription
        if premium:
            self._requests_limit = None  # Effectively unlimited for premium
            self.daily_limit = 500  # Common premium limit
        else:
            self._requests_limit = daily_limit
            self.daily_limit = daily_limit

        self._last_request_time: Optional[datetime] = None
        self._request_times: List[datetime] = []
        self._daily_request_count = 0
        self._daily_reset_date = date.today()

        if self.api_key:
            logger.info(f"AlphaVantageProvider initialized (premium={premium}, limit={self.daily_limit}/day)")
        else:
            logger.warning("AlphaVantageProvider initialized WITHOUT API key - will fail on requests")

    def _check_rate_limit(self) -> None:
        """
        Check and enforce rate limits.

        Raises:
            RateLimitError: If rate limit exceeded
        """
        now = datetime.now()
        today = date.today()

        # Reset daily counter if new day
        if today > self._daily_reset_date:
            self._daily_request_count = 0
            self._daily_reset_date = today
            self.reset_daily_counter()

        # Check daily limit
        if not self.premium and self._daily_request_count >= self.daily_limit:
            raise RateLimitError(
                f"Alpha Vantage daily limit reached ({self.daily_limit} requests). "
                "Resets at midnight. Consider upgrading to premium."
            )

        # Enforce per-minute rate limit
        one_minute_ago = now.timestamp() - 60
        self._request_times = [t for t in self._request_times if t.timestamp() > one_minute_ago]

        if len(self._request_times) >= self.requests_per_minute:
            wait_time = 60 - (now.timestamp() - self._request_times[0].timestamp())
            if wait_time > 0:
                logger.debug(f"Alpha Vantage rate limit, waiting {wait_time:.1f}s")
                time.sleep(wait_time)

    def _record_request(self) -> None:
        """Record that a request was made for rate limiting."""
        now = datetime.now()
        self._request_times.append(now)
        self._last_request_time = now
        self._daily_request_count += 1

    def _make_request(self, params: Dict) -> Dict:
        """
        Make an API request to Alpha Vantage.

        Args:
            params: Query parameters

        Returns:
            JSON response as dictionary

        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limited
            ProviderError: For other errors
        """
        if not self.api_key:
            raise AuthenticationError("Alpha Vantage API key not configured")

        self._check_rate_limit()

        params["apikey"] = self.api_key

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            self._record_request()

            if response.status_code == 401:
                raise AuthenticationError("Invalid Alpha Vantage API key")

            if response.status_code == 429:
                raise RateLimitError("Alpha Vantage rate limit exceeded")

            if response.status_code != 200:
                raise ProviderError(f"Alpha Vantage HTTP error: {response.status_code}")

            data = response.json()

            # Check for API error messages in response
            if "Error Message" in data:
                raise DataNotFoundError(data["Error Message"])

            if "Note" in data and "rate limit" in data["Note"].lower():
                raise RateLimitError(data["Note"])

            if "Information" in data and "rate limit" in data["Information"].lower():
                raise RateLimitError(data["Information"])

            return data

        except (AuthenticationError, RateLimitError, DataNotFoundError):
            raise
        except requests.exceptions.Timeout:
            raise ProviderError("Alpha Vantage request timed out")
        except requests.exceptions.RequestException as e:
            raise ProviderError(f"Alpha Vantage request failed: {e}")

    def get_quote(self, symbol: str) -> Quote:
        """
        Get current quote for a symbol using Alpha Vantage GLOBAL_QUOTE.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Quote object with current market data

        Raises:
            ProviderError: If quote cannot be retrieved
        """
        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
            }

            data = self._make_request(params)

            if "Global Quote" not in data or not data["Global Quote"]:
                raise DataNotFoundError(f"No quote data for {symbol}")

            quote_data = data["Global Quote"]

            # Parse Alpha Vantage response format
            price = float(quote_data.get("05. price", 0))
            open_price = float(quote_data.get("02. open", price))
            high = float(quote_data.get("03. high", price))
            low = float(quote_data.get("04. low", price))
            volume = int(quote_data.get("06. volume", 0))
            previous_close = float(quote_data.get("08. previous close", price))
            change = float(quote_data.get("09. change", 0))
            change_percent_str = quote_data.get("10. change percent", "0%")
            change_percent = float(change_percent_str.replace("%", ""))

            self.record_success()

            return Quote(
                symbol=symbol,
                price=price,
                open=open_price,
                high=high,
                low=low,
                volume=volume,
                previous_close=previous_close,
                change=change,
                change_percent=change_percent,
                timestamp=datetime.now(),
                provider=self.name,
            )

        except (DataNotFoundError, RateLimitError, AuthenticationError):
            self.record_failure(str(symbol))
            raise
        except Exception as e:
            self.record_failure(str(e))
            raise ProviderError(f"Alpha Vantage error getting quote for {symbol}: {e}")

    def get_historical(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> HistoricalData:
        """
        Get historical OHLCV data using Alpha Vantage.

        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)

        Returns:
            HistoricalData object with OHLCV DataFrame

        Raises:
            ProviderError: If historical data cannot be retrieved
        """
        try:
            # Map interval to Alpha Vantage function
            if interval not in self.INTERVAL_MAPPING:
                raise ProviderError(f"Unsupported interval: {interval}")

            function, av_interval = self.INTERVAL_MAPPING[interval]
            output_size = self.PERIOD_MAPPING.get(period, "compact")

            params = {
                "function": function,
                "symbol": symbol,
                "outputsize": output_size,
            }

            if av_interval:
                params["interval"] = av_interval

            data = self._make_request(params)

            # Find the time series key in response
            ts_key = None
            for key in data.keys():
                if "Time Series" in key or "Weekly" in key or "Monthly" in key:
                    ts_key = key
                    break

            if not ts_key or not data[ts_key]:
                raise DataNotFoundError(f"No historical data for {symbol}")

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data[ts_key], orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Rename columns to standard format
            column_mapping = {
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume",
                "5. adjusted close": "adj_close",
                "6. volume": "volume",
            }

            df = df.rename(columns=column_mapping)

            # Convert to numeric
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Keep only OHLCV
            available_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            df = df[available_cols]

            # Filter by period
            df = self._filter_by_period(df, period)

            self.record_success()

            return HistoricalData(
                symbol=symbol,
                data=df,
                period=period,
                interval=interval,
                provider=self.name,
            )

        except (DataNotFoundError, RateLimitError, AuthenticationError):
            self.record_failure(str(symbol))
            raise
        except Exception as e:
            self.record_failure(str(e))
            raise ProviderError(f"Alpha Vantage error getting historical for {symbol}: {e}")

    def _filter_by_period(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """
        Filter DataFrame to match requested period.

        Args:
            df: DataFrame with datetime index
            period: Period string

        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df

        now = datetime.now()

        period_days = {
            "1d": 1,
            "5d": 5,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
            "max": None,
        }

        days = period_days.get(period)
        if days is None:
            return df

        cutoff = now - pd.Timedelta(days=days)
        return df[df.index >= cutoff]

    def get_batch_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """
        Get quotes for multiple symbols.

        Note: Alpha Vantage doesn't have a true batch endpoint for quotes,
        so this makes individual requests with rate limiting.

        Args:
            symbols: List of stock ticker symbols

        Returns:
            Dictionary mapping symbols to Quote objects

        Raises:
            ProviderError: If batch quotes cannot be retrieved
        """
        quotes = {}
        errors = []

        for symbol in symbols:
            try:
                quotes[symbol] = self.get_quote(symbol)
            except RateLimitError:
                # Stop if rate limited
                logger.warning(f"Alpha Vantage rate limited, got {len(quotes)}/{len(symbols)} quotes")
                break
            except DataNotFoundError:
                logger.debug(f"No data for {symbol}")
                continue
            except Exception as e:
                errors.append(f"{symbol}: {e}")
                continue

        if not quotes and errors:
            raise ProviderError(f"Alpha Vantage batch quotes failed: {'; '.join(errors)}")

        return quotes

    def is_available(self) -> bool:
        """
        Check if Alpha Vantage is available.

        Returns:
            True if API key is configured and not rate limited
        """
        if not self.api_key:
            return False

        # Check if daily limit reached
        today = date.today()
        if today > self._daily_reset_date:
            self._daily_request_count = 0
            self._daily_reset_date = today

        if not self.premium and self._daily_request_count >= self.daily_limit:
            return False

        return True

    def get_remaining_requests(self) -> int:
        """
        Get number of remaining requests for today.

        Returns:
            Remaining request count (or -1 for unlimited)
        """
        if self.premium:
            return -1

        today = date.today()
        if today > self._daily_reset_date:
            return self.daily_limit

        return max(0, self.daily_limit - self._daily_request_count)

    def get_intraday_extended(
        self,
        symbol: str,
        interval: str = "5m",
        month: Optional[str] = None
    ) -> HistoricalData:
        """
        Get extended intraday historical data (premium feature).

        Args:
            symbol: Stock ticker symbol
            interval: Data interval (1m, 5m, 15m, 30m, 60m)
            month: Month in format "year-month" (e.g., "2024-01")

        Returns:
            HistoricalData object

        Raises:
            ProviderError: If data cannot be retrieved
        """
        if not self.premium:
            raise ProviderError("Extended intraday requires premium subscription")

        try:
            av_interval_map = {
                "1m": "1min",
                "5m": "5min",
                "15m": "15min",
                "30m": "30min",
                "1h": "60min",
            }

            params = {
                "function": "TIME_SERIES_INTRADAY_EXTENDED",
                "symbol": symbol,
                "interval": av_interval_map.get(interval, "5min"),
            }

            if month:
                params["month"] = month

            data = self._make_request(params)

            # Parse CSV response for extended data
            df = pd.DataFrame(data)
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time").sort_index()

            # Rename and convert columns
            df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            self.record_success()

            return HistoricalData(
                symbol=symbol,
                data=df,
                period=month or "extended",
                interval=interval,
                provider=self.name,
            )

        except Exception as e:
            self.record_failure(str(e))
            raise ProviderError(f"Alpha Vantage extended intraday error: {e}")

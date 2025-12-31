"""
Data Provider
Unified interface for market data
"""

import asyncio
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger

from shared.indicators.rrs import RRSCalculator


class DataProvider:
    """
    Unified market data provider

    Supports:
    - Real-time quotes via Yahoo Finance
    - Historical data
    - Caching to reduce API calls
    """

    def __init__(self, cache_ttl_seconds: int = 30):
        """
        Initialize data provider

        Args:
            cache_ttl_seconds: How long to cache data
        """
        self.cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, Dict] = {}
        self._cache_times: Dict[str, datetime] = {}
        self.rrs_calc = RRSCalculator()

        logger.info("DataProvider initialized")

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self._cache_times:
            return False
        age = (datetime.now() - self._cache_times[key]).total_seconds()
        return age < self.cache_ttl

    async def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """
        Get current stock data including calculated indicators

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with price, volume, ATR, and daily data
        """
        cache_key = f"stock_{symbol}"

        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                self._fetch_stock_data_sync,
                symbol
            )

            if data:
                self._cache[cache_key] = data
                self._cache_times[cache_key] = datetime.now()

            return data

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

    def _fetch_stock_data_sync(self, symbol: str) -> Optional[Dict]:
        """Synchronous data fetch"""
        try:
            ticker = yf.Ticker(symbol)

            # Get daily data for ATR and trends
            daily = ticker.history(period="60d", interval="1d")
            if daily.empty:
                return None

            # Normalize columns
            daily.columns = [c.lower() for c in daily.columns]

            # Get intraday for current price
            intraday = ticker.history(period="1d", interval="5m")
            if intraday.empty:
                current_price = daily['close'].iloc[-1]
            else:
                intraday.columns = [c.lower() for c in intraday.columns]
                current_price = intraday['close'].iloc[-1]

            # Calculate ATR
            atr = self.rrs_calc.calculate_atr(daily).iloc[-1]

            return {
                "symbol": symbol,
                "current_price": float(current_price),
                "previous_close": float(daily['close'].iloc[-2]),
                "open": float(daily['open'].iloc[-1]),
                "high": float(daily['high'].iloc[-1]),
                "low": float(daily['low'].iloc[-1]),
                "volume": int(daily['volume'].iloc[-1]),
                "atr": float(atr),
                "daily_data": daily
            }

        except Exception as e:
            logger.debug(f"Error fetching {symbol}: {e}")
            return None

    async def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get quick quote for a symbol"""
        data = await self.get_stock_data(symbol)
        if data:
            return {
                "symbol": symbol,
                "price": data["current_price"],
                "change": data["current_price"] - data["previous_close"],
                "change_pct": ((data["current_price"] / data["previous_close"]) - 1) * 100,
                "volume": data["volume"]
            }
        return None

    async def get_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get quotes for multiple symbols"""
        tasks = [self.get_quote(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        quotes = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, dict):
                quotes[symbol] = result

        return quotes

    async def get_spy_data(self) -> Optional[Dict]:
        """Get SPY benchmark data"""
        return await self.get_stock_data("SPY")

    def clear_cache(self):
        """Clear all cached data"""
        self._cache.clear()
        self._cache_times.clear()

    def invalidate(self, symbol: str):
        """Invalidate cache for a symbol"""
        cache_key = f"stock_{symbol}"
        if cache_key in self._cache:
            del self._cache[cache_key]
        if cache_key in self._cache_times:
            del self._cache_times[cache_key]

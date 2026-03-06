"""
Intraday (5-minute) Bar Data Service

Fetches and caches 5-minute bars for signal candidates and open positions.
Only used for small sets of symbols (~5-20 candidates, ~1-10 positions),
never for the full 503-symbol scan.

Uses the _run_in_thread() pattern from data_provider.py to avoid blocking
ib_insync's event loop.
"""

import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger


class IntradayDataService:
    """
    Lightweight service that fetches and caches 5-minute bars.

    Cache TTL defaults to 300 seconds (5 minutes), matching the bar interval
    so we don't re-fetch within the same bar.
    """

    def __init__(self, cache_ttl_seconds: int = 300):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_times: Dict[str, datetime] = {}
        self._cache_ttl = cache_ttl_seconds
        self._lock: Optional[asyncio.Lock] = None
        logger.info(
            f"IntradayDataService initialized (cache_ttl={cache_ttl_seconds}s)"
        )

    def _get_lock(self) -> asyncio.Lock:
        """Lazy-init asyncio.Lock (must be created inside a running event loop)."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still fresh."""
        if key not in self._cache_times:
            return False
        age = (datetime.now() - self._cache_times[key]).total_seconds()
        if age >= self._cache_ttl:
            self._cache.pop(key, None)
            self._cache_times.pop(key, None)
            return False
        return True

    async def _run_in_thread(self, func, *args):
        """Run a sync function in a separate thread without blocking ib_insync.

        Uses threading.Thread directly (same pattern as DataProvider._run_in_thread)
        to avoid deadlocks when ib_insync is managing the event loop.
        """
        result = [None]
        error = [None]

        def worker():
            try:
                result[0] = func(*args)
            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while thread.is_alive():
            await asyncio.sleep(0.1)

        if error[0] is not None:
            raise error[0]
        return result[0]

    def _fetch_5m_bars_sync(self, symbol: str) -> Optional[pd.DataFrame]:
        """Synchronously fetch 2 days of 5m bars via ProviderManager."""
        try:
            from data.providers.provider_manager import get_provider_manager
            pm = get_provider_manager()
            hist = pm.get_historical(symbol, period="2d", interval="5m")
            if hist is None or hist.data.empty:
                return None
            df = hist.data.copy()
            return df
        except Exception as e:
            logger.debug(f"IntradayData: failed to fetch 5m bars for {symbol}: {e}")
            return None

    async def get_5m_bars(
        self, symbol: str, min_bars: int = 15
    ) -> Optional[pd.DataFrame]:
        """
        Get 5-minute bars for a symbol, with caching.

        Args:
            symbol: Stock ticker symbol.
            min_bars: Minimum number of bars required.

        Returns:
            DataFrame with columns [open, high, low, close, volume] or None.
        """
        cache_key = f"5m_{symbol}"
        async with self._get_lock():
            if self._is_cache_valid(cache_key):
                cached = self._cache[cache_key]
                if len(cached) >= min_bars:
                    return cached

        try:
            df = await self._run_in_thread(self._fetch_5m_bars_sync, symbol)
            if df is not None and len(df) >= min_bars:
                async with self._get_lock():
                    self._cache[cache_key] = df
                    self._cache_times[cache_key] = datetime.now()
                return df
            elif df is not None:
                logger.debug(
                    f"IntradayData: {symbol} returned only {len(df)} bars "
                    f"(need {min_bars})"
                )
            return None
        except Exception as e:
            logger.warning(f"IntradayData: error fetching {symbol}: {e}")
            return None

    async def get_spy_5m_bars(self) -> Optional[pd.DataFrame]:
        """Get SPY 5-minute bars (shared cache)."""
        return await self.get_5m_bars("SPY", min_bars=15)

    async def get_batch_5m_bars(
        self, symbols: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch 5m bars for multiple symbols concurrently.

        Uses asyncio.gather with up to 4 concurrent fetches to avoid
        overwhelming yfinance.

        Args:
            symbols: List of ticker symbols.

        Returns:
            Dict mapping symbol -> DataFrame (only includes successful fetches).
        """
        semaphore = asyncio.Semaphore(4)

        async def _fetch_with_limit(sym: str) -> tuple:
            async with semaphore:
                df = await self.get_5m_bars(sym)
                return sym, df

        tasks = [_fetch_with_limit(s) for s in symbols]
        results_raw = await asyncio.gather(*tasks, return_exceptions=True)

        results: Dict[str, pd.DataFrame] = {}
        for item in results_raw:
            if isinstance(item, Exception):
                continue
            sym, df = item
            if df is not None:
                results[sym] = df

        logger.info(
            f"IntradayData: batch fetch {len(results)}/{len(symbols)} symbols"
        )
        return results

    def invalidate(self, symbol: str):
        """Remove a symbol from the cache."""
        cache_key = f"5m_{symbol}"
        self._cache.pop(cache_key, None)
        self._cache_times.pop(cache_key, None)

"""
Data Provider
Unified interface for market data

- Streaming quotes from IBKR for actively subscribed symbols (~95 at a time)
- IBKR snapshot quotes for remaining symbols
- Daily history loaded from PostgreSQL cache (populated by IBKR background refresh)
- No yfinance dependency
"""

import asyncio
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger

from shared.indicators.rrs import RRSCalculator


class DataProvider:
    """
    Unified market data provider

    Supports:
    - IBKR streaming quotes (active rotation group, ~95 symbols)
    - IBKR snapshot quotes (remaining symbols, chunked in batches of 50)
    - Daily history from PostgreSQL cache (HistoricalBarCache)
    - Background refresh thread to keep DB cache current via IBKR
    """

    MAX_CACHE_SIZE = 600

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

        # Daily history store: symbol -> DataFrame (refreshed once per day)
        self._daily_history: Dict[str, pd.DataFrame] = {}
        self._daily_history_loaded_at: Optional[datetime] = None

        # Dedicated thread pool to avoid conflicts with ib_insync's event loop
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dataprovider")

        # Optional broker reference for IBKR snapshot quotes
        self._broker = None

        # Historical bar cache (PostgreSQL-backed)
        self._historical_cache = None

        # Background refresh thread
        self._bg_refresh_thread: Optional[threading.Thread] = None
        self._bg_refresh_stop = threading.Event()

        logger.info("DataProvider initialized")

    def set_broker(self, broker):
        """Store a broker reference for fetching IBKR snapshot quotes."""
        self._broker = broker
        logger.info(f"DataProvider: broker set ({type(broker).__name__})")

    def set_historical_cache(self, cache):
        """Store a reference to the HistoricalBarCache for DB-backed daily bars."""
        self._historical_cache = cache
        logger.info("DataProvider: historical cache set")

    def start_background_refresh(self, watchlist: List[str]):
        """Start a daemon thread that refreshes stale symbols from IBKR."""
        if self._bg_refresh_thread is not None:
            return  # Already running

        if not self._historical_cache:
            logger.warning("Cannot start background refresh: no historical cache set")
            return

        if not self._broker or not hasattr(self._broker, 'get_historical_bars'):
            logger.warning("Cannot start background refresh: broker has no get_historical_bars()")
            return

        self._bg_refresh_stop.clear()
        self._bg_refresh_thread = threading.Thread(
            target=self._background_refresh_loop,
            args=(list(watchlist),),
            daemon=True,
            name="dataprovider-bg-refresh",
        )
        self._bg_refresh_thread.start()
        logger.info(f"Background refresh thread started for {len(watchlist)} symbols")

    def _background_refresh_loop(self, watchlist: List[str]):
        """Background thread: refresh stale daily bars from IBKR."""
        # Wait a bit for startup to settle
        if self._bg_refresh_stop.wait(30):
            return

        while not self._bg_refresh_stop.is_set():
            try:
                stale = self._historical_cache.get_stale_symbols(watchlist, max_age_hours=20, min_bars=200)
                if stale:
                    logger.info(f"Background refresh: {len(stale)} stale symbols to update from IBKR")
                    self._historical_cache.refresh_from_ibkr(
                        self._broker, stale, duration='1 Y'
                    )
                    # Reload daily history from DB after refresh
                    refreshed = self._historical_cache.get_bulk_daily_bars(stale, lookback_days=365)
                    if refreshed:
                        self._daily_history.update(refreshed)
                        logger.info(f"Background refresh: updated {len(refreshed)} symbols in memory")
                else:
                    logger.debug("Background refresh: all symbols are fresh")
            except Exception as e:
                logger.error(f"Background refresh error: {e}")

            # Sleep for 10 minutes between cycles
            if self._bg_refresh_stop.wait(600):
                break

        logger.info("Background refresh thread stopped")

    async def _run_in_thread(self, func, *args, timeout=120):
        """Run a sync function in a separate thread, compatible with ib_insync.

        Uses threading.Thread directly instead of loop.run_in_executor to
        avoid deadlocks when ib_insync is managing the event loop.

        Args:
            timeout: Maximum seconds to wait before abandoning the thread.
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

        # Poll until thread completes, yielding control to the event loop
        start = _time.monotonic()
        while thread.is_alive():
            if _time.monotonic() - start > timeout:
                logger.warning(
                    f"DATA_WARN[THREAD_TIMEOUT] _run_in_thread timed out after {timeout}s for {func.__name__}"
                )
                return None
            await asyncio.sleep(0.1)

        if error[0] is not None:
            raise error[0]
        return result[0]

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self._cache_times:
            return False
        age = (datetime.now() - self._cache_times[key]).total_seconds()
        if age >= self.cache_ttl:
            # Evict stale entry
            self._cache.pop(key, None)
            self._cache_times.pop(key, None)
            return False
        return True

    def _evict_cache_if_needed(self):
        """Evict oldest cache entries if cache exceeds MAX_CACHE_SIZE."""
        if len(self._cache) > self.MAX_CACHE_SIZE:
            # Remove oldest entries (by cache time) - drop half
            sorted_keys = sorted(
                self._cache_times.keys(),
                key=lambda k: self._cache_times.get(k, datetime.min)
            )
            for key in sorted_keys[:len(self._cache) // 2]:
                self._cache.pop(key, None)
                self._cache_times.pop(key, None)

    def _is_daily_history_fresh(self) -> bool:
        """Check if daily history was loaded today."""
        if self._daily_history_loaded_at is None:
            return False
        return self._daily_history_loaded_at.date() == datetime.now().date()

    # =========================================================================
    # Single-symbol API (uses ProviderManager)
    # =========================================================================

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
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(
                self._executor,
                self._fetch_stock_data_sync,
                symbol
            )

            if data:
                self._cache[cache_key] = data
                self._cache_times[cache_key] = datetime.now()
                self._evict_cache_if_needed()

            return data

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

    def _fetch_stock_data_sync(self, symbol: str) -> Optional[Dict]:
        """
        Synchronous data fetch via ProviderManager.
        """
        try:
            from data.providers.provider_manager import get_provider_manager
            pm = get_provider_manager()

            # Get daily data for ATR and trends
            hist_daily = pm.get_historical(symbol, period="60d", interval="1d")
            if hist_daily is None or hist_daily.data.empty:
                logger.debug(f"No daily data for {symbol}")
                return None

            daily = hist_daily.data.copy()

            # Get intraday for current price
            try:
                hist_5m = pm.get_historical(symbol, period="1d", interval="5m")
                if hist_5m and not hist_5m.data.empty:
                    current_price = hist_5m.data['close'].iloc[-1]
                else:
                    current_price = daily['close'].iloc[-1]
            except Exception:
                current_price = daily['close'].iloc[-1]

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
                "daily_data": daily,
            }

        except Exception as e:
            logger.debug(f"DataProvider fetch for {symbol} failed: {e}")
            return None

    # =========================================================================
    # Batch API (uses DB cache + IBKR streaming/snapshots)
    # =========================================================================

    async def get_batch_stock_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch data for many symbols using DB-cached daily history + IBKR quotes.

        Architecture:
        1. Daily history: loaded from PostgreSQL cache (populated by IBKR background refresh)
        2. Bulk quotes: IBKR streaming + snapshots for current prices
        3. Merges quotes + cached history into the standard stock data dict

        Returns dict mapping symbol -> stock data dict.
        """
        # Check if all requested symbols are in the short-lived cache
        results = {}
        uncached = []
        for symbol in symbols:
            cache_key = f"stock_{symbol}"
            if self._is_cache_valid(cache_key):
                results[symbol] = self._cache[cache_key]
            else:
                uncached.append(symbol)

        if not uncached:
            logger.info(f"Batch fetch: all {len(symbols)} symbols served from cache")
            return results

        logger.info(f"Batch fetch: {len(results)} cached, {len(uncached)} to fetch")

        # Step 1: Ensure daily history is loaded (once per day from DB)
        if not self._is_daily_history_fresh():
            logger.info(f"Loading daily history for {len(uncached)} symbols from DB cache...")
            history = await self._run_in_thread(
                self._load_daily_history_from_cache, uncached
            )
            if history:
                self._daily_history.update(history)
                logger.info(f"Daily history loaded from cache: {len(history)} symbols")
            else:
                logger.warning("Daily history: no data in DB cache")
            self._daily_history_loaded_at = datetime.now()
        else:
            # Load history for any new symbols not yet in the store
            missing = [s for s in uncached if s not in self._daily_history]
            if missing:
                logger.info(f"Loading daily history for {len(missing)} new symbols from DB cache...")
                history = await self._run_in_thread(
                    self._load_daily_history_from_cache, missing
                )
                if history:
                    self._daily_history.update(history)

        # Step 2: Fetch current quotes — prefer streaming, then IBKR snapshots
        quotes = {}
        if self._broker and hasattr(self._broker, 'is_connected') and self._broker.is_connected:
            # Try streaming cache first (instant, no API calls)
            if hasattr(self._broker, 'has_streaming') and self._broker.has_streaming:
                streaming_quotes = self._broker.get_streaming_quotes(uncached)
                if not isinstance(streaming_quotes, dict):
                    logger.warning(
                        "DATA_WARN[STREAMING_QUOTES_INVALID] Streaming quotes were not a dict; continuing with snapshots"
                    )
                    streaming_quotes = {}
                for sym, q in streaming_quotes.items():
                    price = q.last if q.last and q.last > 0 else (q.bid if q.bid and q.bid > 0 else 0)
                    if price > 0:
                        norm_sym = sym.upper().replace(" ", "-")  # Reverse IBKR normalization
                        orig_sym = sym if sym in uncached else norm_sym if norm_sym in uncached else sym
                        quotes[orig_sym] = {
                            "price": price,
                            "previous_close": q.prev_close if q.prev_close and q.prev_close > 0 else price,
                            "volume": q.volume if q.volume else 0,
                        }
                logger.info(f"Streaming cache: {len(quotes)}/{len(uncached)} quotes")

                # Trigger rotation for next scan cycle
                self._broker.rotate_streaming()

            # Fill missing symbols with IBKR snapshots (30s timeout — don't block scan)
            missing = [s for s in uncached if s not in quotes]
            if missing:
                logger.info(f"Streaming covered {len(quotes)}/{len(uncached)}, fetching {len(missing)} via IBKR snapshots")
                ibkr_quotes = await self._run_in_thread(self._fetch_ibkr_quotes, missing, timeout=30)
                if not isinstance(ibkr_quotes, dict):
                    logger.warning(
                        "DATA_WARN[IBKR_QUOTES_TIMEOUT] IBKR snapshot fetch timed out"
                    )
                    ibkr_quotes = {}
                quotes.update(ibkr_quotes)

                # For still-missing symbols, use last known close from DB as fallback
                still_missing = [s for s in missing if s not in ibkr_quotes]
                if still_missing:
                    logger.info(f"IBKR snapshots missed {len(still_missing)} symbols — using last close from DB")
                    for sym in still_missing:
                        daily = self._daily_history.get(sym)
                        if daily is not None and len(daily) >= 1:
                            last_close = float(daily['close'].iloc[-1])
                            if last_close > 0:
                                quotes[sym] = {
                                    "price": last_close,
                                    "previous_close": float(daily['close'].iloc[-2]) if len(daily) >= 2 else last_close,
                                    "volume": int(daily['volume'].iloc[-1]),
                                }
        else:
            # No broker connected — use last close from DB cache as price
            logger.info("No broker connected — using DB cache closes as prices")
            for sym in uncached:
                daily = self._daily_history.get(sym)
                if daily is not None and len(daily) >= 2:
                    last_close = float(daily['close'].iloc[-1])
                    if last_close > 0:
                        quotes[sym] = {
                            "price": last_close,
                            "previous_close": float(daily['close'].iloc[-2]),
                            "volume": int(daily['volume'].iloc[-1]),
                        }
        logger.info(f"Bulk quotes fetched: {len(quotes)} symbols")

        # Step 3: Merge quotes + daily history into standard data dict
        now = datetime.now()
        for symbol in uncached:
            quote = quotes.get(symbol)
            daily = self._daily_history.get(symbol)

            if not quote:
                continue
            if daily is None or (hasattr(daily, 'empty') and daily.empty) or len(daily) < 3:
                continue

            try:
                atr_val = self.rrs_calc.calculate_atr(daily).iloc[-1]
                if np.isnan(atr_val):
                    continue

                data = {
                    "symbol": symbol,
                    "current_price": quote["price"],
                    "previous_close": quote["previous_close"],
                    "open": float(daily["open"].iloc[-1]),
                    "high": float(daily["high"].iloc[-1]),
                    "low": float(daily["low"].iloc[-1]),
                    "volume": quote["volume"],
                    "atr": float(atr_val),
                    "daily_data": daily,
                }
                results[symbol] = data

                cache_key = f"stock_{symbol}"
                self._cache[cache_key] = data
                self._cache_times[cache_key] = now
            except Exception as e:
                logger.debug(f"Batch merge: skipping {symbol}: {e}")
                continue

        self._evict_cache_if_needed()
        logger.info(f"Batch fetch complete: {len(results)}/{len(symbols)} symbols")
        return results

    def _load_daily_history_from_cache(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load daily OHLCV history from PostgreSQL cache.

        Pure SQL read — no IBKR API calls, no yfinance.
        Returns dict mapping symbol -> DataFrame with lowercase columns.
        """
        if not self._historical_cache:
            logger.warning("No historical cache available for daily history")
            return {}

        return self._historical_cache.get_bulk_daily_bars(symbols, lookback_days=365)

    def _fetch_ibkr_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch current quotes via IBKR snapshots in chunks of 50.

        Uses live or delayed data depending on IBKR_MARKET_DATA_TYPE config.
        Returns dict mapping symbol -> {price, previous_close, volume}.

        Note: Caller must ensure nest_asyncio is applied if running inside
        an async context, since ib_insync needs to run its own event loop.
        """
        if not symbols:
            return {}

        results = {}
        chunk_size = 50
        chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
        logger.info(f"IBKR quotes: fetching {len(symbols)} symbols in {len(chunks)} chunks")

        # Build reverse map: IBKR normalized symbol -> original symbol
        # e.g. "BRK B" -> "BRK-B" so we can key results by the original symbol
        norm_to_orig = {}
        for s in symbols:
            normalized = s.upper().replace("-", " ")
            norm_to_orig[normalized] = s
            norm_to_orig[s.upper()] = s  # identity mapping for non-hyphenated

        for idx, chunk in enumerate(chunks):
            try:
                ibkr_quotes = self._broker.get_quotes(chunk)
                if not isinstance(ibkr_quotes, dict):
                    logger.warning(
                        f"DATA_WARN[IBKR_CHUNK_INVALID] IBKR quotes chunk {idx + 1}/{len(chunks)} returned invalid type"
                    )
                    continue
                for sym, quote in ibkr_quotes.items():
                    price = quote.last if quote.last and quote.last > 0 else (quote.bid if quote.bid and quote.bid > 0 else 0)
                    if price > 0:
                        orig_sym = norm_to_orig.get(sym, sym)
                        results[orig_sym] = {
                            "price": price,
                            "previous_close": quote.prev_close if quote.prev_close and quote.prev_close > 0 else price,
                            "volume": quote.volume if quote.volume else 0,
                        }
            except Exception as e:
                code = "IBKR_CHUNK_TIMEOUT" if "timeout" in str(e).lower() else "IBKR_CHUNK_ERROR"
                logger.warning(
                    f"DATA_WARN[{code}] IBKR quotes chunk {idx + 1}/{len(chunks)} failed: {e}"
                )
        logger.info(f"IBKR quotes fetched: {len(results)}/{len(symbols)} symbols")
        return results

    # =========================================================================
    # Convenience methods
    # =========================================================================

    async def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get quick quote for a symbol"""
        data = await self.get_stock_data(symbol)
        if data:
            return {
                "symbol": symbol,
                "price": data["current_price"],
                "change": data["current_price"] - data["previous_close"],
                "change_pct": ((data["current_price"] / data["previous_close"]) - 1) * 100 if data["previous_close"] else 0.0,
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

    def get_cached_price(self, symbol: str) -> Optional[Dict]:
        """
        Get last known price from the batch cache (synchronous, no network call).

        Returns dict with 'price', 'volume', 'high', 'low', 'open', 'prev_close'
        or None if the symbol is not in cache.
        """
        cache_key = f"stock_{symbol}"
        data = self._cache.get(cache_key)
        if data is not None:
            return {
                'price': data['current_price'],
                'volume': data.get('volume', 0),
                'high': data.get('high', data['current_price']),
                'low': data.get('low', data['current_price']),
                'open': data.get('open', data['current_price']),
                'prev_close': data.get('previous_close', data['current_price']),
            }
        return None

    def invalidate(self, symbol: str):
        """Invalidate cache for a symbol"""
        cache_key = f"stock_{symbol}"
        if cache_key in self._cache:
            del self._cache[cache_key]
        if cache_key in self._cache_times:
            del self._cache_times[cache_key]

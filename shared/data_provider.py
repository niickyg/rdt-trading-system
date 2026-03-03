"""
Data Provider
Unified interface for market data

- Streaming quotes from IBKR for actively subscribed symbols (~95 at a time)
- IBKR snapshot quotes for remaining symbols (no Yahoo dependency)
- Daily history loaded once at startup via yahooquery (~33s for 503 symbols)
- yfinance fallback only when IBKR is unavailable
"""

# Fix curl_cffi chrome136 impersonation issue in Docker
# curl_cffi 0.13.0 maps 'chrome' to 'chrome136' which may not be supported
try:
    from curl_cffi.requests import impersonate
    # Use chrome110 which is widely supported
    impersonate.DEFAULT_CHROME = 'chrome110'
    if hasattr(impersonate, 'REAL_TARGET_MAP'):
        impersonate.REAL_TARGET_MAP['chrome'] = 'chrome110'
except ImportError:
    pass  # curl_cffi not installed

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
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
    - IBKR streaming quotes (active rotation group, ~95 symbols)
    - IBKR snapshot quotes (remaining symbols, chunked in batches of 50)
    - Daily history caching via yahooquery (loaded once, refreshed daily)
    - yfinance fallback when IBKR is unavailable
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

        logger.info("DataProvider initialized")

    def set_broker(self, broker):
        """Store a broker reference for fetching IBKR snapshot quotes."""
        self._broker = broker
        logger.info(f"DataProvider: broker set ({type(broker).__name__})")

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
        import time as _time
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
    # Single-symbol API (unchanged, uses yfinance)
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
        Synchronous data fetch with exponential backoff retry logic.

        Attempts up to 3 fetches with delays of 1 s, 2 s, and 4 s between
        failures before giving up.
        """
        import time as _time

        last_exc: Optional[Exception] = None

        for attempt in range(3):
            wait_time = 1.0 * (2 ** attempt)  # 1s, 2s, 4s
            try:
                ticker = yf.Ticker(symbol)

                # Get daily data for ATR and trends
                daily = ticker.history(period="60d", interval="1d")
                if daily.empty:
                    raise ValueError(f"No daily data returned for {symbol}")

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
                    "daily_data": daily,
                }

            except Exception as e:
                last_exc = e
                if attempt < 2:
                    logger.warning(
                        f"DataProvider fetch for {symbol} failed "
                        f"(attempt {attempt + 1}/3): {e}; "
                        f"retrying in {wait_time}s"
                    )
                    _time.sleep(wait_time)

        logger.debug(
            f"DataProvider fetch for {symbol} failed after 3 attempts: {last_exc}"
        )
        return None

    # =========================================================================
    # Batch API (uses yahooquery for fast bulk fetching)
    # =========================================================================

    async def get_batch_stock_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch data for many symbols using yahooquery bulk API.

        Architecture:
        1. Daily history: loaded once per day (~33s for 503 symbols), cached in memory
        2. Bulk quotes: fetched each call (~26s for 503 symbols) for current prices
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

        loop = asyncio.get_running_loop()

        # Step 1: Ensure daily history is loaded (once per day)
        if not self._is_daily_history_fresh():
            logger.info(f"Loading daily history for {len(uncached)} symbols (once per day)...")
            history = await self._run_in_thread(
                self._fetch_bulk_history_sync, uncached
            )
            if history:
                self._daily_history.update(history)
                logger.info(f"Daily history loaded: {len(history)} symbols")
            else:
                logger.warning("Daily history fetch returned no data")
            self._daily_history_loaded_at = datetime.now()
        else:
            # Load history for any new symbols not yet in the store
            missing = [s for s in uncached if s not in self._daily_history]
            if missing:
                logger.info(f"Loading daily history for {len(missing)} new symbols...")
                history = await self._run_in_thread(
                    self._fetch_bulk_history_sync, missing
                )
                if history:
                    self._daily_history.update(history)

        # Step 2: Fetch current quotes — prefer streaming, then snapshots, then yfinance
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

            # Fill missing symbols with IBKR snapshots
            missing = [s for s in uncached if s not in quotes]
            if missing:
                logger.info(f"Streaming covered {len(quotes)}/{len(uncached)}, fetching {len(missing)} via IBKR snapshots")
                ibkr_quotes = await self._run_in_thread(self._fetch_ibkr_quotes, missing)
                if not isinstance(ibkr_quotes, dict):
                    logger.warning(
                        "DATA_WARN[IBKR_QUOTES_TIMEOUT] IBKR snapshot fetch timed out or returned invalid data; using empty result"
                    )
                    ibkr_quotes = {}
                quotes.update(ibkr_quotes)

                # Fallback to yfinance for any still missing (IBKR failures)
                still_missing = [s for s in missing if s not in ibkr_quotes]
                if still_missing:
                    logger.info(f"IBKR snapshots missed {len(still_missing)}, falling back to yfinance")
                    yf_quotes = await self._run_in_thread(self._fetch_bulk_quotes_sync, still_missing)
                    if not isinstance(yf_quotes, dict):
                        logger.warning(
                            "DATA_WARN[YF_FALLBACK_INVALID] yfinance fallback returned invalid data; using empty result"
                        )
                        yf_quotes = {}
                    quotes.update(yf_quotes)
        else:
            # No broker connected — use yfinance
            quotes = await self._run_in_thread(self._fetch_bulk_quotes_sync, uncached)
            if not isinstance(quotes, dict):
                logger.warning(
                    "DATA_WARN[YF_QUOTES_INVALID] yfinance quote fetch returned invalid data; using empty result"
                )
                quotes = {}
        logger.info(f"Bulk quotes fetched: {len(quotes)} symbols")

        # Step 3: Merge quotes + daily history into standard data dict
        now = datetime.now()
        for symbol in uncached:
            quote = quotes.get(symbol)
            daily = self._daily_history.get(symbol)

            if not quote or daily is None or daily.empty or len(daily) < 3:
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

    def _fetch_ibkr_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch current quotes via IBKR snapshots in chunks of 50.

        Uses live or delayed data depending on IBKR_MARKET_DATA_TYPE config.
        Returns dict mapping symbol -> {price, previous_close, volume}.
        Must be called on the main thread (ib_insync is not thread-safe).
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

    def _fetch_bulk_history_sync(self, symbols: List[str], chunk_size: int = 100) -> Dict[str, pd.DataFrame]:
        """
        Fetch 60-day daily OHLCV history for many symbols using yahooquery.

        Processes in chunks to avoid timeouts with large symbol lists.
        Returns dict mapping symbol -> DataFrame with lowercase columns.
        """
        import time as _time
        from yahooquery import Ticker

        results = {}
        chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
        logger.info(f"Bulk history: fetching {len(symbols)} symbols in {len(chunks)} chunks of {chunk_size}")

        for idx, chunk in enumerate(chunks):
            try:
                t = Ticker(chunk, asynchronous=False, timeout=30)
                hist = t.history(period="60d", interval="1d")

                if isinstance(hist, pd.DataFrame) and not hist.empty:
                    for symbol in hist.index.get_level_values(0).unique():
                        try:
                            df = hist.loc[symbol].copy()
                            df.columns = [c.lower() for c in df.columns]
                            if len(df) >= 3:
                                results[symbol] = df
                        except Exception as e:
                            logger.debug(f"History parse error for {symbol}: {e}")

                logger.info(f"Bulk history chunk {idx + 1}/{len(chunks)}: got {len(results)} symbols so far")
                if idx < len(chunks) - 1:
                    _time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Bulk history chunk {idx + 1}/{len(chunks)} failed: {e}")

        return results

    def _fetch_bulk_quotes_sync(self, symbols: List[str], chunk_size: int = 100) -> Dict[str, Dict]:
        """
        Fetch current quotes for many symbols using yahooquery.

        Processes in chunks to avoid timeouts with large symbol lists.
        Returns dict mapping symbol -> {price, previous_close, volume}.
        """
        import time as _time
        from yahooquery import Ticker

        results = {}
        chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
        for idx, chunk in enumerate(chunks):
            try:
                t = Ticker(chunk, asynchronous=False, timeout=30)
                prices = t.price

                for symbol, data in prices.items():
                    if not isinstance(data, dict):
                        continue
                    try:
                        price = data.get("regularMarketPrice")
                        prev_close = data.get("regularMarketPreviousClose")
                        volume = data.get("regularMarketVolume", 0)

                        if price is not None and prev_close is not None:
                            results[symbol] = {
                                "price": float(price),
                                "previous_close": float(prev_close),
                                "volume": int(volume or 0),
                            }
                    except (TypeError, ValueError) as e:
                        logger.debug(f"Quote parse error for {symbol}: {e}")

                if idx < len(chunks) - 1:
                    _time.sleep(0.3)
            except Exception as e:
                logger.warning(f"Bulk quotes chunk {idx + 1}/{len(chunks)} failed: {e}")

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

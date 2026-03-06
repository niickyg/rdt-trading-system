"""
Historical Bar Cache — PostgreSQL-backed daily OHLCV cache

Replaces yfinance bulk downloads with cached IBKR historical data stored in
the daily_bars table. Provides fast SQL reads at startup and background
refresh from IBKR for stale symbols.

Usage:
    cache = HistoricalBarCache()
    bars = cache.get_daily_bars("AAPL", lookback_days=60)
    bulk = cache.get_bulk_daily_bars(["AAPL", "MSFT"], lookback_days=60)
    stale = cache.get_stale_symbols(watchlist, max_age_hours=20)
    cache.refresh_from_ibkr(ibkr_client, stale, duration="60 D")
"""

import asyncio
import os
import threading
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger
from sqlalchemy import text

from data.database.connection import get_db_manager
from data.database.models import DailyBar

try:
    from ib_insync import IB, Stock
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False


class HistoricalBarCache:
    """
    PostgreSQL-backed cache for daily OHLCV bars.

    Reads are pure SQL (no IBKR API calls). Writes come from background
    refresh via IBKRClient.get_historical_bars().
    """

    def __init__(self):
        self.db_manager = get_db_manager()
        self._refresh_lock = threading.Lock()
        logger.info("HistoricalBarCache initialized")

    def get_daily_bars(
        self, symbol: str, lookback_days: int = 60
    ) -> Optional[pd.DataFrame]:
        """
        Read daily bars from PostgreSQL for a single symbol.

        Returns DataFrame with lowercase columns [open, high, low, close, volume]
        indexed by bar_date, or None if no data.
        """
        cutoff = date.today() - timedelta(days=lookback_days)
        try:
            with self.db_manager.get_session() as session:
                rows = (
                    session.query(DailyBar)
                    .filter(DailyBar.symbol == symbol, DailyBar.bar_date >= cutoff)
                    .order_by(DailyBar.bar_date)
                    .all()
                )
                if not rows:
                    return None

                data = []
                for r in rows:
                    data.append({
                        'open': float(r.open),
                        'high': float(r.high),
                        'low': float(r.low),
                        'close': float(r.close),
                        'volume': int(r.volume),
                    })
                df = pd.DataFrame(data, index=[r.bar_date for r in rows])
                df.index.name = 'date'
                return df if len(df) >= 3 else None

        except Exception as e:
            logger.error(f"HistoricalBarCache.get_daily_bars({symbol}) failed: {e}")
            return None

    def get_bulk_daily_bars(
        self, symbols: List[str], lookback_days: int = 60
    ) -> Dict[str, pd.DataFrame]:
        """
        Read daily bars for many symbols in a single SQL query.

        Returns dict mapping symbol -> DataFrame with lowercase columns.
        """
        if not symbols:
            return {}

        cutoff = date.today() - timedelta(days=lookback_days)
        try:
            with self.db_manager.get_session() as session:
                rows = (
                    session.query(DailyBar)
                    .filter(
                        DailyBar.symbol.in_(symbols),
                        DailyBar.bar_date >= cutoff,
                    )
                    .order_by(DailyBar.symbol, DailyBar.bar_date)
                    .all()
                )

                # Group by symbol
                grouped: Dict[str, list] = {}
                dates_grouped: Dict[str, list] = {}
                for r in rows:
                    sym = r.symbol
                    if sym not in grouped:
                        grouped[sym] = []
                        dates_grouped[sym] = []
                    grouped[sym].append({
                        'open': float(r.open),
                        'high': float(r.high),
                        'low': float(r.low),
                        'close': float(r.close),
                        'volume': int(r.volume),
                    })
                    dates_grouped[sym].append(r.bar_date)

                result = {}
                for sym, data in grouped.items():
                    df = pd.DataFrame(data, index=dates_grouped[sym])
                    df.index.name = 'date'
                    if len(df) >= 3:
                        result[sym] = df

                logger.info(f"HistoricalBarCache: loaded {len(result)}/{len(symbols)} symbols from DB")
                return result

        except Exception as e:
            logger.error(f"HistoricalBarCache.get_bulk_daily_bars failed: {e}")
            return {}

    def save_daily_bars(self, symbol: str, df: pd.DataFrame):
        """
        Upsert daily bars from a DataFrame into PostgreSQL.

        DataFrame must have columns [open, high, low, close, volume] and a
        date-like index.
        """
        if df is None or df.empty:
            return

        try:
            with self.db_manager.get_session() as session:
                saved = 0
                for idx, row in df.iterrows():
                    bar_date_val = idx.date() if hasattr(idx, 'date') else idx

                    # Check existing
                    existing = (
                        session.query(DailyBar)
                        .filter(DailyBar.symbol == symbol, DailyBar.bar_date == bar_date_val)
                        .first()
                    )

                    if existing:
                        existing.open = float(row['open'])
                        existing.high = float(row['high'])
                        existing.low = float(row['low'])
                        existing.close = float(row['close'])
                        existing.volume = int(row['volume'])
                        existing.fetched_at = datetime.utcnow()
                    else:
                        bar = DailyBar(
                            symbol=symbol,
                            bar_date=bar_date_val,
                            open=float(row['open']),
                            high=float(row['high']),
                            low=float(row['low']),
                            close=float(row['close']),
                            volume=int(row['volume']),
                            fetched_at=datetime.utcnow(),
                        )
                        session.add(bar)
                    saved += 1

                logger.debug(f"HistoricalBarCache: saved {saved} bars for {symbol}")

        except Exception as e:
            logger.error(f"HistoricalBarCache.save_daily_bars({symbol}) failed: {e}")

    def get_stale_symbols(
        self, symbols: List[str], max_age_hours: int = 20,
        min_bars: int = 0
    ) -> List[str]:
        """
        Return symbols whose latest bar is older than max_age_hours,
        symbols with no bars at all, or symbols with fewer than min_bars.
        """
        if not symbols:
            return []

        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        try:
            with self.db_manager.get_session() as session:
                from sqlalchemy import func
                subq = (
                    session.query(
                        DailyBar.symbol,
                        func.max(DailyBar.fetched_at).label('latest'),
                        func.count(DailyBar.id).label('bar_count'),
                    )
                    .filter(DailyBar.symbol.in_(symbols))
                    .group_by(DailyBar.symbol)
                    .all()
                )
                fresh = set()
                for row in subq:
                    is_recent = row[1] and row[1].replace(tzinfo=None) >= cutoff
                    has_enough_bars = row[2] >= min_bars if min_bars > 0 else True
                    if is_recent and has_enough_bars:
                        fresh.add(row[0])
                stale = [s for s in symbols if s not in fresh]
                return stale

        except Exception as e:
            logger.error(f"HistoricalBarCache.get_stale_symbols failed: {e}")
            return list(symbols)  # If we can't check, assume all stale

    def _connect_dedicated_ib(self) -> 'IB':
        """
        Create a dedicated IB connection for historical data fetching.

        Uses client_id 11 (separate from trading=20 and data_provider=10).
        Creates its own event loop so it works from any thread.
        """
        if not IB_AVAILABLE:
            raise RuntimeError("ib_insync not available")

        host = os.environ.get("IBKR_HOST", "host.docker.internal")
        port = int(os.environ.get("IBKR_PORT", "4000"))

        # Create a fresh event loop for this thread
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        ib = IB()
        ib.connect(host, port, clientId=11, timeout=15, readonly=True)
        logger.info(f"HistoricalBarCache: dedicated IB connection established (client_id=11)")
        return ib

    def _fetch_bars_via_ib(self, ib: 'IB', symbol: str, duration: str) -> Optional[pd.DataFrame]:
        """Fetch historical bars using a dedicated IB connection."""
        normalized = symbol.upper().replace("-", " ")
        contract = Stock(normalized, 'SMART', 'USD')

        try:
            ib.qualifyContracts(contract)
        except TypeError:
            ib.qualifyContracts(contract)

        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1,
        )

        if not bars:
            return None

        data = []
        dates = []
        for bar in bars:
            data.append({
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume),
            })
            dates.append(bar.date)

        df = pd.DataFrame(data, index=pd.DatetimeIndex(dates))
        df.index.name = 'date'
        return df

    def refresh_from_ibkr(
        self,
        ibkr_client,
        symbols: List[str],
        duration: str = '1 Y',
    ):
        """
        Fetch historical bars from IBKR and save to DB.

        Creates a dedicated IB connection (client_id=11) for thread safety.
        This method is intended to run in a background thread.
        """
        if not symbols:
            return

        if not IB_AVAILABLE:
            logger.warning("HistoricalBarCache: ib_insync not available, skipping refresh")
            return

        with self._refresh_lock:
            logger.info(f"HistoricalBarCache: refreshing {len(symbols)} symbols from IBKR")

            # Create dedicated IB connection for this thread
            ib = None
            try:
                ib = self._connect_dedicated_ib()
            except Exception as e:
                logger.error(f"HistoricalBarCache: failed to connect to IBKR: {e}")
                return

            refreshed = 0
            errors = 0
            try:
                for i, symbol in enumerate(symbols):
                    try:
                        df = self._fetch_bars_via_ib(ib, symbol, duration)
                        if df is not None and not df.empty:
                            self.save_daily_bars(symbol, df)
                            refreshed += 1
                        # Pacing: 1s between requests to avoid overloading IBKR Gateway
                        time.sleep(1.0)
                    except Exception as e:
                        errors += 1
                        logger.debug(f"HistoricalBarCache: refresh failed for {symbol}: {e}")
                        time.sleep(1)  # Back off on errors

                    # Log progress every 50 symbols
                    if (i + 1) % 50 == 0:
                        logger.info(f"HistoricalBarCache: progress {i+1}/{len(symbols)} ({refreshed} saved, {errors} errors)")
            finally:
                try:
                    ib.disconnect()
                    logger.debug("HistoricalBarCache: dedicated IB connection closed")
                except Exception:
                    pass

            logger.info(f"HistoricalBarCache: refreshed {refreshed}/{len(symbols)} symbols from IBKR ({errors} errors)")


# Singleton
_historical_cache = None
_historical_cache_lock = threading.Lock()


def get_historical_cache() -> HistoricalBarCache:
    """Get or create the singleton HistoricalBarCache instance."""
    global _historical_cache
    if _historical_cache is None:
        with _historical_cache_lock:
            if _historical_cache is None:
                _historical_cache = HistoricalBarCache()
    return _historical_cache

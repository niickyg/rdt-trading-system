"""
IBKR Data Provider

Implements DataProvider interface using ib_insync for market data.
Uses a dedicated connection (separate client_id) to avoid conflicts
with the trading connection.
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from data.providers.base import (
    DataProvider,
    Quote,
    HistoricalData,
    ProviderError,
    DataNotFoundError,
)

# Try to import ib_insync
import asyncio
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

try:
    from ib_insync import IB, Stock
    IB_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    IB_AVAILABLE = False
    logger.warning(f"ib_insync not available for IBKRProvider: {e}")


# Period mapping: yfinance-style -> IBKR durationStr
_PERIOD_MAP = {
    "1d": "1 D",
    "2d": "2 D",
    "5d": "5 D",
    "10d": "10 D",
    "30d": "30 D",
    "60d": "60 D",
    "90d": "90 D",
    "1mo": "1 M",
    "3mo": "3 M",
    "6mo": "6 M",
    "1y": "1 Y",
    "2y": "2 Y",
}

# Interval mapping: yfinance-style -> IBKR barSizeSetting
_INTERVAL_MAP = {
    "1m": "1 min",
    "2m": "2 mins",
    "5m": "5 mins",
    "15m": "15 mins",
    "30m": "30 mins",
    "1h": "1 hour",
    "1d": "1 day",
    "1wk": "1 week",
    "1mo": "1 month",
}


def _nan_safe(val, default=0.0):
    """Convert value to float, returning default for None/NaN."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return float(val)


def _nan_safe_int(val, default=0):
    """Convert value to int, returning default for None/NaN."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return int(val)


class IBKRProvider(DataProvider):
    """
    Data provider using IBKR TWS/Gateway via ib_insync.

    Maintains its own ib_insync connection with a dedicated client_id
    (default 10) so it doesn't conflict with the trading connection.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
        priority: int = 1,
    ):
        super().__init__(name="ibkr", priority=priority)

        self._host = host or os.environ.get("IBKR_HOST", "host.docker.internal")
        self._port = port or int(os.environ.get("IBKR_PORT", "4000"))
        self._client_id = client_id or int(os.environ.get("IBKR_DATA_CLIENT_ID", "10"))

        self._ib: Optional[IB] = None
        self._connected = False

        if not IB_AVAILABLE:
            logger.warning("IBKRProvider created but ib_insync is not installed")
            return

        logger.info(
            f"IBKRProvider initialized (host={self._host}, port={self._port}, "
            f"client_id={self._client_id}, priority={priority})"
        )

    def _ensure_connected(self) -> IB:
        """Ensure we have a live connection, reconnecting if needed.

        ib_insync needs an asyncio event loop. If the current thread
        already has one (e.g. from the trading IB connection), we set a
        fresh loop before creating and connecting our own IB instance.
        """
        if not IB_AVAILABLE:
            raise ProviderError("ib_insync is not installed")

        if self._ib is not None and self._ib.isConnected():
            return self._ib

        try:
            if self._ib is not None:
                try:
                    self._ib.disconnect()
                except Exception:
                    pass

            # Ensure a fresh event loop for this IB instance.
            # ib_insync's connect() calls asyncio.get_event_loop() and
            # runs connectAsync on it. If the current loop is already
            # running (from another IB instance), connect() will fail.
            # Setting a brand new loop avoids that conflict.
            old_loop = None
            try:
                old_loop = asyncio.get_event_loop()
                if old_loop.is_running():
                    # Can't reuse a running loop — create a new one
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
            except RuntimeError:
                # No event loop at all — create one
                asyncio.set_event_loop(asyncio.new_event_loop())

            self._ib = IB()
            self._ib.connect(
                self._host,
                self._port,
                clientId=self._client_id,
                timeout=15,
                readonly=True,
            )
            self._connected = True
            logger.info(
                f"IBKRProvider connected to {self._host}:{self._port} "
                f"(client_id={self._client_id})"
            )
            return self._ib
        except Exception as e:
            self._connected = False
            raise ProviderError(f"Failed to connect to IBKR: {e}")

    def _make_contract(self, symbol: str) -> Stock:
        """Create a Stock contract for a symbol."""
        # IBKR uses spaces instead of hyphens (e.g., "BRK B" not "BRK-B")
        normalized = symbol.upper().replace("-", " ")
        return Stock(normalized, "SMART", "USD")

    def get_quote(self, symbol: str) -> Quote:
        """Get current quote for a symbol via IBKR snapshot."""
        ib = self._ensure_connected()

        try:
            contract = self._make_contract(symbol)

            try:
                ib.qualifyContracts(contract, timeout=10)
            except TypeError:
                ib.qualifyContracts(contract)

            ticker = ib.reqMktData(contract, snapshot=True)
            ib.sleep(2.0)

            last = _nan_safe(ticker.last)
            bid = _nan_safe(ticker.bid)
            ask = _nan_safe(ticker.ask)
            price = last if last > 0 else (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0

            if price <= 0:
                ib.cancelMktData(contract)
                raise DataNotFoundError(f"No price data for {symbol}")

            prev_close = _nan_safe(ticker.close, price)
            change = price - prev_close
            change_pct = (change / prev_close * 100) if prev_close > 0 else 0.0

            quote = Quote(
                symbol=symbol.upper(),
                price=price,
                open=_nan_safe(ticker.open, price),
                high=_nan_safe(ticker.high, price),
                low=_nan_safe(ticker.low, price),
                volume=_nan_safe_int(ticker.volume),
                previous_close=prev_close,
                change=change,
                change_percent=change_pct,
                timestamp=datetime.now(),
                provider=self.name,
            )

            ib.cancelMktData(contract)
            return quote

        except (ProviderError, DataNotFoundError):
            raise
        except Exception as e:
            raise ProviderError(f"IBKR quote error for {symbol}: {e}")

    def get_batch_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """Get quotes for multiple symbols via IBKR batch snapshot."""
        ib = self._ensure_connected()

        quotes: Dict[str, Quote] = {}
        if not symbols:
            return quotes

        # Process in chunks of 50
        chunk_size = 50
        chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]

        for chunk in chunks:
            try:
                contracts = [self._make_contract(s) for s in chunk]

                try:
                    ib.qualifyContracts(*contracts, timeout=max(30, len(contracts) * 0.1))
                except TypeError:
                    ib.qualifyContracts(*contracts)

                tickers = []
                for i, contract in enumerate(contracts):
                    t = ib.reqMktData(contract, snapshot=True)
                    tickers.append((chunk[i], contract, t))

                # Wait for data
                wait_time = min(30.0, max(2.0, len(tickers) * 0.06))
                ib.sleep(wait_time)

                for symbol, contract, ticker in tickers:
                    try:
                        last = _nan_safe(ticker.last)
                        bid = _nan_safe(ticker.bid)
                        ask = _nan_safe(ticker.ask)
                        price = last if last > 0 else (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0

                        if price <= 0:
                            continue

                        prev_close = _nan_safe(ticker.close, price)
                        change = price - prev_close
                        change_pct = (change / prev_close * 100) if prev_close > 0 else 0.0

                        quotes[symbol.upper()] = Quote(
                            symbol=symbol.upper(),
                            price=price,
                            open=_nan_safe(ticker.open, price),
                            high=_nan_safe(ticker.high, price),
                            low=_nan_safe(ticker.low, price),
                            volume=_nan_safe_int(ticker.volume),
                            previous_close=prev_close,
                            change=change,
                            change_percent=change_pct,
                            timestamp=datetime.now(),
                            provider=self.name,
                        )
                    except Exception as e:
                        logger.debug(f"IBKRProvider: failed to process quote for {symbol}: {e}")

                # Cancel subscriptions
                for _, contract, _ in tickers:
                    try:
                        ib.cancelMktData(contract)
                    except Exception:
                        pass

            except Exception as e:
                logger.warning(f"IBKRProvider: batch chunk failed: {e}")

        if not quotes:
            raise DataNotFoundError("No quotes returned from IBKR batch request")

        return quotes

    def get_historical(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> HistoricalData:
        """Get historical OHLCV data via IBKR reqHistoricalData."""
        ib = self._ensure_connected()

        duration_str = _PERIOD_MAP.get(period)
        if duration_str is None:
            raise ProviderError(f"Unsupported period: {period}")

        bar_size = _INTERVAL_MAP.get(interval)
        if bar_size is None:
            raise ProviderError(f"Unsupported interval: {interval}")

        try:
            contract = self._make_contract(symbol)

            try:
                ib.qualifyContracts(contract, timeout=10)
            except TypeError:
                ib.qualifyContracts(contract)

            # Use TRADES for daily, MIDPOINT for intraday
            what_to_show = "TRADES" if interval in ("1d", "1wk", "1mo") else "TRADES"

            bars = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration_str,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=True,
                formatDate=1,
            )

            if not bars:
                raise DataNotFoundError(f"No historical data for {symbol} ({period}/{interval})")

            # Convert to DataFrame
            data = []
            for bar in bars:
                data.append({
                    "Open": bar.open,
                    "High": bar.high,
                    "Low": bar.low,
                    "Close": bar.close,
                    "Volume": int(bar.volume),
                })

            df = pd.DataFrame(data)

            # Set datetime index
            dates = [bar.date for bar in bars]
            if dates and isinstance(dates[0], str):
                df.index = pd.to_datetime(dates)
            else:
                df.index = pd.DatetimeIndex(dates)

            return HistoricalData(
                symbol=symbol.upper(),
                data=df,
                period=period,
                interval=interval,
                provider=self.name,
            )

        except (ProviderError, DataNotFoundError):
            raise
        except Exception as e:
            raise ProviderError(f"IBKR historical error for {symbol}: {e}")

    def get_batch_historical(
        self,
        symbols: List[str],
        period: str = "60d",
        interval: str = "1d",
    ) -> Dict[str, HistoricalData]:
        """
        Get historical data for multiple symbols.

        Fetches sequentially to respect IBKR pacing rules.
        ib_insync handles pacing internally.
        """
        result: Dict[str, HistoricalData] = {}

        for symbol in symbols:
            try:
                hist = self.get_historical(symbol, period, interval)
                result[symbol] = hist
            except (DataNotFoundError, ProviderError) as e:
                logger.debug(f"IBKRProvider: no historical data for {symbol}: {e}")
                continue
            except Exception as e:
                logger.debug(f"IBKRProvider: historical fetch error for {symbol}: {e}")
                continue

        return result

    def is_available(self) -> bool:
        """Check if IBKR connection is alive."""
        if not IB_AVAILABLE:
            return False

        try:
            if self._ib is not None and self._ib.isConnected():
                return True
            # Try to connect
            self._ensure_connected()
            return True
        except Exception:
            return False

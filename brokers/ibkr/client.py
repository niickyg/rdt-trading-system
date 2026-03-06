"""
Interactive Brokers Client.

Implements BrokerInterface for live and paper trading via TWS/Gateway API
using the ib_insync library.

Features:
- Full BrokerInterface implementation
- Auto-reconnection on connection loss
- Support for market, limit, stop, stop-limit, and trailing stop orders
- Bracket and OCO order support
- Real-time quote streaming
- Paper and live trading accounts
"""

from __future__ import annotations

import time
import threading
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Tuple

import pandas as pd
from loguru import logger

from brokers.broker_interface import (
    BrokerInterface, Order, Position, Quote, AccountInfo,
    OrderSide, OrderType, OrderStatus,
    BrokerError, ConnectionError, OrderError,
    InsufficientFundsError
)
from brokers.ibkr.config import IBKRConfig, get_ibkr_config
from brokers.ibkr.orders import (
    convert_order_type,
    convert_order_side,
    convert_time_in_force,
    map_ibkr_order_status,
    map_ibkr_order_type,
    map_ibkr_order_side,
    create_bracket_order,
    create_oco_order,
)

# Python 3.14+ requires explicit event loop creation before importing ib_insync
import asyncio
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Try to import ib_insync
try:
    from ib_insync import (
        IB,
        Stock,
        Contract,
        Option,
        ComboLeg,
        Order as IBOrder,
        MarketOrder,
        LimitOrder,
        StopOrder,
        StopLimitOrder,
        Trade,
        AccountValue,
        PortfolioItem,
        Ticker,
    )
    # TrailingStopOrder not available in all versions — use Order with orderType='TRAIL'
    TrailingStopOrder = None
    IB_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    IB_AVAILABLE = False
    logger.warning(f"ib_insync not available: {e}. Install with: pip install ib_insync")


import math


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


class StreamingQuoteManager:
    """
    Manages rotating streaming market data subscriptions for IBKR.

    IBKR enforces ~100 concurrent streaming lines. This manager rotates
    through symbol groups during warmup to populate all tickers, then
    serves cached data from all tickers regardless of age. Ticker objects
    retain their last values after unsubscription, so get_quotes() returns
    data for all symbols that received data during warmup.

    Between scans, rotate_next() refreshes one group at a time to keep
    prices reasonably current across the full symbol list.

    Usage:
        sqm = StreamingQuoteManager(ib)
        sqm.configure(symbols)   # Set up symbol groups
        sqm.warm_up()            # Subscribe to all groups once (~30s)
        quotes = sqm.get_quotes(symbols)  # Instant read from cache
        sqm.rotate_next()        # Call periodically between scans
    """

    MAX_CONCURRENT = 95  # Stay under IBKR's 100-line limit
    ROTATION_SETTLE_TIME = 3.0  # Seconds to let data flow after subscribing

    def __init__(self, ib):
        self._ib = ib
        self._symbols: List[str] = []
        self._groups: List[List[str]] = []
        self._current_group: int = -1
        self._tickers: Dict[str, Any] = {}  # symbol -> Ticker (retains last values)
        self._active_contracts: Dict[str, Any] = {}  # symbol -> Contract (currently subscribed)
        self._contract_cache: Dict[str, Any] = {}  # normalized_symbol -> qualified Contract
        self._initialized_groups: set = set()
        self._active = False
        self._pump_thread: Optional[threading.Thread] = None
        self._pump_stop = threading.Event()

    def configure(self, symbols: List[str]):
        """Set up symbol groups for rotation."""
        self._symbols = list(symbols)
        self._groups = [
            symbols[i:i + self.MAX_CONCURRENT]
            for i in range(0, len(symbols), self.MAX_CONCURRENT)
        ]
        self._initialized_groups.clear()
        logger.info(
            f"StreamingQuoteManager: {len(symbols)} symbols in "
            f"{len(self._groups)} groups of {self.MAX_CONCURRENT}"
        )

    def warm_up(self):
        """
        Subscribe to all groups sequentially to populate initial data.

        After warmup, every symbol has at least one snapshot of data.
        Tickers retain their values after unsubscription, so all symbols
        will have cached prices available via get_quotes().
        """
        if not self._groups:
            return

        logger.info(f"StreamingQuoteManager: warming up {len(self._groups)} groups...")
        for i in range(len(self._groups)):
            self._subscribe_group(i)
            self._ib.sleep(self.ROTATION_SETTLE_TIME)
            self._initialized_groups.add(i)
            filled = sum(1 for s in self._symbols if self._has_data(s))
            logger.info(
                f"  Group {i + 1}/{len(self._groups)}: "
                f"{filled}/{len(self._symbols)} symbols have data"
            )

        # Verify data freshness
        fresh_count = self._count_fresh_tickers()
        if fresh_count == 0 and len(self._symbols) > 0:
            logger.warning(
                f"StreamingQuoteManager: warmup completed but NO tickers have fresh data. "
                f"Gateway may not be delivering market data."
            )
        else:
            logger.info(
                f"StreamingQuoteManager: {fresh_count}/{len(self._symbols)} tickers "
                f"have fresh data after warmup"
            )

        self._active = True
        logger.info(f"StreamingQuoteManager: warmup complete, {len(self._tickers)} tickers cached")

    def _count_fresh_tickers(self) -> int:
        """Count tickers that have any valid price data."""
        return sum(1 for s in self._symbols if self._has_data(s))

    def start_pump(self):
        """Start background thread that pumps ib_insync event loop."""
        if self._pump_thread and self._pump_thread.is_alive():
            return
        self._pump_stop.clear()
        self._pump_thread = threading.Thread(
            target=self._pump_loop, daemon=True, name="ibkr-pump"
        )
        self._pump_thread.start()
        logger.info("StreamingQuoteManager: background pump started")

    def stop_pump(self):
        """Stop the background pump thread."""
        self._pump_stop.set()
        if self._pump_thread:
            self._pump_thread.join(timeout=5)
        logger.info("StreamingQuoteManager: background pump stopped")

    def _pump_loop(self):
        """Background loop that keeps ib_insync processing incoming data."""
        consecutive_errors = 0
        while not self._pump_stop.is_set():
            try:
                if not self._ib.isConnected():
                    logger.warning("StreamingQuoteManager: IB connection lost, pump loop exiting")
                    break
                self._ib.sleep(0.1)
                consecutive_errors = 0
            except Exception as e:
                if self._pump_stop.is_set():
                    break
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    logger.error(f"StreamingQuoteManager: pump loop hit {consecutive_errors} consecutive errors, exiting: {e}")
                    break
                time.sleep(0.5)  # Back off on error

    def rotate_next(self):
        """
        Rotate to next symbol group.

        Call this between scans to cycle through all symbols.
        Returns the group index that was subscribed.
        """
        if not self._groups or not self._active:
            return -1

        next_idx = (self._current_group + 1) % len(self._groups)
        self._subscribe_group(next_idx)
        self._initialized_groups.add(next_idx)
        return next_idx

    def is_warmed_up(self) -> bool:
        """Check if all groups have been subscribed at least once."""
        return len(self._initialized_groups) >= len(self._groups)

    def get_quotes(self, symbols: List[str]) -> Dict[str, 'Quote']:
        """
        Read cached quotes from streaming tickers — NO API round-trips.

        Returns quotes for ALL symbols that have valid cached data.
        No staleness check — tickers retain their last values after
        unsubscription, so warmup data stays available. Rotation keeps
        prices reasonably fresh across all groups.
        """
        quotes = {}
        for symbol in symbols:
            ticker = self._tickers.get(symbol)
            if ticker is None:
                continue
            quote = self._ticker_to_quote(symbol, ticker)
            if quote is not None:
                quotes[symbol] = quote
        return quotes

    def get_coverage(self) -> tuple:
        """Return (symbols_with_data, total_symbols)."""
        with_data = sum(1 for s in self._symbols if self._has_data(s))
        return with_data, len(self._symbols)

    def _subscribe_group(self, group_idx: int):
        """Unsubscribe current group, subscribe to new group."""
        # Cancel current streaming subscriptions
        for sym, contract in self._active_contracts.items():
            try:
                self._ib.cancelMktData(contract)
            except Exception:
                pass
        self._active_contracts.clear()

        group = self._groups[group_idx]
        self._current_group = group_idx

        # Qualify contracts we haven't seen before
        new_contracts = []
        new_symbols = []
        for symbol in group:
            norm = symbol.upper().replace("-", " ")
            if norm not in self._contract_cache:
                contract = Stock(norm, "SMART", "USD")
                new_contracts.append(contract)
                new_symbols.append(symbol)

        if new_contracts:
            try:
                self._ib.qualifyContracts(*new_contracts)
                for symbol, contract in zip(new_symbols, new_contracts):
                    norm = symbol.upper().replace("-", " ")
                    self._contract_cache[norm] = contract
            except Exception as e:
                logger.warning(f"StreamingQuoteManager: qualify failed for group {group_idx}: {e}")

        # Subscribe to streaming data for this group
        for symbol in group:
            norm = symbol.upper().replace("-", " ")
            contract = self._contract_cache.get(norm)
            if contract is None:
                continue
            try:
                ticker = self._ib.reqMktData(contract, snapshot=False)
                self._tickers[symbol] = ticker
                self._active_contracts[symbol] = contract
            except Exception as e:
                logger.debug(f"StreamingQuoteManager: reqMktData failed for {symbol}: {e}")

    def _has_data(self, symbol: str) -> bool:
        """Check if a ticker has any price data."""
        ticker = self._tickers.get(symbol)
        if ticker is None:
            return False
        last = _nan_safe(ticker.last)
        bid = _nan_safe(ticker.bid)
        return last > 0 or bid > 0

    def _ticker_to_quote(self, symbol: str, ticker) -> Optional['Quote']:
        """Convert an ib_insync Ticker to a Quote object."""
        last = _nan_safe(ticker.last)
        bid = _nan_safe(ticker.bid)
        ask = _nan_safe(ticker.ask)

        # Need at least one valid price
        if last <= 0 and bid <= 0:
            return None

        price = last if last > 0 else bid
        if bid <= 0:
            bid = price
        if ask <= 0:
            ask = price

        return Quote(
            symbol=symbol.upper(),
            bid=bid,
            ask=ask,
            last=price,
            volume=_nan_safe_int(ticker.volume),
            timestamp=datetime.now(),
            bid_size=_nan_safe_int(ticker.bidSize),
            ask_size=_nan_safe_int(ticker.askSize),
            high=_nan_safe(ticker.high),
            low=_nan_safe(ticker.low),
            open=_nan_safe(ticker.open),
            prev_close=_nan_safe(ticker.close),
        )

    def shutdown(self):
        """Cancel all subscriptions and stop pump."""
        self.stop_pump()
        for contract in self._active_contracts.values():
            try:
                self._ib.cancelMktData(contract)
            except Exception:
                pass
        self._active_contracts.clear()
        self._active = False
        logger.info("StreamingQuoteManager: shutdown complete")


class IBKRClient(BrokerInterface):
    """
    Interactive Brokers client using ib_insync library.

    Implements the BrokerInterface for trading through Interactive Brokers'
    TWS (Trader Workstation) or IB Gateway API.

    Supports:
    - Paper trading (port 7497/4002)
    - Live trading (port 7496/4001)
    - Stocks, ETFs (options and futures via custom contracts)
    - All standard order types
    - Bracket and OCO orders
    - Auto-reconnection

    Example:
        client = IBKRClient(port=7497, paper_trading=True)
        if client.connect():
            account = client.get_account()
            order = client.place_order("AAPL", OrderSide.BUY, 10)
            client.disconnect()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        paper_trading: bool = True,
        timeout: int = 20,
        auto_reconnect: bool = True,
        config: Optional[IBKRConfig] = None
    ):
        """
        Initialize IBKR client.

        Args:
            host: TWS/Gateway host address
            port: TWS/Gateway port (7497=paper, 7496=live for TWS)
            client_id: Unique client ID for this connection
            paper_trading: Whether using paper trading account
            timeout: Connection timeout in seconds
            auto_reconnect: Enable automatic reconnection
            config: Optional IBKRConfig object (overrides other params)
        """
        if not IB_AVAILABLE:
            raise ImportError(
                "ib_insync is required for IBKR integration. "
                "Install with: pip install ib_insync"
            )

        # Use config if provided, otherwise create from params
        if config is not None:
            self._config = config
        else:
            self._config = IBKRConfig(
                host=host,
                port=port,
                client_id=client_id,
                paper_trading=paper_trading,
                timeout=timeout,
                auto_reconnect=auto_reconnect,
            )

        # IB connection
        self._ib = IB()
        self._connected = False
        self._account_id: Optional[str] = None

        # Order tracking
        self._orders: Dict[str, Order] = {}
        self._trades: Dict[str, Trade] = {}

        # Rate limiting
        self._last_request_time = 0.0

        # Reconnection state
        self._reconnect_attempts = 0
        self._reconnecting = False
        self._reconnect_lock = threading.Lock()

        # Event callbacks
        self._on_order_update: Optional[Callable[[Order], None]] = None
        self._on_position_update: Optional[Callable[[Position], None]] = None

        # Streaming quote manager (initialized after connect)
        self._streaming: Optional[StreamingQuoteManager] = None

        # Set up IB event handlers
        self._setup_event_handlers()

        logger.info(
            f"IBKRClient initialized: {self._config.connection_type} "
            f"@ {self._config.host}:{self._config.port}, "
            f"client_id={self._config.client_id}"
        )

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalize ticker symbol for IBKR (e.g. BF-B -> BF B, BRK-B -> BRK B)."""
        return symbol.upper().replace("-", " ")

    def _setup_event_handlers(self):
        """Set up ib_insync event handlers."""
        self._ib.disconnectedEvent += self._on_disconnected
        self._ib.errorEvent += self._on_error
        self._ib.orderStatusEvent += self._on_order_status
        self._ib.execDetailsEvent += self._on_exec_details

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._config.request_throttle:
            time.sleep(self._config.request_throttle - elapsed)
        self._last_request_time = time.time()

    # ==================== Event Handlers ====================

    def _on_disconnected(self):
        """Handle disconnection event."""
        logger.warning("Disconnected from IBKR")
        self._connected = False

        if self._config.auto_reconnect and not self._reconnecting:
            self._attempt_reconnect()

    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract: Any):
        """Handle error events from IB."""
        # Informational messages (not actual errors)
        if errorCode in (2104, 2106, 2158):  # Market data farm connected
            logger.debug(f"IB Info [{errorCode}]: {errorString}")
            return

        # Market data farm connection broken — invalidate streaming cache
        if errorCode in (2103, 2107):
            logger.warning(f"IB Market Data Farm BROKEN [{errorCode}]: {errorString}")
            if self._streaming:
                self._streaming._ticker_subscribed_at.clear()
                logger.warning("Invalidated all streaming ticker timestamps due to data farm disconnect")
            return

        # Connection-related errors
        if errorCode in (502, 504, 1100, 1101, 1102):
            logger.error(f"IB Connection Error [{errorCode}]: {errorString}")
            if self._config.auto_reconnect:
                self._attempt_reconnect()
            return

        # Order-related errors
        if errorCode in (201, 202, 203, 399):
            logger.warning(f"IB Order Warning [{errorCode}]: {errorString}")
            return

        logger.error(f"IB Error [{errorCode}] reqId={reqId}: {errorString}")

    def _on_order_status(self, trade: Trade):
        """Handle order status updates."""
        order_id = str(trade.order.orderId)

        if order_id in self._orders:
            order = self._orders[order_id]
            order.status = map_ibkr_order_status(trade.orderStatus.status)
            order.filled_quantity = int(trade.orderStatus.filled)

            if trade.orderStatus.avgFillPrice > 0:
                order.avg_fill_price = trade.orderStatus.avgFillPrice

            if order.status == OrderStatus.FILLED:
                order.filled_at = datetime.now()

            logger.debug(
                f"Order {order_id} status: {order.status.value}, "
                f"filled: {order.filled_quantity}/{order.quantity}"
            )

            if self._on_order_update:
                self._on_order_update(order)

    def _on_exec_details(self, trade: Trade, fill: Any):
        """Handle execution details (fills)."""
        order_id = str(trade.order.orderId)
        logger.info(
            f"Execution: Order {order_id} filled {fill.execution.shares} "
            f"@ ${fill.execution.avgPrice:.2f}"
        )

    def _attempt_reconnect(self):
        """
        Schedule an async reconnection attempt.

        ib_insync callbacks run within the event loop, so we schedule
        the reconnect as an async task rather than blocking with time.sleep().
        """
        with self._reconnect_lock:
            if self._reconnecting:
                return
            self._reconnecting = True

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._async_reconnect())
        except RuntimeError:
            # No running event loop — run sync reconnect in a daemon thread
            # to avoid blocking the ib_insync event loop with time.sleep()
            threading.Thread(
                target=self._sync_reconnect, daemon=True
            ).start()

    async def _async_reconnect(self):
        """Async reconnection with exponential backoff."""
        import random

        max_attempts = self._config.max_reconnect_attempts
        limit_label = "inf" if max_attempts == 0 else str(max_attempts)
        logger.info(f"Attempting to reconnect to IBKR (max attempts: {limit_label})...")

        base_delay = self._config.reconnect_delay
        max_delay = 300.0

        while max_attempts == 0 or self._reconnect_attempts < max_attempts:
            self._reconnect_attempts += 1

            delay = min(base_delay * (2 ** (self._reconnect_attempts - 1)), max_delay)
            jitter = random.uniform(0, delay * 0.1)
            actual_delay = delay + jitter

            logger.info(
                f"Reconnection attempt {self._reconnect_attempts}/"
                f"{limit_label} "
                f"(waiting {actual_delay:.1f}s)"
            )

            await asyncio.sleep(actual_delay)

            try:
                await self.connect_async()
                logger.info(
                    f"Reconnected to IBKR successfully after "
                    f"{self._reconnect_attempts} attempt(s)"
                )
                self._reconnect_attempts = 0
                self._reconnecting = False
                return
            except Exception as e:
                logger.warning(f"Reconnection attempt {self._reconnect_attempts} failed: {e}")

        if max_attempts > 0:
            logger.error(
                f"Failed to reconnect to IBKR after "
                f"{max_attempts} attempts. "
                f"Please check TWS/Gateway status and network connectivity."
            )
        self._reconnecting = False

    def _sync_reconnect(self):
        """Synchronous reconnection fallback (for non-async contexts)."""
        import random

        max_attempts = self._config.max_reconnect_attempts
        limit_label = "inf" if max_attempts == 0 else str(max_attempts)
        logger.info(f"Attempting to reconnect to IBKR (sync, max attempts: {limit_label})...")

        base_delay = self._config.reconnect_delay
        max_delay = 300.0

        while max_attempts == 0 or self._reconnect_attempts < max_attempts:
            self._reconnect_attempts += 1

            delay = min(base_delay * (2 ** (self._reconnect_attempts - 1)), max_delay)
            jitter = random.uniform(0, delay * 0.1)
            actual_delay = delay + jitter

            logger.info(
                f"Reconnection attempt {self._reconnect_attempts}/"
                f"{limit_label} "
                f"(waiting {actual_delay:.1f}s)"
            )

            time.sleep(actual_delay)

            try:
                if self.connect():
                    logger.info(
                        f"Reconnected to IBKR successfully after "
                        f"{self._reconnect_attempts} attempt(s)"
                    )
                    self._reconnect_attempts = 0
                    break
            except Exception as e:
                logger.warning(f"Reconnection attempt {self._reconnect_attempts} failed: {e}")

        if not self._connected and max_attempts > 0:
            logger.error(
                f"Failed to reconnect to IBKR after "
                f"{max_attempts} attempts. "
                f"Please check TWS/Gateway status and network connectivity."
            )

        self._reconnecting = False

    # ==================== Connection Methods ====================

    def connect(self) -> bool:
        """
        Connect to TWS/Gateway (synchronous version).

        For use inside an async context, use connect_async() instead.

        Returns:
            True if connection successful, False otherwise.

        Raises:
            ConnectionError: If connection fails.
        """
        if self._connected and self._ib.isConnected():
            logger.debug("Already connected to IBKR")
            return True

        try:
            logger.info(
                f"Connecting to IBKR at {self._config.host}:{self._config.port}..."
            )

            self._ib.connect(
                host=self._config.host,
                port=self._config.port,
                clientId=self._config.client_id,
                timeout=self._config.timeout,
                readonly=self._config.readonly,
            )

            self._post_connect()
            return True

        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            self._connected = False
            raise ConnectionError(f"IBKR connection failed: {e}")

    async def connect_async(self) -> bool:
        """
        Connect to TWS/Gateway (async version).

        Use this when calling from inside an async context (e.g. asyncio.run).

        Returns:
            True if connection successful, False otherwise.

        Raises:
            ConnectionError: If connection fails.
        """
        if self._connected and self._ib.isConnected():
            logger.debug("Already connected to IBKR")
            return True

        try:
            logger.info(
                f"Connecting to IBKR at {self._config.host}:{self._config.port}..."
            )

            await self._ib.connectAsync(
                host=self._config.host,
                port=self._config.port,
                clientId=self._config.client_id,
                timeout=self._config.timeout,
                readonly=self._config.readonly,
            )

            self._post_connect()
            return True

        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            self._connected = False
            raise ConnectionError(f"IBKR connection failed: {e}")

    async def create_order_connection(self) -> 'IB':
        """Create a dedicated IB connection for order execution on the current event loop.

        The main IB connection (self._ib) is bound to the main thread's event loop
        and cannot process responses when called from the agent thread. This method
        creates a second connection (client_id=21) on the CALLER's event loop so
        that order-related calls (qualifyContracts, placeOrder, sleep) can pump
        events correctly.

        Must be called from an async context on the thread where orders will be placed.
        """
        order_ib = IB()
        await order_ib.connectAsync(
            host=self._config.host,
            port=self._config.port,
            clientId=21,
            timeout=self._config.timeout,
            readonly=False,
        )
        # Set same market data type
        order_ib.reqMarketDataType(self._config.market_data_type)
        logger.info("Dedicated order execution IB connection established (client_id=21)")
        return order_ib

    def _post_connect(self):
        """Common post-connection setup."""
        self._connected = True

        # Set market data type from config (1=live, 2=frozen, 3=delayed, 4=delayed-frozen)
        mdt = self._config.market_data_type
        mdt_labels = {1: "live", 2: "frozen", 3: "delayed", 4: "delayed-frozen"}
        self._ib.reqMarketDataType(mdt)
        logger.info(f"Using {mdt_labels.get(mdt, mdt)} market data (type {mdt})")

        # Get account ID
        accounts = self._ib.managedAccounts()
        if accounts:
            self._account_id = self._config.account or accounts[0]
            logger.info(f"Using account: {self._account_id}")
        else:
            logger.warning("No managed accounts found")

        # Reinitialize streaming after reconnection
        if self._streaming and self._streaming._symbols:
            logger.info("Reinitializing streaming market data after reconnect...")
            symbols = list(self._streaming._symbols)
            self._streaming.shutdown()
            self._streaming = StreamingQuoteManager(self._ib)
            self._streaming.configure(symbols)
            self._streaming.warm_up()
            self._streaming.start_pump()
            logger.info("Streaming market data reinitialized")

        logger.info(
            f"Connected to IBKR ({self._config.connection_type})"
        )

    def disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self._streaming:
            # shutdown() calls stop_pump() internally, ensuring pump thread
            # exits before we disconnect the IB connection
            self._streaming.shutdown()
            self._streaming = None

        if self._ib.isConnected():
            self._ib.disconnect()
            logger.info("Disconnected from IBKR")

        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to broker."""
        return self._connected and self._ib.isConnected()

    # ==================== Streaming Market Data ====================

    def start_streaming(self, symbols: List[str]):
        """
        Start streaming market data for a list of symbols.

        Subscribes in groups of 95 (under IBKR's 100-line limit), rotating
        through all groups to populate data. After warmup, call
        rotate_streaming() periodically to keep data fresh.

        Args:
            symbols: Full list of symbols to stream (can exceed 100).
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IBKR")

        if self._streaming:
            self._streaming.shutdown()

        self._streaming = StreamingQuoteManager(self._ib)
        self._streaming.configure(symbols)
        self._streaming.warm_up()
        self._streaming.start_pump()

        coverage, total = self._streaming.get_coverage()
        logger.info(f"Streaming started: {coverage}/{total} symbols with data")

    def rotate_streaming(self):
        """Rotate to next symbol group. Call between scans."""
        if self._streaming and self._streaming.is_warmed_up():
            self._streaming.rotate_next()

    def get_streaming_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """
        Read cached quotes from streaming subscriptions — zero API latency.

        Returns empty dict if streaming not active or not warmed up.
        """
        if not self._streaming or not self._streaming.is_warmed_up():
            return {}
        return self._streaming.get_quotes(symbols)

    @property
    def has_streaming(self) -> bool:
        """Check if streaming is active and warmed up."""
        return (
            self._streaming is not None
            and self._streaming.is_warmed_up()
        )

    # ==================== Account Methods ====================

    def get_account(self) -> AccountInfo:
        """
        Get account information and balances.

        Returns:
            AccountInfo with current account state.

        Raises:
            ConnectionError: If not connected.
            BrokerError: If request fails.
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IBKR")

        self._rate_limit()

        try:
            # Request account summary
            account_values = self._ib.accountSummary(self._account_id)

            # Parse account values into a dict for easier access
            values: Dict[str, float] = {}
            for av in account_values:
                if av.currency == "USD":
                    try:
                        values[av.tag] = float(av.value)
                    except (ValueError, TypeError):
                        pass

            # Get positions for position value calculation
            positions = self.get_positions()
            positions_value = sum(p.market_value for p in positions.values())

            # Extract key values
            buying_power = values.get("BuyingPower", 0.0)
            cash = values.get("AvailableFunds", values.get("CashBalance", 0.0))
            equity = values.get("NetLiquidation", 0.0)
            daily_pnl = values.get("DailyPnL", 0.0)

            # Day trading info
            day_trades_remaining = int(values.get("DayTradesRemaining", 3))

            # Margin info
            margin_enabled = values.get("RegTMargin", 0) > 0

            return AccountInfo(
                account_id=self._account_id or "IBKR",
                buying_power=buying_power,
                cash=cash,
                equity=equity,
                day_trades_remaining=day_trades_remaining,
                pattern_day_trader=day_trades_remaining == 0,
                margin_enabled=margin_enabled,
                positions_value=positions_value,
                daily_pnl=daily_pnl,
            )

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise BrokerError(f"Failed to get account info: {e}")

    # ==================== Position Methods ====================

    def get_positions(self) -> Dict[str, Position]:
        """
        Get all current positions.

        Returns:
            Dictionary mapping symbol to Position.

        Raises:
            ConnectionError: If not connected.
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IBKR")

        self._rate_limit()

        positions: Dict[str, Position] = {}

        try:
            portfolio_items = self._ib.portfolio(self._account_id)

            for item in portfolio_items:
                quantity = int(item.position)

                if quantity == 0:
                    continue

                # Build symbol key — for options include expiry/strike/right
                sec_type = getattr(item.contract, 'secType', 'STK')
                if sec_type == 'OPT':
                    symbol = (
                        f"{item.contract.symbol} "
                        f"{item.contract.lastTradeDateOrContractMonth} "
                        f"{item.contract.strike}"
                        f"{item.contract.right}"
                    )
                else:
                    symbol = item.contract.symbol

                avg_cost = item.averageCost
                current_price = item.marketPrice
                market_value = item.marketValue
                unrealized_pnl = item.unrealizedPNL
                realized_pnl = item.realizedPNL

                # Calculate unrealized P&L percentage
                cost_basis = abs(quantity) * avg_cost
                if cost_basis > 0:
                    unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100
                else:
                    unrealized_pnl_pct = 0.0

                positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_cost=avg_cost,
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    realized_pnl=realized_pnl,
                    cost_basis=cost_basis,
                )

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise BrokerError(f"Failed to get positions: {e}")

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Position if exists, None otherwise.
        """
        positions = self.get_positions()
        return positions.get(symbol.upper())

    # ==================== Order Methods ====================

    @property
    def supports_extended_hours(self) -> bool:
        """IBKR supports extended hours trading via outsideRth flag."""
        return True

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "DAY",
        session: str = "regular"
    ) -> Order:
        """
        Place an order with extended hours support.

        Args:
            symbol: Stock ticker symbol.
            side: Buy or sell direction.
            quantity: Number of shares.
            order_type: Market, limit, stop, etc.
            price: Limit price (required for LIMIT, STOP_LIMIT).
            stop_price: Stop trigger price (required for STOP, STOP_LIMIT).
            time_in_force: Order duration (DAY, GTC, etc.).
            session: Trading session ('regular', 'premarket', 'afterhours', 'extended').

        Returns:
            Order object with status.

        Raises:
            OrderError: If order placement fails.
            ConnectionError: If not connected.
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IBKR")

        if self._config.readonly:
            raise OrderError("Client is in read-only mode")

        # Validate order
        is_valid, error_msg = self.validate_order(
            symbol, side, quantity, order_type, price, stop_price
        )
        if not is_valid:
            raise OrderError(error_msg)

        # Determine if outsideRth (outside Regular Trading Hours) should be enabled
        outside_rth = session in ('premarket', 'afterhours', 'extended')

        # Extended hours typically requires limit orders for safety
        if outside_rth and order_type == OrderType.MARKET:
            logger.warning(
                f"Market orders during extended hours may have poor execution. "
                f"Consider using limit orders for {symbol}."
            )

        self._rate_limit()

        try:
            # Create contract
            contract = Stock(self._normalize_symbol(symbol), "SMART", "USD")
            self._ib.qualifyContracts(contract)

            # Convert order side to IBKR action
            action = convert_order_side(side)

            # Create order based on type with extended hours support
            ib_order = self._create_ib_order(
                action=action,
                quantity=quantity,
                order_type=order_type,
                price=price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                outside_rth=outside_rth,
            )

            # Log extended hours order
            if outside_rth:
                logger.info(
                    f"Placing extended hours order (outsideRth=True): "
                    f"{symbol} {side.value} {quantity} session={session}"
                )

            # Place order
            trade = self._ib.placeOrder(contract, ib_order)

            # Wait for order to be acknowledged
            self._ib.sleep(0.5)

            # Create our Order object
            order_id = str(trade.order.orderId)
            order = Order(
                order_id=order_id,
                symbol=symbol.upper(),
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                stop_price=stop_price,
                status=map_ibkr_order_status(trade.orderStatus.status),
                filled_quantity=int(trade.orderStatus.filled),
                avg_fill_price=(
                    trade.orderStatus.avgFillPrice
                    if trade.orderStatus.avgFillPrice > 0 else None
                ),
                created_at=datetime.now(),
                time_in_force=time_in_force,
                broker_order_id=order_id,
                session=session,
            )

            # Store order
            self._orders[order_id] = order
            self._trades[order_id] = trade

            logger.info(
                f"Placed {order_type.value} {side.value} order: "
                f"{quantity} {symbol} @ {price or 'market'}, id={order_id}"
                f"{' (extended hours)' if outside_rth else ''}"
            )

            return order

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise OrderError(f"Order placement failed: {e}")

    def _create_ib_order(
        self,
        action: str,
        quantity: int,
        order_type: OrderType,
        price: Optional[float],
        stop_price: Optional[float],
        time_in_force: str,
        outside_rth: bool = False,
    ) -> IBOrder:
        """
        Create an ib_insync Order object.

        Args:
            action: IBKR action (BUY, SELL, etc.)
            quantity: Number of shares
            order_type: Order type enum
            price: Limit price
            stop_price: Stop trigger price
            time_in_force: Order duration
            outside_rth: Allow order to execute outside regular trading hours

        Returns:
            ib_insync Order object
        """
        tif = convert_time_in_force(time_in_force)

        if order_type == OrderType.MARKET:
            order = MarketOrder(action, quantity)

        elif order_type == OrderType.LIMIT:
            if price is None:
                raise ValueError("Limit order requires price")
            order = LimitOrder(action, quantity, price)

        elif order_type == OrderType.STOP:
            if stop_price is None:
                raise ValueError("Stop order requires stop_price")
            order = StopOrder(action, quantity, stop_price)

        elif order_type == OrderType.STOP_LIMIT:
            if price is None or stop_price is None:
                raise ValueError("Stop-limit order requires price and stop_price")
            order = StopLimitOrder(action, quantity, price, stop_price)

        elif order_type == OrderType.TRAILING_STOP:
            order = IBOrder(action=action, totalQuantity=quantity, orderType='TRAIL')
            if stop_price is not None:
                order.auxPrice = stop_price  # Trailing amount

        else:
            raise ValueError(f"Unsupported order type: {order_type}")

        order.tif = tif

        # Set outsideRth flag for extended hours trading
        # This allows the order to be filled during pre-market and after-hours sessions
        if outside_rth:
            order.outsideRth = True

        return order

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel.

        Returns:
            True if cancellation successful, False otherwise.
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IBKR")

        self._rate_limit()

        try:
            # Find the trade
            if order_id in self._trades:
                trade = self._trades[order_id]
                self._ib.cancelOrder(trade.order)
                logger.info(f"Cancelled order {order_id}")

                # Update local order status
                if order_id in self._orders:
                    self._orders[order_id].status = OrderStatus.CANCELLED

                return True

            # Try to find in all trades
            for trade in self._ib.trades():
                if str(trade.order.orderId) == order_id:
                    self._ib.cancelOrder(trade.order)
                    logger.info(f"Cancelled order {order_id}")
                    return True

            logger.warning(f"Order {order_id} not found")
            return False

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get current status of an order.

        Args:
            order_id: Order ID to check.

        Returns:
            Order with current status, None if not found.
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IBKR")

        self._rate_limit()

        # Check cached orders first
        if order_id in self._orders:
            # Update from trade if available
            if order_id in self._trades:
                trade = self._trades[order_id]
                order = self._orders[order_id]
                order.status = map_ibkr_order_status(trade.orderStatus.status)
                order.filled_quantity = int(trade.orderStatus.filled)
                if trade.orderStatus.avgFillPrice > 0:
                    order.avg_fill_price = trade.orderStatus.avgFillPrice
            return self._orders[order_id]

        # Search in IB trades
        for trade in self._ib.trades():
            if str(trade.order.orderId) == order_id:
                return self._trade_to_order(trade)

        return None

    def _trade_to_order(self, trade: Trade) -> Order:
        """Convert an IB Trade to our Order object."""
        return Order(
            order_id=str(trade.order.orderId),
            symbol=trade.contract.symbol,
            side=map_ibkr_order_side(trade.order.action),
            quantity=int(trade.order.totalQuantity),
            order_type=map_ibkr_order_type(trade.order.orderType),
            price=getattr(trade.order, 'lmtPrice', None),
            stop_price=getattr(trade.order, 'auxPrice', None),
            status=map_ibkr_order_status(trade.orderStatus.status),
            filled_quantity=int(trade.orderStatus.filled),
            avg_fill_price=(
                trade.orderStatus.avgFillPrice
                if trade.orderStatus.avgFillPrice > 0 else None
            ),
            broker_order_id=str(trade.order.orderId),
        )

    def get_open_orders(self) -> List[Order]:
        """
        Get all open/pending orders.

        Returns:
            List of active orders.
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IBKR")

        self._rate_limit()

        open_orders = []

        for trade in self._ib.openTrades():
            order = self._trade_to_order(trade)
            if order.is_active:
                open_orders.append(order)

        return open_orders

    # ==================== Historical Data Methods ====================

    # Sliding-window rate limiter for IBKR historical data requests
    # IBKR allows ~60 requests per 10 minutes; we use 55 to leave headroom
    _historical_rate_window: deque = deque()
    _historical_rate_lock: threading.Lock = threading.Lock()
    _HIST_RATE_MAX = 55
    _HIST_RATE_PERIOD = 600  # 10 minutes in seconds

    def _wait_for_historical_rate(self):
        """Block until we have capacity under the IBKR historical data rate limit."""
        with self._historical_rate_lock:
            now = time.monotonic()
            # Evict entries older than the rate window
            while self._historical_rate_window and self._historical_rate_window[0] < now - self._HIST_RATE_PERIOD:
                self._historical_rate_window.popleft()

            if len(self._historical_rate_window) >= self._HIST_RATE_MAX:
                # Wait until the oldest request expires
                wait_time = self._historical_rate_window[0] + self._HIST_RATE_PERIOD - now + 1.0
                if wait_time > 0:
                    logger.info(f"IBKR historical rate limit: waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    # Re-evict after sleep
                    now = time.monotonic()
                    while self._historical_rate_window and self._historical_rate_window[0] < now - self._HIST_RATE_PERIOD:
                        self._historical_rate_window.popleft()

            self._historical_rate_window.append(time.monotonic())

    def get_historical_bars(
        self,
        symbol: str,
        duration_str: str = '60 D',
        bar_size: str = '1 day',
        what_to_show: str = 'TRADES',
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV bars from IBKR.

        Args:
            symbol: Ticker symbol (will be normalized for IBKR)
            duration_str: IBKR duration string (e.g. '60 D', '1 Y', '5 Y')
            bar_size: Bar size (e.g. '1 day', '5 mins', '1 hour')
            what_to_show: Data type ('TRADES', 'MIDPOINT', 'BID', 'ASK')

        Returns:
            DataFrame with lowercase columns [open, high, low, close, volume]
            indexed by date/datetime, or None on failure.
        """
        if not self._connected:
            logger.warning(f"IBKRClient.get_historical_bars: not connected")
            return None

        self._wait_for_historical_rate()

        normalized = self._normalize_symbol(symbol)
        try:
            # Get or create contract
            contract = Stock(normalized, 'SMART', 'USD')
            self._ib.qualifyContracts(contract)

            bars = self._ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration_str,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=True,
                formatDate=1,
            )

            if not bars:
                logger.debug(f"IBKR returned no historical bars for {symbol}")
                return None

            # Convert to DataFrame
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
            logger.debug(f"IBKR historical: {len(df)} bars for {symbol} ({duration_str}, {bar_size})")
            return df

        except Exception as e:
            logger.error(f"IBKRClient.get_historical_bars({symbol}) failed: {e}")
            return None

    # ==================== Quote Methods ====================

    def get_quote(self, symbol: str) -> Quote:
        """
        Get current quote for a symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Quote with current market data.

        Raises:
            BrokerError: If quote request fails.
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IBKR")

        self._rate_limit()

        try:
            # Create contract
            contract = Stock(self._normalize_symbol(symbol), "SMART", "USD")
            self._ib.qualifyContracts(contract)

            # Request market data snapshot
            ticker = self._ib.reqMktData(contract, snapshot=True)
            self._ib.sleep(1)  # Wait for data

            # Cancel market data subscription
            self._ib.cancelMktData(contract)

            # Extract quote data (guard against NaN from IBKR)
            import math
            def _nan_safe_float(val):
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    return 0.0
                return float(val) if val > 0 else 0.0
            def _nan_safe_int(val):
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    return 0
                return int(val)

            bid = _nan_safe_float(ticker.bid)
            ask = _nan_safe_float(ticker.ask)
            last = _nan_safe_float(ticker.last)
            volume = _nan_safe_int(ticker.volume)
            bid_size = _nan_safe_int(ticker.bidSize)
            ask_size = _nan_safe_int(ticker.askSize)
            high = _nan_safe_float(ticker.high)
            low = _nan_safe_float(ticker.low)
            open_price = _nan_safe_float(ticker.open)
            close = _nan_safe_float(ticker.close)

            # Use last price as fallback for bid/ask
            if bid == 0 and last > 0:
                bid = last
            if ask == 0 and last > 0:
                ask = last

            return Quote(
                symbol=symbol.upper(),
                bid=bid,
                ask=ask,
                last=last,
                volume=volume,
                timestamp=datetime.now(),
                bid_size=bid_size,
                ask_size=ask_size,
                high=high,
                low=low,
                open=open_price,
                prev_close=close,
            )

        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            raise BrokerError(f"Failed to get quote for {symbol}: {e}")

    def get_extended_hours_quote(self, symbol: str) -> Quote:
        """
        Get extended hours quote for a symbol from IBKR.

        IBKR provides extended hours data automatically when outsideRth is enabled.
        This method requests market data with generic tick types that include
        pre-market and after-hours data.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Quote with extended hours data.
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IBKR")

        self._rate_limit()

        try:
            # Create contract
            contract = Stock(self._normalize_symbol(symbol), "SMART", "USD")
            self._ib.qualifyContracts(contract)

            # Request market data with extended hours tick types
            # Generic tick types for extended hours:
            # 100 = Option Volume (useful), 101 = Option Open Interest
            # 106 = Market Data Off, 233 = RT Volume
            # 375 = RT Trade, 411 = RT Historical Volatility
            ticker = self._ib.reqMktData(
                contract,
                genericTickList="233",  # RT Volume for extended hours
                snapshot=True
            )
            self._ib.sleep(1.5)  # Wait for data (extended hours may be slower)

            # Cancel market data subscription
            self._ib.cancelMktData(contract)

            # Determine current session
            from utils.timezone import get_extended_hours_session
            current_session = get_extended_hours_session()

            # Extract quote data (guard against NaN from IBKR)
            import math
            def _nan_safe_float(val):
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    return 0.0
                return float(val) if val > 0 else 0.0
            def _nan_safe_int(val):
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    return 0
                return int(val)

            bid = _nan_safe_float(ticker.bid)
            ask = _nan_safe_float(ticker.ask)
            last = _nan_safe_float(ticker.last)
            volume = _nan_safe_int(ticker.volume)
            bid_size = _nan_safe_int(ticker.bidSize)
            ask_size = _nan_safe_int(ticker.askSize)
            high = _nan_safe_float(ticker.high)
            low = _nan_safe_float(ticker.low)
            open_price = _nan_safe_float(ticker.open)
            close = _nan_safe_float(ticker.close)

            # Use last price as fallback for bid/ask
            if bid == 0 and last > 0:
                bid = last
            if ask == 0 and last > 0:
                ask = last

            # For extended hours, the regular bid/ask/last should already contain
            # extended hours data when the market is in pre-market or after-hours
            return Quote(
                symbol=symbol.upper(),
                bid=bid,
                ask=ask,
                last=last,
                volume=volume,
                timestamp=datetime.now(),
                bid_size=bid_size,
                ask_size=ask_size,
                high=high,
                low=low,
                open=open_price,
                prev_close=close,
                # Extended hours fields
                extended_hours=current_session in ('premarket', 'afterhours'),
                session=current_session,
                extended_bid=bid,
                extended_ask=ask,
                extended_last=last,
                extended_volume=volume,
            )

        except Exception as e:
            logger.error(f"Failed to get extended hours quote for {symbol}: {e}")
            raise BrokerError(f"Failed to get extended hours quote for {symbol}: {e}")

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """
        Get quotes for multiple symbols.

        More efficient than calling get_quote multiple times as it
        batches the requests.

        Args:
            symbols: List of stock ticker symbols.

        Returns:
            Dictionary mapping symbol to Quote.
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IBKR")

        quotes: Dict[str, Quote] = {}

        if not symbols:
            return quotes

        self._rate_limit()

        try:
            # Create contracts for all symbols
            contracts = [Stock(self._normalize_symbol(s), "SMART", "USD") for s in symbols]

            # Qualify with timeout to prevent hanging on zombie Gateway
            qualify_timeout = max(30, len(contracts) * 0.1)
            try:
                self._ib.qualifyContracts(*contracts, timeout=qualify_timeout)
            except TypeError:
                # Older ib_insync may not support timeout kwarg
                self._ib.qualifyContracts(*contracts)

            # Request market data for all
            tickers = []
            for contract in contracts:
                ticker = self._ib.reqMktData(contract, snapshot=True)
                tickers.append((contract.symbol, ticker))

            # Wait for data — scale with batch size, minimum 2s, max 30s
            wait_time = min(30.0, max(2.0, len(tickers) * 0.06))
            self._ib.sleep(wait_time)

            # Process results
            for symbol, ticker in tickers:
                try:
                    bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0.0
                    ask = ticker.ask if ticker.ask and ticker.ask > 0 else 0.0
                    last = ticker.last if ticker.last and ticker.last > 0 else 0.0

                    if bid == 0 and last > 0:
                        bid = last
                    if ask == 0 and last > 0:
                        ask = last

                    def _safe_int(val):
                        """Convert to int, handling NaN and None."""
                        import math
                        if val is None or (isinstance(val, float) and math.isnan(val)):
                            return 0
                        return int(val)

                    def _safe_float(val):
                        """Convert to float, handling NaN and None."""
                        import math
                        if val is None or (isinstance(val, float) and math.isnan(val)):
                            return 0.0
                        return float(val) if val > 0 else 0.0

                    quotes[symbol] = Quote(
                        symbol=symbol,
                        bid=bid,
                        ask=ask,
                        last=last,
                        volume=_safe_int(ticker.volume),
                        timestamp=datetime.now(),
                        bid_size=_safe_int(ticker.bidSize),
                        ask_size=_safe_int(ticker.askSize),
                        high=_safe_float(ticker.high),
                        low=_safe_float(ticker.low),
                        open=_safe_float(ticker.open),
                        prev_close=_safe_float(ticker.close),
                    )
                except Exception as e:
                    logger.warning(f"Failed to process quote for {symbol}: {e}")

            # Cancel market data subscriptions
            for contract in contracts:
                try:
                    self._ib.cancelMktData(contract)
                except Exception:
                    pass

        except Exception as e:
            logger.warning(f"Batch quote failed, falling back to individual: {e}")
            # Fallback to individual quotes
            for symbol in symbols:
                try:
                    quotes[symbol] = self.get_quote(symbol)
                except Exception:
                    pass

        return quotes

    # ==================== Advanced Order Methods ====================

    def place_bracket_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        entry_price: Optional[float],
        take_profit_price: float,
        stop_loss_price: float,
        entry_type: OrderType = OrderType.LIMIT
    ) -> Tuple[Order, Order, Order]:
        """
        Place a bracket order (entry + profit target + stop loss).

        Args:
            symbol: Stock ticker symbol
            side: BUY or SELL for entry
            quantity: Number of shares
            entry_price: Entry price (None for market order)
            take_profit_price: Profit target price
            stop_loss_price: Stop loss price
            entry_type: Entry order type (MARKET or LIMIT)

        Returns:
            Tuple of (entry_order, take_profit_order, stop_loss_order)
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IBKR")

        if self._config.readonly:
            raise OrderError("Client is in read-only mode")

        self._rate_limit()

        try:
            # Create contract
            contract = Stock(self._normalize_symbol(symbol), "SMART", "USD")
            self._ib.qualifyContracts(contract)

            # Create bracket orders
            action = convert_order_side(side)
            entry_order, tp_order, sl_order = create_bracket_order(
                action=action,
                quantity=quantity,
                entry_price=entry_price,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                entry_type=entry_type,
            )

            # Place orders
            entry_trade = self._ib.placeOrder(contract, entry_order)

            # Set parent ID for child orders
            tp_order.parentId = entry_trade.order.orderId
            sl_order.parentId = entry_trade.order.orderId

            tp_trade = self._ib.placeOrder(contract, tp_order)
            sl_trade = self._ib.placeOrder(contract, sl_order)

            self._ib.sleep(0.5)

            # Create our Order objects
            exit_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY

            entry = Order(
                order_id=str(entry_trade.order.orderId),
                symbol=symbol.upper(),
                side=side,
                quantity=quantity,
                order_type=entry_type,
                price=entry_price,
                status=map_ibkr_order_status(entry_trade.orderStatus.status),
            )

            take_profit = Order(
                order_id=str(tp_trade.order.orderId),
                symbol=symbol.upper(),
                side=exit_side,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                price=take_profit_price,
                status=map_ibkr_order_status(tp_trade.orderStatus.status),
            )

            stop_loss = Order(
                order_id=str(sl_trade.order.orderId),
                symbol=symbol.upper(),
                side=exit_side,
                quantity=quantity,
                order_type=OrderType.STOP,
                stop_price=stop_loss_price,
                status=map_ibkr_order_status(sl_trade.orderStatus.status),
            )

            # Store orders
            self._orders[entry.order_id] = entry
            self._orders[take_profit.order_id] = take_profit
            self._orders[stop_loss.order_id] = stop_loss

            logger.info(
                f"Placed bracket order: {action} {quantity} {symbol}, "
                f"entry={entry_price or 'MKT'}, TP={take_profit_price}, SL={stop_loss_price}"
            )

            return entry, take_profit, stop_loss

        except Exception as e:
            logger.error(f"Failed to place bracket order: {e}")
            raise OrderError(f"Bracket order failed: {e}")

    def place_trailing_stop(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        trail_amount: Optional[float] = None,
        trail_percent: Optional[float] = None,
        activation_price: Optional[float] = None,
        time_in_force: str = "GTC"
    ) -> Order:
        """
        Place a trailing stop order through IBKR.

        Args:
            symbol: Stock ticker symbol
            side: SELL for long exit, BUY_TO_COVER for short exit
            quantity: Number of shares
            trail_amount: Trail by fixed dollar amount
            trail_percent: Trail by percentage
            activation_price: Price at which trailing begins (optional)
            time_in_force: Order duration

        Returns:
            Order object with trailing stop details
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IBKR")

        if self._config.readonly:
            raise OrderError("Client is in read-only mode")

        if trail_amount is None and trail_percent is None:
            raise OrderError("Either trail_amount or trail_percent must be specified")

        self._rate_limit()

        try:
            # Create contract
            contract = Stock(self._normalize_symbol(symbol), "SMART", "USD")
            self._ib.qualifyContracts(contract)

            # Convert order side to IBKR action
            action = convert_order_side(side)

            # Create trailing stop order
            trail_order = IBOrder(action=action, totalQuantity=quantity, orderType='TRAIL')
            trail_order.tif = convert_time_in_force(time_in_force)

            if trail_percent is not None:
                trail_order.trailingPercent = trail_percent
            else:
                trail_order.auxPrice = trail_amount

            # Place order
            trade = self._ib.placeOrder(contract, trail_order)
            self._ib.sleep(0.5)

            # Create our Order object
            order = Order(
                order_id=str(trade.order.orderId),
                symbol=symbol.upper(),
                side=side,
                quantity=quantity,
                order_type=OrderType.TRAILING_STOP,
                stop_price=trail_amount or trail_percent,
                status=map_ibkr_order_status(trade.orderStatus.status),
                created_at=datetime.now(),
                time_in_force=time_in_force,
                broker_order_id=str(trade.order.orderId),
            )

            self._orders[order.order_id] = order
            self._trades[order.order_id] = trade

            logger.info(
                f"Placed trailing stop: {action} {quantity} {symbol}, "
                f"trail={trail_amount or trail_percent}{'%' if trail_percent else ''}"
            )

            return order

        except Exception as e:
            logger.error(f"Failed to place trailing stop: {e}")
            raise OrderError(f"Trailing stop order failed: {e}")

    def place_oco_order(
        self,
        symbol: str,
        orders: List[Dict[str, Any]]
    ) -> List[Order]:
        """
        Place OCO (one-cancels-other) orders.

        Args:
            symbol: Stock ticker symbol
            orders: List of order specifications

        Returns:
            List of Order objects
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IBKR")

        if self._config.readonly:
            raise OrderError("Client is in read-only mode")

        self._rate_limit()

        try:
            # Create contract
            contract = Stock(self._normalize_symbol(symbol), "SMART", "USD")
            self._ib.qualifyContracts(contract)

            # Create OCO orders
            ib_orders = create_oco_order(orders)

            # Place orders
            result_orders = []
            for i, ib_order in enumerate(ib_orders):
                trade = self._ib.placeOrder(contract, ib_order)

                spec = orders[i]
                order = Order(
                    order_id=str(trade.order.orderId),
                    symbol=symbol.upper(),
                    side=OrderSide.SELL if spec.get("action") == "SELL" else OrderSide.BUY,
                    quantity=spec.get("quantity", 1),
                    order_type=spec.get("order_type", OrderType.MARKET),
                    price=spec.get("price"),
                    stop_price=spec.get("stop_price"),
                    status=map_ibkr_order_status(trade.orderStatus.status),
                )

                self._orders[order.order_id] = order
                result_orders.append(order)

            self._ib.sleep(0.5)

            logger.info(f"Placed OCO order group with {len(result_orders)} orders")

            return result_orders

        except Exception as e:
            logger.error(f"Failed to place OCO order: {e}")
            raise OrderError(f"OCO order failed: {e}")

    def modify_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        quantity: Optional[int] = None
    ) -> bool:
        """
        Modify an existing order through IBKR.

        Args:
            order_id: Order ID to modify
            price: New limit price (optional)
            stop_price: New stop trigger price (optional)
            quantity: New quantity (optional)

        Returns:
            True if modification successful
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IBKR")

        if self._config.readonly:
            raise OrderError("Client is in read-only mode")

        self._rate_limit()

        try:
            # Find the trade
            if order_id not in self._trades:
                logger.warning(f"Trade {order_id} not found for modification")
                return False

            trade = self._trades[order_id]
            ib_order = trade.order

            # Modify order parameters
            if quantity is not None:
                ib_order.totalQuantity = quantity

            if price is not None:
                ib_order.lmtPrice = price

            if stop_price is not None:
                ib_order.auxPrice = stop_price

            # Place modified order
            self._ib.placeOrder(trade.contract, ib_order)
            self._ib.sleep(0.5)

            # Update local order
            if order_id in self._orders:
                local_order = self._orders[order_id]
                if price is not None:
                    local_order.price = price
                if stop_price is not None:
                    local_order.stop_price = stop_price
                if quantity is not None:
                    local_order.quantity = quantity

            logger.info(f"Order {order_id} modified successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to modify order {order_id}: {e}")
            return False

    # ==================== Options Methods ====================

    def place_option_order(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.LIMIT,
        price: Optional[float] = None,
        exchange: str = "SMART",
    ) -> Order:
        """
        Place a single-leg option order through IBKR.

        Args:
            symbol: Underlying symbol
            expiry: Expiration date (YYYYMMDD)
            strike: Strike price
            right: "C" for call, "P" for put
            side: Order side
            quantity: Number of contracts
            order_type: Order type (LIMIT recommended for options)
            price: Limit price per contract
            exchange: Options exchange

        Returns:
            Order object with status
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IBKR")
        if self._config.readonly:
            raise OrderError("Client is in read-only mode")

        self._rate_limit()

        try:
            ib_contract = Option(
                symbol.upper(), expiry, strike, right, exchange, "USD"
            )
            qualified = self._ib.qualifyContracts(ib_contract)
            if not qualified:
                raise OrderError(f"Failed to qualify option: {symbol} {expiry} {strike}{right}")

            action = convert_order_side(side)

            if order_type == OrderType.LIMIT:
                if price is None:
                    raise OrderError("Limit order requires price")
                ib_order = LimitOrder(action, quantity, price)
            else:
                ib_order = MarketOrder(action, quantity)

            ib_order.tif = "DAY"

            trade = self._ib.placeOrder(ib_contract, ib_order)
            self._ib.sleep(0.5)

            order_id = str(trade.order.orderId)
            order = Order(
                order_id=order_id,
                symbol=f"{symbol} {expiry} {strike}{right}",
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                status=map_ibkr_order_status(trade.orderStatus.status),
                filled_quantity=int(trade.orderStatus.filled),
                avg_fill_price=(
                    trade.orderStatus.avgFillPrice
                    if trade.orderStatus.avgFillPrice > 0 else None
                ),
                created_at=datetime.now(),
                broker_order_id=order_id,
            )

            self._orders[order_id] = order
            self._trades[order_id] = trade

            logger.info(
                f"Placed option order: {action} {quantity}x "
                f"{symbol} {expiry} {strike}{right} @ {price or 'MKT'}, id={order_id}"
            )
            return order

        except Exception as e:
            logger.error(f"Failed to place option order: {e}")
            raise OrderError(f"Option order failed: {e}")

    def place_combo_order(
        self,
        symbol: str,
        legs: List[Dict[str, Any]],
        quantity: int,
        order_type: OrderType = OrderType.LIMIT,
        price: Optional[float] = None,
    ) -> Order:
        """
        Place a multi-leg combo order using IBKR BAG contract.

        Args:
            symbol: Underlying symbol
            legs: List of dicts with keys: expiry, strike, right, action, ratio
            quantity: Number of spreads
            order_type: Order type
            price: Net limit price (positive=debit to pay)

        Returns:
            Order object
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IBKR")
        if self._config.readonly:
            raise OrderError("Client is in read-only mode")

        self._rate_limit()

        try:
            combo_legs = []
            for leg_spec in legs:
                ib_opt = Option(
                    symbol.upper(),
                    leg_spec["expiry"],
                    leg_spec["strike"],
                    leg_spec["right"],
                    leg_spec.get("exchange", "SMART"),
                    "USD",
                )
                qualified = self._ib.qualifyContracts(ib_opt)
                if not qualified:
                    raise OrderError(
                        f"Failed to qualify leg: {symbol} {leg_spec['expiry']} "
                        f"{leg_spec['strike']}{leg_spec['right']}"
                    )

                combo_legs.append(ComboLeg(
                    conId=ib_opt.conId,
                    ratio=leg_spec.get("ratio", 1),
                    action=leg_spec["action"],
                    exchange=leg_spec.get("exchange", "SMART"),
                ))

            bag = Contract()
            bag.symbol = symbol.upper()
            bag.secType = "BAG"
            bag.currency = "USD"
            bag.exchange = "SMART"
            bag.comboLegs = combo_legs

            if order_type == OrderType.LIMIT:
                if price is None:
                    raise OrderError("Combo limit order requires price")
                ib_order = LimitOrder("BUY", quantity, price)
            else:
                ib_order = MarketOrder("BUY", quantity)

            ib_order.tif = "DAY"

            trade = self._ib.placeOrder(bag, ib_order)
            self._ib.sleep(1.0)

            order_id = str(trade.order.orderId)
            order = Order(
                order_id=order_id,
                symbol=f"{symbol} COMBO",
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=order_type,
                price=price,
                status=map_ibkr_order_status(trade.orderStatus.status),
                filled_quantity=int(trade.orderStatus.filled),
                created_at=datetime.now(),
                broker_order_id=order_id,
            )

            self._orders[order_id] = order
            self._trades[order_id] = trade

            logger.info(
                f"Placed combo order: {symbol} {len(legs)}-leg "
                f"{quantity}x @ net ${price or 'MKT'}, id={order_id}"
            )
            return order

        except Exception as e:
            logger.error(f"Failed to place combo order: {e}")
            raise OrderError(f"Combo order failed: {e}")

    def get_option_chain_params(self, symbol: str) -> Dict[str, Any]:
        """
        Get option chain parameters from IBKR.

        Args:
            symbol: Underlying symbol

        Returns:
            Dict with 'expirations' and 'strikes' lists
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IBKR")

        self._rate_limit()

        try:
            contract = Stock(self._normalize_symbol(symbol), "SMART", "USD")
            self._ib.qualifyContracts(contract)

            chains = self._ib.reqSecDefOptParams(
                contract.symbol, "", contract.secType, contract.conId
            )

            if not chains:
                return {"expirations": [], "strikes": []}

            # Prefer SMART exchange
            chain = None
            for c in chains:
                if c.exchange == "SMART":
                    chain = c
                    break
            if chain is None:
                chain = chains[0]

            return {
                "exchange": chain.exchange,
                "expirations": sorted(list(chain.expirations)),
                "strikes": sorted(list(chain.strikes)),
            }

        except Exception as e:
            logger.error(f"Failed to get option chain for {symbol}: {e}")
            raise BrokerError(f"Option chain query failed: {e}")

    def get_option_greeks(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
        exchange: str = "SMART",
    ) -> Optional[Dict[str, float]]:
        """
        Get Greeks for a specific option contract.

        Returns:
            Dict with delta, gamma, theta, vega, impliedVol, or None
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IBKR")

        self._rate_limit()

        try:
            ib_contract = Option(
                symbol.upper(), expiry, strike, right, exchange, "USD"
            )
            qualified = self._ib.qualifyContracts(ib_contract)
            if not qualified:
                return None

            # Generic ticks: 100=option volume, 106=implied vol
            ticker = self._ib.reqMktData(ib_contract, genericTickList="100,106", snapshot=True)
            self._ib.sleep(1.5)  # Allow time for live Greeks to populate
            self._ib.cancelMktData(ib_contract)

            greeks = ticker.modelGreeks
            if greeks:
                return {
                    "delta": greeks.delta or 0.0,
                    "gamma": greeks.gamma or 0.0,
                    "theta": greeks.theta or 0.0,
                    "vega": greeks.vega or 0.0,
                    "implied_vol": greeks.impliedVol or 0.0,
                    "underlying_price": greeks.undPrice or 0.0,
                    "option_price": greeks.optPrice or 0.0,
                    "bid": ticker.bid if ticker.bid and ticker.bid > 0 else 0.0,
                    "ask": ticker.ask if ticker.ask and ticker.ask > 0 else 0.0,
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get option Greeks: {e}")
            return None

    def qualify_option_contract(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
        exchange: str = "SMART",
    ) -> Optional[int]:
        """
        Qualify an option contract and return its conId.

        Returns:
            IBKR contract ID (conId) or None
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IBKR")

        self._rate_limit()

        try:
            ib_contract = Option(
                symbol.upper(), expiry, strike, right, exchange, "USD"
            )
            qualified = self._ib.qualifyContracts(ib_contract)
            if qualified:
                return ib_contract.conId
            return None

        except Exception as e:
            logger.error(f"Failed to qualify option contract: {e}")
            return None

    # ==================== Event Registration ====================

    def on_order_update(self, callback: Callable[[Order], None]):
        """
        Register callback for order updates.

        Args:
            callback: Function to call with Order when status changes
        """
        self._on_order_update = callback

    def on_position_update(self, callback: Callable[[Position], None]):
        """
        Register callback for position updates.

        Args:
            callback: Function to call with Position when it changes
        """
        self._on_position_update = callback

    # ==================== Utility Methods ====================

    @property
    def paper_trading(self) -> bool:
        """Check if using paper trading account."""
        return self._config.paper_trading

    @property
    def account_id(self) -> Optional[str]:
        """Get current account ID."""
        return self._account_id

    def sleep(self, seconds: float):
        """
        Sleep while processing IB events.

        Args:
            seconds: Time to sleep
        """
        self._ib.sleep(seconds)

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False

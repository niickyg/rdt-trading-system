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

import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Tuple
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
            # No running event loop — fall back to sync reconnect
            self._sync_reconnect()

    async def _async_reconnect(self):
        """Async reconnection with exponential backoff."""
        import random

        logger.info("Attempting to reconnect to IBKR...")

        base_delay = self._config.reconnect_delay
        max_delay = 300.0

        while self._reconnect_attempts < self._config.max_reconnect_attempts:
            self._reconnect_attempts += 1

            delay = min(base_delay * (2 ** (self._reconnect_attempts - 1)), max_delay)
            jitter = random.uniform(0, delay * 0.1)
            actual_delay = delay + jitter

            logger.info(
                f"Reconnection attempt {self._reconnect_attempts}/"
                f"{self._config.max_reconnect_attempts} "
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

        logger.error(
            f"Failed to reconnect to IBKR after "
            f"{self._config.max_reconnect_attempts} attempts. "
            f"Please check TWS/Gateway status and network connectivity."
        )
        self._reconnecting = False

    def _sync_reconnect(self):
        """Synchronous reconnection fallback (for non-async contexts)."""
        import random

        logger.info("Attempting to reconnect to IBKR (sync)...")

        base_delay = self._config.reconnect_delay
        max_delay = 300.0

        while self._reconnect_attempts < self._config.max_reconnect_attempts:
            self._reconnect_attempts += 1

            delay = min(base_delay * (2 ** (self._reconnect_attempts - 1)), max_delay)
            jitter = random.uniform(0, delay * 0.1)
            actual_delay = delay + jitter

            logger.info(
                f"Reconnection attempt {self._reconnect_attempts}/"
                f"{self._config.max_reconnect_attempts} "
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

        if not self._connected:
            logger.error(
                f"Failed to reconnect to IBKR after "
                f"{self._config.max_reconnect_attempts} attempts. "
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

    def _post_connect(self):
        """Common post-connection setup."""
        self._connected = True

        # Use delayed market data for paper trading (free, no subscription needed)
        if self._config.paper_trading:
            self._ib.reqMarketDataType(3)  # 3 = delayed
            logger.info("Using delayed market data (paper trading mode)")

        # Get account ID
        accounts = self._ib.managedAccounts()
        if accounts:
            self._account_id = self._config.account or accounts[0]
            logger.info(f"Using account: {self._account_id}")
        else:
            logger.warning("No managed accounts found")

        logger.info(
            f"Connected to IBKR ({self._config.connection_type})"
        )

    def disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self._ib.isConnected():
            self._ib.disconnect()
            logger.info("Disconnected from IBKR")

        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to broker."""
        return self._connected and self._ib.isConnected()

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

            # Extract quote data
            bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0.0
            ask = ticker.ask if ticker.ask and ticker.ask > 0 else 0.0
            last = ticker.last if ticker.last and ticker.last > 0 else 0.0
            volume = int(ticker.volume) if ticker.volume else 0
            bid_size = int(ticker.bidSize) if ticker.bidSize else 0
            ask_size = int(ticker.askSize) if ticker.askSize else 0
            high = ticker.high if ticker.high and ticker.high > 0 else 0.0
            low = ticker.low if ticker.low and ticker.low > 0 else 0.0
            open_price = ticker.open if ticker.open and ticker.open > 0 else 0.0
            close = ticker.close if ticker.close and ticker.close > 0 else 0.0

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

            # Extract quote data
            bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0.0
            ask = ticker.ask if ticker.ask and ticker.ask > 0 else 0.0
            last = ticker.last if ticker.last and ticker.last > 0 else 0.0
            volume = int(ticker.volume) if ticker.volume else 0
            bid_size = int(ticker.bidSize) if ticker.bidSize else 0
            ask_size = int(ticker.askSize) if ticker.askSize else 0
            high = ticker.high if ticker.high and ticker.high > 0 else 0.0
            low = ticker.low if ticker.low and ticker.low > 0 else 0.0
            open_price = ticker.open if ticker.open and ticker.open > 0 else 0.0
            close = ticker.close if ticker.close and ticker.close > 0 else 0.0

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
            self._ib.qualifyContracts(*contracts)

            # Request market data for all
            tickers = []
            for contract in contracts:
                ticker = self._ib.reqMktData(contract, snapshot=True)
                tickers.append((contract.symbol, ticker))

            # Wait for data
            self._ib.sleep(1.5)

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

                    quotes[symbol] = Quote(
                        symbol=symbol,
                        bid=bid,
                        ask=ask,
                        last=last,
                        volume=int(ticker.volume) if ticker.volume else 0,
                        timestamp=datetime.now(),
                        bid_size=int(ticker.bidSize) if ticker.bidSize else 0,
                        ask_size=int(ticker.askSize) if ticker.askSize else 0,
                        high=ticker.high if ticker.high and ticker.high > 0 else 0.0,
                        low=ticker.low if ticker.low and ticker.low > 0 else 0.0,
                        open=ticker.open if ticker.open and ticker.open > 0 else 0.0,
                        prev_close=ticker.close if ticker.close and ticker.close > 0 else 0.0,
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

            ticker = self._ib.reqMktData(ib_contract, snapshot=True)
            self._ib.sleep(1.0)
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

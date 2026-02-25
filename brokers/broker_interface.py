"""
Abstract Broker Interface for the RDT Trading System.

This module defines the standard interface that all broker implementations
must follow, ensuring consistent behavior across paper trading, Schwab,
Interactive Brokers, and any future broker integrations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class OrderSide(str, Enum):
    """Order direction/side."""
    BUY = "buy"
    SELL = "sell"
    BUY_TO_COVER = "buy_to_cover"
    SELL_SHORT = "sell_short"
    # Options-specific sides
    BUY_TO_OPEN = "buy_to_open"
    SELL_TO_OPEN = "sell_to_open"
    BUY_TO_CLOSE = "buy_to_close"
    SELL_TO_CLOSE = "sell_to_close"


class OrderType(str, Enum):
    """Order execution type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(str, Enum):
    """Order execution status."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TradingSession(str, Enum):
    """Trading session for order placement."""
    REGULAR = "regular"          # Regular market hours (9:30 AM - 4:00 PM ET)
    PREMARKET = "premarket"      # Pre-market hours (4:00 AM - 9:30 AM ET)
    AFTERHOURS = "afterhours"    # After-hours (4:00 PM - 8:00 PM ET)
    EXTENDED = "extended"        # Both pre-market and after-hours


class BrokerError(Exception):
    """Base exception for broker errors."""
    pass


class AuthenticationError(BrokerError):
    """Authentication/authorization failed."""
    pass


class ConnectionError(BrokerError):
    """Connection to broker API failed."""
    pass


class OrderError(BrokerError):
    """Order placement/modification failed."""
    pass


class InsufficientFundsError(OrderError):
    """Not enough buying power for the order."""
    pass


class PositionError(BrokerError):
    """Position-related error (e.g., trying to sell shares you don't own)."""
    pass


@dataclass
class Quote:
    """Market quote data for a symbol."""
    symbol: str
    bid: float
    ask: float
    last: float
    volume: int
    timestamp: datetime
    bid_size: int = 0
    ask_size: int = 0
    high: float = 0.0
    low: float = 0.0
    open: float = 0.0
    prev_close: float = 0.0
    # Extended hours quote fields
    extended_hours: bool = False
    session: str = "regular"  # 'regular', 'premarket', 'afterhours'
    extended_bid: float = 0.0
    extended_ask: float = 0.0
    extended_last: float = 0.0
    extended_volume: int = 0

    @property
    def mid(self) -> float:
        """Calculate mid price."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0

    @property
    def spread_pct(self) -> float:
        """Calculate spread as percentage of mid price."""
        mid = self.mid
        if mid > 0:
            return (self.spread / mid) * 100
        return 0.0

    @property
    def extended_mid(self) -> float:
        """Calculate mid price for extended hours."""
        if self.extended_bid > 0 and self.extended_ask > 0:
            return (self.extended_bid + self.extended_ask) / 2
        return self.extended_last

    @property
    def extended_spread(self) -> float:
        """Calculate bid-ask spread for extended hours."""
        if self.extended_bid > 0 and self.extended_ask > 0:
            return self.extended_ask - self.extended_bid
        return 0.0

    @property
    def extended_spread_pct(self) -> float:
        """Calculate extended hours spread as percentage of mid price."""
        mid = self.extended_mid
        if mid > 0:
            return (self.extended_spread / mid) * 100
        return 0.0


@dataclass
class Order:
    """Order representation."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    time_in_force: str = "DAY"
    broker_order_id: Optional[str] = None
    error_message: Optional[str] = None
    session: str = "regular"  # 'regular', 'premarket', 'afterhours', 'extended'

    @property
    def is_active(self) -> bool:
        """Check if order is still active (not terminal)."""
        return self.status in (
            OrderStatus.PENDING,
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED
        )

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED

    @property
    def remaining_quantity(self) -> int:
        """Calculate unfilled quantity."""
        return self.quantity - self.filled_quantity


@dataclass
class Position:
    """Current position in a security."""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl: float = 0.0
    cost_basis: float = 0.0

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

    def __post_init__(self):
        """Calculate derived fields."""
        if self.cost_basis == 0:
            self.cost_basis = abs(self.quantity) * self.avg_cost


@dataclass
class AccountInfo:
    """Account information and balances."""
    account_id: str
    buying_power: float
    cash: float
    equity: float
    day_trades_remaining: int = 3
    pattern_day_trader: bool = False
    margin_enabled: bool = False
    positions_value: float = 0.0
    daily_pnl: float = 0.0


class BrokerInterface(ABC):
    """
    Abstract base class defining the broker interface.

    All broker implementations (Paper, Schwab, IBKR, etc.) must implement
    this interface to ensure consistent behavior across the trading system.

    Usage:
        broker = get_broker("schwab", app_key="...", app_secret="...")
        if broker.connect():
            account = broker.get_account()
            order = broker.place_order("AAPL", OrderSide.BUY, 100)
            broker.disconnect()
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the broker API.

        Returns:
            True if connection successful, False otherwise.

        Raises:
            ConnectionError: If connection fails with specific error.
            AuthenticationError: If authentication fails.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Disconnect from the broker API.

        Should clean up any resources and connections.
        """
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if currently connected to broker.

        Returns:
            True if connected and authenticated, False otherwise.
        """
        pass

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """
        Get account information and balances.

        Returns:
            AccountInfo with current account state.

        Raises:
            ConnectionError: If not connected.
            BrokerError: If request fails.
        """
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """
        Get all current positions.

        Returns:
            Dictionary mapping symbol to Position.

        Raises:
            ConnectionError: If not connected.
            BrokerError: If request fails.
        """
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Position if exists, None otherwise.
        """
        pass

    @abstractmethod
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
        Place an order.

        Args:
            symbol: Stock ticker symbol.
            side: Buy or sell direction.
            quantity: Number of shares.
            order_type: Market, limit, stop, etc.
            price: Limit price (required for LIMIT, STOP_LIMIT).
            stop_price: Stop trigger price (required for STOP, STOP_LIMIT).
            time_in_force: Order duration (DAY, GTC, etc.).
            session: Trading session for the order:
                - 'regular': Regular market hours only (9:30 AM - 4:00 PM ET)
                - 'premarket': Pre-market hours (4:00 AM - 9:30 AM ET)
                - 'afterhours': After-hours (4:00 PM - 8:00 PM ET)
                - 'extended': Both pre-market and after-hours

        Returns:
            Order object with status.

        Raises:
            OrderError: If order placement fails.
            InsufficientFundsError: If not enough buying power.
            ConnectionError: If not connected.
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel.

        Returns:
            True if cancellation successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get current status of an order.

        Args:
            order_id: Order ID to check.

        Returns:
            Order with current status, None if not found.
        """
        pass

    @abstractmethod
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
        pass

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """
        Get quotes for multiple symbols.

        Default implementation calls get_quote for each symbol.
        Subclasses should override for batch efficiency.

        Args:
            symbols: List of stock ticker symbols.

        Returns:
            Dictionary mapping symbol to Quote.
        """
        quotes = {}
        for symbol in symbols:
            try:
                quotes[symbol] = self.get_quote(symbol)
            except Exception:
                pass
        return quotes

    def get_open_orders(self) -> List[Order]:
        """
        Get all open/pending orders.

        Default implementation returns empty list.
        Subclasses should override.

        Returns:
            List of active orders.
        """
        return []

    def is_market_open(self) -> bool:
        """
        Check if market is currently open for trading.

        Default implementation checks US market hours.
        Subclasses may override for more accurate exchange calendars.

        Returns:
            True if market is open, False otherwise.
        """
        from utils.timezone import is_market_open as check_market
        return check_market()

    @property
    def supports_extended_hours(self) -> bool:
        """
        Check if this broker supports extended hours trading.

        Default implementation returns False.
        Subclasses should override if they support extended hours.

        Returns:
            True if broker supports extended hours trading, False otherwise.
        """
        return False

    def get_extended_hours_quote(self, symbol: str) -> Quote:
        """
        Get extended hours quote for a symbol.

        Default implementation returns regular quote with extended hours fields populated.
        Subclasses should override to provide actual extended hours data.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Quote with extended hours data populated.

        Raises:
            BrokerError: If quote request fails.
        """
        # Default implementation - return regular quote
        quote = self.get_quote(symbol)

        # Populate extended hours fields with regular data as fallback
        from utils.timezone import get_extended_hours_session
        session = get_extended_hours_session()

        quote.extended_hours = session in ('premarket', 'afterhours')
        quote.session = session
        quote.extended_bid = quote.bid
        quote.extended_ask = quote.ask
        quote.extended_last = quote.last
        quote.extended_volume = quote.volume

        return quote

    def is_extended_hours_available(self) -> bool:
        """
        Check if extended hours trading is currently available.

        Returns:
            True if in pre-market or after-hours and broker supports it.
        """
        if not self.supports_extended_hours:
            return False

        from utils.timezone import is_extended_hours
        return is_extended_hours()

    def get_current_session(self) -> str:
        """
        Get the current trading session.

        Returns:
            One of 'premarket', 'regular', 'afterhours', or 'closed'.
        """
        from utils.timezone import get_extended_hours_session
        return get_extended_hours_session()

    def validate_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> tuple[bool, str]:
        """
        Validate order parameters before submission.

        Args:
            symbol: Stock ticker symbol.
            side: Buy or sell direction.
            quantity: Number of shares.
            order_type: Market, limit, stop, etc.
            price: Limit price.
            stop_price: Stop trigger price.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not symbol or not symbol.strip():
            return False, "Symbol is required"

        if quantity <= 0:
            return False, "Quantity must be positive"

        if order_type == OrderType.LIMIT and price is None:
            return False, "Limit orders require a price"

        if order_type == OrderType.STOP and stop_price is None:
            return False, "Stop orders require a stop price"

        if order_type == OrderType.STOP_LIMIT:
            if price is None or stop_price is None:
                return False, "Stop-limit orders require both price and stop_price"

        if price is not None and price <= 0:
            return False, "Price must be positive"

        if stop_price is not None and stop_price <= 0:
            return False, "Stop price must be positive"

        return True, ""

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
    ) -> tuple:
        """
        Place a bracket order (entry + profit target + stop loss).

        A bracket order consists of:
        1. Entry order (market or limit)
        2. Take profit order (limit) - executed when profit target reached
        3. Stop loss order (stop) - executed when stop price reached

        The profit and stop orders are OCO (one-cancels-other).

        Args:
            symbol: Stock ticker symbol.
            side: OrderSide.BUY for long, OrderSide.SELL_SHORT for short.
            quantity: Number of shares.
            entry_price: Entry limit price (None for market order).
            take_profit_price: Profit target price.
            stop_loss_price: Stop loss trigger price.
            entry_type: Entry order type (MARKET or LIMIT).

        Returns:
            Tuple of (entry_order, take_profit_order, stop_loss_order).

        Raises:
            OrderError: If order placement fails.
            NotImplementedError: If broker doesn't support bracket orders.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support native bracket orders. "
            "Use the BracketOrder class from trading.advanced_orders instead."
        )

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
        Place a trailing stop order.

        A trailing stop follows the price by a fixed amount or percentage,
        locking in profits as the price moves favorably.

        Args:
            symbol: Stock ticker symbol.
            side: SELL for long exit, BUY_TO_COVER for short exit.
            quantity: Number of shares.
            trail_amount: Trail by fixed dollar amount (mutually exclusive with trail_percent).
            trail_percent: Trail by percentage (mutually exclusive with trail_amount).
            activation_price: Price at which trailing begins (optional).
            time_in_force: Order duration.

        Returns:
            Order object with trailing stop details.

        Raises:
            OrderError: If order placement fails.
            NotImplementedError: If broker doesn't support trailing stops.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support native trailing stops. "
            "Use the TrailingStopOrder class from trading.advanced_orders instead."
        )

    def place_oco_order(
        self,
        symbol: str,
        orders: List[Dict[str, Any]]
    ) -> List[Order]:
        """
        Place OCO (one-cancels-other) orders.

        When one order in the group is filled, the others are automatically cancelled.

        Args:
            symbol: Stock ticker symbol.
            orders: List of order specifications, each containing:
                - side: OrderSide
                - quantity: int
                - order_type: OrderType
                - price: float (for limit orders)
                - stop_price: float (for stop orders)

        Returns:
            List of Order objects linked as OCO.

        Raises:
            OrderError: If order placement fails.
            NotImplementedError: If broker doesn't support OCO orders.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support native OCO orders. "
            "Use the OCOOrder class from trading.advanced_orders instead."
        )

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
        Place a single-leg option order.

        Args:
            symbol: Underlying symbol
            expiry: Expiration date (YYYYMMDD)
            strike: Strike price
            right: "C" for call, "P" for put
            side: Order side (BUY_TO_OPEN, SELL_TO_CLOSE, etc.)
            quantity: Number of contracts
            order_type: Order type (LIMIT recommended)
            price: Limit price per contract
            exchange: Options exchange

        Returns:
            Order object

        Raises:
            NotImplementedError: If broker doesn't support options
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support options trading"
        )

    def place_combo_order(
        self,
        symbol: str,
        legs: List[Dict[str, Any]],
        quantity: int,
        order_type: OrderType = OrderType.LIMIT,
        price: Optional[float] = None,
    ) -> Order:
        """
        Place a multi-leg combo (spread) option order.

        Args:
            symbol: Underlying symbol
            legs: List of leg specifications, each with:
                - expiry (str), strike (float), right (str "C"/"P")
                - action (str "BUY"/"SELL"), ratio (int)
            quantity: Number of spreads
            order_type: Order type
            price: Net limit price (positive=debit, negative=credit)

        Returns:
            Order object

        Raises:
            NotImplementedError: If broker doesn't support combo orders
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support combo option orders"
        )

    def get_option_chain(
        self,
        symbol: str,
    ) -> Dict[str, Any]:
        """
        Get option chain parameters (expirations, strikes) for a symbol.

        Args:
            symbol: Underlying symbol

        Returns:
            Dict with 'expirations' (list of str) and 'strikes' (list of float)

        Raises:
            NotImplementedError: If broker doesn't support option chains
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support option chain queries"
        )

    def modify_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        quantity: Optional[int] = None
    ) -> bool:
        """
        Modify an existing order.

        Args:
            order_id: Order ID to modify.
            price: New limit price (optional).
            stop_price: New stop trigger price (optional).
            quantity: New quantity (optional).

        Returns:
            True if modification successful, False otherwise.

        Raises:
            OrderError: If modification fails.
            NotImplementedError: If broker doesn't support order modification.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support order modification. "
            "Cancel and replace the order instead."
        )


# Alias for backward compatibility
AbstractBroker = BrokerInterface

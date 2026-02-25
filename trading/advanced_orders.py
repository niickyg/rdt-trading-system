"""
Advanced Order Types for the RDT Trading System.

Provides:
- BracketOrder: Entry + take profit + stop loss as atomic unit
- TrailingStopOrder: Dynamic stop that trails price movement
- OCOOrder: One-cancels-other linked orders

These classes provide a unified interface for complex order types
across all supported brokers (Paper, Schwab, IBKR).
"""

import uuid
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
from loguru import logger

from brokers.broker_interface import (
    BrokerInterface,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    Quote,
    BrokerError,
    OrderError,
)


class BracketOrderStatus(str, Enum):
    """Status of a bracket order group."""
    PENDING = "pending"
    ENTRY_FILLED = "entry_filled"
    TAKE_PROFIT_FILLED = "take_profit_filled"
    STOP_LOSS_FILLED = "stop_loss_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"


class TrailingStopType(str, Enum):
    """Type of trailing stop calculation."""
    FIXED_AMOUNT = "fixed_amount"
    PERCENTAGE = "percentage"


@dataclass
class BracketOrderResult:
    """Result from placing a bracket order."""
    bracket_id: str
    entry_order: Order
    take_profit_order: Order
    stop_loss_order: Order
    status: BracketOrderStatus
    error_message: Optional[str] = None


class BracketOrder:
    """
    Bracket Order: Entry + Take Profit + Stop Loss as atomic unit.

    A bracket order combines three orders:
    1. Entry order (market or limit) - initiates the position
    2. Take profit order (limit) - closes position at profit target
    3. Stop loss order (stop) - closes position at stop price

    The take profit and stop loss are OCO (one-cancels-other), meaning
    when one fills, the other is automatically cancelled.

    Example:
        bracket = BracketOrder(
            broker=broker,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            entry_price=150.00,
            take_profit_price=165.00,  # +10%
            stop_loss_price=142.50,    # -5%
        )
        result = bracket.create()

        # Later, modify the stop
        bracket.modify_stop(new_stop_price=148.00)

        # Or cancel the entire bracket
        bracket.cancel()
    """

    def __init__(
        self,
        broker: BrokerInterface,
        symbol: str,
        side: OrderSide,
        quantity: int,
        entry_price: Optional[float] = None,
        take_profit_price: float = 0.0,
        stop_loss_price: float = 0.0,
        entry_type: OrderType = OrderType.LIMIT,
        time_in_force: str = "GTC",
        on_entry_fill: Optional[Callable[[Order], None]] = None,
        on_exit_fill: Optional[Callable[[Order, str], None]] = None,
    ):
        """
        Initialize bracket order.

        Args:
            broker: Broker interface to use
            symbol: Stock ticker symbol
            side: OrderSide.BUY for long, OrderSide.SELL_SHORT for short
            quantity: Number of shares
            entry_price: Entry limit price (None for market order)
            take_profit_price: Target price for profit
            stop_loss_price: Stop loss trigger price
            entry_type: MARKET or LIMIT for entry
            time_in_force: Order duration (DAY, GTC, etc.)
            on_entry_fill: Callback when entry order fills
            on_exit_fill: Callback when exit order fills (order, "take_profit"|"stop_loss")
        """
        self.broker = broker
        self.symbol = symbol.upper()
        self.side = side
        self.quantity = quantity
        self.entry_price = entry_price
        self.take_profit_price = take_profit_price
        self.stop_loss_price = stop_loss_price
        self.entry_type = entry_type
        self.time_in_force = time_in_force
        self.on_entry_fill = on_entry_fill
        self.on_exit_fill = on_exit_fill

        # Order tracking
        self.bracket_id = f"bracket_{uuid.uuid4().hex[:12]}"
        self.entry_order: Optional[Order] = None
        self.take_profit_order: Optional[Order] = None
        self.stop_loss_order: Optional[Order] = None

        # Status
        self.status = BracketOrderStatus.PENDING
        self.created_at = datetime.now()
        self.filled_at: Optional[datetime] = None
        self.exit_reason: Optional[str] = None

        # Validate prices
        self._validate_prices()

        logger.debug(
            f"BracketOrder initialized: {self.bracket_id} - "
            f"{side.value} {quantity} {symbol}"
        )

    def _validate_prices(self) -> None:
        """Validate bracket order prices are sensible."""
        if self.side in (OrderSide.BUY, OrderSide.BUY_TO_COVER):
            # Long position
            if self.entry_price is not None:
                if self.take_profit_price <= self.entry_price:
                    raise ValueError(
                        f"Take profit ({self.take_profit_price}) must be above "
                        f"entry ({self.entry_price}) for long positions"
                    )
                if self.stop_loss_price >= self.entry_price:
                    raise ValueError(
                        f"Stop loss ({self.stop_loss_price}) must be below "
                        f"entry ({self.entry_price}) for long positions"
                    )
            if self.stop_loss_price >= self.take_profit_price:
                raise ValueError(
                    f"Stop loss ({self.stop_loss_price}) must be below "
                    f"take profit ({self.take_profit_price})"
                )
        else:
            # Short position
            if self.entry_price is not None:
                if self.take_profit_price >= self.entry_price:
                    raise ValueError(
                        f"Take profit ({self.take_profit_price}) must be below "
                        f"entry ({self.entry_price}) for short positions"
                    )
                if self.stop_loss_price <= self.entry_price:
                    raise ValueError(
                        f"Stop loss ({self.stop_loss_price}) must be above "
                        f"entry ({self.entry_price}) for short positions"
                    )
            if self.stop_loss_price <= self.take_profit_price:
                raise ValueError(
                    f"Stop loss ({self.stop_loss_price}) must be above "
                    f"take profit ({self.take_profit_price}) for short positions"
                )

    def _get_exit_side(self) -> OrderSide:
        """Get the exit order side (opposite of entry)."""
        if self.side in (OrderSide.BUY, OrderSide.BUY_TO_COVER):
            return OrderSide.SELL
        return OrderSide.BUY_TO_COVER

    def create(self) -> BracketOrderResult:
        """
        Create and submit the bracket order.

        Returns:
            BracketOrderResult with order details and status

        Raises:
            OrderError: If order placement fails
        """
        try:
            # Check if broker supports native bracket orders
            if hasattr(self.broker, 'place_bracket_order'):
                return self._create_native_bracket()
            else:
                return self._create_simulated_bracket()

        except Exception as e:
            logger.error(f"Failed to create bracket order: {e}")
            self.status = BracketOrderStatus.REJECTED
            return BracketOrderResult(
                bracket_id=self.bracket_id,
                entry_order=self.entry_order or Order(
                    order_id="",
                    symbol=self.symbol,
                    side=self.side,
                    quantity=self.quantity,
                    status=OrderStatus.REJECTED
                ),
                take_profit_order=self.take_profit_order or Order(
                    order_id="",
                    symbol=self.symbol,
                    side=self._get_exit_side(),
                    quantity=self.quantity,
                    status=OrderStatus.REJECTED
                ),
                stop_loss_order=self.stop_loss_order or Order(
                    order_id="",
                    symbol=self.symbol,
                    side=self._get_exit_side(),
                    quantity=self.quantity,
                    status=OrderStatus.REJECTED
                ),
                status=BracketOrderStatus.REJECTED,
                error_message=str(e)
            )

    def _create_native_bracket(self) -> BracketOrderResult:
        """Create bracket using broker's native support."""
        entry, tp, sl = self.broker.place_bracket_order(
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            entry_price=self.entry_price,
            take_profit_price=self.take_profit_price,
            stop_loss_price=self.stop_loss_price,
            entry_type=self.entry_type
        )

        self.entry_order = entry
        self.take_profit_order = tp
        self.stop_loss_order = sl

        self.status = BracketOrderStatus.PENDING
        if entry.status == OrderStatus.FILLED:
            self.status = BracketOrderStatus.ENTRY_FILLED
            self.filled_at = datetime.now()

        logger.info(
            f"Bracket order created: {self.bracket_id} - "
            f"Entry: {entry.order_id}, TP: {tp.order_id}, SL: {sl.order_id}"
        )

        return BracketOrderResult(
            bracket_id=self.bracket_id,
            entry_order=entry,
            take_profit_order=tp,
            stop_loss_order=sl,
            status=self.status
        )

    def _create_simulated_bracket(self) -> BracketOrderResult:
        """Create bracket using individual orders."""
        exit_side = self._get_exit_side()

        # Place entry order
        self.entry_order = self.broker.place_order(
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            order_type=self.entry_type,
            price=self.entry_price,
            time_in_force=self.time_in_force
        )

        # If entry is filled immediately, place exit orders
        if self.entry_order.status == OrderStatus.FILLED:
            self.status = BracketOrderStatus.ENTRY_FILLED
            self.filled_at = datetime.now()

            # Place take profit order
            self.take_profit_order = self.broker.place_order(
                symbol=self.symbol,
                side=exit_side,
                quantity=self.quantity,
                order_type=OrderType.LIMIT,
                price=self.take_profit_price,
                time_in_force=self.time_in_force
            )

            # Place stop loss order
            self.stop_loss_order = self.broker.place_order(
                symbol=self.symbol,
                side=exit_side,
                quantity=self.quantity,
                order_type=OrderType.STOP,
                stop_price=self.stop_loss_price,
                time_in_force=self.time_in_force
            )

            if self.on_entry_fill:
                self.on_entry_fill(self.entry_order)
        else:
            # Entry not filled yet - create placeholder orders
            self.take_profit_order = Order(
                order_id=f"{self.bracket_id}_tp_pending",
                symbol=self.symbol,
                side=exit_side,
                quantity=self.quantity,
                order_type=OrderType.LIMIT,
                price=self.take_profit_price,
                status=OrderStatus.PENDING
            )
            self.stop_loss_order = Order(
                order_id=f"{self.bracket_id}_sl_pending",
                symbol=self.symbol,
                side=exit_side,
                quantity=self.quantity,
                order_type=OrderType.STOP,
                stop_price=self.stop_loss_price,
                status=OrderStatus.PENDING
            )

        logger.info(
            f"Simulated bracket order created: {self.bracket_id}"
        )

        return BracketOrderResult(
            bracket_id=self.bracket_id,
            entry_order=self.entry_order,
            take_profit_order=self.take_profit_order,
            stop_loss_order=self.stop_loss_order,
            status=self.status
        )

    def cancel(self) -> bool:
        """
        Cancel the entire bracket order.

        Returns:
            True if cancellation successful
        """
        success = True

        # Cancel entry if still pending
        if self.entry_order and self.entry_order.is_active:
            try:
                if not self.broker.cancel_order(self.entry_order.order_id):
                    success = False
                    logger.warning(
                        f"Failed to cancel entry order: {self.entry_order.order_id}"
                    )
            except Exception as e:
                logger.error(f"Error cancelling entry order: {e}")
                success = False

        # Cancel take profit
        if self.take_profit_order and self.take_profit_order.is_active:
            try:
                if not self.broker.cancel_order(self.take_profit_order.order_id):
                    success = False
                    logger.warning(
                        f"Failed to cancel take profit order: {self.take_profit_order.order_id}"
                    )
            except Exception as e:
                logger.error(f"Error cancelling take profit order: {e}")
                success = False

        # Cancel stop loss
        if self.stop_loss_order and self.stop_loss_order.is_active:
            try:
                if not self.broker.cancel_order(self.stop_loss_order.order_id):
                    success = False
                    logger.warning(
                        f"Failed to cancel stop loss order: {self.stop_loss_order.order_id}"
                    )
            except Exception as e:
                logger.error(f"Error cancelling stop loss order: {e}")
                success = False

        if success:
            self.status = BracketOrderStatus.CANCELLED
            logger.info(f"Bracket order cancelled: {self.bracket_id}")

        return success

    def modify_stop(self, new_stop_price: float) -> bool:
        """
        Modify the stop loss price.

        Args:
            new_stop_price: New stop loss trigger price

        Returns:
            True if modification successful
        """
        if not self.stop_loss_order:
            logger.warning("No stop loss order to modify")
            return False

        # Validate new price
        if self.side in (OrderSide.BUY, OrderSide.BUY_TO_COVER):
            if new_stop_price >= self.take_profit_price:
                raise ValueError("Stop loss must be below take profit for long positions")
        else:
            if new_stop_price <= self.take_profit_price:
                raise ValueError("Stop loss must be above take profit for short positions")

        # Check if broker supports order modification
        if hasattr(self.broker, 'modify_order'):
            try:
                success = self.broker.modify_order(
                    order_id=self.stop_loss_order.order_id,
                    stop_price=new_stop_price
                )
                if success:
                    self.stop_loss_price = new_stop_price
                    self.stop_loss_order.stop_price = new_stop_price
                    logger.info(
                        f"Stop loss modified to {new_stop_price} for {self.bracket_id}"
                    )
                return success
            except Exception as e:
                logger.error(f"Failed to modify stop loss: {e}")
                return False
        else:
            # Cancel and replace
            try:
                if self.stop_loss_order.is_active:
                    self.broker.cancel_order(self.stop_loss_order.order_id)

                exit_side = self._get_exit_side()
                self.stop_loss_order = self.broker.place_order(
                    symbol=self.symbol,
                    side=exit_side,
                    quantity=self.quantity,
                    order_type=OrderType.STOP,
                    stop_price=new_stop_price,
                    time_in_force=self.time_in_force
                )
                self.stop_loss_price = new_stop_price
                logger.info(
                    f"Stop loss replaced at {new_stop_price} for {self.bracket_id}"
                )
                return True
            except Exception as e:
                logger.error(f"Failed to replace stop loss: {e}")
                return False

    def modify_target(self, new_target_price: float) -> bool:
        """
        Modify the take profit price.

        Args:
            new_target_price: New take profit price

        Returns:
            True if modification successful
        """
        if not self.take_profit_order:
            logger.warning("No take profit order to modify")
            return False

        # Validate new price
        if self.side in (OrderSide.BUY, OrderSide.BUY_TO_COVER):
            if new_target_price <= self.stop_loss_price:
                raise ValueError("Take profit must be above stop loss for long positions")
        else:
            if new_target_price >= self.stop_loss_price:
                raise ValueError("Take profit must be below stop loss for short positions")

        # Check if broker supports order modification
        if hasattr(self.broker, 'modify_order'):
            try:
                success = self.broker.modify_order(
                    order_id=self.take_profit_order.order_id,
                    price=new_target_price
                )
                if success:
                    self.take_profit_price = new_target_price
                    self.take_profit_order.price = new_target_price
                    logger.info(
                        f"Take profit modified to {new_target_price} for {self.bracket_id}"
                    )
                return success
            except Exception as e:
                logger.error(f"Failed to modify take profit: {e}")
                return False
        else:
            # Cancel and replace
            try:
                if self.take_profit_order.is_active:
                    self.broker.cancel_order(self.take_profit_order.order_id)

                exit_side = self._get_exit_side()
                self.take_profit_order = self.broker.place_order(
                    symbol=self.symbol,
                    side=exit_side,
                    quantity=self.quantity,
                    order_type=OrderType.LIMIT,
                    price=new_target_price,
                    time_in_force=self.time_in_force
                )
                self.take_profit_price = new_target_price
                logger.info(
                    f"Take profit replaced at {new_target_price} for {self.bracket_id}"
                )
                return True
            except Exception as e:
                logger.error(f"Failed to replace take profit: {e}")
                return False

    def update_status(self) -> BracketOrderStatus:
        """
        Update and return current bracket status.

        Returns:
            Current BracketOrderStatus
        """
        # Check entry order
        if self.entry_order:
            entry_status = self.broker.get_order_status(self.entry_order.order_id)
            if entry_status:
                self.entry_order = entry_status

        # Check take profit order
        if self.take_profit_order and self.take_profit_order.order_id:
            tp_status = self.broker.get_order_status(self.take_profit_order.order_id)
            if tp_status:
                self.take_profit_order = tp_status

        # Check stop loss order
        if self.stop_loss_order and self.stop_loss_order.order_id:
            sl_status = self.broker.get_order_status(self.stop_loss_order.order_id)
            if sl_status:
                self.stop_loss_order = sl_status

        # Determine overall status
        if self.entry_order and self.entry_order.status == OrderStatus.FILLED:
            if self.take_profit_order and self.take_profit_order.status == OrderStatus.FILLED:
                self.status = BracketOrderStatus.TAKE_PROFIT_FILLED
                self.exit_reason = "take_profit"
                if self.on_exit_fill:
                    self.on_exit_fill(self.take_profit_order, "take_profit")
            elif self.stop_loss_order and self.stop_loss_order.status == OrderStatus.FILLED:
                self.status = BracketOrderStatus.STOP_LOSS_FILLED
                self.exit_reason = "stop_loss"
                if self.on_exit_fill:
                    self.on_exit_fill(self.stop_loss_order, "stop_loss")
            else:
                self.status = BracketOrderStatus.ENTRY_FILLED
        elif self.entry_order and self.entry_order.status == OrderStatus.CANCELLED:
            self.status = BracketOrderStatus.CANCELLED
        elif self.entry_order and self.entry_order.status == OrderStatus.REJECTED:
            self.status = BracketOrderStatus.REJECTED

        return self.status

    def get_order_ids(self) -> Dict[str, str]:
        """
        Get all order IDs in this bracket.

        Returns:
            Dictionary with entry, take_profit, stop_loss order IDs
        """
        return {
            "entry": self.entry_order.order_id if self.entry_order else "",
            "take_profit": self.take_profit_order.order_id if self.take_profit_order else "",
            "stop_loss": self.stop_loss_order.order_id if self.stop_loss_order else "",
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert bracket order to dictionary."""
        return {
            "bracket_id": self.bracket_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "take_profit_price": self.take_profit_price,
            "stop_loss_price": self.stop_loss_price,
            "entry_type": self.entry_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "exit_reason": self.exit_reason,
            "order_ids": self.get_order_ids(),
        }


class TrailingStopOrder:
    """
    Trailing Stop Order with dynamic stop price adjustment.

    A trailing stop follows the price by a fixed amount or percentage,
    locking in profits as the price moves favorably while protecting
    against reversals.

    For long positions:
    - Stop trails below the highest price seen
    - As price rises, stop rises (never decreases)
    - If price falls to stop, position closes

    For short positions:
    - Stop trails above the lowest price seen
    - As price falls, stop falls (never increases)
    - If price rises to stop, position closes

    Example:
        # Fixed amount trailing stop ($2.00)
        trail = TrailingStopOrder(
            broker=broker,
            symbol="AAPL",
            side=OrderSide.SELL,  # Exit side for a long position
            quantity=100,
            trail_amount=2.00,
            activation_price=155.00  # Optional: only start trailing at this price
        )
        result = trail.create()

        # Or percentage trailing stop (5%)
        trail = TrailingStopOrder(
            broker=broker,
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            trail_percent=5.0
        )
    """

    def __init__(
        self,
        broker: BrokerInterface,
        symbol: str,
        side: OrderSide,
        quantity: int,
        trail_amount: Optional[float] = None,
        trail_percent: Optional[float] = None,
        activation_price: Optional[float] = None,
        time_in_force: str = "GTC",
        on_triggered: Optional[Callable[["TrailingStopOrder"], None]] = None,
    ):
        """
        Initialize trailing stop order.

        Args:
            broker: Broker interface to use
            symbol: Stock ticker symbol
            side: SELL for long exit, BUY_TO_COVER for short exit
            quantity: Number of shares
            trail_amount: Trail by fixed dollar amount (mutually exclusive with trail_percent)
            trail_percent: Trail by percentage (mutually exclusive with trail_amount)
            activation_price: Price at which trailing begins (optional)
            time_in_force: Order duration
            on_triggered: Callback when stop is triggered
        """
        if trail_amount is None and trail_percent is None:
            raise ValueError("Either trail_amount or trail_percent must be specified")
        if trail_amount is not None and trail_percent is not None:
            raise ValueError("Cannot specify both trail_amount and trail_percent")

        self.broker = broker
        self.symbol = symbol.upper()
        self.side = side
        self.quantity = quantity
        self.trail_amount = trail_amount
        self.trail_percent = trail_percent
        self.activation_price = activation_price
        self.time_in_force = time_in_force
        self.on_triggered = on_triggered

        # Determine trail type and direction
        self.trail_type = (
            TrailingStopType.FIXED_AMOUNT if trail_amount
            else TrailingStopType.PERCENTAGE
        )

        # Direction: SELL = long position (stop below price), BUY = short position (stop above)
        self.is_long_exit = side in (OrderSide.SELL, OrderSide.SELL_SHORT)

        # Order tracking
        self.order_id = f"trail_{uuid.uuid4().hex[:12]}"
        self.broker_order: Optional[Order] = None

        # Trailing state
        self.current_stop: Optional[float] = None
        self.reference_price: Optional[float] = None  # Highest (long) or lowest (short) price
        self.is_activated = activation_price is None
        self.is_triggered = False
        self.created_at = datetime.now()

        logger.debug(
            f"TrailingStopOrder initialized: {self.order_id} - "
            f"{side.value} {quantity} {symbol}, trail={trail_amount or trail_percent}%"
        )

    def create(self) -> Optional[Order]:
        """
        Create and submit the trailing stop order.

        Returns:
            Order object if placed, None if using simulated trailing
        """
        # Try native trailing stop first
        if hasattr(self.broker, 'place_trailing_stop'):
            try:
                self.broker_order = self.broker.place_trailing_stop(
                    symbol=self.symbol,
                    side=self.side,
                    quantity=self.quantity,
                    trail_amount=self.trail_amount,
                    trail_percent=self.trail_percent,
                    activation_price=self.activation_price,
                    time_in_force=self.time_in_force
                )
                logger.info(
                    f"Native trailing stop created: {self.broker_order.order_id}"
                )
                return self.broker_order
            except Exception as e:
                logger.warning(f"Native trailing stop failed, using simulation: {e}")

        # Initialize simulated trailing
        quote = self.broker.get_quote(self.symbol)
        self.reference_price = quote.last
        self.current_stop = self._calculate_stop(quote.last)

        logger.info(
            f"Simulated trailing stop created: {self.order_id}, "
            f"initial stop={self.current_stop}"
        )

        return None

    def _calculate_stop(self, price: float) -> float:
        """Calculate stop price based on reference price."""
        if self.trail_type == TrailingStopType.FIXED_AMOUNT:
            trail = self.trail_amount
        else:
            trail = price * (self.trail_percent / 100.0)

        if self.is_long_exit:
            return price - trail
        else:
            return price + trail

    def update_trail(self, current_price: Optional[float] = None) -> Tuple[float, bool]:
        """
        Update the trailing stop based on current price.

        Args:
            current_price: Current market price (fetched if not provided)

        Returns:
            Tuple of (current_stop_price, was_triggered)
        """
        if self.is_triggered:
            return self.current_stop or 0.0, True

        # Get current price if not provided
        if current_price is None:
            quote = self.broker.get_quote(self.symbol)
            current_price = quote.last

        # Check activation
        if not self.is_activated and self.activation_price:
            if self.is_long_exit:
                if current_price >= self.activation_price:
                    self.is_activated = True
                    self.reference_price = current_price
                    logger.info(
                        f"Trailing stop activated at {current_price} for {self.order_id}"
                    )
            else:
                if current_price <= self.activation_price:
                    self.is_activated = True
                    self.reference_price = current_price
                    logger.info(
                        f"Trailing stop activated at {current_price} for {self.order_id}"
                    )

        if not self.is_activated:
            return 0.0, False

        # Update reference price (track extreme)
        if self.reference_price is None:
            self.reference_price = current_price

        if self.is_long_exit:
            # Long exit: track highest price
            if current_price > self.reference_price:
                self.reference_price = current_price
                new_stop = self._calculate_stop(current_price)
                if self.current_stop is None or new_stop > self.current_stop:
                    old_stop = self.current_stop
                    self.current_stop = new_stop
                    logger.debug(
                        f"Trail updated: {old_stop} -> {new_stop} for {self.order_id}"
                    )
        else:
            # Short exit: track lowest price
            if current_price < self.reference_price:
                self.reference_price = current_price
                new_stop = self._calculate_stop(current_price)
                if self.current_stop is None or new_stop < self.current_stop:
                    old_stop = self.current_stop
                    self.current_stop = new_stop
                    logger.debug(
                        f"Trail updated: {old_stop} -> {new_stop} for {self.order_id}"
                    )

        # Calculate current stop if not set
        if self.current_stop is None:
            self.current_stop = self._calculate_stop(self.reference_price)

        # Check if stop triggered
        triggered = False
        if self.is_long_exit:
            if current_price <= self.current_stop:
                triggered = True
        else:
            if current_price >= self.current_stop:
                triggered = True

        if triggered and not self.is_triggered:
            self.is_triggered = True
            logger.info(
                f"Trailing stop TRIGGERED at {current_price}, "
                f"stop={self.current_stop} for {self.order_id}"
            )

            # Place actual stop order
            try:
                self.broker_order = self.broker.place_order(
                    symbol=self.symbol,
                    side=self.side,
                    quantity=self.quantity,
                    order_type=OrderType.MARKET
                )
            except Exception as e:
                logger.error(f"Failed to place stop exit order: {e}")

            if self.on_triggered:
                self.on_triggered(self)

        return self.current_stop, triggered

    def get_current_stop(self) -> Optional[float]:
        """Get the current stop price."""
        return self.current_stop

    def cancel(self) -> bool:
        """
        Cancel the trailing stop order.

        Returns:
            True if cancellation successful
        """
        if self.broker_order:
            try:
                result = self.broker.cancel_order(self.broker_order.order_id)
                if result:
                    logger.info(f"Trailing stop cancelled: {self.order_id}")
                return result
            except Exception as e:
                logger.error(f"Failed to cancel trailing stop: {e}")
                return False

        # Mark simulated trail as cancelled
        self.is_triggered = True
        logger.info(f"Simulated trailing stop cancelled: {self.order_id}")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert trailing stop to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "trail_type": self.trail_type.value,
            "trail_amount": self.trail_amount,
            "trail_percent": self.trail_percent,
            "activation_price": self.activation_price,
            "current_stop": self.current_stop,
            "reference_price": self.reference_price,
            "is_activated": self.is_activated,
            "is_triggered": self.is_triggered,
            "created_at": self.created_at.isoformat(),
        }


class OCOOrderStatus(str, Enum):
    """Status of an OCO order group."""
    PENDING = "pending"
    ACTIVE = "active"
    ONE_FILLED = "one_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OCOOrder:
    """
    One-Cancels-Other (OCO) Order.

    Links two or more orders so that when one fills, the others are
    automatically cancelled. Commonly used for:
    - Exit orders: Take profit OR stop loss
    - Entry orders: Buy on breakout OR buy at support

    Example:
        # OCO exit: profit target or stop loss
        oco = OCOOrder(
            broker=broker,
            symbol="AAPL",
            orders=[
                {
                    "side": OrderSide.SELL,
                    "quantity": 100,
                    "order_type": OrderType.LIMIT,
                    "price": 165.00  # Take profit
                },
                {
                    "side": OrderSide.SELL,
                    "quantity": 100,
                    "order_type": OrderType.STOP,
                    "stop_price": 145.00  # Stop loss
                }
            ]
        )
        orders = oco.create()
    """

    def __init__(
        self,
        broker: BrokerInterface,
        symbol: str,
        orders: List[Dict[str, Any]],
        on_fill: Optional[Callable[[Order, int], None]] = None,
    ):
        """
        Initialize OCO order.

        Args:
            broker: Broker interface to use
            symbol: Stock ticker symbol
            orders: List of order specifications, each containing:
                - side: OrderSide
                - quantity: int
                - order_type: OrderType
                - price: float (for limit orders)
                - stop_price: float (for stop orders)
                - time_in_force: str (optional, default GTC)
            on_fill: Callback when one order fills (order, order_index)
        """
        if len(orders) < 2:
            raise ValueError("OCO requires at least 2 orders")

        self.broker = broker
        self.symbol = symbol.upper()
        self.order_specs = orders
        self.on_fill = on_fill

        # Order tracking
        self.oco_id = f"oco_{uuid.uuid4().hex[:12]}"
        self.orders: List[Order] = []

        # Status
        self.status = OCOOrderStatus.PENDING
        self.filled_index: Optional[int] = None
        self.created_at = datetime.now()

        logger.debug(
            f"OCOOrder initialized: {self.oco_id} - "
            f"{len(orders)} orders for {symbol}"
        )

    def create(self) -> List[Order]:
        """
        Create and submit the OCO orders.

        Returns:
            List of Order objects
        """
        # Try native OCO first
        if hasattr(self.broker, 'place_oco_order'):
            try:
                self.orders = self.broker.place_oco_order(
                    symbol=self.symbol,
                    orders=self.order_specs
                )
                self.status = OCOOrderStatus.ACTIVE
                logger.info(
                    f"Native OCO created: {self.oco_id} with "
                    f"{len(self.orders)} orders"
                )
                return self.orders
            except Exception as e:
                logger.warning(f"Native OCO failed, using simulation: {e}")

        # Create individual orders (simulated OCO)
        for spec in self.order_specs:
            try:
                order = self.broker.place_order(
                    symbol=self.symbol,
                    side=spec["side"],
                    quantity=spec["quantity"],
                    order_type=spec["order_type"],
                    price=spec.get("price"),
                    stop_price=spec.get("stop_price"),
                    time_in_force=spec.get("time_in_force", "GTC")
                )
                self.orders.append(order)
            except Exception as e:
                logger.error(f"Failed to place OCO order: {e}")
                # Cancel any orders already placed
                self.cancel()
                raise

        self.status = OCOOrderStatus.ACTIVE
        logger.info(
            f"Simulated OCO created: {self.oco_id} with "
            f"{len(self.orders)} orders"
        )

        return self.orders

    def check_and_cancel_others(self) -> bool:
        """
        Check if any order filled and cancel the others.

        Returns:
            True if one order filled
        """
        if self.status == OCOOrderStatus.ONE_FILLED:
            return True

        filled_idx = None

        for i, order in enumerate(self.orders):
            # Get updated status
            updated = self.broker.get_order_status(order.order_id)
            if updated:
                self.orders[i] = updated

                if updated.status == OrderStatus.FILLED:
                    filled_idx = i
                    break

        if filled_idx is not None:
            self.filled_index = filled_idx
            self.status = OCOOrderStatus.ONE_FILLED

            logger.info(
                f"OCO order {filled_idx} filled for {self.oco_id}, "
                f"cancelling others"
            )

            # Cancel other orders
            for i, order in enumerate(self.orders):
                if i != filled_idx and order.is_active:
                    try:
                        self.broker.cancel_order(order.order_id)
                    except Exception as e:
                        logger.warning(f"Failed to cancel OCO order {i}: {e}")

            if self.on_fill:
                self.on_fill(self.orders[filled_idx], filled_idx)

            return True

        return False

    def cancel(self) -> bool:
        """
        Cancel all orders in the OCO group.

        Returns:
            True if all cancellations successful
        """
        success = True

        for order in self.orders:
            if order.is_active:
                try:
                    if not self.broker.cancel_order(order.order_id):
                        success = False
                except Exception as e:
                    logger.error(f"Failed to cancel OCO order: {e}")
                    success = False

        if success:
            self.status = OCOOrderStatus.CANCELLED
            logger.info(f"OCO group cancelled: {self.oco_id}")

        return success

    def get_order_ids(self) -> List[str]:
        """Get all order IDs in this OCO group."""
        return [order.order_id for order in self.orders]

    def to_dict(self) -> Dict[str, Any]:
        """Convert OCO order to dictionary."""
        return {
            "oco_id": self.oco_id,
            "symbol": self.symbol,
            "status": self.status.value,
            "filled_index": self.filled_index,
            "created_at": self.created_at.isoformat(),
            "orders": [
                {
                    "order_id": o.order_id,
                    "side": o.side.value,
                    "quantity": o.quantity,
                    "order_type": o.order_type.value,
                    "price": o.price,
                    "stop_price": o.stop_price,
                    "status": o.status.value,
                }
                for o in self.orders
            ]
        }


# ============================================================================
# Advanced Order Manager
# ============================================================================

class AdvancedOrderManager:
    """
    Manages all advanced order types (brackets, trailing stops, OCO).

    Provides centralized tracking and update loop for simulated orders.

    Example:
        manager = AdvancedOrderManager(broker)

        # Create bracket order
        bracket = manager.create_bracket(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            entry_price=150.00,
            take_profit_price=165.00,
            stop_loss_price=142.50
        )

        # Update all orders
        manager.update()
    """

    def __init__(self, broker: BrokerInterface):
        """Initialize the advanced order manager."""
        self.broker = broker

        # Active orders
        self.brackets: Dict[str, BracketOrder] = {}
        self.trailing_stops: Dict[str, TrailingStopOrder] = {}
        self.oco_orders: Dict[str, OCOOrder] = {}

        logger.info("AdvancedOrderManager initialized")

    def create_bracket(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        entry_price: Optional[float],
        take_profit_price: float,
        stop_loss_price: float,
        entry_type: OrderType = OrderType.LIMIT,
        **kwargs
    ) -> BracketOrder:
        """Create and track a bracket order."""
        bracket = BracketOrder(
            broker=self.broker,
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            entry_type=entry_type,
            **kwargs
        )

        result = bracket.create()
        self.brackets[bracket.bracket_id] = bracket

        return bracket

    def create_trailing_stop(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        trail_amount: Optional[float] = None,
        trail_percent: Optional[float] = None,
        activation_price: Optional[float] = None,
        **kwargs
    ) -> TrailingStopOrder:
        """Create and track a trailing stop order."""
        trail = TrailingStopOrder(
            broker=self.broker,
            symbol=symbol,
            side=side,
            quantity=quantity,
            trail_amount=trail_amount,
            trail_percent=trail_percent,
            activation_price=activation_price,
            **kwargs
        )

        trail.create()
        self.trailing_stops[trail.order_id] = trail

        return trail

    def create_oco(
        self,
        symbol: str,
        orders: List[Dict[str, Any]],
        **kwargs
    ) -> OCOOrder:
        """Create and track an OCO order."""
        oco = OCOOrder(
            broker=self.broker,
            symbol=symbol,
            orders=orders,
            **kwargs
        )

        oco.create()
        self.oco_orders[oco.oco_id] = oco

        return oco

    def update(self) -> None:
        """
        Update all tracked orders.

        Should be called periodically (e.g., every second) to:
        - Update trailing stops
        - Check OCO fills
        - Update bracket statuses
        """
        # Update trailing stops
        for trail_id, trail in list(self.trailing_stops.items()):
            if not trail.is_triggered:
                trail.update_trail()
            else:
                # Remove triggered trails after some delay
                del self.trailing_stops[trail_id]

        # Check OCO fills
        for oco_id, oco in list(self.oco_orders.items()):
            if oco.status == OCOOrderStatus.ACTIVE:
                oco.check_and_cancel_others()
            elif oco.status in (OCOOrderStatus.ONE_FILLED, OCOOrderStatus.CANCELLED):
                del self.oco_orders[oco_id]

        # Update bracket statuses
        for bracket_id, bracket in list(self.brackets.items()):
            bracket.update_status()
            if bracket.status in (
                BracketOrderStatus.TAKE_PROFIT_FILLED,
                BracketOrderStatus.STOP_LOSS_FILLED,
                BracketOrderStatus.CANCELLED
            ):
                del self.brackets[bracket_id]

    def get_all_orders(self) -> Dict[str, Any]:
        """Get summary of all tracked orders."""
        return {
            "brackets": [b.to_dict() for b in self.brackets.values()],
            "trailing_stops": [t.to_dict() for t in self.trailing_stops.values()],
            "oco_orders": [o.to_dict() for o in self.oco_orders.values()],
        }

    def track_bracket(self, bracket: BracketOrder) -> None:
        """Track an externally created bracket order."""
        self.brackets[bracket.bracket_id] = bracket
        logger.debug(f"Tracking bracket order: {bracket.bracket_id}")

    def track_trailing_stop(self, trailing_stop: TrailingStopOrder) -> None:
        """Track an externally created trailing stop order."""
        self.trailing_stops[trailing_stop.trailing_stop_id] = trailing_stop
        logger.debug(f"Tracking trailing stop: {trailing_stop.trailing_stop_id}")

    def track_oco(self, oco: OCOOrder) -> None:
        """Track an externally created OCO order."""
        self.oco_orders[oco.oco_id] = oco
        logger.debug(f"Tracking OCO order: {oco.oco_id}")

    def get_trailing_stop(self, order_id: str) -> Optional[TrailingStopOrder]:
        """Get a trailing stop by its ID."""
        return self.trailing_stops.get(order_id)

    def get_bracket(self, bracket_id: str) -> Optional[BracketOrder]:
        """Get a bracket order by its ID."""
        return self.brackets.get(bracket_id)

    def get_oco(self, oco_id: str) -> Optional[OCOOrder]:
        """Get an OCO order by its ID."""
        return self.oco_orders.get(oco_id)

    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all tracked advanced orders for API responses."""
        return {
            "brackets": [
                {
                    "bracket_id": b.bracket_id,
                    "symbol": b.symbol,
                    "side": b.side.value,
                    "quantity": b.quantity,
                    "status": b.status.value if hasattr(b, 'status') else "unknown",
                    "entry_price": b.entry_price,
                    "take_profit_price": b.take_profit_price,
                    "stop_loss_price": b.stop_loss_price,
                }
                for b in self.brackets.values()
            ],
            "trailing_stops": [
                {
                    "trailing_stop_id": t.trailing_stop_id,
                    "symbol": t.symbol,
                    "side": t.side.value,
                    "quantity": t.quantity,
                    "trail_type": t.trail_type.value,
                    "trail_amount": t.trail_amount,
                    "trail_percent": t.trail_percent,
                    "current_stop_price": t.get_current_stop(),
                    "activation_price": t.activation_price,
                    "is_activated": t.is_activated,
                    "is_triggered": t.is_triggered,
                }
                for t in self.trailing_stops.values()
            ],
            "oco_orders": [
                {
                    "oco_id": o.oco_id,
                    "symbol": o.symbol,
                    "status": o.status.value if hasattr(o, 'status') else "unknown",
                    "orders": [
                        {"order_id": ord.order_id, "status": ord.status.value}
                        for ord in (o.orders if hasattr(o, 'orders') and o.orders else [])
                    ],
                }
                for o in self.oco_orders.values()
            ],
        }

    def cancel_all(self) -> None:
        """Cancel all tracked orders."""
        for bracket in self.brackets.values():
            bracket.cancel()

        for trail in self.trailing_stops.values():
            trail.cancel()

        for oco in self.oco_orders.values():
            oco.cancel()

        self.brackets.clear()
        self.trailing_stops.clear()
        self.oco_orders.clear()

        logger.info("All advanced orders cancelled")

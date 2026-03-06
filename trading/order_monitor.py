"""
Order Monitor for the RDT Trading System.

Provides comprehensive order lifecycle tracking with:
- State management (pending, submitted, partial_fill, filled, cancelled, rejected)
- Time-to-fill metrics
- Stuck order detection and alerting
- Order history tracking
"""

import time
import threading
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from loguru import logger


class OrderState(str, Enum):
    """Order lifecycle states."""
    PENDING = "pending"           # Order created but not submitted
    SUBMITTED = "submitted"       # Order submitted to broker
    PARTIAL_FILL = "partial_fill" # Order partially filled
    FILLED = "filled"            # Order completely filled
    CANCELLED = "cancelled"       # Order cancelled
    REJECTED = "rejected"        # Order rejected by broker


@dataclass
class MonitoredOrder:
    """An order being tracked by the monitor."""
    order_id: str
    symbol: str
    side: str  # buy, sell, buy_to_cover, sell_short
    quantity: int
    order_type: str  # market, limit, stop, stop_limit
    expected_price: float  # Price at time of order submission
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None

    # State tracking
    state: OrderState = OrderState.PENDING
    filled_quantity: int = 0
    avg_fill_price: Optional[float] = None

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    first_fill_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Broker info
    broker_order_id: Optional[str] = None
    error_message: Optional[str] = None

    # Fills history
    fills: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        """Check if order is in a terminal state."""
        return self.state in (OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED)

    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.state in (OrderState.PENDING, OrderState.SUBMITTED, OrderState.PARTIAL_FILL)

    @property
    def remaining_quantity(self) -> int:
        """Calculate unfilled quantity."""
        return self.quantity - self.filled_quantity

    @property
    def fill_rate(self) -> float:
        """Calculate fill rate as percentage."""
        if self.quantity == 0:
            return 0.0
        return (self.filled_quantity / self.quantity) * 100

    @property
    def time_to_first_fill(self) -> Optional[float]:
        """Time from submission to first fill in seconds."""
        if self.submitted_at and self.first_fill_at:
            return (self.first_fill_at - self.submitted_at).total_seconds()
        return None

    @property
    def time_to_complete(self) -> Optional[float]:
        """Time from submission to completion in seconds."""
        if self.submitted_at and self.completed_at:
            return (self.completed_at - self.submitted_at).total_seconds()
        return None

    @property
    def time_since_submission(self) -> Optional[float]:
        """Time since order was submitted in seconds."""
        if self.submitted_at:
            return (datetime.utcnow() - self.submitted_at).total_seconds()
        return None

    @property
    def slippage(self) -> Optional[float]:
        """Calculate slippage in price terms.
        Positive slippage always means unfavorable (paid more for buy, received less for sell).
        """
        if self.avg_fill_price is None:
            return None
        if self.side in ('buy', 'BUY', 'BUY_TO_COVER', 'buy_to_cover'):
            return self.avg_fill_price - self.expected_price
        else:
            return self.expected_price - self.avg_fill_price

    @property
    def slippage_pct(self) -> Optional[float]:
        """Calculate slippage as percentage."""
        if self.slippage is None or self.expected_price == 0:
            return None
        return (self.slippage / self.expected_price) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "expected_price": self.expected_price,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "state": self.state.value,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "avg_fill_price": self.avg_fill_price,
            "fill_rate": round(self.fill_rate, 2),
            "slippage": round(self.slippage, 4) if self.slippage else None,
            "slippage_pct": round(self.slippage_pct, 4) if self.slippage_pct else None,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "first_fill_at": self.first_fill_at.isoformat() if self.first_fill_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "time_to_first_fill_seconds": self.time_to_first_fill,
            "time_to_complete_seconds": self.time_to_complete,
            "broker_order_id": self.broker_order_id,
            "error_message": self.error_message,
            "fills_count": len(self.fills)
        }


class OrderMonitor:
    """
    Monitors order lifecycle and tracks fill quality metrics.

    Features:
    - Order state tracking with event history
    - Time-to-fill metrics
    - Stuck order detection with configurable thresholds
    - Alert callbacks for various conditions
    - Thread-safe operations

    Usage:
        monitor = OrderMonitor(
            stuck_order_threshold_seconds=60,
            on_fill=my_fill_callback,
            on_stuck_order=my_alert_callback
        )

        # Track a new order
        monitor.track_order(order_id, symbol, side, quantity, ...)

        # Update order state
        monitor.order_submitted(order_id, broker_order_id)
        monitor.order_filled(order_id, fill_price, fill_quantity)
    """

    def __init__(
        self,
        stuck_order_threshold_seconds: float = 60.0,
        partial_fill_alert_seconds: float = 30.0,
        high_slippage_threshold_pct: float = 0.5,
        check_interval_seconds: float = 5.0,
        on_fill: Optional[Callable[[MonitoredOrder, Dict[str, Any]], None]] = None,
        on_complete: Optional[Callable[[MonitoredOrder], None]] = None,
        on_stuck_order: Optional[Callable[[MonitoredOrder], None]] = None,
        on_high_slippage: Optional[Callable[[MonitoredOrder], None]] = None,
        on_rejection: Optional[Callable[[MonitoredOrder], None]] = None
    ):
        """
        Initialize Order Monitor.

        Args:
            stuck_order_threshold_seconds: Time before order is considered stuck
            partial_fill_alert_seconds: Time to alert on slow partial fill completion
            high_slippage_threshold_pct: Slippage percentage to trigger alert
            check_interval_seconds: Interval for stuck order checking
            on_fill: Callback when a fill occurs
            on_complete: Callback when order completes
            on_stuck_order: Callback when order is detected as stuck
            on_high_slippage: Callback when high slippage is detected
            on_rejection: Callback when order is rejected
        """
        self._orders: Dict[str, MonitoredOrder] = {}
        self._completed_orders: List[MonitoredOrder] = []
        self._lock = threading.RLock()

        # Configuration
        self.stuck_order_threshold_seconds = stuck_order_threshold_seconds
        self.partial_fill_alert_seconds = partial_fill_alert_seconds
        self.high_slippage_threshold_pct = high_slippage_threshold_pct
        self.check_interval_seconds = check_interval_seconds

        # Callbacks
        self._on_fill = on_fill
        self._on_complete = on_complete
        self._on_stuck_order = on_stuck_order
        self._on_high_slippage = on_high_slippage
        self._on_rejection = on_rejection

        # Background monitoring
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stuck_orders_alerted: set = set()

        # Metrics
        self._total_orders = 0
        self._total_fills = 0
        self._total_rejections = 0
        self._total_cancellations = 0

        logger.info("OrderMonitor initialized")

    def track_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str,
        expected_price: float,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> MonitoredOrder:
        """
        Start tracking a new order.

        Args:
            order_id: Unique order identifier
            symbol: Stock symbol
            side: Order side (buy, sell, etc.)
            quantity: Order quantity
            order_type: Type of order (market, limit, etc.)
            expected_price: Expected fill price at order creation
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders

        Returns:
            MonitoredOrder object
        """
        with self._lock:
            order = MonitoredOrder(
                order_id=order_id,
                symbol=symbol.upper(),
                side=side.lower(),
                quantity=quantity,
                order_type=order_type.lower(),
                expected_price=expected_price,
                limit_price=limit_price,
                stop_price=stop_price
            )

            self._orders[order_id] = order
            self._total_orders += 1

            logger.info(
                f"Tracking order {order_id}: {side} {quantity} {symbol} "
                f"@ expected ${expected_price:.2f}"
            )

            return order

    def order_submitted(
        self,
        order_id: str,
        broker_order_id: Optional[str] = None
    ) -> Optional[MonitoredOrder]:
        """
        Mark order as submitted to broker.

        Args:
            order_id: Order ID
            broker_order_id: Broker's order ID

        Returns:
            Updated MonitoredOrder or None if not found
        """
        with self._lock:
            order = self._orders.get(order_id)
            if not order:
                logger.warning(f"Order {order_id} not found for submission update")
                return None

            order.state = OrderState.SUBMITTED
            order.submitted_at = datetime.utcnow()
            order.broker_order_id = broker_order_id

            logger.debug(f"Order {order_id} submitted to broker")
            return order

    def order_fill(
        self,
        order_id: str,
        fill_price: float,
        fill_quantity: int,
        timestamp: Optional[datetime] = None
    ) -> Optional[MonitoredOrder]:
        """
        Record a fill for an order.

        Args:
            order_id: Order ID
            fill_price: Price of this fill
            fill_quantity: Quantity of this fill
            timestamp: Fill timestamp (defaults to now)

        Returns:
            Updated MonitoredOrder or None if not found
        """
        with self._lock:
            order = self._orders.get(order_id)
            if not order:
                logger.warning(f"Order {order_id} not found for fill update")
                return None

            fill_time = timestamp or datetime.utcnow()

            # Record first fill time
            if order.first_fill_at is None:
                order.first_fill_at = fill_time

            # Record fill
            fill = {
                "fill_price": fill_price,
                "fill_quantity": fill_quantity,
                "timestamp": fill_time.isoformat()
            }
            order.fills.append(fill)

            # Update filled quantity
            old_filled = order.filled_quantity
            order.filled_quantity += fill_quantity

            # Calculate average fill price
            total_value = 0.0
            total_qty = 0
            for f in order.fills:
                total_value += f["fill_price"] * f["fill_quantity"]
                total_qty += f["fill_quantity"]
            order.avg_fill_price = total_value / total_qty if total_qty > 0 else None

            # Update state
            if order.filled_quantity >= order.quantity:
                order.state = OrderState.FILLED
                order.completed_at = fill_time
                self._complete_order(order)
            elif order.filled_quantity > 0:
                order.state = OrderState.PARTIAL_FILL

            self._total_fills += 1

            logger.info(
                f"Order {order_id} fill: {fill_quantity} @ ${fill_price:.2f} "
                f"({order.filled_quantity}/{order.quantity} filled)"
            )

            # Trigger fill callback
            if self._on_fill:
                try:
                    self._on_fill(order, fill)
                except Exception as e:
                    logger.error(f"Error in fill callback: {e}")

            return order

    def order_cancelled(
        self,
        order_id: str,
        reason: Optional[str] = None
    ) -> Optional[MonitoredOrder]:
        """
        Mark order as cancelled.

        Args:
            order_id: Order ID
            reason: Cancellation reason

        Returns:
            Updated MonitoredOrder or None if not found
        """
        with self._lock:
            order = self._orders.get(order_id)
            if not order:
                logger.warning(f"Order {order_id} not found for cancellation")
                return None

            order.state = OrderState.CANCELLED
            order.completed_at = datetime.utcnow()
            order.error_message = reason

            self._total_cancellations += 1
            self._complete_order(order)

            logger.info(f"Order {order_id} cancelled: {reason or 'No reason provided'}")
            return order

    def order_rejected(
        self,
        order_id: str,
        reason: str
    ) -> Optional[MonitoredOrder]:
        """
        Mark order as rejected.

        Args:
            order_id: Order ID
            reason: Rejection reason

        Returns:
            Updated MonitoredOrder or None if not found
        """
        with self._lock:
            order = self._orders.get(order_id)
            if not order:
                logger.warning(f"Order {order_id} not found for rejection")
                return None

            order.state = OrderState.REJECTED
            order.completed_at = datetime.utcnow()
            order.error_message = reason

            self._total_rejections += 1
            self._complete_order(order)

            logger.warning(f"Order {order_id} rejected: {reason}")

            # Trigger rejection callback
            if self._on_rejection:
                try:
                    self._on_rejection(order)
                except Exception as e:
                    logger.error(f"Error in rejection callback: {e}")

            return order

    def _complete_order(self, order: MonitoredOrder) -> None:
        """Move completed order to history and trigger callbacks."""
        # Move to completed list
        self._completed_orders.append(order)
        # Limit completed orders history to prevent unbounded growth
        MAX_COMPLETED_ORDERS = 1000
        if len(self._completed_orders) > MAX_COMPLETED_ORDERS:
            self._completed_orders = self._completed_orders[-MAX_COMPLETED_ORDERS:]
        if order.order_id in self._orders:
            del self._orders[order.order_id]

        # Check for high slippage
        if order.state == OrderState.FILLED and order.slippage_pct is not None:
            if abs(order.slippage_pct) >= self.high_slippage_threshold_pct:
                logger.warning(
                    f"High slippage on order {order.order_id}: "
                    f"{order.slippage_pct:.4f}% (${order.slippage:.4f})"
                )
                if self._on_high_slippage:
                    try:
                        self._on_high_slippage(order)
                    except Exception as e:
                        logger.error(f"Error in high slippage callback: {e}")

        # Trigger completion callback
        if self._on_complete:
            try:
                self._on_complete(order)
            except Exception as e:
                logger.error(f"Error in completion callback: {e}")

    def get_order(self, order_id: str) -> Optional[MonitoredOrder]:
        """Get an order by ID (active or completed)."""
        with self._lock:
            # Check active orders
            order = self._orders.get(order_id)
            if order:
                return order

            # Check completed orders
            for completed in self._completed_orders:
                if completed.order_id == order_id:
                    return completed

            return None

    def get_active_orders(self) -> List[MonitoredOrder]:
        """Get all active (non-terminal) orders."""
        with self._lock:
            return list(self._orders.values())

    def get_orders_by_symbol(self, symbol: str) -> List[MonitoredOrder]:
        """Get all orders for a symbol (active and completed)."""
        symbol = symbol.upper()
        with self._lock:
            orders = []
            for order in self._orders.values():
                if order.symbol == symbol:
                    orders.append(order)
            for order in self._completed_orders:
                if order.symbol == symbol:
                    orders.append(order)
            return orders

    def get_stuck_orders(self) -> List[MonitoredOrder]:
        """Get orders that appear to be stuck."""
        stuck = []
        with self._lock:
            now = datetime.utcnow()
            for order in self._orders.values():
                if order.state == OrderState.SUBMITTED:
                    if order.submitted_at:
                        elapsed = (now - order.submitted_at).total_seconds()
                        if elapsed >= self.stuck_order_threshold_seconds:
                            stuck.append(order)
                elif order.state == OrderState.PARTIAL_FILL:
                    # Check time since last fill
                    if order.fills:
                        last_fill_time = datetime.fromisoformat(
                            order.fills[-1]["timestamp"]
                        )
                        elapsed = (now - last_fill_time).total_seconds()
                        if elapsed >= self.partial_fill_alert_seconds:
                            stuck.append(order)
        return stuck

    def get_metrics(self) -> Dict[str, Any]:
        """Get order monitoring metrics."""
        with self._lock:
            active_count = len(self._orders)
            completed_count = len(self._completed_orders)

            # Calculate fill metrics from completed orders
            filled_orders = [
                o for o in self._completed_orders
                if o.state == OrderState.FILLED
            ]

            if filled_orders:
                avg_fill_time = sum(
                    o.time_to_complete or 0 for o in filled_orders
                ) / len(filled_orders)

                slippages = [
                    o.slippage_pct for o in filled_orders
                    if o.slippage_pct is not None
                ]
                avg_slippage = sum(slippages) / len(slippages) if slippages else 0
                max_slippage = max(slippages, default=0)
                min_slippage = min(slippages, default=0)
            else:
                avg_fill_time = 0
                avg_slippage = 0
                max_slippage = 0
                min_slippage = 0

            return {
                "active_orders": active_count,
                "completed_orders": completed_count,
                "total_orders_tracked": self._total_orders,
                "total_fills": self._total_fills,
                "total_rejections": self._total_rejections,
                "total_cancellations": self._total_cancellations,
                "filled_orders_count": len(filled_orders),
                "avg_fill_time_seconds": round(avg_fill_time, 3),
                "avg_slippage_pct": round(avg_slippage, 4),
                "max_slippage_pct": round(max_slippage, 4),
                "min_slippage_pct": round(min_slippage, 4),
                "stuck_orders_count": len(self.get_stuck_orders())
            }

    def get_fill_time_stats(self) -> Dict[str, Any]:
        """Get detailed fill time statistics."""
        with self._lock:
            filled_orders = [
                o for o in self._completed_orders
                if o.state == OrderState.FILLED
            ]

            if not filled_orders:
                return {
                    "count": 0,
                    "avg_seconds": 0,
                    "min_seconds": 0,
                    "max_seconds": 0,
                    "median_seconds": 0
                }

            fill_times = [
                o.time_to_complete for o in filled_orders
                if o.time_to_complete is not None
            ]

            if not fill_times:
                return {
                    "count": len(filled_orders),
                    "avg_seconds": 0,
                    "min_seconds": 0,
                    "max_seconds": 0,
                    "median_seconds": 0
                }

            fill_times.sort()
            median_idx = len(fill_times) // 2

            return {
                "count": len(fill_times),
                "avg_seconds": round(sum(fill_times) / len(fill_times), 3),
                "min_seconds": round(min(fill_times), 3),
                "max_seconds": round(max(fill_times), 3),
                "median_seconds": round(fill_times[median_idx], 3)
            }

    def _check_stuck_orders(self) -> None:
        """Background task to check for stuck orders."""
        stuck = self.get_stuck_orders()

        for order in stuck:
            if order.order_id not in self._stuck_orders_alerted:
                self._stuck_orders_alerted.add(order.order_id)
                logger.warning(
                    f"Stuck order detected: {order.order_id} "
                    f"({order.symbol} {order.side} {order.quantity})"
                )

                if self._on_stuck_order:
                    try:
                        self._on_stuck_order(order)
                    except Exception as e:
                        logger.error(f"Error in stuck order callback: {e}")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                self._check_stuck_orders()
            except Exception as e:
                logger.error(f"Error in order monitor loop: {e}")

            time.sleep(self.check_interval_seconds)

    def start(self) -> None:
        """Start background monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="OrderMonitor"
        )
        self._monitor_thread.start()
        logger.info("OrderMonitor background monitoring started")

    def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
        logger.info("OrderMonitor background monitoring stopped")

    def clear_history(self, keep_recent_hours: int = 24) -> int:
        """
        Clear old completed orders from history.

        Args:
            keep_recent_hours: Keep orders completed within this many hours

        Returns:
            Number of orders removed
        """
        cutoff = datetime.utcnow() - timedelta(hours=keep_recent_hours)
        removed = 0

        with self._lock:
            new_completed = []
            for order in self._completed_orders:
                if order.completed_at and order.completed_at >= cutoff:
                    new_completed.append(order)
                else:
                    removed += 1
            self._completed_orders = new_completed

        if removed > 0:
            logger.info(f"Cleared {removed} old orders from history")

        return removed


# Global order monitor instance
_order_monitor: Optional[OrderMonitor] = None
_order_monitor_lock = threading.Lock()


def get_order_monitor() -> OrderMonitor:
    """Get or create the global OrderMonitor instance."""
    global _order_monitor
    if _order_monitor is None:
        with _order_monitor_lock:
            if _order_monitor is None:
                _order_monitor = OrderMonitor()
    return _order_monitor

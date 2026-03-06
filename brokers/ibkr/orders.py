"""
IBKR Order Type Conversion and Advanced Order Support.

Handles conversion between internal order types and IBKR format,
as well as creation of complex order types like brackets and OCO orders.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger

from brokers.broker_interface import (
    OrderSide, OrderType, OrderStatus
)

try:
    import asyncio
    # Python 3.14 removed implicit event loop creation; eventkit (used by
    # ib_insync) crashes on import if no loop exists.  Ensure one is present.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop – create a default one so eventkit can initialise.
        asyncio.set_event_loop(asyncio.new_event_loop())

    from ib_insync import (
        Order as IBOrder,
        MarketOrder,
        LimitOrder,
        StopOrder,
        StopLimitOrder,
        BracketOrder,
    )
    # TrailingStopOrder not available in ib_insync 0.9.86
    try:
        from ib_insync import TrailingStopOrder
    except ImportError:
        TrailingStopOrder = None
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False


# ==================== Order Type Conversion ====================

def convert_order_type(order_type: OrderType) -> str:
    """
    Convert internal OrderType to IBKR order type string.

    Args:
        order_type: Internal OrderType enum

    Returns:
        IBKR order type string
    """
    mapping = {
        OrderType.MARKET: "MKT",
        OrderType.LIMIT: "LMT",
        OrderType.STOP: "STP",
        OrderType.STOP_LIMIT: "STP LMT",
        OrderType.TRAILING_STOP: "TRAIL",
    }
    return mapping.get(order_type, "MKT")


def convert_order_side(side: OrderSide) -> str:
    """
    Convert internal OrderSide to IBKR action string.

    Args:
        side: Internal OrderSide enum

    Returns:
        IBKR action string (BUY/SELL/SSHORT)
    """
    mapping = {
        OrderSide.BUY: "BUY",
        OrderSide.SELL: "SELL",
        OrderSide.BUY_TO_COVER: "BUY",
        OrderSide.SELL_SHORT: "SSHORT",
        # Options sides — IBKR uses BUY/SELL with contract type determining open/close
        OrderSide.BUY_TO_OPEN: "BUY",
        OrderSide.SELL_TO_OPEN: "SELL",
        OrderSide.BUY_TO_CLOSE: "BUY",
        OrderSide.SELL_TO_CLOSE: "SELL",
    }
    return mapping.get(side, "BUY")


def convert_time_in_force(tif: str) -> str:
    """
    Convert time-in-force string to IBKR format.

    Args:
        tif: Time-in-force string (DAY, GTC, IOC, etc.)

    Returns:
        IBKR time-in-force string
    """
    mapping = {
        "DAY": "DAY",
        "GTC": "GTC",
        "IOC": "IOC",  # Immediate-or-cancel
        "GTD": "GTD",  # Good-till-date
        "OPG": "OPG",  # At the opening
        "FOK": "FOK",  # Fill-or-kill
        "DTC": "DTC",  # Day till canceled
    }
    return mapping.get(tif.upper(), "DAY")


# ==================== Status Mapping ====================

def map_ibkr_order_status(ib_status: str) -> OrderStatus:
    """
    Map IBKR order status to internal OrderStatus.

    Args:
        ib_status: IBKR order status string

    Returns:
        Internal OrderStatus enum
    """
    status_map = {
        # Pending states
        "PendingSubmit": OrderStatus.PENDING,
        "PendingCancel": OrderStatus.PENDING,
        "PreSubmitted": OrderStatus.PENDING,
        "ApiPending": OrderStatus.PENDING,

        # Active states
        "Submitted": OrderStatus.OPEN,
        "ApiCancelled": OrderStatus.CANCELLED,

        # Filled states
        "Filled": OrderStatus.FILLED,
        "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,

        # Terminal states
        "Cancelled": OrderStatus.CANCELLED,
        "Inactive": OrderStatus.REJECTED,
        "Error": OrderStatus.REJECTED,
    }
    return status_map.get(ib_status, OrderStatus.PENDING)


def map_ibkr_order_type(ib_order_type: str) -> OrderType:
    """
    Map IBKR order type to internal OrderType.

    Args:
        ib_order_type: IBKR order type string

    Returns:
        Internal OrderType enum
    """
    type_map = {
        "MKT": OrderType.MARKET,
        "LMT": OrderType.LIMIT,
        "STP": OrderType.STOP,
        "STP LMT": OrderType.STOP_LIMIT,
        "TRAIL": OrderType.TRAILING_STOP,
        "TRAIL LIMIT": OrderType.TRAILING_STOP,
    }
    return type_map.get(ib_order_type, OrderType.MARKET)


def map_ibkr_order_side(ib_action: str) -> OrderSide:
    """
    Map IBKR action to internal OrderSide.

    Args:
        ib_action: IBKR action string

    Returns:
        Internal OrderSide enum
    """
    action_map = {
        "BUY": OrderSide.BUY,
        "SELL": OrderSide.SELL,
        "SSHORT": OrderSide.SELL_SHORT,
    }
    return action_map.get(ib_action, OrderSide.BUY)


# ==================== Order Creation ====================

def create_market_order(
    action: str,
    quantity: int,
    time_in_force: str = "DAY"
) -> "IBOrder":
    """
    Create an IBKR market order.

    Args:
        action: BUY or SELL
        quantity: Number of shares
        time_in_force: Order duration

    Returns:
        IBOrder object
    """
    if not IB_AVAILABLE:
        raise ImportError("ib_insync is required")

    order = MarketOrder(action, quantity)
    order.tif = convert_time_in_force(time_in_force)
    return order


def create_limit_order(
    action: str,
    quantity: int,
    limit_price: float,
    time_in_force: str = "DAY"
) -> "IBOrder":
    """
    Create an IBKR limit order.

    Args:
        action: BUY or SELL
        quantity: Number of shares
        limit_price: Limit price
        time_in_force: Order duration

    Returns:
        IBOrder object
    """
    if not IB_AVAILABLE:
        raise ImportError("ib_insync is required")

    order = LimitOrder(action, quantity, limit_price)
    order.tif = convert_time_in_force(time_in_force)
    return order


def create_stop_order(
    action: str,
    quantity: int,
    stop_price: float,
    time_in_force: str = "DAY"
) -> "IBOrder":
    """
    Create an IBKR stop order.

    Args:
        action: BUY or SELL
        quantity: Number of shares
        stop_price: Stop trigger price
        time_in_force: Order duration

    Returns:
        IBOrder object
    """
    if not IB_AVAILABLE:
        raise ImportError("ib_insync is required")

    order = StopOrder(action, quantity, stop_price)
    order.tif = convert_time_in_force(time_in_force)
    return order


def create_stop_limit_order(
    action: str,
    quantity: int,
    stop_price: float,
    limit_price: float,
    time_in_force: str = "DAY"
) -> "IBOrder":
    """
    Create an IBKR stop-limit order.

    Args:
        action: BUY or SELL
        quantity: Number of shares
        stop_price: Stop trigger price
        limit_price: Limit price after trigger
        time_in_force: Order duration

    Returns:
        IBOrder object
    """
    if not IB_AVAILABLE:
        raise ImportError("ib_insync is required")

    order = StopLimitOrder(action, quantity, limit_price, stop_price)
    order.tif = convert_time_in_force(time_in_force)
    return order


def create_trailing_stop_order(
    action: str,
    quantity: int,
    trailing_amount: Optional[float] = None,
    trailing_percent: Optional[float] = None,
    time_in_force: str = "DAY"
) -> "IBOrder":
    """
    Create an IBKR trailing stop order.

    Args:
        action: BUY or SELL
        quantity: Number of shares
        trailing_amount: Trailing amount in dollars (mutually exclusive with percent)
        trailing_percent: Trailing amount as percentage (mutually exclusive with amount)
        time_in_force: Order duration

    Returns:
        IBOrder object
    """
    if not IB_AVAILABLE:
        raise ImportError("ib_insync is required")

    if trailing_amount is None and trailing_percent is None:
        raise ValueError("Either trailing_amount or trailing_percent must be specified")

    if TrailingStopOrder is None:
        raise ImportError(
            "TrailingStopOrder not available in this ib_insync version. "
            "Upgrade to ib_insync>=0.9.87 or use IBOrder directly."
        )

    order = TrailingStopOrder(action, quantity)
    order.tif = convert_time_in_force(time_in_force)

    if trailing_percent is not None:
        order.trailingPercent = trailing_percent
    else:
        order.auxPrice = trailing_amount

    return order


# ==================== Advanced Order Types ====================

@dataclass
class BracketOrderSpec:
    """Specification for a bracket order (entry + profit target + stop loss)."""
    entry_action: str
    quantity: int
    entry_price: Optional[float]  # None for market order
    take_profit_price: float
    stop_loss_price: float
    entry_type: OrderType = OrderType.LIMIT
    time_in_force: str = "GTC"


def create_bracket_order(
    action: str,
    quantity: int,
    entry_price: Optional[float],
    take_profit_price: float,
    stop_loss_price: float,
    entry_type: OrderType = OrderType.LIMIT,
    time_in_force: str = "GTC"
) -> Tuple["IBOrder", "IBOrder", "IBOrder"]:
    """
    Create a bracket order (entry + profit target + stop loss).

    A bracket order consists of:
    1. Entry order (market or limit)
    2. Take profit order (limit) - executed when profit target reached
    3. Stop loss order (stop) - executed when stop price reached

    The profit and stop orders are OCO (one-cancels-other).

    Args:
        action: BUY or SELL for entry
        quantity: Number of shares
        entry_price: Entry limit price (None for market order)
        take_profit_price: Profit target price
        stop_loss_price: Stop loss price
        entry_type: Entry order type (MARKET or LIMIT)
        time_in_force: Order duration

    Returns:
        Tuple of (entry_order, take_profit_order, stop_loss_order)

    Example:
        # Long bracket: buy at $100, take profit at $110, stop at $95
        entry, tp, sl = create_bracket_order(
            action="BUY",
            quantity=100,
            entry_price=100.00,
            take_profit_price=110.00,
            stop_loss_price=95.00
        )
    """
    if not IB_AVAILABLE:
        raise ImportError("ib_insync is required")

    # Determine exit action (opposite of entry)
    exit_action = "SELL" if action == "BUY" else "BUY"

    # Create entry order
    if entry_type == OrderType.MARKET or entry_price is None:
        entry_order = MarketOrder(action, quantity)
    else:
        entry_order = LimitOrder(action, quantity, entry_price)

    entry_order.tif = convert_time_in_force(time_in_force)
    entry_order.transmit = False  # Don't transmit until all orders are ready

    # Create take profit order (limit)
    take_profit_order = LimitOrder(exit_action, quantity, take_profit_price)
    take_profit_order.tif = convert_time_in_force(time_in_force)
    take_profit_order.parentId = 0  # Will be set by IB
    take_profit_order.transmit = False

    # Create stop loss order
    stop_loss_order = StopOrder(exit_action, quantity, stop_loss_price)
    stop_loss_order.tif = convert_time_in_force(time_in_force)
    stop_loss_order.parentId = 0  # Will be set by IB
    stop_loss_order.transmit = True  # Transmit the bracket

    # Mark as OCO group
    oca_group = f"bracket_{id(entry_order)}"
    take_profit_order.ocaGroup = oca_group
    take_profit_order.ocaType = 1  # Cancel other orders in group when filled
    stop_loss_order.ocaGroup = oca_group
    stop_loss_order.ocaType = 1

    logger.info(
        f"Created bracket order: {action} {quantity} @ "
        f"{entry_price or 'MKT'}, TP={take_profit_price}, SL={stop_loss_price}"
    )

    return entry_order, take_profit_order, stop_loss_order


@dataclass
class OCOOrderSpec:
    """Specification for one-cancels-other orders."""
    orders: List[Dict[str, Any]]
    oca_group: Optional[str] = None


def create_oco_order(
    orders: List[Dict[str, Any]],
    oca_group: Optional[str] = None
) -> List["IBOrder"]:
    """
    Create OCO (one-cancels-other) orders.

    When one order in the group is filled, the others are automatically cancelled.

    Args:
        orders: List of order specifications, each containing:
            - action: BUY or SELL
            - quantity: Number of shares
            - order_type: OrderType enum
            - price: Limit price (for limit orders)
            - stop_price: Stop price (for stop orders)
        oca_group: Optional OCO group name (auto-generated if not provided)

    Returns:
        List of IBOrder objects linked as OCO

    Example:
        # OCO order: sell at $110 profit OR $95 stop loss
        orders = create_oco_order([
            {"action": "SELL", "quantity": 100, "order_type": OrderType.LIMIT, "price": 110},
            {"action": "SELL", "quantity": 100, "order_type": OrderType.STOP, "stop_price": 95}
        ])
    """
    if not IB_AVAILABLE:
        raise ImportError("ib_insync is required")

    if len(orders) < 2:
        raise ValueError("OCO requires at least 2 orders")

    oca_group = oca_group or f"oco_{id(orders)}"
    ib_orders = []

    for i, spec in enumerate(orders):
        action = spec.get("action", "BUY")
        quantity = spec.get("quantity", 1)
        order_type = spec.get("order_type", OrderType.MARKET)
        price = spec.get("price")
        stop_price = spec.get("stop_price")
        tif = spec.get("time_in_force", "GTC")

        # Create order based on type
        if order_type == OrderType.MARKET:
            ib_order = MarketOrder(action, quantity)
        elif order_type == OrderType.LIMIT:
            if price is None:
                raise ValueError("Limit order requires price")
            ib_order = LimitOrder(action, quantity, price)
        elif order_type == OrderType.STOP:
            if stop_price is None:
                raise ValueError("Stop order requires stop_price")
            ib_order = StopOrder(action, quantity, stop_price)
        elif order_type == OrderType.STOP_LIMIT:
            if price is None or stop_price is None:
                raise ValueError("Stop-limit order requires price and stop_price")
            ib_order = StopLimitOrder(action, quantity, price, stop_price)
        else:
            raise ValueError(f"Unsupported order type for OCO: {order_type}")

        # Set OCO properties
        ib_order.tif = convert_time_in_force(tif)
        ib_order.ocaGroup = oca_group
        ib_order.ocaType = 1  # Cancel other orders when filled

        # Only transmit the last order
        ib_order.transmit = (i == len(orders) - 1)

        ib_orders.append(ib_order)

    logger.info(f"Created OCO group '{oca_group}' with {len(ib_orders)} orders")

    return ib_orders


def create_adaptive_order(
    action: str,
    quantity: int,
    order_type: OrderType,
    price: Optional[float] = None,
    priority: str = "Normal",
    time_in_force: str = "DAY"
) -> "IBOrder":
    """
    Create an IBKR adaptive order.

    Adaptive orders seek to achieve better execution by using IB's adaptive
    algorithm which adjusts the order based on real-time market conditions.

    Args:
        action: BUY or SELL
        quantity: Number of shares
        order_type: MARKET or LIMIT
        price: Limit price (required for LIMIT orders)
        priority: "Patient", "Normal", or "Urgent"
        time_in_force: Order duration

    Returns:
        IBOrder object with adaptive algo
    """
    if not IB_AVAILABLE:
        raise ImportError("ib_insync is required")

    if order_type == OrderType.MARKET:
        order = MarketOrder(action, quantity)
    elif order_type == OrderType.LIMIT:
        if price is None:
            raise ValueError("Limit order requires price")
        order = LimitOrder(action, quantity, price)
    else:
        raise ValueError(f"Adaptive orders only support MARKET and LIMIT types")

    order.tif = convert_time_in_force(time_in_force)

    # Set adaptive algo parameters
    order.algoStrategy = "Adaptive"
    order.algoParams = []

    # Map priority
    priority_map = {
        "Patient": "Patient",
        "Normal": "Normal",
        "Urgent": "Urgent",
    }
    priority_value = priority_map.get(priority, "Normal")

    from ib_insync import TagValue
    order.algoParams.append(TagValue("adaptivePriority", priority_value))

    logger.info(f"Created adaptive {order_type.value} order: {action} {quantity}, priority={priority}")

    return order


# ==================== Order Validation ====================

def validate_bracket_order(
    action: str,
    entry_price: Optional[float],
    take_profit_price: float,
    stop_loss_price: float
) -> Tuple[bool, str]:
    """
    Validate bracket order prices are sensible.

    Args:
        action: BUY or SELL
        entry_price: Entry price (None for market)
        take_profit_price: Target price
        stop_loss_price: Stop loss price

    Returns:
        Tuple of (is_valid, error_message)
    """
    if action == "BUY":
        # For long positions:
        # - Take profit should be above entry
        # - Stop loss should be below entry
        if entry_price is not None:
            if take_profit_price <= entry_price:
                return False, "Take profit must be above entry price for long positions"
            if stop_loss_price >= entry_price:
                return False, "Stop loss must be below entry price for long positions"
        if stop_loss_price >= take_profit_price:
            return False, "Stop loss must be below take profit"
    else:
        # For short positions:
        # - Take profit should be below entry
        # - Stop loss should be above entry
        if entry_price is not None:
            if take_profit_price >= entry_price:
                return False, "Take profit must be below entry price for short positions"
            if stop_loss_price <= entry_price:
                return False, "Stop loss must be above entry price for short positions"
        if stop_loss_price <= take_profit_price:
            return False, "Stop loss must be above take profit for short positions"

    return True, ""

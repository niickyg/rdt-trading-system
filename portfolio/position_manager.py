"""
Position Manager
Tracks and manages all open positions
"""

from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from agents.events import EventType, Event, get_event_bus, subscribe


class PositionStatus(Enum):
    """Position lifecycle states"""
    PENDING = "pending"  # Order placed, not filled
    OPEN = "open"  # Position active
    PARTIAL = "partial"  # Partially closed
    CLOSED = "closed"  # Fully closed


@dataclass
class Position:
    """Represents an open trading position"""
    symbol: str
    direction: str  # 'long' or 'short'
    shares: int
    entry_price: float
    stop_price: float
    target_price: float

    # Tracking
    entry_time: datetime = field(default_factory=datetime.now)
    status: PositionStatus = PositionStatus.OPEN

    # Current state
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    # Exit tracking
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    realized_pnl: float = 0.0
    exit_reason: str = ""

    # Additional data
    rrs: float = 0.0
    atr: float = 0.0
    adds_count: int = 0
    tags: List[str] = field(default_factory=list)

    @property
    def position_value(self) -> float:
        return self.shares * self.entry_price

    @property
    def current_value(self) -> float:
        return self.shares * self.current_price

    @property
    def risk_amount(self) -> float:
        return abs(self.entry_price - self.stop_price) * self.shares

    @property
    def reward_amount(self) -> float:
        return abs(self.target_price - self.entry_price) * self.shares

    @property
    def is_profitable(self) -> bool:
        return self.unrealized_pnl > 0

    def update_price(self, price: float):
        """Update current price and P&L"""
        self.current_price = price

        if self.direction == "long":
            self.unrealized_pnl = (price - self.entry_price) * self.shares
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.shares

        self.unrealized_pnl_pct = (self.unrealized_pnl / self.position_value) * 100

    def close(self, exit_price: float, reason: str = ""):
        """Close the position"""
        self.exit_price = exit_price
        self.exit_time = datetime.now()
        self.exit_reason = reason
        self.status = PositionStatus.CLOSED

        if self.direction == "long":
            self.realized_pnl = (exit_price - self.entry_price) * self.shares
        else:
            self.realized_pnl = (self.entry_price - exit_price) * self.shares

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "shares": self.shares,
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "status": self.status.value,
            "entry_time": self.entry_time.isoformat(),
            "rrs": self.rrs
        }


class PositionManager:
    """
    Manages all trading positions

    Responsibilities:
    - Track open positions
    - Update position prices
    - Monitor stop/target levels
    - Calculate portfolio P&L
    - Handle position events
    """

    def __init__(self, event_bus=None):
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.event_bus = event_bus or get_event_bus()

        # Portfolio metrics
        self.total_realized_pnl = 0.0
        self.total_unrealized_pnl = 0.0
        self.winning_trades = 0
        self.losing_trades = 0

        # Subscribe to events
        self._setup_subscriptions()

        logger.info("PositionManager initialized")

    def _setup_subscriptions(self):
        """Subscribe to position-related events"""
        subscribe(EventType.POSITION_OPENED, self._on_position_opened)
        subscribe(EventType.ORDER_FILLED, self._on_order_filled)

    def _on_position_opened(self, event: Event):
        """Handle position opened event"""
        data = event.data
        self.open_position(
            symbol=data["symbol"],
            direction=data["direction"],
            shares=data["shares"],
            entry_price=data["entry_price"],
            stop_price=data["stop_price"],
            target_price=data["target_price"],
            rrs=data.get("rrs", 0)
        )

    def _on_order_filled(self, event: Event):
        """Handle order fill event"""
        # Update position if this is an exit order
        pass

    def open_position(
        self,
        symbol: str,
        direction: str,
        shares: int,
        entry_price: float,
        stop_price: float,
        target_price: float,
        rrs: float = 0.0,
        atr: float = 0.0
    ) -> Position:
        """Open a new position"""
        position = Position(
            symbol=symbol,
            direction=direction,
            shares=shares,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            current_price=entry_price,
            rrs=rrs,
            atr=atr
        )

        self.positions[symbol] = position
        logger.info(f"Position opened: {symbol} {direction} {shares} @ ${entry_price:.2f}")

        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = ""
    ) -> Optional[Position]:
        """Close a position"""
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return None

        position = self.positions[symbol]
        position.close(exit_price, reason)

        # Update stats
        self.total_realized_pnl += position.realized_pnl
        if position.realized_pnl >= 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[symbol]

        logger.info(
            f"Position closed: {symbol} @ ${exit_price:.2f} "
            f"P&L: ${position.realized_pnl:+,.2f} ({reason})"
        )

        # Emit event
        self.event_bus.emit(EventType.POSITION_CLOSED, {
            "symbol": symbol,
            "direction": position.direction,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "shares": position.shares,
            "pnl": position.realized_pnl,
            "reason": reason
        })

        return position

    def update_prices(self, quotes: Dict[str, float]):
        """Update all position prices"""
        for symbol, price in quotes.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)

        self._calculate_totals()
        self._check_stops_and_targets()

    def update_position_price(self, symbol: str, price: float):
        """Update a single position's price"""
        if symbol in self.positions:
            self.positions[symbol].update_price(price)
            self._check_position_levels(symbol)

    def _calculate_totals(self):
        """Calculate total unrealized P&L"""
        self.total_unrealized_pnl = sum(
            p.unrealized_pnl for p in self.positions.values()
        )

    def _check_stops_and_targets(self):
        """Check all positions for stop/target hits"""
        for symbol in list(self.positions.keys()):
            self._check_position_levels(symbol)

    def _check_position_levels(self, symbol: str):
        """Check if a position hit stop or target"""
        position = self.positions.get(symbol)
        if not position:
            return

        price = position.current_price

        if position.direction == "long":
            if price <= position.stop_price:
                self._trigger_stop(symbol, price)
            elif price >= position.target_price:
                self._trigger_target(symbol, price)
        else:  # short
            if price >= position.stop_price:
                self._trigger_stop(symbol, price)
            elif price <= position.target_price:
                self._trigger_target(symbol, price)

    def _trigger_stop(self, symbol: str, price: float):
        """Handle stop loss trigger"""
        logger.warning(f"STOP TRIGGERED: {symbol} @ ${price:.2f}")

        self.event_bus.emit(EventType.STOP_HIT, {
            "symbol": symbol,
            "price": price
        })

        self.close_position(symbol, price, "stop_loss")

    def _trigger_target(self, symbol: str, price: float):
        """Handle take profit trigger"""
        logger.info(f"TARGET REACHED: {symbol} @ ${price:.2f}")

        self.event_bus.emit(EventType.TARGET_HIT, {
            "symbol": symbol,
            "price": price
        })

        self.close_position(symbol, price, "take_profit")

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get a specific position"""
        return self.positions.get(symbol)

    def get_all_positions(self) -> List[Position]:
        """Get all open positions"""
        return list(self.positions.values())

    def get_position_count(self) -> int:
        """Get number of open positions"""
        return len(self.positions)

    def get_total_exposure(self) -> float:
        """Get total portfolio exposure"""
        return sum(p.current_value for p in self.positions.values())

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        win_rate = 0
        total_trades = self.winning_trades + self.losing_trades
        if total_trades > 0:
            win_rate = (self.winning_trades / total_trades) * 100

        return {
            "open_positions": len(self.positions),
            "total_exposure": self.get_total_exposure(),
            "unrealized_pnl": self.total_unrealized_pnl,
            "realized_pnl": self.total_realized_pnl,
            "total_pnl": self.total_realized_pnl + self.total_unrealized_pnl,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "positions": [p.to_dict() for p in self.positions.values()]
        }

    def modify_stop(self, symbol: str, new_stop: float):
        """Modify stop price for a position"""
        if symbol in self.positions:
            old_stop = self.positions[symbol].stop_price
            self.positions[symbol].stop_price = new_stop
            logger.info(f"Stop modified: {symbol} ${old_stop:.2f} -> ${new_stop:.2f}")

    def modify_target(self, symbol: str, new_target: float):
        """Modify target price for a position"""
        if symbol in self.positions:
            old_target = self.positions[symbol].target_price
            self.positions[symbol].target_price = new_target
            logger.info(f"Target modified: {symbol} ${old_target:.2f} -> ${new_target:.2f}")

    def trail_stop(self, symbol: str, trail_amount: float):
        """Trail stop by a fixed amount"""
        position = self.positions.get(symbol)
        if not position:
            return

        if position.direction == "long":
            new_stop = position.current_price - trail_amount
            if new_stop > position.stop_price:
                self.modify_stop(symbol, new_stop)
        else:
            new_stop = position.current_price + trail_amount
            if new_stop < position.stop_price:
                self.modify_stop(symbol, new_stop)

    def scale_out(self, symbol: str, shares_to_sell: int, price: float):
        """Scale out of a position"""
        position = self.positions.get(symbol)
        if not position:
            return

        if shares_to_sell >= position.shares:
            self.close_position(symbol, price, "scale_out_full")
        else:
            # Partial close
            pnl_per_share = price - position.entry_price
            if position.direction == "short":
                pnl_per_share = position.entry_price - price

            realized = pnl_per_share * shares_to_sell
            position.shares -= shares_to_sell
            self.total_realized_pnl += realized

            logger.info(
                f"Scaled out: {symbol} sold {shares_to_sell} @ ${price:.2f} "
                f"(P&L: ${realized:+,.2f})"
            )

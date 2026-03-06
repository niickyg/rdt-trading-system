"""
Position Tracker for the RDT Trading System.

Provides real-time position management with:
- Live P&L calculation with current prices
- Stop loss and target tracking
- Position state transitions (pending -> open -> closed)
- Trade journaling (notes, screenshots, lessons learned)
"""

import asyncio
import threading
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from loguru import logger

try:
    from shared.data_provider import DataProvider
    DATA_PROVIDER_AVAILABLE = True
except ImportError:
    DATA_PROVIDER_AVAILABLE = False
    logger.warning("DataProvider not available for real-time prices")

try:
    from data.database import get_trades_repository
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logger.warning("Database not available for position persistence")

# Prometheus metrics support (optional)
try:
    from monitoring.metrics import (
        record_trade,
        update_portfolio_metrics,
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logger.debug("Prometheus metrics not available")


class PositionState(str, Enum):
    """Position lifecycle states"""
    PENDING = "pending"      # Order placed but not filled
    OPEN = "open"           # Position is active
    CLOSED = "closed"       # Position has been closed
    CANCELLED = "cancelled" # Order was cancelled before fill


class PositionDirection(str, Enum):
    """Position direction"""
    LONG = "long"
    SHORT = "short"


@dataclass
class TradeNote:
    """A journal entry for a trade"""
    id: int
    timestamp: datetime
    note: str
    note_type: str = "general"  # general, entry, exit, lesson, screenshot

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "note": self.note,
            "note_type": self.note_type
        }


@dataclass
class PositionEvent:
    """An event in the position lifecycle"""
    timestamp: datetime
    event_type: str  # opened, stop_updated, target_updated, note_added, closed
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "details": self.details
        }


@dataclass
class Position:
    """A trading position with full tracking"""
    symbol: str
    direction: PositionDirection
    entry_price: float
    shares: int
    stop_price: float
    target_price: float
    state: PositionState = PositionState.OPEN
    entry_time: datetime = field(default_factory=datetime.utcnow)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    current_price: Optional[float] = None
    rrs_at_entry: Optional[float] = None
    notes: List[TradeNote] = field(default_factory=list)
    events: List[PositionEvent] = field(default_factory=list)
    _note_counter: int = 0

    def __post_init__(self):
        """Initialize with an opening event"""
        if not self.events:
            self.events.append(PositionEvent(
                timestamp=self.entry_time,
                event_type="opened",
                details={
                    "entry_price": self.entry_price,
                    "shares": self.shares,
                    "direction": self.direction.value if isinstance(self.direction, PositionDirection) else self.direction,
                    "stop_price": self.stop_price,
                    "target_price": self.target_price,
                    "rrs_at_entry": self.rrs_at_entry
                }
            ))
        # Set current price to entry price if not set
        if self.current_price is None:
            self.current_price = self.entry_price

    @property
    def cost_basis(self) -> float:
        """Total cost of the position"""
        return self.entry_price * abs(self.shares)

    @property
    def current_value(self) -> float:
        """Current market value of the position"""
        price = self.current_price or self.entry_price
        return price * self.shares

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss"""
        if self.current_price is None:
            return 0.0

        if self.direction == PositionDirection.LONG or self.direction == "long":
            return (self.current_price - self.entry_price) * abs(self.shares)
        else:
            return (self.entry_price - self.current_price) * abs(self.shares)

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage"""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100

    @property
    def realized_pnl(self) -> Optional[float]:
        """Realized P&L (only for closed positions)"""
        if self.state != PositionState.CLOSED or self.exit_price is None:
            return None

        if self.direction == PositionDirection.LONG or self.direction == "long":
            return (self.exit_price - self.entry_price) * abs(self.shares)
        else:
            return (self.entry_price - self.exit_price) * abs(self.shares)

    @property
    def realized_pnl_pct(self) -> Optional[float]:
        """Realized P&L as percentage"""
        pnl = self.realized_pnl
        if pnl is None or self.cost_basis == 0:
            return None
        return (pnl / self.cost_basis) * 100

    @property
    def risk_to_stop(self) -> float:
        """Risk amount to stop loss"""
        if self.direction == PositionDirection.LONG or self.direction == "long":
            return (self.entry_price - self.stop_price) * abs(self.shares)
        else:
            return (self.stop_price - self.entry_price) * abs(self.shares)

    @property
    def reward_to_target(self) -> float:
        """Potential reward to target"""
        if self.direction == PositionDirection.LONG or self.direction == "long":
            return (self.target_price - self.entry_price) * abs(self.shares)
        else:
            return (self.entry_price - self.target_price) * abs(self.shares)

    @property
    def risk_reward_ratio(self) -> float:
        """Risk/reward ratio"""
        risk = abs(self.risk_to_stop)
        if risk == 0:
            return 0.0
        return abs(self.reward_to_target) / risk

    @property
    def distance_to_stop_pct(self) -> float:
        """Distance to stop as percentage of current price"""
        if self.current_price is None or self.current_price == 0:
            return 0.0
        if self.direction == PositionDirection.LONG or self.direction == "long":
            return ((self.current_price - self.stop_price) / self.current_price) * 100
        else:
            return ((self.stop_price - self.current_price) / self.current_price) * 100

    @property
    def distance_to_target_pct(self) -> float:
        """Distance to target as percentage of current price"""
        if self.current_price is None or self.current_price == 0:
            return 0.0
        if self.direction == PositionDirection.LONG or self.direction == "long":
            return ((self.target_price - self.current_price) / self.current_price) * 100
        else:
            return ((self.current_price - self.target_price) / self.current_price) * 100

    @property
    def holding_duration(self) -> float:
        """Holding duration in days"""
        end_time = self.exit_time or datetime.utcnow()
        delta = end_time - self.entry_time
        return delta.total_seconds() / 86400

    def is_stop_hit(self) -> bool:
        """Check if stop price has been hit"""
        if self.current_price is None:
            return False
        if self.direction == PositionDirection.LONG or self.direction == "long":
            return self.current_price <= self.stop_price
        else:
            return self.current_price >= self.stop_price

    def is_target_hit(self) -> bool:
        """Check if target price has been hit"""
        if self.current_price is None:
            return False
        if self.direction == PositionDirection.LONG or self.direction == "long":
            return self.current_price >= self.target_price
        else:
            return self.current_price <= self.target_price

    def add_note(self, note: str, note_type: str = "general") -> TradeNote:
        """Add a journal note to this position"""
        self._note_counter += 1
        trade_note = TradeNote(
            id=self._note_counter,
            timestamp=datetime.utcnow(),
            note=note,
            note_type=note_type
        )
        self.notes.append(trade_note)
        self.events.append(PositionEvent(
            timestamp=datetime.utcnow(),
            event_type="note_added",
            details={"note_id": trade_note.id, "note_type": note_type}
        ))
        return trade_note

    def update_stop(self, new_stop: float) -> None:
        """Update stop loss price"""
        if new_stop is not None and new_stop <= 0:
            raise ValueError(f"Stop price must be positive, got {new_stop}")
        old_stop = self.stop_price
        self.stop_price = new_stop
        self.events.append(PositionEvent(
            timestamp=datetime.utcnow(),
            event_type="stop_updated",
            details={"old_stop": old_stop, "new_stop": new_stop}
        ))

    def update_target(self, new_target: float) -> None:
        """Update target price"""
        if new_target is not None and new_target <= 0:
            raise ValueError(f"Target price must be positive, got {new_target}")
        old_target = self.target_price
        self.target_price = new_target
        self.events.append(PositionEvent(
            timestamp=datetime.utcnow(),
            event_type="target_updated",
            details={"old_target": old_target, "new_target": new_target}
        ))

    def close(self, exit_price: float, reason: str) -> None:
        """Close the position"""
        self.exit_price = exit_price
        self.exit_time = datetime.utcnow()
        self.exit_reason = reason
        self.state = PositionState.CLOSED
        self.current_price = exit_price

        self.events.append(PositionEvent(
            timestamp=self.exit_time,
            event_type="closed",
            details={
                "exit_price": exit_price,
                "reason": reason,
                "pnl": self.realized_pnl,
                "pnl_pct": self.realized_pnl_pct,
                "holding_days": self.holding_duration
            }
        ))

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary"""
        direction_value = self.direction.value if isinstance(self.direction, PositionDirection) else self.direction
        state_value = self.state.value if isinstance(self.state, PositionState) else self.state

        return {
            "symbol": self.symbol,
            "direction": direction_value,
            "entry_price": self.entry_price,
            "shares": self.shares,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "state": state_value,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "current_price": self.current_price,
            "rrs_at_entry": self.rrs_at_entry,
            "cost_basis": self.cost_basis,
            "current_value": self.current_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 2),
            "realized_pnl": self.realized_pnl,
            "realized_pnl_pct": round(self.realized_pnl_pct, 2) if self.realized_pnl_pct is not None else None,
            "risk_to_stop": self.risk_to_stop,
            "reward_to_target": self.reward_to_target,
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "distance_to_stop_pct": round(self.distance_to_stop_pct, 2),
            "distance_to_target_pct": round(self.distance_to_target_pct, 2),
            "holding_days": round(self.holding_duration, 2),
            "is_stop_hit": self.is_stop_hit(),
            "is_target_hit": self.is_target_hit(),
            "notes_count": len(self.notes),
            "notes": [n.to_dict() for n in self.notes],
            "events": [e.to_dict() for e in self.events]
        }


class PositionTracker:
    """
    Manages live trading positions with real-time updates.

    Features:
    - Real-time P&L calculation with current prices
    - Stop loss and target tracking
    - Position state transitions
    - Trade journaling
    - Database persistence
    """

    def __init__(self, auto_update_prices: bool = True, update_interval: int = 30):
        """
        Initialize position tracker.

        Args:
            auto_update_prices: Whether to automatically update prices
            update_interval: Seconds between price updates
        """
        self._positions: Dict[str, Position] = {}
        self._closed_positions: List[Position] = []
        self._lock = threading.Lock()
        self.auto_update_prices = auto_update_prices
        self.update_interval = update_interval
        self._update_task: Optional[asyncio.Task] = None
        self._running = False

        # Initialize data provider for real-time prices
        if DATA_PROVIDER_AVAILABLE:
            self._data_provider = DataProvider(cache_ttl_seconds=15)
        else:
            self._data_provider = None

        # Load positions from database if available
        if DATABASE_AVAILABLE:
            self._load_positions_from_db()

        logger.info("PositionTracker initialized")

    def _load_positions_from_db(self) -> None:
        """Load open positions from database"""
        try:
            repo = get_trades_repository()
            db_positions = repo.get_open_positions()

            for pos_data in db_positions:
                symbol = pos_data.get('symbol')
                if symbol and symbol not in self._positions:
                    direction = pos_data.get('direction', 'long')
                    if direction == 'long':
                        dir_enum = PositionDirection.LONG
                    else:
                        dir_enum = PositionDirection.SHORT

                    position = Position(
                        symbol=symbol,
                        direction=dir_enum,
                        entry_price=float(pos_data.get('entry_price', 0)),
                        shares=int(pos_data.get('shares', 0)),
                        stop_price=float(pos_data.get('stop_loss', 0) or 0),
                        target_price=float(pos_data.get('take_profit', 0) or 0),
                        current_price=float(pos_data['current_price']) if pos_data.get('current_price') is not None else None,
                        rrs_at_entry=float(pos_data['rrs_at_entry']) if pos_data.get('rrs_at_entry') is not None else None,
                        entry_time=datetime.fromisoformat(pos_data['entry_time']).replace(tzinfo=None) if pos_data.get('entry_time') else datetime.utcnow()
                    )
                    self._positions[symbol] = position

            logger.info(f"Loaded {len(self._positions)} positions from database")
        except Exception as e:
            logger.error(f"Error loading positions from database: {e}")

    def _save_position_to_db(self, position: Position) -> None:
        """Save or update position in database"""
        if not DATABASE_AVAILABLE:
            return

        try:
            repo = get_trades_repository()
            direction_value = position.direction.value if isinstance(position.direction, PositionDirection) else position.direction

            repo.save_position({
                'symbol': position.symbol,
                'direction': direction_value,
                'entry_price': position.entry_price,
                'shares': position.shares,
                'stop_loss': position.stop_price,
                'take_profit': position.target_price,
                'current_price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'rrs_at_entry': position.rrs_at_entry,
                'entry_time': position.entry_time
            })
        except Exception as e:
            logger.error(f"Error saving position to database: {e}")

    def _close_position_in_db(self, position: Position) -> None:
        """Close position in database and create trade record"""
        if not DATABASE_AVAILABLE:
            return

        try:
            repo = get_trades_repository()

            # Close the position record
            repo.close_position(position.symbol)

            # Create a trade record for the closed position
            direction_value = position.direction.value if isinstance(position.direction, PositionDirection) else position.direction

            repo.save_trade({
                'symbol': position.symbol,
                'direction': direction_value,
                'entry_price': position.entry_price,
                'shares': position.shares,
                'entry_time': position.entry_time,
                'stop_loss': position.stop_price,
                'take_profit': position.target_price,
                'rrs_at_entry': position.rrs_at_entry
            })

            # Get the trade ID and close it
            trades = repo.get_trades(symbol=position.symbol, status='open', limit=1)
            if trades:
                repo.close_trade(
                    trade_id=trades[0]['id'],
                    exit_price=position.exit_price,
                    exit_reason=position.exit_reason or 'manual'
                )

            logger.info(f"Closed position {position.symbol} in database")
        except Exception as e:
            logger.error(f"Error closing position in database: {e}")

    def open_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        shares: int,
        stop_price: float,
        target_price: float,
        rrs_at_entry: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Open a new position.

        Args:
            symbol: Stock ticker symbol
            direction: 'long' or 'short'
            entry_price: Entry price
            shares: Number of shares
            stop_price: Stop loss price
            target_price: Take profit price
            rrs_at_entry: RRS value at entry (optional)

        Returns:
            Position data dictionary
        """
        symbol = symbol.upper()

        # Validate direction
        if direction.lower() == 'long':
            dir_enum = PositionDirection.LONG
        elif direction.lower() == 'short':
            dir_enum = PositionDirection.SHORT
        else:
            return {"error": f"Invalid direction: {direction}"}

        # Validate prices are positive
        if entry_price <= 0:
            return {"error": "Entry price must be positive"}
        if stop_price is not None and stop_price <= 0:
            return {"error": "Stop price must be positive"}
        if target_price is not None and target_price <= 0:
            return {"error": "Target price must be positive"}

        # Validate stop/target for direction
        if direction.lower() == 'long':
            if stop_price >= entry_price:
                return {"error": "Stop price must be below entry for long positions"}
            if target_price <= entry_price:
                return {"error": "Target price must be above entry for long positions"}
        else:
            if stop_price <= entry_price:
                return {"error": "Stop price must be above entry for short positions"}
            if target_price >= entry_price:
                return {"error": "Target price must be below entry for short positions"}

        with self._lock:
            # Check if position already exists
            if symbol in self._positions:
                logger.warning(f"Position already exists for {symbol}")
                return {"error": f"Position already exists for {symbol}"}

            # Create position
            position = Position(
                symbol=symbol,
                direction=dir_enum,
                entry_price=entry_price,
                shares=shares,
                stop_price=stop_price,
                target_price=target_price,
                current_price=entry_price,
                rrs_at_entry=rrs_at_entry
            )

            self._positions[symbol] = position

        # Persist to database (outside lock to avoid holding it during I/O)
        self._save_position_to_db(position)

        logger.info(f"Opened {direction} position: {shares} {symbol} @ ${entry_price:.2f}")

        return position.to_dict()

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "manual"
    ) -> Dict[str, Any]:
        """
        Close an open position.

        Args:
            symbol: Stock ticker symbol
            exit_price: Exit price
            reason: Reason for closing (stop_loss, take_profit, manual, etc.)

        Returns:
            Closed position data dictionary
        """
        symbol = symbol.upper()

        with self._lock:
            if symbol not in self._positions:
                return {"error": f"No open position for {symbol}"}

            position = self._positions[symbol]
            position.close(exit_price, reason)

            # Move to closed positions (cap at 500 to prevent memory leak)
            self._closed_positions.append(position)
            if len(self._closed_positions) > 500:
                self._closed_positions = self._closed_positions[-500:]
            del self._positions[symbol]

        # Update database (outside lock)
        self._close_position_in_db(position)

        pnl = position.realized_pnl or 0.0
        logger.info(f"Closed {symbol} position @ ${exit_price:.2f} - P&L: ${pnl:.2f} ({reason})")

        return position.to_dict()

    def update_stop(self, symbol: str, new_stop_price: float) -> Dict[str, Any]:
        """
        Update stop loss price for a position.

        Args:
            symbol: Stock ticker symbol
            new_stop_price: New stop loss price

        Returns:
            Updated position data dictionary
        """
        symbol = symbol.upper()

        with self._lock:
            if symbol not in self._positions:
                return {"error": f"No open position for {symbol}"}

            position = self._positions[symbol]
            old_stop = position.stop_price
            try:
                position.update_stop(new_stop_price)
            except ValueError as e:
                return {"error": str(e)}

        # Persist to database (outside lock)
        self._save_position_to_db(position)

        logger.info(f"Updated {symbol} stop: ${old_stop:.2f} -> ${new_stop_price:.2f}")

        return position.to_dict()

    def update_target(self, symbol: str, new_target_price: float) -> Dict[str, Any]:
        """
        Update target price for a position.

        Args:
            symbol: Stock ticker symbol
            new_target_price: New target price

        Returns:
            Updated position data dictionary
        """
        symbol = symbol.upper()

        with self._lock:
            if symbol not in self._positions:
                return {"error": f"No open position for {symbol}"}

            position = self._positions[symbol]
            old_target = position.target_price
            try:
                position.update_target(new_target_price)
            except ValueError as e:
                return {"error": str(e)}

        # Persist to database (outside lock)
        self._save_position_to_db(position)

        logger.info(f"Updated {symbol} target: ${old_target:.2f} -> ${new_target_price:.2f}")

        return position.to_dict()

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current position with live P&L.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Position data dictionary or None if not found
        """
        symbol = symbol.upper()

        if symbol not in self._positions:
            return None

        return self._positions[symbol].to_dict()

    def get_all_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open positions with live data.

        Returns:
            List of position data dictionaries
        """
        with self._lock:
            return [pos.to_dict() for pos in self._positions.values()]

    def check_stops_and_targets(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Check if any positions have hit stops or targets.

        Returns:
            Dictionary with 'stops_hit' and 'targets_hit' lists
        """
        stops_hit = []
        targets_hit = []

        with self._lock:
            positions_snapshot = list(self._positions.items())

        for symbol, position in positions_snapshot:
            if position.is_stop_hit():
                stops_hit.append({
                    "symbol": symbol,
                    "current_price": position.current_price,
                    "stop_price": position.stop_price,
                    "unrealized_pnl": position.unrealized_pnl
                })
                logger.warning(f"STOP HIT: {symbol} @ ${position.current_price:.2f} (stop: ${position.stop_price:.2f})")

            if position.is_target_hit():
                targets_hit.append({
                    "symbol": symbol,
                    "current_price": position.current_price,
                    "target_price": position.target_price,
                    "unrealized_pnl": position.unrealized_pnl
                })
                logger.info(f"TARGET HIT: {symbol} @ ${position.current_price:.2f} (target: ${position.target_price:.2f})")

        return {
            "stops_hit": stops_hit,
            "targets_hit": targets_hit
        }

    def add_trade_note(self, symbol: str, note: str, note_type: str = "general") -> Dict[str, Any]:
        """
        Add a journal entry to a position.

        Args:
            symbol: Stock ticker symbol
            note: Note text
            note_type: Type of note (general, entry, exit, lesson, screenshot)

        Returns:
            Updated position data dictionary
        """
        symbol = symbol.upper()

        if symbol not in self._positions:
            # Check closed positions
            for pos in self._closed_positions:
                if pos.symbol == symbol:
                    trade_note = pos.add_note(note, note_type)
                    logger.info(f"Added note to closed position {symbol}: {note[:50]}...")
                    return pos.to_dict()
            return {"error": f"No position found for {symbol}"}

        position = self._positions[symbol]
        trade_note = position.add_note(note, note_type)

        logger.info(f"Added note to {symbol}: {note[:50]}...")

        return position.to_dict()

    def get_trade_history(self, symbol: str) -> Dict[str, Any]:
        """
        Get complete history for a symbol including events and notes.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with position data and history
        """
        symbol = symbol.upper()

        # Check open positions
        if symbol in self._positions:
            position = self._positions[symbol]
            return {
                "symbol": symbol,
                "status": "open",
                "position": position.to_dict(),
                "events": [e.to_dict() for e in position.events],
                "notes": [n.to_dict() for n in position.notes]
            }

        # Check closed positions
        for pos in self._closed_positions:
            if pos.symbol == symbol:
                return {
                    "symbol": symbol,
                    "status": "closed",
                    "position": pos.to_dict(),
                    "events": [e.to_dict() for e in pos.events],
                    "notes": [n.to_dict() for n in pos.notes]
                }

        # Check database for historical trades
        if DATABASE_AVAILABLE:
            try:
                repo = get_trades_repository()
                trades = repo.get_trades(symbol=symbol, limit=50)
                if trades:
                    return {
                        "symbol": symbol,
                        "status": "historical",
                        "trades": trades
                    }
            except Exception as e:
                logger.error(f"Error fetching trade history from database: {e}")

        return {"error": f"No history found for {symbol}"}

    async def update_prices(self) -> None:
        """Update current prices for all open positions via IBKR data provider.

        Retries up to 3 times if the data provider call fails.
        """
        if not self._positions:
            return

        if not self._data_provider:
            logger.debug("No data provider configured — skipping price update")
            return

        symbols = list(self._positions.keys())
        import asyncio as _asyncio

        for attempt in range(3):
            try:
                quotes = await self._data_provider.get_quotes(symbols)
                for symbol, quote in quotes.items():
                    if symbol in self._positions and quote:
                        price = quote.get('price')
                        if price and price > 0:
                            self._positions[symbol].current_price = price
                break  # success
            except Exception as e:
                logger.warning(f"Price update attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    await _asyncio.sleep(0.5 * (attempt + 1))

        # Update database with new prices
        for position in self._positions.values():
            self._save_position_to_db(position)

    def update_prices_sync(self) -> None:
        """Synchronous wrapper around async update_prices() for Flask route callers."""
        if not self._positions:
            return
        if not self._data_provider:
            return
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in an async context (e.g. agent thread) — schedule as task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(lambda: asyncio.run(self.update_prices())).result(timeout=15)
            else:
                loop.run_until_complete(self.update_prices())
        except Exception as e:
            logger.warning(f"Sync price update failed: {e}")

    async def _price_update_loop(self) -> None:
        """Background task for automatic price updates"""
        while self._running:
            try:
                await self.update_prices()

                # Check stops and targets
                alerts = self.check_stops_and_targets()
                if alerts['stops_hit'] or alerts['targets_hit']:
                    # Could trigger notifications here
                    pass

            except Exception as e:
                logger.error(f"Error in price update loop: {e}")

            await asyncio.sleep(self.update_interval)

    def start(self) -> None:
        """Start automatic price updates"""
        if self._running:
            return

        self._running = True
        try:
            loop = asyncio.get_running_loop()
            self._update_task = loop.create_task(self._price_update_loop())
        except RuntimeError:
            # No running event loop available
            pass

        logger.info("PositionTracker started with automatic price updates")

    def stop(self) -> None:
        """Stop automatic price updates"""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            self._update_task = None
        logger.info("PositionTracker stopped")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for all positions.

        Returns:
            Dictionary with summary statistics
        """
        if not self._positions:
            return {
                "total_positions": 0,
                "total_exposure": 0.0,
                "total_unrealized_pnl": 0.0,
                "total_unrealized_pnl_pct": 0.0,
                "total_risk": 0.0,
                "long_positions": 0,
                "short_positions": 0,
                "profitable_positions": 0,
                "losing_positions": 0
            }

        total_exposure = 0.0
        total_pnl = 0.0
        total_cost = 0.0
        total_risk = 0.0
        long_count = 0
        short_count = 0
        profitable_count = 0
        losing_count = 0

        for position in self._positions.values():
            total_exposure += position.current_value
            total_pnl += position.unrealized_pnl
            total_cost += position.cost_basis
            total_risk += abs(position.risk_to_stop)

            if position.direction == PositionDirection.LONG or position.direction == "long":
                long_count += 1
            else:
                short_count += 1

            if position.unrealized_pnl >= 0:
                profitable_count += 1
            else:
                losing_count += 1

        return {
            "total_positions": len(self._positions),
            "total_exposure": round(total_exposure, 2),
            "total_unrealized_pnl": round(total_pnl, 2),
            "total_unrealized_pnl_pct": round((total_pnl / total_cost * 100) if total_cost > 0 else 0, 2),
            "total_risk": round(total_risk, 2),
            "long_positions": long_count,
            "short_positions": short_count,
            "profitable_positions": profitable_count,
            "losing_positions": losing_count
        }

    def reconcile_positions(self, broker) -> Dict[str, Any]:
        """
        Reconcile internal position state with the broker's current positions.

        Fetches live positions from the broker and compares them against the
        internally tracked positions. Three cases are handled:

        1. Broker has a position the tracker does not know about — add it with
           ``reconciled=True`` recorded in its opening event details.
        2. Tracker has a position the broker no longer holds — mark it as stale
           by closing it with reason ``"stale_reconciled"``.
        3. Both sides know about the position but the share quantity differs —
           update the tracker's quantity to match the broker.

        Args:
            broker: A connected BrokerInterface instance.

        Returns:
            Dict summarising what was found and changed:
            ``added``, ``stale``, ``quantity_updated``, ``matched`` counts.
        """
        result = {
            "added": [],
            "stale": [],
            "quantity_updated": [],
            "matched": [],
            "errors": []
        }

        # ------------------------------------------------------------------ #
        # Fetch broker positions — bail out gracefully on any error            #
        # ------------------------------------------------------------------ #
        try:
            broker_positions = broker.get_positions()
        except Exception as exc:
            logger.error(f"Reconciliation: failed to fetch broker positions: {exc}")
            result["errors"].append(str(exc))
            return result

        if broker_positions is None:
            broker_positions = {}

        if not broker_positions:
            logger.info("Reconciliation: broker returned no open positions")
        else:
            logger.info(
                f"Reconciliation: broker has {len(broker_positions)} position(s), "
                f"tracker has {len(self._positions)} position(s)"
            )

        broker_symbols = set(broker_positions.keys())
        tracker_symbols = set(self._positions.keys())

        # ------------------------------------------------------------------ #
        # Case 1: in broker but not in tracker — add as reconciled            #
        # ------------------------------------------------------------------ #
        for symbol in broker_symbols - tracker_symbols:
            broker_pos = broker_positions[symbol]
            try:
                direction = PositionDirection.LONG if broker_pos.quantity > 0 else PositionDirection.SHORT
                avg_cost = broker_pos.avg_cost
                shares = abs(broker_pos.quantity)
                current_price = broker_pos.current_price if broker_pos.current_price else avg_cost

                # Use conservative stop/target placeholders so the Position
                # dataclass invariants are satisfied.  A real operator should
                # update these after reconciliation.
                if direction == PositionDirection.LONG:
                    stop_price = avg_cost * 0.95   # 5 % below cost
                    target_price = avg_cost * 1.10  # 10 % above cost
                else:
                    stop_price = avg_cost * 1.05
                    target_price = avg_cost * 0.90

                position = Position(
                    symbol=symbol,
                    direction=direction,
                    entry_price=avg_cost,
                    shares=shares,
                    stop_price=stop_price,
                    target_price=target_price,
                    current_price=current_price,
                )

                # Tag the opening event so downstream code can distinguish
                # reconciled positions from normally opened ones.
                if position.events:
                    position.events[0].details["reconciled"] = True

                self._positions[symbol] = position
                self._save_position_to_db(position)

                logger.warning(
                    f"Reconciliation: added broker-only position "
                    f"{symbol} ({direction.value}) "
                    f"{shares} shares @ ${avg_cost:.2f} [reconciled=True]"
                )
                result["added"].append(symbol)

            except Exception as exc:
                logger.error(f"Reconciliation: error adding {symbol}: {exc}")
                result["errors"].append(f"{symbol}: {exc}")

        # ------------------------------------------------------------------ #
        # Case 2: in tracker but not in broker — mark as stale/closed         #
        # ------------------------------------------------------------------ #
        for symbol in tracker_symbols - broker_symbols:
            try:
                position = self._positions[symbol]
                current_price = position.current_price or position.entry_price

                logger.warning(
                    f"Reconciliation: tracker position {symbol} not found at broker "
                    f"— marking as stale/closed"
                )

                position.close(exit_price=current_price, reason="stale_reconciled")
                self._closed_positions.append(position)
                del self._positions[symbol]
                self._close_position_in_db(position)

                result["stale"].append(symbol)

            except Exception as exc:
                logger.error(f"Reconciliation: error closing stale {symbol}: {exc}")
                result["errors"].append(f"{symbol}: {exc}")

        # ------------------------------------------------------------------ #
        # Case 3: in both — check for quantity mismatch                       #
        # ------------------------------------------------------------------ #
        for symbol in broker_symbols & tracker_symbols:
            try:
                broker_pos = broker_positions[symbol]
                tracked_pos = self._positions[symbol]

                broker_shares = abs(broker_pos.quantity)
                tracked_shares = abs(tracked_pos.shares)

                if broker_shares != tracked_shares:
                    logger.warning(
                        f"Reconciliation: quantity mismatch for {symbol}: "
                        f"broker={broker_shares}, tracker={tracked_shares} "
                        f"— updating tracker to broker value"
                    )
                    tracked_pos.shares = broker_shares
                    tracked_pos.events.append(PositionEvent(
                        timestamp=datetime.utcnow(),
                        event_type="quantity_reconciled",
                        details={
                            "old_shares": tracked_shares,
                            "new_shares": broker_shares,
                            "source": "broker_reconciliation"
                        }
                    ))
                    self._save_position_to_db(tracked_pos)
                    result["quantity_updated"].append(symbol)
                else:
                    logger.debug(f"Reconciliation: {symbol} matches broker — no action needed")
                    result["matched"].append(symbol)

            except Exception as exc:
                logger.error(f"Reconciliation: error checking {symbol}: {exc}")
                result["errors"].append(f"{symbol}: {exc}")

        # ------------------------------------------------------------------ #
        # Summary log                                                          #
        # ------------------------------------------------------------------ #
        logger.info(
            f"Reconciliation complete: "
            f"{len(result['added'])} added, "
            f"{len(result['stale'])} marked stale, "
            f"{len(result['quantity_updated'])} quantity-updated, "
            f"{len(result['matched'])} matched, "
            f"{len(result['errors'])} error(s)"
        )

        return result

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics from closed positions.

        Returns:
            Dictionary with performance metrics
        """
        if not self._closed_positions:
            # Try to get from database
            if DATABASE_AVAILABLE:
                try:
                    repo = get_trades_repository()
                    return repo.calculate_performance_stats()
                except Exception as e:
                    logger.error(f"Error getting performance stats from database: {e}")

            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "avg_holding_days": 0.0
            }

        winners = [p for p in self._closed_positions if (p.realized_pnl or 0) > 0]
        losers = [p for p in self._closed_positions if (p.realized_pnl or 0) <= 0]

        total_trades = len(self._closed_positions)
        num_winners = len(winners)
        num_losers = len(losers)

        total_pnl = sum(p.realized_pnl or 0 for p in self._closed_positions)
        gross_profit = sum(p.realized_pnl or 0 for p in winners)
        gross_loss = abs(sum(p.realized_pnl or 0 for p in losers))

        avg_win = gross_profit / num_winners if num_winners > 0 else 0
        avg_loss = gross_loss / num_losers if num_losers > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        avg_holding = sum(p.holding_duration for p in self._closed_positions) / total_trades

        return {
            "total_trades": total_trades,
            "wins": num_winners,
            "losses": num_losers,
            "win_rate": round(num_winners / total_trades if total_trades > 0 else 0, 2),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_holding_days": round(avg_holding, 2)
        }


# Global position tracker instance
_position_tracker: Optional[PositionTracker] = None
_position_tracker_lock = threading.Lock()


def get_position_tracker() -> PositionTracker:
    """Get or create the global position tracker instance"""
    global _position_tracker
    if _position_tracker is None:
        with _position_tracker_lock:
            if _position_tracker is None:
                _position_tracker = PositionTracker()
    return _position_tracker

"""
Event System for Inter-Agent Communication
Implements publish-subscribe pattern for agent messaging
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import uuid


class EventType(Enum):
    """Types of events in the trading system"""
    # System events
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_ERROR = "system.error"

    # Market events
    MARKET_OPEN = "market.open"
    MARKET_CLOSE = "market.close"
    MARKET_DATA = "market.data"

    # Scanner events
    SCAN_STARTED = "scanner.started"
    SCAN_COMPLETED = "scanner.completed"
    SIGNAL_FOUND = "scanner.signal_found"

    # Analyzer events
    ANALYSIS_REQUESTED = "analyzer.requested"
    ANALYSIS_COMPLETED = "analyzer.completed"
    SETUP_VALID = "analyzer.setup_valid"
    SETUP_INVALID = "analyzer.setup_invalid"

    # Executor events
    ORDER_REQUESTED = "executor.order_requested"
    ORDER_PLACED = "executor.order_placed"
    ORDER_FILLED = "executor.order_filled"
    ORDER_CANCELLED = "executor.order_cancelled"
    ORDER_REJECTED = "executor.order_rejected"

    # Position events
    POSITION_OPENED = "position.opened"
    POSITION_UPDATED = "position.updated"
    POSITION_CLOSED = "position.closed"
    STOP_HIT = "position.stop_hit"
    TARGET_HIT = "position.target_hit"

    # Risk events
    RISK_CHECK_PASSED = "risk.passed"
    RISK_CHECK_FAILED = "risk.failed"
    DAILY_LIMIT_HIT = "risk.daily_limit"
    DRAWDOWN_WARNING = "risk.drawdown_warning"
    TRADING_HALTED = "risk.trading_halted"

    # Portfolio events
    PORTFOLIO_UPDATED = "portfolio.updated"
    PNL_UPDATED = "portfolio.pnl_updated"


@dataclass
class Event:
    """Event message container"""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source: str = "system"
    priority: int = 5  # 1=highest, 10=lowest

    def __str__(self):
        return f"Event({self.event_type.value}, source={self.source}, id={self.event_id})"


class EventBus:
    """
    Central event bus for agent communication

    Implements async publish-subscribe pattern for decoupled
    communication between trading agents.
    """

    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_queue: asyncio.Queue = None
        self._running = False
        self._event_history: List[Event] = []
        self._max_history = 1000

    async def start(self):
        """Start the event bus"""
        self._event_queue = asyncio.Queue()
        self._running = True
        logger.info("EventBus started")

    async def stop(self):
        """Stop the event bus"""
        self._running = False
        logger.info("EventBus stopped")

    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], Any]
    ):
        """
        Subscribe to an event type

        Args:
            event_type: Type of event to subscribe to
            callback: Async or sync function to call when event occurs
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to {event_type.value}")

    def unsubscribe(
        self,
        event_type: EventType,
        callback: Callable
    ):
        """Unsubscribe from an event type"""
        if event_type in self._subscribers:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)

    async def publish(self, event: Event):
        """
        Publish an event to all subscribers

        Args:
            event: Event to publish
        """
        if not self._running:
            logger.warning("EventBus not running, event not published")
            return

        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)

        logger.debug(f"Publishing: {event}")

        # Get subscribers for this event type
        subscribers = self._subscribers.get(event.event_type, [])

        for callback in subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")

    def publish_sync(self, event: Event):
        """
        Synchronously publish an event (creates new event loop if needed)
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.publish(event))
            else:
                loop.run_until_complete(self.publish(event))
        except RuntimeError:
            asyncio.run(self.publish(event))

    def emit(
        self,
        event_type: EventType,
        data: Dict = None,
        source: str = "system",
        priority: int = 5
    ):
        """
        Convenience method to create and publish an event

        Args:
            event_type: Type of event
            data: Event data
            source: Source agent/component
            priority: Event priority (1=highest)
        """
        event = Event(
            event_type=event_type,
            data=data or {},
            source=source,
            priority=priority
        )
        self.publish_sync(event)

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[Event]:
        """Get event history, optionally filtered by type"""
        if event_type:
            events = [e for e in self._event_history if e.event_type == event_type]
        else:
            events = self._event_history

        return events[-limit:]


# Global event bus instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get or create global event bus instance"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


# Convenience functions
def subscribe(event_type: EventType, callback: Callable):
    """Subscribe to an event type"""
    get_event_bus().subscribe(event_type, callback)


def emit(
    event_type: EventType,
    data: Dict = None,
    source: str = "system"
):
    """Emit an event"""
    get_event_bus().emit(event_type, data, source)

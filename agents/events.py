"""
Event System for Inter-Agent Communication
Implements publish-subscribe pattern for agent messaging
"""

import asyncio
import json
import threading
from datetime import datetime
from typing import Dict, List, Callable, Any, Optional, Set
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
    SYSTEM_MODEL_UPDATED = "system.model_updated"

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
    RISK_ALERT = "risk.alert"
    DAILY_LIMIT_HIT = "risk.daily_limit"
    DRAWDOWN_WARNING = "risk.drawdown_warning"
    TRADING_HALTED = "risk.trading_halted"

    # Portfolio events
    PORTFOLIO_UPDATED = "portfolio.updated"
    PNL_UPDATED = "portfolio.pnl_updated"

    # Market regime events
    REGIME_CHANGE = "regime.change"
    REGIME_DETECTED = "regime.detected"


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
        self._lock = threading.Lock()
        self._background_tasks: Set[asyncio.Task] = set()

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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)

        logger.debug(f"Publishing: {event}")

        # Get snapshot of subscribers under lock
        with self._lock:
            subscribers = list(self._subscribers.get(event.event_type, []))

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
            loop = asyncio.get_running_loop()
            task = asyncio.create_task(self.publish(event))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
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
        with self._lock:
            if event_type:
                events = [e for e in self._event_history if e.event_type == event_type]
            else:
                events = list(self._event_history)

        return events[-limit:]


class PersistentEventBus(EventBus):
    """
    Event bus with database persistence for recovery.

    Persists events before publishing and marks them as processed
    after successful delivery. Supports replaying unprocessed events
    after system restart.
    """

    MAX_RETRY_COUNT = 3

    def __init__(self, db_manager=None):
        """
        Initialize persistent event bus.

        Args:
            db_manager: Database manager instance (lazy loaded if None)
        """
        super().__init__()
        self._db_manager = db_manager

    def _get_db_manager(self):
        """Lazy load database manager."""
        if self._db_manager is None:
            from data.database.connection import get_db_manager
            self._db_manager = get_db_manager()
        return self._db_manager

    def _persist_event(self, event: Event) -> bool:
        """
        Persist event to database before publishing.

        Args:
            event: Event to persist

        Returns:
            True if persisted successfully
        """
        try:
            from data.database.models import EventRecord, EventStatus

            db = self._get_db_manager()
            with db.get_session() as session:
                record = EventRecord(
                    event_id=event.event_id,
                    event_type=event.event_type.value,
                    source=event.source,
                    priority=event.priority,
                    data=json.dumps(event.data, default=str),
                    status=EventStatus.PENDING,
                    created_at=event.timestamp
                )
                session.add(record)
                session.commit()
                logger.debug(f"Persisted event {event.event_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to persist event {event.event_id}: {e}")
            return False

    def _mark_published(self, event_id: str) -> None:
        """Mark event as published in database."""
        try:
            from data.database.models import EventRecord, EventStatus

            db = self._get_db_manager()
            with db.get_session() as session:
                record = session.query(EventRecord).filter_by(event_id=event_id).first()
                if record:
                    record.status = EventStatus.PUBLISHED
                    record.published_at = datetime.now()
                    session.commit()

        except Exception as e:
            logger.error(f"Failed to mark event {event_id} as published: {e}")

    def _mark_processed(self, event_id: str) -> None:
        """Mark event as fully processed in database."""
        try:
            from data.database.models import EventRecord, EventStatus

            db = self._get_db_manager()
            with db.get_session() as session:
                record = session.query(EventRecord).filter_by(event_id=event_id).first()
                if record:
                    record.status = EventStatus.PROCESSED
                    record.processed_at = datetime.now()
                    session.commit()

        except Exception as e:
            logger.error(f"Failed to mark event {event_id} as processed: {e}")

    def _mark_failed(self, event_id: str, error_message: str) -> None:
        """Mark event as failed in database."""
        try:
            from data.database.models import EventRecord, EventStatus

            db = self._get_db_manager()
            with db.get_session() as session:
                record = session.query(EventRecord).filter_by(event_id=event_id).first()
                if record:
                    record.retry_count += 1
                    record.error_message = error_message
                    if record.retry_count >= self.MAX_RETRY_COUNT:
                        record.status = EventStatus.FAILED
                    session.commit()

        except Exception as e:
            logger.error(f"Failed to mark event {event_id} as failed: {e}")

    async def publish(self, event: Event):
        """
        Publish an event with persistence.

        Persists event before publishing, then marks as processed
        after successful delivery to all subscribers.

        Args:
            event: Event to publish
        """
        if not self._running:
            logger.warning("EventBus not running, event not published")
            return

        # Persist before publishing
        persisted = self._persist_event(event)

        # Store in memory history
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)

        logger.debug(f"Publishing: {event}")

        # Get snapshot of subscribers under lock
        with self._lock:
            subscribers = list(self._subscribers.get(event.event_type, []))
        all_succeeded = True
        error_msg = None

        for callback in subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                all_succeeded = False
                error_msg = str(e)
                logger.error(f"Error in event handler: {e}")

        # Update persistence status
        if persisted:
            if all_succeeded:
                self._mark_processed(event.event_id)
            elif error_msg:
                self._mark_failed(event.event_id, error_msg)
            else:
                self._mark_published(event.event_id)

    async def replay_unprocessed(self, max_events: int = 100) -> int:
        """
        Replay unprocessed events from database.

        Used for recovery after system restart. Replays events that
        were persisted but not marked as processed.

        Args:
            max_events: Maximum number of events to replay

        Returns:
            Number of events replayed
        """
        if not self._running:
            logger.warning("EventBus not running, cannot replay events")
            return 0

        try:
            from data.database.models import EventRecord, EventStatus

            db = self._get_db_manager()
            replayed = 0

            with db.get_session() as session:
                # Get unprocessed events (PENDING or PUBLISHED, not FAILED)
                records = (
                    session.query(EventRecord)
                    .filter(EventRecord.status.in_([EventStatus.PENDING, EventStatus.PUBLISHED]))
                    .filter(EventRecord.retry_count < self.MAX_RETRY_COUNT)
                    .order_by(EventRecord.created_at)
                    .limit(max_events)
                    .all()
                )

                for record in records:
                    try:
                        # Reconstruct event
                        event_type = EventType(record.event_type)
                        event = Event(
                            event_type=event_type,
                            data=json.loads(record.data),
                            timestamp=record.created_at,
                            event_id=record.event_id,
                            source=record.source,
                            priority=record.priority
                        )

                        logger.info(f"Replaying event {event.event_id} ({event.event_type.value})")

                        # Publish to subscribers (without re-persisting)
                        with self._lock:
                            subscribers = list(self._subscribers.get(event.event_type, []))
                        all_succeeded = True
                        error_msg = None

                        for callback in subscribers:
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(event)
                                else:
                                    callback(event)
                            except Exception as e:
                                all_succeeded = False
                                error_msg = str(e)
                                logger.error(f"Error replaying to handler: {e}")

                        # Update status
                        if all_succeeded:
                            record.status = EventStatus.PROCESSED
                            record.processed_at = datetime.now()
                            replayed += 1
                        else:
                            record.retry_count += 1
                            record.error_message = error_msg
                            if record.retry_count >= self.MAX_RETRY_COUNT:
                                record.status = EventStatus.FAILED

                    except ValueError:
                        # Invalid event type, mark as failed
                        logger.error(f"Invalid event type: {record.event_type}")
                        record.status = EventStatus.FAILED
                        record.error_message = f"Invalid event type: {record.event_type}"

                session.commit()

            logger.info(f"Replayed {replayed} unprocessed events")
            return replayed

        except Exception as e:
            logger.error(f"Failed to replay unprocessed events: {e}")
            return 0

    def get_unprocessed_count(self) -> int:
        """Get count of unprocessed events in database."""
        try:
            from data.database.models import EventRecord, EventStatus

            db = self._get_db_manager()
            with db.get_session() as session:
                count = (
                    session.query(EventRecord)
                    .filter(EventRecord.status.in_([EventStatus.PENDING, EventStatus.PUBLISHED]))
                    .filter(EventRecord.retry_count < self.MAX_RETRY_COUNT)
                    .count()
                )
                return count

        except Exception as e:
            logger.error(f"Failed to get unprocessed count: {e}")
            return 0

    def cleanup_old_events(self, days: int = 7) -> int:
        """
        Clean up old processed events from database.

        Args:
            days: Delete events older than this many days

        Returns:
            Number of events deleted
        """
        try:
            from datetime import timedelta
            from data.database.models import EventRecord, EventStatus

            db = self._get_db_manager()
            cutoff = datetime.now() - timedelta(days=days)

            with db.get_session() as session:
                deleted = (
                    session.query(EventRecord)
                    .filter(EventRecord.status == EventStatus.PROCESSED)
                    .filter(EventRecord.processed_at < cutoff)
                    .delete()
                )
                session.commit()

            logger.info(f"Cleaned up {deleted} old events")
            return deleted

        except Exception as e:
            logger.error(f"Failed to cleanup old events: {e}")
            return 0


# Global event bus instance
_event_bus: Optional[EventBus] = None
_event_bus_lock = threading.Lock()


def get_event_bus(persistent: bool = False) -> EventBus:
    """
    Get or create global event bus instance.

    Args:
        persistent: If True and no bus exists, create a PersistentEventBus

    Returns:
        EventBus instance (possibly PersistentEventBus)
    """
    global _event_bus
    if _event_bus is None:
        with _event_bus_lock:
            if _event_bus is None:
                if persistent:
                    _event_bus = PersistentEventBus()
                    logger.info("Created PersistentEventBus for event recovery support")
                else:
                    _event_bus = EventBus()
    return _event_bus


def get_persistent_event_bus() -> PersistentEventBus:
    """Get or create a persistent event bus instance."""
    global _event_bus
    with _event_bus_lock:
        if _event_bus is None or not isinstance(_event_bus, PersistentEventBus):
            _event_bus = PersistentEventBus()
            logger.info("Created PersistentEventBus")
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

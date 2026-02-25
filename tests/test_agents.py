"""
Tests for Agent System

Tests:
- Agent lifecycle management
- Event bus communication
- Orchestrator coordination
- Agent state persistence
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from agents.events import (
    EventBus, Event, EventType, get_event_bus, PersistentEventBus
)


class TestEventClass:
    """Tests for Event dataclass"""

    def test_event_creation(self):
        """Test event creation with required fields"""
        event = Event(
            event_type=EventType.SIGNAL_FOUND,
            data={"symbol": "AAPL", "rrs": 2.5}
        )

        assert event.event_type == EventType.SIGNAL_FOUND
        assert event.data["symbol"] == "AAPL"
        assert event.source == "system"
        assert event.priority == 5

    def test_event_with_custom_fields(self):
        """Test event creation with custom fields"""
        event = Event(
            event_type=EventType.ORDER_FILLED,
            data={"order_id": "123"},
            source="executor",
            priority=1
        )

        assert event.source == "executor"
        assert event.priority == 1

    def test_event_has_id(self):
        """Test event has unique ID"""
        event1 = Event(event_type=EventType.MARKET_OPEN, data={})
        event2 = Event(event_type=EventType.MARKET_OPEN, data={})

        assert event1.event_id is not None
        assert event2.event_id is not None
        assert event1.event_id != event2.event_id

    def test_event_has_timestamp(self):
        """Test event has timestamp"""
        event = Event(event_type=EventType.MARKET_OPEN, data={})

        assert event.timestamp is not None
        assert isinstance(event.timestamp, datetime)

    def test_event_str_representation(self):
        """Test event string representation"""
        event = Event(
            event_type=EventType.SIGNAL_FOUND,
            data={},
            source="scanner"
        )

        str_repr = str(event)
        assert "scanner.signal_found" in str_repr
        assert "scanner" in str_repr


class TestEventBus:
    """Tests for EventBus class"""

    @pytest.fixture
    def event_bus(self):
        """Create a fresh event bus for each test"""
        bus = EventBus()
        return bus

    @pytest.mark.asyncio
    async def test_start_stop(self, event_bus):
        """Test event bus start and stop"""
        await event_bus.start()
        assert event_bus._running is True

        await event_bus.stop()
        assert event_bus._running is False

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, event_bus):
        """Test subscribe and publish pattern"""
        await event_bus.start()

        received_events = []

        def handler(event):
            received_events.append(event)

        event_bus.subscribe(EventType.SIGNAL_FOUND, handler)

        event = Event(
            event_type=EventType.SIGNAL_FOUND,
            data={"symbol": "AAPL"}
        )
        await event_bus.publish(event)

        assert len(received_events) == 1
        assert received_events[0].data["symbol"] == "AAPL"

        await event_bus.stop()

    @pytest.mark.asyncio
    async def test_async_handler(self, event_bus):
        """Test async event handler"""
        await event_bus.start()

        received = []

        async def async_handler(event):
            await asyncio.sleep(0.01)  # Simulate async work
            received.append(event)

        event_bus.subscribe(EventType.MARKET_OPEN, async_handler)

        event = Event(event_type=EventType.MARKET_OPEN, data={})
        await event_bus.publish(event)

        assert len(received) == 1

        await event_bus.stop()

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, event_bus):
        """Test multiple subscribers for same event"""
        await event_bus.start()

        count1 = []
        count2 = []

        def handler1(event):
            count1.append(1)

        def handler2(event):
            count2.append(1)

        event_bus.subscribe(EventType.SIGNAL_FOUND, handler1)
        event_bus.subscribe(EventType.SIGNAL_FOUND, handler2)

        event = Event(event_type=EventType.SIGNAL_FOUND, data={})
        await event_bus.publish(event)

        assert len(count1) == 1
        assert len(count2) == 1

        await event_bus.stop()

    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus):
        """Test unsubscribe from event"""
        await event_bus.start()

        received = []

        def handler(event):
            received.append(event)

        event_bus.subscribe(EventType.SIGNAL_FOUND, handler)
        event_bus.unsubscribe(EventType.SIGNAL_FOUND, handler)

        event = Event(event_type=EventType.SIGNAL_FOUND, data={})
        await event_bus.publish(event)

        assert len(received) == 0

        await event_bus.stop()

    @pytest.mark.asyncio
    async def test_event_not_published_when_stopped(self, event_bus):
        """Test events not published when bus is stopped"""
        # Don't start the bus

        received = []

        def handler(event):
            received.append(event)

        event_bus.subscribe(EventType.SIGNAL_FOUND, handler)

        event = Event(event_type=EventType.SIGNAL_FOUND, data={})
        await event_bus.publish(event)

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_handler_error_doesnt_break_bus(self, event_bus):
        """Test that handler errors don't break the bus"""
        await event_bus.start()

        good_received = []

        def bad_handler(event):
            raise ValueError("Test error")

        def good_handler(event):
            good_received.append(event)

        event_bus.subscribe(EventType.SIGNAL_FOUND, bad_handler)
        event_bus.subscribe(EventType.SIGNAL_FOUND, good_handler)

        event = Event(event_type=EventType.SIGNAL_FOUND, data={})
        await event_bus.publish(event)

        # Good handler should still receive event
        assert len(good_received) == 1

        await event_bus.stop()

    @pytest.mark.asyncio
    async def test_event_history(self, event_bus):
        """Test event history tracking"""
        await event_bus.start()

        for i in range(5):
            event = Event(
                event_type=EventType.SIGNAL_FOUND,
                data={"index": i}
            )
            await event_bus.publish(event)

        history = event_bus.get_history(limit=10)
        assert len(history) == 5

        # Test filtered history
        filtered = event_bus.get_history(EventType.SIGNAL_FOUND, limit=3)
        assert len(filtered) == 3

        await event_bus.stop()

    def test_emit_sync(self, event_bus):
        """Test synchronous emit method"""
        event_bus._running = True

        received = []

        def handler(event):
            received.append(event)

        event_bus.subscribe(EventType.MARKET_OPEN, handler)
        event_bus.emit(EventType.MARKET_OPEN, {"session": "regular"}, source="test")

        # Give async task time to complete
        import time
        time.sleep(0.1)


class TestPersistentEventBus:
    """Tests for PersistentEventBus class"""

    @pytest.fixture
    def mock_db(self):
        """Create mock database manager"""
        mock = MagicMock()
        mock.get_session.return_value.__enter__ = MagicMock()
        mock.get_session.return_value.__exit__ = MagicMock()
        return mock

    def test_persistent_bus_creation(self, mock_db):
        """Test PersistentEventBus can be created"""
        bus = PersistentEventBus(db_manager=mock_db)

        assert bus._db_manager is mock_db
        assert bus.MAX_RETRY_COUNT == 3

    @pytest.mark.asyncio
    async def test_event_persistence(self, mock_db):
        """Test events are persisted to database"""
        bus = PersistentEventBus(db_manager=mock_db)
        await bus.start()

        event = Event(
            event_type=EventType.SIGNAL_FOUND,
            data={"symbol": "AAPL"}
        )

        # Mock the persistence
        with patch.object(bus, '_persist_event', return_value=True) as mock_persist:
            with patch.object(bus, '_mark_processed'):
                await bus.publish(event)
                mock_persist.assert_called_once()

        await bus.stop()


class TestEventTypes:
    """Tests for EventType enum"""

    def test_all_system_events(self):
        """Test system events are defined"""
        assert EventType.SYSTEM_START.value == "system.start"
        assert EventType.SYSTEM_STOP.value == "system.stop"
        assert EventType.SYSTEM_ERROR.value == "system.error"

    def test_all_market_events(self):
        """Test market events are defined"""
        assert EventType.MARKET_OPEN.value == "market.open"
        assert EventType.MARKET_CLOSE.value == "market.close"
        assert EventType.MARKET_DATA.value == "market.data"

    def test_all_scanner_events(self):
        """Test scanner events are defined"""
        assert EventType.SCAN_STARTED.value == "scanner.started"
        assert EventType.SCAN_COMPLETED.value == "scanner.completed"
        assert EventType.SIGNAL_FOUND.value == "scanner.signal_found"

    def test_all_risk_events(self):
        """Test risk events are defined"""
        assert EventType.RISK_CHECK_PASSED.value == "risk.passed"
        assert EventType.RISK_CHECK_FAILED.value == "risk.failed"
        assert EventType.TRADING_HALTED.value == "risk.trading_halted"

    def test_all_order_events(self):
        """Test order events are defined"""
        assert EventType.ORDER_REQUESTED.value == "executor.order_requested"
        assert EventType.ORDER_PLACED.value == "executor.order_placed"
        assert EventType.ORDER_FILLED.value == "executor.order_filled"
        assert EventType.ORDER_CANCELLED.value == "executor.order_cancelled"


class TestGlobalEventBus:
    """Tests for global event bus functions"""

    def test_get_event_bus_returns_singleton(self):
        """Test get_event_bus returns same instance"""
        # Reset global for clean test
        import agents.events as events_module
        events_module._event_bus = None

        bus1 = get_event_bus()
        bus2 = get_event_bus()

        assert bus1 is bus2

        # Cleanup
        events_module._event_bus = None

    def test_get_event_bus_persistent(self):
        """Test get_event_bus with persistent flag"""
        import agents.events as events_module
        events_module._event_bus = None

        bus = get_event_bus(persistent=True)

        assert isinstance(bus, PersistentEventBus)

        # Cleanup
        events_module._event_bus = None


class TestAgentLifecycle:
    """Tests for agent lifecycle patterns"""

    @pytest.mark.asyncio
    async def test_agent_startup_sequence(self):
        """Test typical agent startup sequence"""
        bus = EventBus()
        await bus.start()

        startup_events = []

        def track_startup(event):
            startup_events.append(event.event_type)

        bus.subscribe(EventType.SYSTEM_START, track_startup)

        # Simulate system start
        await bus.publish(Event(
            event_type=EventType.SYSTEM_START,
            data={"component": "orchestrator"}
        ))

        assert EventType.SYSTEM_START in startup_events

        await bus.stop()

    @pytest.mark.asyncio
    async def test_agent_shutdown_sequence(self):
        """Test typical agent shutdown sequence"""
        bus = EventBus()
        await bus.start()

        shutdown_events = []

        def track_shutdown(event):
            shutdown_events.append(event.event_type)

        bus.subscribe(EventType.SYSTEM_STOP, track_shutdown)

        # Simulate system stop
        await bus.publish(Event(
            event_type=EventType.SYSTEM_STOP,
            data={"reason": "user_requested"}
        ))

        assert EventType.SYSTEM_STOP in shutdown_events

        await bus.stop()


class TestAgentCommunication:
    """Tests for inter-agent communication patterns"""

    @pytest.mark.asyncio
    async def test_scanner_to_analyzer_flow(self):
        """Test scanner -> analyzer event flow"""
        bus = EventBus()
        await bus.start()

        analysis_requests = []

        def analyzer_handler(event):
            analysis_requests.append(event.data)

        bus.subscribe(EventType.SIGNAL_FOUND, analyzer_handler)

        # Scanner publishes signal
        await bus.publish(Event(
            event_type=EventType.SIGNAL_FOUND,
            data={
                "symbol": "AAPL",
                "rrs": 2.5,
                "price": 175.50
            },
            source="scanner"
        ))

        assert len(analysis_requests) == 1
        assert analysis_requests[0]["symbol"] == "AAPL"

        await bus.stop()

    @pytest.mark.asyncio
    async def test_risk_check_flow(self):
        """Test risk check event flow"""
        bus = EventBus()
        await bus.start()

        risk_results = []

        def risk_handler(event):
            risk_results.append(event.data)

        bus.subscribe(EventType.RISK_CHECK_PASSED, risk_handler)
        bus.subscribe(EventType.RISK_CHECK_FAILED, risk_handler)

        # Simulate passed risk check
        await bus.publish(Event(
            event_type=EventType.RISK_CHECK_PASSED,
            data={
                "symbol": "AAPL",
                "approved_shares": 50,
                "risk_amount": 250.0
            },
            source="risk_agent"
        ))

        assert len(risk_results) == 1
        assert risk_results[0]["approved_shares"] == 50

        await bus.stop()

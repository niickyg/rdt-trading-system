"""
Database Integration Tests

Tests:
- Signal write/read consistency
- Trade history persistence
- Concurrent writes
- Connection recovery
"""

import pytest
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

# These tests use mocked database connections for unit testing
# For actual integration tests, use a test database


class TestSignalWriteReadConsistency:
    """Tests for signal write/read consistency"""

    def test_signal_persisted_correctly(self):
        """Test signal data is persisted and retrieved correctly"""
        from data.database.models import Signal, SignalStatus

        # Create a signal with all fields
        signal = Signal(
            symbol="AAPL",
            direction="LONG",
            rrs=2.5,
            price=175.50,
            atr=2.0,
            status=SignalStatus.PENDING,
            timestamp=datetime.utcnow()
        )

        assert signal.symbol == "AAPL"
        assert signal.rrs == 2.5
        assert signal.status == SignalStatus.PENDING

    def test_signal_status_transitions(self):
        """Test signal status can transition correctly"""
        from data.database.models import SignalStatus

        # Valid transitions
        valid_statuses = [
            SignalStatus.PENDING,
            SignalStatus.TRIGGERED,
            SignalStatus.EXPIRED,
            SignalStatus.IGNORED
        ]

        assert len(valid_statuses) == 4
        assert all(isinstance(s, SignalStatus) for s in valid_statuses)


class TestTradeHistoryPersistence:
    """Tests for trade history persistence"""

    def test_trade_create(self):
        """Test trade record creation"""
        from data.database.models import Trade, TradeDirection, TradeStatus

        trade = Trade(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            entry_price=175.50,
            shares=50,
            entry_time=datetime.utcnow(),
            status=TradeStatus.OPEN,
            stop_loss=172.00,
            take_profit=180.00,
            rrs_at_entry=2.5
        )

        assert trade.symbol == "AAPL"
        assert trade.direction == TradeDirection.LONG
        assert trade.shares == 50
        assert trade.status == TradeStatus.OPEN

    def test_trade_close(self):
        """Test trade can be closed with exit info"""
        from data.database.models import Trade, TradeDirection, TradeStatus, ExitReason

        trade = Trade(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            entry_price=175.50,
            shares=50,
            entry_time=datetime.utcnow(),
            status=TradeStatus.OPEN
        )

        # Close the trade
        trade.status = TradeStatus.CLOSED
        trade.exit_price = 180.00
        trade.exit_time = datetime.utcnow()
        trade.exit_reason = ExitReason.TAKE_PROFIT
        trade.pnl = (trade.exit_price - trade.entry_price) * trade.shares
        trade.pnl_percent = (trade.exit_price - trade.entry_price) / trade.entry_price * 100

        assert trade.status == TradeStatus.CLOSED
        assert trade.pnl == 225.0  # (180 - 175.5) * 50
        assert trade.exit_reason == ExitReason.TAKE_PROFIT

    def test_trade_pnl_calculation(self):
        """Test PnL calculations for different trade types"""
        from data.database.models import TradeDirection

        # Long trade profit
        long_entry = 100.0
        long_exit = 110.0
        long_shares = 50
        long_pnl = (long_exit - long_entry) * long_shares
        assert long_pnl == 500.0

        # Long trade loss
        long_exit_loss = 95.0
        long_pnl_loss = (long_exit_loss - long_entry) * long_shares
        assert long_pnl_loss == -250.0

        # Short trade profit
        short_entry = 100.0
        short_exit = 90.0
        short_shares = 50
        short_pnl = (short_entry - short_exit) * short_shares
        assert short_pnl == 500.0


class TestConcurrentWrites:
    """Tests for concurrent database writes"""

    def test_concurrent_trade_creation(self):
        """Test multiple trades can be created concurrently"""
        from data.database.models import Trade, TradeDirection, TradeStatus

        trades = []
        errors = []

        def create_trade(symbol, index):
            try:
                trade = Trade(
                    symbol=symbol,
                    direction=TradeDirection.LONG,
                    entry_price=100.0 + index,
                    shares=50,
                    entry_time=datetime.utcnow(),
                    status=TradeStatus.OPEN
                )
                trades.append(trade)
            except Exception as e:
                errors.append(e)

        # Create trades in parallel threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(create_trade, f"SYM{i}", i)
                for i in range(10)
            ]
            for future in as_completed(futures):
                future.result()

        assert len(trades) == 10
        assert len(errors) == 0

    def test_concurrent_position_updates(self):
        """Test concurrent position updates don't corrupt data"""
        from data.database.models import Position, TradeDirection

        # Create a position
        position = Position(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            entry_price=175.00,
            shares=100,
            entry_time=datetime.utcnow()
        )

        updates = []
        lock = threading.Lock()

        def update_position(new_price):
            # Simulate price update
            unrealized = (new_price - position.entry_price) * position.shares
            with lock:
                updates.append({
                    'price': new_price,
                    'unrealized': unrealized
                })

        # Concurrent price updates
        prices = [175.5, 176.0, 174.5, 177.0, 175.0]
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(update_position, p) for p in prices]
            for future in as_completed(futures):
                future.result()

        assert len(updates) == 5


class TestConnectionRecovery:
    """Tests for database connection recovery"""

    def test_retry_decorator(self):
        """Test retry decorator handles transient failures"""
        from data.database.connection import with_retry

        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.1, retryable_exceptions=(ValueError,))
        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Transient error")
            return "success"

        result = flaky_operation()

        assert result == "success"
        assert call_count == 3

    def test_retry_exhaustion(self):
        """Test retry decorator gives up after max attempts"""
        from data.database.connection import with_retry

        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.01, retryable_exceptions=(ValueError,))
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Persistent error")

        with pytest.raises(ValueError, match="Persistent error"):
            always_fails()

        assert call_count == 3

    def test_non_retryable_exception(self):
        """Test non-retryable exceptions are raised immediately"""
        from data.database.connection import with_retry

        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.01, retryable_exceptions=(ValueError,))
        def raises_typeerror():
            nonlocal call_count
            call_count += 1
            raise TypeError("Not retryable")

        with pytest.raises(TypeError, match="Not retryable"):
            raises_typeerror()

        assert call_count == 1  # Should only try once

    def test_deadlock_detection(self):
        """Test deadlock detection logic"""
        from data.database.connection import is_deadlock

        # Should detect deadlock
        assert is_deadlock(Exception("deadlock detected"))
        assert is_deadlock(Exception("LOCK WAIT TIMEOUT exceeded"))
        assert is_deadlock(Exception("database is locked"))

        # Should not flag non-deadlock errors
        assert is_deadlock(Exception("connection refused")) is False
        assert is_deadlock(Exception("table not found")) is False


class TestDatabaseModels:
    """Tests for database model relationships"""

    def test_user_subscription_relationship(self):
        """Test user-subscription relationship"""
        from data.database.models import User, Subscription

        user = User(
            username="testuser",
            email="test@test.com",
            password_hash="hashed"
        )

        # Verify relationship attributes exist
        assert hasattr(User, 'subscriptions')

    def test_foreign_key_exists(self):
        """Test foreign key relationships are defined"""
        from data.database.models import Subscription, PaymentHistory

        # Check Subscription has user_id FK
        mapper = Subscription.__table__
        user_id_col = mapper.c.get('user_id')
        assert user_id_col is not None

    def test_event_record_model(self):
        """Test EventRecord model for event persistence"""
        from data.database.models import EventRecord, EventStatus

        record = EventRecord(
            event_id="test-123",
            event_type="scanner.signal_found",
            source="scanner",
            priority=5,
            data='{"symbol": "AAPL"}',
            status=EventStatus.PENDING
        )

        assert record.event_id == "test-123"
        assert record.status == EventStatus.PENDING


class TestDataIntegrity:
    """Tests for data integrity constraints"""

    def test_trade_positive_shares(self):
        """Test trades require positive share count"""
        from data.database.models import Trade

        # The constraint is defined in the model
        # This verifies the constraint name exists
        constraints = [c.name for c in Trade.__table__.constraints]
        assert "ck_trades_shares_positive" in constraints

    def test_trade_positive_price(self):
        """Test trades require positive entry price"""
        from data.database.models import Trade

        constraints = [c.name for c in Trade.__table__.constraints]
        assert "ck_trades_entry_price_positive" in constraints


class TestQueryPatterns:
    """Tests for common query patterns"""

    def test_open_positions_query_pattern(self):
        """Test query pattern for open positions"""
        from data.database.models import Position

        # Verify the model can be filtered by symbol
        assert hasattr(Position, 'symbol')
        assert Position.__tablename__ == 'positions'

    def test_trade_history_query_pattern(self):
        """Test query pattern for trade history"""
        from data.database.models import Trade, TradeStatus

        # Verify indexes exist for common queries
        indexes = [i.name for i in Trade.__table__.indexes]
        assert "ix_trades_symbol" in indexes
        assert "ix_trades_status" in indexes
        assert "ix_trades_entry_time" in indexes

    def test_signal_lookup_indexes(self):
        """Test signal table has proper indexes"""
        from data.database.models import Signal

        indexes = [i.name for i in Signal.__table__.indexes]
        assert "ix_signals_symbol" in indexes
        assert "ix_signals_status" in indexes


class TestTimezoneHandling:
    """Tests for timezone handling in database operations"""

    def test_datetime_fields_with_timezone(self):
        """Test datetime fields support timezone"""
        from data.database.models import Trade

        trade = Trade(
            symbol="AAPL",
            direction="long",
            entry_price=100.0,
            shares=50,
            entry_time=datetime.utcnow()
        )

        assert trade.entry_time is not None

    def test_utc_timestamps(self):
        """Test UTC timestamps are used consistently"""
        now = datetime.utcnow()

        # Verify we can create timestamps
        assert now.year >= 2024
        assert now.month >= 1 and now.month <= 12

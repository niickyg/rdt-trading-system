"""
Comprehensive Unit Tests for Position Tracker

Tests cover:
- Position opening/closing
- P&L calculations
- Stop/target updates
- Trade notes functionality

Run with: pytest tests/test_position_tracker.py -v
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.position_tracker import (
    Position,
    PositionTracker,
    PositionState,
    PositionDirection,
    TradeNote,
    PositionEvent
)


class TestPositionOpening:
    """Test position opening functionality."""

    def test_open_long_position(self):
        """Test opening a long position."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                result = tracker.open_position(
                    symbol="AAPL",
                    direction="long",
                    entry_price=150.00,
                    shares=100,
                    stop_price=145.00,
                    target_price=160.00,
                    rrs_at_entry=2.5
                )

                assert 'error' not in result
                assert result['symbol'] == "AAPL"
                assert result['direction'] == "long"
                assert result['entry_price'] == 150.00
                assert result['shares'] == 100
                assert result['stop_price'] == 145.00
                assert result['target_price'] == 160.00
                assert result['state'] == "open"

    def test_open_short_position(self):
        """Test opening a short position."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                result = tracker.open_position(
                    symbol="TSLA",
                    direction="short",
                    entry_price=200.00,
                    shares=50,
                    stop_price=210.00,  # Stop above entry for short
                    target_price=180.00,  # Target below entry for short
                    rrs_at_entry=-2.5
                )

                assert 'error' not in result
                assert result['direction'] == "short"
                assert result['stop_price'] == 210.00
                assert result['target_price'] == 180.00

    def test_duplicate_position_rejected(self):
        """Test opening duplicate position is rejected."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                tracker.open_position(
                    symbol="AAPL",
                    direction="long",
                    entry_price=150.00,
                    shares=100,
                    stop_price=145.00,
                    target_price=160.00
                )

                result = tracker.open_position(
                    symbol="AAPL",
                    direction="long",
                    entry_price=151.00,
                    shares=50,
                    stop_price=146.00,
                    target_price=161.00
                )

                assert 'error' in result
                assert 'already exists' in result['error']

    def test_invalid_direction_rejected(self):
        """Test invalid direction is rejected."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                result = tracker.open_position(
                    symbol="AAPL",
                    direction="invalid",
                    entry_price=150.00,
                    shares=100,
                    stop_price=145.00,
                    target_price=160.00
                )

                assert 'error' in result
                assert 'Invalid direction' in result['error']

    def test_invalid_stop_for_long_rejected(self):
        """Test stop above entry is rejected for long positions."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                result = tracker.open_position(
                    symbol="AAPL",
                    direction="long",
                    entry_price=150.00,
                    shares=100,
                    stop_price=155.00,  # Stop above entry - invalid for long
                    target_price=160.00
                )

                assert 'error' in result
                assert 'Stop price must be below' in result['error']

    def test_invalid_target_for_long_rejected(self):
        """Test target below entry is rejected for long positions."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                result = tracker.open_position(
                    symbol="AAPL",
                    direction="long",
                    entry_price=150.00,
                    shares=100,
                    stop_price=145.00,
                    target_price=140.00  # Target below entry - invalid for long
                )

                assert 'error' in result
                assert 'Target price must be above' in result['error']

    def test_invalid_stop_for_short_rejected(self):
        """Test stop below entry is rejected for short positions."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                result = tracker.open_position(
                    symbol="TSLA",
                    direction="short",
                    entry_price=200.00,
                    shares=50,
                    stop_price=195.00,  # Stop below entry - invalid for short
                    target_price=180.00
                )

                assert 'error' in result
                assert 'Stop price must be above' in result['error']

    def test_symbol_normalized_uppercase(self):
        """Test symbol is normalized to uppercase."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                result = tracker.open_position(
                    symbol="aapl",  # lowercase
                    direction="long",
                    entry_price=150.00,
                    shares=100,
                    stop_price=145.00,
                    target_price=160.00
                )

                assert result['symbol'] == "AAPL"

    def test_position_event_created(self):
        """Test opening event is recorded."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                result = tracker.open_position(
                    symbol="AAPL",
                    direction="long",
                    entry_price=150.00,
                    shares=100,
                    stop_price=145.00,
                    target_price=160.00
                )

                assert len(result['events']) >= 1
                assert result['events'][0]['event_type'] == 'opened'


class TestPositionClosing:
    """Test position closing functionality."""

    def test_close_position_manual(self):
        """Test closing a position manually."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                tracker.open_position(
                    symbol="AAPL",
                    direction="long",
                    entry_price=150.00,
                    shares=100,
                    stop_price=145.00,
                    target_price=160.00
                )

                result = tracker.close_position(
                    symbol="AAPL",
                    exit_price=155.00,
                    reason="manual"
                )

                assert 'error' not in result
                assert result['state'] == "closed"
                assert result['exit_price'] == 155.00
                assert result['exit_reason'] == "manual"

    def test_close_nonexistent_position(self):
        """Test closing non-existent position returns error."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                result = tracker.close_position(
                    symbol="NVDA",
                    exit_price=500.00,
                    reason="manual"
                )

                assert 'error' in result
                assert 'No open position' in result['error']

    def test_close_position_at_stop_loss(self):
        """Test closing position at stop loss."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                tracker.open_position(
                    symbol="AAPL",
                    direction="long",
                    entry_price=150.00,
                    shares=100,
                    stop_price=145.00,
                    target_price=160.00
                )

                result = tracker.close_position(
                    symbol="AAPL",
                    exit_price=145.00,
                    reason="stop_loss"
                )

                assert result['exit_reason'] == "stop_loss"
                assert result['realized_pnl'] == -500.00  # (145 - 150) * 100

    def test_close_position_at_take_profit(self):
        """Test closing position at take profit."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                tracker.open_position(
                    symbol="AAPL",
                    direction="long",
                    entry_price=150.00,
                    shares=100,
                    stop_price=145.00,
                    target_price=160.00
                )

                result = tracker.close_position(
                    symbol="AAPL",
                    exit_price=160.00,
                    reason="take_profit"
                )

                assert result['exit_reason'] == "take_profit"
                assert result['realized_pnl'] == 1000.00  # (160 - 150) * 100

    def test_close_position_records_event(self):
        """Test closing event is recorded."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                tracker.open_position(
                    symbol="AAPL",
                    direction="long",
                    entry_price=150.00,
                    shares=100,
                    stop_price=145.00,
                    target_price=160.00
                )

                result = tracker.close_position(
                    symbol="AAPL",
                    exit_price=155.00,
                    reason="manual"
                )

                close_event = [e for e in result['events'] if e['event_type'] == 'closed']
                assert len(close_event) == 1


class TestPnLCalculations:
    """Test P&L calculations."""

    def test_unrealized_pnl_long_profit(self):
        """Test unrealized P&L for profitable long position."""
        position = Position(
            symbol="AAPL",
            direction=PositionDirection.LONG,
            entry_price=150.00,
            shares=100,
            stop_price=145.00,
            target_price=160.00,
            current_price=155.00
        )

        assert position.unrealized_pnl == 500.00  # (155 - 150) * 100
        assert position.unrealized_pnl_pct == pytest.approx(3.33, rel=0.01)  # 500 / 15000 * 100

    def test_unrealized_pnl_long_loss(self):
        """Test unrealized P&L for losing long position."""
        position = Position(
            symbol="AAPL",
            direction=PositionDirection.LONG,
            entry_price=150.00,
            shares=100,
            stop_price=145.00,
            target_price=160.00,
            current_price=147.00
        )

        assert position.unrealized_pnl == -300.00  # (147 - 150) * 100

    def test_unrealized_pnl_short_profit(self):
        """Test unrealized P&L for profitable short position."""
        position = Position(
            symbol="TSLA",
            direction=PositionDirection.SHORT,
            entry_price=200.00,
            shares=50,
            stop_price=210.00,
            target_price=180.00,
            current_price=190.00
        )

        # Short profit: (entry - current) * shares = (200 - 190) * 50 = 500
        assert position.unrealized_pnl == 500.00

    def test_unrealized_pnl_short_loss(self):
        """Test unrealized P&L for losing short position."""
        position = Position(
            symbol="TSLA",
            direction=PositionDirection.SHORT,
            entry_price=200.00,
            shares=50,
            stop_price=210.00,
            target_price=180.00,
            current_price=205.00
        )

        # Short loss: (entry - current) * shares = (200 - 205) * 50 = -250
        assert position.unrealized_pnl == -250.00

    def test_realized_pnl_only_for_closed(self):
        """Test realized P&L is only available for closed positions."""
        position = Position(
            symbol="AAPL",
            direction=PositionDirection.LONG,
            entry_price=150.00,
            shares=100,
            stop_price=145.00,
            target_price=160.00,
            current_price=155.00
        )

        assert position.realized_pnl is None

        position.close(exit_price=155.00, reason="manual")
        assert position.realized_pnl == 500.00

    def test_realized_pnl_percentage(self):
        """Test realized P&L percentage calculation."""
        position = Position(
            symbol="AAPL",
            direction=PositionDirection.LONG,
            entry_price=100.00,
            shares=100,
            stop_price=95.00,
            target_price=110.00,
            current_price=100.00
        )

        position.close(exit_price=110.00, reason="take_profit")

        assert position.realized_pnl == 1000.00
        assert position.realized_pnl_pct == 10.0  # 1000 / 10000 * 100

    def test_cost_basis(self):
        """Test cost basis calculation."""
        position = Position(
            symbol="AAPL",
            direction=PositionDirection.LONG,
            entry_price=150.00,
            shares=100,
            stop_price=145.00,
            target_price=160.00
        )

        assert position.cost_basis == 15000.00

    def test_current_value(self):
        """Test current value calculation."""
        position = Position(
            symbol="AAPL",
            direction=PositionDirection.LONG,
            entry_price=150.00,
            shares=100,
            stop_price=145.00,
            target_price=160.00,
            current_price=155.00
        )

        assert position.current_value == 15500.00


class TestStopTargetUpdates:
    """Test stop and target update functionality."""

    def test_update_stop(self):
        """Test updating stop price."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                tracker.open_position(
                    symbol="AAPL",
                    direction="long",
                    entry_price=150.00,
                    shares=100,
                    stop_price=145.00,
                    target_price=160.00
                )

                result = tracker.update_stop("AAPL", 148.00)

                assert 'error' not in result
                assert result['stop_price'] == 148.00

    def test_update_stop_records_event(self):
        """Test stop update is recorded as event."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                tracker.open_position(
                    symbol="AAPL",
                    direction="long",
                    entry_price=150.00,
                    shares=100,
                    stop_price=145.00,
                    target_price=160.00
                )

                result = tracker.update_stop("AAPL", 148.00)

                stop_events = [e for e in result['events'] if e['event_type'] == 'stop_updated']
                assert len(stop_events) == 1
                assert stop_events[0]['details']['old_stop'] == 145.00
                assert stop_events[0]['details']['new_stop'] == 148.00

    def test_update_target(self):
        """Test updating target price."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                tracker.open_position(
                    symbol="AAPL",
                    direction="long",
                    entry_price=150.00,
                    shares=100,
                    stop_price=145.00,
                    target_price=160.00
                )

                result = tracker.update_target("AAPL", 165.00)

                assert 'error' not in result
                assert result['target_price'] == 165.00

    def test_update_target_records_event(self):
        """Test target update is recorded as event."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                tracker.open_position(
                    symbol="AAPL",
                    direction="long",
                    entry_price=150.00,
                    shares=100,
                    stop_price=145.00,
                    target_price=160.00
                )

                result = tracker.update_target("AAPL", 165.00)

                target_events = [e for e in result['events'] if e['event_type'] == 'target_updated']
                assert len(target_events) == 1

    def test_update_nonexistent_position(self):
        """Test updating non-existent position returns error."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                result = tracker.update_stop("NVDA", 500.00)
                assert 'error' in result

                result = tracker.update_target("NVDA", 600.00)
                assert 'error' in result


class TestTradeNotes:
    """Test trade notes functionality."""

    def test_add_note_to_open_position(self):
        """Test adding note to open position."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                tracker.open_position(
                    symbol="AAPL",
                    direction="long",
                    entry_price=150.00,
                    shares=100,
                    stop_price=145.00,
                    target_price=160.00
                )

                result = tracker.add_trade_note(
                    symbol="AAPL",
                    note="Strong volume breakout",
                    note_type="entry"
                )

                assert 'error' not in result
                assert result['notes_count'] == 1
                assert result['notes'][0]['note'] == "Strong volume breakout"
                assert result['notes'][0]['note_type'] == "entry"

    def test_add_multiple_notes(self):
        """Test adding multiple notes."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                tracker.open_position(
                    symbol="AAPL",
                    direction="long",
                    entry_price=150.00,
                    shares=100,
                    stop_price=145.00,
                    target_price=160.00
                )

                tracker.add_trade_note("AAPL", "Entry note", "entry")
                tracker.add_trade_note("AAPL", "Update note", "general")
                result = tracker.add_trade_note("AAPL", "Exit planning", "exit")

                assert result['notes_count'] == 3

    def test_note_records_event(self):
        """Test note addition is recorded as event."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                tracker.open_position(
                    symbol="AAPL",
                    direction="long",
                    entry_price=150.00,
                    shares=100,
                    stop_price=145.00,
                    target_price=160.00
                )

                result = tracker.add_trade_note("AAPL", "Test note", "general")

                note_events = [e for e in result['events'] if e['event_type'] == 'note_added']
                assert len(note_events) == 1

    def test_add_note_to_nonexistent_position(self):
        """Test adding note to non-existent position returns error."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                result = tracker.add_trade_note("NVDA", "Test note", "general")
                assert 'error' in result


class TestStopTargetHitDetection:
    """Test stop and target hit detection."""

    def test_is_stop_hit_long(self):
        """Test stop hit detection for long position."""
        position = Position(
            symbol="AAPL",
            direction=PositionDirection.LONG,
            entry_price=150.00,
            shares=100,
            stop_price=145.00,
            target_price=160.00,
            current_price=144.00  # Below stop
        )

        assert position.is_stop_hit() is True

    def test_is_stop_not_hit_long(self):
        """Test stop not hit for long position."""
        position = Position(
            symbol="AAPL",
            direction=PositionDirection.LONG,
            entry_price=150.00,
            shares=100,
            stop_price=145.00,
            target_price=160.00,
            current_price=148.00  # Above stop
        )

        assert position.is_stop_hit() is False

    def test_is_target_hit_long(self):
        """Test target hit detection for long position."""
        position = Position(
            symbol="AAPL",
            direction=PositionDirection.LONG,
            entry_price=150.00,
            shares=100,
            stop_price=145.00,
            target_price=160.00,
            current_price=161.00  # Above target
        )

        assert position.is_target_hit() is True

    def test_is_stop_hit_short(self):
        """Test stop hit detection for short position."""
        position = Position(
            symbol="TSLA",
            direction=PositionDirection.SHORT,
            entry_price=200.00,
            shares=50,
            stop_price=210.00,
            target_price=180.00,
            current_price=211.00  # Above stop for short
        )

        assert position.is_stop_hit() is True

    def test_is_target_hit_short(self):
        """Test target hit detection for short position."""
        position = Position(
            symbol="TSLA",
            direction=PositionDirection.SHORT,
            entry_price=200.00,
            shares=50,
            stop_price=210.00,
            target_price=180.00,
            current_price=179.00  # Below target for short
        )

        assert position.is_target_hit() is True

    def test_check_stops_and_targets(self):
        """Test checking all positions for stops/targets."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                tracker.open_position(
                    symbol="AAPL",
                    direction="long",
                    entry_price=150.00,
                    shares=100,
                    stop_price=145.00,
                    target_price=160.00
                )

                tracker.open_position(
                    symbol="MSFT",
                    direction="long",
                    entry_price=300.00,
                    shares=50,
                    stop_price=290.00,
                    target_price=320.00
                )

                # Simulate prices
                tracker._positions['AAPL'].current_price = 144.00  # Stop hit
                tracker._positions['MSFT'].current_price = 321.00  # Target hit

                alerts = tracker.check_stops_and_targets()

                assert len(alerts['stops_hit']) == 1
                assert alerts['stops_hit'][0]['symbol'] == 'AAPL'
                assert len(alerts['targets_hit']) == 1
                assert alerts['targets_hit'][0]['symbol'] == 'MSFT'


class TestPositionProperties:
    """Test position property calculations."""

    def test_risk_to_stop(self):
        """Test risk to stop calculation."""
        position = Position(
            symbol="AAPL",
            direction=PositionDirection.LONG,
            entry_price=150.00,
            shares=100,
            stop_price=145.00,
            target_price=160.00
        )

        # Risk = (150 - 145) * 100 = 500
        assert position.risk_to_stop == 500.00

    def test_reward_to_target(self):
        """Test reward to target calculation."""
        position = Position(
            symbol="AAPL",
            direction=PositionDirection.LONG,
            entry_price=150.00,
            shares=100,
            stop_price=145.00,
            target_price=160.00
        )

        # Reward = (160 - 150) * 100 = 1000
        assert position.reward_to_target == 1000.00

    def test_risk_reward_ratio(self):
        """Test risk/reward ratio calculation."""
        position = Position(
            symbol="AAPL",
            direction=PositionDirection.LONG,
            entry_price=150.00,
            shares=100,
            stop_price=145.00,
            target_price=160.00
        )

        # R:R = 1000 / 500 = 2.0
        assert position.risk_reward_ratio == 2.0

    def test_distance_to_stop_pct(self):
        """Test distance to stop percentage."""
        position = Position(
            symbol="AAPL",
            direction=PositionDirection.LONG,
            entry_price=150.00,
            shares=100,
            stop_price=145.00,
            target_price=160.00,
            current_price=152.00
        )

        # Distance = (152 - 145) / 152 * 100 = 4.6%
        assert position.distance_to_stop_pct == pytest.approx(4.6, rel=0.01)

    def test_holding_duration(self):
        """Test holding duration calculation."""
        position = Position(
            symbol="AAPL",
            direction=PositionDirection.LONG,
            entry_price=150.00,
            shares=100,
            stop_price=145.00,
            target_price=160.00,
            entry_time=datetime.utcnow() - timedelta(days=2)
        )

        assert position.holding_duration >= 2.0

    def test_to_dict(self):
        """Test position to_dict conversion."""
        position = Position(
            symbol="AAPL",
            direction=PositionDirection.LONG,
            entry_price=150.00,
            shares=100,
            stop_price=145.00,
            target_price=160.00,
            current_price=155.00
        )

        data = position.to_dict()

        assert data['symbol'] == "AAPL"
        assert data['direction'] == "long"
        assert data['entry_price'] == 150.00
        assert data['shares'] == 100
        assert 'unrealized_pnl' in data
        assert 'risk_reward_ratio' in data


class TestTrackerMethods:
    """Test PositionTracker methods."""

    def test_get_position(self):
        """Test getting a single position."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                tracker.open_position(
                    symbol="AAPL",
                    direction="long",
                    entry_price=150.00,
                    shares=100,
                    stop_price=145.00,
                    target_price=160.00
                )

                result = tracker.get_position("AAPL")
                assert result is not None
                assert result['symbol'] == "AAPL"

                result = tracker.get_position("NVDA")
                assert result is None

    def test_get_all_positions(self):
        """Test getting all positions."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                tracker.open_position("AAPL", "long", 150.00, 100, 145.00, 160.00)
                tracker.open_position("MSFT", "long", 300.00, 50, 290.00, 320.00)
                tracker.open_position("GOOGL", "short", 140.00, 75, 150.00, 120.00)

                positions = tracker.get_all_positions()
                assert len(positions) == 3

    def test_get_summary(self):
        """Test summary statistics."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                tracker.open_position("AAPL", "long", 150.00, 100, 145.00, 160.00)
                tracker._positions['AAPL'].current_price = 155.00  # Profitable

                tracker.open_position("MSFT", "long", 300.00, 50, 290.00, 320.00)
                tracker._positions['MSFT'].current_price = 295.00  # Losing

                summary = tracker.get_summary()

                assert summary['total_positions'] == 2
                assert summary['long_positions'] == 2
                assert summary['short_positions'] == 0
                assert summary['profitable_positions'] == 1
                assert summary['losing_positions'] == 1
                assert summary['total_exposure'] > 0

    def test_get_trade_history(self):
        """Test trade history retrieval."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                tracker.open_position("AAPL", "long", 150.00, 100, 145.00, 160.00)
                tracker.add_trade_note("AAPL", "Entry note", "entry")

                history = tracker.get_trade_history("AAPL")

                assert 'error' not in history
                assert history['symbol'] == "AAPL"
                assert history['status'] == "open"
                assert 'events' in history
                assert 'notes' in history

    def test_get_performance_stats_empty(self):
        """Test performance stats with no closed positions."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                stats = tracker.get_performance_stats()

                assert stats['total_trades'] == 0
                assert stats['win_rate'] == 0.0

    def test_get_performance_stats_with_trades(self):
        """Test performance stats with closed positions."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                # Win
                tracker.open_position("AAPL", "long", 150.00, 100, 145.00, 160.00)
                tracker.close_position("AAPL", 160.00, "take_profit")

                # Win
                tracker.open_position("MSFT", "long", 300.00, 50, 290.00, 320.00)
                tracker.close_position("MSFT", 320.00, "take_profit")

                # Loss
                tracker.open_position("TSLA", "long", 200.00, 25, 190.00, 220.00)
                tracker.close_position("TSLA", 190.00, "stop_loss")

                stats = tracker.get_performance_stats()

                assert stats['total_trades'] == 3
                assert stats['wins'] == 2
                assert stats['losses'] == 1
                assert stats['win_rate'] == pytest.approx(0.67, rel=0.01)


class TestTrackerStartStop:
    """Test tracker start/stop functionality."""

    def test_start_stop(self):
        """Test starting and stopping tracker."""
        with patch('trading.position_tracker.DATABASE_AVAILABLE', False):
            with patch('trading.position_tracker.DATA_PROVIDER_AVAILABLE', False):
                tracker = PositionTracker(auto_update_prices=False)

                tracker.start()
                assert tracker._running is True

                tracker.stop()
                assert tracker._running is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

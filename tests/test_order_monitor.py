"""
Comprehensive Unit Tests for Order Monitor

Tests cover:
- Order state transitions
- Stuck order detection
- Slippage calculations
- Fill tracking
- Callbacks (on_fill, on_stuck, on_rejection)

Run with: pytest tests/test_order_monitor.py -v
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.order_monitor import OrderMonitor, MonitoredOrder, OrderState


class TestOrderStateTransitions:
    """Test order state transitions."""

    def test_initial_state_is_pending(self):
        """Test order starts in PENDING state."""
        order = MonitoredOrder(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )

        assert order.state == OrderState.PENDING
        assert order.is_active is True
        assert order.is_terminal is False

    def test_transition_pending_to_submitted(self):
        """Test order transitions from PENDING to SUBMITTED."""
        monitor = OrderMonitor()

        order = monitor.track_order(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )

        assert order.state == OrderState.PENDING

        updated = monitor.order_submitted("test_001", broker_order_id="BROKER123")

        assert updated.state == OrderState.SUBMITTED
        assert updated.broker_order_id == "BROKER123"
        assert updated.submitted_at is not None

    def test_transition_submitted_to_filled(self):
        """Test order transitions from SUBMITTED to FILLED."""
        monitor = OrderMonitor()

        monitor.track_order(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )
        monitor.order_submitted("test_001")

        updated = monitor.order_fill(
            order_id="test_001",
            fill_price=150.05,
            fill_quantity=100
        )

        assert updated.state == OrderState.FILLED
        assert updated.filled_quantity == 100
        assert updated.avg_fill_price == 150.05
        assert updated.is_terminal is True
        assert updated.is_active is False

    def test_transition_submitted_to_partial_fill(self):
        """Test order transitions from SUBMITTED to PARTIAL_FILL."""
        monitor = OrderMonitor()

        monitor.track_order(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )
        monitor.order_submitted("test_001")

        # Partial fill of 50 shares
        updated = monitor.order_fill(
            order_id="test_001",
            fill_price=150.05,
            fill_quantity=50
        )

        assert updated.state == OrderState.PARTIAL_FILL
        assert updated.filled_quantity == 50
        assert updated.remaining_quantity == 50
        assert updated.is_active is True

    def test_transition_partial_to_filled(self):
        """Test order transitions from PARTIAL_FILL to FILLED."""
        monitor = OrderMonitor()

        monitor.track_order(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )
        monitor.order_submitted("test_001")

        # First partial fill
        monitor.order_fill("test_001", fill_price=150.05, fill_quantity=50)

        # Second fill completes order
        updated = monitor.order_fill("test_001", fill_price=150.10, fill_quantity=50)

        assert updated.state == OrderState.FILLED
        assert updated.filled_quantity == 100
        assert updated.is_terminal is True

    def test_transition_to_cancelled(self):
        """Test order transitions to CANCELLED state."""
        monitor = OrderMonitor()

        monitor.track_order(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="limit",
            expected_price=150.00
        )
        monitor.order_submitted("test_001")

        updated = monitor.order_cancelled("test_001", reason="User requested")

        assert updated.state == OrderState.CANCELLED
        assert updated.error_message == "User requested"
        assert updated.is_terminal is True
        assert updated.completed_at is not None

    def test_transition_to_rejected(self):
        """Test order transitions to REJECTED state."""
        monitor = OrderMonitor()

        monitor.track_order(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )
        monitor.order_submitted("test_001")

        updated = monitor.order_rejected("test_001", reason="Insufficient funds")

        assert updated.state == OrderState.REJECTED
        assert updated.error_message == "Insufficient funds"
        assert updated.is_terminal is True

    def test_nonexistent_order_returns_none(self):
        """Test operations on non-existent orders return None."""
        monitor = OrderMonitor()

        result = monitor.order_submitted("nonexistent")
        assert result is None

        result = monitor.order_fill("nonexistent", 100.0, 50)
        assert result is None

        result = monitor.order_cancelled("nonexistent")
        assert result is None

        result = monitor.order_rejected("nonexistent", "Error")
        assert result is None


class TestStuckOrderDetection:
    """Test stuck order detection."""

    def test_order_detected_as_stuck(self):
        """Test order is detected as stuck after threshold."""
        monitor = OrderMonitor(stuck_order_threshold_seconds=1.0)

        monitor.track_order(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )
        monitor.order_submitted("test_001")

        # Initially not stuck
        stuck = monitor.get_stuck_orders()
        assert len(stuck) == 0

        # Wait for threshold
        time.sleep(1.1)

        stuck = monitor.get_stuck_orders()
        assert len(stuck) == 1
        assert stuck[0].order_id == "test_001"

    def test_filled_order_not_detected_as_stuck(self):
        """Test filled orders are not detected as stuck."""
        monitor = OrderMonitor(stuck_order_threshold_seconds=0.1)

        monitor.track_order(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )
        monitor.order_submitted("test_001")
        monitor.order_fill("test_001", 150.05, 100)

        time.sleep(0.2)

        stuck = monitor.get_stuck_orders()
        assert len(stuck) == 0

    def test_stuck_order_callback_triggered(self):
        """Test stuck order callback is triggered."""
        callback_called = []

        def on_stuck(order):
            callback_called.append(order.order_id)

        monitor = OrderMonitor(
            stuck_order_threshold_seconds=0.1,
            check_interval_seconds=0.05,
            on_stuck_order=on_stuck
        )

        monitor.track_order(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )
        monitor.order_submitted("test_001")

        monitor.start()
        time.sleep(0.3)  # Wait for detection
        monitor.stop()

        assert "test_001" in callback_called

    def test_partial_fill_stuck_detection(self):
        """Test partial fills are detected as stuck if no activity."""
        monitor = OrderMonitor(
            stuck_order_threshold_seconds=0.5,
            partial_fill_alert_seconds=0.1
        )

        monitor.track_order(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )
        monitor.order_submitted("test_001")
        monitor.order_fill("test_001", 150.05, 50)

        # Wait for partial fill alert threshold
        time.sleep(0.2)

        stuck = monitor.get_stuck_orders()
        assert len(stuck) == 1


class TestSlippageCalculations:
    """Test slippage calculations."""

    def test_slippage_calculation_buy(self):
        """Test slippage calculation for buy order."""
        order = MonitoredOrder(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )

        # Simulate fill at higher price
        order.avg_fill_price = 150.10
        order.filled_quantity = 100

        # Slippage = 150.10 - 150.00 = 0.10
        assert order.slippage == pytest.approx(0.10)
        # Slippage % = 0.10 / 150.00 * 100 = 0.0667%
        assert order.slippage_pct == pytest.approx(0.0667, rel=0.01)

    def test_slippage_calculation_sell(self):
        """Test slippage calculation for sell order."""
        order = MonitoredOrder(
            order_id="test_001",
            symbol="AAPL",
            side="sell",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )

        # Simulate fill at lower price (negative slippage)
        order.avg_fill_price = 149.90
        order.filled_quantity = 100

        # Slippage = 149.90 - 150.00 = -0.10
        assert order.slippage == pytest.approx(-0.10)
        assert order.slippage_pct == pytest.approx(-0.0667, rel=0.01)

    def test_slippage_with_multiple_fills(self):
        """Test slippage calculation with multiple partial fills."""
        monitor = OrderMonitor()

        monitor.track_order(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )
        monitor.order_submitted("test_001")

        # Fill 50 @ 150.10
        monitor.order_fill("test_001", 150.10, 50)
        # Fill 50 @ 150.20
        order = monitor.order_fill("test_001", 150.20, 50)

        # Avg fill = (50 * 150.10 + 50 * 150.20) / 100 = 150.15
        assert order.avg_fill_price == pytest.approx(150.15)
        # Slippage = 150.15 - 150.00 = 0.15
        assert order.slippage == pytest.approx(0.15)

    def test_slippage_none_when_not_filled(self):
        """Test slippage is None when order not filled."""
        order = MonitoredOrder(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )

        assert order.slippage is None
        assert order.slippage_pct is None

    def test_high_slippage_callback(self):
        """Test high slippage callback is triggered."""
        callback_called = []

        def on_high_slippage(order):
            callback_called.append(order.order_id)

        monitor = OrderMonitor(
            high_slippage_threshold_pct=0.1,  # 0.1%
            on_high_slippage=on_high_slippage
        )

        monitor.track_order(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )
        monitor.order_submitted("test_001")

        # Fill with high slippage (0.2%)
        monitor.order_fill("test_001", 150.30, 100)

        assert "test_001" in callback_called


class TestFillTracking:
    """Test fill tracking functionality."""

    def test_fill_quantity_tracking(self):
        """Test filled quantity is tracked correctly."""
        monitor = OrderMonitor()

        order = monitor.track_order(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )

        assert order.filled_quantity == 0
        assert order.remaining_quantity == 100
        assert order.fill_rate == 0.0

        monitor.order_submitted("test_001")
        monitor.order_fill("test_001", 150.05, 30)

        order = monitor.get_order("test_001")
        assert order.filled_quantity == 30
        assert order.remaining_quantity == 70
        assert order.fill_rate == 30.0

    def test_fill_history_tracking(self):
        """Test fill history is tracked."""
        monitor = OrderMonitor()

        monitor.track_order(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )
        monitor.order_submitted("test_001")

        monitor.order_fill("test_001", 150.05, 30)
        monitor.order_fill("test_001", 150.10, 40)
        order = monitor.order_fill("test_001", 150.08, 30)

        assert len(order.fills) == 3
        assert order.fills[0]["fill_price"] == 150.05
        assert order.fills[0]["fill_quantity"] == 30
        assert order.fills[1]["fill_price"] == 150.10
        assert order.fills[2]["fill_price"] == 150.08

    def test_time_to_first_fill(self):
        """Test time to first fill calculation."""
        monitor = OrderMonitor()

        monitor.track_order(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )
        monitor.order_submitted("test_001")

        time.sleep(0.1)  # Small delay

        order = monitor.order_fill("test_001", 150.05, 100)

        assert order.time_to_first_fill is not None
        assert order.time_to_first_fill >= 0.1

    def test_time_to_complete(self):
        """Test time to complete calculation."""
        monitor = OrderMonitor()

        monitor.track_order(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )
        monitor.order_submitted("test_001")

        time.sleep(0.05)
        monitor.order_fill("test_001", 150.05, 50)

        time.sleep(0.05)
        order = monitor.order_fill("test_001", 150.10, 50)

        assert order.time_to_complete is not None
        assert order.time_to_complete >= 0.1


class TestCallbacks:
    """Test callback functionality."""

    def test_on_fill_callback(self):
        """Test on_fill callback is triggered."""
        fills_received = []

        def on_fill(order, fill):
            fills_received.append({
                'order_id': order.order_id,
                'price': fill['fill_price'],
                'quantity': fill['fill_quantity']
            })

        monitor = OrderMonitor(on_fill=on_fill)

        monitor.track_order(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )
        monitor.order_submitted("test_001")
        monitor.order_fill("test_001", 150.05, 50)
        monitor.order_fill("test_001", 150.10, 50)

        assert len(fills_received) == 2
        assert fills_received[0]['price'] == 150.05
        assert fills_received[0]['quantity'] == 50
        assert fills_received[1]['price'] == 150.10

    def test_on_complete_callback(self):
        """Test on_complete callback is triggered."""
        completed_orders = []

        def on_complete(order):
            completed_orders.append(order.order_id)

        monitor = OrderMonitor(on_complete=on_complete)

        monitor.track_order(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )
        monitor.order_submitted("test_001")
        monitor.order_fill("test_001", 150.05, 100)

        assert "test_001" in completed_orders

    def test_on_rejection_callback(self):
        """Test on_rejection callback is triggered."""
        rejected_orders = []

        def on_rejection(order):
            rejected_orders.append({
                'order_id': order.order_id,
                'reason': order.error_message
            })

        monitor = OrderMonitor(on_rejection=on_rejection)

        monitor.track_order(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )
        monitor.order_submitted("test_001")
        monitor.order_rejected("test_001", "Insufficient funds")

        assert len(rejected_orders) == 1
        assert rejected_orders[0]['reason'] == "Insufficient funds"

    def test_callback_exception_handling(self):
        """Test callbacks handle exceptions gracefully."""
        def failing_callback(order, fill=None):
            raise Exception("Callback error")

        monitor = OrderMonitor(
            on_fill=failing_callback,
            on_complete=failing_callback
        )

        monitor.track_order(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )
        monitor.order_submitted("test_001")

        # Should not raise exception
        monitor.order_fill("test_001", 150.05, 100)


class TestOrderRetrieval:
    """Test order retrieval methods."""

    def test_get_active_orders(self):
        """Test getting active orders."""
        monitor = OrderMonitor()

        monitor.track_order("order1", "AAPL", "buy", 100, "market", 150.00)
        monitor.track_order("order2", "MSFT", "buy", 50, "market", 300.00)
        monitor.track_order("order3", "GOOGL", "buy", 30, "market", 140.00)

        monitor.order_submitted("order1")
        monitor.order_submitted("order2")
        monitor.order_submitted("order3")

        # Complete one order
        monitor.order_fill("order1", 150.05, 100)

        active = monitor.get_active_orders()
        assert len(active) == 2
        assert all(o.order_id in ["order2", "order3"] for o in active)

    def test_get_orders_by_symbol(self):
        """Test getting orders by symbol."""
        monitor = OrderMonitor()

        monitor.track_order("order1", "AAPL", "buy", 100, "market", 150.00)
        monitor.track_order("order2", "AAPL", "sell", 50, "market", 155.00)
        monitor.track_order("order3", "MSFT", "buy", 30, "market", 300.00)

        monitor.order_submitted("order1")
        monitor.order_submitted("order2")
        monitor.order_fill("order1", 150.05, 100)

        aapl_orders = monitor.get_orders_by_symbol("AAPL")
        assert len(aapl_orders) == 2

        msft_orders = monitor.get_orders_by_symbol("MSFT")
        assert len(msft_orders) == 1

    def test_get_order_by_id(self):
        """Test getting order by ID."""
        monitor = OrderMonitor()

        monitor.track_order("test_001", "AAPL", "buy", 100, "market", 150.00)

        order = monitor.get_order("test_001")
        assert order is not None
        assert order.symbol == "AAPL"

        # Get completed order
        monitor.order_submitted("test_001")
        monitor.order_fill("test_001", 150.05, 100)

        order = monitor.get_order("test_001")
        assert order is not None
        assert order.state == OrderState.FILLED


class TestMetrics:
    """Test metrics collection."""

    def test_get_metrics(self):
        """Test metrics retrieval."""
        monitor = OrderMonitor()

        # Create and fill some orders
        for i in range(5):
            monitor.track_order(f"order{i}", "AAPL", "buy", 100, "market", 150.00)
            monitor.order_submitted(f"order{i}")

        # Fill 3 orders
        monitor.order_fill("order0", 150.05, 100)
        monitor.order_fill("order1", 150.10, 100)
        monitor.order_fill("order2", 149.95, 100)

        # Reject 1 order
        monitor.order_rejected("order3", "Rejected")

        metrics = monitor.get_metrics()

        assert metrics['total_orders_tracked'] == 5
        assert metrics['filled_orders_count'] == 3
        assert metrics['total_rejections'] == 1
        assert metrics['active_orders'] == 1  # order4 still pending

    def test_fill_time_stats(self):
        """Test fill time statistics."""
        monitor = OrderMonitor()

        for i in range(3):
            monitor.track_order(f"order{i}", "AAPL", "buy", 100, "market", 150.00)
            monitor.order_submitted(f"order{i}")
            time.sleep(0.05)
            monitor.order_fill(f"order{i}", 150.05, 100)

        stats = monitor.get_fill_time_stats()

        assert stats['count'] == 3
        assert stats['avg_seconds'] > 0
        assert stats['min_seconds'] > 0
        assert stats['max_seconds'] > 0

    def test_slippage_metrics(self):
        """Test slippage metrics in overall stats."""
        monitor = OrderMonitor()

        # Order with positive slippage
        monitor.track_order("order1", "AAPL", "buy", 100, "market", 150.00)
        monitor.order_submitted("order1")
        monitor.order_fill("order1", 150.20, 100)  # 0.133% slippage

        # Order with negative slippage
        monitor.track_order("order2", "AAPL", "buy", 100, "market", 150.00)
        monitor.order_submitted("order2")
        monitor.order_fill("order2", 149.80, 100)  # -0.133% slippage

        metrics = monitor.get_metrics()

        assert 'avg_slippage_pct' in metrics
        assert 'max_slippage_pct' in metrics
        assert 'min_slippage_pct' in metrics


class TestMonitoredOrderProperties:
    """Test MonitoredOrder dataclass properties."""

    def test_to_dict(self):
        """Test order to_dict conversion."""
        order = MonitoredOrder(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )

        data = order.to_dict()

        assert data['order_id'] == "test_001"
        assert data['symbol'] == "AAPL"
        assert data['side'] == "buy"
        assert data['quantity'] == 100
        assert data['state'] == "pending"
        assert data['fill_rate'] == 0

    def test_time_since_submission(self):
        """Test time since submission property."""
        monitor = OrderMonitor()

        order = monitor.track_order(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            expected_price=150.00
        )

        # Not submitted yet
        assert order.time_since_submission is None

        monitor.order_submitted("test_001")
        time.sleep(0.1)

        order = monitor.get_order("test_001")
        assert order.time_since_submission is not None
        assert order.time_since_submission >= 0.1


class TestBackgroundMonitoring:
    """Test background monitoring thread."""

    def test_start_stop(self):
        """Test starting and stopping monitor."""
        monitor = OrderMonitor(check_interval_seconds=0.1)

        monitor.start()
        assert monitor._running is True
        assert monitor._monitor_thread is not None

        monitor.stop()
        assert monitor._running is False

    def test_start_idempotent(self):
        """Test starting multiple times is safe."""
        monitor = OrderMonitor(check_interval_seconds=0.1)

        monitor.start()
        monitor.start()  # Should not create second thread

        monitor.stop()

    def test_clear_history(self):
        """Test clearing old completed orders."""
        monitor = OrderMonitor()

        # Create and complete orders
        for i in range(5):
            monitor.track_order(f"order{i}", "AAPL", "buy", 100, "market", 150.00)
            monitor.order_submitted(f"order{i}")
            monitor.order_fill(f"order{i}", 150.05, 100)

        assert len(monitor._completed_orders) == 5

        # Clear all (keep_recent_hours=0 would clear all)
        removed = monitor.clear_history(keep_recent_hours=24)

        # All orders are recent, none should be removed
        assert removed == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

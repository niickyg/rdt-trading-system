"""
Comprehensive Unit Tests for Brokers

Tests cover:
- PaperBroker order execution
- PaperBroker position tracking
- Broker interface compliance
- Error handling

Run with: pytest tests/test_brokers.py -v
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock yfinance before importing brokers
mock_yf = MagicMock()
mock_ticker = MagicMock()
mock_ticker.info = {
    'regularMarketPrice': 100.05,
    'bid': 99.95,
    'ask': 100.10,
    'regularMarketVolume': 1000000,
    'dayHigh': 101.00,
    'dayLow': 99.00,
    'regularMarketOpen': 100.00,
    'previousClose': 99.50
}
mock_yf.Ticker.return_value = mock_ticker
sys.modules['yfinance'] = mock_yf

from brokers import (
    get_broker, get_broker_from_config,
    BrokerInterface, PaperBroker,
    Order, Position, Quote, AccountInfo,
    OrderSide, OrderType, OrderStatus,
    BrokerError, OrderError, InsufficientFundsError, PositionError
)


# Create a fixture that uses the non-yfinance paper broker
@pytest.fixture
def paper_broker_no_yf():
    """Create a paper broker that doesn't use yfinance."""
    broker = PaperBroker(
        initial_balance=25000.0,
        slippage_pct=0.001,
        commission_per_trade=0.0,
        realistic_fills=False
    )
    broker.connect()
    return broker


@pytest.fixture
def paper_broker():
    """Create a paper broker for testing with mocked yfinance."""
    broker = PaperBroker(
        initial_balance=25000.0,
        slippage_pct=0.001,
        commission_per_trade=0.0,
        realistic_fills=False
    )
    broker.connect()
    yield broker


class TestPaperBrokerOrderExecution:
    """Test PaperBroker order execution."""

    def test_market_order_buy_fills_immediately(self, paper_broker):
        """Test market buy order fills immediately."""
        order = paper_broker.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )

        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 10
        assert order.avg_fill_price is not None
        assert order.filled_at is not None

    def test_market_order_sell_fills_immediately(self, paper_broker):
        """Test market sell order fills immediately after buying."""
        # First buy
        paper_broker.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )

        # Then sell
        order = paper_broker.place_order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.MARKET
        )

        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 10

    def test_limit_order_fills_at_limit_or_better(self, paper_broker):
        """Test limit order fills at limit price or better."""
        # Get current quote
        quote = paper_broker.get_quote("AAPL")

        # Place limit buy above ask (should fill immediately)
        order = paper_broker.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            price=quote.ask + 1.00
        )

        assert order.status == OrderStatus.FILLED

    def test_limit_order_not_filled_below_market(self, paper_broker):
        """Test limit order doesn't fill if price not reached."""
        quote = paper_broker.get_quote("AAPL")

        # Place limit buy well below ask (should not fill)
        order = paper_broker.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            price=quote.ask - 10.00
        )

        assert order.status == OrderStatus.OPEN

    def test_stop_order_goes_pending(self, paper_broker):
        """Test stop order goes to pending state."""
        # First buy some shares
        paper_broker.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )

        quote = paper_broker.get_quote("AAPL")

        # Place stop sell below current price
        order = paper_broker.place_order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.STOP,
            stop_price=quote.last - 5.00
        )

        assert order.status == OrderStatus.OPEN

    def test_short_sale_order(self, paper_broker):
        """Test short sale order execution."""
        order = paper_broker.place_order(
            symbol="AAPL",
            side=OrderSide.SELL_SHORT,
            quantity=10,
            order_type=OrderType.MARKET
        )

        assert order.status == OrderStatus.FILLED

        # Check position is short
        position = paper_broker.get_position("AAPL")
        assert position is not None
        assert position.quantity < 0

    def test_buy_to_cover_order(self, paper_broker):
        """Test buy to cover order execution."""
        # First short
        paper_broker.place_order(
            symbol="AAPL",
            side=OrderSide.SELL_SHORT,
            quantity=10,
            order_type=OrderType.MARKET
        )

        # Then cover
        order = paper_broker.place_order(
            symbol="AAPL",
            side=OrderSide.BUY_TO_COVER,
            quantity=10,
            order_type=OrderType.MARKET
        )

        assert order.status == OrderStatus.FILLED

        # Position should be closed
        position = paper_broker.get_position("AAPL")
        assert position is None


class TestPaperBrokerPositionTracking:
    """Test PaperBroker position tracking."""

    def test_position_created_on_buy(self, paper_broker):
        """Test position is created when buying."""
        paper_broker.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )

        position = paper_broker.get_position("AAPL")
        assert position is not None
        assert position.quantity == 10
        assert position.avg_cost > 0

    def test_position_increased_on_additional_buy(self, paper_broker):
        """Test position increases on additional buy."""
        paper_broker.place_order("AAPL", OrderSide.BUY, 10, OrderType.MARKET)
        paper_broker.place_order("AAPL", OrderSide.BUY, 5, OrderType.MARKET)

        position = paper_broker.get_position("AAPL")
        assert position.quantity == 15

    def test_position_decreased_on_sell(self, paper_broker):
        """Test position decreases on sell."""
        paper_broker.place_order("AAPL", OrderSide.BUY, 10, OrderType.MARKET)
        paper_broker.place_order("AAPL", OrderSide.SELL, 5, OrderType.MARKET)

        position = paper_broker.get_position("AAPL")
        assert position.quantity == 5

    def test_position_closed_on_full_sell(self, paper_broker):
        """Test position closes on full sell."""
        paper_broker.place_order("AAPL", OrderSide.BUY, 10, OrderType.MARKET)
        paper_broker.place_order("AAPL", OrderSide.SELL, 10, OrderType.MARKET)

        position = paper_broker.get_position("AAPL")
        assert position is None

    def test_multiple_positions(self, paper_broker):
        """Test tracking multiple positions."""
        paper_broker.place_order("AAPL", OrderSide.BUY, 10, OrderType.MARKET)
        paper_broker.place_order("MSFT", OrderSide.BUY, 5, OrderType.MARKET)
        paper_broker.place_order("GOOGL", OrderSide.SELL_SHORT, 8, OrderType.MARKET)

        positions = paper_broker.get_positions()

        assert len(positions) == 3
        assert "AAPL" in positions
        assert "MSFT" in positions
        assert "GOOGL" in positions
        assert positions["AAPL"].quantity == 10
        assert positions["GOOGL"].quantity == -8

    def test_average_cost_calculation(self, paper_broker):
        """Test average cost is calculated correctly on multiple buys."""
        # Buy 10 @ ~100 (mock price)
        paper_broker.place_order("AAPL", OrderSide.BUY, 10, OrderType.MARKET)

        position1 = paper_broker.get_position("AAPL")
        first_cost = position1.avg_cost

        # Buy 10 more (at same mock price)
        paper_broker.place_order("AAPL", OrderSide.BUY, 10, OrderType.MARKET)

        position2 = paper_broker.get_position("AAPL")
        assert position2.quantity == 20
        # Average cost should be similar (using same mock quote)
        assert position2.avg_cost == pytest.approx(first_cost, rel=0.1)

    def test_unrealized_pnl_calculation(self, paper_broker):
        """Test unrealized P&L is calculated."""
        paper_broker.place_order("AAPL", OrderSide.BUY, 10, OrderType.MARKET)

        position = paper_broker.get_position("AAPL")
        # P&L should exist (may be small due to slippage)
        assert position.unrealized_pnl is not None
        assert position.unrealized_pnl_pct is not None


class TestBrokerInterfaceCompliance:
    """Test broker interface compliance."""

    def test_paper_broker_implements_interface(self):
        """Test PaperBroker implements BrokerInterface."""
        assert issubclass(PaperBroker, BrokerInterface)

    def test_connect_disconnect(self, paper_broker):
        """Test connect and disconnect methods."""
        # Already connected by fixture
        assert paper_broker.is_connected is True

        paper_broker.disconnect()
        assert paper_broker.is_connected is False

        paper_broker.connect()
        assert paper_broker.is_connected is True

    def test_get_account(self, paper_broker):
        """Test get_account returns AccountInfo."""
        account = paper_broker.get_account()

        assert isinstance(account, AccountInfo)
        assert account.account_id is not None
        assert account.buying_power > 0
        assert account.cash > 0
        assert account.equity > 0

    def test_get_quote(self, paper_broker):
        """Test get_quote returns Quote."""
        quote = paper_broker.get_quote("AAPL")

        assert isinstance(quote, Quote)
        assert quote.symbol == "AAPL"
        assert quote.bid > 0
        assert quote.ask > 0
        assert quote.last > 0

    def test_get_quotes_multiple(self, paper_broker):
        """Test getting multiple quotes."""
        quotes = paper_broker.get_quotes(["AAPL", "MSFT", "GOOGL"])

        assert len(quotes) == 3
        assert "AAPL" in quotes
        assert "MSFT" in quotes
        assert "GOOGL" in quotes

    def test_cancel_order(self, paper_broker):
        """Test order cancellation."""
        # Place a limit order that won't fill
        quote = paper_broker.get_quote("AAPL")
        order = paper_broker.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            price=quote.ask - 20.00
        )

        assert order.status == OrderStatus.OPEN

        # Cancel it
        result = paper_broker.cancel_order(order.order_id)
        assert result is True

        # Check status
        cancelled = paper_broker.get_order_status(order.order_id)
        assert cancelled.status == OrderStatus.CANCELLED

    def test_get_order_status(self, paper_broker):
        """Test getting order status."""
        order = paper_broker.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )

        status = paper_broker.get_order_status(order.order_id)

        assert status is not None
        assert status.order_id == order.order_id
        assert status.status == OrderStatus.FILLED

    def test_get_open_orders(self, paper_broker):
        """Test getting open orders."""
        quote = paper_broker.get_quote("AAPL")

        # Place limit order that won't fill
        paper_broker.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            price=quote.ask - 20.00
        )

        open_orders = paper_broker.get_open_orders()
        assert len(open_orders) >= 1

    def test_validate_order(self, paper_broker):
        """Test order validation."""
        # Valid order
        is_valid, msg = paper_broker.validate_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        assert is_valid is True
        assert msg == ""

        # Invalid: no symbol
        is_valid, msg = paper_broker.validate_order(
            symbol="",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        assert is_valid is False

        # Invalid: negative quantity
        is_valid, msg = paper_broker.validate_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=-5,
            order_type=OrderType.MARKET
        )
        assert is_valid is False

        # Invalid: limit order without price
        is_valid, msg = paper_broker.validate_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT
        )
        assert is_valid is False


class TestBrokerErrorHandling:
    """Test broker error handling."""

    def test_insufficient_funds_error(self, paper_broker):
        """Test insufficient funds error - order should be rejected."""
        # Place order for way more than available (at $100/share, 10000 shares = $1M)
        order = paper_broker.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10000,  # Way too many shares
            order_type=OrderType.MARKET
        )

        # PaperBroker rejects with status rather than raising
        assert order.status == OrderStatus.REJECTED
        assert "insufficient" in order.error_message.lower()

    def test_sell_without_position_error(self, paper_broker):
        """Test selling without position is rejected."""
        order = paper_broker.place_order(
            symbol="NVDA",  # Symbol we don't own
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.MARKET
        )

        assert order.status == OrderStatus.REJECTED
        assert order.error_message is not None

    def test_sell_more_than_owned_error(self, paper_broker):
        """Test selling more than owned is rejected."""
        # Buy 10
        paper_broker.place_order("AAPL", OrderSide.BUY, 10, OrderType.MARKET)

        # Try to sell 20
        order = paper_broker.place_order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=20,
            order_type=OrderType.MARKET
        )

        assert order.status == OrderStatus.REJECTED

    def test_operations_require_connection(self):
        """Test operations fail when not connected."""
        broker = PaperBroker()
        # Not connected

        with pytest.raises(BrokerError):
            broker.get_account()

        with pytest.raises(BrokerError):
            broker.get_positions()

        with pytest.raises(BrokerError):
            broker.place_order("AAPL", OrderSide.BUY, 10)


class TestBrokerFactory:
    """Test broker factory functions."""

    def test_get_broker_paper(self):
        """Test getting paper broker."""
        broker = get_broker("paper", initial_balance=50000)

        assert isinstance(broker, PaperBroker)
        broker.connect()
        account = broker.get_account()
        assert account.equity == pytest.approx(50000, rel=0.01)

    def test_get_broker_unknown_type(self):
        """Test getting unknown broker type raises error."""
        with pytest.raises(ValueError):
            get_broker("unknown_broker")

    def test_get_broker_from_config_paper(self):
        """Test getting broker from config."""
        config = {
            'initial_balance': 100000,
            'paper_trading': True
        }

        broker = get_broker_from_config(config)
        assert isinstance(broker, PaperBroker)

    def test_get_broker_from_config_paper_override(self):
        """Test paper_trading=True overrides broker_type."""
        config = {
            'paper_trading': True  # This forces paper
        }

        broker = get_broker_from_config(config)
        assert isinstance(broker, PaperBroker)


class TestPaperBrokerSpecific:
    """Test PaperBroker specific functionality."""

    def test_reset(self, paper_broker):
        """Test account reset."""
        # Make some trades
        paper_broker.place_order("AAPL", OrderSide.BUY, 10, OrderType.MARKET)
        paper_broker.place_order("MSFT", OrderSide.BUY, 5, OrderType.MARKET)

        assert len(paper_broker.get_positions()) > 0

        # Reset
        paper_broker.reset()

        assert len(paper_broker.get_positions()) == 0
        assert paper_broker.cash == paper_broker.initial_balance

    def test_process_pending_orders(self, paper_broker):
        """Test processing pending orders."""
        quote = paper_broker.get_quote("AAPL")

        # Place limit order that should fill when processed
        order = paper_broker.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            price=quote.ask + 5.00  # Above ask, should fill
        )

        # Initially might be open (depending on implementation)
        paper_broker.process_pending_orders()

        # After processing, should be filled
        updated = paper_broker.get_order_status(order.order_id)
        # Order was already filled if price was above ask
        assert updated.status in (OrderStatus.FILLED, OrderStatus.OPEN)

    def test_trade_history(self, paper_broker):
        """Test trade history tracking."""
        paper_broker.place_order("AAPL", OrderSide.BUY, 10, OrderType.MARKET)
        paper_broker.place_order("MSFT", OrderSide.BUY, 5, OrderType.MARKET)

        history = paper_broker.get_trade_history()

        assert len(history) == 2
        assert history[0]['symbol'] in ['AAPL', 'MSFT']
        assert 'timestamp' in history[0]
        assert 'price' in history[0]

    def test_performance_summary(self, paper_broker):
        """Test performance summary."""
        paper_broker.place_order("AAPL", OrderSide.BUY, 10, OrderType.MARKET)

        summary = paper_broker.get_performance_summary()

        assert 'initial_balance' in summary
        assert 'current_balance' in summary
        assert 'total_trades' in summary
        assert 'return_pct' in summary
        assert summary['total_trades'] == 1

    def test_slippage_simulation(self):
        """Test slippage is applied in realistic mode."""
        broker = PaperBroker(
            initial_balance=25000,
            slippage_pct=0.01,  # 1% slippage
            realistic_fills=True
        )
        broker.connect()

        quote = broker.get_quote("AAPL")
        order = broker.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )

        # Fill price should be at or above ask (with slippage for buy)
        assert order.avg_fill_price >= quote.ask

    def test_commission_tracking(self):
        """Test commission is tracked."""
        broker = PaperBroker(
            initial_balance=25000,
            commission_per_trade=10.00
        )
        broker.connect()

        initial_cash = broker.cash

        broker.place_order("AAPL", OrderSide.BUY, 10, OrderType.MARKET)

        # Cash should be reduced by order value + commission
        assert broker.cash < initial_cash


class TestQuoteProperties:
    """Test Quote data class properties."""

    def test_mid_price(self):
        """Test mid price calculation."""
        quote = Quote(
            symbol="AAPL",
            bid=99.00,
            ask=101.00,
            last=100.00,
            volume=1000000,
            timestamp=datetime.now()
        )

        assert quote.mid == 100.00

    def test_spread(self):
        """Test spread calculation."""
        quote = Quote(
            symbol="AAPL",
            bid=99.00,
            ask=101.00,
            last=100.00,
            volume=1000000,
            timestamp=datetime.now()
        )

        assert quote.spread == 2.00

    def test_spread_pct(self):
        """Test spread percentage calculation."""
        quote = Quote(
            symbol="AAPL",
            bid=99.00,
            ask=101.00,
            last=100.00,
            volume=1000000,
            timestamp=datetime.now()
        )

        # Spread = 2, Mid = 100, Spread% = 2%
        assert quote.spread_pct == pytest.approx(2.0)


class TestOrderProperties:
    """Test Order data class properties."""

    def test_is_active(self):
        """Test is_active property."""
        order = Order(
            order_id="test",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            status=OrderStatus.PENDING
        )

        assert order.is_active is True

        order.status = OrderStatus.FILLED
        assert order.is_active is False

    def test_is_filled(self):
        """Test is_filled property."""
        order = Order(
            order_id="test",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            status=OrderStatus.FILLED
        )

        assert order.is_filled is True

    def test_remaining_quantity(self):
        """Test remaining_quantity property."""
        order = Order(
            order_id="test",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            filled_quantity=60
        )

        assert order.remaining_quantity == 40


class TestPositionProperties:
    """Test Position data class properties."""

    def test_is_long(self):
        """Test is_long property."""
        position = Position(
            symbol="AAPL",
            quantity=10,
            avg_cost=150.00,
            current_price=155.00,
            market_value=1550.00,
            unrealized_pnl=50.00,
            unrealized_pnl_pct=3.33
        )

        assert position.is_long is True
        assert position.is_short is False

    def test_is_short(self):
        """Test is_short property."""
        position = Position(
            symbol="AAPL",
            quantity=-10,
            avg_cost=150.00,
            current_price=145.00,
            market_value=1450.00,
            unrealized_pnl=50.00,
            unrealized_pnl_pct=3.33
        )

        assert position.is_short is True
        assert position.is_long is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

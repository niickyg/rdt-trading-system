"""
Shared pytest fixtures for RDT Trading System tests.
Provides mock data, fixtures, and test utilities.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any
from zoneinfo import ZoneInfo

# Eastern timezone for market hours
EASTERN_TZ = ZoneInfo("America/New_York")


# =============================================================================
# Stock Data Fixtures
# =============================================================================

@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """
    Create sample OHLCV DataFrame with realistic price data.
    60 days of daily data suitable for ATR and RRS calculations.
    """
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')

    # Generate realistic price movements
    base_price = 150.0
    returns = np.random.normal(0.001, 0.02, 60)
    prices = base_price * np.cumprod(1 + returns)

    data = {
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, 60)),
        'high': prices * (1 + np.random.uniform(0.005, 0.02, 60)),
        'low': prices * (1 - np.random.uniform(0.005, 0.02, 60)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 60),
    }

    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def sample_5m_data() -> pd.DataFrame:
    """
    Create sample 5-minute intraday OHLCV data.
    One trading day of 5-minute bars (78 bars from 9:30 to 16:00).
    """
    np.random.seed(42)
    today = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    dates = pd.date_range(start=today, periods=78, freq='5min')

    base_price = 150.0
    returns = np.random.normal(0.0001, 0.003, 78)
    prices = base_price * np.cumprod(1 + returns)

    data = {
        'open': prices * (1 + np.random.uniform(-0.002, 0.002, 78)),
        'high': prices * (1 + np.random.uniform(0.001, 0.004, 78)),
        'low': prices * (1 - np.random.uniform(0.001, 0.004, 78)),
        'close': prices,
        'volume': np.random.randint(10000, 100000, 78),
    }

    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def strong_rs_daily_data() -> pd.DataFrame:
    """
    Create daily data showing strong relative strength pattern.
    3 consecutive green days with EMA3 > EMA8.
    """
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')

    # Uptrending price data
    base_price = 100.0
    prices = np.linspace(base_price, base_price * 1.15, 60)  # 15% uptrend

    # Ensure last 3 days are green (close > open)
    opens = prices - 1.0
    closes = prices.copy()

    data = {
        'open': opens,
        'high': prices + 0.5,
        'low': opens - 0.3,
        'close': closes,
        'volume': np.random.randint(1000000, 5000000, 60),
    }

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def weak_rw_daily_data() -> pd.DataFrame:
    """
    Create daily data showing relative weakness pattern.
    3 consecutive red days with EMA8 > EMA3.
    """
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')

    # Downtrending price data
    base_price = 100.0
    prices = np.linspace(base_price, base_price * 0.85, 60)  # 15% downtrend

    # Ensure last 3 days are red (close < open)
    opens = prices + 1.0
    closes = prices.copy()

    data = {
        'open': opens,
        'high': opens + 0.3,
        'low': prices - 0.5,
        'close': closes,
        'volume': np.random.randint(1000000, 5000000, 60),
    }

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def spy_data_fixture(sample_ohlcv_data, sample_5m_data) -> Dict:
    """
    Create SPY data fixture with both daily and 5-minute data.
    """
    return {
        '5m': sample_5m_data.copy(),
        'daily': sample_ohlcv_data.copy(),
        'current_price': 450.0,
        'previous_close': 448.0,
    }


@pytest.fixture
def stock_data_fixture(sample_ohlcv_data, sample_5m_data) -> Dict:
    """
    Create individual stock data fixture.
    """
    return {
        '5m': sample_5m_data.copy(),
        'daily': sample_ohlcv_data.copy(),
        'current_price': 155.0,
        'previous_close': 150.0,
        'atr': 3.5,
        'volume': 5000000,
    }


# =============================================================================
# Quote and Historical Data Fixtures (for providers)
# =============================================================================

@pytest.fixture
def mock_quote():
    """Create a mock Quote object."""
    from data.providers.base import Quote

    return Quote(
        symbol="AAPL",
        price=175.50,
        open=174.00,
        high=176.25,
        low=173.50,
        volume=50000000,
        previous_close=173.25,
        change=2.25,
        change_percent=1.30,
        timestamp=datetime.now(),
        provider="test_provider",
    )


@pytest.fixture
def mock_historical_data(sample_ohlcv_data):
    """Create a mock HistoricalData object."""
    from data.providers.base import HistoricalData

    return HistoricalData(
        symbol="AAPL",
        data=sample_ohlcv_data.copy(),
        period="60d",
        interval="1d",
        provider="test_provider",
    )


# =============================================================================
# Batch Data Fixtures
# =============================================================================

@pytest.fixture
def mock_batch_5m_data() -> pd.DataFrame:
    """
    Create mock batch 5-minute data in yfinance download format.
    Multi-level columns: (ticker, OHLCV).
    """
    np.random.seed(42)
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY']
    today = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    dates = pd.date_range(start=today, periods=78, freq='5min')

    data_dict = {}
    for symbol in symbols:
        base_price = {'AAPL': 175, 'MSFT': 380, 'GOOGL': 140, 'SPY': 450}[symbol]
        returns = np.random.normal(0.0001, 0.003, 78)
        prices = base_price * np.cumprod(1 + returns)

        data_dict[(symbol, 'Open')] = prices * (1 + np.random.uniform(-0.002, 0.002, 78))
        data_dict[(symbol, 'High')] = prices * (1 + np.random.uniform(0.001, 0.004, 78))
        data_dict[(symbol, 'Low')] = prices * (1 - np.random.uniform(0.001, 0.004, 78))
        data_dict[(symbol, 'Close')] = prices
        data_dict[(symbol, 'Volume')] = np.random.randint(10000, 100000, 78)

    df = pd.DataFrame(data_dict, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


@pytest.fixture
def mock_batch_daily_data() -> pd.DataFrame:
    """
    Create mock batch daily data in yfinance download format.
    Multi-level columns: (ticker, OHLCV).
    """
    np.random.seed(42)
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY']
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')

    data_dict = {}
    for symbol in symbols:
        base_price = {'AAPL': 170, 'MSFT': 375, 'GOOGL': 135, 'SPY': 445}[symbol]
        returns = np.random.normal(0.001, 0.02, 60)
        prices = base_price * np.cumprod(1 + returns)

        data_dict[(symbol, 'Open')] = prices * (1 + np.random.uniform(-0.01, 0.01, 60))
        data_dict[(symbol, 'High')] = prices * (1 + np.random.uniform(0.005, 0.02, 60))
        data_dict[(symbol, 'Low')] = prices * (1 - np.random.uniform(0.005, 0.02, 60))
        data_dict[(symbol, 'Close')] = prices
        data_dict[(symbol, 'Volume')] = np.random.randint(1000000, 10000000, 60)

    df = pd.DataFrame(data_dict, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


# =============================================================================
# Scanner Configuration Fixtures
# =============================================================================

@pytest.fixture
def scanner_config() -> Dict:
    """Standard scanner configuration for testing."""
    return {
        'atr_period': 14,
        'rrs_strong_threshold': 2.0,
        'rrs_weak_threshold': -2.0,
        'scan_interval_seconds': 60,
        'min_volume': 500000,
        'min_price': 5.0,
        'max_price': 500.0,
        'alert_method': 'desktop',
        'use_data_providers': False,  # Disable providers for basic tests
    }


@pytest.fixture
def scanner_config_with_providers() -> Dict:
    """Scanner configuration with providers enabled."""
    return {
        'atr_period': 14,
        'rrs_strong_threshold': 2.0,
        'rrs_weak_threshold': -2.0,
        'scan_interval_seconds': 60,
        'min_volume': 500000,
        'min_price': 5.0,
        'max_price': 500.0,
        'alert_method': 'desktop',
        'use_data_providers': True,
    }


# =============================================================================
# Mock Provider Fixtures
# =============================================================================

@pytest.fixture
def mock_yfinance_provider():
    """Create a mock YFinance provider."""
    from data.providers.base import ProviderStatus

    provider = Mock()
    provider.name = "yfinance"
    provider.priority = 10
    provider.is_available.return_value = True
    provider.get_health.return_value = Mock(
        status=ProviderStatus.HEALTHY,
        consecutive_failures=0,
        requests_today=10,
        requests_limit=None,
    )
    return provider


@pytest.fixture
def mock_alpha_vantage_provider():
    """Create a mock Alpha Vantage provider."""
    from data.providers.base import ProviderStatus

    provider = Mock()
    provider.name = "alpha_vantage"
    provider.priority = 20
    provider.is_available.return_value = True
    provider.get_health.return_value = Mock(
        status=ProviderStatus.HEALTHY,
        consecutive_failures=0,
        requests_today=5,
        requests_limit=25,
    )
    return provider


@pytest.fixture
def mock_failing_provider():
    """Create a mock provider that always fails."""
    from data.providers.base import ProviderError

    provider = Mock()
    provider.name = "failing_provider"
    provider.priority = 5
    provider.is_available.return_value = False
    provider.get_quote.side_effect = ProviderError("Provider unavailable")
    provider.get_historical.side_effect = ProviderError("Provider unavailable")
    return provider


# =============================================================================
# Market Hours Fixtures
# =============================================================================

@pytest.fixture
def market_open_time() -> datetime:
    """Return a datetime during market hours (10:30 AM ET on a weekday)."""
    now = datetime.now(EASTERN_TZ)
    # Find next Monday
    days_until_monday = (7 - now.weekday()) % 7
    if days_until_monday == 0 and now.weekday() >= 5:
        days_until_monday = 7 - now.weekday()

    market_day = now + timedelta(days=days_until_monday)
    return market_day.replace(hour=10, minute=30, second=0, microsecond=0)


@pytest.fixture
def market_closed_time() -> datetime:
    """Return a datetime outside market hours (8:00 PM ET)."""
    now = datetime.now(EASTERN_TZ)
    return now.replace(hour=20, minute=0, second=0, microsecond=0)


@pytest.fixture
def weekend_time() -> datetime:
    """Return a datetime on Saturday."""
    now = datetime.now(EASTERN_TZ)
    days_until_saturday = (5 - now.weekday()) % 7
    saturday = now + timedelta(days=days_until_saturday)
    return saturday.replace(hour=12, minute=0, second=0, microsecond=0)


# =============================================================================
# File System Fixtures
# =============================================================================

@pytest.fixture
def temp_signals_dir(tmp_path):
    """Create a temporary signals directory."""
    signals_dir = tmp_path / "data" / "signals"
    signals_dir.mkdir(parents=True)
    return signals_dir


@pytest.fixture
def mock_signals_file(temp_signals_dir):
    """Create a mock signals file with sample data."""
    import json

    signals = [
        {
            'symbol': 'AAPL',
            'direction': 'long',
            'strength': 'strong',
            'rrs': 2.85,
            'entry_price': 175.50,
            'stop_price': 172.00,
            'target_price': 182.50,
            'atr': 3.50,
            'stock_change_pct': 1.25,
            'spy_change_pct': 0.45,
            'daily_strong': True,
            'generated_at': datetime.now().isoformat(),
            'strategy': 'RRS_Momentum',
        }
    ]

    signals_file = temp_signals_dir / "active_signals.json"
    with open(signals_file, 'w') as f:
        json.dump(signals, f)

    return signals_file


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_provider_manager():
    """Reset global provider manager after each test."""
    yield
    try:
        from data.providers.provider_manager import reset_provider_manager
        reset_provider_manager()
    except ImportError:
        pass


# =============================================================================
# WebSocket Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_websocket_broadcast():
    """Mock WebSocket broadcast functions."""
    with patch.multiple(
        'scanner.realtime_scanner',
        WEBSOCKET_AVAILABLE=True,
        broadcast_signal=Mock(),
        broadcast_scan_started=Mock(),
        broadcast_scan_progress=Mock(),
        broadcast_scan_completed=Mock(),
        broadcast_scan_error=Mock(),
    ) as mocks:
        yield mocks


# =============================================================================
# Utility Functions
# =============================================================================

def create_mock_ticker(symbol: str, price: float = 150.0):
    """Create a mock yfinance Ticker object."""
    mock_ticker = Mock()
    mock_ticker.info = {'symbol': symbol, 'regularMarketPrice': price}

    # Create mock history DataFrame
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
    np.random.seed(hash(symbol) % 2**32)
    returns = np.random.normal(0.001, 0.02, 60)
    prices = price * np.cumprod(1 + returns)

    history_df = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, 60)),
        'High': prices * (1 + np.random.uniform(0.005, 0.02, 60)),
        'Low': prices * (1 - np.random.uniform(0.005, 0.02, 60)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 60),
    }, index=dates)

    mock_ticker.history.return_value = history_df
    return mock_ticker


# Export utility function
@pytest.fixture
def create_ticker():
    """Fixture that returns the create_mock_ticker function."""
    return create_mock_ticker


# =============================================================================
# Broker and Trading Fixtures
# =============================================================================

# Import broker types
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from brokers.broker_interface import (
        BrokerInterface, Quote as BrokerQuote, Order, Position, AccountInfo,
        OrderSide, OrderType, OrderStatus,
        BrokerError, OrderError, InsufficientFundsError
    )
    from brokers.paper_broker import PaperBroker
    BROKER_IMPORTS_AVAILABLE = True
except ImportError:
    BROKER_IMPORTS_AVAILABLE = False


class MockBroker(BrokerInterface):
    """
    Mock broker for testing that allows full control over behavior.

    Supports configurable responses for:
    - Connection state
    - Account information
    - Quotes
    - Order execution
    - Positions
    """

    def __init__(
        self,
        initial_balance: float = 25000.0,
        fail_connection: bool = False,
        fail_orders: bool = False,
        order_fill_delay: bool = False,
        reject_orders: bool = False,
        insufficient_funds: bool = False,
    ):
        self._connected = False
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.fail_connection = fail_connection
        self.fail_orders = fail_orders
        self.order_fill_delay = order_fill_delay
        self.reject_orders = reject_orders
        self.insufficient_funds = insufficient_funds

        self._positions: Dict[str, dict] = {}
        self._orders: Dict[str, Order] = {}
        self._quotes: Dict[str, BrokerQuote] = {}
        self._order_counter = 0

        # Track method calls for assertions
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.place_order_calls = []
        self.cancel_order_calls = []

    def connect(self) -> bool:
        self.connect_calls += 1
        if self.fail_connection:
            raise BrokerError("Mock connection failure")
        self._connected = True
        return True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_account(self) -> AccountInfo:
        if not self._connected:
            raise BrokerError("Not connected")

        positions_value = sum(
            p['quantity'] * p['current_price']
            for p in self._positions.values()
        )

        return AccountInfo(
            account_id="MOCK_ACCOUNT",
            buying_power=self.cash,
            cash=self.cash,
            equity=self.cash + positions_value,
            day_trades_remaining=3,
            pattern_day_trader=False,
            positions_value=positions_value,
            daily_pnl=0.0
        )

    def get_positions(self) -> Dict[str, Position]:
        if not self._connected:
            raise BrokerError("Not connected")

        result = {}
        for symbol, pos in self._positions.items():
            result[symbol] = Position(
                symbol=symbol,
                quantity=pos['quantity'],
                avg_cost=pos['avg_cost'],
                current_price=pos['current_price'],
                market_value=pos['quantity'] * pos['current_price'],
                unrealized_pnl=(pos['current_price'] - pos['avg_cost']) * pos['quantity'],
                unrealized_pnl_pct=((pos['current_price'] / pos['avg_cost']) - 1) * 100 if pos['avg_cost'] > 0 else 0,
                cost_basis=pos['quantity'] * pos['avg_cost']
            )
        return result

    def get_position(self, symbol: str):
        positions = self.get_positions()
        return positions.get(symbol.upper())

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float = None,
        stop_price: float = None,
        time_in_force: str = "DAY"
    ) -> Order:
        if not self._connected:
            raise BrokerError("Not connected")

        self.place_order_calls.append({
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'order_type': order_type,
            'price': price,
            'stop_price': stop_price
        })

        if self.fail_orders:
            raise OrderError("Mock order failure")

        if self.insufficient_funds:
            raise InsufficientFundsError("Insufficient funds for order")

        self._order_counter += 1
        order_id = f"mock_order_{self._order_counter}"

        # Get quote for fill price
        quote = self.get_quote(symbol)
        fill_price = quote.ask if side in (OrderSide.BUY, OrderSide.BUY_TO_COVER) else quote.bid

        if self.reject_orders:
            order = Order(
                order_id=order_id,
                symbol=symbol.upper(),
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                stop_price=stop_price,
                status=OrderStatus.REJECTED,
                error_message="Order rejected by mock broker"
            )
        elif self.order_fill_delay:
            order = Order(
                order_id=order_id,
                symbol=symbol.upper(),
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                stop_price=stop_price,
                status=OrderStatus.PENDING
            )
        else:
            order = Order(
                order_id=order_id,
                symbol=symbol.upper(),
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                stop_price=stop_price,
                status=OrderStatus.FILLED,
                filled_quantity=quantity,
                avg_fill_price=fill_price,
                filled_at=datetime.now()
            )

            # Update positions
            self._update_position_from_order(symbol.upper(), side, quantity, fill_price)

        self._orders[order_id] = order
        return order

    def _update_position_from_order(self, symbol: str, side: OrderSide, quantity: int, price: float):
        """Update positions based on filled order."""
        if side == OrderSide.BUY:
            if symbol in self._positions:
                pos = self._positions[symbol]
                total_qty = pos['quantity'] + quantity
                pos['avg_cost'] = (pos['avg_cost'] * pos['quantity'] + price * quantity) / total_qty
                pos['quantity'] = total_qty
            else:
                self._positions[symbol] = {
                    'quantity': quantity,
                    'avg_cost': price,
                    'current_price': price
                }
            self.cash -= price * quantity
        elif side == OrderSide.SELL:
            if symbol in self._positions:
                pos = self._positions[symbol]
                pos['quantity'] -= quantity
                if pos['quantity'] <= 0:
                    del self._positions[symbol]
                self.cash += price * quantity

    def cancel_order(self, order_id: str) -> bool:
        self.cancel_order_calls.append(order_id)
        if order_id in self._orders:
            order = self._orders[order_id]
            if order.status in (OrderStatus.PENDING, OrderStatus.OPEN):
                order.status = OrderStatus.CANCELLED
                return True
        return False

    def get_order_status(self, order_id: str):
        return self._orders.get(order_id)

    def get_quote(self, symbol: str) -> BrokerQuote:
        symbol = symbol.upper()
        if symbol in self._quotes:
            return self._quotes[symbol]

        # Default quote
        return BrokerQuote(
            symbol=symbol,
            bid=99.95,
            ask=100.05,
            last=100.00,
            volume=1000000,
            timestamp=datetime.now(),
            high=101.00,
            low=99.00,
            open=100.00,
            prev_close=99.50
        )

    def set_quote(self, symbol: str, bid: float, ask: float, last: float):
        """Set a custom quote for testing."""
        self._quotes[symbol.upper()] = BrokerQuote(
            symbol=symbol.upper(),
            bid=bid,
            ask=ask,
            last=last,
            volume=1000000,
            timestamp=datetime.now()
        )

    def set_position(self, symbol: str, quantity: int, avg_cost: float, current_price: float = None):
        """Set a position for testing."""
        self._positions[symbol.upper()] = {
            'quantity': quantity,
            'avg_cost': avg_cost,
            'current_price': current_price or avg_cost
        }

    def simulate_fill(self, order_id: str, fill_price: float, fill_quantity: int = None):
        """Simulate a fill for a pending order."""
        if order_id in self._orders:
            order = self._orders[order_id]
            fill_qty = fill_quantity or order.quantity
            order.filled_quantity = fill_qty
            order.avg_fill_price = fill_price
            order.status = OrderStatus.FILLED if fill_qty >= order.quantity else OrderStatus.PARTIALLY_FILLED
            order.filled_at = datetime.now()
            self._update_position_from_order(order.symbol, order.side, fill_qty, fill_price)
            return order
        return None


@pytest.fixture
def mock_broker():
    """Create a fresh mock broker for each test."""
    if not BROKER_IMPORTS_AVAILABLE:
        pytest.skip("Broker imports not available")
    return MockBroker(initial_balance=25000.0)


@pytest.fixture
def mock_broker_connected(mock_broker):
    """Create a connected mock broker."""
    mock_broker.connect()
    return mock_broker


@pytest.fixture
def mock_broker_failing():
    """Create a mock broker that fails to connect."""
    if not BROKER_IMPORTS_AVAILABLE:
        pytest.skip("Broker imports not available")
    return MockBroker(fail_connection=True)


@pytest.fixture
def mock_broker_insufficient_funds():
    """Create a mock broker that rejects orders due to insufficient funds."""
    if not BROKER_IMPORTS_AVAILABLE:
        pytest.skip("Broker imports not available")
    broker = MockBroker(insufficient_funds=True)
    broker.connect()
    return broker


@pytest.fixture
def mock_broker_rejecting():
    """Create a mock broker that rejects all orders."""
    if not BROKER_IMPORTS_AVAILABLE:
        pytest.skip("Broker imports not available")
    broker = MockBroker(reject_orders=True)
    broker.connect()
    return broker


@pytest.fixture
def mock_broker_delayed_fills():
    """Create a mock broker with delayed order fills."""
    if not BROKER_IMPORTS_AVAILABLE:
        pytest.skip("Broker imports not available")
    broker = MockBroker(order_fill_delay=True)
    broker.connect()
    return broker


@pytest.fixture
def paper_broker():
    """Create a paper broker for testing."""
    if not BROKER_IMPORTS_AVAILABLE:
        pytest.skip("Broker imports not available")
    broker = PaperBroker(
        initial_balance=25000.0,
        slippage_pct=0.001,
        commission_per_trade=0.0,
        realistic_fills=False  # Disable random slippage for deterministic tests
    )
    broker.connect()
    return broker


# =============================================================================
# Sample Trading Signal Fixtures
# =============================================================================

@pytest.fixture
def sample_long_signal():
    """Sample RRS long signal."""
    return {
        'symbol': 'AAPL',
        'price': 150.00,
        'atr': 3.00,
        'rrs': 2.5,
        'daily_strong': True,
        'daily_weak': False,
        'volume': 50000000,
        'timestamp': datetime.now()
    }


@pytest.fixture
def sample_short_signal():
    """Sample RRS short signal."""
    return {
        'symbol': 'TSLA',
        'price': 200.00,
        'atr': 8.00,
        'rrs': -3.0,
        'daily_strong': False,
        'daily_weak': True,
        'volume': 80000000,
        'timestamp': datetime.now()
    }


@pytest.fixture
def sample_weak_signal():
    """Sample signal that doesn't meet entry criteria."""
    return {
        'symbol': 'MSFT',
        'price': 300.00,
        'atr': 5.00,
        'rrs': 0.5,  # Below threshold
        'daily_strong': True,
        'daily_weak': False,
        'volume': 30000000,
        'timestamp': datetime.now()
    }


@pytest.fixture
def sample_signals_batch():
    """Batch of sample signals."""
    return [
        {
            'symbol': 'AAPL',
            'price': 150.00,
            'atr': 3.00,
            'rrs': 2.5,
            'daily_strong': True,
            'daily_weak': False,
        },
        {
            'symbol': 'GOOGL',
            'price': 140.00,
            'atr': 4.00,
            'rrs': 3.0,
            'daily_strong': True,
            'daily_weak': False,
        },
        {
            'symbol': 'TSLA',
            'price': 200.00,
            'atr': 8.00,
            'rrs': -2.5,
            'daily_strong': False,
            'daily_weak': True,
        }
    ]


# =============================================================================
# Sample Position Fixtures
# =============================================================================

@pytest.fixture
def sample_long_position():
    """Sample long position."""
    return {
        'symbol': 'AAPL',
        'direction': 'long',
        'entry_price': 150.00,
        'shares': 50,
        'stop_loss': 145.50,
        'take_profit': 159.00,
        'entry_time': datetime.now() - timedelta(hours=2),
        'rrs': 2.5,
        'executed': True
    }


@pytest.fixture
def sample_short_position():
    """Sample short position."""
    return {
        'symbol': 'TSLA',
        'direction': 'short',
        'entry_price': 200.00,
        'shares': 25,
        'stop_loss': 212.00,
        'take_profit': 176.00,
        'entry_time': datetime.now() - timedelta(hours=1),
        'rrs': -3.0,
        'executed': True
    }


@pytest.fixture
def sample_positions_dict():
    """Dictionary of sample positions."""
    return {
        'AAPL': {
            'direction': 'long',
            'entry_price': 150.00,
            'shares': 50,
            'stop_loss': 145.50,
            'take_profit': 159.00,
            'entry_time': datetime.now() - timedelta(hours=2),
            'rrs': 2.5,
            'executed': True
        },
        'TSLA': {
            'direction': 'short',
            'entry_price': 200.00,
            'shares': 25,
            'stop_loss': 212.00,
            'take_profit': 176.00,
            'entry_time': datetime.now() - timedelta(hours=1),
            'rrs': -3.0,
            'executed': True
        }
    }


# =============================================================================
# Trading Bot Configuration Fixtures
# =============================================================================

@pytest.fixture
def default_bot_config():
    """Default trading bot configuration."""
    return {
        'account_size': 25000,
        'max_risk_per_trade': 0.01,  # 1%
        'max_daily_loss': 0.03,  # 3%
        'max_position_size': 0.10,  # 10%
        'paper_trading': True,
        'auto_trade': False,
        'broker_type': 'paper',
        'rrs_strong_threshold': 2.0,
        'stop_atr_multiplier': 1.5,
        'target_atr_multiplier': 3.0,
        'scan_interval_seconds': 60,
        'high_slippage_threshold_pct': 0.5,
        'stuck_order_timeout_seconds': 60,
        'fill_confirmation_timeout_seconds': 30
    }


@pytest.fixture
def auto_trade_config(default_bot_config):
    """Configuration with auto-trade enabled."""
    config = default_bot_config.copy()
    config['auto_trade'] = True
    return config


@pytest.fixture
def live_trading_config(default_bot_config):
    """Configuration for live trading (paper_trading=False)."""
    config = default_bot_config.copy()
    config['paper_trading'] = False
    config['broker_type'] = 'paper'  # Still use paper broker for testing
    return config


# =============================================================================
# Order Monitor Fixtures
# =============================================================================

@pytest.fixture
def sample_order_data():
    """Sample order data for OrderMonitor tests."""
    return {
        'order_id': 'test_order_001',
        'symbol': 'AAPL',
        'side': 'buy',
        'quantity': 100,
        'order_type': 'market',
        'expected_price': 150.00,
        'limit_price': None,
        'stop_price': None
    }


@pytest.fixture
def sample_fill_data():
    """Sample fill data for OrderMonitor tests."""
    return {
        'fill_price': 150.05,
        'fill_quantity': 100,
        'timestamp': datetime.now()
    }


# =============================================================================
# Mock Scanner and RRS Calculator
# =============================================================================

@pytest.fixture
def mock_scanner():
    """Mock scanner that returns predefined signals."""
    scanner = Mock()
    scanner.scan_once = Mock(return_value=[])
    scanner.get_signals = Mock(return_value=[])
    return scanner


@pytest.fixture
def mock_rrs_calculator():
    """Mock RRS calculator."""
    calc = Mock()
    calc.calculate = Mock(return_value={
        'rrs': 2.5,
        'atr': 3.00,
        'strength': 'strong'
    })
    return calc


# =============================================================================
# Capture Logs Fixture
# =============================================================================

@pytest.fixture
def capture_logs(caplog):
    """Capture log messages for assertions."""
    import logging
    caplog.set_level(logging.DEBUG)
    return caplog

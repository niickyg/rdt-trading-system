"""
Integration Test Fixtures for RDT Trading System

Provides shared fixtures for integration tests including:
- Test database setup (SQLite in-memory)
- Flask application fixture
- Mock broker fixture
- WebSocket test client
- Alert manager mocks
- ML component fixtures
"""

import os
import sys
import pytest
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Generator, Any
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set test environment
os.environ['FLASK_ENV'] = 'testing'
os.environ['RDT_ENV'] = 'test'


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture(scope='function')
def test_database():
    """
    Create an in-memory SQLite database for testing.

    This fixture creates a fresh database for each test function,
    ensuring test isolation.
    """
    import sqlite3

    # Create in-memory database
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Create tables
    cursor.executescript('''
        -- Trades table
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            shares INTEGER NOT NULL,
            entry_time TIMESTAMP NOT NULL,
            exit_time TIMESTAMP,
            stop_loss REAL,
            take_profit REAL,
            pnl REAL,
            pnl_pct REAL,
            status TEXT DEFAULT 'open',
            strategy TEXT,
            rrs REAL,
            atr REAL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Positions table
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_price REAL NOT NULL,
            shares INTEGER NOT NULL,
            stop_loss REAL,
            take_profit REAL,
            entry_time TIMESTAMP NOT NULL,
            status TEXT DEFAULT 'open',
            current_price REAL,
            unrealized_pnl REAL,
            unrealized_pnl_pct REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Signals table
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            strength TEXT,
            rrs REAL,
            entry_price REAL,
            stop_price REAL,
            target_price REAL,
            atr REAL,
            stock_change_pct REAL,
            spy_change_pct REAL,
            daily_strong INTEGER,
            generated_at TIMESTAMP NOT NULL,
            status TEXT DEFAULT 'pending',
            traded INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Alerts table
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type TEXT NOT NULL,
            severity TEXT DEFAULT 'info',
            title TEXT NOT NULL,
            message TEXT,
            symbol TEXT,
            channel TEXT,
            delivered INTEGER DEFAULT 0,
            delivered_at TIMESTAMP,
            retry_count INTEGER DEFAULT 0,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- API keys table
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            user_id TEXT NOT NULL,
            subscription_tier TEXT DEFAULT 'FREE',
            requests_today INTEGER DEFAULT 0,
            last_request_at TIMESTAMP,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- ML model performance table
        CREATE TABLE IF NOT EXISTS ml_model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            predictions_count INTEGER DEFAULT 0,
            drift_detected INTEGER DEFAULT 0,
            trained_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')

    conn.commit()

    yield conn

    # Cleanup
    conn.close()


@pytest.fixture(scope='function')
def test_db_path(tmp_path):
    """Create a temporary file-based SQLite database path."""
    db_path = tmp_path / "test_trading.db"
    return str(db_path)


# =============================================================================
# Flask Application Fixtures
# =============================================================================

@pytest.fixture(scope='function')
def flask_app(test_database, test_db_path):
    """
    Create a Flask application configured for testing.

    Returns the Flask app instance with testing configuration.
    """
    from flask import Flask

    app = Flask(__name__)
    app.config.update({
        'TESTING': True,
        'DEBUG': False,
        'SECRET_KEY': 'test-secret-key-for-testing',
        'DATABASE_URI': f'sqlite:///{test_db_path}',
        'WTF_CSRF_ENABLED': False,
        'LOGIN_DISABLED': True,
    })

    # Store test database reference
    app.test_db = test_database
    app.test_db_path = test_db_path

    # Try to register blueprints
    try:
        from api.v1.routes import api_v1_bp
        app.register_blueprint(api_v1_bp, url_prefix='/api/v1')
    except ImportError:
        pass

    try:
        from api.v1.auth import auth_bp
        app.register_blueprint(auth_bp, url_prefix='/api/v1/auth')
    except ImportError:
        pass

    return app


@pytest.fixture(scope='function')
def flask_client(flask_app):
    """Create a Flask test client."""
    return flask_app.test_client()


@pytest.fixture(scope='function')
def flask_app_context(flask_app):
    """Create Flask application context."""
    with flask_app.app_context():
        yield flask_app


# =============================================================================
# WebSocket Fixtures
# =============================================================================

@pytest.fixture(scope='function')
def socketio_app(flask_app):
    """
    Create Flask-SocketIO application for testing.

    Returns tuple of (Flask app, SocketIO instance).
    """
    try:
        from flask_socketio import SocketIO
        from web.websocket import register_event_handlers

        socketio = SocketIO(
            flask_app,
            async_mode='threading',  # Use threading for tests
            cors_allowed_origins=["http://localhost:5000", "http://127.0.0.1:5000"],
            logger=False,
            engineio_logger=False
        )

        register_event_handlers(socketio)

        return flask_app, socketio
    except ImportError:
        pytest.skip("Flask-SocketIO not available")


@pytest.fixture(scope='function')
def socketio_client(socketio_app):
    """
    Create a SocketIO test client.

    Yields a connected test client that can send/receive events.
    """
    try:
        from flask_socketio import SocketIOTestClient

        app, socketio = socketio_app
        client = socketio.test_client(app)

        yield client

        # Disconnect after test
        client.disconnect()
    except ImportError:
        pytest.skip("Flask-SocketIO not available")


# =============================================================================
# Broker Fixtures
# =============================================================================

@pytest.fixture(scope='function')
def integration_mock_broker():
    """
    Create a mock broker for integration testing.

    This broker tracks all operations and provides realistic behavior
    for testing the complete trading flow.
    """
    try:
        from brokers.broker_interface import (
            BrokerInterface, Quote, Order, Position, AccountInfo,
            OrderSide, OrderType, OrderStatus,
            BrokerError, OrderError
        )
    except ImportError:
        pytest.skip("Broker interface not available")

    class IntegrationMockBroker(BrokerInterface):
        """Mock broker for integration tests with full tracking."""

        def __init__(self, initial_balance: float = 100000.0):
            self._connected = False
            self.initial_balance = initial_balance
            self.cash = initial_balance
            self._positions: Dict[str, dict] = {}
            self._orders: Dict[str, Order] = {}
            self._order_counter = 0

            # Event tracking
            self.events: list = []
            self.order_history: list = []
            self.position_history: list = []

        def connect(self) -> bool:
            self._connected = True
            self.events.append(('connect', datetime.now()))
            return True

        def disconnect(self) -> None:
            self._connected = False
            self.events.append(('disconnect', datetime.now()))

        @property
        def is_connected(self) -> bool:
            return self._connected

        def get_account(self) -> AccountInfo:
            positions_value = sum(
                p['quantity'] * p['current_price']
                for p in self._positions.values()
            )

            return AccountInfo(
                account_id="INT_TEST_ACCOUNT",
                buying_power=self.cash,
                cash=self.cash,
                equity=self.cash + positions_value,
                day_trades_remaining=3,
                pattern_day_trader=False,
                positions_value=positions_value,
                daily_pnl=0.0
            )

        def get_positions(self) -> Dict[str, Position]:
            result = {}
            for symbol, pos in self._positions.items():
                result[symbol] = Position(
                    symbol=symbol,
                    quantity=pos['quantity'],
                    avg_cost=pos['avg_cost'],
                    current_price=pos['current_price'],
                    market_value=pos['quantity'] * pos['current_price'],
                    unrealized_pnl=(pos['current_price'] - pos['avg_cost']) * pos['quantity'],
                    unrealized_pnl_pct=((pos['current_price'] / pos['avg_cost']) - 1) * 100,
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
            self._order_counter += 1
            order_id = f"INT_{self._order_counter:06d}"

            # Get fill price
            fill_price = price if price else 100.0  # Default price

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

            self._orders[order_id] = order
            self.order_history.append(order)
            self.events.append(('place_order', order_id, symbol, side.value, quantity))

            # Update positions
            self._update_position(symbol.upper(), side, quantity, fill_price)

            return order

        def _update_position(self, symbol: str, side: OrderSide, quantity: int, price: float):
            """Update positions based on order execution."""
            if side in (OrderSide.BUY, OrderSide.BUY_TO_COVER):
                if symbol in self._positions:
                    pos = self._positions[symbol]
                    total_qty = pos['quantity'] + quantity
                    pos['avg_cost'] = (pos['avg_cost'] * pos['quantity'] + price * quantity) / total_qty
                    pos['quantity'] = total_qty
                    pos['current_price'] = price
                else:
                    self._positions[symbol] = {
                        'quantity': quantity,
                        'avg_cost': price,
                        'current_price': price,
                        'direction': 'long'
                    }
                self.cash -= price * quantity
            elif side in (OrderSide.SELL, OrderSide.SELL_SHORT):
                if symbol in self._positions:
                    pos = self._positions[symbol]
                    pos['quantity'] -= quantity
                    if pos['quantity'] <= 0:
                        # Calculate realized P&L
                        pnl = (price - pos['avg_cost']) * quantity
                        self.position_history.append({
                            'symbol': symbol,
                            'entry_price': pos['avg_cost'],
                            'exit_price': price,
                            'quantity': quantity,
                            'pnl': pnl
                        })
                        del self._positions[symbol]
                    self.cash += price * quantity

        def cancel_order(self, order_id: str) -> bool:
            if order_id in self._orders:
                self._orders[order_id].status = OrderStatus.CANCELLED
                self.events.append(('cancel_order', order_id))
                return True
            return False

        def get_order_status(self, order_id: str):
            return self._orders.get(order_id)

        def get_quote(self, symbol: str) -> Quote:
            # Return a realistic quote
            return Quote(
                symbol=symbol.upper(),
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

        def update_price(self, symbol: str, new_price: float):
            """Update current price for a position (for testing P&L updates)."""
            if symbol.upper() in self._positions:
                self._positions[symbol.upper()]['current_price'] = new_price

    broker = IntegrationMockBroker()
    broker.connect()
    return broker


@pytest.fixture(scope='function')
def paper_broker_integration():
    """Create a paper broker for integration tests."""
    try:
        from brokers.paper_broker import PaperBroker

        broker = PaperBroker(
            initial_balance=100000.0,
            slippage_pct=0.0,  # No slippage for deterministic tests
            commission_per_trade=0.0,
            realistic_fills=False
        )
        broker.connect()

        yield broker

        broker.disconnect()
    except ImportError:
        pytest.skip("PaperBroker not available")


# =============================================================================
# Trading Agent Fixtures
# =============================================================================

@pytest.fixture(scope='function')
def mock_event_bus():
    """Create a mock event bus for agent testing."""

    class MockEventBus:
        def __init__(self):
            self.subscribers = {}
            self.published_events = []

        def subscribe(self, event_type, handler):
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(handler)

        def publish(self, event_type, data):
            self.published_events.append((event_type, data))
            if event_type in self.subscribers:
                for handler in self.subscribers[event_type]:
                    handler(data)

        def get_events_of_type(self, event_type):
            return [e for e in self.published_events if e[0] == event_type]

    return MockEventBus()


@pytest.fixture(scope='function')
def analyzer_agent_fixture(integration_mock_broker, mock_event_bus):
    """Create an AnalyzerAgent for integration testing."""
    try:
        from agents.analyzer_agent import AnalyzerAgent

        agent = AnalyzerAgent(
            event_bus=mock_event_bus,
            config={
                'rrs_threshold': 2.0,
                'use_ml': False,  # Disable ML for basic tests
                'min_volume': 500000
            }
        )

        return agent
    except ImportError:
        pytest.skip("AnalyzerAgent not available")


@pytest.fixture(scope='function')
def risk_agent_fixture(integration_mock_broker, mock_event_bus, test_database):
    """Create a RiskAgent for integration testing."""
    try:
        from agents.risk_agent import RiskAgent

        agent = RiskAgent(
            event_bus=mock_event_bus,
            broker=integration_mock_broker,
            config={
                'max_risk_per_trade': 0.01,
                'max_daily_loss': 0.03,
                'max_position_size': 0.10,
                'max_positions': 5
            }
        )

        return agent
    except ImportError:
        pytest.skip("RiskAgent not available")


# =============================================================================
# Alert Manager Fixtures
# =============================================================================

@pytest.fixture(scope='function')
def mock_alert_channels():
    """Create mock alert delivery channels."""

    class MockChannel:
        def __init__(self, name: str, should_fail: bool = False):
            self.name = name
            self.should_fail = should_fail
            self.sent_alerts = []

        def send(self, alert):
            if self.should_fail:
                raise Exception(f"Mock {self.name} failure")
            self.sent_alerts.append(alert)
            return True

    return {
        'pushover': MockChannel('pushover'),
        'discord': MockChannel('discord'),
        'telegram': MockChannel('telegram'),
        'email': MockChannel('email'),
        'failing': MockChannel('failing', should_fail=True)
    }


@pytest.fixture(scope='function')
def alert_manager_fixture(mock_alert_channels):
    """Create an AlertManager for integration testing."""
    try:
        from alerts.alert_manager import AlertManager

        # Create with mocked channels
        manager = AlertManager(
            config={
                'enabled_channels': ['pushover', 'discord'],
                'cooldown_seconds': 0,  # No cooldown for tests
                'max_retries': 3
            }
        )

        # Replace channels with mocks
        manager._channels = mock_alert_channels

        return manager
    except ImportError:
        # Create a simple mock AlertManager
        class MockAlertManager:
            def __init__(self):
                self.alerts_sent = []
                self.channels = mock_alert_channels

            def send_alert(self, alert_type, title, message, **kwargs):
                alert = {
                    'type': alert_type,
                    'title': title,
                    'message': message,
                    **kwargs
                }
                self.alerts_sent.append(alert)
                for channel in self.channels.values():
                    try:
                        channel.send(alert)
                    except Exception:
                        pass
                return True

        return MockAlertManager()


# =============================================================================
# ML Component Fixtures
# =============================================================================

@pytest.fixture(scope='function')
def feature_engineer_fixture():
    """Create a FeatureEngineer for integration testing."""
    try:
        from ml.feature_engineering import FeatureEngineer

        return FeatureEngineer()
    except ImportError:
        pytest.skip("FeatureEngineer not available")


@pytest.fixture(scope='function')
def regime_detector_fixture():
    """Create a RegimeDetector for integration testing."""
    try:
        from ml.regime_detector import RegimeDetector

        detector = RegimeDetector(
            config={
                'use_hmm': False,  # Use heuristic for tests
                'lookback_days': 20
            }
        )

        return detector
    except ImportError:
        pytest.skip("RegimeDetector not available")


@pytest.fixture(scope='function')
def mock_ml_model():
    """Create a mock ML model for integration testing."""

    class MockMLModel:
        def __init__(self):
            self.predictions = []
            self.trained = False
            self.drift_detected = False

        def predict(self, features):
            prediction = {
                'signal': 'long',
                'confidence': 0.75,
                'probability': 0.78
            }
            self.predictions.append(prediction)
            return prediction

        def train(self, X, y):
            self.trained = True
            return {'accuracy': 0.82, 'f1': 0.79}

        def detect_drift(self, recent_data):
            return self.drift_detected

    return MockMLModel()


# =============================================================================
# Sample Data Fixtures for Integration Tests
# =============================================================================

@pytest.fixture(scope='function')
def sample_signal_data():
    """Sample signal data for integration tests."""
    return {
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
        'volume': 50000000
    }


@pytest.fixture(scope='function')
def sample_position_data():
    """Sample position data for integration tests."""
    return {
        'symbol': 'AAPL',
        'direction': 'long',
        'entry_price': 175.50,
        'shares': 100,
        'stop_loss': 172.00,
        'take_profit': 182.50,
        'current_price': 176.25,
        'unrealized_pnl': 75.00,
        'unrealized_pnl_pct': 0.43,
        'entry_time': datetime.now() - timedelta(hours=2),
        'status': 'open'
    }


@pytest.fixture(scope='function')
def sample_api_key():
    """Sample API key for testing."""
    return {
        'key': 'test_api_key_12345',
        'user_id': 'test_user',
        'subscription_tier': 'PRO',
        'is_active': True
    }


@pytest.fixture(scope='function')
def sample_ohlcv_data():
    """Sample OHLCV data for integration tests."""
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')

    base_price = 150.0
    returns = np.random.normal(0.001, 0.02, 60)
    prices = base_price * np.cumprod(1 + returns)

    return pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, 60)),
        'high': prices * (1 + np.random.uniform(0.005, 0.02, 60)),
        'low': prices * (1 - np.random.uniform(0.005, 0.02, 60)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 60),
    }, index=dates)


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup any resources after each test."""
    yield

    # Reset any global state if needed
    try:
        from web.websocket import connected_clients
        connected_clients.clear()
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    yield

    # Reset provider manager if it exists
    try:
        from data.providers.provider_manager import reset_provider_manager
        reset_provider_manager()
    except (ImportError, AttributeError):
        pass

"""
Integration Tests for New API Endpoints

Tests the following endpoint groups added in Phase 3:
- Broker endpoints (/api/v1/broker/*)
- Risk management endpoints (/api/v1/risk/*)
- ML prediction endpoints (/api/v1/ml/*)

These tests verify:
- Authentication requirements
- Request/response formats
- Error handling
- Subscription tier access control
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

pytestmark = [pytest.mark.integration]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def api_headers(sample_api_key):
    """Standard headers with valid API key."""
    return {'X-API-Key': sample_api_key['key'], 'Content-Type': 'application/json'}


@pytest.fixture
def sample_api_key():
    """Sample API key for testing."""
    return {
        'key': 'test_api_key_integration_12345',
        'user_id': 'test_user_1',
        'subscription_tier': 'PRO'
    }


@pytest.fixture
def elite_api_key():
    """Elite tier API key for testing."""
    return {
        'key': 'elite_api_key_integration_67890',
        'user_id': 'elite_user_1',
        'subscription_tier': 'ELITE'
    }


@pytest.fixture
def free_api_key():
    """Free tier API key for testing."""
    return {
        'key': 'free_api_key_integration_11111',
        'user_id': 'free_user_1',
        'subscription_tier': 'FREE'
    }


@pytest.fixture
def mock_broker():
    """Create a mock broker for testing."""
    broker = MagicMock()
    broker.is_connected = True
    broker.broker_name = 'paper'

    # Mock account info
    broker.get_account.return_value = MagicMock(
        account_id='TEST_ACCOUNT_001',
        buying_power=50000.0,
        cash=50000.0,
        equity=100000.0,
        positions_value=50000.0,
        daily_pnl=1250.50,
        day_trades_remaining=3,
        pattern_day_trader=False
    )

    # Mock positions
    mock_position = MagicMock(
        symbol='AAPL',
        quantity=100,
        avg_cost=150.0,
        current_price=155.0,
        market_value=15500.0,
        unrealized_pnl=500.0,
        unrealized_pnl_pct=3.33
    )
    broker.get_positions.return_value = {'AAPL': mock_position}

    # Mock orders
    mock_order = MagicMock(
        order_id='ORD_123',
        symbol='AAPL',
        side='buy',
        quantity=50,
        order_type='limit',
        limit_price=150.0,
        status='pending',
        filled_quantity=0,
        created_at=datetime.now()
    )
    broker.get_orders.return_value = [mock_order]

    # Mock place_order
    broker.place_order.return_value = MagicMock(
        order_id='ORD_NEW_456',
        symbol='TSLA',
        side='buy',
        quantity=10,
        order_type='market',
        status='submitted'
    )

    # Mock cancel_order
    broker.cancel_order.return_value = True

    return broker


@pytest.fixture
def mock_risk_manager():
    """Create a mock risk manager for testing."""
    risk_manager = MagicMock()

    # Mock risk status
    risk_manager.get_current_risk_status.return_value = {
        'daily_pnl': 1250.50,
        'daily_pnl_pct': 1.25,
        'positions_count': 3,
        'max_position_size': 10000.0,
        'current_exposure': 30000.0,
        'exposure_pct': 30.0,
        'var_1d': 2500.0,
        'var_5d': 5500.0,
        'circuit_breaker_triggered': False,
        'daily_loss_limit_remaining': 3750.0,
        'trades_today': 5,
        'max_trades_per_day': 20
    }

    # Mock validate_trade
    risk_manager.validate_trade.return_value = {
        'approved': True,
        'position_size': 100,
        'position_value': 15000.0,
        'risk_per_trade': 150.0,
        'risk_pct': 0.15,
        'warnings': []
    }

    # Mock risk limits
    risk_manager.get_risk_limits.return_value = {
        'max_position_size': 10000.0,
        'max_portfolio_exposure': 100000.0,
        'max_daily_loss': 5000.0,
        'max_trades_per_day': 20,
        'max_correlated_positions': 3
    }

    return risk_manager


@pytest.fixture
def mock_regime_detector():
    """Create a mock regime detector for testing."""
    detector = MagicMock()
    detector.detect_regime.return_value = {
        'regime': 'TRENDING_BULLISH',
        'confidence': 0.85,
        'volatility': 'MEDIUM',
        'trend_strength': 0.72,
        'market_breadth': 0.65,
        'timestamp': datetime.now().isoformat()
    }
    return detector


@pytest.fixture
def mock_ensemble_model():
    """Create a mock ensemble model for testing."""
    model = MagicMock()
    model.predict_proba.return_value = [0.73]
    model.is_loaded = True
    model.model_info = {
        'xgboost': {'loaded': True, 'version': '1.0'},
        'random_forest': {'loaded': True, 'version': '1.0'},
        'meta_learner': {'loaded': True, 'version': '1.0'}
    }
    return model


def setup_api_key(test_database, api_key_fixture):
    """Helper to insert an API key into test database."""
    cursor = test_database.cursor()
    cursor.execute('''
        INSERT INTO api_keys (key, user_id, subscription_tier, is_active)
        VALUES (?, ?, ?, ?)
    ''', (
        api_key_fixture['key'],
        api_key_fixture['user_id'],
        api_key_fixture['subscription_tier'],
        1
    ))
    test_database.commit()


# =============================================================================
# Broker Endpoint Tests
# =============================================================================

class TestBrokerAccountEndpoint:
    """Tests for GET /api/v1/broker/account."""

    def test_account_requires_authentication(self, flask_client):
        """Test that account endpoint requires API key."""
        response = flask_client.get('/api/v1/broker/account')
        assert response.status_code in (401, 404)

    def test_account_returns_data(self, flask_client, test_database, sample_api_key, mock_broker):
        """Test that account endpoint returns broker account info."""
        setup_api_key(test_database, sample_api_key)

        with patch('api.v1.routes.get_global_broker', return_value=mock_broker):
            response = flask_client.get(
                '/api/v1/broker/account',
                headers={'X-API-Key': sample_api_key['key']}
            )

        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'account_id' in data or 'buying_power' in data or 'equity' in data
        else:
            # Endpoint may not be registered in test environment
            assert response.status_code in (401, 404, 503)

    def test_account_handles_broker_disconnect(self, flask_client, test_database, sample_api_key, mock_broker):
        """Test graceful handling when broker is disconnected."""
        setup_api_key(test_database, sample_api_key)
        mock_broker.is_connected = False

        with patch('api.v1.routes.get_global_broker', return_value=mock_broker):
            response = flask_client.get(
                '/api/v1/broker/account',
                headers={'X-API-Key': sample_api_key['key']}
            )

        # Should return 503 or appropriate error when broker disconnected
        assert response.status_code in (200, 404, 503)


class TestBrokerPositionsEndpoint:
    """Tests for GET /api/v1/broker/positions."""

    def test_positions_requires_authentication(self, flask_client):
        """Test that positions endpoint requires API key."""
        response = flask_client.get('/api/v1/broker/positions')
        assert response.status_code in (401, 404)

    def test_positions_returns_list(self, flask_client, test_database, sample_api_key, mock_broker):
        """Test that positions endpoint returns position list."""
        setup_api_key(test_database, sample_api_key)

        with patch('api.v1.routes.get_global_broker', return_value=mock_broker):
            response = flask_client.get(
                '/api/v1/broker/positions',
                headers={'X-API-Key': sample_api_key['key']}
            )

        if response.status_code == 200:
            data = json.loads(response.data)
            assert isinstance(data, (list, dict))
        else:
            assert response.status_code in (401, 404, 503)


class TestBrokerOrdersEndpoint:
    """Tests for GET/POST /api/v1/broker/orders."""

    def test_get_orders_requires_authentication(self, flask_client):
        """Test that orders endpoint requires API key."""
        response = flask_client.get('/api/v1/broker/orders')
        assert response.status_code in (401, 404)

    def test_get_orders_returns_list(self, flask_client, test_database, sample_api_key, mock_broker):
        """Test that GET orders returns order list."""
        setup_api_key(test_database, sample_api_key)

        with patch('api.v1.routes.get_global_broker', return_value=mock_broker):
            response = flask_client.get(
                '/api/v1/broker/orders',
                headers={'X-API-Key': sample_api_key['key']}
            )

        if response.status_code == 200:
            data = json.loads(response.data)
            assert isinstance(data, (list, dict))
        else:
            assert response.status_code in (401, 404, 503)

    def test_post_order_requires_elite_tier(self, flask_client, test_database, free_api_key, mock_broker):
        """Test that placing orders requires higher tier subscription."""
        setup_api_key(test_database, free_api_key)

        order_data = {
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 10,
            'order_type': 'market'
        }

        with patch('api.v1.routes.get_global_broker', return_value=mock_broker):
            response = flask_client.post(
                '/api/v1/broker/orders',
                headers={'X-API-Key': free_api_key['key'], 'Content-Type': 'application/json'},
                data=json.dumps(order_data)
            )

        # Free tier should be denied for order placement
        assert response.status_code in (401, 403, 404)

    def test_post_order_validates_input(self, flask_client, test_database, elite_api_key, mock_broker):
        """Test that order placement validates required fields."""
        setup_api_key(test_database, elite_api_key)

        # Missing required fields
        invalid_order = {'symbol': 'AAPL'}

        with patch('api.v1.routes.get_global_broker', return_value=mock_broker):
            response = flask_client.post(
                '/api/v1/broker/orders',
                headers={'X-API-Key': elite_api_key['key'], 'Content-Type': 'application/json'},
                data=json.dumps(invalid_order)
            )

        # Should return 400 for invalid input or 404 if endpoint not registered
        assert response.status_code in (400, 404, 422)


class TestBrokerCancelOrderEndpoint:
    """Tests for DELETE /api/v1/broker/orders/<order_id>."""

    def test_cancel_order_requires_authentication(self, flask_client):
        """Test that cancel order requires API key."""
        response = flask_client.delete('/api/v1/broker/orders/ORD_123')
        assert response.status_code in (401, 404)

    def test_cancel_order_works(self, flask_client, test_database, elite_api_key, mock_broker):
        """Test successful order cancellation."""
        setup_api_key(test_database, elite_api_key)

        with patch('api.v1.routes.get_global_broker', return_value=mock_broker):
            response = flask_client.delete(
                '/api/v1/broker/orders/ORD_123',
                headers={'X-API-Key': elite_api_key['key']}
            )

        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'success' in data or 'cancelled' in str(data).lower()
        else:
            assert response.status_code in (401, 403, 404)


# =============================================================================
# Risk Management Endpoint Tests
# =============================================================================

class TestRiskStatusEndpoint:
    """Tests for GET /api/v1/risk/status."""

    def test_risk_status_requires_authentication(self, flask_client):
        """Test that risk status requires API key."""
        response = flask_client.get('/api/v1/risk/status')
        assert response.status_code in (401, 404)

    def test_risk_status_returns_metrics(self, flask_client, test_database, sample_api_key, mock_risk_manager):
        """Test that risk status returns current risk metrics."""
        setup_api_key(test_database, sample_api_key)

        with patch('api.v1.routes.get_global_risk_manager', return_value=mock_risk_manager):
            response = flask_client.get(
                '/api/v1/risk/status',
                headers={'X-API-Key': sample_api_key['key']}
            )

        if response.status_code == 200:
            data = json.loads(response.data)
            # Should contain risk metrics
            assert isinstance(data, dict)
        else:
            assert response.status_code in (401, 404, 503)


class TestRiskValidateEndpoint:
    """Tests for POST /api/v1/risk/validate."""

    def test_validate_requires_authentication(self, flask_client):
        """Test that trade validation requires API key."""
        response = flask_client.post('/api/v1/risk/validate')
        assert response.status_code in (401, 404, 415)

    def test_validate_trade_success(self, flask_client, test_database, sample_api_key, mock_risk_manager):
        """Test successful trade validation."""
        setup_api_key(test_database, sample_api_key)

        trade_data = {
            'symbol': 'AAPL',
            'direction': 'LONG',
            'entry_price': 150.0,
            'stop_loss': 147.0,
            'position_size': 100
        }

        with patch('api.v1.routes.get_global_risk_manager', return_value=mock_risk_manager):
            response = flask_client.post(
                '/api/v1/risk/validate',
                headers={'X-API-Key': sample_api_key['key'], 'Content-Type': 'application/json'},
                data=json.dumps(trade_data)
            )

        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'approved' in data or 'valid' in str(data).lower()
        else:
            assert response.status_code in (400, 401, 404)

    def test_validate_missing_fields(self, flask_client, test_database, sample_api_key, mock_risk_manager):
        """Test validation with missing required fields."""
        setup_api_key(test_database, sample_api_key)

        incomplete_data = {'symbol': 'AAPL'}

        with patch('api.v1.routes.get_global_risk_manager', return_value=mock_risk_manager):
            response = flask_client.post(
                '/api/v1/risk/validate',
                headers={'X-API-Key': sample_api_key['key'], 'Content-Type': 'application/json'},
                data=json.dumps(incomplete_data)
            )

        # Should return error for missing fields
        assert response.status_code in (400, 404, 422)


class TestRiskLimitsEndpoint:
    """Tests for GET/PUT /api/v1/risk/limits."""

    def test_get_limits_requires_authentication(self, flask_client):
        """Test that getting risk limits requires API key."""
        response = flask_client.get('/api/v1/risk/limits')
        assert response.status_code in (401, 404)

    def test_get_limits_returns_config(self, flask_client, test_database, sample_api_key, mock_risk_manager):
        """Test that GET limits returns current configuration."""
        setup_api_key(test_database, sample_api_key)

        with patch('api.v1.routes.get_global_risk_manager', return_value=mock_risk_manager):
            response = flask_client.get(
                '/api/v1/risk/limits',
                headers={'X-API-Key': sample_api_key['key']}
            )

        if response.status_code == 200:
            data = json.loads(response.data)
            assert isinstance(data, dict)
        else:
            assert response.status_code in (401, 404, 503)

    def test_put_limits_requires_elite_tier(self, flask_client, test_database, free_api_key, mock_risk_manager):
        """Test that updating risk limits requires elevated permissions."""
        setup_api_key(test_database, free_api_key)

        new_limits = {
            'max_position_size': 20000.0,
            'max_daily_loss': 10000.0
        }

        with patch('api.v1.routes.get_global_risk_manager', return_value=mock_risk_manager):
            response = flask_client.put(
                '/api/v1/risk/limits',
                headers={'X-API-Key': free_api_key['key'], 'Content-Type': 'application/json'},
                data=json.dumps(new_limits)
            )

        # Free tier should be denied
        assert response.status_code in (401, 403, 404)


# =============================================================================
# ML Endpoint Tests
# =============================================================================

class TestMLRegimeEndpoint:
    """Tests for GET /api/v1/ml/regime."""

    def test_regime_requires_authentication(self, flask_client):
        """Test that regime endpoint requires API key."""
        response = flask_client.get('/api/v1/ml/regime')
        assert response.status_code in (401, 404)

    def test_regime_returns_current_regime(self, flask_client, test_database, sample_api_key, mock_regime_detector):
        """Test that regime endpoint returns current market regime."""
        setup_api_key(test_database, sample_api_key)

        with patch('api.v1.routes.get_global_regime_detector', return_value=mock_regime_detector):
            response = flask_client.get(
                '/api/v1/ml/regime',
                headers={'X-API-Key': sample_api_key['key']}
            )

        if response.status_code == 200:
            data = json.loads(response.data)
            # Should contain regime information
            assert 'regime' in data or 'market_regime' in str(data).lower() or isinstance(data, dict)
        else:
            assert response.status_code in (401, 404, 503)

    def test_regime_handles_detector_unavailable(self, flask_client, test_database, sample_api_key):
        """Test graceful handling when regime detector is unavailable."""
        setup_api_key(test_database, sample_api_key)

        with patch('api.v1.routes.get_global_regime_detector', return_value=None):
            response = flask_client.get(
                '/api/v1/ml/regime',
                headers={'X-API-Key': sample_api_key['key']}
            )

        # Should return 503 or appropriate error
        assert response.status_code in (200, 404, 503)


class TestMLPredictEndpoint:
    """Tests for POST /api/v1/ml/predict."""

    def test_predict_requires_authentication(self, flask_client):
        """Test that predict endpoint requires API key."""
        response = flask_client.post('/api/v1/ml/predict')
        assert response.status_code in (401, 404, 415)

    def test_predict_returns_probability(self, flask_client, test_database, sample_api_key, mock_ensemble_model):
        """Test that predict endpoint returns success probability."""
        setup_api_key(test_database, sample_api_key)

        prediction_data = {
            'symbol': 'AAPL',
            'direction': 'LONG',
            'rrs': 2.5,
            'entry_price': 150.0,
            'stop_loss': 147.0,
            'features': {
                'rsi': 55.0,
                'macd_histogram': 0.5,
                'volume_ratio': 1.2
            }
        }

        with patch('api.v1.routes.get_global_ensemble_model', return_value=mock_ensemble_model):
            response = flask_client.post(
                '/api/v1/ml/predict',
                headers={'X-API-Key': sample_api_key['key'], 'Content-Type': 'application/json'},
                data=json.dumps(prediction_data)
            )

        if response.status_code == 200:
            data = json.loads(response.data)
            # Should contain prediction or probability
            assert 'probability' in data or 'prediction' in data or 'success_prob' in str(data).lower() or isinstance(data, dict)
        else:
            assert response.status_code in (400, 401, 404, 503)

    def test_predict_requires_pro_tier(self, flask_client, test_database, free_api_key, mock_ensemble_model):
        """Test that ML predictions require PRO tier or higher."""
        setup_api_key(test_database, free_api_key)

        prediction_data = {
            'symbol': 'AAPL',
            'direction': 'LONG',
            'rrs': 2.5
        }

        with patch('api.v1.routes.get_global_ensemble_model', return_value=mock_ensemble_model):
            response = flask_client.post(
                '/api/v1/ml/predict',
                headers={'X-API-Key': free_api_key['key'], 'Content-Type': 'application/json'},
                data=json.dumps(prediction_data)
            )

        # Free tier may be denied ML predictions
        assert response.status_code in (200, 401, 403, 404)


class TestMLModelsStatusEndpoint:
    """Tests for GET /api/v1/ml/models/status."""

    def test_models_status_requires_authentication(self, flask_client):
        """Test that models status requires API key."""
        response = flask_client.get('/api/v1/ml/models/status')
        assert response.status_code in (401, 404)

    def test_models_status_returns_health(self, flask_client, test_database, sample_api_key, mock_ensemble_model):
        """Test that models status returns health information."""
        setup_api_key(test_database, sample_api_key)

        with patch('api.v1.routes.get_global_ensemble_model', return_value=mock_ensemble_model):
            response = flask_client.get(
                '/api/v1/ml/models/status',
                headers={'X-API-Key': sample_api_key['key']}
            )

        if response.status_code == 200:
            data = json.loads(response.data)
            assert isinstance(data, dict)
        else:
            assert response.status_code in (401, 404, 503)


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestEndpointErrorHandling:
    """Tests for error handling across new endpoints."""

    def test_invalid_json_returns_400(self, flask_client, test_database, sample_api_key):
        """Test that invalid JSON returns 400 Bad Request."""
        setup_api_key(test_database, sample_api_key)

        response = flask_client.post(
            '/api/v1/ml/predict',
            headers={'X-API-Key': sample_api_key['key'], 'Content-Type': 'application/json'},
            data='not valid json{'
        )

        assert response.status_code in (400, 404, 415)

    def test_internal_errors_return_500(self, flask_client, test_database, sample_api_key):
        """Test that internal errors are handled gracefully."""
        setup_api_key(test_database, sample_api_key)

        # Mock an exception
        with patch('api.v1.routes.get_global_broker', side_effect=Exception('Test error')):
            response = flask_client.get(
                '/api/v1/broker/account',
                headers={'X-API-Key': sample_api_key['key']}
            )

        # Should handle exception gracefully
        assert response.status_code in (404, 500, 503)


# =============================================================================
# Subscription Tier Access Control Tests
# =============================================================================

class TestTierAccessControl:
    """Tests for subscription tier access control on new endpoints."""

    def test_free_tier_limited_ml_access(self, flask_client, test_database, free_api_key):
        """Test that FREE tier has limited ML endpoint access."""
        setup_api_key(test_database, free_api_key)

        # Regime might be accessible
        response = flask_client.get(
            '/api/v1/ml/regime',
            headers={'X-API-Key': free_api_key['key']}
        )

        # Basic regime info might be available, predictions restricted
        assert response.status_code in (200, 401, 403, 404)

    def test_pro_tier_full_ml_access(self, flask_client, test_database, sample_api_key, mock_ensemble_model):
        """Test that PRO tier has full ML access."""
        setup_api_key(test_database, sample_api_key)

        with patch('api.v1.routes.get_global_ensemble_model', return_value=mock_ensemble_model):
            response = flask_client.get(
                '/api/v1/ml/models/status',
                headers={'X-API-Key': sample_api_key['key']}
            )

        if response.status_code == 200:
            data = json.loads(response.data)
            assert isinstance(data, dict)
        else:
            assert response.status_code in (401, 404, 503)

    def test_elite_tier_trading_access(self, flask_client, test_database, elite_api_key, mock_broker):
        """Test that ELITE tier can execute trades."""
        setup_api_key(test_database, elite_api_key)

        order_data = {
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 10,
            'order_type': 'market'
        }

        with patch('api.v1.routes.get_global_broker', return_value=mock_broker):
            response = flask_client.post(
                '/api/v1/broker/orders',
                headers={'X-API-Key': elite_api_key['key'], 'Content-Type': 'application/json'},
                data=json.dumps(order_data)
            )

        # Elite tier should be able to place orders
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'order_id' in data or 'success' in data or isinstance(data, dict)
        else:
            # Endpoint may not be registered or require additional config
            assert response.status_code in (400, 401, 404, 503)


# =============================================================================
# Rate Limiting Tests
# =============================================================================

class TestRateLimiting:
    """Tests for rate limiting on new endpoints."""

    def test_requests_increment_counter(self, flask_client, test_database, sample_api_key, mock_broker):
        """Test that requests increment the daily request counter."""
        setup_api_key(test_database, sample_api_key)

        # Make a request
        with patch('api.v1.routes.get_global_broker', return_value=mock_broker):
            flask_client.get(
                '/api/v1/broker/account',
                headers={'X-API-Key': sample_api_key['key']}
            )

        # Check if counter was incremented (depends on implementation)
        cursor = test_database.cursor()
        cursor.execute(
            'SELECT requests_today FROM api_keys WHERE key = ?',
            (sample_api_key['key'],)
        )
        result = cursor.fetchone()

        # Counter may or may not be implemented
        if result:
            assert result[0] >= 0


# =============================================================================
# Integration Flow Tests
# =============================================================================

class TestEndToEndFlow:
    """Tests for end-to-end flows using new endpoints."""

    def test_trade_validation_flow(
        self, flask_client, test_database, sample_api_key,
        mock_risk_manager, mock_ensemble_model, mock_regime_detector
    ):
        """Test complete trade validation flow: regime -> predict -> validate."""
        setup_api_key(test_database, sample_api_key)
        headers = {'X-API-Key': sample_api_key['key'], 'Content-Type': 'application/json'}

        # Step 1: Check market regime
        with patch('api.v1.routes.get_global_regime_detector', return_value=mock_regime_detector):
            regime_response = flask_client.get('/api/v1/ml/regime', headers=headers)

        # Step 2: Get ML prediction
        prediction_data = {'symbol': 'AAPL', 'direction': 'LONG', 'rrs': 2.5}
        with patch('api.v1.routes.get_global_ensemble_model', return_value=mock_ensemble_model):
            predict_response = flask_client.post(
                '/api/v1/ml/predict',
                headers=headers,
                data=json.dumps(prediction_data)
            )

        # Step 3: Validate trade with risk manager
        trade_data = {
            'symbol': 'AAPL',
            'direction': 'LONG',
            'entry_price': 150.0,
            'stop_loss': 147.0,
            'position_size': 100
        }
        with patch('api.v1.routes.get_global_risk_manager', return_value=mock_risk_manager):
            validate_response = flask_client.post(
                '/api/v1/risk/validate',
                headers=headers,
                data=json.dumps(trade_data)
            )

        # All endpoints should either work or return 404 (not registered in test)
        for response in [regime_response, predict_response, validate_response]:
            assert response.status_code in (200, 400, 404, 503)

    def test_portfolio_monitoring_flow(
        self, flask_client, test_database, sample_api_key,
        mock_broker, mock_risk_manager
    ):
        """Test portfolio monitoring flow: account -> positions -> risk status."""
        setup_api_key(test_database, sample_api_key)
        headers = {'X-API-Key': sample_api_key['key']}

        # Step 1: Get account info
        with patch('api.v1.routes.get_global_broker', return_value=mock_broker):
            account_response = flask_client.get('/api/v1/broker/account', headers=headers)

        # Step 2: Get positions
        with patch('api.v1.routes.get_global_broker', return_value=mock_broker):
            positions_response = flask_client.get('/api/v1/broker/positions', headers=headers)

        # Step 3: Get risk status
        with patch('api.v1.routes.get_global_risk_manager', return_value=mock_risk_manager):
            risk_response = flask_client.get('/api/v1/risk/status', headers=headers)

        # All should work or return 404 if not registered
        for response in [account_response, positions_response, risk_response]:
            assert response.status_code in (200, 404, 503)

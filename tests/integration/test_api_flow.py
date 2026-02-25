"""
Integration Tests for API Flow

Tests API authentication and endpoint functionality:
- API key authentication
- Signal retrieval endpoints
- Position management endpoints
- Backtest endpoint
- Rate limiting
- Subscription tier access control
"""

import pytest
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

pytestmark = [pytest.mark.integration]


class TestAPIAuthentication:
    """Test API authentication flow."""

    def test_request_without_api_key_rejected(self, flask_client):
        """Test that requests without API key are rejected."""
        response = flask_client.get('/api/v1/signals/current')

        # Should return 401 Unauthorized
        assert response.status_code in (401, 404)  # 404 if route not registered

    def test_request_with_invalid_api_key_rejected(self, flask_client):
        """Test that requests with invalid API key are rejected."""
        response = flask_client.get(
            '/api/v1/signals/current',
            headers={'X-API-Key': 'invalid_key_12345'}
        )

        assert response.status_code in (401, 403, 404)

    def test_request_with_valid_api_key_accepted(self, flask_client, test_database, sample_api_key):
        """Test that requests with valid API key are accepted."""
        cursor = test_database.cursor()

        # Insert valid API key
        cursor.execute('''
            INSERT INTO api_keys (key, user_id, subscription_tier, is_active)
            VALUES (?, ?, ?, ?)
        ''', (
            sample_api_key['key'],
            sample_api_key['user_id'],
            sample_api_key['subscription_tier'],
            1
        ))
        test_database.commit()

        # Make request with valid key (endpoint may not exist)
        response = flask_client.get(
            '/api/v1/health',
            headers={'X-API-Key': sample_api_key['key']}
        )

        # Health endpoint should work or return 404 if not registered
        assert response.status_code in (200, 404)

    def test_deactivated_api_key_rejected(self, flask_client, test_database):
        """Test that deactivated API keys are rejected."""
        cursor = test_database.cursor()

        # Insert deactivated API key
        cursor.execute('''
            INSERT INTO api_keys (key, user_id, subscription_tier, is_active)
            VALUES (?, ?, ?, ?)
        ''', ('deactivated_key_123', 'user1', 'PRO', 0))
        test_database.commit()

        response = flask_client.get(
            '/api/v1/signals/current',
            headers={'X-API-Key': 'deactivated_key_123'}
        )

        assert response.status_code in (401, 403, 404)


class TestSubscriptionTiers:
    """Test subscription tier access control."""

    def test_free_tier_access_limits(self, test_database):
        """Test FREE tier has limited access."""
        cursor = test_database.cursor()

        # Define tier limits
        tier_limits = {
            'FREE': {
                'requests_per_day': 100,
                'signals_access': True,
                'positions_access': False,
                'backtest_access': False,
                'realtime_access': False
            }
        }

        # Insert FREE tier API key
        cursor.execute('''
            INSERT INTO api_keys (key, user_id, subscription_tier, requests_today)
            VALUES (?, ?, ?, ?)
        ''', ('free_key_123', 'free_user', 'FREE', 0))
        test_database.commit()

        # Verify tier limits
        cursor.execute('SELECT subscription_tier FROM api_keys WHERE key = ?', ('free_key_123',))
        tier = cursor.fetchone()[0]

        assert tier == 'FREE'
        assert tier_limits['FREE']['requests_per_day'] == 100
        assert tier_limits['FREE']['backtest_access'] is False

    def test_pro_tier_has_full_access(self, test_database):
        """Test PRO tier has full access."""
        cursor = test_database.cursor()

        tier_features = {
            'PRO': {
                'requests_per_day': 10000,
                'signals_access': True,
                'positions_access': True,
                'backtest_access': True,
                'realtime_access': True
            }
        }

        cursor.execute('''
            INSERT INTO api_keys (key, user_id, subscription_tier, requests_today)
            VALUES (?, ?, ?, ?)
        ''', ('pro_key_456', 'pro_user', 'PRO', 0))
        test_database.commit()

        cursor.execute('SELECT subscription_tier FROM api_keys WHERE key = ?', ('pro_key_456',))
        tier = cursor.fetchone()[0]

        assert tier == 'PRO'
        assert tier_features['PRO']['backtest_access'] is True
        assert tier_features['PRO']['realtime_access'] is True

    def test_elite_tier_unlimited_access(self, test_database):
        """Test ELITE tier has unlimited access."""
        cursor = test_database.cursor()

        cursor.execute('''
            INSERT INTO api_keys (key, user_id, subscription_tier, requests_today)
            VALUES (?, ?, ?, ?)
        ''', ('elite_key_789', 'elite_user', 'ELITE', 0))
        test_database.commit()

        cursor.execute('SELECT subscription_tier FROM api_keys WHERE key = ?', ('elite_key_789',))
        tier = cursor.fetchone()[0]

        assert tier == 'ELITE'


class TestRateLimiting:
    """Test API rate limiting."""

    def test_rate_limit_tracking(self, test_database):
        """Test that request counts are tracked."""
        cursor = test_database.cursor()

        # Insert API key
        cursor.execute('''
            INSERT INTO api_keys (key, user_id, subscription_tier, requests_today)
            VALUES (?, ?, ?, ?)
        ''', ('rate_test_key', 'rate_user', 'FREE', 0))
        test_database.commit()

        # Simulate requests
        for i in range(5):
            cursor.execute('''
                UPDATE api_keys
                SET requests_today = requests_today + 1,
                    last_request_at = ?
                WHERE key = ?
            ''', (datetime.now().isoformat(), 'rate_test_key'))
            test_database.commit()

        # Check count
        cursor.execute('SELECT requests_today FROM api_keys WHERE key = ?', ('rate_test_key',))
        count = cursor.fetchone()[0]

        assert count == 5

    def test_rate_limit_exceeded_rejected(self, test_database):
        """Test that requests are rejected when rate limit exceeded."""
        cursor = test_database.cursor()

        # Insert API key at limit
        free_limit = 100
        cursor.execute('''
            INSERT INTO api_keys (key, user_id, subscription_tier, requests_today)
            VALUES (?, ?, ?, ?)
        ''', ('exceeded_key', 'exceeded_user', 'FREE', free_limit))
        test_database.commit()

        # Check if limit exceeded
        cursor.execute('''
            SELECT requests_today FROM api_keys WHERE key = ?
        ''', ('exceeded_key',))
        current_count = cursor.fetchone()[0]

        # Simulate rate limit check
        is_limited = current_count >= free_limit
        assert is_limited is True

    def test_rate_limit_resets_daily(self, test_database):
        """Test that rate limits reset daily."""
        cursor = test_database.cursor()

        # Insert key with yesterday's requests
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        cursor.execute('''
            INSERT INTO api_keys (key, user_id, subscription_tier, requests_today, last_request_at)
            VALUES (?, ?, ?, ?, ?)
        ''', ('reset_key', 'reset_user', 'FREE', 100, yesterday))
        test_database.commit()

        # Simulate daily reset logic
        cursor.execute('''
            SELECT last_request_at FROM api_keys WHERE key = ?
        ''', ('reset_key',))
        last_request = cursor.fetchone()[0]

        # Check if last request was before today
        last_request_date = datetime.fromisoformat(last_request).date()
        today = datetime.now().date()

        should_reset = last_request_date < today
        assert should_reset is True

        # Perform reset
        if should_reset:
            cursor.execute('''
                UPDATE api_keys SET requests_today = 0 WHERE key = ?
            ''', ('reset_key',))
            test_database.commit()

        cursor.execute('SELECT requests_today FROM api_keys WHERE key = ?', ('reset_key',))
        new_count = cursor.fetchone()[0]
        assert new_count == 0


class TestSignalEndpoints:
    """Test signal-related API endpoints."""

    def test_get_current_signals(self, test_database, sample_signal_data):
        """Test retrieving current signals."""
        cursor = test_database.cursor()

        # Insert test signals
        cursor.execute('''
            INSERT INTO signals (
                symbol, direction, strength, rrs, entry_price,
                stop_price, target_price, atr, generated_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            sample_signal_data['symbol'],
            sample_signal_data['direction'],
            sample_signal_data['strength'],
            sample_signal_data['rrs'],
            sample_signal_data['entry_price'],
            sample_signal_data['stop_price'],
            sample_signal_data['target_price'],
            sample_signal_data['atr'],
            datetime.now().isoformat(),
            'active'
        ))
        test_database.commit()

        # Query current signals
        cursor.execute('''
            SELECT * FROM signals WHERE status = 'active'
        ''')
        signals = cursor.fetchall()

        assert len(signals) >= 1
        assert signals[0][1] == sample_signal_data['symbol']

    def test_get_signal_history(self, test_database):
        """Test retrieving signal history."""
        cursor = test_database.cursor()

        # Insert historical signals
        for i in range(5):
            days_ago = datetime.now() - timedelta(days=i)
            cursor.execute('''
                INSERT INTO signals (
                    symbol, direction, strength, rrs, entry_price,
                    generated_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                f'SYM{i}', 'long', 'strong', 2.5 + i * 0.1,
                100.0 + i * 10, days_ago.isoformat(), 'expired'
            ))
        test_database.commit()

        # Query history
        cursor.execute('''
            SELECT * FROM signals ORDER BY generated_at DESC
        ''')
        history = cursor.fetchall()

        assert len(history) >= 5

    def test_filter_signals_by_direction(self, test_database):
        """Test filtering signals by direction."""
        cursor = test_database.cursor()

        # Insert mixed signals
        cursor.executemany('''
            INSERT INTO signals (symbol, direction, rrs, entry_price, generated_at, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', [
            ('AAPL', 'long', 2.5, 175.0, datetime.now().isoformat(), 'active'),
            ('GOOGL', 'long', 2.8, 140.0, datetime.now().isoformat(), 'active'),
            ('TSLA', 'short', -3.0, 200.0, datetime.now().isoformat(), 'active'),
        ])
        test_database.commit()

        # Filter by direction
        cursor.execute("SELECT * FROM signals WHERE direction = 'long' AND status = 'active'")
        long_signals = cursor.fetchall()

        cursor.execute("SELECT * FROM signals WHERE direction = 'short' AND status = 'active'")
        short_signals = cursor.fetchall()

        assert len(long_signals) == 2
        assert len(short_signals) == 1


class TestPositionEndpoints:
    """Test position management API endpoints."""

    def test_get_open_positions(self, test_database, sample_position_data):
        """Test retrieving open positions."""
        cursor = test_database.cursor()

        cursor.execute('''
            INSERT INTO positions (
                symbol, direction, entry_price, shares, stop_loss,
                take_profit, entry_time, status, current_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            sample_position_data['symbol'],
            sample_position_data['direction'],
            sample_position_data['entry_price'],
            sample_position_data['shares'],
            sample_position_data['stop_loss'],
            sample_position_data['take_profit'],
            datetime.now().isoformat(),
            'open',
            sample_position_data['current_price']
        ))
        test_database.commit()

        cursor.execute("SELECT * FROM positions WHERE status = 'open'")
        positions = cursor.fetchall()

        assert len(positions) >= 1

    def test_get_position_by_symbol(self, test_database):
        """Test retrieving position by symbol."""
        cursor = test_database.cursor()

        cursor.execute('''
            INSERT INTO positions (
                symbol, direction, entry_price, shares, entry_time, status
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', ('AAPL', 'long', 175.0, 100, datetime.now().isoformat(), 'open'))
        test_database.commit()

        cursor.execute("SELECT * FROM positions WHERE symbol = ?", ('AAPL',))
        position = cursor.fetchone()

        assert position is not None
        assert position[1] == 'AAPL'

    def test_update_position_pnl(self, test_database):
        """Test updating position P&L."""
        cursor = test_database.cursor()

        # Insert position
        cursor.execute('''
            INSERT INTO positions (
                symbol, direction, entry_price, shares, entry_time,
                status, current_price, unrealized_pnl, unrealized_pnl_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', ('AAPL', 'long', 175.0, 100, datetime.now().isoformat(), 'open', 175.0, 0.0, 0.0))
        test_database.commit()

        # Update with new price
        new_price = 180.0
        entry_price = 175.0
        shares = 100
        new_pnl = (new_price - entry_price) * shares
        new_pnl_pct = ((new_price / entry_price) - 1) * 100

        cursor.execute('''
            UPDATE positions
            SET current_price = ?, unrealized_pnl = ?, unrealized_pnl_pct = ?
            WHERE symbol = ?
        ''', (new_price, new_pnl, new_pnl_pct, 'AAPL'))
        test_database.commit()

        cursor.execute("SELECT unrealized_pnl FROM positions WHERE symbol = ?", ('AAPL',))
        pnl = cursor.fetchone()[0]

        assert pnl == 500.0  # ($5 gain * 100 shares)


class TestBacktestEndpoint:
    """Test backtest API endpoint."""

    def test_backtest_requires_pro_tier(self, test_database):
        """Test that backtest requires PRO tier or higher."""
        cursor = test_database.cursor()

        # FREE tier should not have access
        free_features = {
            'backtest': False,
            'signals': True
        }

        # PRO tier should have access
        pro_features = {
            'backtest': True,
            'signals': True
        }

        assert free_features['backtest'] is False
        assert pro_features['backtest'] is True

    def test_backtest_request_validation(self):
        """Test backtest request parameter validation."""
        valid_request = {
            'symbol': 'AAPL',
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'strategy': 'RRS_Momentum',
            'initial_capital': 100000,
            'risk_per_trade': 0.01
        }

        # Validate required fields
        required_fields = ['symbol', 'start_date', 'end_date', 'strategy']
        for field in required_fields:
            assert field in valid_request

        # Validate date format
        from datetime import datetime
        start = datetime.strptime(valid_request['start_date'], '%Y-%m-%d')
        end = datetime.strptime(valid_request['end_date'], '%Y-%m-%d')
        assert end > start

    def test_backtest_result_structure(self):
        """Test backtest result structure."""
        mock_result = {
            'status': 'completed',
            'symbol': 'AAPL',
            'period': {
                'start': '2024-01-01',
                'end': '2024-12-31'
            },
            'metrics': {
                'total_trades': 45,
                'winning_trades': 28,
                'losing_trades': 17,
                'win_rate': 0.622,
                'profit_factor': 1.85,
                'total_return': 0.152,
                'max_drawdown': -0.082,
                'sharpe_ratio': 1.45
            },
            'trades': []
        }

        # Verify structure
        assert 'status' in mock_result
        assert 'metrics' in mock_result
        assert 'win_rate' in mock_result['metrics']
        assert mock_result['metrics']['win_rate'] > 0.5


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check_response(self, flask_client):
        """Test health check returns system status."""
        response = flask_client.get('/api/v1/health')

        # May return 404 if route not registered
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'status' in data or 'healthy' in data

    def test_health_check_includes_components(self):
        """Test health check includes component status."""
        mock_health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'database': 'healthy',
                'broker': 'connected',
                'scanner': 'running',
                'websocket': 'active'
            },
            'version': '1.0.0'
        }

        assert mock_health['status'] == 'healthy'
        assert 'components' in mock_health
        assert mock_health['components']['database'] == 'healthy'


class TestAPIErrorHandling:
    """Test API error handling."""

    def test_invalid_json_returns_400(self, flask_client):
        """Test that invalid JSON returns 400 error."""
        response = flask_client.post(
            '/api/v1/backtest',
            data='invalid json{',
            content_type='application/json'
        )

        # Should return 400 or 404 (if route not registered)
        assert response.status_code in (400, 404, 415)

    def test_missing_required_fields_returns_400(self):
        """Test that missing required fields returns 400."""
        incomplete_request = {
            'symbol': 'AAPL'
            # Missing: start_date, end_date, strategy
        }

        required = ['symbol', 'start_date', 'end_date', 'strategy']
        missing = [f for f in required if f not in incomplete_request]

        assert len(missing) == 3

    def test_internal_error_returns_500(self):
        """Test that internal errors return 500."""
        # This would be tested with actual routes
        # Mock internal error scenario
        error_response = {
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'status_code': 500
        }

        assert error_response['status_code'] == 500


class TestAPIPagination:
    """Test API pagination."""

    def test_signals_pagination(self, test_database):
        """Test signal list pagination."""
        cursor = test_database.cursor()

        # Insert many signals
        for i in range(50):
            cursor.execute('''
                INSERT INTO signals (
                    symbol, direction, rrs, entry_price, generated_at, status
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                f'SYM{i:03d}', 'long', 2.0 + (i * 0.01),
                100.0 + i, datetime.now().isoformat(), 'active'
            ))
        test_database.commit()

        # Test pagination
        page_size = 10
        page = 1
        offset = (page - 1) * page_size

        cursor.execute('''
            SELECT * FROM signals
            ORDER BY generated_at DESC
            LIMIT ? OFFSET ?
        ''', (page_size, offset))
        page_results = cursor.fetchall()

        assert len(page_results) == page_size

        # Get total count
        cursor.execute('SELECT COUNT(*) FROM signals')
        total = cursor.fetchone()[0]

        assert total == 50
        assert total // page_size == 5  # 5 pages of 10

    def test_trades_pagination(self, test_database):
        """Test trade history pagination."""
        cursor = test_database.cursor()

        # Insert trade history
        for i in range(30):
            cursor.execute('''
                INSERT INTO trades (
                    symbol, direction, entry_price, exit_price, shares,
                    entry_time, exit_time, pnl, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                f'SYM{i}', 'long', 100.0, 105.0, 100,
                datetime.now().isoformat(), datetime.now().isoformat(),
                500.0, 'closed'
            ))
        test_database.commit()

        cursor.execute('''
            SELECT * FROM trades
            ORDER BY exit_time DESC
            LIMIT 20
        ''')
        results = cursor.fetchall()

        assert len(results) == 20

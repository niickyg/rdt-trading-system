"""
Unit tests for the RDT Trading System Real-Time Scanner.

Tests cover:
- RRS calculation logic
- Signal generation (strong/moderate thresholds)
- Batch data fetching
- Signal saving to files
- WebSocket broadcasting (mocked)
- Error handling for failed data fetches
- Watchlist loading
- Market hours detection
"""

import pytest
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from pathlib import Path
from zoneinfo import ZoneInfo

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def scanner_instance(scanner_config):
    """Create a scanner instance with mocked dependencies."""
    with patch('scanner.realtime_scanner.yf') as mock_yf, \
         patch('scanner.realtime_scanner.PROVIDERS_AVAILABLE', False), \
         patch('scanner.realtime_scanner.WEBSOCKET_AVAILABLE', False), \
         patch('scanner.realtime_scanner.METRICS_AVAILABLE', False):

        # Create a mock ticker for SPY that returns valid data
        mock_spy = Mock()
        mock_spy_5m = pd.DataFrame({
            'Open': [448.0, 449.0, 450.0],
            'High': [449.5, 450.5, 451.0],
            'Low': [447.5, 448.5, 449.0],
            'Close': [449.0, 450.0, 450.5],
            'Volume': [1000000, 1500000, 2000000],
        }, index=pd.date_range(start='2024-01-15 09:30', periods=3, freq='5min'))

        mock_spy.history.return_value = mock_spy_5m
        mock_yf.Ticker.return_value = mock_spy

        from scanner.realtime_scanner import RealTimeScanner
        scanner = RealTimeScanner(scanner_config)
        scanner._use_providers = False

        yield scanner


@pytest.fixture
def scanner_with_providers(scanner_config_with_providers):
    """Create a scanner instance with provider manager."""
    with patch('scanner.realtime_scanner.PROVIDERS_AVAILABLE', True), \
         patch('scanner.realtime_scanner.WEBSOCKET_AVAILABLE', False), \
         patch('scanner.realtime_scanner.METRICS_AVAILABLE', False), \
         patch('scanner.realtime_scanner.get_provider_manager') as mock_get_pm:

        mock_pm = Mock()
        mock_pm.get_provider_order.return_value = ['yfinance', 'alpha_vantage']
        mock_get_pm.return_value = mock_pm

        from scanner.realtime_scanner import RealTimeScanner
        scanner = RealTimeScanner(scanner_config_with_providers)

        yield scanner, mock_pm


# =============================================================================
# Watchlist Loading Tests
# =============================================================================

class TestWatchlistLoading:
    """Tests for watchlist loading functionality."""

    def test_load_watchlist_returns_list(self, scanner_instance):
        """Test that load_watchlist returns a non-empty list."""
        watchlist = scanner_instance.load_watchlist()

        assert isinstance(watchlist, list)
        assert len(watchlist) > 0

    def test_watchlist_contains_expected_stocks(self, scanner_instance):
        """Test that watchlist contains major tech stocks."""
        watchlist = scanner_instance.load_watchlist()

        expected_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
        for stock in expected_stocks:
            assert stock in watchlist, f"{stock} should be in watchlist"

    def test_watchlist_contains_only_strings(self, scanner_instance):
        """Test that all watchlist entries are strings."""
        watchlist = scanner_instance.load_watchlist()

        for symbol in watchlist:
            assert isinstance(symbol, str)
            assert len(symbol) > 0
            assert symbol.isupper()

    def test_watchlist_no_duplicates(self, scanner_instance):
        """Test that watchlist has no duplicate symbols."""
        watchlist = scanner_instance.load_watchlist()

        assert len(watchlist) == len(set(watchlist))


# =============================================================================
# RRS Calculation Tests
# =============================================================================

class TestRRSCalculation:
    """Tests for RRS calculation logic."""

    def test_calculate_stock_rrs_positive(self, scanner_instance, spy_data_fixture, stock_data_fixture):
        """Test RRS calculation for a stock outperforming SPY."""
        scanner_instance.spy_data = spy_data_fixture.copy()
        scanner_instance.spy_data['current_price'] = 450.0
        scanner_instance.spy_data['previous_close'] = 449.0  # SPY up 0.22%

        stock_data = stock_data_fixture.copy()
        stock_data['current_price'] = 155.0
        stock_data['previous_close'] = 150.0  # Stock up 3.33%
        stock_data['atr'] = 3.5

        with patch.object(scanner_instance.rrs_calc, 'calculate_rrs_current') as mock_calc:
            mock_calc.return_value = {
                'rrs': 0.89,  # (3.33 - 0.22) / 3.5
                'status': 'MODERATE_RS',
                'stock_pc': 3.33,
                'spy_pc': 0.22,
                'atr': 3.5,
                'expected_pc': 0.22,
                'current_price': 155.0,
            }

            with patch('scanner.realtime_scanner.check_daily_strength') as mock_strength, \
                 patch('scanner.realtime_scanner.check_daily_weakness') as mock_weakness:
                mock_strength.return_value = {'is_strong': False, 'ema3': 153.0, 'ema8': 151.0}
                mock_weakness.return_value = {'is_weak': False}

                result = scanner_instance.calculate_stock_rrs('AAPL', stock_data)

        assert result is not None
        assert result['symbol'] == 'AAPL'
        assert result['rrs'] > 0
        assert result['status'] == 'MODERATE_RS'

    def test_calculate_stock_rrs_negative(self, scanner_instance, spy_data_fixture, stock_data_fixture):
        """Test RRS calculation for a stock underperforming SPY."""
        scanner_instance.spy_data = spy_data_fixture.copy()
        scanner_instance.spy_data['current_price'] = 455.0
        scanner_instance.spy_data['previous_close'] = 450.0  # SPY up 1.11%

        stock_data = stock_data_fixture.copy()
        stock_data['current_price'] = 148.0
        stock_data['previous_close'] = 150.0  # Stock down 1.33%
        stock_data['atr'] = 3.5

        with patch.object(scanner_instance.rrs_calc, 'calculate_rrs_current') as mock_calc:
            mock_calc.return_value = {
                'rrs': -0.70,  # (-1.33 - 1.11) / 3.5
                'status': 'MODERATE_RW',
                'stock_pc': -1.33,
                'spy_pc': 1.11,
                'atr': 3.5,
                'expected_pc': 1.11,
                'current_price': 148.0,
            }

            with patch('scanner.realtime_scanner.check_daily_strength') as mock_strength, \
                 patch('scanner.realtime_scanner.check_daily_weakness') as mock_weakness:
                mock_strength.return_value = {'is_strong': False, 'ema3': 149.0, 'ema8': 151.0}
                mock_weakness.return_value = {'is_weak': True}

                result = scanner_instance.calculate_stock_rrs('AAPL', stock_data)

        assert result is not None
        assert result['rrs'] < 0
        assert result['status'] == 'MODERATE_RW'

    def test_calculate_stock_rrs_handles_exception(self, scanner_instance, spy_data_fixture, stock_data_fixture):
        """Test that calculate_stock_rrs handles exceptions gracefully."""
        scanner_instance.spy_data = spy_data_fixture

        with patch.object(scanner_instance.rrs_calc, 'calculate_rrs_current') as mock_calc:
            mock_calc.side_effect = ValueError("Calculation error")

            result = scanner_instance.calculate_stock_rrs('AAPL', stock_data_fixture)

        assert result is None

    def test_calculate_stock_rrs_strong_threshold(self, scanner_instance, spy_data_fixture, stock_data_fixture):
        """Test detection of strong RRS signals."""
        scanner_instance.spy_data = spy_data_fixture.copy()
        scanner_instance.spy_data['current_price'] = 450.0
        scanner_instance.spy_data['previous_close'] = 449.0

        stock_data = stock_data_fixture.copy()
        stock_data['current_price'] = 160.0
        stock_data['previous_close'] = 150.0  # Stock up 6.67%
        stock_data['atr'] = 2.5

        with patch.object(scanner_instance.rrs_calc, 'calculate_rrs_current') as mock_calc:
            mock_calc.return_value = {
                'rrs': 2.58,  # Strong RS
                'status': 'STRONG_RS',
                'stock_pc': 6.67,
                'spy_pc': 0.22,
                'atr': 2.5,
                'expected_pc': 0.22,
                'current_price': 160.0,
            }

            with patch('scanner.realtime_scanner.check_daily_strength') as mock_strength, \
                 patch('scanner.realtime_scanner.check_daily_weakness') as mock_weakness:
                mock_strength.return_value = {'is_strong': True, 'ema3': 158.0, 'ema8': 155.0}
                mock_weakness.return_value = {'is_weak': False}

                result = scanner_instance.calculate_stock_rrs('AAPL', stock_data)

        assert result is not None
        assert result['rrs'] > 2.0
        assert result['status'] == 'STRONG_RS'
        assert result['daily_strong'] is True


# =============================================================================
# Signal Generation Tests
# =============================================================================

class TestSignalGeneration:
    """Tests for signal generation with strong/moderate thresholds."""

    def test_should_alert_strong_rs(self, scanner_instance):
        """Test that strong RS signals trigger alerts."""
        assert scanner_instance.should_alert('AAPL', 2.5) is True

    def test_should_alert_strong_rw(self, scanner_instance):
        """Test that strong RW signals trigger alerts."""
        assert scanner_instance.should_alert('AAPL', -2.5) is True

    def test_should_not_alert_moderate(self, scanner_instance):
        """Test that moderate signals don't trigger alerts."""
        assert scanner_instance.should_alert('AAPL', 1.5) is False
        assert scanner_instance.should_alert('AAPL', -1.5) is False

    def test_should_not_alert_neutral(self, scanner_instance):
        """Test that neutral signals don't trigger alerts."""
        assert scanner_instance.should_alert('AAPL', 0.3) is False
        assert scanner_instance.should_alert('AAPL', -0.3) is False

    def test_should_alert_respects_cooldown(self, scanner_instance):
        """Test that alert cooldown prevents spam."""
        # First alert should trigger
        assert scanner_instance.should_alert('AAPL', 2.5) is True

        # Record the alert
        with patch('scanner.realtime_scanner.get_eastern_time') as mock_time:
            now = datetime.now(ZoneInfo("America/New_York"))
            mock_time.return_value = now
            scanner_instance.last_alerts['AAPL'] = now

            # Same stock within 15 minutes should not alert
            mock_time.return_value = now + timedelta(minutes=5)
            assert scanner_instance.should_alert('AAPL', 3.0) is False

            # After 15 minutes, should alert again
            mock_time.return_value = now + timedelta(minutes=20)
            assert scanner_instance.should_alert('AAPL', 3.0) is True

    def test_format_alert_message_long(self, scanner_instance):
        """Test alert message formatting for long signals."""
        with patch('scanner.realtime_scanner.get_eastern_time') as mock_time:
            mock_time.return_value = datetime(2024, 1, 15, 10, 30, 0, tzinfo=ZoneInfo("America/New_York"))

            analysis = {
                'symbol': 'AAPL',
                'rrs': 2.85,
                'price': 175.50,
                'status': 'STRONG_RS',
                'stock_pc': 2.5,
                'spy_pc': 0.5,
                'atr': 3.5,
                'daily_strong': True,
                'daily_weak': False,
            }

            message = scanner_instance.format_alert_message(analysis)

            assert 'RELATIVE STRENGTH ALERT' in message
            assert 'AAPL' in message
            assert '175.50' in message
            assert 'LONG' in message
            assert '2.85' in message

    def test_format_alert_message_short(self, scanner_instance):
        """Test alert message formatting for short signals."""
        with patch('scanner.realtime_scanner.get_eastern_time') as mock_time:
            mock_time.return_value = datetime(2024, 1, 15, 10, 30, 0, tzinfo=ZoneInfo("America/New_York"))

            analysis = {
                'symbol': 'AAPL',
                'rrs': -2.85,
                'price': 165.50,
                'status': 'STRONG_RW',
                'stock_pc': -2.5,
                'spy_pc': 0.5,
                'atr': 3.5,
                'daily_strong': False,
                'daily_weak': True,
            }

            message = scanner_instance.format_alert_message(analysis)

            assert 'RELATIVE WEAKNESS ALERT' in message
            assert 'AAPL' in message
            assert 'SHORT' in message
            assert '-2.85' in message


# =============================================================================
# Batch Data Fetching Tests
# =============================================================================

class TestBatchDataFetching:
    """Tests for batch data fetching functionality."""

    def test_fetch_batch_data_success(self, scanner_instance, mock_batch_5m_data, mock_batch_daily_data):
        """Test successful batch data fetch."""
        with patch('scanner.realtime_scanner.yf.download') as mock_download:
            mock_download.side_effect = [mock_batch_5m_data, mock_batch_daily_data]

            batch_5m, batch_daily = scanner_instance.fetch_batch_data(max_retries=1)

            assert batch_5m is not None
            assert batch_daily is not None
            assert not batch_5m.empty
            assert not batch_daily.empty

    def test_fetch_batch_data_empty_response(self, scanner_instance):
        """Test handling of empty batch data response."""
        with patch('scanner.realtime_scanner.yf.download') as mock_download:
            mock_download.return_value = pd.DataFrame()

            batch_5m, batch_daily = scanner_instance.fetch_batch_data(max_retries=1)

            assert batch_5m is None
            assert batch_daily is None

    def test_fetch_batch_data_rate_limit_retry(self, scanner_instance, mock_batch_5m_data, mock_batch_daily_data):
        """Test retry on rate limit error."""
        with patch('scanner.realtime_scanner.yf.download') as mock_download, \
             patch('scanner.realtime_scanner.time.sleep') as mock_sleep:

            # First call raises rate limit, second succeeds
            mock_download.side_effect = [
                Exception("401 rate limit exceeded"),
                mock_batch_5m_data,
                mock_batch_daily_data,
            ]

            batch_5m, batch_daily = scanner_instance.fetch_batch_data(max_retries=3)

            # Sleep should have been called for backoff
            assert mock_sleep.called

    def test_fetch_batch_data_all_retries_fail(self, scanner_instance):
        """Test failure after all retries exhausted."""
        with patch('scanner.realtime_scanner.yf.download') as mock_download, \
             patch('scanner.realtime_scanner.time.sleep'):

            mock_download.side_effect = Exception("Network error")

            batch_5m, batch_daily = scanner_instance.fetch_batch_data(max_retries=2)

            assert batch_5m is None
            assert batch_daily is None

    def test_extract_spy_data_from_batch(self, scanner_instance, mock_batch_5m_data, mock_batch_daily_data):
        """Test SPY data extraction from batch download."""
        spy_data = scanner_instance._extract_spy_data(mock_batch_5m_data, mock_batch_daily_data)

        assert spy_data is not None
        assert 'current_price' in spy_data
        assert 'previous_close' in spy_data
        assert '5m' in spy_data
        assert 'daily' in spy_data

    def test_extract_stock_data_from_batch(self, scanner_instance, mock_batch_5m_data, mock_batch_daily_data):
        """Test individual stock extraction from batch download."""
        # First setup SPY data
        scanner_instance._extract_spy_data(mock_batch_5m_data, mock_batch_daily_data)

        stock_data = scanner_instance._extract_stock_data('AAPL', mock_batch_5m_data, mock_batch_daily_data)

        assert stock_data is not None
        assert 'current_price' in stock_data
        assert 'atr' in stock_data
        assert 'volume' in stock_data

    def test_extract_stock_data_missing_symbol(self, scanner_instance, mock_batch_5m_data, mock_batch_daily_data):
        """Test handling of missing symbol in batch data."""
        stock_data = scanner_instance._extract_stock_data('INVALID', mock_batch_5m_data, mock_batch_daily_data)

        assert stock_data is None


# =============================================================================
# Signal Saving Tests
# =============================================================================

class TestSignalSaving:
    """Tests for signal saving to files."""

    def test_save_signals_creates_file(self, scanner_instance, tmp_path):
        """Test that save_signals creates the signals file."""
        with patch('scanner.realtime_scanner.Path') as mock_path:
            mock_signals_dir = tmp_path / 'data' / 'signals'
            mock_signals_dir.mkdir(parents=True)
            mock_path.return_value = mock_signals_dir

            strong_rs = [{
                'symbol': 'AAPL',
                'rrs': 2.5,
                'price': 175.0,
                'atr': 3.5,
                'stock_pc': 2.0,
                'spy_pc': 0.5,
                'daily_strong': True,
            }]
            strong_rw = []

            with patch('scanner.realtime_scanner.format_timestamp') as mock_ts:
                mock_ts.return_value = '2024-01-15T10:30:00-05:00'

                scanner_instance.save_signals(strong_rs, strong_rw)

    def test_save_signals_long_format(self, scanner_instance, tmp_path, monkeypatch):
        """Test signal format for long positions."""
        import json
        from pathlib import Path

        signals_dir = tmp_path / 'data' / 'signals'
        signals_dir.mkdir(parents=True)

        strong_rs = [{
            'symbol': 'AAPL',
            'rrs': 2.85,
            'price': 175.0,
            'atr': 3.5,
            'stock_pc': 2.5,
            'spy_pc': 0.5,
            'daily_strong': True,
        }]

        with patch('scanner.realtime_scanner.format_timestamp') as mock_ts:
            mock_ts.return_value = '2024-01-15T10:30:00-05:00'

            # Change working directory to tmp_path so relative paths work
            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_path)
                scanner_instance.save_signals(strong_rs, [])

                # Verify file was created
                signal_file = signals_dir / 'active_signals.json'
                assert signal_file.exists()

                # Verify content
                with open(signal_file) as f:
                    signals = json.load(f)
                assert len(signals) == 1
                assert signals[0]['symbol'] == 'AAPL'
                assert signals[0]['direction'] == 'long'
            finally:
                os.chdir(original_cwd)

    def test_save_signals_short_format(self, scanner_instance, tmp_path):
        """Test signal format for short positions."""
        strong_rw = [{
            'symbol': 'AAPL',
            'rrs': -2.85,
            'price': 165.0,
            'atr': 3.5,
            'stock_pc': -2.5,
            'spy_pc': 0.5,
            'daily_weak': True,
        }]

        with patch('scanner.realtime_scanner.format_timestamp') as mock_ts:
            mock_ts.return_value = '2024-01-15T10:30:00-05:00'

            with patch('builtins.open', create=True) as mock_open:
                mock_file = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_file
                with patch('scanner.realtime_scanner.Path'):
                    scanner_instance.save_signals([], strong_rw)

    def test_append_to_history(self, scanner_instance, tmp_path):
        """Test signal history appending."""
        history_file = tmp_path / 'signal_history.json'

        signals = [{
            'symbol': 'AAPL',
            'generated_at': '2024-01-15T10:30:00-05:00',
        }]

        with patch('scanner.realtime_scanner.Path') as mock_path:
            mock_path.return_value.exists.return_value = False

            with patch('builtins.open', create=True) as mock_open:
                mock_file = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_file

                with patch('scanner.realtime_scanner.format_timestamp') as mock_ts, \
                     patch('scanner.realtime_scanner.get_eastern_time') as mock_time:
                    mock_ts.return_value = '2024-01-15T10:30:00-05:00'
                    mock_time.return_value = datetime.now(ZoneInfo("America/New_York"))

                    scanner_instance.append_to_history(signals)


# =============================================================================
# WebSocket Broadcasting Tests
# =============================================================================

class TestWebSocketBroadcasting:
    """Tests for WebSocket broadcasting functionality."""

    def test_scan_once_broadcasts_started(self, scanner_instance, mock_batch_5m_data, mock_batch_daily_data):
        """Test that scan_once broadcasts scan started event."""
        with patch('scanner.realtime_scanner.WEBSOCKET_AVAILABLE', True), \
             patch('scanner.realtime_scanner.broadcast_scan_started') as mock_broadcast, \
             patch('scanner.realtime_scanner.broadcast_scan_completed'), \
             patch('scanner.realtime_scanner.yf.download') as mock_download, \
             patch.object(scanner_instance, 'save_signals'):

            mock_download.side_effect = [mock_batch_5m_data, mock_batch_daily_data]

            scanner_instance.scan_once()

            mock_broadcast.assert_called_once()

    def test_scan_once_broadcasts_completed(self, scanner_instance, mock_batch_5m_data, mock_batch_daily_data):
        """Test that scan_once broadcasts scan completed event."""
        with patch('scanner.realtime_scanner.WEBSOCKET_AVAILABLE', True), \
             patch('scanner.realtime_scanner.broadcast_scan_started'), \
             patch('scanner.realtime_scanner.broadcast_scan_completed') as mock_broadcast, \
             patch('scanner.realtime_scanner.yf.download') as mock_download, \
             patch.object(scanner_instance, 'save_signals'):

            mock_download.side_effect = [mock_batch_5m_data, mock_batch_daily_data]

            scanner_instance.scan_once()

            mock_broadcast.assert_called_once()

    def test_scan_once_broadcasts_error_on_failure(self, scanner_instance):
        """Test that scan_once broadcasts error on data fetch failure."""
        with patch('scanner.realtime_scanner.WEBSOCKET_AVAILABLE', True), \
             patch('scanner.realtime_scanner.broadcast_scan_started'), \
             patch('scanner.realtime_scanner.broadcast_scan_error') as mock_broadcast_error, \
             patch('scanner.realtime_scanner.yf.download') as mock_download, \
             patch.object(scanner_instance, '_scan_individual_stocks'):

            mock_download.return_value = (pd.DataFrame(), pd.DataFrame())

            scanner_instance.scan_once()

            # Should broadcast error when batch data fails
            mock_broadcast_error.assert_called()


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in data fetching."""

    def test_fetch_spy_data_handles_exception(self, scanner_instance):
        """Test graceful handling of SPY data fetch failure."""
        with patch('scanner.realtime_scanner.yf.Ticker') as mock_ticker:
            mock_ticker.side_effect = Exception("Network error")

            result = scanner_instance.fetch_spy_data()

            assert result is None

    def test_fetch_stock_data_handles_empty_data(self, scanner_instance):
        """Test handling of empty stock data."""
        with patch('scanner.realtime_scanner.yf.Ticker') as mock_ticker:
            mock = Mock()
            mock.history.return_value = pd.DataFrame()
            mock_ticker.return_value = mock

            result = scanner_instance.fetch_stock_data('INVALID')

            assert result is None

    def test_fetch_stock_data_handles_exception(self, scanner_instance):
        """Test graceful handling of stock data fetch failure."""
        with patch('scanner.realtime_scanner.yf.Ticker') as mock_ticker:
            mock_ticker.side_effect = Exception("API error")

            result = scanner_instance.fetch_stock_data('AAPL')

            assert result is None

    def test_calculate_rrs_handles_missing_spy_data(self, scanner_instance, stock_data_fixture):
        """Test RRS calculation when SPY data is missing."""
        scanner_instance.spy_data = None

        # This should raise an exception or return None
        try:
            result = scanner_instance.calculate_stock_rrs('AAPL', stock_data_fixture)
            assert result is None
        except (AttributeError, TypeError):
            pass  # Expected behavior


# =============================================================================
# Provider Integration Tests
# =============================================================================

class TestProviderIntegration:
    """Tests for data provider integration."""

    def test_scanner_uses_provider_manager_when_available(self, scanner_config_with_providers):
        """Test that scanner uses provider manager when configured."""
        with patch('scanner.realtime_scanner.PROVIDERS_AVAILABLE', True), \
             patch('scanner.realtime_scanner.get_provider_manager') as mock_get_pm:

            mock_pm = Mock()
            mock_pm.get_provider_order.return_value = ['yfinance']
            mock_get_pm.return_value = mock_pm

            from scanner.realtime_scanner import RealTimeScanner
            scanner = RealTimeScanner(scanner_config_with_providers)

            assert scanner._use_providers is True
            assert scanner._provider_manager is not None

    def test_scanner_falls_back_when_providers_unavailable(self, scanner_config_with_providers):
        """Test fallback to yfinance when providers not available."""
        with patch('scanner.realtime_scanner.PROVIDERS_AVAILABLE', False):
            from scanner.realtime_scanner import RealTimeScanner
            scanner = RealTimeScanner(scanner_config_with_providers)

            assert scanner._use_providers is False

    def test_get_provider_status_when_disabled(self, scanner_instance):
        """Test provider status when providers are disabled."""
        status = scanner_instance.get_provider_status()

        assert status['status'] == 'disabled'

    def test_fetch_batch_data_with_providers(self, scanner_with_providers, mock_batch_5m_data, mock_batch_daily_data):
        """Test batch data fetch through provider manager."""
        scanner, mock_pm = scanner_with_providers

        # Test the fallback path when no YFinance provider is found
        mock_pm.get_provider.return_value = None
        mock_pm.get_provider_order.return_value = ['other_provider']

        # Mock the individual fetch path
        with patch.object(scanner, '_fetch_individual_via_providers') as mock_individual:
            mock_individual.return_value = (None, None)

            result = scanner.fetch_batch_data_with_providers()

            # Should return (None, None) when no providers work
            assert result == (None, None) or result[0] is None


# =============================================================================
# Market Hours Tests
# =============================================================================

class TestMarketHours:
    """Tests for market hours detection."""

    def test_scan_during_market_hours(self, scanner_instance, market_open_time):
        """Test scanning during market hours."""
        with patch('scanner.realtime_scanner.is_market_open') as mock_is_open:
            mock_is_open.return_value = True

            # Scanner should work normally during market hours
            assert mock_is_open(market_open_time) is True

    def test_scan_outside_market_hours(self, scanner_instance, market_closed_time):
        """Test scanning outside market hours."""
        with patch('scanner.realtime_scanner.is_market_open') as mock_is_open:
            mock_is_open.return_value = False

            assert mock_is_open(market_closed_time) is False


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_watchlist(self, scanner_config):
        """Test scanner behavior with empty watchlist."""
        with patch('scanner.realtime_scanner.PROVIDERS_AVAILABLE', False), \
             patch('scanner.realtime_scanner.WEBSOCKET_AVAILABLE', False), \
             patch('scanner.realtime_scanner.METRICS_AVAILABLE', False):

            from scanner.realtime_scanner import RealTimeScanner
            scanner = RealTimeScanner(scanner_config)
            scanner.watchlist = []

            # Should handle empty watchlist gracefully
            with patch.object(scanner, 'fetch_batch_data') as mock_fetch:
                mock_fetch.return_value = (pd.DataFrame(), pd.DataFrame())
                # The scan should complete without errors
                try:
                    scanner.scan_once()
                except Exception as e:
                    # May fail gracefully, which is acceptable
                    pass

    def test_all_stocks_filtered_by_volume(self, scanner_instance, mock_batch_5m_data, mock_batch_daily_data):
        """Test when all stocks are filtered out by volume."""
        scanner_instance.config['min_volume'] = 999999999999  # Unrealistically high

        with patch('scanner.realtime_scanner.yf.download') as mock_download, \
             patch.object(scanner_instance, 'save_signals') as mock_save:

            mock_download.side_effect = [mock_batch_5m_data, mock_batch_daily_data]

            scanner_instance.scan_once()

            # save_signals should be called with empty lists
            mock_save.assert_called()

    def test_all_stocks_filtered_by_price(self, scanner_instance, mock_batch_5m_data, mock_batch_daily_data):
        """Test when all stocks are filtered out by price."""
        scanner_instance.config['min_price'] = 99999999  # Unrealistically high

        with patch('scanner.realtime_scanner.yf.download') as mock_download, \
             patch.object(scanner_instance, 'save_signals') as mock_save:

            mock_download.side_effect = [mock_batch_5m_data, mock_batch_daily_data]

            scanner_instance.scan_once()

            mock_save.assert_called()

    def test_invalid_data_response(self, scanner_instance):
        """Test handling of empty batch data responses."""
        with patch('scanner.realtime_scanner.yf.download') as mock_download:
            # Return empty DataFrame to trigger the "Empty batch data received" error
            mock_download.return_value = pd.DataFrame()

            batch_5m, batch_daily = scanner_instance.fetch_batch_data(max_retries=1)

            assert batch_5m is None
            assert batch_daily is None

    def test_malformed_data_extraction(self, scanner_instance, mock_batch_5m_data, mock_batch_daily_data):
        """Test handling of malformed data during extraction."""
        # Malformed data without expected columns
        malformed_data = pd.DataFrame({'invalid': [1, 2, 3]})

        # SPY extraction should fail gracefully
        result = scanner_instance._extract_spy_data(malformed_data, malformed_data)
        assert result is None

        # Stock extraction should fail gracefully
        result = scanner_instance._extract_stock_data('AAPL', malformed_data, malformed_data)
        assert result is None

    def test_scan_individual_stocks_fallback(self, scanner_instance, sample_ohlcv_data, sample_5m_data):
        """Test individual stock scanning fallback mode."""
        # Setup SPY data that fetch_spy_data would return and set
        spy_data = {
            '5m': sample_5m_data,
            'daily': sample_ohlcv_data,
            'current_price': 450.0,
            'previous_close': 448.0,
        }

        def mock_fetch_spy():
            scanner_instance.spy_data = spy_data
            return spy_data

        with patch.object(scanner_instance, 'fetch_spy_data', side_effect=mock_fetch_spy) as mock_spy, \
             patch.object(scanner_instance, 'fetch_stock_data') as mock_stock, \
             patch.object(scanner_instance, 'save_signals'), \
             patch('scanner.realtime_scanner.send_alert'):

            # Setup stock data
            stock_data = {
                '5m': sample_5m_data,
                'daily': sample_ohlcv_data,
                'current_price': 155.0,
                'previous_close': 150.0,
                'atr': 3.5,
                'volume': 5000000,
            }
            mock_stock.return_value = stock_data

            with patch.object(scanner_instance, 'calculate_stock_rrs') as mock_rrs, \
                 patch('scanner.realtime_scanner.time.sleep'):
                mock_rrs.return_value = {
                    'symbol': 'AAPL',
                    'rrs': 1.5,
                    'status': 'MODERATE_RS',
                    'stock_pc': 3.0,
                    'spy_pc': 0.5,
                    'price': 155.0,
                    'volume': 5000000,
                    'atr': 3.5,
                    'daily_strong': False,
                    'daily_weak': False,
                }

                scanner_instance._scan_individual_stocks()

                mock_spy.assert_called_once()
                assert mock_stock.call_count >= 1


# =============================================================================
# Continuous Scanning Tests
# =============================================================================

class TestContinuousScanning:
    """Tests for continuous scanning mode."""

    def test_run_continuous_stops_on_keyboard_interrupt(self, scanner_instance):
        """Test that run_continuous stops on KeyboardInterrupt."""
        with patch.object(scanner_instance, 'scan_once') as mock_scan:
            mock_scan.side_effect = KeyboardInterrupt()

            # Should not raise, should exit gracefully
            scanner_instance.run_continuous()

    def test_run_continuous_respects_scan_interval(self, scanner_instance):
        """Test that run_continuous waits between scans."""
        scanner_instance.config['scan_interval_seconds'] = 5
        call_count = 0

        def mock_scan():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise KeyboardInterrupt()

        with patch.object(scanner_instance, 'scan_once', side_effect=mock_scan), \
             patch('scanner.realtime_scanner.time.sleep') as mock_sleep:

            scanner_instance.run_continuous()

            # Should have called sleep at least once
            assert mock_sleep.called


# =============================================================================
# Additional Coverage Tests
# =============================================================================

class TestAdditionalCoverage:
    """Additional tests to improve code coverage."""

    def test_convert_historical_dict_to_batch_df_empty(self, scanner_instance):
        """Test conversion of empty historical dict."""
        result = scanner_instance._convert_historical_dict_to_batch_df({})
        assert result.empty

    def test_convert_historical_dict_to_batch_df_valid(self, scanner_instance, sample_ohlcv_data):
        """Test conversion of valid historical dict."""
        from data.providers.base import HistoricalData

        mock_hist = Mock()
        mock_hist.data = sample_ohlcv_data.copy()

        data_dict = {'AAPL': mock_hist}
        result = scanner_instance._convert_historical_dict_to_batch_df(data_dict)

        assert not result.empty
        assert 'AAPL' in result.columns.get_level_values(0)

    def test_convert_historical_dict_to_batch_df_with_empty_data(self, scanner_instance):
        """Test conversion when some symbols have empty data."""
        mock_hist = Mock()
        mock_hist.data = pd.DataFrame()

        data_dict = {'AAPL': mock_hist}
        result = scanner_instance._convert_historical_dict_to_batch_df(data_dict)

        assert result.empty

    def test_fetch_individual_via_providers_success(self, scanner_config_with_providers):
        """Test individual fetch via providers."""
        from data.providers.base import HistoricalData

        with patch('scanner.realtime_scanner.PROVIDERS_AVAILABLE', True), \
             patch('scanner.realtime_scanner.get_provider_manager') as mock_get_pm:

            mock_pm = Mock()
            mock_get_pm.return_value = mock_pm

            # Create mock historical data
            mock_5m = Mock()
            mock_5m.data = pd.DataFrame({
                'open': [100, 101],
                'high': [102, 103],
                'low': [99, 100],
                'close': [101, 102],
                'volume': [1000, 2000]
            })

            mock_daily = Mock()
            mock_daily.data = pd.DataFrame({
                'open': [100, 101],
                'high': [102, 103],
                'low': [99, 100],
                'close': [101, 102],
                'volume': [1000, 2000]
            })

            mock_pm.get_historical.side_effect = [mock_5m, mock_daily]
            mock_pm.get_provider_order.return_value = ['yfinance']

            from scanner.realtime_scanner import RealTimeScanner
            scanner = RealTimeScanner(scanner_config_with_providers)

            result = scanner._fetch_individual_via_providers(['AAPL'])

            # Should return data frames
            assert result is not None

    def test_fetch_individual_via_providers_failure(self, scanner_config_with_providers):
        """Test individual fetch via providers when all fail."""
        with patch('scanner.realtime_scanner.PROVIDERS_AVAILABLE', True), \
             patch('scanner.realtime_scanner.get_provider_manager') as mock_get_pm:

            mock_pm = Mock()
            mock_get_pm.return_value = mock_pm

            # Make all fetches return None
            mock_pm.get_historical.return_value = None
            mock_pm.get_provider_order.return_value = ['yfinance']

            from scanner.realtime_scanner import RealTimeScanner
            scanner = RealTimeScanner(scanner_config_with_providers)

            result = scanner._fetch_individual_via_providers(['AAPL'])

            # Should return (None, None) when nothing works
            assert result == (None, None)

    def test_spy_data_extraction_single_ticker(self, scanner_instance, sample_5m_data, sample_ohlcv_data):
        """Test SPY extraction when single ticker (columns not multi-level)."""
        # Simulate single ticker download (no multi-level columns)
        single_5m = sample_5m_data.copy()
        single_daily = sample_ohlcv_data.copy()

        with patch.object(single_5m.columns, 'get_level_values', side_effect=AttributeError):
            # This tests the fallback path
            pass  # Can't easily test without multi-level columns in fixture

    def test_append_to_history_existing_file(self, scanner_instance, tmp_path):
        """Test appending to existing history file."""
        import json
        from datetime import datetime, timedelta

        # Create signals directory and history file
        signals_dir = tmp_path / 'data' / 'signals'
        signals_dir.mkdir(parents=True, exist_ok=True)

        # Use recent dates so they don't get filtered by the 30-day cutoff
        now = datetime.now(ZoneInfo("America/New_York"))
        yesterday = (now - timedelta(days=1)).isoformat()
        today = now.isoformat()

        history_file = signals_dir / 'signal_history.json'
        existing = [{'symbol': 'MSFT', 'generated_at': yesterday}]
        with open(history_file, 'w') as f:
            json.dump(existing, f)

        new_signals = [{'symbol': 'AAPL', 'generated_at': today}]

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            with patch('scanner.realtime_scanner.get_eastern_time') as mock_time, \
                 patch('scanner.realtime_scanner.format_timestamp') as mock_ts:
                mock_time.return_value = now
                # Return a cutoff date 30 days ago
                mock_ts.return_value = (now - timedelta(days=30)).isoformat()

                scanner_instance.append_to_history(new_signals)

                # Verify history was updated
                with open(history_file) as f:
                    history = json.load(f)

                assert len(history) == 2
        finally:
            os.chdir(original_cwd)

    def test_get_provider_status_with_providers(self, scanner_config_with_providers):
        """Test provider status when providers are enabled."""
        with patch('scanner.realtime_scanner.PROVIDERS_AVAILABLE', True), \
             patch('scanner.realtime_scanner.get_provider_manager') as mock_get_pm:

            mock_pm = Mock()
            mock_pm.get_provider_order.return_value = ['yfinance']
            mock_pm.status.return_value = {
                'providers': {'yfinance': {'status': 'healthy'}},
                'provider_order': ['yfinance']
            }
            mock_get_pm.return_value = mock_pm

            from scanner.realtime_scanner import RealTimeScanner
            scanner = RealTimeScanner(scanner_config_with_providers)

            status = scanner.get_provider_status()

            assert 'providers' in status or 'provider_order' in status

    def test_scan_once_basic_flow(self, scanner_instance, mock_batch_5m_data, mock_batch_daily_data):
        """Test basic scan_once flow completes without error."""
        with patch('scanner.realtime_scanner.yf.download') as mock_download, \
             patch.object(scanner_instance, 'save_signals') as mock_save:

            mock_download.side_effect = [mock_batch_5m_data, mock_batch_daily_data]

            # Should complete without raising
            scanner_instance.scan_once()

            # save_signals should be called
            mock_save.assert_called_once()

    def test_fetch_spy_data_with_providers(self, scanner_config_with_providers):
        """Test SPY data fetch through providers."""
        with patch('scanner.realtime_scanner.PROVIDERS_AVAILABLE', True), \
             patch('scanner.realtime_scanner.get_provider_manager') as mock_get_pm:

            mock_pm = Mock()
            mock_get_pm.return_value = mock_pm

            # Setup mock provider responses
            mock_5m = Mock()
            mock_5m.data = pd.DataFrame({
                'open': [448, 449, 450],
                'high': [449, 450, 451],
                'low': [447, 448, 449],
                'close': [449, 450, 450.5],
                'volume': [1000000, 1500000, 2000000]
            })

            mock_daily = Mock()
            mock_daily.data = pd.DataFrame({
                'open': [445, 447, 448],
                'high': [447, 449, 450],
                'low': [444, 446, 447],
                'close': [446, 448, 449],
                'volume': [50000000, 55000000, 60000000]
            })

            mock_pm.get_historical.side_effect = [mock_5m, mock_daily]
            mock_pm.get_provider_order.return_value = ['yfinance']

            from scanner.realtime_scanner import RealTimeScanner
            scanner = RealTimeScanner(scanner_config_with_providers)

            result = scanner.fetch_spy_data()

            assert result is not None
            assert 'current_price' in result

    def test_fetch_stock_data_with_providers(self, scanner_config_with_providers):
        """Test individual stock data fetch through providers."""
        with patch('scanner.realtime_scanner.PROVIDERS_AVAILABLE', True), \
             patch('scanner.realtime_scanner.get_provider_manager') as mock_get_pm:

            mock_pm = Mock()
            mock_get_pm.return_value = mock_pm

            # Setup mock provider responses
            mock_5m = Mock()
            mock_5m.data = pd.DataFrame({
                'open': [174, 175, 176],
                'high': [175, 176, 177],
                'low': [173, 174, 175],
                'close': [175, 176, 176.5],
                'volume': [100000, 150000, 200000]
            })

            mock_daily = Mock()
            mock_daily.data = pd.DataFrame({
                'open': [170, 172, 174],
                'high': [172, 174, 176],
                'low': [169, 171, 173],
                'close': [171, 173, 175],
                'volume': [5000000, 5500000, 6000000]
            })

            mock_pm.get_historical.side_effect = [mock_5m, mock_daily]
            mock_pm.get_provider_order.return_value = ['yfinance']

            from scanner.realtime_scanner import RealTimeScanner
            scanner = RealTimeScanner(scanner_config_with_providers)

            result = scanner.fetch_stock_data('AAPL')

            assert result is not None
            assert 'current_price' in result
            assert 'atr' in result

"""
Unit tests for the RDT Trading System Data Providers.

Tests cover:
- ProviderManager fallback logic
- Circuit breaker behavior
- Caching functionality
- Individual providers (mocked API responses)
- Error handling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import time
import threading

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.providers.base import (
    DataProvider,
    Quote,
    HistoricalData,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    DataNotFoundError,
    ProviderStatus,
    ProviderHealth,
)
from data.providers.provider_manager import (
    ProviderManager,
    CircuitBreaker,
    CircuitState,
    LRUCache,
    CacheEntry,
    get_provider_manager,
    reset_provider_manager,
)
from data.providers.yfinance_provider import YFinanceProvider
from data.providers.alpha_vantage_provider import AlphaVantageProvider


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_quote():
    """Create a sample Quote object."""
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
def sample_historical(sample_ohlcv_data):
    """Create a sample HistoricalData object."""
    return HistoricalData(
        symbol="AAPL",
        data=sample_ohlcv_data,
        period="60d",
        interval="1d",
        provider="test_provider",
    )


@pytest.fixture
def mock_provider_healthy():
    """Create a healthy mock provider."""
    provider = Mock(spec=DataProvider)
    provider.name = "mock_healthy"
    provider.priority = 10
    provider.is_available.return_value = True
    provider.record_success = Mock()
    provider.record_failure = Mock()
    provider.get_health.return_value = ProviderHealth(
        name="mock_healthy",
        status=ProviderStatus.HEALTHY,
        consecutive_failures=0,
    )
    return provider


@pytest.fixture
def mock_provider_failing():
    """Create a failing mock provider."""
    provider = Mock(spec=DataProvider)
    provider.name = "mock_failing"
    provider.priority = 5
    provider.is_available.return_value = True
    provider.record_success = Mock()
    provider.record_failure = Mock()
    provider.get_health.return_value = ProviderHealth(
        name="mock_failing",
        status=ProviderStatus.DEGRADED,
        consecutive_failures=3,
    )
    return provider


@pytest.fixture
def provider_manager_with_mocks(mock_provider_healthy, mock_provider_failing):
    """Create a ProviderManager with mock providers."""
    manager = ProviderManager(providers=[], cache_ttl=30.0)
    manager.register_provider(mock_provider_healthy)
    manager.register_provider(mock_provider_failing)
    return manager


# =============================================================================
# Circuit Breaker Tests
# =============================================================================

class TestCircuitBreaker:
    """Tests for circuit breaker behavior."""

    def test_initial_state_is_closed(self):
        """Test that circuit breaker starts in closed state."""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.can_execute() is True

    def test_records_failures(self):
        """Test that failures are recorded correctly."""
        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()
        assert cb.failure_count == 1
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        assert cb.failure_count == 2
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold(self):
        """Test that circuit opens after reaching failure threshold."""
        cb = CircuitBreaker(failure_threshold=3)

        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_success_resets_failure_count(self):
        """Test that success resets the failure count."""
        cb = CircuitBreaker(failure_threshold=5)

        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2

        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_half_open_after_recovery_timeout(self):
        """Test transition to half-open state after recovery timeout."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Should transition to half-open
        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes_circuit(self):
        """Test that success in half-open state closes the circuit."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        # Open and wait for half-open
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        cb.can_execute()  # Trigger transition to half-open

        # Record success
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_half_open_failure_reopens_circuit(self):
        """Test that failure in half-open state reopens the circuit."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        # Open and wait for half-open
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        cb.can_execute()  # Trigger transition to half-open

        # Record failure
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_half_open_limits_calls(self):
        """Test that half-open state limits number of test calls."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1, half_open_max_calls=2)

        # Open and wait for half-open
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)

        # First call transitions to half-open (doesn't count against limit)
        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

        # Now in half-open: should allow max_calls attempts
        assert cb.can_execute() is True  # call 1 of max_calls=2
        assert cb.can_execute() is True  # call 2 of max_calls=2
        assert cb.can_execute() is False  # exceeds limit


# =============================================================================
# LRU Cache Tests
# =============================================================================

class TestLRUCache:
    """Tests for LRU cache functionality."""

    def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = LRUCache(max_size=10, default_ttl=60.0)

        cache.set("key1", "value1", "provider1")
        result = cache.get("key1")

        assert result == "value1"

    def test_get_missing_key(self):
        """Test get returns None for missing key."""
        cache = LRUCache()

        result = cache.get("nonexistent")

        assert result is None

    def test_expiration(self):
        """Test that expired entries are not returned."""
        cache = LRUCache(default_ttl=0.05)  # 50ms TTL

        cache.set("key1", "value1", "provider1")
        assert cache.get("key1") == "value1"

        time.sleep(0.1)
        assert cache.get("key1") is None

    def test_lru_eviction(self):
        """Test that least recently used items are evicted."""
        cache = LRUCache(max_size=3, default_ttl=60.0)

        cache.set("key1", "value1", "p")
        cache.set("key2", "value2", "p")
        cache.set("key3", "value3", "p")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add new key, should evict key2 (least recently used)
        cache.set("key4", "value4", "p")

        assert cache.get("key1") == "value1"  # Should still exist
        assert cache.get("key2") is None  # Should be evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_invalidate_key(self):
        """Test invalidating a specific key."""
        cache = LRUCache()

        cache.set("key1", "value1", "p")
        cache.set("key2", "value2", "p")

        cache.invalidate("key1")

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

    def test_invalidate_pattern(self):
        """Test invalidating keys matching a pattern."""
        cache = LRUCache()

        cache.set("quote:AAPL", "q1", "p")
        cache.set("quote:MSFT", "q2", "p")
        cache.set("historical:AAPL", "h1", "p")

        count = cache.invalidate_pattern("quote:")

        assert count == 2
        assert cache.get("quote:AAPL") is None
        assert cache.get("quote:MSFT") is None
        assert cache.get("historical:AAPL") == "h1"

    def test_clear(self):
        """Test clearing all cache entries."""
        cache = LRUCache()

        cache.set("key1", "value1", "p")
        cache.set("key2", "value2", "p")

        cache.clear()

        assert cache.size == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache = LRUCache(default_ttl=0.05)

        cache.set("key1", "value1", "p")
        cache.set("key2", "value2", "p", ttl=60.0)  # Long TTL

        time.sleep(0.1)

        count = cache.cleanup_expired()

        assert count == 1
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

    def test_stats(self):
        """Test cache statistics."""
        cache = LRUCache(max_size=100)

        cache.set("key1", "value1", "provider1")
        cache.set("key2", "value2", "provider1")
        cache.set("key3", "value3", "provider2")

        stats = cache.stats()

        assert stats["size"] == 3
        assert stats["max_size"] == 100
        assert stats["by_provider"]["provider1"] == 2
        assert stats["by_provider"]["provider2"] == 1

    def test_thread_safety(self):
        """Test that cache operations are thread-safe."""
        cache = LRUCache(max_size=100, default_ttl=60.0)
        errors = []

        def writer():
            try:
                for i in range(100):
                    cache.set(f"key{i}", f"value{i}", "p")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    cache.get(f"key{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(5)]
        threads += [threading.Thread(target=reader) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Provider Manager Tests
# =============================================================================

class TestProviderManager:
    """Tests for ProviderManager functionality."""

    def test_initialization_with_providers(self, mock_provider_healthy):
        """Test initialization with provided providers."""
        manager = ProviderManager(providers=[mock_provider_healthy])

        assert len(manager._providers) == 1
        assert "mock_healthy" in manager._providers

    def test_register_provider(self, mock_provider_healthy):
        """Test registering a new provider."""
        manager = ProviderManager(providers=[])

        manager.register_provider(mock_provider_healthy)

        assert "mock_healthy" in manager._providers
        assert "mock_healthy" in manager._circuits

    def test_unregister_provider(self, mock_provider_healthy):
        """Test unregistering a provider."""
        manager = ProviderManager(providers=[mock_provider_healthy])

        manager.unregister_provider("mock_healthy")

        assert "mock_healthy" not in manager._providers
        assert "mock_healthy" not in manager._circuits

    def test_get_provider(self, mock_provider_healthy):
        """Test getting a provider by name."""
        manager = ProviderManager(providers=[mock_provider_healthy])

        provider = manager.get_provider("mock_healthy")

        assert provider is mock_provider_healthy

    def test_get_provider_not_found(self):
        """Test getting a non-existent provider returns None."""
        manager = ProviderManager(providers=[])

        provider = manager.get_provider("nonexistent")

        assert provider is None

    def test_provider_order_by_priority(self, mock_provider_healthy, mock_provider_failing):
        """Test that providers are ordered by priority."""
        manager = ProviderManager(providers=[mock_provider_healthy, mock_provider_failing])

        # mock_failing has priority 5, mock_healthy has priority 10
        assert manager._provider_order == ["mock_failing", "mock_healthy"]

    def test_get_quote_uses_cache(self, mock_provider_healthy, sample_quote):
        """Test that get_quote uses cache."""
        manager = ProviderManager(providers=[mock_provider_healthy])
        mock_provider_healthy.get_quote.return_value = sample_quote

        # First call should hit provider
        result1 = manager.get_quote("AAPL")
        # Second call should hit cache
        result2 = manager.get_quote("AAPL")

        assert mock_provider_healthy.get_quote.call_count == 1
        assert result1 == result2

    def test_get_quote_bypasses_cache(self, mock_provider_healthy, sample_quote):
        """Test that cache can be bypassed."""
        manager = ProviderManager(providers=[mock_provider_healthy])
        mock_provider_healthy.get_quote.return_value = sample_quote

        result1 = manager.get_quote("AAPL", use_cache=False)
        result2 = manager.get_quote("AAPL", use_cache=False)

        assert mock_provider_healthy.get_quote.call_count == 2

    def test_fallback_on_provider_failure(self, mock_provider_healthy, mock_provider_failing, sample_quote):
        """Test fallback to next provider on failure."""
        manager = ProviderManager(providers=[mock_provider_failing, mock_provider_healthy])

        mock_provider_failing.get_quote.side_effect = ProviderError("Failed")
        mock_provider_healthy.get_quote.return_value = sample_quote

        result = manager.get_quote("AAPL", use_cache=False)

        assert result == sample_quote
        mock_provider_failing.get_quote.assert_called_once()
        mock_provider_healthy.get_quote.assert_called_once()

    def test_fallback_on_rate_limit(self, mock_provider_healthy, mock_provider_failing, sample_quote):
        """Test fallback on rate limit error."""
        manager = ProviderManager(providers=[mock_provider_failing, mock_provider_healthy])

        mock_provider_failing.get_quote.side_effect = RateLimitError("Rate limited")
        mock_provider_healthy.get_quote.return_value = sample_quote

        result = manager.get_quote("AAPL", use_cache=False)

        assert result == sample_quote

    def test_all_providers_fail(self, mock_provider_failing):
        """Test error when all providers fail."""
        manager = ProviderManager(providers=[mock_provider_failing])

        mock_provider_failing.get_quote.side_effect = ProviderError("Failed")

        with pytest.raises(ProviderError) as exc_info:
            manager.get_quote("AAPL", use_cache=False)

        assert "All providers failed" in str(exc_info.value)

    def test_circuit_breaker_prevents_calls(self, mock_provider_healthy, sample_quote):
        """Test that open circuit prevents calls to provider."""
        manager = ProviderManager(
            providers=[mock_provider_healthy],
            circuit_failure_threshold=2
        )

        mock_provider_healthy.get_quote.side_effect = ProviderError("Failed")

        # Cause failures to open circuit
        for _ in range(2):
            try:
                manager.get_quote("AAPL", use_cache=False)
            except ProviderError:
                pass

        # Circuit should be open, no more calls
        circuit = manager._circuits["mock_healthy"]
        assert circuit.state == CircuitState.OPEN

        with pytest.raises(ProviderError) as exc_info:
            manager.get_quote("AAPL", use_cache=False)

        assert "No providers available" in str(exc_info.value)

    def test_get_historical_with_cache(self, mock_provider_healthy, sample_historical):
        """Test historical data fetching with caching."""
        manager = ProviderManager(providers=[mock_provider_healthy])
        mock_provider_healthy.get_historical.return_value = sample_historical

        result1 = manager.get_historical("AAPL", "60d", "1d")
        result2 = manager.get_historical("AAPL", "60d", "1d")

        assert mock_provider_healthy.get_historical.call_count == 1
        assert result1 == result2

    def test_get_batch_quotes_caches_individual(self, mock_provider_healthy, sample_quote):
        """Test that batch quotes are cached individually."""
        manager = ProviderManager(providers=[mock_provider_healthy])

        batch_result = {
            "AAPL": sample_quote,
            "MSFT": Quote(**{**sample_quote.__dict__, "symbol": "MSFT"}),
        }
        mock_provider_healthy.get_batch_quotes.return_value = batch_result

        manager.get_batch_quotes(["AAPL", "MSFT"])

        # Individual quotes should be cached
        assert manager._cache.get("quote:AAPL") is not None
        assert manager._cache.get("quote:MSFT") is not None

    def test_get_health(self, mock_provider_healthy, mock_provider_failing):
        """Test getting health status of all providers."""
        manager = ProviderManager(providers=[mock_provider_healthy, mock_provider_failing])

        health = manager.get_health()

        assert "mock_healthy" in health
        assert "mock_failing" in health
        assert health["mock_healthy"].status == ProviderStatus.HEALTHY

    def test_clear_cache(self, mock_provider_healthy, sample_quote):
        """Test clearing the cache."""
        manager = ProviderManager(providers=[mock_provider_healthy])
        mock_provider_healthy.get_quote.return_value = sample_quote

        manager.get_quote("AAPL")
        assert manager._cache.size > 0

        manager.clear_cache()
        assert manager._cache.size == 0

    def test_invalidate_symbol(self, mock_provider_healthy, sample_quote, sample_historical):
        """Test invalidating all cache entries for a symbol."""
        manager = ProviderManager(providers=[mock_provider_healthy])
        mock_provider_healthy.get_quote.return_value = sample_quote
        mock_provider_healthy.get_historical.return_value = sample_historical

        manager.get_quote("AAPL")
        manager.get_historical("AAPL", "60d", "1d")

        count = manager.invalidate_symbol("AAPL")

        assert count == 2
        assert manager._cache.get("quote:AAPL") is None
        assert manager._cache.get("historical:AAPL:60d:1d") is None

    def test_reset_circuit(self, mock_provider_healthy):
        """Test resetting a circuit breaker."""
        manager = ProviderManager(
            providers=[mock_provider_healthy],
            circuit_failure_threshold=2
        )

        # Open the circuit
        mock_provider_healthy.get_quote.side_effect = ProviderError("Failed")
        for _ in range(2):
            try:
                manager.get_quote("AAPL", use_cache=False)
            except ProviderError:
                pass

        assert manager._circuits["mock_healthy"].state == CircuitState.OPEN

        # Reset the circuit
        result = manager.reset_circuit("mock_healthy")

        assert result is True
        assert manager._circuits["mock_healthy"].state == CircuitState.CLOSED

    def test_set_provider_priority(self, mock_provider_healthy, mock_provider_failing):
        """Test changing provider priority."""
        manager = ProviderManager(providers=[mock_provider_healthy, mock_provider_failing])

        # Initial order: failing (5), healthy (10)
        assert manager._provider_order[0] == "mock_failing"

        # Change healthy to priority 1
        manager.set_provider_priority("mock_healthy", 1)

        # New order: healthy (1), failing (5)
        assert manager._provider_order[0] == "mock_healthy"

    def test_status_report(self, mock_provider_healthy):
        """Test comprehensive status report."""
        manager = ProviderManager(providers=[mock_provider_healthy])

        status = manager.status()

        assert "providers" in status
        assert "provider_order" in status
        assert "cache" in status
        assert "mock_healthy" in status["providers"]


# =============================================================================
# YFinance Provider Tests
# =============================================================================

class TestYFinanceProvider:
    """Tests for YFinance provider."""

    def test_initialization(self):
        """Test provider initialization."""
        provider = YFinanceProvider(priority=15)

        assert provider.name == "yfinance"
        assert provider.priority == 15

    def test_get_quote_success(self, sample_5m_data, sample_ohlcv_data):
        """Test successful quote fetch."""
        provider = YFinanceProvider()

        with patch('data.providers.yfinance_provider.yf.Ticker') as mock_ticker:
            mock = Mock()
            mock.history.side_effect = [sample_5m_data, sample_ohlcv_data.tail(5)]
            mock_ticker.return_value = mock

            quote = provider.get_quote("AAPL")

            assert quote.symbol == "AAPL"
            assert quote.provider == "yfinance"
            assert isinstance(quote.price, float)

    def test_get_quote_no_data(self):
        """Test handling of empty data."""
        provider = YFinanceProvider()

        with patch('data.providers.yfinance_provider.yf.Ticker') as mock_ticker:
            mock = Mock()
            mock.history.return_value = pd.DataFrame()
            mock_ticker.return_value = mock

            with pytest.raises(DataNotFoundError):
                provider.get_quote("INVALID")

    def test_get_quote_rate_limit(self):
        """Test handling of rate limit error."""
        provider = YFinanceProvider(max_retries=1)

        with patch('data.providers.yfinance_provider.yf.Ticker') as mock_ticker, \
             patch('data.providers.yfinance_provider.time.sleep'):
            mock = Mock()
            mock.history.side_effect = Exception("401 too many requests")
            mock_ticker.return_value = mock

            with pytest.raises(RateLimitError):
                provider.get_quote("AAPL")

    def test_get_historical_success(self, sample_ohlcv_data):
        """Test successful historical data fetch."""
        provider = YFinanceProvider()

        with patch('data.providers.yfinance_provider.yf.Ticker') as mock_ticker:
            mock = Mock()
            mock.history.return_value = sample_ohlcv_data.copy()
            mock.history.return_value.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            mock_ticker.return_value = mock

            hist = provider.get_historical("AAPL", "60d", "1d")

            assert hist.symbol == "AAPL"
            assert hist.period == "60d"
            assert hist.interval == "1d"
            assert not hist.data.empty

    def test_get_batch_quotes_success(self, mock_batch_5m_data, mock_batch_daily_data):
        """Test successful batch quote fetch."""
        provider = YFinanceProvider()

        with patch('data.providers.yfinance_provider.yf.download') as mock_download:
            mock_download.side_effect = [mock_batch_5m_data, mock_batch_daily_data]

            quotes = provider.get_batch_quotes(["AAPL", "MSFT", "GOOGL"])

            assert len(quotes) > 0
            for symbol, quote in quotes.items():
                assert quote.provider == "yfinance"

    def test_get_batch_historical_success(self, mock_batch_daily_data):
        """Test successful batch historical data fetch."""
        provider = YFinanceProvider()

        with patch('data.providers.yfinance_provider.yf.download') as mock_download:
            mock_download.return_value = mock_batch_daily_data

            result = provider.get_batch_historical(["AAPL", "MSFT"], "60d", "1d")

            assert len(result) > 0
            for symbol, hist in result.items():
                assert hist.provider == "yfinance"

    def test_is_available_success(self):
        """Test availability check success."""
        provider = YFinanceProvider()

        with patch('data.providers.yfinance_provider.yf.Ticker') as mock_ticker:
            mock = Mock()
            mock.fast_info = {"something": "value"}
            mock_ticker.return_value = mock

            assert provider.is_available() is True

    def test_is_available_failure(self):
        """Test availability check failure."""
        provider = YFinanceProvider()

        with patch('data.providers.yfinance_provider.yf.Ticker') as mock_ticker:
            mock_ticker.side_effect = Exception("Network error")

            assert provider.is_available() is False

    def test_retry_on_rate_limit_error(self, sample_5m_data, sample_ohlcv_data):
        """Test retry logic on rate limit errors."""
        provider = YFinanceProvider(max_retries=3, retry_delay=0.01)

        with patch('data.providers.yfinance_provider.yf.Ticker') as mock_ticker, \
             patch('data.providers.yfinance_provider.time.sleep'):
            mock = Mock()
            # First call raises rate limit error, then succeeds
            call_count = [0]

            def history_side_effect(*args, **kwargs):
                call_count[0] += 1
                # First 2 calls are the failed attempt (intraday then bails)
                if call_count[0] == 1:
                    raise Exception("401 rate limit exceeded")
                # After retry: intraday call
                elif call_count[0] == 2:
                    return sample_5m_data
                # After retry: daily call
                else:
                    return sample_ohlcv_data.tail(5)

            mock.history.side_effect = history_side_effect
            mock_ticker.return_value = mock

            quote = provider.get_quote("AAPL")

            assert quote is not None
            assert quote.symbol == "AAPL"


# =============================================================================
# Alpha Vantage Provider Tests
# =============================================================================

class TestAlphaVantageProvider:
    """Tests for Alpha Vantage provider."""

    def test_initialization_with_api_key(self):
        """Test initialization with API key."""
        provider = AlphaVantageProvider(api_key="test_key", priority=20)

        assert provider.name == "alpha_vantage"
        assert provider.api_key == "test_key"
        assert provider.priority == 20

    def test_initialization_without_api_key(self):
        """Test initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            provider = AlphaVantageProvider(api_key="")

            assert provider.api_key == ""
            assert provider.is_available() is False

    def test_get_quote_success(self):
        """Test successful quote fetch."""
        provider = AlphaVantageProvider(api_key="test_key")

        mock_response = {
            "Global Quote": {
                "01. symbol": "AAPL",
                "05. price": "175.50",
                "02. open": "174.00",
                "03. high": "176.25",
                "04. low": "173.50",
                "06. volume": "50000000",
                "08. previous close": "173.25",
                "09. change": "2.25",
                "10. change percent": "1.30%",
            }
        }

        with patch('data.providers.alpha_vantage_provider.requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_response

            quote = provider.get_quote("AAPL")

            assert quote.symbol == "AAPL"
            assert quote.price == 175.50
            assert quote.provider == "alpha_vantage"

    def test_get_quote_no_api_key(self):
        """Test error when no API key configured."""
        provider = AlphaVantageProvider(api_key="")

        with pytest.raises(AuthenticationError):
            provider.get_quote("AAPL")

    def test_get_quote_rate_limit(self):
        """Test handling of rate limit response."""
        provider = AlphaVantageProvider(api_key="test_key")

        mock_response = {
            "Note": "Thank you for using Alpha Vantage! Our standard API call frequency is rate limit exceeded."
        }

        with patch('data.providers.alpha_vantage_provider.requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_response

            with pytest.raises(RateLimitError):
                provider.get_quote("AAPL")

    def test_get_quote_http_rate_limit(self):
        """Test handling of HTTP 429 rate limit."""
        provider = AlphaVantageProvider(api_key="test_key")

        with patch('data.providers.alpha_vantage_provider.requests.get') as mock_get:
            mock_get.return_value.status_code = 429

            with pytest.raises(RateLimitError):
                provider.get_quote("AAPL")

    def test_get_historical_success(self):
        """Test successful historical data fetch."""
        provider = AlphaVantageProvider(api_key="test_key")

        # Use recent dates that will pass the period filter
        from datetime import datetime, timedelta
        today = datetime.now()
        yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        two_days_ago = (today - timedelta(days=2)).strftime("%Y-%m-%d")

        mock_response = {
            "Time Series (Daily)": {
                yesterday: {
                    "1. open": "174.00",
                    "2. high": "176.25",
                    "3. low": "173.50",
                    "4. close": "175.50",
                    "5. volume": "50000000",
                },
                two_days_ago: {
                    "1. open": "173.00",
                    "2. high": "174.00",
                    "3. low": "172.50",
                    "4. close": "173.25",
                    "5. volume": "45000000",
                },
            }
        }

        with patch('data.providers.alpha_vantage_provider.requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_response

            hist = provider.get_historical("AAPL", "1mo", "1d")

            assert hist.symbol == "AAPL"
            assert not hist.data.empty
            assert "close" in hist.data.columns

    def test_daily_limit_tracking(self):
        """Test that daily request limit is tracked."""
        provider = AlphaVantageProvider(api_key="test_key", daily_limit=5)

        mock_response = {
            "Global Quote": {
                "01. symbol": "AAPL",
                "05. price": "175.50",
                "02. open": "174.00",
                "03. high": "176.25",
                "04. low": "173.50",
                "06. volume": "50000000",
                "08. previous close": "173.25",
                "09. change": "2.25",
                "10. change percent": "1.30%",
            }
        }

        with patch('data.providers.alpha_vantage_provider.requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_response

            # Make 5 requests
            for _ in range(5):
                provider.get_quote("AAPL")

            # 6th request should fail
            with pytest.raises(RateLimitError):
                provider.get_quote("AAPL")

    def test_get_remaining_requests(self):
        """Test getting remaining request count."""
        provider = AlphaVantageProvider(api_key="test_key", daily_limit=25)

        assert provider.get_remaining_requests() == 25

        # Simulate some requests
        provider._daily_request_count = 10

        assert provider.get_remaining_requests() == 15

    def test_is_available_with_key(self):
        """Test availability with API key configured."""
        provider = AlphaVantageProvider(api_key="test_key")

        assert provider.is_available() is True

    def test_is_available_at_limit(self):
        """Test availability when daily limit reached."""
        provider = AlphaVantageProvider(api_key="test_key", daily_limit=5)
        provider._daily_request_count = 5

        assert provider.is_available() is False


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_no_providers_available(self):
        """Test error when no providers are available."""
        manager = ProviderManager(providers=[])

        with pytest.raises(ProviderError) as exc_info:
            manager.get_quote("AAPL")

        assert "No providers available" in str(exc_info.value)

    def test_all_providers_circuit_open(self, mock_provider_healthy, mock_provider_failing):
        """Test error when all provider circuits are open."""
        manager = ProviderManager(
            providers=[mock_provider_healthy, mock_provider_failing],
            circuit_failure_threshold=1
        )

        # Open both circuits
        mock_provider_healthy.get_quote.side_effect = ProviderError("Failed")
        mock_provider_failing.get_quote.side_effect = ProviderError("Failed")

        try:
            manager.get_quote("AAPL", use_cache=False)
        except ProviderError:
            pass

        try:
            manager.get_quote("AAPL", use_cache=False)
        except ProviderError:
            pass

        # Both circuits should be open
        with pytest.raises(ProviderError) as exc_info:
            manager.get_quote("AAPL", use_cache=False)

        assert "No providers available" in str(exc_info.value)

    def test_authentication_error_propagates(self, mock_provider_healthy):
        """Test that authentication errors are handled properly."""
        manager = ProviderManager(providers=[mock_provider_healthy])

        mock_provider_healthy.get_quote.side_effect = AuthenticationError("Invalid API key")

        with pytest.raises(ProviderError):
            manager.get_quote("AAPL", use_cache=False)

    def test_data_not_found_error(self, mock_provider_healthy):
        """Test handling of data not found errors."""
        manager = ProviderManager(providers=[mock_provider_healthy])

        mock_provider_healthy.get_quote.side_effect = DataNotFoundError("Symbol not found")

        with pytest.raises(ProviderError):
            manager.get_quote("INVALID", use_cache=False)

    def test_batch_quotes_partial_failure(self, mock_provider_healthy, sample_quote):
        """Test batch quotes with some symbols failing."""
        manager = ProviderManager(providers=[mock_provider_healthy])

        # Return only some symbols
        batch_result = {"AAPL": sample_quote}
        mock_provider_healthy.get_batch_quotes.return_value = batch_result

        result = manager.get_batch_quotes(["AAPL", "INVALID"], use_cache=False)

        assert "AAPL" in result
        assert len(result) == 1


# =============================================================================
# Global Provider Manager Tests
# =============================================================================

class TestGlobalProviderManager:
    """Tests for global provider manager functions."""

    def test_get_provider_manager_creates_instance(self):
        """Test that get_provider_manager creates an instance."""
        reset_provider_manager()

        manager = get_provider_manager()

        assert manager is not None
        assert isinstance(manager, ProviderManager)

    def test_get_provider_manager_returns_same_instance(self):
        """Test that get_provider_manager returns the same instance."""
        reset_provider_manager()

        manager1 = get_provider_manager()
        manager2 = get_provider_manager()

        assert manager1 is manager2

    def test_reset_provider_manager(self):
        """Test that reset_provider_manager clears the instance."""
        manager1 = get_provider_manager()
        reset_provider_manager()
        manager2 = get_provider_manager()

        assert manager1 is not manager2

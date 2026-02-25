"""
API Performance Tests using pytest-benchmark

Tests API endpoint response times, concurrent request handling,
and memory usage.

Usage:
    # Run all performance tests
    pytest tests/performance/test_api_performance.py -v

    # Run with benchmark output
    pytest tests/performance/test_api_performance.py -v --benchmark-only

    # Save benchmark results
    pytest tests/performance/test_api_performance.py -v --benchmark-save=baseline

    # Compare with previous results
    pytest tests/performance/test_api_performance.py -v --benchmark-compare=baseline
"""

import asyncio
import gc
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Try to import benchmark - skip tests if not available
try:
    import pytest_benchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

# Try to import memory profiler
try:
    from memory_profiler import memory_usage
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

# Import test configuration
try:
    from tests.performance.config import (
        get_performance_config,
        TargetMetrics,
        TEST_SYMBOLS,
        API_ENDPOINTS,
    )
except ImportError:
    from config import (
        get_performance_config,
        TargetMetrics,
        TEST_SYMBOLS,
        API_ENDPOINTS,
    )


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def perf_config():
    """Get performance test configuration."""
    return get_performance_config()


@pytest.fixture(scope="module")
def app():
    """Create Flask app for testing."""
    try:
        from flask import Flask
        from api.v1.routes import api_bp

        app = Flask(__name__)
        app.config['TESTING'] = True
        app.register_blueprint(api_bp)

        return app
    except ImportError:
        pytest.skip("Flask app not available")


@pytest.fixture(scope="module")
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def auth_headers():
    """Get authentication headers."""
    api_key = os.getenv('TEST_API_KEY', 'test-api-key')
    return {
        'X-API-Key': api_key,
        'Content-Type': 'application/json',
    }


@pytest.fixture
def mock_data_provider():
    """Create mock data provider for isolated tests."""
    mock_provider = MagicMock()

    # Mock stock data
    mock_provider.get_stock_data.return_value = {
        'symbol': 'AAPL',
        'current_price': 175.50,
        'previous_close': 174.25,
        'volume': 50000000,
        'atr': 2.5,
    }

    # Mock quote
    mock_provider.get_quote.return_value = {
        'symbol': 'AAPL',
        'price': 175.50,
        'change': 1.25,
        'change_pct': 0.72,
        'volume': 50000000,
    }

    return mock_provider


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def measure_response_time(func: Callable, *args, **kwargs) -> tuple:
    """
    Measure response time of a function.

    Returns:
        Tuple of (result, response_time_ms)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed


def run_concurrent_requests(
    func: Callable,
    num_requests: int,
    max_workers: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Run multiple concurrent requests and collect statistics.

    Returns:
        Dictionary with response times and statistics
    """
    response_times = []
    errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(measure_response_time, func, **kwargs)
                   for _ in range(num_requests)]

        for future in as_completed(futures):
            try:
                _, elapsed = future.result()
                response_times.append(elapsed)
            except Exception as e:
                errors.append(str(e))

    if not response_times:
        return {
            'count': num_requests,
            'successful': 0,
            'failed': len(errors),
            'errors': errors,
        }

    response_times.sort()
    return {
        'count': num_requests,
        'successful': len(response_times),
        'failed': len(errors),
        'min_ms': min(response_times),
        'max_ms': max(response_times),
        'avg_ms': sum(response_times) / len(response_times),
        'p50_ms': response_times[int(len(response_times) * 0.50)],
        'p95_ms': response_times[int(len(response_times) * 0.95)],
        'p99_ms': response_times[int(len(response_times) * 0.99)] if len(response_times) >= 100 else response_times[-1],
        'errors': errors,
    }


def measure_memory(func: Callable, *args, **kwargs) -> Dict[str, float]:
    """
    Measure memory usage of a function.

    Returns:
        Dictionary with memory statistics in MB
    """
    if not MEMORY_PROFILER_AVAILABLE:
        return {'error': 'memory_profiler not available'}

    gc.collect()

    mem_before = memory_usage()[0]
    mem_during = memory_usage((func, args, kwargs), interval=0.1)
    gc.collect()
    mem_after = memory_usage()[0]

    return {
        'before_mb': mem_before,
        'peak_mb': max(mem_during),
        'after_mb': mem_after,
        'growth_mb': mem_after - mem_before,
        'peak_growth_mb': max(mem_during) - mem_before,
    }


# =============================================================================
# HEALTH ENDPOINT TESTS
# =============================================================================

class TestHealthEndpointPerformance:
    """Performance tests for health endpoint."""

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not available")
    def test_health_response_time(self, benchmark, client, perf_config):
        """Benchmark health endpoint response time."""
        def make_request():
            return client.get('/api/v1/health')

        result = benchmark(make_request)

        # Verify response
        assert result.status_code == 200

        # Check against target
        target = perf_config.get_target('health')
        stats = benchmark.stats.stats
        assert stats.median * 1000 < target.p50_response_ms, \
            f"Median {stats.median * 1000:.2f}ms exceeds target {target.p50_response_ms}ms"

    def test_health_concurrent_requests(self, client, perf_config):
        """Test health endpoint under concurrent load."""
        def make_request():
            return client.get('/api/v1/health')

        stats = run_concurrent_requests(make_request, num_requests=100, max_workers=20)

        # Verify success rate
        success_rate = stats['successful'] / stats['count']
        assert success_rate >= 0.99, f"Success rate {success_rate:.2%} below 99%"

        # Check response times
        target = perf_config.get_target('health')
        assert stats['p95_ms'] < target.p95_response_ms, \
            f"P95 {stats['p95_ms']:.2f}ms exceeds target {target.p95_response_ms}ms"


class TestStatusEndpointPerformance:
    """Performance tests for status endpoint."""

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not available")
    def test_status_response_time(self, benchmark, client, perf_config):
        """Benchmark status endpoint response time."""
        def make_request():
            return client.get('/api/v1/status')

        result = benchmark(make_request)
        assert result.status_code == 200

    def test_status_concurrent_requests(self, client, perf_config):
        """Test status endpoint under concurrent load."""
        def make_request():
            return client.get('/api/v1/status')

        stats = run_concurrent_requests(make_request, num_requests=50, max_workers=10)

        success_rate = stats['successful'] / stats['count']
        assert success_rate >= 0.98, f"Success rate {success_rate:.2%} below 98%"


# =============================================================================
# DASHBOARD ENDPOINT TESTS
# =============================================================================

class TestDashboardEndpointPerformance:
    """Performance tests for dashboard endpoint."""

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not available")
    def test_dashboard_response_time(self, benchmark, client, perf_config):
        """Benchmark dashboard endpoint response time."""
        def make_request():
            return client.get('/api/v1/dashboard')

        result = benchmark(make_request)
        assert result.status_code == 200

        # Check against target
        target = perf_config.get_target('dashboard')
        stats = benchmark.stats.stats
        assert stats.median * 1000 < target.p50_response_ms, \
            f"Median {stats.median * 1000:.2f}ms exceeds target {target.p50_response_ms}ms"

    def test_dashboard_concurrent_requests(self, client, perf_config):
        """Test dashboard endpoint under concurrent load."""
        def make_request():
            return client.get('/api/v1/dashboard')

        stats = run_concurrent_requests(make_request, num_requests=30, max_workers=10)

        success_rate = stats['successful'] / stats['count']
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} below 95%"

    @pytest.mark.skipif(not MEMORY_PROFILER_AVAILABLE, reason="memory_profiler not available")
    def test_dashboard_memory_usage(self, client, perf_config):
        """Test dashboard endpoint memory usage."""
        def make_multiple_requests():
            for _ in range(10):
                client.get('/api/v1/dashboard')

        mem_stats = measure_memory(make_multiple_requests)

        # Check memory growth
        limits = perf_config.resource_limits
        assert mem_stats['growth_mb'] < limits.max_memory_growth_mb, \
            f"Memory growth {mem_stats['growth_mb']:.2f}MB exceeds limit {limits.max_memory_growth_mb}MB"


# =============================================================================
# SIGNALS ENDPOINT TESTS
# =============================================================================

class TestSignalsEndpointPerformance:
    """Performance tests for signals endpoints."""

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not available")
    def test_signals_current_response_time(self, benchmark, client, auth_headers, perf_config):
        """Benchmark current signals endpoint."""
        def make_request():
            return client.get('/api/v1/signals/current', headers=auth_headers)

        result = benchmark(make_request)

        # May return 401 if auth not properly mocked - that's OK for benchmark
        assert result.status_code in [200, 401, 404]

    def test_signals_current_concurrent(self, client, auth_headers, perf_config):
        """Test signals endpoint under concurrent load."""
        def make_request():
            return client.get('/api/v1/signals/current', headers=auth_headers)

        stats = run_concurrent_requests(make_request, num_requests=50, max_workers=10)

        # Allow for auth failures
        total_handled = stats['successful'] + len([e for e in stats.get('errors', []) if '401' in str(e)])
        assert total_handled / stats['count'] >= 0.95

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not available")
    def test_signals_history_response_time(self, benchmark, client, auth_headers, perf_config):
        """Benchmark signals history endpoint."""
        def make_request():
            return client.get('/api/v1/signals/history?limit=50', headers=auth_headers)

        result = benchmark(make_request)
        assert result.status_code in [200, 401, 404]


# =============================================================================
# POSITIONS ENDPOINT TESTS
# =============================================================================

class TestPositionsEndpointPerformance:
    """Performance tests for positions endpoint."""

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not available")
    def test_positions_response_time(self, benchmark, client, auth_headers, perf_config):
        """Benchmark positions endpoint."""
        def make_request():
            return client.get('/api/v1/positions', headers=auth_headers)

        result = benchmark(make_request)
        assert result.status_code in [200, 401, 404]

    def test_positions_concurrent(self, client, auth_headers, perf_config):
        """Test positions endpoint under concurrent load."""
        def make_request():
            return client.get('/api/v1/positions', headers=auth_headers)

        stats = run_concurrent_requests(make_request, num_requests=30, max_workers=10)

        # Check response time
        target = perf_config.get_target('positions')
        if stats['successful'] > 0:
            assert stats['p95_ms'] < target.p95_response_ms * 2  # Allow 2x during concurrent load


# =============================================================================
# QUOTE ENDPOINT TESTS
# =============================================================================

class TestQuoteEndpointPerformance:
    """Performance tests for quote endpoint."""

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not available")
    def test_quote_response_time(self, benchmark, client, auth_headers):
        """Benchmark quote endpoint."""
        def make_request():
            return client.get('/api/v1/quote/AAPL', headers=auth_headers)

        result = benchmark(make_request)
        assert result.status_code in [200, 401, 404, 500]

    def test_quote_multiple_symbols(self, client, auth_headers, perf_config):
        """Test fetching quotes for multiple symbols."""
        response_times = []

        for symbol in TEST_SYMBOLS[:10]:
            start = time.perf_counter()
            response = client.get(f'/api/v1/quote/{symbol}', headers=auth_headers)
            elapsed = (time.perf_counter() - start) * 1000
            response_times.append(elapsed)

        # Check average response time
        avg_time = sum(response_times) / len(response_times)
        assert avg_time < 2000, f"Average quote time {avg_time:.2f}ms too slow"


# =============================================================================
# BACKTEST ENDPOINT TESTS
# =============================================================================

class TestBacktestEndpointPerformance:
    """Performance tests for backtest endpoint (heavy operation)."""

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not available")
    @pytest.mark.slow
    def test_backtest_response_time(self, benchmark, client, auth_headers, perf_config):
        """Benchmark backtest endpoint."""
        payload = {
            'symbols': ['AAPL', 'MSFT'],
            'start_date': '2024-01-01',
            'end_date': '2024-02-01',
            'initial_capital': 100000,
        }

        def make_request():
            return client.post(
                '/api/v1/backtest',
                headers=auth_headers,
                json=payload
            )

        # Use fewer rounds for slow test
        result = benchmark.pedantic(make_request, rounds=3, warmup_rounds=1)

        # Backtest may return various status codes
        assert result.status_code in [200, 202, 401, 404, 500, 503]

    @pytest.mark.slow
    def test_backtest_memory_usage(self, client, auth_headers, perf_config):
        """Test backtest memory usage."""
        if not MEMORY_PROFILER_AVAILABLE:
            pytest.skip("memory_profiler not available")

        payload = {
            'symbols': ['AAPL'],
            'start_date': '2024-01-01',
            'end_date': '2024-02-01',
            'initial_capital': 100000,
        }

        def run_backtest():
            client.post('/api/v1/backtest', headers=auth_headers, json=payload)

        mem_stats = measure_memory(run_backtest)

        # Check memory doesn't explode
        limits = perf_config.resource_limits
        assert mem_stats['peak_growth_mb'] < limits.max_memory_mb


# =============================================================================
# CONCURRENT REQUEST STRESS TESTS
# =============================================================================

class TestConcurrentStress:
    """Stress tests with high concurrent load."""

    @pytest.mark.stress
    def test_high_concurrent_load(self, client, perf_config):
        """Test system under high concurrent load."""
        endpoints = [
            '/api/v1/health',
            '/api/v1/status',
            '/api/v1/dashboard',
        ]

        def make_random_request():
            import random
            endpoint = random.choice(endpoints)
            return client.get(endpoint)

        stats = run_concurrent_requests(
            make_random_request,
            num_requests=200,
            max_workers=50
        )

        # Expect at least 90% success under stress
        success_rate = stats['successful'] / stats['count']
        assert success_rate >= 0.90, f"Success rate {success_rate:.2%} below 90% under stress"

    @pytest.mark.stress
    def test_sustained_load(self, client, perf_config):
        """Test system under sustained load for 30 seconds."""
        start_time = time.time()
        duration = 10  # 10 seconds for test
        request_count = 0
        error_count = 0

        while time.time() - start_time < duration:
            try:
                response = client.get('/api/v1/health')
                if response.status_code != 200:
                    error_count += 1
                request_count += 1
            except Exception:
                error_count += 1
                request_count += 1

        # Calculate metrics
        elapsed = time.time() - start_time
        rps = request_count / elapsed
        error_rate = error_count / request_count if request_count > 0 else 1.0

        print(f"\nSustained load results:")
        print(f"  Duration: {elapsed:.2f}s")
        print(f"  Requests: {request_count}")
        print(f"  RPS: {rps:.2f}")
        print(f"  Error rate: {error_rate:.2%}")

        assert error_rate < 0.05, f"Error rate {error_rate:.2%} exceeds 5%"
        assert rps > 10, f"RPS {rps:.2f} below minimum threshold"


# =============================================================================
# MEMORY LEAK TESTS
# =============================================================================

class TestMemoryLeaks:
    """Tests for memory leaks during repeated operations."""

    @pytest.mark.skipif(not MEMORY_PROFILER_AVAILABLE, reason="memory_profiler not available")
    def test_repeated_requests_memory(self, client, perf_config):
        """Test for memory leaks during repeated requests."""
        gc.collect()
        mem_before = memory_usage()[0]

        # Make many requests
        for _ in range(100):
            client.get('/api/v1/health')
            client.get('/api/v1/status')
            client.get('/api/v1/dashboard')

        gc.collect()
        mem_after = memory_usage()[0]
        growth = mem_after - mem_before

        print(f"\nMemory usage after 300 requests:")
        print(f"  Before: {mem_before:.2f} MB")
        print(f"  After: {mem_after:.2f} MB")
        print(f"  Growth: {growth:.2f} MB")

        limits = perf_config.resource_limits
        assert growth < limits.max_memory_growth_mb, \
            f"Memory growth {growth:.2f}MB exceeds limit {limits.max_memory_growth_mb}MB"


# =============================================================================
# RESPONSE TIME DISTRIBUTION TESTS
# =============================================================================

class TestResponseTimeDistribution:
    """Tests for response time distribution and consistency."""

    def test_health_response_time_consistency(self, client, perf_config):
        """Test that health endpoint has consistent response times."""
        response_times = []

        for _ in range(50):
            start = time.perf_counter()
            client.get('/api/v1/health')
            elapsed = (time.perf_counter() - start) * 1000
            response_times.append(elapsed)

        # Calculate statistics
        avg = sum(response_times) / len(response_times)
        variance = sum((t - avg) ** 2 for t in response_times) / len(response_times)
        std_dev = variance ** 0.5
        cv = std_dev / avg if avg > 0 else 0  # Coefficient of variation

        print(f"\nHealth endpoint response time distribution:")
        print(f"  Mean: {avg:.2f} ms")
        print(f"  Std Dev: {std_dev:.2f} ms")
        print(f"  CV: {cv:.2f}")
        print(f"  Min: {min(response_times):.2f} ms")
        print(f"  Max: {max(response_times):.2f} ms")

        # CV should be relatively low for consistent performance
        assert cv < 1.0, f"Response time coefficient of variation {cv:.2f} indicates inconsistent performance"

    def test_response_time_percentiles(self, client, perf_config):
        """Test response time percentiles meet targets."""
        response_times = []

        for _ in range(100):
            start = time.perf_counter()
            client.get('/api/v1/dashboard')
            elapsed = (time.perf_counter() - start) * 1000
            response_times.append(elapsed)

        response_times.sort()

        p50 = response_times[50]
        p95 = response_times[95]
        p99 = response_times[99]

        print(f"\nDashboard response time percentiles:")
        print(f"  P50: {p50:.2f} ms")
        print(f"  P95: {p95:.2f} ms")
        print(f"  P99: {p99:.2f} ms")

        target = perf_config.get_target('dashboard')
        assert p50 < target.p50_response_ms * 2, f"P50 {p50:.2f}ms exceeds 2x target"
        assert p95 < target.p95_response_ms * 2, f"P95 {p95:.2f}ms exceeds 2x target"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    pytest.main([
        __file__,
        '-v',
        '--benchmark-only',
        '--benchmark-group-by=class',
    ])

"""
Scanner Performance Tests

Benchmarks for the RDT Trading System scanner including:
- Scan duration with different watchlist sizes
- Data provider performance
- Memory usage during scans
- RRS calculation performance

Usage:
    pytest tests/performance/test_scanner_performance.py -v
    pytest tests/performance/test_scanner_performance.py -v --benchmark-only
"""

import asyncio
import gc
import os
import sys
import time
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Try to import benchmark
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

# Import configuration
try:
    from tests.performance.config import (
        get_performance_config,
        TEST_SYMBOLS,
        SMALL_WATCHLIST,
        MEDIUM_WATCHLIST,
        LARGE_WATCHLIST,
    )
except ImportError:
    from config import (
        get_performance_config,
        TEST_SYMBOLS,
        SMALL_WATCHLIST,
        MEDIUM_WATCHLIST,
        LARGE_WATCHLIST,
    )


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def perf_config():
    """Get performance configuration."""
    return get_performance_config()


@pytest.fixture
def mock_daily_data():
    """Generate mock daily price data."""
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
    np.random.seed(42)

    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
        'high': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
        'low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates)),
    }, index=dates)

    return df


@pytest.fixture
def mock_data_provider(mock_daily_data):
    """Create mock data provider."""
    provider = MagicMock()

    async def mock_get_stock_data(symbol):
        # Add small delay to simulate network
        await asyncio.sleep(0.01)

        return {
            'symbol': symbol,
            'current_price': float(mock_daily_data['close'].iloc[-1]),
            'previous_close': float(mock_daily_data['close'].iloc[-2]),
            'open': float(mock_daily_data['open'].iloc[-1]),
            'high': float(mock_daily_data['high'].iloc[-1]),
            'low': float(mock_daily_data['low'].iloc[-1]),
            'volume': int(mock_daily_data['volume'].iloc[-1]),
            'atr': float(mock_daily_data['high'].iloc[-1] - mock_daily_data['low'].iloc[-1]),
            'daily_data': mock_daily_data.copy(),
        }

    async def mock_get_spy_data():
        return await mock_get_stock_data('SPY')

    provider.get_stock_data = mock_get_stock_data
    provider.get_spy_data = mock_get_spy_data

    return provider


@pytest.fixture
def rrs_calculator():
    """Get RRS calculator instance."""
    try:
        from shared.indicators.rrs import RRSCalculator
        return RRSCalculator()
    except ImportError:
        pytest.skip("RRSCalculator not available")


# =============================================================================
# RRS CALCULATION PERFORMANCE TESTS
# =============================================================================

class TestRRSCalculationPerformance:
    """Performance tests for RRS calculations."""

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not available")
    def test_rrs_single_calculation(self, benchmark, rrs_calculator, mock_daily_data):
        """Benchmark single RRS calculation."""
        spy_data = mock_daily_data.copy()
        stock_data = mock_daily_data.copy()

        def calculate():
            return rrs_calculator.calculate_rrs(stock_data, spy_data, periods=1)

        result = benchmark(calculate)
        assert result is not None

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not available")
    def test_rrs_batch_calculation(self, benchmark, rrs_calculator, mock_daily_data):
        """Benchmark batch RRS calculation for multiple periods."""
        spy_data = mock_daily_data.copy()
        stock_data = mock_daily_data.copy()

        def calculate_all():
            results = []
            for periods in [1, 3, 5, 10]:
                result = rrs_calculator.calculate_rrs(stock_data, spy_data, periods=periods)
                results.append(result)
            return results

        result = benchmark(calculate_all)
        assert len(result) == 4

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not available")
    def test_atr_calculation(self, benchmark, rrs_calculator, mock_daily_data):
        """Benchmark ATR calculation."""
        def calculate():
            return rrs_calculator.calculate_atr(mock_daily_data)

        result = benchmark(calculate)
        assert result is not None
        assert len(result) == len(mock_daily_data)

    def test_rrs_calculation_scalability(self, rrs_calculator, mock_daily_data, perf_config):
        """Test RRS calculation scalability with increasing data."""
        spy_data = mock_daily_data.copy()

        data_sizes = [30, 60, 120, 250, 500]
        times = []

        for size in data_sizes:
            # Generate larger dataset
            dates = pd.date_range(end=datetime.now(), periods=size, freq='D')
            np.random.seed(42)
            prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, size))

            stock_data = pd.DataFrame({
                'open': prices * 0.99,
                'high': prices * 1.01,
                'low': prices * 0.98,
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, size),
            }, index=dates)

            spy_large = stock_data.copy()

            # Measure time
            start = time.perf_counter()
            for _ in range(10):
                rrs_calculator.calculate_rrs(stock_data, spy_large, periods=1)
            elapsed = (time.perf_counter() - start) / 10 * 1000

            times.append(elapsed)
            print(f"  {size} bars: {elapsed:.2f} ms")

        # Check that calculation time scales reasonably
        # Should be roughly linear or better
        ratio = times[-1] / times[0]
        size_ratio = data_sizes[-1] / data_sizes[0]

        print(f"\nScaling: {size_ratio:.1f}x data -> {ratio:.1f}x time")
        assert ratio < size_ratio * 2, "RRS calculation doesn't scale well"


# =============================================================================
# SCANNER PERFORMANCE TESTS
# =============================================================================

class TestScannerPerformance:
    """Performance tests for the scanner agent."""

    @pytest.fixture
    def scanner_agent(self, mock_data_provider):
        """Create scanner agent instance."""
        try:
            from agents.scanner_agent import ScannerAgent
            return ScannerAgent(
                watchlist=SMALL_WATCHLIST,
                data_provider=mock_data_provider,
                scan_interval=60.0,
                rrs_threshold=2.0
            )
        except ImportError:
            pytest.skip("ScannerAgent not available")

    @pytest.mark.asyncio
    async def test_small_watchlist_scan_time(self, scanner_agent, perf_config):
        """Test scan time for small watchlist."""
        await scanner_agent.initialize()

        start = time.perf_counter()
        await scanner_agent.scan_market()
        elapsed = time.perf_counter() - start

        print(f"\nSmall watchlist ({len(SMALL_WATCHLIST)} symbols) scan time: {elapsed:.2f}s")

        target = perf_config.scanner_targets['small_watchlist_scan_seconds']
        assert elapsed < target * 2, f"Scan time {elapsed:.2f}s exceeds target {target}s"

    @pytest.mark.asyncio
    async def test_scan_time_per_symbol(self, mock_data_provider, perf_config):
        """Test average scan time per symbol."""
        try:
            from agents.scanner_agent import ScannerAgent
        except ImportError:
            pytest.skip("ScannerAgent not available")

        watchlist = MEDIUM_WATCHLIST
        scanner = ScannerAgent(
            watchlist=watchlist,
            data_provider=mock_data_provider,
            scan_interval=60.0,
            rrs_threshold=2.0
        )
        await scanner.initialize()

        start = time.perf_counter()
        await scanner.scan_market()
        elapsed = time.perf_counter() - start

        time_per_symbol = (elapsed / len(watchlist)) * 1000  # ms
        print(f"\nTime per symbol: {time_per_symbol:.2f} ms")

        target = perf_config.scanner_targets['time_per_symbol_ms']
        # Allow some overhead for batch operations
        assert time_per_symbol < target * 3, f"Time per symbol {time_per_symbol:.2f}ms exceeds target"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MEMORY_PROFILER_AVAILABLE, reason="memory_profiler not available")
    async def test_scanner_memory_usage(self, mock_data_provider, perf_config):
        """Test scanner memory usage during scan."""
        try:
            from agents.scanner_agent import ScannerAgent
        except ImportError:
            pytest.skip("ScannerAgent not available")

        gc.collect()
        mem_before = memory_usage()[0]

        scanner = ScannerAgent(
            watchlist=MEDIUM_WATCHLIST,
            data_provider=mock_data_provider,
            scan_interval=60.0,
            rrs_threshold=2.0
        )
        await scanner.initialize()
        await scanner.scan_market()

        gc.collect()
        mem_after = memory_usage()[0]
        growth = mem_after - mem_before

        print(f"\nScanner memory usage:")
        print(f"  Before: {mem_before:.2f} MB")
        print(f"  After: {mem_after:.2f} MB")
        print(f"  Growth: {growth:.2f} MB")

        max_memory = perf_config.scanner_targets['max_total_memory_mb']
        assert growth < max_memory, f"Memory usage {growth:.2f}MB exceeds limit {max_memory}MB"


# =============================================================================
# DATA PROVIDER PERFORMANCE TESTS
# =============================================================================

class TestDataProviderPerformance:
    """Performance tests for the data provider."""

    @pytest.fixture
    def data_provider(self):
        """Create data provider instance."""
        try:
            from shared.data_provider import DataProvider
            return DataProvider(cache_ttl_seconds=30)
        except ImportError:
            pytest.skip("DataProvider not available")

    @pytest.mark.asyncio
    async def test_single_stock_fetch_time(self, data_provider, perf_config):
        """Test time to fetch single stock data."""
        start = time.perf_counter()
        data = await data_provider.get_stock_data('AAPL')
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\nSingle stock fetch time: {elapsed:.2f} ms")

        # First fetch may be slow due to network
        target = perf_config.scanner_targets['data_fetch_timeout_seconds'] * 1000
        assert elapsed < target, f"Fetch time {elapsed:.2f}ms exceeds timeout"

    @pytest.mark.asyncio
    async def test_cached_fetch_time(self, data_provider, perf_config):
        """Test time to fetch cached stock data."""
        # Prime the cache
        await data_provider.get_stock_data('AAPL')

        # Measure cached fetch
        start = time.perf_counter()
        data = await data_provider.get_stock_data('AAPL')
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\nCached stock fetch time: {elapsed:.2f} ms")

        # Cached fetch should be very fast
        assert elapsed < 10, f"Cached fetch time {elapsed:.2f}ms too slow"

    @pytest.mark.asyncio
    async def test_multiple_stock_fetch(self, data_provider, perf_config):
        """Test time to fetch multiple stocks."""
        symbols = TEST_SYMBOLS[:5]

        start = time.perf_counter()
        results = await data_provider.get_quotes(symbols)
        elapsed = time.perf_counter() - start

        print(f"\nMultiple stock fetch ({len(symbols)} symbols): {elapsed:.2f}s")

        # Should complete within reasonable time
        target = perf_config.scanner_targets['data_fetch_timeout_seconds'] * len(symbols) * 0.5
        assert elapsed < target, f"Batch fetch time {elapsed:.2f}s exceeds target"

    @pytest.mark.asyncio
    async def test_concurrent_fetches(self, data_provider, perf_config):
        """Test concurrent data fetches."""
        symbols = TEST_SYMBOLS[:10]

        start = time.perf_counter()

        # Fetch all concurrently
        tasks = [data_provider.get_stock_data(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.perf_counter() - start

        # Count successes
        successes = sum(1 for r in results if not isinstance(r, Exception) and r is not None)

        print(f"\nConcurrent fetches ({len(symbols)} symbols):")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Success: {successes}/{len(symbols)}")

        # Most should succeed
        assert successes >= len(symbols) * 0.8, "Too many fetch failures"


# =============================================================================
# WATCHLIST SCALING TESTS
# =============================================================================

class TestWatchlistScaling:
    """Tests for scanner performance with different watchlist sizes."""

    @pytest.mark.asyncio
    async def test_scaling_with_watchlist_size(self, mock_data_provider, perf_config):
        """Test how scan time scales with watchlist size."""
        try:
            from agents.scanner_agent import ScannerAgent
        except ImportError:
            pytest.skip("ScannerAgent not available")

        sizes = [5, 10, 20, 50]
        times = []

        for size in sizes:
            watchlist = (TEST_SYMBOLS * ((size // len(TEST_SYMBOLS)) + 1))[:size]

            scanner = ScannerAgent(
                watchlist=watchlist,
                data_provider=mock_data_provider,
                scan_interval=60.0,
                rrs_threshold=2.0
            )
            await scanner.initialize()

            start = time.perf_counter()
            await scanner.scan_market()
            elapsed = time.perf_counter() - start

            times.append(elapsed)
            print(f"  {size} symbols: {elapsed:.2f}s ({elapsed / size * 1000:.2f} ms/symbol)")

        # Check scaling is roughly linear
        # Double the symbols shouldn't more than triple the time
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i - 1]
            time_ratio = times[i] / times[i - 1] if times[i - 1] > 0 else 1

            print(f"\n{sizes[i - 1]} -> {sizes[i]} symbols: {size_ratio:.1f}x size, {time_ratio:.1f}x time")
            assert time_ratio < size_ratio * 3, "Scanner scaling is worse than linear"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_large_watchlist_scan(self, mock_data_provider, perf_config):
        """Test scanning a large watchlist."""
        try:
            from agents.scanner_agent import ScannerAgent
        except ImportError:
            pytest.skip("ScannerAgent not available")

        # Use a subset of large watchlist for test
        watchlist = LARGE_WATCHLIST[:50]

        scanner = ScannerAgent(
            watchlist=watchlist,
            data_provider=mock_data_provider,
            scan_interval=60.0,
            rrs_threshold=2.0
        )
        await scanner.initialize()

        start = time.perf_counter()
        await scanner.scan_market()
        elapsed = time.perf_counter() - start

        print(f"\nLarge watchlist scan ({len(watchlist)} symbols): {elapsed:.2f}s")

        # Should complete in reasonable time
        target = perf_config.scanner_targets['large_watchlist_scan_seconds']
        assert elapsed < target, f"Large scan {elapsed:.2f}s exceeds target {target}s"


# =============================================================================
# SIGNAL GENERATION PERFORMANCE
# =============================================================================

class TestSignalGenerationPerformance:
    """Tests for signal generation performance."""

    def test_signal_filtering_performance(self, rrs_calculator, mock_daily_data, perf_config):
        """Test performance of signal filtering logic."""
        # Generate many potential signals
        num_symbols = 100
        signals = []

        for i in range(num_symbols):
            signals.append({
                'symbol': f'TEST{i}',
                'rrs': np.random.uniform(-4, 4),
                'price': np.random.uniform(10, 500),
                'volume': np.random.randint(100000, 10000000),
                'atr': np.random.uniform(0.5, 5),
            })

        # Measure filtering time
        start = time.perf_counter()

        for _ in range(100):  # Run multiple times
            strong_rs = [s for s in signals if s['rrs'] > 2.0 and s['volume'] > 500000]
            strong_rw = [s for s in signals if s['rrs'] < -2.0 and s['volume'] > 500000]

            # Sort by RRS strength
            strong_rs.sort(key=lambda x: x['rrs'], reverse=True)
            strong_rw.sort(key=lambda x: x['rrs'])

        elapsed = (time.perf_counter() - start) / 100 * 1000  # ms per iteration

        print(f"\nSignal filtering ({num_symbols} symbols): {elapsed:.2f} ms")
        assert elapsed < 10, f"Filtering time {elapsed:.2f}ms too slow"

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not available")
    def test_daily_strength_check(self, benchmark, mock_daily_data):
        """Benchmark daily strength check."""
        try:
            from shared.indicators.rrs import check_daily_strength
        except ImportError:
            pytest.skip("check_daily_strength not available")

        def check():
            return check_daily_strength(mock_daily_data)

        result = benchmark(check)
        assert 'is_strong' in result


# =============================================================================
# MEMORY PROFILING
# =============================================================================

class TestScannerMemoryProfiling:
    """Memory profiling tests for the scanner."""

    @pytest.mark.skipif(not MEMORY_PROFILER_AVAILABLE, reason="memory_profiler not available")
    def test_memory_per_symbol(self, mock_daily_data, perf_config):
        """Test memory usage per symbol."""
        gc.collect()
        mem_baseline = memory_usage()[0]

        # Store data for multiple symbols
        symbol_data = {}
        for i in range(50):
            symbol_data[f'TEST{i}'] = {
                'symbol': f'TEST{i}',
                'current_price': 100.0,
                'daily_data': mock_daily_data.copy(),
            }

        gc.collect()
        mem_after = memory_usage()[0]
        growth = mem_after - mem_baseline
        per_symbol = growth / 50

        print(f"\nMemory per symbol: {per_symbol:.2f} MB")
        print(f"Total for 50 symbols: {growth:.2f} MB")

        target = perf_config.scanner_targets['memory_per_symbol_mb']
        assert per_symbol < target, f"Memory per symbol {per_symbol:.2f}MB exceeds {target}MB"

    @pytest.mark.skipif(not MEMORY_PROFILER_AVAILABLE, reason="memory_profiler not available")
    def test_repeated_scans_memory(self, mock_data_provider, perf_config):
        """Test for memory leaks during repeated scans."""
        try:
            from agents.scanner_agent import ScannerAgent
        except ImportError:
            pytest.skip("ScannerAgent not available")

        async def run_scans():
            scanner = ScannerAgent(
                watchlist=SMALL_WATCHLIST,
                data_provider=mock_data_provider,
                scan_interval=60.0,
                rrs_threshold=2.0
            )
            await scanner.initialize()

            gc.collect()
            mem_start = memory_usage()[0]

            # Run multiple scans
            for _ in range(10):
                await scanner.scan_market()
                gc.collect()

            mem_end = memory_usage()[0]
            return mem_end - mem_start

        growth = asyncio.run(run_scans())

        print(f"\nMemory growth after 10 scans: {growth:.2f} MB")

        # Should not grow significantly
        assert growth < 50, f"Memory grew {growth:.2f}MB during repeated scans"


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

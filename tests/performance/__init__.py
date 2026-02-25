"""
RDT Trading System Performance Tests

This module contains performance and load testing for the RDT Trading System.

Test modules:
- locustfile.py: Locust load testing configuration
- test_api_performance.py: pytest-benchmark tests for API endpoints
- test_scanner_performance.py: Scanner performance benchmarks
- test_ml_performance.py: ML model inference benchmarks
- config.py: Performance test configuration

Usage:
    # Run API benchmarks
    pytest tests/performance/test_api_performance.py -v

    # Run load tests with Locust
    locust -f tests/performance/locustfile.py --host=http://localhost:5000

    # Run all performance tests
    pytest tests/performance/ -v --benchmark-only
"""

from tests.performance.config import (
    PerformanceConfig,
    get_performance_config,
    TargetMetrics,
    ResourceLimits,
)

__all__ = [
    'PerformanceConfig',
    'get_performance_config',
    'TargetMetrics',
    'ResourceLimits',
]

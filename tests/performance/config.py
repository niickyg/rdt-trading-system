"""
Performance Test Configuration

Centralized configuration for performance testing including:
- Target response times for API endpoints
- Acceptable error rates
- Resource limits (CPU, memory)
- Load testing parameters
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class EndpointCategory(Enum):
    """Categories of API endpoints with different performance targets."""
    FAST = "fast"          # Health checks, simple reads
    NORMAL = "normal"      # Standard API calls
    HEAVY = "heavy"        # Backtests, batch operations
    REALTIME = "realtime"  # WebSocket, streaming


@dataclass
class TargetMetrics:
    """Target performance metrics for API endpoints."""

    # Response times in milliseconds
    p50_response_ms: float = 100.0    # 50th percentile (median)
    p95_response_ms: float = 500.0    # 95th percentile
    p99_response_ms: float = 1000.0   # 99th percentile
    max_response_ms: float = 5000.0   # Maximum acceptable

    # Error rates
    max_error_rate: float = 0.01      # 1% maximum error rate
    max_timeout_rate: float = 0.005   # 0.5% timeout rate

    # Throughput
    min_requests_per_second: float = 10.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'p50_response_ms': self.p50_response_ms,
            'p95_response_ms': self.p95_response_ms,
            'p99_response_ms': self.p99_response_ms,
            'max_response_ms': self.max_response_ms,
            'max_error_rate': self.max_error_rate,
            'max_timeout_rate': self.max_timeout_rate,
            'min_requests_per_second': self.min_requests_per_second,
        }


@dataclass
class ResourceLimits:
    """Resource usage limits for performance tests."""

    # Memory limits in MB
    max_memory_mb: float = 512.0
    max_memory_growth_mb: float = 100.0    # Max growth during test

    # CPU limits (percentage)
    max_cpu_percent: float = 80.0

    # Connection limits
    max_concurrent_connections: int = 100
    max_websocket_connections: int = 50

    # Database limits
    max_db_connections: int = 20
    max_query_time_ms: float = 100.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'max_memory_mb': self.max_memory_mb,
            'max_memory_growth_mb': self.max_memory_growth_mb,
            'max_cpu_percent': self.max_cpu_percent,
            'max_concurrent_connections': self.max_concurrent_connections,
            'max_websocket_connections': self.max_websocket_connections,
            'max_db_connections': self.max_db_connections,
            'max_query_time_ms': self.max_query_time_ms,
        }


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""

    # User counts
    min_users: int = 1
    max_users: int = 100
    spawn_rate: float = 5.0      # Users per second

    # Test duration
    ramp_up_time_seconds: int = 30
    steady_state_seconds: int = 60
    ramp_down_time_seconds: int = 10

    # Request parameters
    request_timeout_seconds: float = 30.0
    think_time_min_ms: int = 100
    think_time_max_ms: int = 1000

    # Failure thresholds
    max_failure_rate: float = 0.05  # 5% failure rate triggers stop

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'min_users': self.min_users,
            'max_users': self.max_users,
            'spawn_rate': self.spawn_rate,
            'ramp_up_time_seconds': self.ramp_up_time_seconds,
            'steady_state_seconds': self.steady_state_seconds,
            'ramp_down_time_seconds': self.ramp_down_time_seconds,
            'request_timeout_seconds': self.request_timeout_seconds,
            'think_time_min_ms': self.think_time_min_ms,
            'think_time_max_ms': self.think_time_max_ms,
            'max_failure_rate': self.max_failure_rate,
        }


@dataclass
class PerformanceConfig:
    """
    Main performance test configuration.

    This configuration defines performance targets and limits for
    all performance tests in the RDT Trading System.
    """

    # Base URL for API tests
    base_url: str = field(default_factory=lambda: os.getenv('API_BASE_URL', 'http://localhost:5000'))

    # WebSocket URL
    websocket_url: str = field(default_factory=lambda: os.getenv('WS_BASE_URL', 'ws://localhost:5000'))

    # API key for authenticated endpoints
    api_key: str = field(default_factory=lambda: os.getenv('TEST_API_KEY', 'test-api-key'))

    # Endpoint-specific targets
    endpoint_targets: Dict[str, TargetMetrics] = field(default_factory=dict)

    # Resource limits
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)

    # Load test configuration
    load_test_config: LoadTestConfig = field(default_factory=LoadTestConfig)

    # Scanner performance targets
    scanner_targets: Dict[str, float] = field(default_factory=dict)

    # ML model performance targets
    ml_targets: Dict[str, float] = field(default_factory=dict)

    # Benchmark configuration
    benchmark_rounds: int = 10
    benchmark_warmup_rounds: int = 2

    def __post_init__(self):
        """Initialize default targets after instance creation."""
        if not self.endpoint_targets:
            self._set_default_endpoint_targets()
        if not self.scanner_targets:
            self._set_default_scanner_targets()
        if not self.ml_targets:
            self._set_default_ml_targets()

    def _set_default_endpoint_targets(self):
        """Set default performance targets for each endpoint category."""
        # Fast endpoints (health, status)
        self.endpoint_targets['health'] = TargetMetrics(
            p50_response_ms=20.0,
            p95_response_ms=50.0,
            p99_response_ms=100.0,
            max_response_ms=200.0,
            min_requests_per_second=100.0
        )

        self.endpoint_targets['status'] = TargetMetrics(
            p50_response_ms=50.0,
            p95_response_ms=100.0,
            p99_response_ms=200.0,
            max_response_ms=500.0,
            min_requests_per_second=50.0
        )

        # Normal endpoints (signals, positions)
        self.endpoint_targets['signals'] = TargetMetrics(
            p50_response_ms=100.0,
            p95_response_ms=300.0,
            p99_response_ms=500.0,
            max_response_ms=1000.0,
            min_requests_per_second=20.0
        )

        self.endpoint_targets['positions'] = TargetMetrics(
            p50_response_ms=100.0,
            p95_response_ms=300.0,
            p99_response_ms=500.0,
            max_response_ms=1000.0,
            min_requests_per_second=20.0
        )

        self.endpoint_targets['dashboard'] = TargetMetrics(
            p50_response_ms=200.0,
            p95_response_ms=500.0,
            p99_response_ms=1000.0,
            max_response_ms=2000.0,
            min_requests_per_second=10.0
        )

        # Heavy endpoints (backtest, orders)
        self.endpoint_targets['backtest'] = TargetMetrics(
            p50_response_ms=2000.0,
            p95_response_ms=5000.0,
            p99_response_ms=10000.0,
            max_response_ms=30000.0,
            min_requests_per_second=1.0
        )

        self.endpoint_targets['order'] = TargetMetrics(
            p50_response_ms=500.0,
            p95_response_ms=1000.0,
            p99_response_ms=2000.0,
            max_response_ms=5000.0,
            min_requests_per_second=5.0
        )

        # WebSocket endpoints
        self.endpoint_targets['websocket_connect'] = TargetMetrics(
            p50_response_ms=100.0,
            p95_response_ms=300.0,
            p99_response_ms=500.0,
            max_response_ms=1000.0,
            min_requests_per_second=10.0
        )

        self.endpoint_targets['websocket_subscribe'] = TargetMetrics(
            p50_response_ms=50.0,
            p95_response_ms=100.0,
            p99_response_ms=200.0,
            max_response_ms=500.0,
            min_requests_per_second=20.0
        )

    def _set_default_scanner_targets(self):
        """Set default scanner performance targets."""
        self.scanner_targets = {
            # Time per symbol in milliseconds
            'time_per_symbol_ms': 100.0,

            # Scan durations in seconds
            'small_watchlist_scan_seconds': 5.0,     # 50 symbols
            'medium_watchlist_scan_seconds': 15.0,   # 175 symbols
            'large_watchlist_scan_seconds': 60.0,    # 500 symbols

            # Memory usage
            'memory_per_symbol_mb': 2.0,
            'max_total_memory_mb': 500.0,

            # Data provider limits
            'data_fetch_timeout_seconds': 5.0,
            'max_concurrent_fetches': 10,
        }

    def _set_default_ml_targets(self):
        """Set default ML model performance targets."""
        self.ml_targets = {
            # Single prediction time in milliseconds
            'single_inference_ms': 10.0,

            # Batch prediction (100 samples)
            'batch_100_inference_ms': 50.0,

            # Batch prediction (1000 samples)
            'batch_1000_inference_ms': 200.0,

            # Feature engineering time per sample
            'feature_engineering_ms': 20.0,

            # Model loading time
            'model_load_seconds': 2.0,

            # Memory usage
            'model_memory_mb': 100.0,
            'inference_memory_overhead_mb': 50.0,
        }

    def get_target(self, endpoint: str) -> TargetMetrics:
        """Get performance target for an endpoint."""
        return self.endpoint_targets.get(endpoint, TargetMetrics())

    def to_dict(self) -> Dict:
        """Convert entire configuration to dictionary."""
        return {
            'base_url': self.base_url,
            'websocket_url': self.websocket_url,
            'endpoint_targets': {k: v.to_dict() for k, v in self.endpoint_targets.items()},
            'resource_limits': self.resource_limits.to_dict(),
            'load_test_config': self.load_test_config.to_dict(),
            'scanner_targets': self.scanner_targets,
            'ml_targets': self.ml_targets,
            'benchmark_rounds': self.benchmark_rounds,
            'benchmark_warmup_rounds': self.benchmark_warmup_rounds,
        }


# Global configuration instance
_config: Optional[PerformanceConfig] = None


def get_performance_config() -> PerformanceConfig:
    """
    Get the global performance configuration instance.

    Creates a new instance if one doesn't exist.
    """
    global _config
    if _config is None:
        _config = PerformanceConfig()
    return _config


def set_performance_config(config: PerformanceConfig) -> None:
    """Set the global performance configuration instance."""
    global _config
    _config = config


# Test symbols for performance testing
TEST_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'NVDA', 'TSLA', 'JPM', 'V', 'JNJ',
    'WMT', 'PG', 'MA', 'UNH', 'HD',
    'DIS', 'PYPL', 'BAC', 'ADBE', 'CRM',
]

# Small watchlist for quick tests
SMALL_WATCHLIST = TEST_SYMBOLS[:10]

# Medium watchlist for standard tests
MEDIUM_WATCHLIST = TEST_SYMBOLS

# Large watchlist for stress tests
LARGE_WATCHLIST = TEST_SYMBOLS * 5  # 100 symbols


# API endpoint definitions for testing
API_ENDPOINTS = {
    'health': {
        'method': 'GET',
        'path': '/api/v1/health',
        'auth_required': False,
        'category': EndpointCategory.FAST,
    },
    'status': {
        'method': 'GET',
        'path': '/api/v1/status',
        'auth_required': False,
        'category': EndpointCategory.FAST,
    },
    'dashboard': {
        'method': 'GET',
        'path': '/api/v1/dashboard',
        'auth_required': False,
        'category': EndpointCategory.NORMAL,
    },
    'signals_current': {
        'method': 'GET',
        'path': '/api/v1/signals/current',
        'auth_required': True,
        'category': EndpointCategory.NORMAL,
    },
    'signals_history': {
        'method': 'GET',
        'path': '/api/v1/signals/history',
        'auth_required': True,
        'category': EndpointCategory.NORMAL,
    },
    'positions': {
        'method': 'GET',
        'path': '/api/v1/positions',
        'auth_required': True,
        'category': EndpointCategory.NORMAL,
    },
    'performance': {
        'method': 'GET',
        'path': '/api/v1/performance',
        'auth_required': True,
        'category': EndpointCategory.NORMAL,
    },
    'backtest': {
        'method': 'POST',
        'path': '/api/v1/backtest',
        'auth_required': True,
        'category': EndpointCategory.HEAVY,
    },
    'order': {
        'method': 'POST',
        'path': '/api/v1/orders',
        'auth_required': True,
        'category': EndpointCategory.HEAVY,
    },
}

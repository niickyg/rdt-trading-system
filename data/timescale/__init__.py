"""
TimescaleDB integration for the RDT Trading System.

This module provides TimescaleDB-specific functionality for time-series optimization:
- Hypertable creation and management
- Compression policies
- Retention policies
- Optimized time-series queries

TimescaleDB extends PostgreSQL with time-series capabilities:
- Automatic partitioning via hypertables
- Built-in compression for storage efficiency
- Time-series optimized functions (time_bucket, etc.)
- Continuous aggregates for pre-computed analytics

Usage:
    from data.timescale import (
        TimescaleManager,
        check_timescale_available,
        get_signals_time_bucket,
        get_performance_over_time
    )

    # Check if TimescaleDB is available
    if check_timescale_available():
        manager = TimescaleManager()
        manager.setup_hypertables()
"""

from .setup import (
    TimescaleManager,
    check_timescale_available,
    create_hypertable,
    add_compression_policy,
    add_retention_policy,
    get_timescale_info,
)

from .queries import (
    get_signals_time_bucket,
    get_performance_over_time,
    get_continuous_aggregate,
    get_market_data_aggregated,
    get_order_execution_stats,
)

from .migrations import (
    migrate_to_hypertables,
    rollback_hypertables,
    get_migration_status,
)

__all__ = [
    # Setup functions
    'TimescaleManager',
    'check_timescale_available',
    'create_hypertable',
    'add_compression_policy',
    'add_retention_policy',
    'get_timescale_info',

    # Query functions
    'get_signals_time_bucket',
    'get_performance_over_time',
    'get_continuous_aggregate',
    'get_market_data_aggregated',
    'get_order_execution_stats',

    # Migration functions
    'migrate_to_hypertables',
    'rollback_hypertables',
    'get_migration_status',
]

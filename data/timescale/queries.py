"""
Optimized TimescaleDB queries for the RDT Trading System.

This module provides time-series optimized query functions that leverage
TimescaleDB-specific functions like time_bucket, first(), last(), etc.

All functions gracefully fall back to standard SQL when TimescaleDB is not available.
"""

from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta, date
from decimal import Decimal

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from .setup import check_timescale_available

from loguru import logger


# =============================================================================
# Signal Queries
# =============================================================================

def get_signals_time_bucket(
    interval: str = '1 hour',
    symbol: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000,
    engine: Optional[Engine] = None
) -> List[Dict[str, Any]]:
    """
    Get aggregated signals using time_bucket for time-series analysis.

    Args:
        interval: Time bucket interval (e.g., '1 hour', '1 day', '15 minutes').
        symbol: Optional symbol to filter by.
        start_time: Optional start time filter.
        end_time: Optional end time filter.
        limit: Maximum number of results.
        engine: SQLAlchemy engine.

    Returns:
        List of dicts with aggregated signal data per time bucket.
    """
    if engine is None:
        from data.database.connection import get_db_manager
        manager = get_db_manager()
        engine = manager.engine

    use_timescale = check_timescale_available(engine)

    # Build the query
    if use_timescale:
        # TimescaleDB optimized query
        bucket_sql = f"time_bucket(INTERVAL '{interval}', timestamp)"
        first_last = """
            first(rrs, timestamp) as first_rrs,
            last(rrs, timestamp) as last_rrs,
        """
    else:
        # Standard PostgreSQL fallback using date_trunc
        # Convert interval to date_trunc unit
        trunc_unit = _interval_to_trunc_unit(interval)
        bucket_sql = f"date_trunc('{trunc_unit}', timestamp)"
        first_last = """
            MIN(rrs) as first_rrs,
            MAX(rrs) as last_rrs,
        """

    where_clauses = ["1=1"]
    params = {'limit': limit}

    if symbol:
        where_clauses.append("symbol = :symbol")
        params['symbol'] = symbol
    if start_time:
        where_clauses.append("timestamp >= :start_time")
        params['start_time'] = start_time
    if end_time:
        where_clauses.append("timestamp <= :end_time")
        params['end_time'] = end_time

    where_sql = " AND ".join(where_clauses)

    query = text(f"""
        SELECT
            {bucket_sql} as bucket,
            COUNT(*) as signal_count,
            COUNT(DISTINCT symbol) as unique_symbols,
            AVG(rrs) as avg_rrs,
            MIN(rrs) as min_rrs,
            MAX(rrs) as max_rrs,
            {first_last}
            COUNT(*) FILTER (WHERE direction = 'long') as long_signals,
            COUNT(*) FILTER (WHERE direction = 'short') as short_signals,
            COUNT(*) FILTER (WHERE status = 'triggered') as triggered_count
        FROM signals
        WHERE {where_sql}
        GROUP BY bucket
        ORDER BY bucket DESC
        LIMIT :limit
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query, params)
            rows = []
            for row in result:
                rows.append({
                    'bucket': row[0],
                    'signal_count': row[1],
                    'unique_symbols': row[2],
                    'avg_rrs': float(row[3]) if row[3] else None,
                    'min_rrs': float(row[4]) if row[4] else None,
                    'max_rrs': float(row[5]) if row[5] else None,
                    'first_rrs': float(row[6]) if row[6] else None,
                    'last_rrs': float(row[7]) if row[7] else None,
                    'long_signals': row[8],
                    'short_signals': row[9],
                    'triggered_count': row[10]
                })
            return rows
    except Exception as e:
        logger.error(f"Error in get_signals_time_bucket: {e}")
        return []


# =============================================================================
# Performance Queries
# =============================================================================

def get_performance_over_time(
    period: str = '1 day',
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    include_cumulative: bool = True,
    engine: Optional[Engine] = None
) -> List[Dict[str, Any]]:
    """
    Get P&L performance over time using time-series aggregation.

    Args:
        period: Aggregation period ('1 day', '1 week', '1 month').
        start_date: Optional start date filter.
        end_date: Optional end date filter.
        include_cumulative: Whether to include cumulative P&L.
        engine: SQLAlchemy engine.

    Returns:
        List of dicts with performance metrics per period.
    """
    if engine is None:
        from data.database.connection import get_db_manager
        manager = get_db_manager()
        engine = manager.engine

    use_timescale = check_timescale_available(engine)

    # Build bucket expression
    if use_timescale:
        bucket_sql = f"time_bucket(INTERVAL '{period}', date)"
    else:
        trunc_unit = _interval_to_trunc_unit(period)
        bucket_sql = f"date_trunc('{trunc_unit}', date)"

    where_clauses = ["1=1"]
    params = {}

    if start_date:
        where_clauses.append("date >= :start_date")
        params['start_date'] = start_date
    if end_date:
        where_clauses.append("date <= :end_date")
        params['end_date'] = end_date

    where_sql = " AND ".join(where_clauses)

    # Build cumulative column
    cumulative_sql = ""
    if include_cumulative:
        cumulative_sql = """
            , SUM(SUM(pnl)) OVER (ORDER BY bucket) as cumulative_pnl
            , SUM(SUM(num_trades)) OVER (ORDER BY bucket) as cumulative_trades
        """

    query = text(f"""
        SELECT
            {bucket_sql} as bucket,
            SUM(pnl) as total_pnl,
            AVG(pnl_percent) as avg_pnl_percent,
            SUM(num_trades) as total_trades,
            SUM(winners) as total_winners,
            SUM(losers) as total_losers,
            CASE WHEN SUM(winners) + SUM(losers) > 0
                 THEN ROUND(100.0 * SUM(winners) / (SUM(winners) + SUM(losers)), 2)
                 ELSE 0 END as win_rate,
            AVG(avg_win) as avg_win,
            AVG(avg_loss) as avg_loss,
            MAX(largest_win) as best_trade,
            MIN(largest_loss) as worst_trade
            {cumulative_sql}
        FROM daily_stats
        WHERE {where_sql}
        GROUP BY bucket
        ORDER BY bucket
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query, params)
            rows = []
            for row in result:
                data = {
                    'period': row[0],
                    'total_pnl': float(row[1]) if row[1] else 0,
                    'avg_pnl_percent': float(row[2]) if row[2] else 0,
                    'total_trades': row[3] or 0,
                    'total_winners': row[4] or 0,
                    'total_losers': row[5] or 0,
                    'win_rate': float(row[6]) if row[6] else 0,
                    'avg_win': float(row[7]) if row[7] else 0,
                    'avg_loss': float(row[8]) if row[8] else 0,
                    'best_trade': float(row[9]) if row[9] else 0,
                    'worst_trade': float(row[10]) if row[10] else 0,
                }
                if include_cumulative:
                    data['cumulative_pnl'] = float(row[11]) if row[11] else 0
                    data['cumulative_trades'] = row[12] or 0
                rows.append(data)
            return rows
    except Exception as e:
        logger.error(f"Error in get_performance_over_time: {e}")
        return []


def get_continuous_aggregate(
    metric: str,
    interval: str = '1 hour',
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    symbol: Optional[str] = None,
    limit: int = 1000,
    engine: Optional[Engine] = None
) -> List[Dict[str, Any]]:
    """
    Get pre-aggregated data from continuous aggregates.

    If continuous aggregates are not set up, falls back to real-time aggregation.

    Args:
        metric: Metric type ('signals', 'performance', 'market_data', 'executions').
        interval: Time bucket interval.
        start_time: Optional start time filter.
        end_time: Optional end time filter.
        symbol: Optional symbol filter.
        limit: Maximum number of results.
        engine: SQLAlchemy engine.

    Returns:
        List of aggregated metric data.
    """
    if engine is None:
        from data.database.connection import get_db_manager
        manager = get_db_manager()
        engine = manager.engine

    # Map metrics to their aggregation queries
    metric_queries = {
        'signals': lambda: get_signals_time_bucket(
            interval, symbol, start_time, end_time, limit, engine
        ),
        'performance': lambda: get_performance_over_time(
            interval,
            start_time.date() if start_time else None,
            end_time.date() if end_time else None,
            True, engine
        ),
        'market_data': lambda: get_market_data_aggregated(
            interval, symbol, start_time, end_time, limit, engine
        ),
        'executions': lambda: get_order_execution_stats(
            interval, symbol, start_time, end_time, limit, engine
        ),
    }

    if metric not in metric_queries:
        logger.warning(f"Unknown metric: {metric}")
        return []

    # Check for continuous aggregate view
    view_name = f"cagg_{metric}_{interval.replace(' ', '_')}"
    has_cagg = _check_continuous_aggregate_exists(view_name, engine)

    if has_cagg and check_timescale_available(engine):
        # Query from continuous aggregate
        return _query_continuous_aggregate(
            view_name, start_time, end_time, symbol, limit, engine
        )
    else:
        # Fall back to real-time aggregation
        return metric_queries[metric]()


# =============================================================================
# Market Data Queries
# =============================================================================

def get_market_data_aggregated(
    interval: str = '1 hour',
    symbol: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000,
    engine: Optional[Engine] = None
) -> List[Dict[str, Any]]:
    """
    Get aggregated market data from cache using time buckets.

    Args:
        interval: Time bucket interval.
        symbol: Optional symbol filter.
        start_time: Optional start time filter.
        end_time: Optional end time filter.
        limit: Maximum number of results.
        engine: SQLAlchemy engine.

    Returns:
        List of aggregated market data.
    """
    if engine is None:
        from data.database.connection import get_db_manager
        manager = get_db_manager()
        engine = manager.engine

    use_timescale = check_timescale_available(engine)

    if use_timescale:
        bucket_sql = f"time_bucket(INTERVAL '{interval}', timestamp)"
    else:
        trunc_unit = _interval_to_trunc_unit(interval)
        bucket_sql = f"date_trunc('{trunc_unit}', timestamp)"

    where_clauses = ["1=1"]
    params = {'limit': limit}

    if symbol:
        where_clauses.append("symbol = :symbol")
        params['symbol'] = symbol
    if start_time:
        where_clauses.append("timestamp >= :start_time")
        params['start_time'] = start_time
    if end_time:
        where_clauses.append("timestamp <= :end_time")
        params['end_time'] = end_time

    where_sql = " AND ".join(where_clauses)

    query = text(f"""
        SELECT
            {bucket_sql} as bucket,
            symbol,
            data_type,
            COUNT(*) as record_count,
            MIN(timestamp) as first_record,
            MAX(timestamp) as last_record
        FROM market_data_cache
        WHERE {where_sql}
        GROUP BY bucket, symbol, data_type
        ORDER BY bucket DESC, symbol
        LIMIT :limit
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query, params)
            rows = []
            for row in result:
                rows.append({
                    'bucket': row[0],
                    'symbol': row[1],
                    'data_type': row[2],
                    'record_count': row[3],
                    'first_record': row[4],
                    'last_record': row[5]
                })
            return rows
    except Exception as e:
        logger.error(f"Error in get_market_data_aggregated: {e}")
        return []


# =============================================================================
# Order Execution Queries
# =============================================================================

def get_order_execution_stats(
    interval: str = '1 day',
    symbol: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000,
    engine: Optional[Engine] = None
) -> List[Dict[str, Any]]:
    """
    Get aggregated order execution statistics using time buckets.

    Args:
        interval: Time bucket interval.
        symbol: Optional symbol filter.
        start_time: Optional start time filter.
        end_time: Optional end time filter.
        limit: Maximum number of results.
        engine: SQLAlchemy engine.

    Returns:
        List of execution statistics per time bucket.
    """
    if engine is None:
        from data.database.connection import get_db_manager
        manager = get_db_manager()
        engine = manager.engine

    use_timescale = check_timescale_available(engine)

    if use_timescale:
        bucket_sql = f"time_bucket(INTERVAL '{interval}', fill_time)"
    else:
        trunc_unit = _interval_to_trunc_unit(interval)
        bucket_sql = f"date_trunc('{trunc_unit}', fill_time)"

    where_clauses = ["1=1"]
    params = {'limit': limit}

    if symbol:
        where_clauses.append("symbol = :symbol")
        params['symbol'] = symbol
    if start_time:
        where_clauses.append("fill_time >= :start_time")
        params['start_time'] = start_time
    if end_time:
        where_clauses.append("fill_time <= :end_time")
        params['end_time'] = end_time

    where_sql = " AND ".join(where_clauses)

    query = text(f"""
        SELECT
            {bucket_sql} as bucket,
            COUNT(*) as execution_count,
            COUNT(DISTINCT symbol) as unique_symbols,
            SUM(quantity) as total_shares,
            AVG(slippage) as avg_slippage,
            AVG(slippage_pct) as avg_slippage_pct,
            MAX(slippage_pct) as max_slippage_pct,
            AVG(time_to_fill_seconds) as avg_fill_time,
            COUNT(*) FILTER (WHERE status = 'filled') as filled_count,
            COUNT(*) FILTER (WHERE status = 'partial_fill') as partial_count,
            COUNT(*) FILTER (WHERE status IN ('cancelled', 'rejected')) as failed_count
        FROM order_executions
        WHERE {where_sql}
        GROUP BY bucket
        ORDER BY bucket DESC
        LIMIT :limit
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query, params)
            rows = []
            for row in result:
                rows.append({
                    'bucket': row[0],
                    'execution_count': row[1],
                    'unique_symbols': row[2],
                    'total_shares': row[3] or 0,
                    'avg_slippage': float(row[4]) if row[4] else 0,
                    'avg_slippage_pct': float(row[5]) if row[5] else 0,
                    'max_slippage_pct': float(row[6]) if row[6] else 0,
                    'avg_fill_time': float(row[7]) if row[7] else 0,
                    'filled_count': row[8],
                    'partial_count': row[9],
                    'failed_count': row[10]
                })
            return rows
    except Exception as e:
        logger.error(f"Error in get_order_execution_stats: {e}")
        return []


def get_slippage_analysis(
    symbol: Optional[str] = None,
    days: int = 30,
    engine: Optional[Engine] = None
) -> Dict[str, Any]:
    """
    Get detailed slippage analysis for a symbol or all symbols.

    Args:
        symbol: Optional symbol filter.
        days: Number of days to analyze.
        engine: SQLAlchemy engine.

    Returns:
        Dict with slippage statistics and distribution.
    """
    if engine is None:
        from data.database.connection import get_db_manager
        manager = get_db_manager()
        engine = manager.engine

    where_clauses = ["fill_time >= NOW() - INTERVAL :days"]
    params = {'days': f'{days} days'}

    if symbol:
        where_clauses.append("symbol = :symbol")
        params['symbol'] = symbol

    where_sql = " AND ".join(where_clauses)

    query = text(f"""
        SELECT
            COUNT(*) as total_executions,
            AVG(slippage_pct) as avg_slippage_pct,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY slippage_pct) as median_slippage_pct,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY slippage_pct) as p95_slippage_pct,
            STDDEV(slippage_pct) as stddev_slippage_pct,
            MIN(slippage_pct) as min_slippage_pct,
            MAX(slippage_pct) as max_slippage_pct,
            SUM(CASE WHEN slippage > 0 THEN slippage ELSE 0 END) as total_negative_slippage,
            COUNT(*) FILTER (WHERE slippage > 0) as unfavorable_fills,
            COUNT(*) FILTER (WHERE slippage <= 0) as favorable_fills
        FROM order_executions
        WHERE {where_sql}
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query, params)
            row = result.fetchone()
            if row:
                return {
                    'total_executions': row[0],
                    'avg_slippage_pct': float(row[1]) if row[1] else 0,
                    'median_slippage_pct': float(row[2]) if row[2] else 0,
                    'p95_slippage_pct': float(row[3]) if row[3] else 0,
                    'stddev_slippage_pct': float(row[4]) if row[4] else 0,
                    'min_slippage_pct': float(row[5]) if row[5] else 0,
                    'max_slippage_pct': float(row[6]) if row[6] else 0,
                    'total_negative_slippage': float(row[7]) if row[7] else 0,
                    'unfavorable_fills': row[8],
                    'favorable_fills': row[9],
                }
    except Exception as e:
        logger.error(f"Error in get_slippage_analysis: {e}")

    return {}


# =============================================================================
# Utility Functions
# =============================================================================

def _interval_to_trunc_unit(interval: str) -> str:
    """Convert an interval string to a date_trunc unit for fallback."""
    interval = interval.lower()
    if 'minute' in interval:
        return 'minute'
    elif 'hour' in interval:
        return 'hour'
    elif 'day' in interval:
        return 'day'
    elif 'week' in interval:
        return 'week'
    elif 'month' in interval:
        return 'month'
    elif 'year' in interval:
        return 'year'
    else:
        return 'hour'  # Default


def _check_continuous_aggregate_exists(view_name: str, engine: Engine) -> bool:
    """Check if a continuous aggregate view exists."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM timescaledb_information.continuous_aggregates
                    WHERE view_name = :view_name
                )
            """), {'view_name': view_name})
            return result.scalar() or False
    except Exception:
        return False


def _query_continuous_aggregate(
    view_name: str,
    start_time: Optional[datetime],
    end_time: Optional[datetime],
    symbol: Optional[str],
    limit: int,
    engine: Engine
) -> List[Dict[str, Any]]:
    """Query a continuous aggregate view."""
    where_clauses = ["1=1"]
    params = {'limit': limit}

    if start_time:
        where_clauses.append("bucket >= :start_time")
        params['start_time'] = start_time
    if end_time:
        where_clauses.append("bucket <= :end_time")
        params['end_time'] = end_time
    if symbol:
        where_clauses.append("symbol = :symbol")
        params['symbol'] = symbol

    where_sql = " AND ".join(where_clauses)

    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT * FROM {view_name}
                WHERE {where_sql}
                ORDER BY bucket DESC
                LIMIT :limit
            """), params)

            # Convert rows to dicts
            columns = result.keys()
            rows = []
            for row in result:
                rows.append(dict(zip(columns, row)))
            return rows
    except Exception as e:
        logger.error(f"Error querying continuous aggregate {view_name}: {e}")
        return []


# =============================================================================
# Real-time Time-Series Functions
# =============================================================================

def get_last_n_periods(
    table_name: str,
    time_column: str,
    n_periods: int,
    interval: str = '1 day',
    columns: Optional[List[str]] = None,
    engine: Optional[Engine] = None
) -> List[Dict[str, Any]]:
    """
    Get data for the last N time periods.

    Args:
        table_name: Name of the table to query.
        time_column: Name of the time column.
        n_periods: Number of periods to retrieve.
        interval: Time period interval.
        columns: Columns to select (default: all).
        engine: SQLAlchemy engine.

    Returns:
        List of rows for the last N periods.
    """
    if engine is None:
        from data.database.connection import get_db_manager
        manager = get_db_manager()
        engine = manager.engine

    use_timescale = check_timescale_available(engine)

    if columns:
        select_sql = ", ".join(columns)
    else:
        select_sql = "*"

    if use_timescale:
        # Use TimescaleDB's efficient last() function if aggregating
        bucket_sql = f"time_bucket(INTERVAL '{interval}', {time_column})"
    else:
        trunc_unit = _interval_to_trunc_unit(interval)
        bucket_sql = f"date_trunc('{trunc_unit}', {time_column})"

    query = text(f"""
        SELECT {select_sql}
        FROM {table_name}
        WHERE {time_column} >= NOW() - INTERVAL :lookback
        ORDER BY {time_column} DESC
    """)

    # Calculate lookback
    lookback = f"{n_periods} {interval.split()[1] if len(interval.split()) > 1 else 'days'}"

    try:
        with engine.connect() as conn:
            result = conn.execute(query, {'lookback': lookback})
            columns = result.keys()
            rows = []
            for row in result:
                rows.append(dict(zip(columns, row)))
            return rows
    except Exception as e:
        logger.error(f"Error in get_last_n_periods: {e}")
        return []


def get_time_weighted_average(
    table_name: str,
    value_column: str,
    time_column: str,
    start_time: datetime,
    end_time: datetime,
    engine: Optional[Engine] = None
) -> Optional[float]:
    """
    Calculate time-weighted average for a value over a period.

    Args:
        table_name: Name of the table.
        value_column: Column containing values to average.
        time_column: Time column.
        start_time: Start of the period.
        end_time: End of the period.
        engine: SQLAlchemy engine.

    Returns:
        Time-weighted average or None if no data.
    """
    if engine is None:
        from data.database.connection import get_db_manager
        manager = get_db_manager()
        engine = manager.engine

    # Using a simple weighted average based on time between points
    query = text(f"""
        WITH ordered AS (
            SELECT
                {value_column},
                {time_column},
                LEAD({time_column}) OVER (ORDER BY {time_column}) as next_time
            FROM {table_name}
            WHERE {time_column} >= :start_time AND {time_column} < :end_time
        )
        SELECT
            SUM({value_column} * EXTRACT(EPOCH FROM (COALESCE(next_time, :end_time) - {time_column}))) /
            NULLIF(EXTRACT(EPOCH FROM (:end_time - :start_time)), 0) as twa
        FROM ordered
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query, {
                'start_time': start_time,
                'end_time': end_time
            })
            row = result.fetchone()
            if row and row[0]:
                return float(row[0])
    except Exception as e:
        logger.error(f"Error in get_time_weighted_average: {e}")

    return None

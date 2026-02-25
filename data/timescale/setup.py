"""
TimescaleDB setup and management for the RDT Trading System.

This module provides functions to:
- Detect TimescaleDB availability
- Create hypertables from existing tables
- Configure compression policies
- Configure retention policies
- Manage TimescaleDB-specific settings
"""

from typing import Optional, Dict, Any, List, Tuple
from datetime import timedelta

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from loguru import logger


# =============================================================================
# Constants
# =============================================================================

# Default chunk intervals for different table types
DEFAULT_CHUNK_INTERVALS = {
    'signals': timedelta(days=7),           # Weekly chunks for signals
    'daily_stats': timedelta(days=30),      # Monthly chunks for daily stats
    'market_data_cache': timedelta(hours=6), # 6-hour chunks for market data
    'order_executions': timedelta(days=7),  # Weekly chunks for order executions
    'audit_logs': timedelta(days=30),       # Monthly chunks for audit logs
}

# Default compression settings
DEFAULT_COMPRESSION_AFTER = {
    'signals': timedelta(days=7),
    'daily_stats': timedelta(days=30),
    'market_data_cache': timedelta(days=1),
    'order_executions': timedelta(days=7),
    'audit_logs': timedelta(days=30),
}

# Default retention settings (how long to keep data)
DEFAULT_RETENTION = {
    'signals': timedelta(days=90),
    'daily_stats': timedelta(days=730),      # 2 years
    'market_data_cache': timedelta(days=7),  # Short cache
    'order_executions': timedelta(days=365), # 1 year
    'audit_logs': timedelta(days=2555),      # ~7 years
}


# =============================================================================
# Detection Functions
# =============================================================================

def check_timescale_available(engine: Optional[Engine] = None) -> bool:
    """
    Check if TimescaleDB extension is available and enabled.

    Args:
        engine: SQLAlchemy engine. If None, uses the default database manager.

    Returns:
        bool: True if TimescaleDB is available and enabled.
    """
    if engine is None:
        from data.database.connection import get_db_manager
        manager = get_db_manager()
        engine = manager.engine

    try:
        with engine.connect() as conn:
            # Check if extension exists
            result = conn.execute(text(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb')"
            ))
            return result.scalar() or False
    except Exception as e:
        logger.debug(f"TimescaleDB check failed (likely not PostgreSQL): {e}")
        return False


def get_timescale_info(engine: Optional[Engine] = None) -> Dict[str, Any]:
    """
    Get detailed information about TimescaleDB installation.

    Args:
        engine: SQLAlchemy engine. If None, uses the default database manager.

    Returns:
        dict: TimescaleDB version and configuration info.
    """
    if engine is None:
        from data.database.connection import get_db_manager
        manager = get_db_manager()
        engine = manager.engine

    info = {
        'available': False,
        'version': None,
        'hypertables': [],
        'compression_enabled': False,
    }

    if not check_timescale_available(engine):
        return info

    try:
        with engine.connect() as conn:
            # Get TimescaleDB version
            result = conn.execute(text(
                "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
            ))
            row = result.fetchone()
            if row:
                info['available'] = True
                info['version'] = row[0]

            # Get list of hypertables
            result = conn.execute(text("""
                SELECT hypertable_schema, hypertable_name,
                       num_chunks, compression_enabled
                FROM timescaledb_information.hypertables
            """))
            hypertables = []
            for row in result:
                hypertables.append({
                    'schema': row[0],
                    'name': row[1],
                    'num_chunks': row[2],
                    'compression_enabled': row[3]
                })
            info['hypertables'] = hypertables

            # Check if any have compression
            info['compression_enabled'] = any(h['compression_enabled'] for h in hypertables)

    except Exception as e:
        logger.warning(f"Error getting TimescaleDB info: {e}")

    return info


def is_hypertable(table_name: str, engine: Optional[Engine] = None) -> bool:
    """
    Check if a table is already a hypertable.

    Args:
        table_name: Name of the table to check.
        engine: SQLAlchemy engine.

    Returns:
        bool: True if the table is a hypertable.
    """
    if engine is None:
        from data.database.connection import get_db_manager
        manager = get_db_manager()
        engine = manager.engine

    if not check_timescale_available(engine):
        return False

    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM timescaledb_information.hypertables
                    WHERE hypertable_name = :table_name
                )
            """), {'table_name': table_name})
            return result.scalar() or False
    except Exception as e:
        logger.debug(f"Hypertable check failed: {e}")
        return False


# =============================================================================
# Hypertable Management
# =============================================================================

def create_hypertable(
    table_name: str,
    time_column: str,
    chunk_time_interval: Optional[timedelta] = None,
    if_not_exists: bool = True,
    engine: Optional[Engine] = None
) -> bool:
    """
    Convert an existing table to a hypertable.

    Args:
        table_name: Name of the table to convert.
        time_column: Name of the timestamp column to partition on.
        chunk_time_interval: Time interval for each chunk. Defaults based on table type.
        if_not_exists: If True, skip if already a hypertable.
        engine: SQLAlchemy engine.

    Returns:
        bool: True if hypertable was created or already exists.
    """
    if engine is None:
        from data.database.connection import get_db_manager
        manager = get_db_manager()
        engine = manager.engine

    if not check_timescale_available(engine):
        logger.warning("TimescaleDB not available, cannot create hypertable")
        return False

    # Check if already a hypertable
    if if_not_exists and is_hypertable(table_name, engine):
        logger.info(f"Table {table_name} is already a hypertable")
        return True

    # Get default chunk interval if not specified
    if chunk_time_interval is None:
        chunk_time_interval = DEFAULT_CHUNK_INTERVALS.get(
            table_name, timedelta(days=7)
        )

    # Convert to interval string
    interval_str = _timedelta_to_interval(chunk_time_interval)

    try:
        with engine.connect() as conn:
            # Create hypertable with migrate_data option for existing data
            sql = text(f"""
                SELECT create_hypertable(
                    :table_name,
                    :time_column,
                    chunk_time_interval => INTERVAL :interval,
                    migrate_data => true,
                    if_not_exists => :if_not_exists
                )
            """)
            conn.execute(sql, {
                'table_name': table_name,
                'time_column': time_column,
                'interval': interval_str,
                'if_not_exists': if_not_exists
            })
            conn.commit()
            logger.info(f"Created hypertable: {table_name} on {time_column} with interval {interval_str}")
            return True

    except Exception as e:
        logger.error(f"Failed to create hypertable {table_name}: {e}")
        return False


def add_compression_policy(
    table_name: str,
    compress_after: Optional[timedelta] = None,
    segment_by: Optional[List[str]] = None,
    order_by: Optional[str] = None,
    engine: Optional[Engine] = None
) -> bool:
    """
    Add compression policy to a hypertable.

    Args:
        table_name: Name of the hypertable.
        compress_after: Compress chunks older than this interval.
        segment_by: Columns to segment compression by (for better query performance).
        order_by: Column to order compressed data by.
        engine: SQLAlchemy engine.

    Returns:
        bool: True if compression policy was added successfully.
    """
    if engine is None:
        from data.database.connection import get_db_manager
        manager = get_db_manager()
        engine = manager.engine

    if not check_timescale_available(engine):
        logger.warning("TimescaleDB not available, cannot add compression policy")
        return False

    if not is_hypertable(table_name, engine):
        logger.warning(f"Table {table_name} is not a hypertable")
        return False

    # Get defaults
    if compress_after is None:
        compress_after = DEFAULT_COMPRESSION_AFTER.get(
            table_name, timedelta(days=7)
        )

    interval_str = _timedelta_to_interval(compress_after)

    try:
        with engine.connect() as conn:
            # First, enable compression with segment/order by settings
            segment_sql = ""
            if segment_by:
                segment_sql = f", timescaledb.compress_segmentby = '{','.join(segment_by)}'"

            order_sql = ""
            if order_by:
                order_sql = f", timescaledb.compress_orderby = '{order_by}'"

            # Enable compression on the hypertable
            conn.execute(text(f"""
                ALTER TABLE {table_name} SET (
                    timescaledb.compress
                    {segment_sql}
                    {order_sql}
                )
            """))

            # Add compression policy
            conn.execute(text("""
                SELECT add_compression_policy(
                    :table_name,
                    INTERVAL :interval,
                    if_not_exists => true
                )
            """), {
                'table_name': table_name,
                'interval': interval_str
            })

            conn.commit()
            logger.info(f"Added compression policy to {table_name}: compress after {interval_str}")
            return True

    except Exception as e:
        logger.error(f"Failed to add compression policy to {table_name}: {e}")
        return False


def add_retention_policy(
    table_name: str,
    drop_after: Optional[timedelta] = None,
    engine: Optional[Engine] = None
) -> bool:
    """
    Add retention policy to a hypertable (automatically drops old chunks).

    Args:
        table_name: Name of the hypertable.
        drop_after: Drop chunks older than this interval.
        engine: SQLAlchemy engine.

    Returns:
        bool: True if retention policy was added successfully.
    """
    if engine is None:
        from data.database.connection import get_db_manager
        manager = get_db_manager()
        engine = manager.engine

    if not check_timescale_available(engine):
        logger.warning("TimescaleDB not available, cannot add retention policy")
        return False

    if not is_hypertable(table_name, engine):
        logger.warning(f"Table {table_name} is not a hypertable")
        return False

    # Get default retention if not specified
    if drop_after is None:
        drop_after = DEFAULT_RETENTION.get(table_name)
        if drop_after is None:
            logger.warning(f"No default retention for {table_name}, skipping")
            return False

    interval_str = _timedelta_to_interval(drop_after)

    try:
        with engine.connect() as conn:
            conn.execute(text("""
                SELECT add_retention_policy(
                    :table_name,
                    INTERVAL :interval,
                    if_not_exists => true
                )
            """), {
                'table_name': table_name,
                'interval': interval_str
            })
            conn.commit()
            logger.info(f"Added retention policy to {table_name}: drop after {interval_str}")
            return True

    except Exception as e:
        logger.error(f"Failed to add retention policy to {table_name}: {e}")
        return False


def remove_compression_policy(table_name: str, engine: Optional[Engine] = None) -> bool:
    """Remove compression policy from a hypertable."""
    if engine is None:
        from data.database.connection import get_db_manager
        manager = get_db_manager()
        engine = manager.engine

    if not check_timescale_available(engine):
        return False

    try:
        with engine.connect() as conn:
            conn.execute(text("""
                SELECT remove_compression_policy(:table_name, if_exists => true)
            """), {'table_name': table_name})
            conn.commit()
            logger.info(f"Removed compression policy from {table_name}")
            return True
    except Exception as e:
        logger.error(f"Failed to remove compression policy from {table_name}: {e}")
        return False


def remove_retention_policy(table_name: str, engine: Optional[Engine] = None) -> bool:
    """Remove retention policy from a hypertable."""
    if engine is None:
        from data.database.connection import get_db_manager
        manager = get_db_manager()
        engine = manager.engine

    if not check_timescale_available(engine):
        return False

    try:
        with engine.connect() as conn:
            conn.execute(text("""
                SELECT remove_retention_policy(:table_name, if_exists => true)
            """), {'table_name': table_name})
            conn.commit()
            logger.info(f"Removed retention policy from {table_name}")
            return True
    except Exception as e:
        logger.error(f"Failed to remove retention policy from {table_name}: {e}")
        return False


# =============================================================================
# TimescaleDB Manager Class
# =============================================================================

class TimescaleManager:
    """
    Manager class for TimescaleDB operations.

    Provides a high-level interface for managing hypertables, compression,
    and retention policies in the RDT Trading System.
    """

    # Table configurations: (time_column, segment_by, order_by)
    HYPERTABLE_CONFIGS = {
        'signals': ('timestamp', ['symbol'], 'timestamp DESC'),
        'daily_stats': ('date', None, 'date DESC'),
        'market_data_cache': ('timestamp', ['symbol', 'data_type'], 'timestamp DESC'),
        'order_executions': ('fill_time', ['symbol'], 'fill_time DESC'),
    }

    def __init__(self, engine: Optional[Engine] = None):
        """
        Initialize TimescaleManager.

        Args:
            engine: SQLAlchemy engine. If None, uses the default database manager.
        """
        if engine is None:
            from data.database.connection import get_db_manager
            manager = get_db_manager()
            engine = manager.engine

        self.engine = engine
        self._available = None

    @property
    def available(self) -> bool:
        """Check if TimescaleDB is available."""
        if self._available is None:
            self._available = check_timescale_available(self.engine)
        return self._available

    def get_info(self) -> Dict[str, Any]:
        """Get TimescaleDB information."""
        return get_timescale_info(self.engine)

    def setup_hypertables(
        self,
        tables: Optional[List[str]] = None,
        with_compression: bool = True,
        with_retention: bool = True
    ) -> Dict[str, bool]:
        """
        Set up hypertables for the specified tables.

        Args:
            tables: List of table names. If None, sets up all configured tables.
            with_compression: If True, adds compression policies.
            with_retention: If True, adds retention policies.

        Returns:
            dict: Mapping of table names to success status.
        """
        if not self.available:
            logger.warning("TimescaleDB not available")
            return {}

        if tables is None:
            tables = list(self.HYPERTABLE_CONFIGS.keys())

        results = {}
        for table_name in tables:
            if table_name not in self.HYPERTABLE_CONFIGS:
                logger.warning(f"Unknown table configuration: {table_name}")
                results[table_name] = False
                continue

            time_column, segment_by, order_by = self.HYPERTABLE_CONFIGS[table_name]

            # Create hypertable
            success = create_hypertable(
                table_name, time_column, engine=self.engine
            )

            if success:
                # Add compression policy
                if with_compression:
                    add_compression_policy(
                        table_name,
                        segment_by=segment_by,
                        order_by=order_by,
                        engine=self.engine
                    )

                # Add retention policy
                if with_retention:
                    add_retention_policy(table_name, engine=self.engine)

            results[table_name] = success

        return results

    def compress_chunks(
        self,
        table_name: str,
        older_than: Optional[timedelta] = None
    ) -> int:
        """
        Manually compress chunks older than specified interval.

        Args:
            table_name: Name of the hypertable.
            older_than: Compress chunks older than this. Defaults to compression policy.

        Returns:
            int: Number of chunks compressed.
        """
        if not self.available:
            return 0

        if older_than is None:
            older_than = DEFAULT_COMPRESSION_AFTER.get(table_name, timedelta(days=7))

        interval_str = _timedelta_to_interval(older_than)

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT compress_chunk(c.chunk_schema || '.' || c.chunk_name)
                    FROM timescaledb_information.chunks c
                    WHERE c.hypertable_name = :table_name
                    AND c.range_end < NOW() - INTERVAL :interval
                    AND NOT c.is_compressed
                """), {
                    'table_name': table_name,
                    'interval': interval_str
                })
                count = result.rowcount
                conn.commit()
                logger.info(f"Compressed {count} chunks for {table_name}")
                return count
        except Exception as e:
            logger.error(f"Failed to compress chunks for {table_name}: {e}")
            return 0

    def get_chunk_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Get statistics about chunks for a hypertable.

        Args:
            table_name: Name of the hypertable.

        Returns:
            dict: Chunk statistics including count, size, and compression ratio.
        """
        if not self.available:
            return {}

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT
                        COUNT(*) as total_chunks,
                        COUNT(*) FILTER (WHERE is_compressed) as compressed_chunks,
                        SUM(CASE WHEN NOT is_compressed THEN range_end - range_start END) as uncompressed_interval,
                        MIN(range_start) as oldest_data,
                        MAX(range_end) as newest_data
                    FROM timescaledb_information.chunks
                    WHERE hypertable_name = :table_name
                """), {'table_name': table_name})
                row = result.fetchone()
                if row:
                    return {
                        'total_chunks': row[0],
                        'compressed_chunks': row[1],
                        'oldest_data': row[3],
                        'newest_data': row[4],
                    }
        except Exception as e:
            logger.error(f"Failed to get chunk stats for {table_name}: {e}")
        return {}


# =============================================================================
# Utility Functions
# =============================================================================

def _timedelta_to_interval(td: timedelta) -> str:
    """Convert a timedelta to a PostgreSQL interval string."""
    total_seconds = int(td.total_seconds())

    if total_seconds >= 86400:  # Days
        days = total_seconds // 86400
        return f"{days} days"
    elif total_seconds >= 3600:  # Hours
        hours = total_seconds // 3600
        return f"{hours} hours"
    elif total_seconds >= 60:  # Minutes
        minutes = total_seconds // 60
        return f"{minutes} minutes"
    else:
        return f"{total_seconds} seconds"

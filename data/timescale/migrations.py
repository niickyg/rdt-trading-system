"""
TimescaleDB migration scripts for the RDT Trading System.

This module provides migration functions to:
- Convert existing PostgreSQL tables to TimescaleDB hypertables
- Roll back hypertable conversions (restore to regular tables)
- Track migration status

Note: These migrations are designed to be run separately from Alembic migrations
as they are TimescaleDB-specific and require the extension to be available.
"""

from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.engine import Engine

from .setup import (
    check_timescale_available,
    is_hypertable,
    create_hypertable,
    add_compression_policy,
    add_retention_policy,
    remove_compression_policy,
    remove_retention_policy,
    TimescaleManager,
)

from loguru import logger


# =============================================================================
# Migration Configuration
# =============================================================================

# Tables to migrate and their configurations
# Format: table_name -> (time_column, chunk_interval_days, segment_by, order_by)
MIGRATION_CONFIG = {
    'signals': {
        'time_column': 'timestamp',
        'chunk_interval': '7 days',
        'segment_by': ['symbol'],
        'order_by': 'timestamp DESC',
        'compress_after': '7 days',
        'retain_for': '90 days',
    },
    'daily_stats': {
        'time_column': 'date',
        'chunk_interval': '30 days',
        'segment_by': None,
        'order_by': 'date DESC',
        'compress_after': '30 days',
        'retain_for': '730 days',  # 2 years
    },
    'market_data_cache': {
        'time_column': 'timestamp',
        'chunk_interval': '6 hours',
        'segment_by': ['symbol', 'data_type'],
        'order_by': 'timestamp DESC',
        'compress_after': '1 day',
        'retain_for': '7 days',
    },
    'order_executions': {
        'time_column': 'fill_time',
        'chunk_interval': '7 days',
        'segment_by': ['symbol'],
        'order_by': 'fill_time DESC',
        'compress_after': '7 days',
        'retain_for': '365 days',
    },
}

# Migration tracking table
MIGRATION_TABLE = 'timescale_migrations'


# =============================================================================
# Migration Status Functions
# =============================================================================

def _ensure_migration_table(engine: Engine) -> None:
    """Create the migration tracking table if it doesn't exist."""
    with engine.connect() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {MIGRATION_TABLE} (
                id SERIAL PRIMARY KEY,
                table_name VARCHAR(64) NOT NULL UNIQUE,
                migrated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                time_column VARCHAR(64) NOT NULL,
                chunk_interval VARCHAR(32),
                compression_enabled BOOLEAN DEFAULT FALSE,
                retention_enabled BOOLEAN DEFAULT FALSE,
                rollback_sql TEXT
            )
        """))
        conn.commit()


def _record_migration(
    engine: Engine,
    table_name: str,
    time_column: str,
    chunk_interval: str,
    compression_enabled: bool = False,
    retention_enabled: bool = False,
    rollback_sql: Optional[str] = None
) -> None:
    """Record a migration in the tracking table."""
    with engine.connect() as conn:
        conn.execute(text(f"""
            INSERT INTO {MIGRATION_TABLE}
                (table_name, time_column, chunk_interval, compression_enabled, retention_enabled, rollback_sql)
            VALUES
                (:table_name, :time_column, :chunk_interval, :compression_enabled, :retention_enabled, :rollback_sql)
            ON CONFLICT (table_name) DO UPDATE SET
                migrated_at = NOW(),
                time_column = EXCLUDED.time_column,
                chunk_interval = EXCLUDED.chunk_interval,
                compression_enabled = EXCLUDED.compression_enabled,
                retention_enabled = EXCLUDED.retention_enabled,
                rollback_sql = EXCLUDED.rollback_sql
        """), {
            'table_name': table_name,
            'time_column': time_column,
            'chunk_interval': chunk_interval,
            'compression_enabled': compression_enabled,
            'retention_enabled': retention_enabled,
            'rollback_sql': rollback_sql
        })
        conn.commit()


def _remove_migration_record(engine: Engine, table_name: str) -> None:
    """Remove a migration record from the tracking table."""
    with engine.connect() as conn:
        conn.execute(text(f"""
            DELETE FROM {MIGRATION_TABLE} WHERE table_name = :table_name
        """), {'table_name': table_name})
        conn.commit()


def get_migration_status(engine: Optional[Engine] = None) -> Dict[str, Any]:
    """
    Get the current status of TimescaleDB migrations.

    Args:
        engine: SQLAlchemy engine. If None, uses the default database manager.

    Returns:
        dict: Migration status including:
            - timescale_available: Whether TimescaleDB is installed
            - migrations: Dict of table migrations
            - pending: List of tables not yet migrated
    """
    if engine is None:
        from data.database.connection import get_db_manager
        manager = get_db_manager()
        engine = manager.engine

    status = {
        'timescale_available': check_timescale_available(engine),
        'migrations': {},
        'pending': [],
        'hypertables': []
    }

    if not status['timescale_available']:
        status['pending'] = list(MIGRATION_CONFIG.keys())
        return status

    _ensure_migration_table(engine)

    try:
        with engine.connect() as conn:
            # Get recorded migrations
            result = conn.execute(text(f"""
                SELECT table_name, migrated_at, time_column, chunk_interval,
                       compression_enabled, retention_enabled
                FROM {MIGRATION_TABLE}
            """))
            for row in result:
                status['migrations'][row[0]] = {
                    'migrated_at': row[1],
                    'time_column': row[2],
                    'chunk_interval': row[3],
                    'compression_enabled': row[4],
                    'retention_enabled': row[5]
                }

            # Get actual hypertables
            result = conn.execute(text("""
                SELECT hypertable_name, compression_enabled
                FROM timescaledb_information.hypertables
            """))
            for row in result:
                status['hypertables'].append({
                    'name': row[0],
                    'compression_enabled': row[1]
                })

    except Exception as e:
        logger.error(f"Error getting migration status: {e}")

    # Determine pending migrations
    migrated = set(status['migrations'].keys())
    configured = set(MIGRATION_CONFIG.keys())
    status['pending'] = list(configured - migrated)

    return status


# =============================================================================
# Migration Functions
# =============================================================================

def migrate_table_to_hypertable(
    table_name: str,
    config: Dict[str, Any],
    engine: Engine,
    with_compression: bool = True,
    with_retention: bool = True
) -> Tuple[bool, str]:
    """
    Migrate a single table to a hypertable.

    Args:
        table_name: Name of the table to migrate.
        config: Migration configuration for the table.
        engine: SQLAlchemy engine.
        with_compression: Whether to enable compression.
        with_retention: Whether to enable retention policy.

    Returns:
        Tuple of (success, message).
    """
    time_column = config['time_column']
    chunk_interval = config['chunk_interval']

    # Check if already a hypertable
    if is_hypertable(table_name, engine):
        return True, f"Table {table_name} is already a hypertable"

    # Check if table exists
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = :table_name
            )
        """), {'table_name': table_name})
        if not result.scalar():
            return False, f"Table {table_name} does not exist"

    try:
        with engine.connect() as conn:
            # Create hypertable
            conn.execute(text(f"""
                SELECT create_hypertable(
                    '{table_name}',
                    '{time_column}',
                    chunk_time_interval => INTERVAL '{chunk_interval}',
                    migrate_data => true,
                    if_not_exists => true
                )
            """))
            conn.commit()
            logger.info(f"Converted {table_name} to hypertable on {time_column}")

            # Enable compression
            compression_enabled = False
            if with_compression and config.get('compress_after'):
                segment_sql = ""
                if config.get('segment_by'):
                    segment_sql = f", timescaledb.compress_segmentby = '{','.join(config['segment_by'])}'"

                order_sql = ""
                if config.get('order_by'):
                    order_sql = f", timescaledb.compress_orderby = '{config['order_by']}'"

                conn.execute(text(f"""
                    ALTER TABLE {table_name} SET (
                        timescaledb.compress
                        {segment_sql}
                        {order_sql}
                    )
                """))

                conn.execute(text(f"""
                    SELECT add_compression_policy(
                        '{table_name}',
                        INTERVAL '{config['compress_after']}',
                        if_not_exists => true
                    )
                """))
                compression_enabled = True
                conn.commit()
                logger.info(f"Added compression policy to {table_name}")

            # Add retention policy
            retention_enabled = False
            if with_retention and config.get('retain_for'):
                conn.execute(text(f"""
                    SELECT add_retention_policy(
                        '{table_name}',
                        INTERVAL '{config['retain_for']}',
                        if_not_exists => true
                    )
                """))
                retention_enabled = True
                conn.commit()
                logger.info(f"Added retention policy to {table_name}")

            # Record the migration
            _record_migration(
                engine,
                table_name,
                time_column,
                chunk_interval,
                compression_enabled,
                retention_enabled,
                f"-- Rollback SQL for {table_name}\n-- Note: Converting back from hypertable requires data export/import"
            )

            return True, f"Successfully migrated {table_name} to hypertable"

    except Exception as e:
        error_msg = f"Failed to migrate {table_name}: {e}"
        logger.error(error_msg)
        return False, error_msg


def migrate_to_hypertables(
    tables: Optional[List[str]] = None,
    with_compression: bool = True,
    with_retention: bool = True,
    engine: Optional[Engine] = None
) -> Dict[str, Tuple[bool, str]]:
    """
    Migrate tables to TimescaleDB hypertables.

    Args:
        tables: List of table names to migrate. If None, migrates all configured tables.
        with_compression: Whether to enable compression policies.
        with_retention: Whether to enable retention policies.
        engine: SQLAlchemy engine. If None, uses the default database manager.

    Returns:
        dict: Mapping of table names to (success, message) tuples.
    """
    if engine is None:
        from data.database.connection import get_db_manager
        manager = get_db_manager()
        engine = manager.engine

    if not check_timescale_available(engine):
        return {t: (False, "TimescaleDB not available") for t in (tables or MIGRATION_CONFIG.keys())}

    _ensure_migration_table(engine)

    if tables is None:
        tables = list(MIGRATION_CONFIG.keys())

    results = {}
    for table_name in tables:
        if table_name not in MIGRATION_CONFIG:
            results[table_name] = (False, f"No migration configuration for {table_name}")
            continue

        config = MIGRATION_CONFIG[table_name]
        results[table_name] = migrate_table_to_hypertable(
            table_name, config, engine, with_compression, with_retention
        )

    return results


# =============================================================================
# Rollback Functions
# =============================================================================

def rollback_hypertable(table_name: str, engine: Engine) -> Tuple[bool, str]:
    """
    Roll back a hypertable to a regular table.

    Note: This is a complex operation that requires:
    1. Removing policies
    2. Decompressing all chunks
    3. Exporting data
    4. Dropping the hypertable
    5. Creating a regular table
    6. Importing data

    For safety, this function only removes policies and decompresses chunks.
    Full rollback to a regular table is not automated.

    Args:
        table_name: Name of the hypertable to roll back.
        engine: SQLAlchemy engine.

    Returns:
        Tuple of (success, message).
    """
    if not is_hypertable(table_name, engine):
        return False, f"Table {table_name} is not a hypertable"

    try:
        with engine.connect() as conn:
            # Remove retention policy
            conn.execute(text(f"""
                SELECT remove_retention_policy('{table_name}', if_exists => true)
            """))
            logger.info(f"Removed retention policy from {table_name}")

            # Remove compression policy
            conn.execute(text(f"""
                SELECT remove_compression_policy('{table_name}', if_exists => true)
            """))
            logger.info(f"Removed compression policy from {table_name}")

            # Decompress all chunks
            conn.execute(text(f"""
                SELECT decompress_chunk(c.chunk_schema || '.' || c.chunk_name, if_compressed => true)
                FROM timescaledb_information.chunks c
                WHERE c.hypertable_name = '{table_name}'
                AND c.is_compressed
            """))
            logger.info(f"Decompressed chunks for {table_name}")

            conn.commit()

            # Remove migration record
            _remove_migration_record(engine, table_name)

            return True, f"Removed policies and decompressed chunks for {table_name}. " \
                         f"Table remains a hypertable - full rollback requires manual data migration."

    except Exception as e:
        error_msg = f"Failed to rollback {table_name}: {e}"
        logger.error(error_msg)
        return False, error_msg


def rollback_hypertables(
    tables: Optional[List[str]] = None,
    engine: Optional[Engine] = None
) -> Dict[str, Tuple[bool, str]]:
    """
    Roll back hypertables (remove policies and decompress).

    Args:
        tables: List of table names. If None, rolls back all migrated tables.
        engine: SQLAlchemy engine.

    Returns:
        dict: Mapping of table names to (success, message) tuples.
    """
    if engine is None:
        from data.database.connection import get_db_manager
        manager = get_db_manager()
        engine = manager.engine

    if not check_timescale_available(engine):
        return {}

    if tables is None:
        status = get_migration_status(engine)
        tables = list(status['migrations'].keys())

    results = {}
    for table_name in tables:
        results[table_name] = rollback_hypertable(table_name, engine)

    return results


# =============================================================================
# Utility Functions
# =============================================================================

def create_continuous_aggregate(
    name: str,
    source_table: str,
    time_column: str,
    bucket_interval: str,
    aggregations: str,
    group_by: Optional[List[str]] = None,
    refresh_interval: str = '1 hour',
    engine: Optional[Engine] = None
) -> bool:
    """
    Create a continuous aggregate for pre-computed time-series analytics.

    Args:
        name: Name of the continuous aggregate view.
        source_table: Source hypertable name.
        time_column: Time column to bucket.
        bucket_interval: Time bucket interval (e.g., '1 hour', '1 day').
        aggregations: SQL aggregation expressions (e.g., 'COUNT(*) as count, AVG(price) as avg_price').
        group_by: Additional columns to group by.
        refresh_interval: How often to refresh (for materialized data).
        engine: SQLAlchemy engine.

    Returns:
        bool: True if created successfully.
    """
    if engine is None:
        from data.database.connection import get_db_manager
        manager = get_db_manager()
        engine = manager.engine

    if not check_timescale_available(engine):
        logger.warning("TimescaleDB not available")
        return False

    group_by_sql = ""
    if group_by:
        group_by_sql = ", " + ", ".join(group_by)

    try:
        with engine.connect() as conn:
            # Create continuous aggregate
            conn.execute(text(f"""
                CREATE MATERIALIZED VIEW IF NOT EXISTS {name}
                WITH (timescaledb.continuous) AS
                SELECT
                    time_bucket(INTERVAL '{bucket_interval}', {time_column}) AS bucket,
                    {aggregations}
                    {group_by_sql}
                FROM {source_table}
                GROUP BY bucket{group_by_sql}
                WITH NO DATA
            """))

            # Add refresh policy
            conn.execute(text(f"""
                SELECT add_continuous_aggregate_policy(
                    '{name}',
                    start_offset => INTERVAL '3 days',
                    end_offset => INTERVAL '1 hour',
                    schedule_interval => INTERVAL '{refresh_interval}',
                    if_not_exists => true
                )
            """))

            # Initial refresh
            conn.execute(text(f"""
                CALL refresh_continuous_aggregate('{name}', NULL, NULL)
            """))

            conn.commit()
            logger.info(f"Created continuous aggregate: {name}")
            return True

    except Exception as e:
        logger.error(f"Failed to create continuous aggregate {name}: {e}")
        return False

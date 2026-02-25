# TimescaleDB Setup Guide for RDT Trading System

This guide explains how to set up and use TimescaleDB for optimized time-series data storage in the RDT Trading System.

## Overview

TimescaleDB is a PostgreSQL extension that provides optimized storage and queries for time-series data. The RDT Trading System uses it to efficiently store and query:

- **Trading signals** - Time-series of generated signals
- **Daily statistics** - Daily P&L and performance metrics
- **Market data cache** - Cached market data for reduced API calls
- **Order executions** - Order fill tracking with slippage analysis

### Benefits

1. **Automatic partitioning** - Data is automatically partitioned by time (hypertables)
2. **Compression** - Older data is compressed to save storage
3. **Retention policies** - Automatic cleanup of old data
4. **Optimized queries** - Special functions like `time_bucket()` for time-series aggregations
5. **Continuous aggregates** - Pre-computed rollups for fast dashboards

## Quick Start

### Using Docker (Recommended)

The default Docker Compose configuration uses TimescaleDB:

```bash
# Start the database (TimescaleDB by default)
docker-compose up -d db

# Run migrations to create tables
docker-compose run --rm db-init

# The TimescaleDB initialization script will automatically:
# - Enable the TimescaleDB extension
# - Convert tables to hypertables
# - Set up compression policies
# - Set up retention policies
```

### Using Plain PostgreSQL (Fallback)

If you need to use plain PostgreSQL without TimescaleDB:

```bash
# Start PostgreSQL only
docker-compose --profile postgres-only up -d db-postgres
```

The system automatically detects TimescaleDB availability and falls back to standard SQL queries when not available.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_USER` | `rdt` | Database username |
| `POSTGRES_PASSWORD` | Required | Database password |
| `POSTGRES_DB` | `rdt_trading` | Database name |
| `POSTGRES_PORT` | `5432` | Database port |

### Chunk Intervals

Hypertables are partitioned into chunks. Default intervals:

| Table | Chunk Interval | Description |
|-------|----------------|-------------|
| `signals` | 7 days | Weekly chunks for trading signals |
| `daily_stats` | 30 days | Monthly chunks for daily stats |
| `market_data_cache` | 6 hours | Small chunks for frequently updated cache |
| `order_executions` | 7 days | Weekly chunks for order fills |

### Compression Settings

Data is compressed after a configurable period:

| Table | Compress After | Segment By | Order By |
|-------|----------------|------------|----------|
| `signals` | 7 days | symbol | timestamp DESC |
| `daily_stats` | 30 days | - | date DESC |
| `market_data_cache` | 1 day | symbol, data_type | timestamp DESC |
| `order_executions` | 7 days | symbol | fill_time DESC |

### Retention Settings

Old data is automatically dropped:

| Table | Retention Period | Notes |
|-------|------------------|-------|
| `signals` | 90 days | Short retention for signals |
| `daily_stats` | 730 days (2 years) | Longer retention for performance history |
| `market_data_cache` | 7 days | Cache cleaned frequently |
| `order_executions` | 365 days (1 year) | Keep 1 year of execution data |

## Python API

### Checking TimescaleDB Availability

```python
from data.database import get_db_manager

manager = get_db_manager()

# Check if TimescaleDB is available
if manager.is_timescale:
    print(f"TimescaleDB {manager.timescale_version} is available")
else:
    print("Using standard PostgreSQL")
```

### Using Optimized Queries

```python
from data.timescale import (
    get_signals_time_bucket,
    get_performance_over_time,
    get_order_execution_stats,
)

# Get hourly signal aggregations
signals = get_signals_time_bucket(
    interval='1 hour',
    symbol='AAPL',
    start_time=datetime.now() - timedelta(days=7)
)

# Get daily P&L over time
performance = get_performance_over_time(
    period='1 day',
    start_date=date(2024, 1, 1),
    include_cumulative=True
)

# Get execution statistics
stats = get_order_execution_stats(
    interval='1 day',
    days=30
)
```

### Managing Hypertables

```python
from data.timescale import TimescaleManager

# Create manager
ts_manager = TimescaleManager()

# Check current setup
info = ts_manager.get_info()
print(f"Hypertables: {info['hypertables']}")

# Set up hypertables (idempotent)
results = ts_manager.setup_hypertables(
    with_compression=True,
    with_retention=True
)

# Get chunk statistics
stats = ts_manager.get_chunk_stats('signals')
print(f"Total chunks: {stats['total_chunks']}")
print(f"Compressed: {stats['compressed_chunks']}")
```

### Running Migrations

```python
from data.timescale import migrate_to_hypertables, get_migration_status

# Check migration status
status = get_migration_status()
print(f"Pending migrations: {status['pending']}")

# Run migrations
results = migrate_to_hypertables(
    with_compression=True,
    with_retention=True
)

for table, (success, message) in results.items():
    print(f"{table}: {message}")
```

## Manual Setup

If you need to manually set up TimescaleDB:

### 1. Enable Extension

```sql
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
```

### 2. Convert Tables to Hypertables

```sql
-- Signals table
SELECT create_hypertable(
    'signals',
    'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    migrate_data => true,
    if_not_exists => true
);

-- Daily stats table
SELECT create_hypertable(
    'daily_stats',
    'date',
    chunk_time_interval => INTERVAL '30 days',
    migrate_data => true,
    if_not_exists => true
);
```

### 3. Enable Compression

```sql
-- Enable compression on signals
ALTER TABLE signals SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'timestamp DESC'
);

-- Add compression policy
SELECT add_compression_policy('signals', INTERVAL '7 days');
```

### 4. Add Retention Policy

```sql
SELECT add_retention_policy('signals', INTERVAL '90 days');
```

## Monitoring

### View Hypertables

```sql
SELECT * FROM timescaledb_information.hypertables;
```

### View Chunks

```sql
SELECT * FROM timescaledb_information.chunks
WHERE hypertable_name = 'signals'
ORDER BY range_start DESC;
```

### View Compression Stats

```sql
SELECT
    hypertable_name,
    chunk_name,
    range_start,
    range_end,
    is_compressed
FROM timescaledb_information.chunks
WHERE hypertable_name = 'signals';
```

### View Policies

```sql
SELECT * FROM timescaledb_information.jobs;
```

## Troubleshooting

### TimescaleDB Not Detected

1. Ensure you're using the TimescaleDB Docker image:
   ```yaml
   image: timescale/timescaledb:latest-pg15
   ```

2. Check if extension is enabled:
   ```sql
   SELECT * FROM pg_extension WHERE extname = 'timescaledb';
   ```

3. Enable manually if needed:
   ```sql
   CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
   ```

### Tables Not Converted to Hypertables

Run the migration script:

```python
from data.timescale import migrate_to_hypertables
results = migrate_to_hypertables()
```

Or manually:

```sql
SELECT create_hypertable('signals', 'timestamp', migrate_data => true, if_not_exists => true);
```

### Compression Not Working

Check if compression is enabled:

```sql
SELECT hypertable_name, compression_enabled
FROM timescaledb_information.hypertables;
```

Enable compression policy:

```sql
ALTER TABLE signals SET (timescaledb.compress);
SELECT add_compression_policy('signals', INTERVAL '7 days', if_not_exists => true);
```

### Query Performance

Use `EXPLAIN ANALYZE` to check query plans:

```sql
EXPLAIN ANALYZE
SELECT time_bucket('1 hour', timestamp) as bucket, COUNT(*)
FROM signals
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY bucket;
```

## Backward Compatibility

The RDT Trading System is designed to work with or without TimescaleDB:

1. **TimescaleDB available**: Uses optimized time-series functions
2. **Plain PostgreSQL**: Falls back to standard SQL with `date_trunc()`
3. **SQLite**: Uses standard SQL for development/testing

All query functions automatically detect the database type and use appropriate SQL syntax.

## Resources

- [TimescaleDB Documentation](https://docs.timescale.com/)
- [Hypertable Best Practices](https://docs.timescale.com/timescaledb/latest/how-to-guides/hypertables/)
- [Compression Guide](https://docs.timescale.com/timescaledb/latest/how-to-guides/compression/)
- [Continuous Aggregates](https://docs.timescale.com/timescaledb/latest/how-to-guides/continuous-aggregates/)

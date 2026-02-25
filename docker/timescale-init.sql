-- =============================================================================
-- RDT Trading System - TimescaleDB Initialization Script
-- =============================================================================
-- This script runs automatically when the TimescaleDB container is first created.
-- It enables TimescaleDB extension and sets up hypertables with policies.
--
-- Note: This script requires the base tables to exist first. Run Alembic
-- migrations before this script, or use the two-stage initialization approach.
-- =============================================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Enable useful PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";      -- UUID generation
CREATE EXTENSION IF NOT EXISTS "pg_trgm";        -- Trigram matching for fuzzy search
CREATE EXTENSION IF NOT EXISTS "btree_gin";      -- GIN index support for btree types

-- Set timezone for the database
SET timezone = 'America/New_York';

-- =============================================================================
-- Notification that we're using TimescaleDB
-- =============================================================================
DO $$
BEGIN
    RAISE NOTICE '==============================================';
    RAISE NOTICE 'RDT Trading System - TimescaleDB Mode';
    RAISE NOTICE 'TimescaleDB extension enabled';
    RAISE NOTICE '==============================================';
END $$;

-- =============================================================================
-- Function to safely convert tables to hypertables
-- =============================================================================
-- This function checks if a table exists and converts it to a hypertable
-- if it isn't already one. This allows the script to be idempotent.

CREATE OR REPLACE FUNCTION setup_hypertable_if_table_exists(
    p_table_name TEXT,
    p_time_column TEXT,
    p_chunk_interval INTERVAL DEFAULT INTERVAL '7 days'
) RETURNS TEXT AS $$
DECLARE
    v_result TEXT;
BEGIN
    -- Check if the table exists
    IF NOT EXISTS (
        SELECT FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name = p_table_name
    ) THEN
        RETURN 'Table ' || p_table_name || ' does not exist yet - skipping';
    END IF;

    -- Check if already a hypertable
    IF EXISTS (
        SELECT FROM timescaledb_information.hypertables
        WHERE hypertable_name = p_table_name
    ) THEN
        RETURN 'Table ' || p_table_name || ' is already a hypertable';
    END IF;

    -- Convert to hypertable
    BEGIN
        PERFORM create_hypertable(
            p_table_name,
            p_time_column,
            chunk_time_interval => p_chunk_interval,
            migrate_data => true,
            if_not_exists => true
        );
        v_result := 'Created hypertable: ' || p_table_name || ' on ' || p_time_column;
    EXCEPTION WHEN OTHERS THEN
        v_result := 'Error creating hypertable ' || p_table_name || ': ' || SQLERRM;
    END;

    RETURN v_result;
END;
$$ LANGUAGE plpgsql;


-- =============================================================================
-- Function to add compression policy
-- =============================================================================
CREATE OR REPLACE FUNCTION setup_compression_policy(
    p_table_name TEXT,
    p_compress_after INTERVAL,
    p_segment_by TEXT DEFAULT NULL,
    p_order_by TEXT DEFAULT NULL
) RETURNS TEXT AS $$
DECLARE
    v_alter_sql TEXT;
    v_result TEXT;
BEGIN
    -- Check if table is a hypertable
    IF NOT EXISTS (
        SELECT FROM timescaledb_information.hypertables
        WHERE hypertable_name = p_table_name
    ) THEN
        RETURN 'Table ' || p_table_name || ' is not a hypertable - skipping compression';
    END IF;

    -- Build ALTER TABLE statement
    v_alter_sql := 'ALTER TABLE ' || p_table_name || ' SET (timescaledb.compress';

    IF p_segment_by IS NOT NULL THEN
        v_alter_sql := v_alter_sql || ', timescaledb.compress_segmentby = ''' || p_segment_by || '''';
    END IF;

    IF p_order_by IS NOT NULL THEN
        v_alter_sql := v_alter_sql || ', timescaledb.compress_orderby = ''' || p_order_by || '''';
    END IF;

    v_alter_sql := v_alter_sql || ')';

    BEGIN
        -- Enable compression
        EXECUTE v_alter_sql;

        -- Add compression policy
        PERFORM add_compression_policy(
            p_table_name,
            p_compress_after,
            if_not_exists => true
        );

        v_result := 'Compression enabled for ' || p_table_name || ' after ' || p_compress_after;
    EXCEPTION WHEN OTHERS THEN
        v_result := 'Error enabling compression for ' || p_table_name || ': ' || SQLERRM;
    END;

    RETURN v_result;
END;
$$ LANGUAGE plpgsql;


-- =============================================================================
-- Function to add retention policy
-- =============================================================================
CREATE OR REPLACE FUNCTION setup_retention_policy(
    p_table_name TEXT,
    p_drop_after INTERVAL
) RETURNS TEXT AS $$
DECLARE
    v_result TEXT;
BEGIN
    -- Check if table is a hypertable
    IF NOT EXISTS (
        SELECT FROM timescaledb_information.hypertables
        WHERE hypertable_name = p_table_name
    ) THEN
        RETURN 'Table ' || p_table_name || ' is not a hypertable - skipping retention';
    END IF;

    BEGIN
        PERFORM add_retention_policy(
            p_table_name,
            p_drop_after,
            if_not_exists => true
        );
        v_result := 'Retention policy added for ' || p_table_name || ': drop after ' || p_drop_after;
    EXCEPTION WHEN OTHERS THEN
        v_result := 'Error adding retention policy for ' || p_table_name || ': ' || SQLERRM;
    END;

    RETURN v_result;
END;
$$ LANGUAGE plpgsql;


-- =============================================================================
-- Setup Hypertables
-- =============================================================================
-- Note: These will only work after Alembic migrations have created the tables.
-- Run this script again after migrations or use the Python migration scripts.

-- Signals table - time-series of trading signals
SELECT setup_hypertable_if_table_exists(
    'signals',
    'timestamp',
    INTERVAL '7 days'
);

-- Daily stats table - daily performance metrics
SELECT setup_hypertable_if_table_exists(
    'daily_stats',
    'date',
    INTERVAL '30 days'
);

-- Market data cache - cached market data
SELECT setup_hypertable_if_table_exists(
    'market_data_cache',
    'timestamp',
    INTERVAL '6 hours'
);

-- Order executions - order fill tracking
SELECT setup_hypertable_if_table_exists(
    'order_executions',
    'fill_time',
    INTERVAL '7 days'
);


-- =============================================================================
-- Setup Compression Policies
-- =============================================================================

SELECT setup_compression_policy(
    'signals',
    INTERVAL '7 days',
    'symbol',
    'timestamp DESC'
);

SELECT setup_compression_policy(
    'daily_stats',
    INTERVAL '30 days',
    NULL,
    'date DESC'
);

SELECT setup_compression_policy(
    'market_data_cache',
    INTERVAL '1 day',
    'symbol,data_type',
    'timestamp DESC'
);

SELECT setup_compression_policy(
    'order_executions',
    INTERVAL '7 days',
    'symbol',
    'fill_time DESC'
);


-- =============================================================================
-- Setup Retention Policies
-- =============================================================================

SELECT setup_retention_policy(
    'signals',
    INTERVAL '90 days'
);

SELECT setup_retention_policy(
    'daily_stats',
    INTERVAL '730 days'  -- 2 years
);

SELECT setup_retention_policy(
    'market_data_cache',
    INTERVAL '7 days'
);

SELECT setup_retention_policy(
    'order_executions',
    INTERVAL '365 days'  -- 1 year
);


-- =============================================================================
-- Create additional indexes for common queries (same as postgres-init.sql)
-- =============================================================================

-- Function to safely create indexes (checks if table/index exists)
CREATE OR REPLACE FUNCTION create_index_if_not_exists(
    index_name TEXT,
    table_name TEXT,
    index_definition TEXT
) RETURNS VOID AS $$
BEGIN
    IF EXISTS (
        SELECT FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name = create_index_if_not_exists.table_name
    ) THEN
        IF NOT EXISTS (
            SELECT FROM pg_indexes
            WHERE schemaname = 'public'
            AND indexname = create_index_if_not_exists.index_name
        ) THEN
            EXECUTE index_definition;
            RAISE NOTICE 'Created index: %', index_name;
        ELSE
            RAISE NOTICE 'Index already exists: %', index_name;
        END IF;
    ELSE
        RAISE NOTICE 'Table does not exist yet: %. Index will be created by migrations.', table_name;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Partial indexes
SELECT create_index_if_not_exists(
    'ix_trades_open_status',
    'trades',
    'CREATE INDEX ix_trades_open_status ON trades (symbol, entry_time) WHERE status = ''open'''
);

SELECT create_index_if_not_exists(
    'ix_signals_recent',
    'signals',
    'CREATE INDEX ix_signals_recent ON signals (timestamp DESC) WHERE status = ''pending'''
);

SELECT create_index_if_not_exists(
    'ix_users_active',
    'users',
    'CREATE INDEX ix_users_active ON users (username) WHERE is_active = true'
);

SELECT create_index_if_not_exists(
    'ix_api_users_tier',
    'api_users',
    'CREATE INDEX ix_api_users_tier ON api_users (tier) WHERE is_active = true'
);

-- Composite indexes
SELECT create_index_if_not_exists(
    'ix_trades_date_symbol',
    'trades',
    'CREATE INDEX ix_trades_date_symbol ON trades (entry_time, symbol, status)'
);

SELECT create_index_if_not_exists(
    'ix_daily_stats_date_range',
    'daily_stats',
    'CREATE INDEX ix_daily_stats_date_range ON daily_stats (date DESC)'
);


-- =============================================================================
-- Create continuous aggregates for common queries (optional, requires data)
-- =============================================================================
-- Uncomment these after the tables have data and you've confirmed the setup works.

/*
-- Hourly signal aggregation
CREATE MATERIALIZED VIEW IF NOT EXISTS signals_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket(INTERVAL '1 hour', timestamp) AS bucket,
    symbol,
    COUNT(*) as signal_count,
    AVG(rrs) as avg_rrs,
    COUNT(*) FILTER (WHERE direction = 'long') as long_count,
    COUNT(*) FILTER (WHERE direction = 'short') as short_count
FROM signals
GROUP BY bucket, symbol
WITH NO DATA;

-- Add refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy(
    'signals_hourly',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => true
);


-- Daily execution statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS executions_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket(INTERVAL '1 day', fill_time) AS bucket,
    symbol,
    COUNT(*) as execution_count,
    AVG(slippage_pct) as avg_slippage_pct,
    SUM(quantity) as total_shares
FROM order_executions
GROUP BY bucket, symbol
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'executions_daily',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => true
);
*/


-- =============================================================================
-- View current TimescaleDB setup
-- =============================================================================
DO $$
DECLARE
    v_hypertable RECORD;
    v_policy RECORD;
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '=== TimescaleDB Setup Summary ===';
    RAISE NOTICE '';

    -- List hypertables
    RAISE NOTICE 'Hypertables:';
    FOR v_hypertable IN
        SELECT hypertable_name, num_chunks, compression_enabled
        FROM timescaledb_information.hypertables
        ORDER BY hypertable_name
    LOOP
        RAISE NOTICE '  - %: % chunks, compression=%',
            v_hypertable.hypertable_name,
            v_hypertable.num_chunks,
            v_hypertable.compression_enabled;
    END LOOP;

    -- List compression policies
    RAISE NOTICE '';
    RAISE NOTICE 'Compression Policies:';
    FOR v_policy IN
        SELECT hypertable_name, compress_after
        FROM timescaledb_information.jobs j
        JOIN timescaledb_information.job_stats js ON j.job_id = js.job_id
        WHERE j.proc_name = 'policy_compression'
    LOOP
        RAISE NOTICE '  - %: compress after %',
            v_policy.hypertable_name,
            v_policy.compress_after;
    END LOOP;

    RAISE NOTICE '';
    RAISE NOTICE '=== Setup Complete ===';
END $$;

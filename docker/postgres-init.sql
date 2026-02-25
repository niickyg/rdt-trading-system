-- =============================================================================
-- RDT Trading System - PostgreSQL Initialization Script
-- =============================================================================
-- This script runs automatically when the PostgreSQL container is first created.
-- It sets up extensions, creates indexes, and configures the database.
-- =============================================================================

-- Enable useful PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";      -- UUID generation
CREATE EXTENSION IF NOT EXISTS "pg_trgm";        -- Trigram matching for fuzzy search
CREATE EXTENSION IF NOT EXISTS "btree_gin";      -- GIN index support for btree types

-- Set timezone for the database
SET timezone = 'America/New_York';

-- =============================================================================
-- Performance Tuning Configuration
-- =============================================================================
-- Note: These are suggestions. Actual values should be tuned based on your
-- server's resources. These can also be set in postgresql.conf.

-- Increase work memory for complex queries (default is 4MB)
-- ALTER SYSTEM SET work_mem = '64MB';

-- Increase maintenance work memory for VACUUM, CREATE INDEX, etc.
-- ALTER SYSTEM SET maintenance_work_mem = '256MB';

-- Enable parallel query execution
-- ALTER SYSTEM SET max_parallel_workers_per_gather = 2;

-- =============================================================================
-- Create additional indexes for common queries
-- =============================================================================
-- Note: These indexes improve query performance for common trading operations.
-- They will be created on tables after Alembic migrations create them.

-- Function to safely create indexes (checks if table/index exists)
CREATE OR REPLACE FUNCTION create_index_if_not_exists(
    index_name TEXT,
    table_name TEXT,
    index_definition TEXT
) RETURNS VOID AS $$
BEGIN
    -- Check if the table exists
    IF EXISTS (
        SELECT FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name = create_index_if_not_exists.table_name
    ) THEN
        -- Check if the index already exists
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

-- =============================================================================
-- Partial indexes for common query patterns
-- =============================================================================

-- Index for open trades (most frequently queried)
SELECT create_index_if_not_exists(
    'ix_trades_open_status',
    'trades',
    'CREATE INDEX ix_trades_open_status ON trades (symbol, entry_time) WHERE status = ''open'''
);

-- Index for recent signals
SELECT create_index_if_not_exists(
    'ix_signals_recent',
    'signals',
    'CREATE INDEX ix_signals_recent ON signals (timestamp DESC) WHERE status = ''pending'''
);

-- Index for active users
SELECT create_index_if_not_exists(
    'ix_users_active',
    'users',
    'CREATE INDEX ix_users_active ON users (username) WHERE is_active = true'
);

-- Index for API users by tier
SELECT create_index_if_not_exists(
    'ix_api_users_tier',
    'api_users',
    'CREATE INDEX ix_api_users_tier ON api_users (tier) WHERE is_active = true'
);

-- =============================================================================
-- Composite indexes for common join patterns
-- =============================================================================

-- Trades by date range and symbol (for reporting)
SELECT create_index_if_not_exists(
    'ix_trades_date_symbol',
    'trades',
    'CREATE INDEX ix_trades_date_symbol ON trades (entry_time, symbol, status)'
);

-- Daily stats lookup
SELECT create_index_if_not_exists(
    'ix_daily_stats_date_range',
    'daily_stats',
    'CREATE INDEX ix_daily_stats_date_range ON daily_stats (date DESC)'
);

-- =============================================================================
-- GIN indexes for full-text search (if needed)
-- =============================================================================

-- Index for searching watchlist notes
SELECT create_index_if_not_exists(
    'ix_watchlist_notes_gin',
    'watchlist',
    'CREATE INDEX ix_watchlist_notes_gin ON watchlist USING gin (notes gin_trgm_ops) WHERE notes IS NOT NULL'
);

-- =============================================================================
-- Cleanup
-- =============================================================================

-- Drop the helper function (optional, can keep for future use)
-- DROP FUNCTION IF EXISTS create_index_if_not_exists;

-- =============================================================================
-- Grant permissions (if using separate application user)
-- =============================================================================
-- Uncomment and modify if you want to use a separate application user

-- CREATE USER IF NOT EXISTS rdt_app WITH PASSWORD 'app_password';
-- GRANT CONNECT ON DATABASE rdt_trading TO rdt_app;
-- GRANT USAGE ON SCHEMA public TO rdt_app;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO rdt_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO rdt_app;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO rdt_app;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO rdt_app;

-- =============================================================================
-- Initialization complete
-- =============================================================================
DO $$
BEGIN
    RAISE NOTICE '==============================================';
    RAISE NOTICE 'RDT Trading System Database Initialized';
    RAISE NOTICE 'Extensions: uuid-ossp, pg_trgm, btree_gin';
    RAISE NOTICE 'Timezone: America/New_York';
    RAISE NOTICE '==============================================';
END $$;

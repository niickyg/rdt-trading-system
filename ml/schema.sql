-- TimescaleDB Schema for ML Feature Storage
--
-- This schema creates tables for storing calculated features for ML models.
-- Features are stored in JSONB format for flexibility and queried efficiently
-- using TimescaleDB's time-series capabilities.

-- Enable TimescaleDB extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create ml_features table
CREATE TABLE IF NOT EXISTS ml_features (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    features JSONB NOT NULL,
    feature_version VARCHAR(10) NOT NULL DEFAULT '1.0',
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS ix_ml_features_symbol ON ml_features(symbol);
CREATE INDEX IF NOT EXISTS ix_ml_features_timestamp ON ml_features(timestamp);
CREATE INDEX IF NOT EXISTS ix_ml_features_symbol_timestamp ON ml_features(symbol, timestamp);

-- Create GIN index for JSONB queries (allows fast feature searches)
CREATE INDEX IF NOT EXISTS ix_ml_features_features ON ml_features USING GIN (features);

-- Convert to TimescaleDB hypertable (partitioned by time)
SELECT create_hypertable(
    'ml_features',
    'timestamp',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Set up data retention policy (optional - keep 90 days of features)
-- SELECT add_retention_policy('ml_features', INTERVAL '90 days');

-- Create compression policy for older data (optional)
-- ALTER TABLE ml_features SET (
--     timescaledb.compress,
--     timescaledb.compress_segmentby = 'symbol'
-- );
-- SELECT add_compression_policy('ml_features', INTERVAL '7 days');

-- Useful queries for feature analysis

-- Get latest features for a symbol
COMMENT ON TABLE ml_features IS 'Stores calculated ML features for stocks';

-- Example query: Get latest features for AAPL
-- SELECT timestamp, features
-- FROM ml_features
-- WHERE symbol = 'AAPL'
-- ORDER BY timestamp DESC
-- LIMIT 1;

-- Example query: Get features for specific time range
-- SELECT timestamp, features->>'rrs' as rrs, features->>'rsi_14' as rsi
-- FROM ml_features
-- WHERE symbol = 'AAPL'
--   AND timestamp >= NOW() - INTERVAL '1 day'
-- ORDER BY timestamp DESC;

-- Example query: Find symbols with high RRS
-- SELECT DISTINCT symbol,
--        timestamp,
--        (features->>'rrs')::float as rrs
-- FROM ml_features
-- WHERE timestamp >= NOW() - INTERVAL '1 hour'
--   AND (features->>'rrs')::float > 2.0
-- ORDER BY (features->>'rrs')::float DESC;

-- Example query: Time series of features for a symbol
-- SELECT timestamp,
--        (features->>'rrs')::float as rrs,
--        (features->>'rsi_14')::float as rsi,
--        (features->>'volume_ratio')::float as volume_ratio
-- FROM ml_features
-- WHERE symbol = 'AAPL'
--   AND timestamp >= NOW() - INTERVAL '1 day'
-- ORDER BY timestamp ASC;

-- Example query: Get feature statistics over time
-- SELECT symbol,
--        COUNT(*) as feature_count,
--        MIN(timestamp) as first_calculation,
--        MAX(timestamp) as last_calculation,
--        AVG((features->>'rrs')::float) as avg_rrs,
--        AVG((features->>'rsi_14')::float) as avg_rsi
-- FROM ml_features
-- WHERE timestamp >= NOW() - INTERVAL '7 days'
-- GROUP BY symbol
-- ORDER BY avg_rrs DESC;

-- Example query: Find breakout candidates
-- SELECT symbol,
--        timestamp,
--        (features->>'rrs')::float as rrs,
--        (features->>'volume_ratio')::float as volume_ratio,
--        (features->>'breakout_probability')::float as breakout_prob
-- FROM ml_features
-- WHERE timestamp >= NOW() - INTERVAL '1 hour'
--   AND (features->>'breakout_probability')::float > 0.7
--   AND (features->>'volume_ratio')::float > 1.5
-- ORDER BY timestamp DESC;

-- Create view for latest features per symbol
CREATE OR REPLACE VIEW latest_features AS
SELECT DISTINCT ON (symbol)
    symbol,
    timestamp,
    features,
    feature_version,
    created_at
FROM ml_features
ORDER BY symbol, timestamp DESC;

COMMENT ON VIEW latest_features IS 'Latest calculated features for each symbol';

-- Create view for strong signals
CREATE OR REPLACE VIEW strong_signals AS
SELECT
    symbol,
    timestamp,
    (features->>'rrs')::float as rrs,
    (features->>'rsi_14')::float as rsi,
    (features->>'volume_ratio')::float as volume_ratio,
    (features->>'breakout_probability')::float as breakout_prob,
    (features->>'daily_alignment_score')::float as alignment_score
FROM ml_features
WHERE timestamp >= NOW() - INTERVAL '1 hour'
  AND (
    ABS((features->>'rrs')::float) > 2.0
    OR (features->>'breakout_probability')::float > 0.7
    OR (features->>'volume_ratio')::float > 2.0
  )
ORDER BY timestamp DESC;

COMMENT ON VIEW strong_signals IS 'Recent features with strong signals';

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT ON ml_features TO trading_app;
-- GRANT USAGE, SELECT ON SEQUENCE ml_features_id_seq TO trading_app;
-- GRANT SELECT ON latest_features, strong_signals TO trading_app;

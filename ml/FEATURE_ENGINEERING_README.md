# Feature Engineering Pipeline

A comprehensive feature engineering system for ML-based trading that calculates 60+ features from market data.

## Overview

The Feature Engineering Pipeline extracts and calculates features across 5 categories:

1. **Technical Indicators (20 features)** - Traditional technical analysis
2. **Microstructure Features (15 features)** - Market microstructure and order flow
3. **Regime Features (10 features)** - Market regime and correlation analysis
4. **Temporal Features (15 features)** - Time-based and session features
5. **Derived Features (10 features)** - Feature interactions and composites

## Features

### Technical Indicators (20 features)

| Feature | Description |
|---------|-------------|
| `rrs` | Real Relative Strength (1-bar) |
| `rrs_3bar` | Real Relative Strength (3-bar) |
| `rrs_5bar` | Real Relative Strength (5-bar) |
| `atr` | Average True Range |
| `atr_percent` | ATR as percentage of price |
| `rsi_14` | Relative Strength Index (14-period) |
| `rsi_9` | Relative Strength Index (9-period) |
| `macd` | MACD line |
| `macd_signal` | MACD signal line |
| `macd_histogram` | MACD histogram |
| `bb_upper` | Bollinger Band upper |
| `bb_middle` | Bollinger Band middle (SMA) |
| `bb_lower` | Bollinger Band lower |
| `bb_width` | Bollinger Band width |
| `bb_percent` | %B indicator (position in bands) |
| `ema_3` | 3-period EMA |
| `ema_8` | 8-period EMA |
| `ema_21` | 21-period EMA |
| `ema_50` | 50-period EMA |
| `volume_sma_20` | 20-period volume SMA |

### Microstructure Features (15 features)

| Feature | Description |
|---------|-------------|
| `bid_ask_spread` | Estimated bid-ask spread |
| `vwap` | Volume Weighted Average Price |
| `vwap_distance` | Distance from VWAP |
| `vwap_distance_percent` | Distance from VWAP (%) |
| `price_momentum_1` | 1-day price momentum |
| `price_momentum_5` | 5-day price momentum |
| `price_momentum_15` | 15-day price momentum |
| `volume_ratio` | Current vs average volume |
| `volume_trend` | 5-day vs 20-day volume |
| `price_range_percent` | Daily range as % of price |
| `intraday_high_low_range` | Intraday range |
| `price_position_in_range` | Position in daily range (0-1) |
| `relative_volume` | Volume vs 20-day average |
| `tick_direction` | Last tick direction (-1, 0, 1) |
| `order_flow_imbalance` | Estimated order flow (-1 to 1) |

### Regime Features (10 features)

| Feature | Description |
|---------|-------------|
| `vix` | VIX volatility index |
| `vix_change` | VIX % change |
| `spy_trend` | SPY 10-day trend |
| `spy_ema_alignment` | SPY EMA alignment (bullish/bearish) |
| `spy_rsi` | SPY RSI |
| `spy_momentum` | SPY 5-day momentum |
| `sector_relative_strength` | Stock vs SPY performance |
| `market_breadth` | Market breadth estimate |
| `spy_volume_ratio` | SPY volume ratio |
| `correlation_with_spy` | 20-day correlation with SPY |

### Temporal Features (15 features)

| Feature | Description |
|---------|-------------|
| `hour_of_day` | Current hour (0-23) |
| `minute_of_hour` | Current minute (0-59) |
| `day_of_week` | Day of week (0=Monday) |
| `day_of_month` | Day of month (1-31) |
| `week_of_year` | Week of year |
| `time_since_open_minutes` | Minutes since market open |
| `time_until_close_minutes` | Minutes until market close |
| `is_market_open` | Market open flag |
| `is_pre_market` | Pre-market flag |
| `is_after_hours` | After-hours flag |
| `is_first_hour` | First trading hour flag |
| `is_last_hour` | Last trading hour flag |
| `is_power_hour` | Power hour flag |
| `is_monday` | Monday flag |
| `is_friday` | Friday flag |

### Derived Features (10 features)

| Feature | Description |
|---------|-------------|
| `rrs_rsi_interaction` | RRS × RSI interaction |
| `momentum_volume_interaction` | Momentum × Volume interaction |
| `volatility_regime_score` | VIX-ATR composite |
| `trend_strength_composite` | EMA alignment strength |
| `reversal_probability` | Mean reversion probability |
| `breakout_probability` | Breakout probability |
| `risk_reward_ratio` | Risk/reward estimate |
| `sharpe_estimate` | Sharpe ratio estimate |
| `daily_alignment_score` | Daily trend alignment |
| `feature_complexity_score` | Signal complexity (0-1) |

## Usage

### Basic Usage

```python
import asyncio
from ml.feature_engineering import FeatureEngineer

async def calculate_features():
    # Initialize
    engineer = FeatureEngineer()

    # Calculate features for a symbol
    features = await engineer.calculate_features("AAPL")

    print(f"Calculated {len(features.columns)} features")
    print(features.head())

asyncio.run(calculate_features())
```

### Batch Processing

```python
# Calculate features for multiple symbols
engineer = FeatureEngineer()
symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
features_dict = await engineer.calculate_batch_features(symbols)

for symbol, df in features_dict.items():
    print(f"{symbol}: RRS = {df['rrs'].iloc[0]:.4f}")
```

### With Caching

```python
# Enable caching for performance
engineer = FeatureEngineer(cache_ttl_seconds=300)  # 5 minutes

# First call - fetches data
features1 = await engineer.calculate_features("AAPL", use_cache=True)

# Second call - uses cache (faster)
features2 = await engineer.calculate_features("AAPL", use_cache=True)
```

### With Database Storage

```python
# Store features to TimescaleDB
engineer = FeatureEngineer(
    db_url='postgresql://user:pass@localhost:5432/trading_db',
    enable_db_storage=True
)

# Features are automatically stored to database
features = await engineer.calculate_features("AAPL")

# Retrieve historical features
historical = engineer.get_historical_features(
    symbol="AAPL",
    start_date=datetime(2024, 1, 1),
    limit=1000
)
```

### Get Features by Category

```python
engineer = FeatureEngineer()

# Get all technical features
technical = engineer.get_feature_names('technical')

# Get all microstructure features
microstructure = engineer.get_feature_names('microstructure')

# Get all feature names
all_features = engineer.get_feature_names()
```

### Cache Persistence

```python
# Save cache to disk
engineer.save_cache('/tmp/feature_cache.pkl')

# Load cache in new session
engineer.load_cache('/tmp/feature_cache.pkl')
```

### Convenience Function

```python
from ml.feature_engineering import calculate_features_for_symbol

# Quick one-liner
features = await calculate_features_for_symbol("AAPL")
```

## Database Schema

When database storage is enabled, features are stored in a TimescaleDB hypertable:

```sql
CREATE TABLE ml_features (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    features JSONB NOT NULL,
    feature_version VARCHAR(10) NOT NULL DEFAULT '1.0',
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX ix_ml_features_symbol ON ml_features(symbol);
CREATE INDEX ix_ml_features_timestamp ON ml_features(timestamp);
CREATE INDEX ix_ml_features_symbol_timestamp ON ml_features(symbol, timestamp);

-- TimescaleDB hypertable (optional)
SELECT create_hypertable('ml_features', 'timestamp');
```

## Integration with Existing Code

The feature engineering pipeline integrates seamlessly with existing components:

### DataProvider Integration

```python
from shared.data_provider import DataProvider
from ml.feature_engineering import FeatureEngineer

# Use existing data provider
data_provider = DataProvider(cache_ttl_seconds=60)
engineer = FeatureEngineer(data_provider=data_provider)

features = await engineer.calculate_features("AAPL")
```

### RRS Calculator Integration

The pipeline uses the existing `RRSCalculator` from `shared.indicators.rrs`:

```python
# Internally uses:
from shared.indicators.rrs import RRSCalculator, calculate_ema, calculate_sma, calculate_vwap

# RRS features are calculated using the standard methodology
rrs_calculator = RRSCalculator(atr_period=14)
```

### ML Model Integration

```python
from ml.feature_engineering import FeatureEngineer
from ml.models.random_forest_model import RandomForestModel

# Calculate features
engineer = FeatureEngineer()
features = await engineer.calculate_features("AAPL")

# Use with ML model
model = RandomForestModel()
prediction = model.predict(features)
```

## Performance Optimization

### 1. Feature Caching

Features are cached in memory for the specified TTL:

```python
engineer = FeatureEngineer(cache_ttl_seconds=300)  # 5-minute cache
```

### 2. Batch Processing

Process multiple symbols in parallel:

```python
# Calculate features for 10 symbols concurrently
symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA',
           'AMD', 'INTC', 'META', 'NFLX', 'AMZN']
features = await engineer.calculate_batch_features(symbols)
```

### 3. Database Storage

Store features to avoid recalculation:

```python
engineer = FeatureEngineer(
    enable_db_storage=True,
    db_url='postgresql://localhost/trading'
)
```

## Error Handling

The pipeline includes comprehensive error handling:

```python
features = await engineer.calculate_features("INVALID_SYMBOL")
# Returns None on error, logs warning

# Check for None before using
if features is not None:
    # Process features
    pass
else:
    # Handle error
    logger.warning("Failed to calculate features")
```

## Logging

All operations are logged using loguru:

```python
from loguru import logger

# Feature calculation logs
logger.info("FeatureEngineer initialized with 70 features")
logger.debug("Calculated 70 features for AAPL")
logger.warning("Failed to fetch data for INVALID")
logger.error("Error calculating technical features: ...")
```

## Testing

Run the comprehensive test suite:

```bash
cd /home/user0/rdt-trading-system
python examples/test_feature_engineering.py
```

This will demonstrate:
- Basic feature calculation
- Batch processing
- Feature categories
- Caching performance
- Convenience functions
- Feature analysis

## Requirements

- Python 3.8+
- pandas
- numpy
- yfinance
- loguru
- sqlalchemy (for database storage)
- asyncio

Optional:
- PostgreSQL/TimescaleDB (for feature storage)

## Advanced Usage

### Custom Feature Selection

```python
# Get specific feature categories
engineer = FeatureEngineer()
features_df = await engineer.calculate_features("AAPL")

# Select only technical and regime features
technical = engineer.get_feature_names('technical')
regime = engineer.get_feature_names('regime')
selected_features = technical + regime

subset = features_df[selected_features]
```

### Feature Importance Analysis

```python
# Identify strong signals
features = await engineer.calculate_features("AAPL")

# Check for strong RRS
if abs(features['rrs'].iloc[0]) > 2.0:
    print("Strong relative strength detected!")

# Check for breakout conditions
if features['breakout_probability'].iloc[0] > 0.7:
    print("High breakout probability!")

# Check trend alignment
if features['daily_alignment_score'].iloc[0] > 0.5:
    print("Strong bullish alignment!")
```

### Historical Analysis

```python
# Retrieve and analyze historical features
engineer = FeatureEngineer(
    db_url='postgresql://localhost/trading',
    enable_db_storage=True
)

# Store features over time
for symbol in watchlist:
    await engineer.calculate_features(symbol)

# Later: analyze historical patterns
historical = engineer.get_historical_features(
    symbol="AAPL",
    start_date=datetime(2024, 1, 1),
    limit=1000
)

# Analyze feature evolution
print(historical[['rrs', 'rsi_14', 'volume_ratio']].describe())
```

## Architecture

```
FeatureEngineer
├── DataProvider Integration
│   └── Fetches market data (OHLCV)
├── RRSCalculator Integration
│   └── Calculates RRS indicators
├── Feature Categories
│   ├── Technical (20 features)
│   ├── Microstructure (15 features)
│   ├── Regime (10 features)
│   ├── Temporal (15 features)
│   └── Derived (10 features)
├── Caching Layer
│   ├── In-memory cache
│   └── Disk persistence
└── Storage Layer
    └── TimescaleDB (optional)
```

## Best Practices

1. **Use caching** for repeated calculations
2. **Batch process** multiple symbols for efficiency
3. **Store to database** for historical analysis
4. **Handle None returns** gracefully
5. **Monitor logs** for errors and warnings
6. **Clear cache** periodically to free memory
7. **Version features** when schema changes

## Troubleshooting

### Features return None

- Check symbol is valid
- Verify market data is available
- Check internet connection
- Review logs for specific errors

### Slow performance

- Enable caching
- Use batch processing
- Store to database
- Reduce cache TTL

### Database errors

- Verify PostgreSQL is running
- Check connection string
- Ensure table permissions
- Review database logs

## Contributing

When adding new features:

1. Add feature name to appropriate category in `_init_feature_names()`
2. Implement calculation in corresponding `_calculate_*_features()` method
3. Add error handling with default values
4. Update this documentation
5. Add tests in test file

## License

Part of the RDT Trading System

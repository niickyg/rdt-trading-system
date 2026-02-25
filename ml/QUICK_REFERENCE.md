# Feature Engineering Quick Reference

## Quick Start

```python
from ml.feature_engineering import FeatureEngineer
import asyncio

async def get_features():
    engineer = FeatureEngineer()
    features = await engineer.calculate_features("AAPL")
    print(features[['rrs', 'rsi_14', 'volume_ratio']])

asyncio.run(get_features())
```

## Common Use Cases

### 1. Screen for Strong Signals

```python
engineer = FeatureEngineer()
watchlist = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
features_dict = await engineer.calculate_batch_features(watchlist)

for symbol, df in features_dict.items():
    rrs = df['rrs'].iloc[0]
    if rrs > 2.0:  # Strong relative strength
        print(f"{symbol}: Strong RS - RRS={rrs:.2f}")
```

### 2. Find Breakout Candidates

```python
engineer = FeatureEngineer()
features = await engineer.calculate_features("AAPL")

breakout_prob = features['breakout_probability'].iloc[0]
volume_ratio = features['volume_ratio'].iloc[0]

if breakout_prob > 0.7 and volume_ratio > 1.5:
    print("Potential breakout setup!")
```

### 3. Analyze Market Regime

```python
engineer = FeatureEngineer()
spy = await engineer.calculate_features("SPY")

vix = spy['vix'].iloc[0]
spy_trend = spy['spy_trend'].iloc[0]

if vix < 15 and spy_trend > 0:
    print("Low volatility uptrend - favorable conditions")
```

### 4. Calculate Risk Metrics

```python
engineer = FeatureEngineer()
features = await engineer.calculate_features("TSLA")

atr_percent = features['atr_percent'].iloc[0]
volatility_regime = features['volatility_regime_score'].iloc[0]

print(f"ATR: {atr_percent:.2f}% of price")
print(f"Volatility Score: {volatility_regime:.2f}")
```

### 5. Monitor Feature Changes

```python
engineer = FeatureEngineer(cache_ttl_seconds=10)

# Get features at T0
features_t0 = await engineer.calculate_features("AAPL", use_cache=False)
rrs_t0 = features_t0['rrs'].iloc[0]

await asyncio.sleep(60)

# Get features at T1
features_t1 = await engineer.calculate_features("AAPL", use_cache=False)
rrs_t1 = features_t1['rrs'].iloc[0]

print(f"RRS change: {rrs_t1 - rrs_t0:+.2f}")
```

## Feature Categories

### Technical (20 features)
Most important: `rrs`, `rsi_14`, `macd`, `atr_percent`, `bb_percent`

### Microstructure (15 features)
Most important: `volume_ratio`, `vwap_distance_percent`, `price_momentum_5`

### Regime (10 features)
Most important: `vix`, `spy_trend`, `correlation_with_spy`

### Temporal (15 features)
Most important: `is_market_open`, `is_first_hour`, `time_since_open_minutes`

### Derived (10 features)
Most important: `breakout_probability`, `daily_alignment_score`, `trend_strength_composite`

## Signal Interpretation

### Strong Bullish Setup
```python
rrs > 2.0                           # Strong relative strength
30 < rsi_14 < 70                    # Not overbought
volume_ratio > 1.5                  # High volume
daily_alignment_score > 0.5         # Trend aligned
breakout_probability > 0.7          # Breakout likely
```

### Strong Bearish Setup
```python
rrs < -2.0                          # Strong relative weakness
30 < rsi_14 < 70                    # Not oversold
volume_ratio > 1.5                  # High volume
daily_alignment_score < -0.5        # Downtrend aligned
reversal_probability < 0.3          # No reversal expected
```

### Overbought Warning
```python
rsi_14 > 70                         # Overbought RSI
bb_percent > 0.95                   # At upper Bollinger Band
rrs > 3.0                           # Extended move
reversal_probability > 0.7          # Reversal likely
```

### Oversold Bounce
```python
rsi_14 < 30                         # Oversold RSI
bb_percent < 0.05                   # At lower Bollinger Band
volume_ratio > 2.0                  # High volume
reversal_probability > 0.7          # Reversal likely
```

## Performance Tips

1. **Use caching** for repeated queries:
   ```python
   engineer = FeatureEngineer(cache_ttl_seconds=300)
   features = await engineer.calculate_features("AAPL", use_cache=True)
   ```

2. **Batch process** multiple symbols:
   ```python
   features_dict = await engineer.calculate_batch_features(symbols)
   ```

3. **Store to database** for historical analysis:
   ```python
   engineer = FeatureEngineer(
       db_url='postgresql://localhost/trading',
       enable_db_storage=True
   )
   ```

4. **Save cache to disk**:
   ```python
   engineer.save_cache('/tmp/features.pkl')
   # Later...
   engineer.load_cache('/tmp/features.pkl')
   ```

## Database Queries

### Get latest features for symbol
```sql
SELECT features
FROM ml_features
WHERE symbol = 'AAPL'
ORDER BY timestamp DESC
LIMIT 1;
```

### Find high RRS stocks
```sql
SELECT symbol, (features->>'rrs')::float as rrs
FROM ml_features
WHERE timestamp >= NOW() - INTERVAL '1 hour'
  AND (features->>'rrs')::float > 2.0
ORDER BY rrs DESC;
```

### Track feature over time
```sql
SELECT timestamp,
       (features->>'rrs')::float as rrs
FROM ml_features
WHERE symbol = 'AAPL'
  AND timestamp >= NOW() - INTERVAL '1 day'
ORDER BY timestamp;
```

## Error Handling

Always check for None:
```python
features = await engineer.calculate_features("SYMBOL")
if features is not None:
    # Use features
    rrs = features['rrs'].iloc[0]
else:
    # Handle error
    logger.warning("Failed to calculate features")
```

## Integration Examples

### With Scanner
```python
from scanner.realtime_scanner import RealtimeScanner
from ml.feature_engineering import FeatureEngineer

async def enhanced_scan():
    scanner = RealtimeScanner()
    engineer = FeatureEngineer()

    signals = await scanner.scan()
    for signal in signals:
        features = await engineer.calculate_features(signal['symbol'])
        if features is not None:
            signal['features'] = features.to_dict('records')[0]

    return signals
```

### With Trading Bot
```python
from automation.trading_bot import TradingBot
from ml.feature_engineering import FeatureEngineer

async def trade_with_features():
    bot = TradingBot()
    engineer = FeatureEngineer()

    # Get signal
    signal = await bot.find_signal()

    # Calculate features
    features = await engineer.calculate_features(signal['symbol'])

    # Enhanced decision
    if features is not None:
        if (features['rrs'].iloc[0] > 2.0 and
            features['volume_ratio'].iloc[0] > 1.5 and
            features['breakout_probability'].iloc[0] > 0.7):
            await bot.enter_trade(signal)
```

### With Risk Manager
```python
from risk.risk_manager import RiskManager
from ml.feature_engineering import FeatureEngineer

async def calculate_position_size():
    risk_mgr = RiskManager()
    engineer = FeatureEngineer()

    features = await engineer.calculate_features("AAPL")

    # Adjust position size based on volatility
    atr_percent = features['atr_percent'].iloc[0]
    base_size = 100

    if atr_percent > 3.0:  # High volatility
        position_size = base_size * 0.5
    elif atr_percent > 2.0:  # Medium volatility
        position_size = base_size * 0.75
    else:  # Low volatility
        position_size = base_size

    return position_size
```

## Testing

Run the test suite:
```bash
cd /home/user0/rdt-trading-system
python examples/test_feature_engineering.py
```

Run integration examples:
```bash
python examples/feature_engineering_ml_integration.py
```

## Documentation

- Full documentation: `ml/FEATURE_ENGINEERING_README.md`
- Database schema: `ml/schema.sql`
- API reference: See docstrings in `ml/feature_engineering.py`

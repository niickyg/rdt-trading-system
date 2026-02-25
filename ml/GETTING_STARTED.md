# Getting Started with Feature Engineering

## 5-Minute Quick Start

### 1. Import and Initialize

```python
from ml.feature_engineering import FeatureEngineer
import asyncio

async def main():
    # Create feature engineer instance
    engineer = FeatureEngineer()

    # That's it! You're ready to calculate features
    print("Feature Engineer initialized with 70 features")

asyncio.run(main())
```

### 2. Calculate Features for a Stock

```python
async def calculate():
    engineer = FeatureEngineer()

    # Calculate features for AAPL
    features = await engineer.calculate_features("AAPL")

    if features is not None:
        # Print some key features
        print(f"RRS: {features['rrs'].iloc[0]:.4f}")
        print(f"RSI: {features['rsi_14'].iloc[0]:.2f}")
        print(f"Volume Ratio: {features['volume_ratio'].iloc[0]:.2f}")
        print(f"Breakout Probability: {features['breakout_probability'].iloc[0]:.2f}")
    else:
        print("Failed to calculate features")

asyncio.run(calculate())
```

### 3. Process Your Watchlist

```python
async def scan_watchlist():
    engineer = FeatureEngineer()

    # Your watchlist
    watchlist = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

    # Calculate features for all symbols (in parallel!)
    features_dict = await engineer.calculate_batch_features(watchlist)

    # Find strong signals
    for symbol, df in features_dict.items():
        if df is not None:
            rrs = df['rrs'].iloc[0]
            if rrs > 2.0:
                print(f"{symbol}: STRONG RELATIVE STRENGTH - RRS={rrs:.2f}")

asyncio.run(scan_watchlist())
```

## Common Use Cases

### Find Breakout Candidates

```python
async def find_breakouts():
    engineer = FeatureEngineer()
    watchlist = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD', 'META']

    features_dict = await engineer.calculate_batch_features(watchlist)

    breakout_candidates = []
    for symbol, df in features_dict.items():
        if df is None:
            continue

        # Check breakout conditions
        if (df['breakout_probability'].iloc[0] > 0.7 and
            df['volume_ratio'].iloc[0] > 1.5 and
            df['rrs'].iloc[0] > 1.0):

            breakout_candidates.append({
                'symbol': symbol,
                'breakout_prob': df['breakout_probability'].iloc[0],
                'volume_ratio': df['volume_ratio'].iloc[0],
                'rrs': df['rrs'].iloc[0]
            })

    if breakout_candidates:
        print("Breakout Candidates Found:")
        for candidate in breakout_candidates:
            print(f"  {candidate['symbol']}: "
                  f"Breakout={candidate['breakout_prob']:.2f}, "
                  f"Vol={candidate['volume_ratio']:.2f}, "
                  f"RRS={candidate['rrs']:.2f}")
    else:
        print("No breakout candidates found")

asyncio.run(find_breakouts())
```

### Check Market Regime

```python
async def check_market():
    engineer = FeatureEngineer()

    # Get SPY features to determine market regime
    spy = await engineer.calculate_features("SPY")

    if spy is not None:
        vix = spy['vix'].iloc[0]
        spy_trend = spy['spy_trend'].iloc[0]
        spy_rsi = spy['spy_rsi'].iloc[0]

        print(f"Market Regime:")
        print(f"  VIX: {vix:.2f}")
        print(f"  SPY Trend (10d): {spy_trend:.2f}%")
        print(f"  SPY RSI: {spy_rsi:.2f}")

        # Interpret
        if vix < 15:
            print("  Volatility: LOW (calm market)")
        elif vix < 20:
            print("  Volatility: NORMAL")
        elif vix < 30:
            print("  Volatility: ELEVATED")
        else:
            print("  Volatility: HIGH (fear)")

        if spy_trend > 2:
            print("  Trend: STRONG UPTREND")
        elif spy_trend > 0:
            print("  Trend: UPTREND")
        elif spy_trend > -2:
            print("  Trend: DOWNTREND")
        else:
            print("  Trend: STRONG DOWNTREND")

asyncio.run(check_market())
```

### Rank Stocks by Strength

```python
async def rank_stocks():
    engineer = FeatureEngineer()
    watchlist = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD', 'META', 'NFLX']

    features_dict = await engineer.calculate_batch_features(watchlist)

    rankings = []
    for symbol, df in features_dict.items():
        if df is not None:
            rankings.append({
                'symbol': symbol,
                'rrs': df['rrs'].iloc[0],
                'rsi': df['rsi_14'].iloc[0],
                'volume_ratio': df['volume_ratio'].iloc[0]
            })

    # Sort by RRS
    rankings.sort(key=lambda x: x['rrs'], reverse=True)

    print("Stock Rankings (by RRS):")
    print(f"{'Symbol':<8} {'RRS':>8} {'RSI':>8} {'Volume':>10}")
    print("-" * 40)
    for r in rankings:
        print(f"{r['symbol']:<8} {r['rrs']:>8.2f} {r['rsi']:>8.2f} {r['volume_ratio']:>10.2f}")

asyncio.run(rank_stocks())
```

## Performance Tips

### 1. Enable Caching

```python
# Cache features for 5 minutes
engineer = FeatureEngineer(cache_ttl_seconds=300)

# First call - fetches data (slow)
features = await engineer.calculate_features("AAPL", use_cache=True)

# Second call within 5 minutes - uses cache (fast!)
features = await engineer.calculate_features("AAPL", use_cache=True)
```

### 2. Use Batch Processing

```python
# Good: Parallel processing
watchlist = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
features_dict = await engineer.calculate_batch_features(watchlist)

# Avoid: Sequential processing
for symbol in watchlist:
    features = await engineer.calculate_features(symbol)
```

### 3. Convenience Function

```python
# Quick one-liner for single symbol
from ml.feature_engineering import calculate_features_for_symbol

features = await calculate_features_for_symbol("AAPL")
```

## Integration Examples

### With Existing Scanner

```python
from scanner.realtime_scanner import RealtimeScanner
from ml.feature_engineering import FeatureEngineer

async def enhanced_scan():
    scanner = RealtimeScanner()
    engineer = FeatureEngineer()

    # Get signals from scanner
    signals = await scanner.scan()

    # Enhance with features
    for signal in signals:
        features = await engineer.calculate_features(signal['symbol'])

        if features is not None:
            # Add key features to signal
            signal['volume_ratio'] = features['volume_ratio'].iloc[0]
            signal['breakout_prob'] = features['breakout_probability'].iloc[0]
            signal['rsi'] = features['rsi_14'].iloc[0]

    return signals
```

### With Trading Bot

```python
from automation.trading_bot import TradingBot
from ml.feature_engineering import FeatureEngineer

async def smart_trading():
    bot = TradingBot()
    engineer = FeatureEngineer()

    # Get potential signal
    signal = await bot.find_signal()

    # Calculate features
    features = await engineer.calculate_features(signal['symbol'])

    # Enhanced entry decision
    if features is not None:
        # Only enter if multiple conditions align
        if (features['rrs'].iloc[0] > 2.0 and
            features['volume_ratio'].iloc[0] > 1.5 and
            features['breakout_probability'].iloc[0] > 0.7 and
            30 < features['rsi_14'].iloc[0] < 70):

            print(f"Strong setup detected for {signal['symbol']}")
            await bot.enter_trade(signal)
        else:
            print(f"Signal filtered out by feature analysis")
    else:
        print("Could not calculate features")
```

## Next Steps

### Learn More
- **Feature Catalog**: See `FEATURE_CATALOG.md` for all 70 features
- **Full Documentation**: See `FEATURE_ENGINEERING_README.md`
- **Quick Reference**: See `QUICK_REFERENCE.md`

### Run Examples
```bash
# Basic examples
python examples/test_feature_engineering.py

# Integration examples
python examples/feature_engineering_ml_integration.py
```

### Advanced Features
1. **Database Storage** - Store features to TimescaleDB for historical analysis
2. **Custom Feature Selection** - Select specific feature categories
3. **Feature Monitoring** - Track feature changes over time
4. **Risk Analysis** - Use volatility features for position sizing

## Troubleshooting

### Features return None
- Check symbol is valid
- Verify internet connection
- Check logs for specific error

### Slow performance
- Enable caching with `cache_ttl_seconds=300`
- Use batch processing for multiple symbols
- Consider database storage to avoid recalculation

### Import errors
```bash
# Make sure you're in the right directory
cd /home/user0/rdt-trading-system

# Or add to Python path
export PYTHONPATH="/home/user0/rdt-trading-system:$PYTHONPATH"
```

## Support

For questions or issues:
1. Check the documentation in `FEATURE_ENGINEERING_README.md`
2. Review examples in `examples/`
3. Check the feature catalog in `FEATURE_CATALOG.md`

## Summary

You now have a powerful feature engineering pipeline that:
- ✅ Calculates 70+ features automatically
- ✅ Works with any stock symbol
- ✅ Supports batch processing
- ✅ Includes caching for performance
- ✅ Integrates with existing trading system
- ✅ Ready to use immediately

Start with the simple examples above and build from there!

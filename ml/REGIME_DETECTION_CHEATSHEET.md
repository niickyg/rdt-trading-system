# Regime Detection Cheat Sheet

Quick reference for the Market Regime Detection System.

## Installation

```bash
pip install hmmlearn scikit-learn matplotlib seaborn yfinance loguru
```

## Training

```bash
# Basic
python scripts/train_regime_detector.py

# With plots
python scripts/train_regime_detector.py --visualize

# Full evaluation
python scripts/train_regime_detector.py --visualize --evaluate --verbose
```

## Python Usage

### Quick Start

```python
from ml.regime_detector import MarketRegimeDetector

# Load model
detector = MarketRegimeDetector()
detector.load_model()

# Get current regime
regime, info = detector.predict(return_confidence=True)
```

### Training

```python
# Create and train
detector = MarketRegimeDetector(n_regimes=4, n_iter=100)
metrics = detector.train(symbol="SPY", period="5y")

# Save
detector.save_model()
```

### Prediction

```python
# Current regime
regime, info = detector.predict()

# Access results
print(f"Regime: {regime}")
print(f"Confidence: {info['confidence_scores'][regime]:.1%}")
print(f"Allocation: {info['strategy_allocation']}")
```

### Historical Analysis

```python
import yfinance as yf

# Get data
data = yf.Ticker("SPY").history(period="1y")

# Predict sequence
results = detector.predict_sequence(data)

# Get transitions
transitions = detector.get_regime_transitions(results)

# Get statistics
stats = detector.get_regime_statistics(data)
```

### Strategy Integration

```python
from examples.regime_strategy_integration import RegimeAwareStrategyManager

# Initialize
manager = RegimeAwareStrategyManager()
manager.update_regime()

# Get weights
weights = manager.get_strategy_weights()

# Adjust position size
adjusted = manager.adjust_position_size(1000, 'momentum')

# Filter trades
should_take = manager.should_take_trade('momentum', signal_strength=0.75)

# Get risk params
risk_params = manager.get_risk_parameters()
```

## Regimes

| Regime | Returns | Volatility | Allocation (M/MR/P/O) |
|--------|---------|------------|----------------------|
| Bull Trending | High + | Medium | 60/10/20/10 |
| Bear Trending | Negative | High | 30/30/20/20 |
| High Volatility | Mixed | Very High | 20/20/20/40 |
| Low Volatility | Steady | Low | 40/40/10/10 |

M=Momentum, MR=Mean Reversion, P=Pairs, O=Options

## Features (17)

**Returns**: 1d, 5d, 20d
**Volatility**: 10d, 30d std
**Volume**: 5d, 20d ratios
**Trend**: SMA crosses, momentum
**Price**: HL ratio, ATR, RSI

## Commands

```bash
# Train
python scripts/train_regime_detector.py [--symbol SPY] [--period 5y] [--visualize]

# Demo
python examples/regime_detection_demo.py

# Integration
python examples/regime_strategy_integration.py

# Test
pytest tests/test_regime_detector.py -v
```

## File Locations

```
models/regime_detector.pkl          # Trained model
models/training_metrics.json        # Metrics
ml/data/visualizations/             # Plots
logs/regime_training_*.log          # Logs
```

## API Quick Reference

```python
# Initialize
detector = MarketRegimeDetector(n_regimes=4, n_iter=100, random_state=42)

# Train
metrics = detector.train(symbol="SPY", period="5y")

# Save/Load
detector.save_model(path)
detector.load_model(path)

# Predict
regime, info = detector.predict(data=None, return_confidence=True)
results = detector.predict_sequence(data)

# Analysis
stats = detector.get_regime_statistics(data, regimes)
transitions = detector.get_regime_transitions(results)
allocation = detector.get_strategy_allocation(regime_name)
```

## Confidence Interpretation

- **>70%**: High confidence, strong signal
- **50-70%**: Moderate confidence
- **<50%**: Low confidence, possible transition

## Strategy Actions by Regime

### Bull Trending
- ✓ Long momentum stocks
- ✓ Tight stops
- ✓ Sector rotation to growth
- ✗ Avoid heavy mean reversion

### Bear Trending
- ✓ Inverse ETFs/shorts
- ✓ Mean reversion on oversold
- ✓ Protective puts
- ✗ Reduce overall exposure

### High Volatility
- ✓ Reduce position sizes
- ✓ Widen stops
- ✓ Sell options premium
- ✗ Avoid high leverage

### Low Volatility
- ✓ Higher leverage OK
- ✓ Tight range strategies
- ✓ Buy options (low IV)
- ✗ Don't sell premium

## Troubleshooting

```bash
# No hmmlearn
pip install hmmlearn

# No model
python scripts/train_regime_detector.py

# Convergence warning
python scripts/train_regime_detector.py --n-iter 200

# Check logs
tail -f logs/regime_training_*.log
```

## Performance

- **Training**: 30-60s for 5 years
- **Prediction**: <1s
- **Memory**: ~50-100 MB
- **Update**: Daily or weekly

## Integration Pattern

```python
# Daily regime check
import schedule

def daily_check():
    detector = MarketRegimeDetector()
    detector.load_model()
    regime, info = detector.predict()

    # Update your system
    update_strategy_weights(info['strategy_allocation'])

schedule.every().day.at("09:30").do(daily_check)
```

## Advanced

```python
# Custom features
class CustomDetector(MarketRegimeDetector):
    def extract_features(self, data):
        features = super().extract_features(data)
        features['custom'] = my_indicator(data)
        return features

# Different regimes
detector = MarketRegimeDetector(n_regimes=3)  # bull, bear, neutral

# Different asset
detector.train(symbol="QQQ", period="10y")
```

## Links

- Full Docs: `ml/REGIME_DETECTION_README.md`
- Quick Start: `ml/QUICKSTART_REGIME_DETECTION.md`
- Setup Guide: `REGIME_DETECTION_SETUP.md`

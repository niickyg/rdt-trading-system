# Regime Detection Quick Start Guide

Get up and running with the Market Regime Detection system in 5 minutes.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install just the ML dependencies
pip install hmmlearn scikit-learn matplotlib seaborn yfinance loguru
```

## 1. Train Your First Model (2 minutes)

```bash
cd /home/user0/rdt-trading-system

# Basic training
python scripts/train_regime_detector.py

# With visualizations (recommended for first run)
python scripts/train_regime_detector.py --visualize --verbose
```

**What this does:**
- Downloads 5 years of SPY data
- Extracts 17 technical features
- Trains Hidden Markov Model with 4 regimes
- Saves model to `models/regime_detector.pkl`
- Shows current market regime and allocation

**Expected output:**
```
Current Market Regime: BULL_TRENDING
Confidence: 72.4%

Strategy Allocation:
  momentum:          60.0% ████████████████████████
  mean_reversion:    10.0% ████
  pairs:             20.0% ████████
  options:           10.0% ████
```

## 2. Use in Your Code (1 minute)

```python
from ml.regime_detector import MarketRegimeDetector

# Load pre-trained model
detector = MarketRegimeDetector()
detector.load_model()

# Get current regime
regime, info = detector.predict(return_confidence=True)

print(f"Current regime: {regime}")
print(f"Confidence: {info['confidence_scores'][regime]:.1%}")
print(f"Allocation: {info['strategy_allocation']}")
```

## 3. Run Demo (1 minute)

```bash
# Complete demo with all features
python examples/regime_detection_demo.py

# Strategy integration example
python examples/regime_strategy_integration.py
```

## 4. Integrate with Trading System (1 minute)

```python
from examples.regime_strategy_integration import RegimeAwareStrategyManager

# Initialize manager
manager = RegimeAwareStrategyManager()

# Update regime
manager.update_regime()
manager.log_regime_status()

# Get strategy weights
weights = manager.get_strategy_weights()
# {'momentum': 0.6, 'mean_reversion': 0.1, 'pairs': 0.2, 'options': 0.1}

# Adjust position sizing
base_size = 1000
adjusted_size = manager.adjust_position_size(base_size, 'momentum')

# Get risk parameters
risk_params = manager.get_risk_parameters()
```

## Common Use Cases

### 1. Daily Regime Check

```python
# Check regime each morning
detector = MarketRegimeDetector()
detector.load_model()

regime, info = detector.predict()
print(f"Today's regime: {regime}")

# Adjust your trading plan accordingly
if regime == 'high_volatility':
    print("Reduce position sizes today!")
```

### 2. Automated Strategy Switching

```python
from examples.regime_strategy_integration import RegimeAwareStrategyManager

manager = RegimeAwareStrategyManager()
manager.update_regime()

# Get optimal strategy weights
weights = manager.get_strategy_weights()

# Apply to your multi-strategy engine
# engine.set_weights(weights)
```

### 3. Historical Analysis

```python
import yfinance as yf

# Get historical data
spy = yf.Ticker("SPY")
data = spy.history(period="1y")

# Predict regime sequence
results = detector.predict_sequence(data)

# Analyze transitions
transitions = detector.get_regime_transitions(results)
print(f"Number of transitions: {len(transitions)}")

# Get statistics
stats = detector.get_regime_statistics(data)
print(stats)
```

### 4. Real-time Monitoring

```python
import schedule
import time

def check_regime():
    detector = MarketRegimeDetector()
    detector.load_model()
    regime, info = detector.predict()

    # Log or send alert if regime changed
    print(f"Current regime: {regime}")

# Check every hour during market hours
schedule.every().hour.do(check_regime)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Understanding the Output

### Regimes

1. **Bull Trending** (Green)
   - Strong upward momentum
   - Low to moderate volatility
   - Positive returns
   - **Action**: Favor momentum strategies

2. **Bear Trending** (Red)
   - Strong downward momentum
   - Moderate to high volatility
   - Negative returns
   - **Action**: Reduce exposure, use mean reversion

3. **High Volatility** (Orange)
   - Large price swings
   - High uncertainty
   - Mixed returns
   - **Action**: Reduce position sizes, sell options premium

4. **Low Volatility** (Blue)
   - Small price movements
   - Low uncertainty
   - Steady trends
   - **Action**: Can use leverage, tight mean reversion

### Confidence Scores

- **> 70%**: High confidence, strong regime signal
- **50-70%**: Moderate confidence, clear regime
- **< 50%**: Low confidence, regime may be transitioning

### Strategy Allocations

Automatically calculated based on regime:

```python
# Bull Trending
{'momentum': 0.60, 'mean_reversion': 0.10, 'pairs': 0.20, 'options': 0.10}

# Bear Trending
{'momentum': 0.30, 'mean_reversion': 0.30, 'pairs': 0.20, 'options': 0.20}

# High Volatility
{'momentum': 0.20, 'mean_reversion': 0.20, 'pairs': 0.20, 'options': 0.40}

# Low Volatility
{'momentum': 0.40, 'mean_reversion': 0.40, 'pairs': 0.10, 'options': 0.10}
```

## Advanced Options

### Custom Training

```bash
# Train on different symbol
python scripts/train_regime_detector.py --symbol QQQ

# Use more data
python scripts/train_regime_detector.py --period 10y

# More iterations for better fit
python scripts/train_regime_detector.py --n-iter 200

# Run evaluation
python scripts/train_regime_detector.py --evaluate --test-symbol DIA
```

### Custom Features

```python
class CustomDetector(MarketRegimeDetector):
    def extract_features(self, data):
        features = super().extract_features(data)
        # Add your custom features
        features['my_indicator'] = calculate_my_indicator(data)
        return features

detector = CustomDetector()
detector.train()
```

### Different Regime Count

```python
# 3 regimes (bull, bear, neutral)
detector = MarketRegimeDetector(n_regimes=3)

# 5 regimes (more granular)
detector = MarketRegimeDetector(n_regimes=5)

metrics = detector.train()
```

## Troubleshooting

### "No module named 'hmmlearn'"
```bash
pip install hmmlearn
```

### "Model file not found"
Train a model first:
```bash
python scripts/train_regime_detector.py
```

### "No convergence" warning
Increase iterations:
```python
detector = MarketRegimeDetector(n_iter=200)
```

### Poor regime separation
- Try different features
- Adjust number of regimes
- Use more training data
- Check data quality

## Next Steps

1. **Backtest**: Test strategy allocations on historical data
2. **Monitor**: Set up daily regime checks
3. **Automate**: Integrate with your trading bot
4. **Customize**: Add your own features or regimes
5. **Visualize**: Generate plots to understand regime behavior

## File Locations

```
models/
  └── regime_detector.pkl          # Trained model
  └── training_metrics.json        # Training statistics

ml/data/visualizations/
  ├── regime_overlay.png           # Price chart with regimes
  ├── regime_distribution.png      # Regime frequency
  ├── confidence_scores.png        # Confidence over time
  ├── transition_matrix.png        # Regime transitions
  └── regime_statistics.png        # Performance by regime

logs/
  └── regime_training_*.log        # Training logs
```

## Resources

- **Full Documentation**: `ml/REGIME_DETECTION_README.md`
- **API Reference**: See docstrings in `ml/regime_detector.py`
- **Examples**: `examples/regime_detection_demo.py`
- **Tests**: `tests/test_regime_detector.py`

## Performance Tips

1. **Training**: ~30-60 seconds for 5 years of data
2. **Prediction**: <1 second for current regime
3. **Memory**: ~50-100 MB for trained model
4. **Update Frequency**: Check regime daily or weekly

## Need Help?

1. Check logs: `logs/regime_training_*.log`
2. Run tests: `pytest tests/test_regime_detector.py -v`
3. Run demo: `python examples/regime_detection_demo.py`
4. Enable verbose: `--verbose` flag

---

**Ready to start?**

```bash
# Train your first model
python scripts/train_regime_detector.py --visualize

# Run the demo
python examples/regime_detection_demo.py

# Start trading!
```

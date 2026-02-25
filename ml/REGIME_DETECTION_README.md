# Market Regime Detection System

A sophisticated market regime detection system using Hidden Markov Models (HMM) to identify market states and dynamically allocate trading strategies.

## Overview

This system uses HMM to detect four distinct market regimes:

1. **Bull Trending** - Strong upward momentum
2. **Bear Trending** - Strong downward momentum
3. **High Volatility** - Elevated market uncertainty
4. **Low Volatility** - Calm market conditions

## Features

- **Advanced Feature Engineering**: 17+ technical indicators including returns, volatility, volume ratios, and trend indicators
- **Real-time Prediction**: Current regime detection with confidence scores
- **Sequence Analysis**: Historical regime identification over time series
- **Strategy Allocation**: Automatic portfolio allocation based on detected regime
- **Model Persistence**: Save and load trained models
- **Validation Metrics**: Silhouette score, Davies-Bouldin index, log-likelihood
- **Visualization**: Comprehensive plots for regime analysis
- **Error Handling**: Robust exception handling and logging with loguru

## Installation

### Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- `hmmlearn>=0.3.0` - Hidden Markov Models
- `scikit-learn>=1.3.0` - Machine learning utilities
- `yfinance>=0.2.28` - Market data
- `loguru>=0.7.0` - Logging
- `matplotlib>=3.7.0` - Visualization
- `seaborn>=0.12.0` - Statistical visualization

## Quick Start

### 1. Train a Model

```bash
# Basic training
python scripts/train_regime_detector.py

# With visualization and evaluation
python scripts/train_regime_detector.py --visualize --evaluate --verbose

# Custom parameters
python scripts/train_regime_detector.py --symbol SPY --period 10y --n-regimes 4 --n-iter 200
```

### 2. Use in Python

```python
from ml.regime_detector import MarketRegimeDetector

# Initialize and train
detector = MarketRegimeDetector(n_regimes=4, n_iter=100)
metrics = detector.train(symbol="SPY", period="5y")

# Save model
detector.save_model()

# Predict current regime
regime, info = detector.predict(return_confidence=True)
print(f"Current regime: {regime}")
print(f"Confidence: {info['confidence_scores'][regime]:.2%}")
print(f"Allocation: {info['strategy_allocation']}")
```

### 3. Load Pre-trained Model

```python
# Load existing model
detector = MarketRegimeDetector()
detector.load_model()

# Make predictions
regime, info = detector.predict()
```

## Usage Examples

### Basic Prediction

```python
from ml.regime_detector import MarketRegimeDetector

detector = MarketRegimeDetector()
detector.load_model()

# Get current regime
regime, info = detector.predict(return_confidence=True)

print(f"Regime: {regime}")
print(f"Timestamp: {info['timestamp']}")
print(f"Confidence Scores: {info['confidence_scores']}")
print(f"Strategy Allocation: {info['strategy_allocation']}")
```

### Sequence Prediction

```python
import yfinance as yf

# Fetch data
spy = yf.Ticker("SPY")
data = spy.history(period="1y")

# Predict regime sequence
results = detector.predict_sequence(data)

# Analyze transitions
transitions = detector.get_regime_transitions(results)
print(f"Found {len(transitions)} regime transitions")
```

### Regime Statistics

```python
# Get performance metrics for each regime
stats = detector.get_regime_statistics(data)
print(stats)

# Output includes:
# - Average returns per regime
# - Volatility per regime
# - Sharpe ratio per regime
# - Maximum drawdown per regime
# - Sample count per regime
```

### Strategy Integration

```python
# Get recommended strategy allocation
regime, info = detector.predict()
allocation = info['strategy_allocation']

# Apply to portfolio
portfolio_value = 100000
for strategy, weight in allocation.items():
    amount = portfolio_value * weight
    print(f"{strategy}: ${amount:,.2f} ({weight:.1%})")
```

## Strategy Allocations

### Bull Trending
- Momentum: 60%
- Mean Reversion: 10%
- Pairs Trading: 20%
- Options: 10%

**Rationale**: Focus on momentum during strong uptrends with some diversification

### Bear Trending
- Momentum: 30%
- Mean Reversion: 30%
- Pairs Trading: 20%
- Options: 20%

**Rationale**: Balanced allocation with increased mean reversion and protection

### High Volatility
- Momentum: 20%
- Mean Reversion: 20%
- Pairs Trading: 20%
- Options: 40%

**Rationale**: Heavy options allocation to capitalize on elevated implied volatility

### Low Volatility
- Momentum: 40%
- Mean Reversion: 40%
- Pairs Trading: 10%
- Options: 10%

**Rationale**: Split between momentum and mean reversion in calm markets

## Feature Engineering

The system extracts 17 features from OHLCV data:

### Returns
- 1-day return
- 5-day return
- 20-day return

### Volatility
- 10-day rolling standard deviation
- 30-day rolling standard deviation

### Volume Analysis
- 5-day volume ratio
- 20-day volume ratio

### Trend Indicators
- 10/50 SMA crossover
- 50/200 SMA crossover
- 5-day momentum
- 20-day momentum

### Price Metrics
- High-Low ratio
- 14-day ATR (Average True Range)
- 14-day RSI (Relative Strength Index)

## Model Architecture

### Hidden Markov Model Configuration

```python
GaussianHMM(
    n_components=4,           # Number of regimes
    covariance_type="full",   # Full covariance matrix
    n_iter=100,               # EM algorithm iterations
    random_state=42           # Reproducibility
)
```

### Training Process

1. **Data Fetching**: Download 5 years of SPY daily data
2. **Feature Extraction**: Calculate 17 technical indicators
3. **Scaling**: Standardize features using StandardScaler
4. **HMM Training**: Fit Gaussian HMM using EM algorithm
5. **Regime Mapping**: Map HMM states to meaningful labels
6. **Validation**: Calculate clustering metrics

### Regime Mapping Logic

States are mapped based on feature characteristics:
- **Bull Trending**: Highest average 20-day return
- **Bear Trending**: Lowest average 20-day return
- **High Volatility**: Highest average 30-day volatility
- **Low Volatility**: Remaining state (lowest volatility)

## Training Script Options

```bash
python scripts/train_regime_detector.py [OPTIONS]

Options:
  --symbol SYMBOL          Ticker symbol (default: SPY)
  --period PERIOD          Data period (default: 5y)
  --n-regimes N           Number of regimes (default: 4)
  --n-iter N              HMM iterations (default: 100)
  --output PATH           Model output path
  --visualize             Generate plots
  --evaluate              Run evaluation on QQQ
  --test-symbol SYMBOL    Test evaluation symbol (default: QQQ)
  --verbose               Enable verbose logging
```

## Visualization

When using `--visualize`, the system generates:

1. **Regime Overlay**: Price chart with colored regime regions
2. **Regime Distribution**: Pie and bar charts showing regime frequency
3. **Confidence Scores**: Time series of confidence for each regime
4. **Transition Matrix**: Heatmap of regime transition probabilities
5. **Regime Statistics**: Bar charts comparing regime performance

Plots are saved to `/ml/data/visualizations/`

## Validation Metrics

### Training Metrics
- **Log Likelihood**: Model fit quality (higher is better)
- **Silhouette Score**: Cluster separation (-1 to 1, higher is better)
- **Davies-Bouldin Score**: Cluster compactness (lower is better)
- **Regime Distribution**: Sample count per regime

### Evaluation Metrics
- Out-of-sample performance on different ticker (QQQ)
- Regime transition analysis
- Strategy performance statistics
- Sharpe ratios by regime

## Integration with Trading System

### Real-time Monitoring

```python
import schedule
import time

def check_regime():
    detector = MarketRegimeDetector()
    detector.load_model()

    regime, info = detector.predict()
    allocation = info['strategy_allocation']

    # Update your trading system
    update_strategy_weights(allocation)
    log_regime_change(regime)

# Check every hour during market hours
schedule.every().hour.do(check_regime)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Multi-Strategy Engine Integration

```python
from ml.regime_detector import MarketRegimeDetector
from strategies.multi_strategy_engine import MultiStrategyEngine

# Initialize detector
detector = MarketRegimeDetector()
detector.load_model()

# Get current regime
regime, info = detector.predict()

# Configure strategy engine
engine = MultiStrategyEngine()
engine.set_weights(info['strategy_allocation'])

# Execute trades
engine.run()
```

## API Reference

### MarketRegimeDetector

#### Methods

##### `__init__(n_regimes=4, n_iter=100, random_state=42, model_path=None)`
Initialize the detector.

##### `train(symbol='SPY', period='5y', interval='1d')`
Train the HMM model on historical data.

Returns: `dict` - Training metrics

##### `predict(data=None, return_confidence=True)`
Predict current market regime.

Returns: `tuple` - (regime_name, info_dict)

##### `predict_sequence(data, window_size=None)`
Predict regime sequence for time series.

Returns: `pd.DataFrame` - Regime predictions with confidence scores

##### `get_strategy_allocation(regime_name)`
Get strategy allocation for a regime.

Returns: `dict` - Strategy weights

##### `save_model(path=None)`
Save trained model to disk.

##### `load_model(path=None)`
Load trained model from disk.

##### `get_regime_statistics(data, regimes=None)`
Calculate performance statistics per regime.

Returns: `pd.DataFrame` - Regime statistics

##### `get_regime_transitions(results)`
Analyze regime transitions.

Returns: `pd.DataFrame` - Transition events

## Performance Considerations

- **Training Time**: ~30-60 seconds for 5 years of daily data
- **Prediction Time**: <1 second for current regime
- **Memory Usage**: ~50-100 MB for trained model
- **Data Requirements**: Minimum 200 trading days recommended

## Troubleshooting

### Common Issues

**ImportError: hmmlearn not found**
```bash
pip install hmmlearn
```

**Warning: No convergence**
- Increase `n_iter` parameter
- Try different random seed
- Check data quality

**Low silhouette score**
- Regimes may overlap naturally
- Consider different feature set
- Try different number of regimes

**Model file not found**
- Train model first: `python scripts/train_regime_detector.py`
- Check model path in config

## Advanced Usage

### Custom Features

```python
from ml.regime_detector import MarketRegimeDetector

class CustomRegimeDetector(MarketRegimeDetector):
    def extract_features(self, data):
        # Add custom features
        features = super().extract_features(data)
        features['custom_indicator'] = calculate_custom_indicator(data)
        return features

detector = CustomRegimeDetector()
detector.train()
```

### Different Regimes

```python
# Train with 3 regimes (bull, bear, neutral)
detector = MarketRegimeDetector(n_regimes=3)
metrics = detector.train()

# Train with 5 regimes
detector = MarketRegimeDetector(n_regimes=5)
metrics = detector.train()
```

### Different Assets

```python
# Train on QQQ instead of SPY
detector = MarketRegimeDetector()
metrics = detector.train(symbol="QQQ", period="10y")

# Train on bonds
detector = MarketRegimeDetector()
metrics = detector.train(symbol="TLT", period="5y")
```

## Testing

Run the demo to verify installation:

```bash
python examples/regime_detection_demo.py
```

This will:
1. Train or load a model
2. Make current prediction
3. Analyze historical regimes
4. Show regime statistics
5. Demonstrate strategy integration

## File Structure

```
rdt-trading-system/
├── ml/
│   ├── regime_detector.py          # Main detector class
│   ├── data/
│   │   └── visualizations/         # Generated plots
│   └── REGIME_DETECTION_README.md  # This file
├── scripts/
│   └── train_regime_detector.py    # Training script
├── examples/
│   └── regime_detection_demo.py    # Usage examples
├── models/
│   ├── regime_detector.pkl         # Trained model
│   └── training_metrics.json       # Training results
└── logs/
    └── regime_training_*.log       # Training logs
```

## References

- Hidden Markov Models: Rabiner, L. R. (1989). "A tutorial on hidden Markov models and selected applications in speech recognition"
- Market Regimes: Ang, A., & Bekaert, G. (2002). "Regime switches in interest rates"
- Feature Engineering: Technical Analysis from A to Z by Steven Achelis

## Contributing

To contribute improvements:
1. Add new features to `extract_features()`
2. Implement custom regime mapping logic
3. Add new validation metrics
4. Enhance visualization capabilities

## License

Part of the RDT Trading System. See main LICENSE file.

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review training metrics in `models/training_metrics.json`
3. Run demo script for debugging
4. Enable verbose logging with `--verbose` flag

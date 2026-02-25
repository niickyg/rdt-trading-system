# ML Ensemble Integration Guide

Quick guide for integrating the ML ensemble into the RDT Trading System.

## Quick Start

### 1. Install Dependencies

```bash
pip install xgboost scikit-learn tensorflow joblib
```

### 2. Train Your First Model

```python
from ml.ensemble import StackedEnsemble
from ml.feature_engineering import FeatureEngineer
import numpy as np
import pandas as pd

# Step 1: Prepare historical data
# Load your historical trades with outcomes
data = pd.read_csv('historical_trades.csv')

# Step 2: Extract features
engineer = FeatureEngineer()
features_list = []
labels = []

for _, trade in data.iterrows():
    signal = {
        'rrs': trade['rrs'],
        'atr': trade['atr'],
        'price': trade['entry_price'],
        'daily_strong': trade['daily_strong'],
        'daily_weak': trade['daily_weak'],
        'volume_ratio': trade['volume_ratio']
    }

    feat_dict = engineer.extract_features(signal)
    features_list.append(feat_dict['features'])

    # Label: 1 if reached 2R, 0 otherwise
    labels.append(1 if trade['max_profit'] >= 2.0 else 0)

X = np.array(features_list)
y = np.array(labels)

# Step 3: Train ensemble
ensemble = StackedEnsemble(
    use_xgboost=True,
    use_random_forest=True,
    use_lstm=False  # Disable if no sequence data
)

metrics = ensemble.train(
    X=X,
    y=y,
    feature_names=engineer.feature_names
)

print(f"Ensemble AUC: {metrics.auc:.4f}")
print(f"Model weights: {metrics.model_weights}")

# Step 4: Save model
ensemble.save('ml/data/production_ensemble')
```

### 3. Use in Trading Bot

```python
from ml.ensemble import StackedEnsemble
from ml.feature_engineering import FeatureEngineer

# Load trained model (once at startup)
ensemble = StackedEnsemble()
ensemble.load('ml/data/production_ensemble')
engineer = FeatureEngineer()

# In your signal evaluation function
def evaluate_signal(signal):
    """
    Evaluate a trading signal using ML ensemble

    Args:
        signal: Dict with signal data

    Returns:
        float: Probability of success (0-1)
    """
    # Extract features
    feat_dict = engineer.extract_features(signal)
    X = feat_dict['features'].reshape(1, -1)  # Shape: (1, n_features)

    # Predict
    probability = ensemble.predict_success_probability(X)[0]

    return probability

# Example usage in scanner
for signal in scanner.get_signals():
    ml_probability = evaluate_signal(signal)

    # Only take high-confidence trades
    if ml_probability >= 0.70:
        print(f"{signal['symbol']}: {ml_probability:.2%} success probability")
        # Execute trade...
```

## Integration Points

### A. Scanner Agent (`agents/scanner_agent.py`)

Add ML scoring to signal evaluation:

```python
from ml.ensemble import StackedEnsemble
from ml.feature_engineering import FeatureEngineer

class ScannerAgent:
    def __init__(self):
        # ... existing code ...
        self.ml_ensemble = StackedEnsemble()
        self.ml_ensemble.load('ml/data/production_ensemble')
        self.feature_engineer = FeatureEngineer()

    async def evaluate_signal(self, signal):
        # ... existing evaluation ...

        # Add ML score
        features = self.feature_engineer.extract_features(signal)
        X = features['features'].reshape(1, -1)
        ml_score = self.ml_ensemble.predict_success_probability(X)[0]

        signal['ml_score'] = ml_score
        signal['ml_confidence'] = 'HIGH' if ml_score > 0.7 else 'MEDIUM' if ml_score > 0.5 else 'LOW'

        return signal
```

### B. Risk Manager (`risk/risk_manager.py`)

Adjust position sizing based on ML confidence:

```python
class RiskManager:
    def __init__(self):
        # ... existing code ...
        self.ml_ensemble = StackedEnsemble()
        self.ml_ensemble.load('ml/data/production_ensemble')

    def calculate_position_size(self, signal, account_value):
        base_size = self._base_position_size(signal, account_value)

        # Adjust based on ML confidence
        ml_prob = signal.get('ml_score', 0.5)

        if ml_prob >= 0.80:
            multiplier = 1.5  # Increase size for very high confidence
        elif ml_prob >= 0.70:
            multiplier = 1.2  # Slightly increase
        elif ml_prob >= 0.60:
            multiplier = 1.0  # Standard size
        elif ml_prob >= 0.50:
            multiplier = 0.7  # Reduce size
        else:
            multiplier = 0.5  # Minimum size or skip

        adjusted_size = base_size * multiplier

        return min(adjusted_size, self.max_position_size)
```

### C. Strategy (`strategies/base_strategy.py`)

Filter trades by ML score:

```python
class BaseStrategy:
    def __init__(self):
        # ... existing code ...
        self.ml_threshold = 0.65  # Minimum ML score to take trade

    def should_enter_trade(self, signal):
        # Existing checks
        if not self._passes_technical_checks(signal):
            return False

        # ML check
        ml_score = signal.get('ml_score', 0)
        if ml_score < self.ml_threshold:
            logger.info(f"ML score too low: {ml_score:.2%}")
            return False

        return True
```

### D. Learning Agent (`agents/learning_agent.py`)

Add ML predictions to trade analysis:

```python
class LearningAgent:
    def analyze_trade_outcome(self, trade):
        # ... existing analysis ...

        # Compare ML prediction vs actual
        ml_predicted_success = trade.get('ml_score', 0) > 0.5
        actual_success = trade['profit_loss_r'] >= 2.0

        # Track accuracy
        self.ml_accuracy_tracker.update(ml_predicted_success == actual_success)

        # If ML was wrong, analyze why
        if ml_predicted_success != actual_success:
            self._analyze_ml_error(trade)
```

## Training Data Collection

### Collect Historical Outcomes

```python
import pandas as pd
from datetime import datetime

class TradeDataCollector:
    """Collect trade data for ML training"""

    def __init__(self, db_path='data/trades.db'):
        self.db_path = db_path

    def record_trade_entry(self, signal):
        """Record when trade is entered"""
        trade_record = {
            'symbol': signal['symbol'],
            'entry_date': datetime.now(),
            'entry_price': signal['price'],
            'rrs': signal['rrs'],
            'atr_percent': signal['atr'] / signal['price'] * 100,
            # ... all features ...
            'target_2r': signal['price'] + (2 * signal['stop_distance'])
        }
        # Save to database
        self._save_to_db(trade_record)

    def update_trade_outcome(self, symbol, entry_date):
        """Update trade outcome after 10 days"""
        # Fetch price history
        prices = self._get_price_history(symbol, entry_date, days=10)
        max_price = max(prices)

        # Check if reached 2R
        reached_2r = max_price >= trade_record['target_2r']

        # Update database
        self._update_db(symbol, entry_date, {
            'reached_2r': reached_2r,
            'max_profit_r': (max_price - entry_price) / stop_distance,
            'days_to_2r': self._days_to_target(prices, target_2r)
        })

    def export_training_data(self, output_path='ml/data/training_data.csv'):
        """Export all trades for model training"""
        df = self._load_from_db()
        df.to_csv(output_path, index=False)
        return df
```

### Automated Retraining

```python
from apscheduler.schedulers.background import BackgroundScheduler

def retrain_models():
    """Automated model retraining (run monthly)"""
    logger.info("Starting monthly model retraining...")

    # Load latest data
    collector = TradeDataCollector()
    df = collector.export_training_data()

    # Prepare features and labels
    X, y = prepare_training_data(df)

    # Train new model
    ensemble = StackedEnsemble()
    metrics = ensemble.train(X, y)

    # Save with timestamp
    timestamp = datetime.now().strftime('%Y%m%d')
    ensemble.save(f'ml/data/ensemble_{timestamp}')

    # A/B test: Compare with production model
    if metrics.auc > production_model_auc:
        logger.info(f"New model better! AUC: {metrics.auc:.4f}")
        ensemble.save('ml/data/production_ensemble')
    else:
        logger.info("Keeping existing model")

# Schedule monthly retraining
scheduler = BackgroundScheduler()
scheduler.add_job(retrain_models, 'cron', day=1, hour=2)  # 1st of month at 2am
scheduler.start()
```

## Performance Monitoring

### Track ML Predictions

```python
class MLMonitor:
    """Monitor ML model performance in production"""

    def __init__(self):
        self.predictions = []
        self.outcomes = []

    def record_prediction(self, symbol, probability):
        """Record ML prediction"""
        self.predictions.append({
            'symbol': symbol,
            'timestamp': datetime.now(),
            'probability': probability,
            'predicted_success': probability > 0.5
        })

    def record_outcome(self, symbol, reached_2r):
        """Record actual outcome"""
        # Find matching prediction
        pred = [p for p in self.predictions if p['symbol'] == symbol][0]

        self.outcomes.append({
            'symbol': symbol,
            'predicted': pred['predicted_success'],
            'actual': reached_2r,
            'correct': pred['predicted_success'] == reached_2r
        })

    def get_accuracy(self, days=30):
        """Get accuracy over last N days"""
        recent = [o for o in self.outcomes
                 if (datetime.now() - o['timestamp']).days <= days]

        if not recent:
            return 0.0

        correct = sum(1 for o in recent if o['correct'])
        return correct / len(recent)

    def get_calibration(self):
        """Check if probabilities are well-calibrated"""
        # Group predictions by probability bins
        bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        for i in range(len(bins)-1):
            low, high = bins[i], bins[i+1]
            bin_preds = [o for o in self.outcomes
                        if low <= o['probability'] < high]

            if bin_preds:
                actual_rate = sum(o['actual'] for o in bin_preds) / len(bin_preds)
                expected_rate = (low + high) / 2

                logger.info(f"Bin {low:.1f}-{high:.1f}: "
                          f"Expected {expected_rate:.2%}, "
                          f"Actual {actual_rate:.2%}")
```

## Troubleshooting

### Common Issues

**1. Model returns all same predictions**
- Not enough training data (need >500 samples)
- Features not diverse enough
- Check class balance in training data

**2. Poor accuracy on new data**
- Model overfitting (reduce complexity)
- Data drift (retrain with recent data)
- Features changed (verify feature extraction)

**3. Slow predictions**
- Batch predictions instead of one-by-one
- Disable LSTM if not needed
- Cache model in memory

**4. Import errors**
- Install dependencies: `pip install xgboost scikit-learn tensorflow`
- Check Python version (3.8+ required)

### Debugging

```python
# Enable debug logging
from loguru import logger
logger.add("ml_debug.log", level="DEBUG")

# Check model state
summary = ensemble.get_metrics_summary()
print(summary)

# Verify predictions
X_test = np.random.randn(5, 15)
probas = ensemble.predict_success_probability(X_test)
base_preds = ensemble.get_base_model_predictions(X_test)

print("Ensemble:", probas)
print("Base models:", base_preds)
```

## Best Practices

1. **Start Simple**: Use XGBoost + Random Forest only
2. **Collect Data**: Run scanner for 1-2 months to gather training data
3. **Validate Properly**: Use time-series validation, never random splits
4. **Monitor Performance**: Track accuracy and calibration weekly
5. **Retrain Regularly**: Monthly retraining with new data
6. **A/B Test**: Always compare new models vs production
7. **Set Thresholds**: Use 70%+ probability for high confidence trades
8. **Adjust Position Size**: Scale positions by ML confidence
9. **Feature Quality**: Ensure features are consistently calculated
10. **Version Control**: Save models with timestamps

## Production Checklist

- [ ] Install all dependencies
- [ ] Collect at least 500 historical trades with outcomes
- [ ] Train initial model and validate metrics
- [ ] Save production model
- [ ] Integrate into scanner agent
- [ ] Integrate into risk manager
- [ ] Add trade outcome tracking
- [ ] Set up monitoring dashboard
- [ ] Schedule monthly retraining
- [ ] Document threshold settings
- [ ] Test on paper trading first

## Support

For detailed documentation, see:
- `ml/README.md` - Full technical documentation
- `ml/ENSEMBLE_SUMMARY.md` - Implementation summary
- `examples/ensemble_example.py` - Code examples

The ML ensemble is now ready for integration into your trading system!

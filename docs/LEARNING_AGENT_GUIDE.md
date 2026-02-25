# Learning Agent Guide

## Overview

The **Learning Agent** is a critical component of the RDT Trading System that continuously improves trading performance through machine learning. It implements a feedback loop that:

1. Tracks all trading signals and their features
2. Labels outcomes based on position results (2R achievement)
3. Accumulates labeled training data
4. Periodically retrains the ML model
5. Validates new models before deployment
6. Publishes model update events

## Architecture

### Key Components

#### 1. SignalFeatures (Data Structure)
Captures all relevant features from a trading signal:
- **Technical**: RRS, ATR, ATR%, price
- **Daily Chart**: EMA alignment, strength/weakness
- **Market Context**: SPY movement, stock movement, volume
- **Timestamp**: When the signal occurred

#### 2. LabeledSignal (Data Structure)
Pairs a signal with its outcome:
- **Features**: The original signal features
- **Label**: 1 for success (≥2R), 0 for failure
- **Metrics**: Actual P&L, R-multiple achieved
- **Metadata**: Position ID, exit reason, timestamps

#### 3. SimpleTradingModel
A lightweight ML model for signal classification:
- Uses logistic regression with learned weights
- Takes feature vectors as input
- Outputs probability of success (0-1)
- Can be easily saved/loaded from disk

#### 4. LearningAgent (Main Class)
Orchestrates the entire learning process:
- Subscribes to relevant events
- Manages data collection
- Triggers retraining
- Validates models
- Deploys improvements

## Event Flow

```
SIGNAL_FOUND
    ↓
[Extract & Store Features]
    ↓
POSITION_OPENED
    ↓
[Link Signal to Position]
    ↓
POSITION_CLOSED
    ↓
[Calculate Outcome & Label]
    ↓
[Check Retrain Triggers]
    ↓
[Retrain Model if Needed]
    ↓
[Validate New Model]
    ↓
[Deploy if Better]
    ↓
SYSTEM_MODEL_UPDATED (Event)
```

## Configuration

### Initialization Parameters

```python
learning_agent = LearningAgent(
    model_dir="models",              # Where to save models
    retrain_threshold=100,           # Retrain after N trades
    retrain_interval_days=7,         # Or weekly (whichever first)
    lookback_months=6,               # Use last 6 months of data
    min_improvement=0.02,            # Require 2% improvement to deploy
    event_bus=event_bus,             # Event bus instance
    config={}                        # Additional config
)
```

### Retrain Triggers

The agent retrains when **either** condition is met:

1. **Data Threshold**: Accumulated N new labeled trades
2. **Time Threshold**: X days since last retrain

This ensures:
- Frequent updates with active trading
- Regular updates even with low activity

## Success Criteria

A trade is labeled as **SUCCESS** (1) if:
- **R-multiple ≥ 2.0**

A trade is labeled as **FAILURE** (0) if:
- **R-multiple < 2.0**

**R-multiple calculation**:
```
For LONG:  R = (exit_price - entry_price) / ATR
For SHORT: R = (entry_price - exit_price) / ATR
```

## Model Deployment Logic

A new model is deployed **only if**:

1. **No current model exists** AND **new accuracy > 50%**
   - Bootstrap the system with a reasonable model

2. **Improvement threshold met**: `new_F1 - current_F1 >= min_improvement`
   - Prevents deploying marginally better models
   - Default: 2% improvement required

3. **Validation passes**: Model performs well on held-out data

When deployed:
- Old model saved as backup with timestamp
- New model becomes current
- Metrics saved to disk
- `SYSTEM_MODEL_UPDATED` event published

## Usage Examples

### Basic Integration

```python
from agents.learning_agent import LearningAgent
from agents.events import get_event_bus

# Initialize
event_bus = get_event_bus()
await event_bus.start()

learning_agent = LearningAgent(
    event_bus=event_bus,
    retrain_threshold=100
)

# Start the agent
await learning_agent.start()

# The agent now automatically:
# - Tracks signals
# - Labels outcomes
# - Retrains periodically
# - Deploys better models

# ... trading system runs ...

# Get statistics
stats = learning_agent.get_learning_stats()
print(f"Success Rate: {stats['success_rate']:.1%}")
print(f"Model Accuracy: {stats['model_info']['accuracy']:.1%}")

# Predict signal quality
quality = learning_agent.predict_signal_quality({
    "symbol": "AAPL",
    "direction": "long",
    "rrs": 3.5,
    "price": 180.0,
    "atr": 3.5,
    "daily_strong": True,
    "daily_weak": False,
    "volume": 80000000
})
print(f"Predicted success probability: {quality:.1%}")

# Stop the agent
await learning_agent.stop()
```

### Manual Retrain Trigger

```python
# Force immediate retraining (useful for testing)
learning_agent.force_retrain()
```

### Subscribe to Model Updates

```python
from agents.events import EventType, subscribe

async def on_model_updated(event):
    data = event.data
    print(f"New model deployed!")
    print(f"  Version: {data['model_version']}")
    print(f"  Accuracy: {data['accuracy']:.1%}")
    print(f"  F1 Score: {data['f1_score']:.3f}")

subscribe(EventType.SYSTEM_MODEL_UPDATED, on_model_updated)
```

## Metrics Tracked

### Learning Metrics
- `signals_tracked`: Total signals captured
- `outcomes_labeled`: Total outcomes labeled
- `success_rate`: Overall success rate (%)
- `labeled_data_size`: Size of training dataset
- `new_labels_since_retrain`: Pending labels

### Training Metrics
- `total_retrains`: Number of retraining cycles
- `total_deployments`: Number of model deployments
- `last_retrain`: Timestamp of last retrain

### Model Metrics
- `accuracy`: Overall prediction accuracy
- `precision`: Positive predictive value
- `recall`: True positive rate
- `f1_score`: Harmonic mean of precision/recall
- `training_samples`: Number of training samples
- `validation_samples`: Number of validation samples

## Database Integration

The Learning Agent integrates with the system database:

### Saves to Database:
- **Signals table**: All signals with features
- Uses existing Trade/Position tables for historical data

### Reads from Database:
- Historical trades for initial training
- Signal-trade associations for feature extraction

### Schema Requirements:
```sql
-- Signals are saved with:
- symbol
- timestamp
- rrs
- direction
- price
- atr
- daily_strong
- daily_weak
```

## File System Structure

```
models/
├── current_model.pkl          # Active model
├── current_metrics.json       # Active model metrics
├── model_backup_*.pkl         # Previous model versions
└── labeled_data.pkl          # Backup of labeled data

~/.rdt-trading/models/         # Default location
```

## Advanced Features

### Feature Engineering

Features extracted from each signal:
1. **RRS**: Relative strength value
2. **ATR %**: Volatility as % of price
3. **Daily Strong**: Boolean (1/0)
4. **Daily Weak**: Boolean (1/0)
5. **Direction**: Long=1, Short=-1
6. **SPY % Change**: Market movement
7. **Stock % Change**: Individual movement
8. **Volume**: Log-transformed for normalization

### Model Validation

Uses train/test split:
- **80%** of data for training
- **20%** of data for validation

Calculates comprehensive metrics:
- Confusion matrix (TP, FP, TN, FN)
- Accuracy, Precision, Recall, F1
- ROC-AUC approximation

### Historical Data Loading

On startup, loads:
- Last 6 months of closed trades
- Associated signal data
- Reconstructs labeled samples
- Enables immediate model training

## Production Considerations

### Error Handling

All event handlers wrapped in try/except:
- Errors logged but don't crash agent
- Metrics track error counts
- Failed signals skipped gracefully

### Performance

- Lightweight model training (< 1 second)
- Asynchronous event handling
- Minimal memory footprint
- Periodic cleanup of old data

### Monitoring

Monitor these key indicators:
- **Success rate trending**: Should improve over time
- **Model accuracy**: Should be > 60% for effectiveness
- **Retrain frequency**: Ensure happening regularly
- **Label accumulation**: Confirm outcomes being captured

### Scaling Considerations

For high-volume trading:
- Increase `retrain_threshold` (e.g., 500 trades)
- Reduce `lookback_months` (e.g., 3 months)
- Consider batching label processing
- Use more sophisticated models (XGBoost, LightGBM)

## Integration with Other Agents

### AnalyzerAgent Integration

The AnalyzerAgent can use predictions:

```python
# In AnalyzerAgent
quality_score = learning_agent.predict_signal_quality(signal)

if quality_score < 0.60:  # 60% threshold
    rejection_reasons.append(f"ML quality too low: {quality_score:.1%}")
```

### Risk Agent Integration

Adjust position sizing based on model confidence:

```python
# Higher confidence = larger position (within limits)
quality = learning_agent.predict_signal_quality(signal)
position_multiplier = 0.5 + (quality * 0.5)  # 0.5x to 1.0x
shares = base_shares * position_multiplier
```

## Troubleshooting

### No Model Deployed
**Issue**: Model never gets deployed
**Solutions**:
- Check if `labeled_data_size >= 20`
- Verify events are being received (check logs)
- Ensure database connection working
- Try `force_retrain()`

### Low Accuracy
**Issue**: Model accuracy < 50%
**Solutions**:
- Need more training data (wait for more trades)
- Features may need tuning
- Success criteria may be too strict
- Market conditions may have changed

### Not Retraining
**Issue**: Agent not retraining automatically
**Solutions**:
- Verify `new_labels_since_retrain` incrementing
- Check retrain thresholds are reasonable
- Ensure positions are closing (providing labels)
- Look for errors in logs

### Database Errors
**Issue**: Cannot save/load data
**Solutions**:
- Verify database initialized (`init_database()`)
- Check database path/permissions
- Ensure tables exist
- Review database logs

## Future Enhancements

Potential improvements:
1. **Advanced Models**: XGBoost, LightGBM, Neural Networks
2. **Feature Selection**: Automated feature importance analysis
3. **Ensemble Methods**: Combine multiple model types
4. **Online Learning**: Update model incrementally
5. **A/B Testing**: Run multiple models in parallel
6. **Backtesting**: Validate on historical data before deployment
7. **Explainability**: SHAP values for feature importance
8. **Drift Detection**: Alert when model performance degrades

## API Reference

### Main Methods

#### `predict_signal_quality(signal_data: Dict) -> float`
Predict probability of signal success.

**Returns**: Probability (0.0 to 1.0)

#### `get_learning_stats() -> Dict`
Get comprehensive learning statistics.

**Returns**: Dictionary with all metrics

#### `get_model_info() -> Dict`
Get current model information.

**Returns**: Model version, accuracy, training info

#### `force_retrain()`
Manually trigger immediate model retraining.

### Events Subscribed

- `SIGNAL_FOUND`: Capture signal features
- `POSITION_OPENED`: Link signal to position
- `POSITION_CLOSED`: Label outcome

### Events Published

- `SYSTEM_MODEL_UPDATED`: When new model deployed
  - Contains: version, accuracy, precision, recall, F1, sample counts

## Testing

Run the test suite:

```bash
cd /home/user0/rdt-trading-system
python examples/test_learning_agent.py
```

This will:
1. Simulate multiple trading scenarios
2. Test signal tracking and labeling
3. Trigger model retraining
4. Validate predictions
5. Display comprehensive statistics

## Conclusion

The Learning Agent provides a production-ready feedback loop for continuous system improvement. It automatically:
- Learns from every trade
- Adapts to market conditions
- Improves prediction accuracy
- Deploys better models safely

By labeling outcomes based on 2R achievement and retraining periodically, the system becomes progressively better at identifying high-quality trading opportunities.

# ML Ensemble System for Trade Success Prediction

A sophisticated machine learning ensemble that combines XGBoost, Random Forest, and LSTM models using a Logistic Regression meta-learner to predict the probability of trade success.

## Overview

The ensemble predicts whether a trade will reach its 2R (2x risk) target within 10 days based on various technical and market features.

### Architecture

```
Input Features
    │
    ├─────────────┬─────────────┬─────────────┐
    ▼             ▼             ▼             ▼
XGBoost    Random Forest    LSTM    Meta-Learner
(Tree-based)  (Bagging)   (Deep Learning) (Logistic Regression)
    │             │             │             │
    └─────────────┴─────────────┴─────────────┘
                      ▼
            Stacked Predictions
                      ▼
              Final Probability
```

## Models

### 1. XGBoost Classifier (`xgboost_model.py`)

**Purpose**: Gradient boosted decision trees for high-accuracy predictions

**Features**:
- Time-series cross-validation
- GPU acceleration support
- Feature importance tracking
- Binary classification (success/failure)

**Key Parameters**:
- `n_estimators`: Number of boosting rounds (default: 100)
- `max_depth`: Maximum tree depth (default: 6)
- `learning_rate`: Boosting learning rate (default: 0.1)
- `use_gpu`: Enable GPU acceleration (default: False)

**Metrics**:
- Accuracy, AUC, Precision, Recall, F1 Score
- Confusion Matrix
- Feature Importance

### 2. Random Forest Classifier (`random_forest_model.py`)

**Purpose**: Bagging ensemble for algorithm diversity and robustness

**Features**:
- Out-of-bag (OOB) score estimation
- Balanced class weights
- Tree statistics tracking
- Feature importance via Gini index

**Key Parameters**:
- `n_estimators`: Number of trees (default: 100)
- `max_depth`: Maximum tree depth (default: 15)
- `min_samples_split`: Minimum samples to split (default: 5)
- `oob_score`: Use out-of-bag samples (default: True)

**Metrics**:
- Same as XGBoost
- OOB Score for unbiased validation
- Tree depth and leaf statistics

### 3. LSTM Time-Series Model (`lstm_model.py`)

**Purpose**: Deep learning sequence prediction for temporal patterns

**Features**:
- TensorFlow/Keras implementation
- Sequence-based prediction (last N bars)
- 3-class output (up/neutral/down)
- StandardScaler normalization
- Early stopping and learning rate reduction

**Key Parameters**:
- `sequence_length`: Time steps in sequence (default: 20)
- `lstm_units`: LSTM layer sizes (default: [64, 32])
- `dropout`: Dropout rate (default: 0.2)
- `learning_rate`: Adam optimizer LR (default: 0.001)

**Metrics**:
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix (3-class)
- Training history (loss curves)

### 4. Stacked Ensemble (`ensemble.py`)

**Purpose**: Meta-learner that combines all base models

**Features**:
- Logistic Regression meta-learner
- Automatic model weight optimization
- Cross-validation for meta-learner
- Individual and ensemble metrics tracking

**Key Parameters**:
- `use_xgboost`: Include XGBoost (default: True)
- `use_random_forest`: Include Random Forest (default: True)
- `use_lstm`: Include LSTM (default: False)
- `meta_learner_C`: Regularization strength (default: 1.0)

**Metrics**:
- All ensemble metrics (accuracy, AUC, etc.)
- Log loss
- Model weights (learned importance)
- Base model performance comparison

## Installation

### Required Dependencies

```bash
# Core ML libraries
pip install scikit-learn>=1.3.0
pip install xgboost>=2.0.0
pip install tensorflow>=2.14.0

# Supporting libraries
pip install numpy>=1.24.0
pip install pandas>=2.1.0
pip install joblib>=1.3.0
pip install loguru>=0.7.0
```

### Optional: GPU Support

For XGBoost GPU acceleration:
```bash
# NVIDIA GPU with CUDA support
pip install xgboost[gpu]
```

For TensorFlow GPU:
```bash
pip install tensorflow[and-cuda]
```

## Usage

### Quick Start

```python
from ml.ensemble import StackedEnsemble
import numpy as np

# Prepare your data
X = np.array([...])  # Features (n_samples, n_features)
y = np.array([...])  # Binary labels (0=failure, 1=success)

feature_names = [
    'rrs', 'atr_percent', 'price_momentum_5d',
    'volume_ratio', 'trend_strength', ...
]

# Create and train ensemble
ensemble = StackedEnsemble(
    use_xgboost=True,
    use_random_forest=True,
    use_lstm=False
)

metrics = ensemble.train(
    X=X,
    y=y,
    feature_names=feature_names
)

# Make predictions
X_new = np.array([...])  # New samples
probabilities = ensemble.predict_success_probability(X_new)

# Get predictions
for i, prob in enumerate(probabilities):
    print(f"Trade {i}: {prob:.2%} success probability")
```

### Training Individual Models

```python
from ml.models.xgboost_model import XGBoostTradeClassifier

# Train XGBoost
xgb = XGBoostTradeClassifier(
    n_estimators=100,
    max_depth=6,
    use_gpu=True
)
metrics = xgb.train(X, y, feature_names=feature_names)

# Get top features
top_features = xgb.get_top_features(10)
for feature, importance in top_features:
    print(f"{feature}: {importance:.4f}")
```

### Using LSTM for Sequence Prediction

```python
from ml.models.lstm_model import LSTMTradePredictor

# Prepare sequential data
X_sequences = np.array([...])  # Shape: (n_samples, 20, n_features)
y_direction = np.array([...])  # Class labels: 0=up, 1=neutral, 2=down

# Train LSTM
lstm = LSTMTradePredictor(
    sequence_length=20,
    n_features=10,
    lstm_units=[64, 32]
)
metrics = lstm.train(X_sequences, y_direction, epochs=50)

# Predict directions
directions = lstm.predict_direction(X_sequences)
print(directions)  # ['up', 'down', 'neutral', ...]
```

### Model Persistence

```python
# Save ensemble
ensemble.save('/path/to/models/ensemble')

# Load ensemble
new_ensemble = StackedEnsemble()
new_ensemble.load('/path/to/models/ensemble')

# Make predictions
probabilities = new_ensemble.predict_success_probability(X_new)
```

## Feature Engineering

### Expected Features

The models expect the following types of features:

1. **Relative Strength** (`rrs`): Relative strength indicator value
2. **Volatility** (`atr_percent`): ATR as percentage of price
3. **Momentum** (`price_momentum_*d`): Price change over N days
4. **Volume** (`volume_ratio`): Current vs average volume
5. **Trend** (`trend_strength`): Trend quality score
6. **Technical Indicators**: RSI, MACD, Bollinger position
7. **Market Context**: SPY trend, sector strength, market volatility
8. **Price Levels**: Distance to SMA20, SMA50

### Example Feature Extraction

```python
from ml.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()

# Extract features from signal
signal = {
    'rrs': 3.5,
    'atr': 2.5,
    'price': 100.0,
    'daily_strong': True,
    'volume_ratio': 1.8
}

features = engineer.extract_features(signal)
print(features['features'])  # NumPy array
print(features['feature_names'])  # Feature names
print(features['top_features'])  # Most important for this signal
```

## Performance Metrics

### Validation Metrics

All models track comprehensive metrics:

- **Accuracy**: Percentage of correct predictions
- **AUC**: Area under ROC curve (probability ranking quality)
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Breakdown of predictions vs actuals

### Ensemble-Specific Metrics

- **Log Loss**: Probability calibration quality
- **Model Weights**: Learned importance of each base model
- **Base Model Performance**: Individual model metrics for comparison

## Examples

See `examples/ensemble_example.py` for comprehensive examples:

1. Training individual models
2. Training stacked ensemble
3. Making predictions
4. Analyzing feature importance
5. Saving and loading models
6. Getting comprehensive metrics

Run examples:
```bash
cd /home/user0/rdt-trading-system
python examples/ensemble_example.py
```

## File Structure

```
ml/
├── __init__.py
├── README.md (this file)
├── ensemble.py              # Stacked ensemble + meta-learner
├── feature_engineering.py   # Feature extraction
├── regime_detector.py       # Market regime detection
├── models/
│   ├── __init__.py
│   ├── xgboost_model.py    # XGBoost classifier
│   ├── random_forest_model.py  # Random Forest classifier
│   └── lstm_model.py       # LSTM time-series model
├── data/                   # Saved models and training data
└── examples/
    └── ensemble_example.py # Usage examples
```

## Best Practices

### 1. Data Preparation

- **Time-series order**: Always maintain chronological order
- **Feature scaling**: LSTM uses StandardScaler automatically
- **Class balance**: Models handle imbalance, but check distribution
- **Validation split**: Use time-based splits (not random)

### 2. Model Training

- **Cross-validation**: Use time-series CV for proper evaluation
- **Hyperparameters**: Start with defaults, tune if needed
- **Early stopping**: LSTM uses automatic early stopping
- **GPU usage**: Enable for XGBoost if available

### 3. Production Deployment

- **Model versioning**: Save models with timestamps
- **Monitoring**: Track prediction distribution drift
- **Retraining**: Retrain periodically with new data
- **A/B testing**: Compare new models against production

### 4. Feature Importance

- **Regular analysis**: Check feature importance after training
- **Feature stability**: Monitor changes in importance over time
- **Feature selection**: Remove low-importance features
- **Domain knowledge**: Validate that important features make sense

## Performance Tips

### Training Speed

1. **XGBoost**: Use GPU acceleration if available
2. **Random Forest**: Increase `n_jobs` for parallel training
3. **LSTM**: Use smaller batch sizes for faster convergence
4. **Ensemble**: Train base models in parallel (future enhancement)

### Memory Usage

1. **Large datasets**: Use batch processing for LSTM
2. **Feature selection**: Remove redundant features
3. **Model pruning**: Use smaller architectures if needed
4. **Data types**: Use float32 instead of float64 where possible

### Prediction Speed

1. **Batch predictions**: Predict multiple samples at once
2. **Model caching**: Load models once, reuse for predictions
3. **Feature caching**: Pre-compute features when possible
4. **LSTM optional**: Disable LSTM in production if too slow

## Troubleshooting

### Common Issues

**ImportError: No module named 'xgboost'**
```bash
pip install xgboost scikit-learn tensorflow
```

**GPU not detected**
```python
# Check XGBoost GPU support
import xgboost as xgb
print(xgb.dask.predict(..., tree_method='gpu_hist'))
```

**LSTM training too slow**
- Reduce `sequence_length`
- Reduce `lstm_units` sizes
- Use fewer epochs
- Enable GPU for TensorFlow

**Poor ensemble performance**
- Check base model performance first
- Ensure sufficient training data (>500 samples)
- Verify feature quality and relevance
- Check for data leakage

### Debugging

Enable detailed logging:
```python
from loguru import logger
logger.add("ml_training.log", level="DEBUG")
```

Check model metrics:
```python
summary = ensemble.get_metrics_summary()
print(summary)
```

## Contributing

When adding new models:

1. Inherit from base model pattern
2. Implement `train()`, `predict()`, `predict_proba()` methods
3. Track metrics using dataclass
4. Add save/load functionality with joblib or h5
5. Update ensemble to integrate new model

## License

Part of the RDT Trading System. See main README for license information.

## Support

For issues or questions:
- Check examples in `examples/ensemble_example.py`
- Review this README thoroughly
- Examine model source code for implementation details
- Check logs for detailed error messages

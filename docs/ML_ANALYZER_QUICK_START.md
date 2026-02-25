# ML-Enhanced Analyzer Agent - Quick Start Guide

## Overview

The Analyzer Agent now includes ML capabilities that improve signal filtering by analyzing trade probability, market regime, and feature importance.

## Key Changes

### 1. Enhanced Output

All approved setups now include ML metadata:

```python
{
    # Original signal data
    "symbol": "AAPL",
    "direction": "long",
    "price": 175.50,
    "rrs": 3.8,

    # Position sizing
    "entry_price": 175.50,
    "stop_price": 172.25,
    "target_price": 185.00,
    "shares": 100,

    # NEW: ML enhancements
    "ml_probability": 88.0,              # ML prediction confidence (0-100%)
    "confidence": 96.7,                  # Regime-adjusted confidence
    "market_regime": "bull_trending",    # Current market state
    "regime_confidence": 0.80,           # Regime detection confidence
    "top_features": [                    # Key decision factors
        "rrs",
        "daily_strong",
        "trend_strength"
    ]
}
```

### 2. New Rejection Criteria

Signals can now be rejected for ML reasons:

- **ML probability too low**: < 65%
- **Overall confidence too low**: < 70%
- **ML analysis error**: Feature extraction or prediction failure

### 3. Market Regime Awareness

The analyzer adjusts confidence based on market conditions:

| Regime | Description | Confidence Multiplier |
|--------|-------------|---------------------|
| `bull_trending` | Strong upward momentum | 1.1x (boost) |
| `bear_trending` | Strong downward momentum | 0.9x (reduce) |
| `high_volatility` | Elevated uncertainty | 0.85x (reduce) |
| `low_volatility` | Calm market | 1.05x (slight boost) |

## Configuration

### Adjust ML Thresholds

```python
analyzer = AnalyzerAgent(risk_manager=risk_manager)

# Set ML probability threshold (default: 65%)
analyzer.min_ml_probability = 70.0  # More conservative

# Set overall confidence threshold (default: 70%)
analyzer.min_overall_confidence = 75.0  # More conservative

# Traditional thresholds also available
analyzer.set_min_rrs(2.5)  # Increase RRS requirement
analyzer.set_min_rr_ratio(2.5)  # Increase R/R requirement
```

### Disable ML (Fallback Mode)

If ML libraries are not installed, the analyzer automatically falls back to rule-based analysis:

- Feature extraction uses simple heuristics
- Regime detection uses signal-based logic
- All functionality remains available

## Understanding ML Analysis

### ML Probability

The Ensemble model combines multiple factors:

- **RRS Strength** (30% weight): How strong is the relative strength?
- **Trend Alignment** (25% weight): Do timeframes agree?
- **Volatility** (20% weight): Is ATR in optimal range?
- **Technical Setup** (25% weight): Quality of technical patterns

A score of 80%+ indicates high confidence in the signal.

### Overall Confidence

Combines ML probability with market regime:

```
Overall Confidence = ML Probability × Regime Multiplier
```

Example:
- ML Probability: 80%
- Market Regime: bull_trending (1.1x)
- Overall Confidence: 88%

### Top Features

Shows which factors most influenced the decision:

```python
"top_features": ["rrs", "daily_strong", "volume_ratio"]
```

This means:
- Strong RRS was a key factor
- Daily chart alignment was important
- Above-average volume supported the signal

## Monitoring ML Performance

Track ML metrics:

```python
# Get approval rate
approval_rate = analyzer.metrics.custom_metrics["approval_rate"]

# Check if ML is enabled
ml_enabled = analyzer.metrics.custom_metrics["ml_enabled"]

# View analysis counts
signals_analyzed = analyzer.metrics.custom_metrics["signals_analyzed"]
setups_approved = analyzer.metrics.custom_metrics["setups_approved"]
setups_rejected = analyzer.metrics.custom_metrics["setups_rejected"]
```

## Example Analysis Flow

### High-Quality Signal

```
Input Signal:
  Symbol: TSLA
  Direction: short
  Price: $245.75
  RRS: -3.20
  Daily Weak: True
  ATR: $8.30

ML Analysis:
  Features Extracted: [rrs=-3.20, atr_percent=3.4, ...]
  ML Probability: 80.0%
  Market Regime: bull_trending (confidence: 0.80)
  Overall Confidence: 88.1%
  Top Features: ["rrs", "daily_weak", "trend_strength"]

Result: ✓ APPROVED
  40 shares @ $245.75
  Stop: $258.20
  Target: $220.85
```

### Low-Quality Signal

```
Input Signal:
  Symbol: MSFT
  Direction: long
  Price: $380.25
  RRS: 1.50
  Daily Strong: False
  ATR: $5.50

ML Analysis:
  Features Extracted: [rrs=1.50, atr_percent=1.4, ...]
  ML Probability: 56.2%
  Market Regime: bull_trending (confidence: 0.80)
  Overall Confidence: 61.9%

Result: ✗ REJECTED
  Reasons:
    - RRS too weak: 1.50 < 2.0
    - Daily chart not strong for long
    - ML probability too low: 56.2% < 65.0%
    - Overall confidence too low: 61.9% < 70.0%
```

## Troubleshooting

### ML Analysis Errors

If you see "ML analysis error" in rejections:

1. Check feature extraction:
   ```python
   signal = {...}
   features = analyzer.feature_engineer.extract_features(signal)
   print(features)
   ```

2. Check regime detection:
   ```python
   regime, confidence = analyzer.regime_detector.detect_regime(signal)
   print(f"Regime: {regime}, Confidence: {confidence}")
   ```

3. Check ensemble prediction:
   ```python
   probability = analyzer.ensemble.predict(features['features'], signal)
   print(f"ML Probability: {probability}%")
   ```

### Low Approval Rates

If approval rate is too low:

1. Lower ML thresholds:
   ```python
   analyzer.min_ml_probability = 60.0  # From 65%
   analyzer.min_overall_confidence = 65.0  # From 70%
   ```

2. Check if regime multipliers are too conservative:
   - High volatility regime reduces confidence significantly
   - Bear regime also reduces confidence

3. Review rejected signals to understand patterns:
   - Are ML probabilities consistently low?
   - Is one feature causing most rejections?

### High Approval Rates

If approval rate is too high:

1. Increase ML thresholds:
   ```python
   analyzer.min_ml_probability = 70.0  # From 65%
   analyzer.min_overall_confidence = 75.0  # From 70%
   ```

2. Tighten traditional criteria:
   ```python
   analyzer.set_min_rrs(2.5)  # From 2.0
   analyzer.set_min_rr_ratio(2.5)  # From 2.0
   ```

## Best Practices

1. **Start Conservative**: Begin with higher thresholds (70%+ ML probability)

2. **Monitor Regime Changes**: Adjust strategy when regime shifts
   - Bull trending: More aggressive on longs
   - Bear trending: More aggressive on shorts
   - High volatility: Reduce position sizes

3. **Review Top Features**: Learn which features drive decisions
   - Consistently important features are reliable
   - Rarely important features may need refinement

4. **Track Performance**: Log approved setups and outcomes
   - Calculate actual win rate vs ML probability
   - Adjust thresholds based on real results

5. **Use ML as Filter**: Let ML enhance, not replace, traditional analysis
   - Strong traditional signal + high ML probability = best setups
   - Weak traditional signal with high ML probability = review carefully

## Integration with Trading System

The ML enhancements are fully integrated:

```python
# No code changes needed!
# Just listen for SETUP_VALID events

async def handle_setup(event):
    setup = event.data

    # Access ML data
    ml_prob = setup.get('ml_probability', 0)
    confidence = setup.get('confidence', 0)
    regime = setup.get('market_regime', 'unknown')

    # Use for position sizing or filtering
    if confidence > 90:
        # High confidence = larger position
        pass

    if regime == 'high_volatility':
        # Reduce size in volatile markets
        pass
```

## Summary

The ML enhancement provides:
- **Better Signal Quality**: Only high-probability setups pass
- **Market Awareness**: Adjusts for current conditions
- **Explainability**: Know why each decision was made
- **Flexibility**: Configurable thresholds for your strategy
- **Reliability**: Graceful fallback if ML unavailable

Start with default settings (65% ML probability, 70% overall confidence) and adjust based on your results and risk tolerance.

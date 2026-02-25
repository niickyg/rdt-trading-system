# Complete Feature Catalog

## 70 Features Organized by Category

---

## 📊 Technical Indicators (20 features)

### Relative Strength Features
| Feature | Range | Description | Usage |
|---------|-------|-------------|-------|
| `rrs` | -5 to +5 | Real Relative Strength (1-bar) | >2.0 = strong RS, <-2.0 = strong RW |
| `rrs_3bar` | -5 to +5 | RRS over 3 bars | Smoother, less noise |
| `rrs_5bar` | -5 to +5 | RRS over 5 bars | Trend confirmation |

### Volatility Features
| Feature | Range | Description | Usage |
|---------|-------|-------------|-------|
| `atr` | 0+ | Average True Range | Absolute volatility measure |
| `atr_percent` | 0-10% | ATR as % of price | >3% = high volatility |

### Momentum Features
| Feature | Range | Description | Usage |
|---------|-------|-------------|-------|
| `rsi_14` | 0-100 | 14-period RSI | >70 overbought, <30 oversold |
| `rsi_9` | 0-100 | 9-period RSI | Faster, more responsive |
| `macd` | Any | MACD line | Trend direction |
| `macd_signal` | Any | MACD signal line | Entry/exit timing |
| `macd_histogram` | Any | MACD histogram | Momentum strength |

### Bollinger Bands
| Feature | Range | Description | Usage |
|---------|-------|-------------|-------|
| `bb_upper` | Price | Upper band (2σ) | Resistance level |
| `bb_middle` | Price | Middle band (SMA) | Mean price |
| `bb_lower` | Price | Lower band (2σ) | Support level |
| `bb_width` | 0+ | Band width | Volatility measure |
| `bb_percent` | 0-1 | Position in bands | >0.95 upper, <0.05 lower |

### Moving Averages
| Feature | Range | Description | Usage |
|---------|-------|-------------|-------|
| `ema_3` | Price | 3-period EMA | Very short-term trend |
| `ema_8` | Price | 8-period EMA | Short-term trend (RDT standard) |
| `ema_21` | Price | 21-period EMA | Medium-term trend |
| `ema_50` | Price | 50-period EMA | Long-term trend |
| `volume_sma_20` | Volume | 20-period volume avg | Volume baseline |

---

## 🔬 Microstructure Features (15 features)

### VWAP Features
| Feature | Range | Description | Usage |
|---------|-------|-------------|-------|
| `vwap` | Price | Volume Weighted Avg Price | Intraday fair value |
| `vwap_distance` | Any | Distance from VWAP | Absolute deviation |
| `vwap_distance_percent` | -10 to +10% | Distance from VWAP (%) | >2% = extended |

### Spread & Range
| Feature | Range | Description | Usage |
|---------|-------|-------------|-------|
| `bid_ask_spread` | 0+ | Estimated bid-ask spread | Liquidity measure |
| `price_range_percent` | 0-20% | Daily range as % | Volatility indicator |
| `intraday_high_low_range` | 0+ | Day's high-low range | Intraday volatility |
| `price_position_in_range` | 0-1 | Position in daily range | 0=low, 1=high |

### Momentum
| Feature | Range | Description | Usage |
|---------|-------|-------------|-------|
| `price_momentum_1` | -20 to +20% | 1-day price change | Very short-term |
| `price_momentum_5` | -30 to +30% | 5-day price change | Short-term momentum |
| `price_momentum_15` | -50 to +50% | 15-day price change | Medium-term momentum |

### Volume Analysis
| Feature | Range | Description | Usage |
|---------|-------|-------------|-------|
| `volume_ratio` | 0-5+ | Current vs avg volume | >1.5 = high volume |
| `volume_trend` | 0-3 | 5-day vs 20-day volume | Volume momentum |
| `relative_volume` | 0-5+ | Volume vs 20-day avg | Same as volume_ratio |

### Order Flow
| Feature | Range | Description | Usage |
|---------|-------|-------------|-------|
| `tick_direction` | -1, 0, 1 | Last tick direction | Buy/sell pressure |
| `order_flow_imbalance` | -1 to +1 | Upticks vs downticks | >0.5 = buying, <-0.5 = selling |

---

## 🌍 Regime Features (10 features)

### Volatility Regime
| Feature | Range | Description | Usage |
|---------|-------|-------------|-------|
| `vix` | 10-80 | VIX volatility index | <15 calm, >30 fear |
| `vix_change` | -50 to +50% | VIX daily change | Regime shift |

### Market Trend
| Feature | Range | Description | Usage |
|---------|-------|-------------|-------|
| `spy_trend` | -20 to +20% | SPY 10-day trend | Market direction |
| `spy_ema_alignment` | 0 or 1 | SPY EMA bullish/bearish | 1=bullish, 0=bearish |
| `spy_rsi` | 0-100 | SPY RSI | Market momentum |
| `spy_momentum` | -20 to +20% | SPY 5-day momentum | Short-term market |

### Relative Strength
| Feature | Range | Description | Usage |
|---------|-------|-------------|-------|
| `sector_relative_strength` | -30 to +30% | Stock vs SPY performance | Sector rotation |
| `correlation_with_spy` | -1 to +1 | 20-day correlation | Beta estimate |

### Market Breadth
| Feature | Range | Description | Usage |
|---------|-------|-------------|-------|
| `market_breadth` | 0-3 | SPY volume vs avg | Market participation |
| `spy_volume_ratio` | 0-3 | SPY volume ratio | Market volume |

---

## ⏰ Temporal Features (15 features)

### Time of Day
| Feature | Range | Description | Usage |
|---------|-------|-------------|-------|
| `hour_of_day` | 0-23 | Current hour | Session analysis |
| `minute_of_hour` | 0-59 | Current minute | Precision timing |
| `time_since_open_minutes` | 0-390 | Minutes since 9:30 AM | Session position |
| `time_until_close_minutes` | 0-390 | Minutes until 4:00 PM | Time remaining |

### Calendar Features
| Feature | Range | Description | Usage |
|---------|-------|-------------|-------|
| `day_of_week` | 0-6 | Day of week (0=Monday) | Weekly patterns |
| `day_of_month` | 1-31 | Day of month | Monthly patterns |
| `week_of_year` | 1-52 | Week of year | Seasonal patterns |

### Session Indicators
| Feature | Range | Description | Usage |
|---------|-------|-------------|-------|
| `is_market_open` | 0 or 1 | Market hours flag | 1=open (9:30-4:00) |
| `is_pre_market` | 0 or 1 | Pre-market flag | 1=pre-market |
| `is_after_hours` | 0 or 1 | After-hours flag | 1=after 4:00 PM |
| `is_first_hour` | 0 or 1 | First hour of trading | 1=9:00-10:00 |
| `is_last_hour` | 0 or 1 | Last hour of trading | 1=3:00-4:00 |
| `is_power_hour` | 0 or 1 | Power hour flag | 1=3:00-4:00 |
| `is_monday` | 0 or 1 | Monday flag | Monday patterns |
| `is_friday` | 0 or 1 | Friday flag | Friday patterns |

---

## 🧬 Derived Features (10 features)

### Interaction Features
| Feature | Range | Description | Usage |
|---------|-------|-------------|-------|
| `rrs_rsi_interaction` | -10 to +10 | RRS × RSI composite | Combined momentum |
| `momentum_volume_interaction` | -50 to +50 | Momentum × Volume | Quality of move |
| `volatility_regime_score` | 0-5 | VIX × ATR composite | Risk environment |

### Trend Composites
| Feature | Range | Description | Usage |
|---------|-------|-------------|-------|
| `trend_strength_composite` | -10 to +10 | EMA alignment strength | Trend quality |
| `daily_alignment_score` | -1 to +1 | EMA + RRS composite | Overall alignment |

### Probability Estimates
| Feature | Range | Description | Usage |
|---------|-------|-------------|-------|
| `reversal_probability` | 0-1 | Mean reversion likelihood | >0.7 = likely reversal |
| `breakout_probability` | 0-1 | Breakout likelihood | >0.7 = likely breakout |

### Risk Metrics
| Feature | Range | Description | Usage |
|---------|-------|-------------|-------|
| `risk_reward_ratio` | 0-5 | Risk/reward estimate | Position sizing |
| `sharpe_estimate` | -5 to +5 | Return/volatility ratio | Trade quality |
| `feature_complexity_score` | 0-1 | Signal complexity | 0=simple, 1=complex |

---

## 📈 Common Feature Combinations

### Strong Bullish Setup
```
rrs > 2.0
30 < rsi_14 < 70
volume_ratio > 1.5
daily_alignment_score > 0.5
breakout_probability > 0.7
```

### Strong Bearish Setup
```
rrs < -2.0
30 < rsi_14 < 70
volume_ratio > 1.5
daily_alignment_score < -0.5
reversal_probability < 0.3
```

### High Probability Breakout
```
bb_percent > 0.95 or bb_percent < 0.05
volume_ratio > 2.0
breakout_probability > 0.7
trend_strength_composite > 5.0
```

### Mean Reversion Setup
```
rsi_14 > 70 or rsi_14 < 30
bb_percent > 0.95 or bb_percent < 0.05
reversal_probability > 0.7
trend_strength_composite < 2.0
```

### Low Risk Entry
```
volatility_regime_score < 1.0
atr_percent < 2.0
vix < 20
correlation_with_spy > 0.7 (for longs in uptrend)
```

---

## 🎯 Feature Selection Guide

### For Momentum Trading
**Primary**: `rrs`, `rrs_5bar`, `volume_ratio`, `trend_strength_composite`
**Secondary**: `macd_histogram`, `spy_ema_alignment`, `breakout_probability`

### For Mean Reversion
**Primary**: `rsi_14`, `bb_percent`, `reversal_probability`
**Secondary**: `vwap_distance_percent`, `price_position_in_range`

### For Trend Following
**Primary**: `ema_3`, `ema_8`, `ema_21`, `daily_alignment_score`
**Secondary**: `macd`, `trend_strength_composite`, `spy_trend`

### For Breakout Trading
**Primary**: `bb_percent`, `volume_ratio`, `breakout_probability`
**Secondary**: `atr_percent`, `price_range_percent`, `order_flow_imbalance`

### For Risk Management
**Primary**: `atr_percent`, `vix`, `volatility_regime_score`
**Secondary**: `bb_width`, `correlation_with_spy`, `risk_reward_ratio`

---

## 📊 Data Types & Formats

All features are returned as `float` values in a pandas DataFrame:

```python
features_df.shape  # (1, 70+)  # Single row, 70+ columns
features_df.dtypes  # All float64
features_df.index   # [0]
```

Access features:
```python
# Single feature
rrs = features_df['rrs'].iloc[0]

# Multiple features
subset = features_df[['rrs', 'rsi_14', 'volume_ratio']]

# All technical features
technical = engineer.get_feature_names('technical')
tech_subset = features_df[technical]
```

---

## 🔄 Feature Updates

Features are calculated in real-time from market data:
- **Update frequency**: As often as you call `calculate_features()`
- **Data latency**: ~1-5 seconds (Yahoo Finance delay)
- **Cache duration**: Configurable (default 5 minutes)

For live trading:
```python
# Disable cache for real-time updates
features = await engineer.calculate_features("AAPL", use_cache=False)
```

---

## 📚 Additional Resources

- **Full Documentation**: `FEATURE_ENGINEERING_README.md`
- **Quick Reference**: `QUICK_REFERENCE.md`
- **Database Schema**: `schema.sql`
- **Usage Examples**: `examples/test_feature_engineering.py`
- **Integration Examples**: `examples/feature_engineering_ml_integration.py`

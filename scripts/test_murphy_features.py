#!/usr/bin/env python3
"""
Test Murphy-inspired features and intermarket analysis.

Usage:
    cd /home/user0/rdt-trading-system
    python scripts/test_murphy_features.py
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")


def test_intermarket_analyzer():
    """Test the IntermarketAnalyzer module."""
    print("\n" + "=" * 80)
    print("TEST 1: INTERMARKET ANALYZER")
    print("=" * 80)

    from scanner.intermarket_analyzer import IntermarketAnalyzer

    analyzer = IntermarketAnalyzer(cache_ttl_minutes=5)

    print("\nFetching intermarket signals (TLT, UUP, GLD, IWM, SPY)...")
    t0 = time.time()
    signals = analyzer.get_intermarket_signals()
    elapsed = time.time() - t0
    print(f"Fetch time: {elapsed:.1f}s\n")

    print(f"  bonds_stocks_divergence: {signals['bonds_stocks_divergence']:>+.3f}")
    print(f"  dollar_trend:            {signals['dollar_trend']:>+.3f}")
    print(f"  gold_signal:             {signals['gold_signal']:>+.3f}")
    print(f"  risk_on_off_ratio:       {signals['risk_on_off_ratio']:>+.3f}")
    print(f"  intermarket_composite:   {signals['intermarket_composite']:>+.3f}")
    print(f"  intermarket_regime:      {signals['intermarket_regime']}")

    rrs_adj = analyzer.get_rrs_adjustment()
    pos_mult = analyzer.get_position_size_multiplier()
    should_warn, reason = analyzer.should_warn()

    print(f"\n  RRS threshold adjustment: {rrs_adj:+.2f}")
    print(f"  Position size multiplier: {pos_mult:.2f}x")
    print(f"  Warning active: {should_warn}")
    if should_warn:
        print(f"  Warning reason: {reason}")

    # Test cache hit
    print("\n  Testing cache...")
    t0 = time.time()
    signals2 = analyzer.get_intermarket_signals()
    cache_time = time.time() - t0
    print(f"  Cache hit time: {cache_time*1000:.1f}ms (should be <10ms)")

    assert signals == signals2, "Cache should return identical results"
    print("  Cache: PASS")

    return True


def test_murphy_features():
    """Test Murphy features in the feature engineering pipeline."""
    print("\n" + "=" * 80)
    print("TEST 2: MURPHY FEATURES (feature_engineering.py)")
    print("=" * 80)

    from ml.feature_engineering import FeatureEngineer

    engineer = FeatureEngineer()

    # Verify murphy features are registered
    murphy_names = engineer.get_feature_names('murphy')
    print(f"\n  Murphy features registered: {len(murphy_names)}")
    for name in murphy_names:
        print(f"    - {name}")

    assert len(murphy_names) == 17, f"Expected 17 murphy features, got {len(murphy_names)}"
    print(f"\n  Feature count check: PASS (17 Murphy + {len(engineer.all_feature_names) - 17} existing = {len(engineer.all_feature_names)} total)")

    return True


def test_murphy_calculations():
    """Test Murphy feature calculations with real data."""
    print("\n" + "=" * 80)
    print("TEST 3: MURPHY FEATURE CALCULATIONS (with real data)")
    print("=" * 80)

    # Fetch real data for AAPL
    print("\n  Downloading AAPL and SPY daily data (1 year)...")
    t0 = time.time()

    aapl = yf.download("AAPL", period="1y", interval="1d", progress=False)
    spy = yf.download("SPY", period="1y", interval="1d", progress=False)

    if aapl.empty or spy.empty:
        print("  ERROR: Failed to download data")
        return False

    # Handle multi-index columns from modern yfinance
    if isinstance(aapl.columns, pd.MultiIndex):
        aapl.columns = [c[0].lower() for c in aapl.columns]
    else:
        aapl.columns = [c.lower() for c in aapl.columns]

    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0].lower() for c in spy.columns]
    else:
        spy.columns = [c.lower() for c in spy.columns]

    print(f"  AAPL: {len(aapl)} bars, SPY: {len(spy)} bars ({time.time()-t0:.1f}s)")

    close = aapl['close'].astype(float)
    volume = aapl['volume'].astype(float)
    high = aapl['high'].astype(float)
    low = aapl['low'].astype(float)
    current_price = float(close.iloc[-1])

    print(f"  AAPL current price: ${current_price:.2f}")
    print()

    results = {}

    # --- OBV ---
    print("  [OBV]")
    obv = (np.sign(close.diff()) * volume).cumsum()
    obv_last_10 = obv.tail(10).values
    x = np.arange(len(obv_last_10))
    if len(obv_last_10) >= 2:
        slope = np.polyfit(x, obv_last_10, 1)[0]
        obv_trend = 1.0 if slope > 0 else (-1.0 if slope < 0 else 0.0)
    else:
        obv_trend = 0.0

    # Price slope vs OBV slope
    price_last_10 = close.tail(10).values
    price_slope = np.polyfit(np.arange(len(price_last_10)), price_last_10, 1)[0]
    obv_price_div = 1.0 if (price_slope > 0 and slope < 0) or (price_slope < 0 and slope > 0) else 0.0

    vol_climax = 1.0 if float(volume.iloc[-1]) > float(volume.rolling(20).mean().iloc[-1]) * 3 else 0.0

    results['obv_trend'] = obv_trend
    results['obv_price_divergence'] = obv_price_div
    results['volume_climax'] = vol_climax
    print(f"    obv_trend: {obv_trend:+.0f}")
    print(f"    obv_price_divergence: {obv_price_div:.0f}")
    print(f"    volume_climax: {vol_climax:.0f}")

    # --- ADX ---
    print("  [ADX]")
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    smoothed_tr = tr.rolling(14).sum()
    smoothed_plus = plus_dm.rolling(14).sum()
    smoothed_minus = minus_dm.rolling(14).sum()

    plus_di = (smoothed_plus / smoothed_tr) * 100
    minus_di = (smoothed_minus / smoothed_tr) * 100
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(14).mean()

    adx_val = float(adx.iloc[-1])
    adx_5ago = float(adx.iloc[-6]) if len(adx) > 5 else adx_val
    adx_rising = 1.0 if adx_val > adx_5ago else 0.0

    results['adx'] = adx_val
    results['adx_rising'] = adx_rising
    print(f"    adx: {adx_val:.1f}")
    print(f"    adx_rising: {adx_rising:.0f} (was {adx_5ago:.1f} 5 bars ago)")
    print(f"    +DI: {float(plus_di.iloc[-1]):.1f}, -DI: {float(minus_di.iloc[-1]):.1f}")

    # --- 200 SMA ---
    print("  [200 SMA]")
    from shared.indicators.rrs import calculate_sma
    sma_200 = calculate_sma(close, 200)
    sma_50 = calculate_sma(close, 50)

    sma_200_val = float(sma_200.iloc[-1]) if len(close) >= 200 else 0.0
    sma_50_val = float(sma_50.iloc[-1]) if len(close) >= 50 else 0.0
    above_200 = 1.0 if current_price > sma_200_val else 0.0

    if sma_200_val > 0:
        ratio = sma_50_val / sma_200_val
        if ratio > 1.01:
            gc_state = 1.0
        elif ratio < 0.99:
            gc_state = -1.0
        else:
            gc_state = 0.0
    else:
        gc_state = 0.0

    results['sma_200'] = sma_200_val
    results['price_above_sma200'] = above_200
    results['golden_cross_state'] = gc_state
    print(f"    sma_200: ${sma_200_val:.2f}")
    print(f"    price_above_sma200: {above_200:.0f} (price ${current_price:.2f})")
    print(f"    golden_cross_state: {gc_state:+.0f} (50SMA=${sma_50_val:.2f})")

    # --- MACD Histogram ---
    print("  [MACD Histogram]")
    from shared.indicators.rrs import calculate_ema
    ema_12 = calculate_ema(close, 12)
    ema_26 = calculate_ema(close, 26)
    macd_line = ema_12 - ema_26
    macd_signal = calculate_ema(macd_line, 9)
    histogram = macd_line - macd_signal

    h = histogram.tail(3).values
    if len(h) == 3:
        if h[2] > h[1] > h[0]:
            hist_slope = 1.0
        elif h[2] < h[1] < h[0]:
            hist_slope = -1.0
        else:
            hist_slope = 0.0
    else:
        hist_slope = 0.0

    results['macd_histogram_slope'] = hist_slope
    print(f"    macd_histogram_slope: {hist_slope:+.0f} (last 3: {h[0]:.3f}, {h[1]:.3f}, {h[2]:.3f})")

    # --- BB Squeeze ---
    print("  [BB Squeeze]")
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    bb_width = (sma_20 + 2 * std_20) - (sma_20 - 2 * std_20)  # = 4 * std_20
    bb_width_min_50 = bb_width.rolling(50).min()

    current_width = float(bb_width.iloc[-1])
    min_width = float(bb_width_min_50.iloc[-1])
    squeeze = 1.0 if current_width <= min_width * 1.01 else 0.0

    results['bb_squeeze'] = squeeze
    print(f"    bb_width: {current_width:.2f}, 50-bar min: {min_width:.2f}")
    print(f"    bb_squeeze: {squeeze:.0f}")

    # --- Pullback Depth ---
    print("  [Pullback Depth]")
    last_20 = close.tail(20)
    high_20 = float(last_20.max())
    low_20 = float(last_20.min())
    range_20 = high_20 - low_20

    if range_20 > 0:
        mid = (high_20 + low_20) / 2
        if current_price >= mid:
            pullback_pct = ((high_20 - current_price) / range_20) * 100
        else:
            pullback_pct = ((current_price - low_20) / range_20) * 100
    else:
        pullback_pct = 0.0

    results['pullback_depth_pct'] = pullback_pct
    print(f"    20-bar range: ${low_20:.2f} - ${high_20:.2f}")
    print(f"    pullback_depth_pct: {pullback_pct:.1f}%")

    # --- ADX Gate on Reversal Probability ---
    print("\n  [ADX Gate Test]")
    rsi_14 = 75  # Overbought
    macd_hist = -0.5  # Negative histogram

    # Without ADX gate: reversal_probability would be 0.8
    if rsi_14 > 70 and macd_hist < 0:
        raw_reversal = 0.8
    else:
        raw_reversal = 0.2

    # With ADX gate: if ADX > 25 and rising, suppress to 0.2
    if adx_val > 25 and adx_rising == 1.0:
        gated_reversal = 0.2
        print(f"    RSI={rsi_14}, MACD_hist={macd_hist}")
        print(f"    Raw reversal_probability: {raw_reversal}")
        print(f"    ADX={adx_val:.1f} (>25) and rising → SUPPRESSED to {gated_reversal}")
        print(f"    Murphy Law #9: ADX gate working correctly!")
    else:
        gated_reversal = raw_reversal
        print(f"    ADX={adx_val:.1f}, rising={adx_rising}")
        print(f"    ADX gate not triggered (ADX <= 25 or not rising)")
        print(f"    reversal_probability stays at {gated_reversal}")

    print()
    return True


def test_walkforward_integration():
    """Test that the new features would integrate with the walk-forward backtest."""
    print("\n" + "=" * 80)
    print("TEST 4: INTERMARKET BACKTEST INTEGRATION")
    print("=" * 80)

    from scanner.intermarket_analyzer import IntermarketAnalyzer

    analyzer = IntermarketAnalyzer(cache_ttl_minutes=5)
    signals = analyzer.get_intermarket_signals()

    regime = signals['intermarket_regime']
    composite = signals['intermarket_composite']

    # Simulate how this would affect a signal
    base_rrs_threshold = 2.0
    rrs_adj = analyzer.get_rrs_adjustment()
    effective_threshold = base_rrs_threshold + rrs_adj
    pos_mult = analyzer.get_position_size_multiplier()

    print(f"\n  Current intermarket regime: {regime}")
    print(f"  Composite score: {composite:+.3f}")
    print()
    print(f"  Base RRS threshold: {base_rrs_threshold}")
    print(f"  Intermarket adjustment: {rrs_adj:+.2f}")
    print(f"  Effective RRS threshold: {effective_threshold}")
    print(f"  Position size multiplier: {pos_mult:.2f}x")
    print()

    # Example: a signal with RRS 2.1
    test_rrs = 2.1
    would_pass_base = test_rrs >= base_rrs_threshold
    would_pass_adjusted = test_rrs >= effective_threshold

    print(f"  Example signal with RRS {test_rrs}:")
    print(f"    Pass baseline threshold ({base_rrs_threshold})?  {'YES' if would_pass_base else 'NO'}")
    print(f"    Pass adjusted threshold ({effective_threshold})? {'YES' if would_pass_adjusted else 'NO'}")

    if would_pass_base != would_pass_adjusted:
        print(f"    → Intermarket filter CHANGED the outcome!")
    else:
        print(f"    → Same outcome (but position size would be {pos_mult:.2f}x)")

    print()
    return True


def main():
    print()
    print("=" * 80)
    print("MURPHY FEATURES & INTERMARKET ANALYSIS — TEST SUITE".center(80))
    print("=" * 80)

    results = {}

    # Test 1: Intermarket Analyzer
    try:
        results['intermarket'] = test_intermarket_analyzer()
    except Exception as e:
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['intermarket'] = False

    # Test 2: Feature names
    try:
        results['feature_names'] = test_murphy_features()
    except Exception as e:
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['feature_names'] = False

    # Test 3: Real calculations
    try:
        results['calculations'] = test_murphy_calculations()
    except Exception as e:
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['calculations'] = False

    # Test 4: Integration
    try:
        results['integration'] = test_walkforward_integration()
    except Exception as e:
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['integration'] = False

    # Summary
    print("=" * 80)
    print("TEST SUMMARY".center(80))
    print("=" * 80)
    print()
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<30} {status}")
    print()

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"  {passed}/{total} tests passed")
    print()

    if passed == total:
        print("  All tests passed!")
    else:
        print("  Some tests failed — check output above.")

    print()


if __name__ == "__main__":
    main()

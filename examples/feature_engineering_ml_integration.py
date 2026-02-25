"""
Feature Engineering + ML Model Integration Example

Demonstrates how to use the comprehensive feature engineering pipeline
with the existing ML models (Random Forest, XGBoost, LSTM).
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger

from ml.feature_engineering import FeatureEngineer
from shared.data_provider import DataProvider


async def demo_feature_calculation_for_watchlist():
    """Calculate features for entire watchlist."""
    logger.info("=== Feature Calculation for Watchlist ===")

    # Sample watchlist
    watchlist = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD', 'META']

    # Initialize feature engineer
    engineer = FeatureEngineer(cache_ttl_seconds=300)

    logger.info(f"Calculating features for {len(watchlist)} symbols...")

    # Calculate features in parallel
    features_dict = await engineer.calculate_batch_features(watchlist)

    # Create summary DataFrame
    summary = []
    for symbol, features_df in features_dict.items():
        if features_df is not None:
            summary.append({
                'symbol': symbol,
                'rrs': features_df['rrs'].iloc[0],
                'rsi_14': features_df['rsi_14'].iloc[0],
                'volume_ratio': features_df['volume_ratio'].iloc[0],
                'breakout_prob': features_df['breakout_probability'].iloc[0],
                'alignment_score': features_df['daily_alignment_score'].iloc[0]
            })

    summary_df = pd.DataFrame(summary)
    logger.info(f"\n{summary_df.to_string(index=False)}")

    return features_dict


async def demo_signal_screening():
    """Screen for trading signals based on features."""
    logger.info("\n=== Signal Screening ===")

    watchlist = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD', 'META', 'NFLX', 'AMZN', 'SPY']
    engineer = FeatureEngineer()

    features_dict = await engineer.calculate_batch_features(watchlist)

    # Screen for strong bullish signals
    logger.info("\n--- Strong Bullish Signals ---")
    bullish_signals = []

    for symbol, features_df in features_dict.items():
        if features_df is None:
            continue

        rrs = features_df['rrs'].iloc[0]
        rsi = features_df['rsi_14'].iloc[0]
        volume_ratio = features_df['volume_ratio'].iloc[0]
        alignment = features_df['daily_alignment_score'].iloc[0]
        breakout_prob = features_df['breakout_probability'].iloc[0]

        # Bullish criteria
        is_bullish = (
            rrs > 2.0 and  # Strong relative strength
            30 < rsi < 70 and  # Not overbought
            volume_ratio > 1.2 and  # Above average volume
            alignment > 0.3  # Bullish alignment
        )

        if is_bullish:
            bullish_signals.append({
                'symbol': symbol,
                'rrs': rrs,
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'alignment': alignment,
                'breakout_prob': breakout_prob
            })

    if bullish_signals:
        bullish_df = pd.DataFrame(bullish_signals)
        bullish_df = bullish_df.sort_values('rrs', ascending=False)
        logger.info(f"\nFound {len(bullish_signals)} bullish signals:")
        logger.info(f"\n{bullish_df.to_string(index=False)}")
    else:
        logger.info("No bullish signals found")

    # Screen for strong bearish signals
    logger.info("\n--- Strong Bearish Signals ---")
    bearish_signals = []

    for symbol, features_df in features_dict.items():
        if features_df is None:
            continue

        rrs = features_df['rrs'].iloc[0]
        rsi = features_df['rsi_14'].iloc[0]
        volume_ratio = features_df['volume_ratio'].iloc[0]
        alignment = features_df['daily_alignment_score'].iloc[0]

        # Bearish criteria
        is_bearish = (
            rrs < -2.0 and  # Strong relative weakness
            30 < rsi < 70 and  # Not oversold
            volume_ratio > 1.2 and  # Above average volume
            alignment < -0.3  # Bearish alignment
        )

        if is_bearish:
            bearish_signals.append({
                'symbol': symbol,
                'rrs': rrs,
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'alignment': alignment
            })

    if bearish_signals:
        bearish_df = pd.DataFrame(bearish_signals)
        bearish_df = bearish_df.sort_values('rrs', ascending=True)
        logger.info(f"\nFound {len(bearish_signals)} bearish signals:")
        logger.info(f"\n{bearish_df.to_string(index=False)}")
    else:
        logger.info("No bearish signals found")


async def demo_feature_monitoring():
    """Monitor feature changes over time."""
    logger.info("\n=== Feature Monitoring ===")

    symbol = "AAPL"
    engineer = FeatureEngineer(cache_ttl_seconds=10)

    logger.info(f"Monitoring features for {symbol} (3 snapshots, 15 seconds apart)...")

    snapshots = []
    for i in range(3):
        logger.info(f"\nSnapshot {i+1}/3...")

        # Calculate features (force refresh)
        features = await engineer.calculate_features(symbol, use_cache=False)

        if features is not None:
            snapshots.append({
                'timestamp': datetime.now(),
                'rrs': features['rrs'].iloc[0],
                'rsi_14': features['rsi_14'].iloc[0],
                'volume_ratio': features['volume_ratio'].iloc[0],
                'vwap_distance_pct': features['vwap_distance_percent'].iloc[0]
            })

        if i < 2:
            await asyncio.sleep(15)

    # Display monitoring results
    if snapshots:
        monitor_df = pd.DataFrame(snapshots)
        logger.info(f"\n{monitor_df.to_string(index=False)}")

        # Calculate changes
        if len(snapshots) >= 2:
            rrs_change = snapshots[-1]['rrs'] - snapshots[0]['rrs']
            rsi_change = snapshots[-1]['rsi_14'] - snapshots[0]['rsi_14']

            logger.info(f"\nChanges over monitoring period:")
            logger.info(f"  RRS change: {rrs_change:+.4f}")
            logger.info(f"  RSI change: {rsi_change:+.4f}")


async def demo_feature_based_ranking():
    """Rank symbols by composite feature score."""
    logger.info("\n=== Feature-Based Symbol Ranking ===")

    watchlist = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD', 'META', 'NFLX', 'AMZN', 'DIS']
    engineer = FeatureEngineer()

    features_dict = await engineer.calculate_batch_features(watchlist)

    # Calculate composite scores
    rankings = []
    for symbol, features_df in features_dict.items():
        if features_df is None:
            continue

        # Extract key features
        rrs = features_df['rrs'].iloc[0]
        rsi = features_df['rsi_14'].iloc[0]
        volume_ratio = features_df['volume_ratio'].iloc[0]
        alignment = features_df['daily_alignment_score'].iloc[0]
        breakout_prob = features_df['breakout_probability'].iloc[0]
        trend_strength = features_df['trend_strength_composite'].iloc[0]

        # Calculate composite score (weighted)
        # Higher score = better bullish setup
        score = (
            rrs * 0.30 +  # 30% weight on RRS
            ((rsi - 50) / 50) * 0.15 +  # 15% weight on RSI momentum
            (volume_ratio - 1) * 0.20 +  # 20% weight on volume
            alignment * 0.20 +  # 20% weight on alignment
            breakout_prob * 0.15  # 15% weight on breakout probability
        )

        rankings.append({
            'symbol': symbol,
            'composite_score': score,
            'rrs': rrs,
            'rsi': rsi,
            'volume_ratio': volume_ratio,
            'alignment': alignment,
            'breakout_prob': breakout_prob
        })

    # Sort by composite score
    rankings_df = pd.DataFrame(rankings)
    rankings_df = rankings_df.sort_values('composite_score', ascending=False)

    logger.info("\nSymbol Rankings (by composite score):")
    logger.info(f"\n{rankings_df.to_string(index=False)}")


async def demo_risk_analysis():
    """Analyze risk metrics from features."""
    logger.info("\n=== Risk Analysis ===")

    symbols = ['AAPL', 'TSLA', 'NVDA']  # Different risk profiles
    engineer = FeatureEngineer()

    features_dict = await engineer.calculate_batch_features(symbols)

    risk_analysis = []
    for symbol, features_df in features_dict.items():
        if features_df is None:
            continue

        atr_percent = features_df['atr_percent'].iloc[0]
        vix = features_df['vix'].iloc[0]
        volatility_regime = features_df['volatility_regime_score'].iloc[0]
        bb_width = features_df['bb_width'].iloc[0]
        price_range_pct = features_df['price_range_percent'].iloc[0]

        # Risk score (higher = riskier)
        risk_score = (
            atr_percent * 0.30 +
            (vix / 20) * 0.25 +
            volatility_regime * 0.25 +
            price_range_pct * 0.20
        )

        risk_analysis.append({
            'symbol': symbol,
            'risk_score': risk_score,
            'atr_percent': atr_percent,
            'vix': vix,
            'volatility_regime': volatility_regime,
            'bb_width': bb_width,
            'price_range_pct': price_range_pct
        })

    risk_df = pd.DataFrame(risk_analysis)
    risk_df = risk_df.sort_values('risk_score', ascending=False)

    logger.info("\nRisk Analysis (higher score = higher risk):")
    logger.info(f"\n{risk_df.to_string(index=False)}")


async def demo_market_regime_detection():
    """Detect current market regime using features."""
    logger.info("\n=== Market Regime Detection ===")

    engineer = FeatureEngineer()

    # Get SPY features to determine market regime
    spy_features = await engineer.calculate_features('SPY')

    if spy_features is not None:
        vix = spy_features['vix'].iloc[0]
        spy_trend = spy_features['spy_trend'].iloc[0]
        spy_rsi = spy_features['spy_rsi'].iloc[0]
        spy_momentum = spy_features['spy_momentum'].iloc[0]
        spy_ema_alignment = spy_features['spy_ema_alignment'].iloc[0]

        logger.info("\n--- Market Regime Indicators ---")
        logger.info(f"VIX: {vix:.2f}")
        logger.info(f"SPY Trend (10-day): {spy_trend:.2f}%")
        logger.info(f"SPY RSI: {spy_rsi:.2f}")
        logger.info(f"SPY Momentum (5-day): {spy_momentum:.2f}%")
        logger.info(f"SPY EMA Alignment: {'Bullish' if spy_ema_alignment > 0.5 else 'Bearish'}")

        # Determine regime
        if vix < 15:
            volatility_regime = "Low Volatility"
        elif vix < 20:
            volatility_regime = "Normal Volatility"
        elif vix < 30:
            volatility_regime = "Elevated Volatility"
        else:
            volatility_regime = "High Volatility / Fear"

        if spy_trend > 2 and spy_ema_alignment > 0.5:
            trend_regime = "Strong Uptrend"
        elif spy_trend > 0 and spy_ema_alignment > 0.5:
            trend_regime = "Moderate Uptrend"
        elif spy_trend < -2 and spy_ema_alignment < 0.5:
            trend_regime = "Strong Downtrend"
        elif spy_trend < 0 and spy_ema_alignment < 0.5:
            trend_regime = "Moderate Downtrend"
        else:
            trend_regime = "Choppy / Range-bound"

        logger.info(f"\n--- Market Regime ---")
        logger.info(f"Volatility Regime: {volatility_regime}")
        logger.info(f"Trend Regime: {trend_regime}")

        # Trading recommendations based on regime
        logger.info(f"\n--- Trading Recommendations ---")
        if trend_regime in ["Strong Uptrend", "Moderate Uptrend"]:
            logger.info("• Focus on long setups with strong RRS")
            logger.info("• Look for pullbacks to support")
            logger.info("• Favor momentum strategies")
        elif trend_regime in ["Strong Downtrend", "Moderate Downtrend"]:
            logger.info("• Focus on short setups with strong RW")
            logger.info("• Look for rallies to resistance")
            logger.info("• Consider hedging long positions")
        else:
            logger.info("• Reduce position sizes")
            logger.info("• Focus on range-trading strategies")
            logger.info("• Wait for clearer trend signals")

        if volatility_regime in ["High Volatility / Fear", "Elevated Volatility"]:
            logger.info("• Widen stop losses")
            logger.info("• Reduce position sizes")
            logger.info("• Consider options strategies")


async def main():
    """Run all integration examples."""
    logger.info("=" * 70)
    logger.info("Feature Engineering + ML Integration Examples")
    logger.info("=" * 70)

    try:
        # Demo 1: Watchlist feature calculation
        await demo_feature_calculation_for_watchlist()

        # Demo 2: Signal screening
        await demo_signal_screening()

        # Demo 3: Feature ranking
        await demo_feature_based_ranking()

        # Demo 4: Risk analysis
        await demo_risk_analysis()

        # Demo 5: Market regime detection
        await demo_market_regime_detection()

        # Note: Demo 6 (monitoring) takes 30+ seconds, so it's optional
        # Uncomment to run:
        # await demo_feature_monitoring()

        logger.info("\n" + "=" * 70)
        logger.info("All integration examples completed!")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Error in integration examples: {e}", exc_info=True)


if __name__ == "__main__":
    # Configure logger
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # Run examples
    asyncio.run(main())

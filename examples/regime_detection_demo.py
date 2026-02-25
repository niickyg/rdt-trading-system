#!/usr/bin/env python3
"""
Market Regime Detection Demo

This script demonstrates how to use the MarketRegimeDetector for:
1. Training a new model
2. Loading a pre-trained model
3. Making predictions
4. Integrating with trading strategies

Usage:
    python examples/regime_detection_demo.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.regime_detector import MarketRegimeDetector
from loguru import logger
import pandas as pd


def demo_training():
    """Demonstrate model training."""
    logger.info("\n" + "="*80)
    logger.info("DEMO 1: Training a New Regime Detector")
    logger.info("="*80 + "\n")

    # Initialize detector
    detector = MarketRegimeDetector(
        n_regimes=4,
        n_iter=100,
        random_state=42
    )

    # Train on SPY data
    logger.info("Training on 5 years of SPY data...")
    metrics = detector.train(symbol="SPY", period="5y")

    logger.info("\nTraining Metrics:")
    logger.info(f"  Log Likelihood: {metrics['log_likelihood']:.2f}")
    logger.info(f"  Samples: {metrics['n_samples']}")
    logger.info(f"  Features: {metrics['n_features']}")

    logger.info("\nRegime Distribution:")
    for regime, count in metrics['regime_distribution'].items():
        pct = count / metrics['n_samples'] * 100
        logger.info(f"  {regime}: {count} ({pct:.1f}%)")

    # Save model
    detector.save_model()
    logger.success("\nModel saved successfully!")

    return detector


def demo_prediction(detector: MarketRegimeDetector):
    """Demonstrate regime prediction."""
    logger.info("\n" + "="*80)
    logger.info("DEMO 2: Predicting Current Market Regime")
    logger.info("="*80 + "\n")

    # Predict current regime
    regime, info = detector.predict(return_confidence=True)

    logger.info(f"Current Market Regime: {regime.upper()}")
    logger.info(f"Timestamp: {info['timestamp']}")

    logger.info("\nConfidence Scores:")
    for regime_name, score in info['confidence_scores'].items():
        bar = '█' * int(score * 50)
        logger.info(f"  {regime_name:20s}: {score:.4f} {bar}")

    logger.info("\nRecommended Strategy Allocation:")
    for strategy, weight in info['strategy_allocation'].items():
        logger.info(f"  {strategy:20s}: {weight*100:5.1f}%")

    return regime, info


def demo_sequence_prediction(detector: MarketRegimeDetector):
    """Demonstrate regime prediction over time series."""
    logger.info("\n" + "="*80)
    logger.info("DEMO 3: Predicting Regime Sequence")
    logger.info("="*80 + "\n")

    # Fetch recent data
    data = detector.fetch_data(symbol="SPY", period="6mo")

    # Predict regime sequence
    results = detector.predict_sequence(data)

    logger.info(f"Predicted regimes for {len(results)} periods")
    logger.info("\nRegime Distribution (last 6 months):")

    regime_counts = results['regime_name'].value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(results) * 100
        logger.info(f"  {regime}: {count} ({pct:.1f}%)")

    # Show recent regimes
    logger.info("\nMost Recent 10 Predictions:")
    recent = results[['regime_name']].tail(10)
    for date, row in recent.iterrows():
        logger.info(f"  {date.strftime('%Y-%m-%d')}: {row['regime_name']}")

    # Analyze transitions
    transitions = detector.get_regime_transitions(results)
    logger.info(f"\nNumber of regime transitions: {len(transitions)}")

    if len(transitions) > 0:
        logger.info("\nMost Recent Transitions:")
        for _, trans in transitions.tail(5).iterrows():
            logger.info(
                f"  {trans['date'].strftime('%Y-%m-%d')}: "
                f"{trans['from_regime']} -> {trans['to_regime']}"
            )

    return results


def demo_statistics(detector: MarketRegimeDetector):
    """Demonstrate regime statistics calculation."""
    logger.info("\n" + "="*80)
    logger.info("DEMO 4: Regime Performance Statistics")
    logger.info("="*80 + "\n")

    # Get statistics
    stats = detector.get_regime_statistics(detector.raw_data)

    logger.info("Performance Metrics by Regime:\n")
    logger.info(stats.to_string())

    # Highlight best/worst regimes
    logger.info("\nKey Insights:")

    best_return = stats.loc[stats['avg_return'].idxmax()]
    logger.info(f"  Best average returns: {best_return['regime']} "
               f"({best_return['avg_return']*100:.4f}%)")

    worst_return = stats.loc[stats['avg_return'].idxmin()]
    logger.info(f"  Worst average returns: {worst_return['regime']} "
               f"({worst_return['avg_return']*100:.4f}%)")

    best_sharpe = stats.loc[stats['sharpe_ratio'].idxmax()]
    logger.info(f"  Best Sharpe ratio: {best_sharpe['regime']} "
               f"({best_sharpe['sharpe_ratio']:.4f})")

    highest_vol = stats.loc[stats['volatility'].idxmax()]
    logger.info(f"  Highest volatility: {highest_vol['regime']} "
               f"({highest_vol['volatility']*100:.4f}%)")


def demo_strategy_integration(regime: str, info: dict):
    """Demonstrate integration with trading strategies."""
    logger.info("\n" + "="*80)
    logger.info("DEMO 5: Strategy Integration")
    logger.info("="*80 + "\n")

    allocation = info['strategy_allocation']

    logger.info(f"Current Regime: {regime}")
    logger.info("\nStrategy Recommendations:\n")

    # Simulate portfolio of $100,000
    portfolio_value = 100000

    logger.info("Portfolio Allocation ($100,000):")
    for strategy, weight in allocation.items():
        amount = portfolio_value * weight
        logger.info(f"  {strategy.title():20s}: ${amount:>10,.2f} ({weight*100:5.1f}%)")

    # Strategy-specific recommendations
    logger.info("\nStrategy-Specific Actions:")

    if regime == 'bull_trending':
        logger.info("  Momentum Strategy:")
        logger.info("    - Focus on strong uptrend stocks")
        logger.info("    - Use tight stops to protect gains")
        logger.info("    - Consider sector rotation into growth")
        logger.info("  Mean Reversion Strategy:")
        logger.info("    - Reduce position sizes")
        logger.info("    - Focus on oversold quality stocks")
        logger.info("  Options Strategy:")
        logger.info("    - Sell cash-secured puts on strong stocks")
        logger.info("    - Use covered calls sparingly")

    elif regime == 'bear_trending':
        logger.info("  Momentum Strategy:")
        logger.info("    - Consider inverse ETFs or short positions")
        logger.info("    - Reduce overall exposure")
        logger.info("  Mean Reversion Strategy:")
        logger.info("    - Increase allocation to oversold stocks")
        logger.info("    - Look for capitulation signals")
        logger.info("  Options Strategy:")
        logger.info("    - Buy protective puts")
        logger.info("    - Consider bear call spreads")

    elif regime == 'high_volatility':
        logger.info("  All Strategies:")
        logger.info("    - Reduce position sizes")
        logger.info("    - Widen stop losses")
        logger.info("    - Increase cash reserves")
        logger.info("  Options Strategy:")
        logger.info("    - Sell premium through strangles/straddles")
        logger.info("    - Capitalize on elevated IV")

    elif regime == 'low_volatility':
        logger.info("  Momentum Strategy:")
        logger.info("    - Can use higher leverage safely")
        logger.info("    - Focus on breakout patterns")
        logger.info("  Mean Reversion Strategy:")
        logger.info("    - Tight ranges offer good opportunities")
        logger.info("    - Use Bollinger Bands strategy")
        logger.info("  Options Strategy:")
        logger.info("    - Buy options due to low IV")
        logger.info("    - Avoid selling premium")


def demo_load_model():
    """Demonstrate loading a pre-trained model."""
    logger.info("\n" + "="*80)
    logger.info("DEMO: Loading Pre-trained Model")
    logger.info("="*80 + "\n")

    detector = MarketRegimeDetector()

    try:
        detector.load_model()
        logger.success("Model loaded successfully!")

        # Quick prediction
        regime, info = detector.predict(return_confidence=True)
        logger.info(f"\nCurrent regime: {regime}")

        return detector

    except FileNotFoundError:
        logger.warning("No pre-trained model found. Training new model...")
        return None


def main():
    """Run all demos."""
    logger.info("\n" + "="*80)
    logger.info("Market Regime Detection System - Complete Demo")
    logger.info("="*80)

    # Try to load existing model
    detector = demo_load_model()

    # If no model exists, train a new one
    if detector is None:
        detector = demo_training()

    # Run all demos
    regime, info = demo_prediction(detector)
    results = demo_sequence_prediction(detector)
    demo_statistics(detector)
    demo_strategy_integration(regime, info)

    logger.info("\n" + "="*80)
    logger.success("Demo completed successfully!")
    logger.info("="*80 + "\n")

    logger.info("Next Steps:")
    logger.info("  1. Run training script: python scripts/train_regime_detector.py --visualize")
    logger.info("  2. Integrate with your trading system")
    logger.info("  3. Backtest strategy allocations")
    logger.info("  4. Set up automated regime monitoring")


if __name__ == "__main__":
    main()

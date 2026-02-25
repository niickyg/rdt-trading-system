#!/usr/bin/env python3
"""
Regime-Based Strategy Integration Example

This example shows how to integrate the Market Regime Detector with the
existing multi-strategy trading system to dynamically adjust strategy weights
based on detected market regimes.

Usage:
    python examples/regime_strategy_integration.py
"""

import sys
from pathlib import Path
from typing import Dict
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.regime_detector import MarketRegimeDetector
from loguru import logger
import pandas as pd


class RegimeAwareStrategyManager:
    """
    Manages trading strategies based on detected market regimes.

    Integrates with the RDT trading system's multi-strategy engine to
    dynamically adjust strategy allocations.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize the regime-aware strategy manager.

        Args:
            model_path: Path to trained regime detector model
        """
        self.detector = MarketRegimeDetector(model_path=model_path)

        # Try to load existing model
        try:
            self.detector.load_model()
            logger.success("Loaded pre-trained regime detector model")
        except FileNotFoundError:
            logger.warning("No pre-trained model found. Train with: python scripts/train_regime_detector.py")
            raise

        self.current_regime = None
        self.current_allocation = None
        self.last_update = None

    def update_regime(self) -> str:
        """
        Update the current market regime.

        Returns:
            Current regime name
        """
        logger.info("Updating market regime...")

        regime, info = self.detector.predict(return_confidence=True)

        self.current_regime = regime
        self.current_allocation = info['strategy_allocation']
        self.last_update = datetime.now()

        logger.info(f"Current regime: {regime}")
        logger.info(f"Confidence: {info['confidence_scores'][regime]:.2%}")

        return regime

    def get_strategy_weights(self) -> Dict[str, float]:
        """
        Get current strategy allocation weights.

        Returns:
            Dictionary of strategy weights
        """
        if self.current_allocation is None:
            self.update_regime()

        return self.current_allocation.copy()

    def adjust_position_size(self, base_size: float, strategy: str) -> float:
        """
        Adjust position size based on regime and strategy.

        Args:
            base_size: Base position size
            strategy: Strategy name

        Returns:
            Adjusted position size
        """
        if self.current_allocation is None:
            self.update_regime()

        # Get weight for this strategy
        weight = self.current_allocation.get(strategy, 0.25)

        # Adjust size
        adjusted_size = base_size * weight * 4  # Normalize back from allocation

        logger.debug(f"Position size adjustment for {strategy}: "
                    f"{base_size} -> {adjusted_size} (weight: {weight:.2%})")

        return adjusted_size

    def should_take_trade(self, strategy: str, signal_strength: float = 0.5) -> bool:
        """
        Determine if a trade should be taken based on regime and signal strength.

        Args:
            strategy: Strategy name
            signal_strength: Signal strength (0 to 1)

        Returns:
            True if trade should be taken
        """
        if self.current_allocation is None:
            self.update_regime()

        # Get allocation for this strategy
        allocation = self.current_allocation.get(strategy, 0.25)

        # Higher allocation = lower threshold
        threshold = 1.0 - allocation

        # Adjust threshold by signal strength
        should_take = signal_strength >= (threshold * 0.5)

        logger.debug(f"Trade decision for {strategy}: signal={signal_strength:.2f}, "
                    f"threshold={threshold:.2f}, take={should_take}")

        return should_take

    def get_regime_recommendations(self) -> Dict:
        """
        Get detailed recommendations for current regime.

        Returns:
            Dictionary with regime-specific recommendations
        """
        if self.current_regime is None:
            self.update_regime()

        recommendations = {
            'regime': self.current_regime,
            'allocation': self.current_allocation,
            'timestamp': self.last_update
        }

        # Add regime-specific guidance
        if self.current_regime == 'bull_trending':
            recommendations['guidance'] = {
                'momentum': 'Focus on strong uptrend stocks, use tight stops',
                'mean_reversion': 'Reduce position sizes, oversold quality stocks only',
                'pairs': 'Standard allocation, focus on sector pairs',
                'options': 'Sell cash-secured puts, limited covered calls',
                'risk_level': 'medium',
                'leverage_multiplier': 1.2
            }

        elif self.current_regime == 'bear_trending':
            recommendations['guidance'] = {
                'momentum': 'Consider inverse ETFs, reduce overall exposure',
                'mean_reversion': 'Increase allocation, look for capitulation',
                'pairs': 'Standard allocation, defensive pairs',
                'options': 'Buy protective puts, bear call spreads',
                'risk_level': 'high',
                'leverage_multiplier': 0.7
            }

        elif self.current_regime == 'high_volatility':
            recommendations['guidance'] = {
                'momentum': 'Reduce position sizes, widen stops',
                'mean_reversion': 'Reduce position sizes, quick profits',
                'pairs': 'Reduce allocation, tight risk management',
                'options': 'Sell premium through strangles/straddles',
                'risk_level': 'very_high',
                'leverage_multiplier': 0.5
            }

        elif self.current_regime == 'low_volatility':
            recommendations['guidance'] = {
                'momentum': 'Can use higher leverage safely, breakouts',
                'mean_reversion': 'Tight ranges, Bollinger Band strategy',
                'pairs': 'Standard allocation',
                'options': 'Buy options due to low IV, avoid selling',
                'risk_level': 'low',
                'leverage_multiplier': 1.5
            }

        return recommendations

    def get_risk_parameters(self) -> Dict:
        """
        Get risk management parameters for current regime.

        Returns:
            Dictionary with risk parameters
        """
        if self.current_regime is None:
            self.update_regime()

        # Base parameters
        risk_params = {
            'max_position_size': 0.10,
            'max_portfolio_heat': 0.15,
            'stop_loss_multiplier': 2.0,
            'profit_target_multiplier': 3.0
        }

        # Adjust based on regime
        if self.current_regime == 'bull_trending':
            risk_params['max_position_size'] = 0.12
            risk_params['max_portfolio_heat'] = 0.20
            risk_params['stop_loss_multiplier'] = 1.5

        elif self.current_regime == 'bear_trending':
            risk_params['max_position_size'] = 0.08
            risk_params['max_portfolio_heat'] = 0.12
            risk_params['stop_loss_multiplier'] = 2.5

        elif self.current_regime == 'high_volatility':
            risk_params['max_position_size'] = 0.06
            risk_params['max_portfolio_heat'] = 0.10
            risk_params['stop_loss_multiplier'] = 3.0

        elif self.current_regime == 'low_volatility':
            risk_params['max_position_size'] = 0.15
            risk_params['max_portfolio_heat'] = 0.25
            risk_params['stop_loss_multiplier'] = 1.5

        return risk_params

    def log_regime_status(self):
        """Log current regime status and allocations."""
        if self.current_regime is None:
            self.update_regime()

        logger.info("\n" + "="*60)
        logger.info("REGIME STATUS")
        logger.info("="*60)
        logger.info(f"Current Regime: {self.current_regime.upper()}")
        logger.info(f"Last Updated: {self.last_update}")

        logger.info("\nStrategy Allocation:")
        for strategy, weight in self.current_allocation.items():
            bar = '█' * int(weight * 40)
            logger.info(f"  {strategy:20s}: {weight:5.1%} {bar}")

        # Get recommendations
        recs = self.get_regime_recommendations()
        logger.info("\nGuidance:")
        for strategy, guidance in recs['guidance'].items():
            if strategy not in ['risk_level', 'leverage_multiplier']:
                logger.info(f"  {strategy}: {guidance}")

        logger.info(f"\nRisk Level: {recs['guidance']['risk_level'].upper()}")
        logger.info(f"Leverage Multiplier: {recs['guidance']['leverage_multiplier']:.1f}x")

        # Get risk parameters
        risk_params = self.get_risk_parameters()
        logger.info("\nRisk Parameters:")
        for param, value in risk_params.items():
            logger.info(f"  {param}: {value}")

        logger.info("="*60 + "\n")


def example_trading_loop():
    """
    Example trading loop with regime-based adjustments.
    """
    logger.info("Starting regime-aware trading example...\n")

    try:
        # Initialize manager
        manager = RegimeAwareStrategyManager()

        # Update regime
        manager.update_regime()
        manager.log_regime_status()

        # Simulate trading decisions
        logger.info("\nSimulating Trading Decisions:\n")

        # Example 1: Momentum trade
        strategy = "momentum"
        signal_strength = 0.75
        should_take = manager.should_take_trade(strategy, signal_strength)

        logger.info(f"Momentum Trade Signal (strength: {signal_strength}):")
        logger.info(f"  Decision: {'TAKE' if should_take else 'SKIP'}")

        if should_take:
            base_size = 1000  # shares
            adjusted_size = manager.adjust_position_size(base_size, strategy)
            logger.info(f"  Position Size: {adjusted_size:.0f} shares (from {base_size})")

        # Example 2: Mean reversion trade
        strategy = "mean_reversion"
        signal_strength = 0.60
        should_take = manager.should_take_trade(strategy, signal_strength)

        logger.info(f"\nMean Reversion Trade Signal (strength: {signal_strength}):")
        logger.info(f"  Decision: {'TAKE' if should_take else 'SKIP'}")

        if should_take:
            base_size = 1000
            adjusted_size = manager.adjust_position_size(base_size, strategy)
            logger.info(f"  Position Size: {adjusted_size:.0f} shares (from {base_size})")

        # Example 3: Options trade
        strategy = "options"
        signal_strength = 0.55
        should_take = manager.should_take_trade(strategy, signal_strength)

        logger.info(f"\nOptions Trade Signal (strength: {signal_strength}):")
        logger.info(f"  Decision: {'TAKE' if should_take else 'SKIP'}")

        if should_take:
            base_size = 5  # contracts
            adjusted_size = manager.adjust_position_size(base_size, strategy)
            logger.info(f"  Position Size: {adjusted_size:.0f} contracts (from {base_size})")

        # Example 4: Check risk parameters
        logger.info("\nRisk Management:")
        risk_params = manager.get_risk_parameters()
        logger.info(f"  Max Position Size: {risk_params['max_position_size']:.1%}")
        logger.info(f"  Max Portfolio Heat: {risk_params['max_portfolio_heat']:.1%}")
        logger.info(f"  Stop Loss: {risk_params['stop_loss_multiplier']:.1f}x ATR")
        logger.info(f"  Profit Target: {risk_params['profit_target_multiplier']:.1f}x ATR")

        logger.success("\nExample completed successfully!")

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.info("\nTo use this example, first train a model:")
        logger.info("  python scripts/train_regime_detector.py")


def example_multi_strategy_integration():
    """
    Example integration with multi-strategy engine.
    """
    logger.info("\n" + "="*60)
    logger.info("MULTI-STRATEGY ENGINE INTEGRATION")
    logger.info("="*60 + "\n")

    try:
        manager = RegimeAwareStrategyManager()
        manager.update_regime()

        # Get strategy weights
        weights = manager.get_strategy_weights()

        logger.info("Integration with MultiStrategyEngine:\n")
        logger.info("```python")
        logger.info("from strategies.multi_strategy_engine import MultiStrategyEngine")
        logger.info("from examples.regime_strategy_integration import RegimeAwareStrategyManager")
        logger.info("")
        logger.info("# Initialize")
        logger.info("manager = RegimeAwareStrategyManager()")
        logger.info("engine = MultiStrategyEngine()")
        logger.info("")
        logger.info("# Update based on regime")
        logger.info("manager.update_regime()")
        logger.info("weights = manager.get_strategy_weights()")
        logger.info("")
        logger.info("# Apply to engine")
        logger.info("engine.set_weights(weights)")
        logger.info("```")

        logger.info(f"\nCurrent Weights: {weights}")

    except Exception as e:
        logger.error(f"Error: {e}")


def main():
    """Run all examples."""
    logger.info("="*60)
    logger.info("REGIME-BASED STRATEGY INTEGRATION EXAMPLES")
    logger.info("="*60)

    # Run trading loop example
    example_trading_loop()

    # Run multi-strategy integration example
    example_multi_strategy_integration()

    logger.info("\n" + "="*60)
    logger.info("All examples completed!")
    logger.info("="*60)


if __name__ == "__main__":
    main()

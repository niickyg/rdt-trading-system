"""
Example: Integrating Learning Agent into the Trading System

This shows how to add the Learning Agent to your existing trading system.
"""

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import (
    EventBus,
    ScannerAgent,
    AnalyzerAgent,
    ExecutorAgent,
    LearningAgent,
    get_event_bus
)
from loguru import logger


async def run_trading_system_with_learning():
    """
    Run the complete trading system with Learning Agent integrated
    """

    logger.info("=" * 60)
    logger.info("Trading System with Learning Agent")
    logger.info("=" * 60)

    # 1. Initialize Event Bus
    event_bus = get_event_bus()
    await event_bus.start()

    # 2. Create Learning Agent
    learning_agent = LearningAgent(
        event_bus=event_bus,
        model_dir=Path.home() / ".rdt-trading" / "models",
        retrain_threshold=100,      # Retrain after 100 trades
        retrain_interval_days=7,    # Or weekly
        lookback_months=6,          # Use 6 months of data
        min_improvement=0.02        # Require 2% improvement
    )

    # 3. Create other agents (simplified - you'd use real implementations)
    # scanner_agent = ScannerAgent(...)
    # analyzer_agent = AnalyzerAgent(...)
    # executor_agent = ExecutorAgent(...)

    # 4. Start all agents
    agents = [learning_agent]

    for agent in agents:
        await agent.start()
        logger.info(f"Started: {agent.name}")

    logger.info("\nAll agents running. Learning Agent is now:")
    logger.info("  ✓ Tracking all signals")
    logger.info("  ✓ Labeling trade outcomes")
    logger.info("  ✓ Accumulating training data")
    logger.info("  ✓ Will retrain automatically")
    logger.info("  ✓ Will deploy better models")

    # 5. Subscribe to model updates (optional)
    async def on_model_updated(event):
        data = event.data
        logger.info("\n" + "=" * 60)
        logger.info("🎉 NEW MODEL DEPLOYED!")
        logger.info("=" * 60)
        logger.info(f"Version:    {data['model_version']}")
        logger.info(f"Accuracy:   {data['accuracy']:.1%}")
        logger.info(f"Precision:  {data['precision']:.1%}")
        logger.info(f"Recall:     {data['recall']:.1%}")
        logger.info(f"F1 Score:   {data['f1_score']:.3f}")
        logger.info(f"Train Size: {data.get('training_samples', 'N/A')}")
        logger.info(f"Val Size:   {data.get('validation_samples', 'N/A')}")
        logger.info("=" * 60 + "\n")

    from agents.events import EventType, subscribe
    subscribe(EventType.SYSTEM_MODEL_UPDATED, on_model_updated)

    # 6. Run for a while (in production, this runs continuously)
    try:
        logger.info("\nSystem running... (Press Ctrl+C to stop)\n")

        # Periodic status updates
        for i in range(10):
            await asyncio.sleep(5)

            # Show stats every 5 seconds
            stats = learning_agent.get_learning_stats()
            logger.info(
                f"Status: Tracked={stats['signals_tracked']}, "
                f"Labeled={stats['outcomes_labeled']}, "
                f"Success Rate={stats['success_rate']:.1%}, "
                f"Pending={stats['new_labels_since_retrain']}"
            )

    except KeyboardInterrupt:
        logger.info("\nShutting down...")

    # 7. Cleanup
    for agent in agents:
        await agent.stop()
        logger.info(f"Stopped: {agent.name}")

    await event_bus.stop()

    # 8. Final statistics
    logger.info("\n" + "=" * 60)
    logger.info("Final Learning Statistics")
    logger.info("=" * 60)

    stats = learning_agent.get_learning_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for k, v in value.items():
                logger.info(f"  {k}: {v}")
        else:
            logger.info(f"{key}: {value}")

    logger.info("=" * 60)


async def enhanced_analyzer_example():
    """
    Example: Using Learning Agent predictions in AnalyzerAgent

    This shows how to enhance signal analysis with ML predictions.
    """

    logger.info("\n\n" + "=" * 60)
    logger.info("Enhanced Analyzer with ML Predictions")
    logger.info("=" * 60 + "\n")

    event_bus = get_event_bus()
    await event_bus.start()

    learning_agent = LearningAgent(event_bus=event_bus)
    await learning_agent.start()

    # Simulate some signals
    test_signals = [
        {
            "symbol": "AAPL",
            "direction": "long",
            "rrs": 3.5,
            "price": 180.0,
            "atr": 3.5,
            "daily_strong": True,
            "daily_weak": False,
            "volume": 80000000,
            "expected": "HIGH QUALITY"
        },
        {
            "symbol": "WEAK",
            "direction": "long",
            "rrs": 2.1,
            "price": 100.0,
            "atr": 8.0,
            "daily_strong": False,
            "daily_weak": False,
            "volume": 500000,
            "expected": "LOW QUALITY"
        },
        {
            "symbol": "NVDA",
            "direction": "short",
            "rrs": -3.8,
            "price": 500.0,
            "atr": 12.0,
            "daily_strong": False,
            "daily_weak": True,
            "volume": 90000000,
            "expected": "HIGH QUALITY"
        }
    ]

    logger.info("Testing ML-enhanced signal analysis:\n")

    for signal in test_signals:
        # Get ML prediction
        quality_score = learning_agent.predict_signal_quality(signal)

        # Decision logic
        if quality_score >= 0.70:
            decision = "✓ APPROVE"
            color = "GREEN"
        elif quality_score >= 0.60:
            decision = "⚠ CAUTION"
            color = "YELLOW"
        else:
            decision = "✗ REJECT"
            color = "RED"

        logger.info(f"{signal['symbol']:6s} | {signal['direction']:5s} | "
                   f"RRS={signal['rrs']:+5.2f} | "
                   f"ML Quality={quality_score:.1%} | "
                   f"{decision:12s} | Expected: {signal['expected']}")

    await learning_agent.stop()
    await event_bus.stop()

    logger.info("\n" + "=" * 60)


async def risk_adjusted_position_sizing():
    """
    Example: Adjust position sizing based on ML confidence

    Higher ML confidence = larger position (within risk limits)
    """

    logger.info("\n\n" + "=" * 60)
    logger.info("ML-Adjusted Position Sizing")
    logger.info("=" * 60 + "\n")

    event_bus = get_event_bus()
    await event_bus.start()

    learning_agent = LearningAgent(event_bus=event_bus)
    await learning_agent.start()

    # Base position parameters
    account_size = 100000
    risk_per_trade = 0.01  # 1%
    max_position_value = 20000  # $20k max

    signals = [
        {"symbol": "HIGH", "rrs": 3.8, "price": 100, "atr": 2.0, "daily_strong": True, "daily_weak": False, "direction": "long", "volume": 10000000},
        {"symbol": "MED", "rrs": 2.8, "price": 100, "atr": 3.0, "daily_strong": True, "daily_weak": False, "direction": "long", "volume": 5000000},
        {"symbol": "LOW", "rrs": 2.2, "price": 100, "atr": 5.0, "daily_strong": False, "daily_weak": False, "direction": "long", "volume": 1000000},
    ]

    logger.info(f"Account: ${account_size:,}")
    logger.info(f"Base Risk: {risk_per_trade:.1%} per trade")
    logger.info(f"Max Position: ${max_position_value:,}\n")

    for signal in signals:
        # Get ML confidence
        confidence = learning_agent.predict_signal_quality(signal)

        # Adjust risk based on confidence
        # Low confidence (50%) = 0.5x position
        # High confidence (100%) = 1.0x position
        confidence_multiplier = 0.5 + (confidence * 0.5)

        # Calculate position
        risk_amount = account_size * risk_per_trade * confidence_multiplier
        stop_distance = signal['atr']
        shares = int(risk_amount / stop_distance)
        position_value = shares * signal['price']

        # Cap at max position
        if position_value > max_position_value:
            shares = int(max_position_value / signal['price'])
            position_value = shares * signal['price']

        logger.info(f"{signal['symbol']:6s} | "
                   f"Confidence={confidence:.1%} | "
                   f"Multiplier={confidence_multiplier:.2f}x | "
                   f"Shares={shares:4d} | "
                   f"Value=${position_value:7,.0f}")

    await learning_agent.stop()
    await event_bus.stop()

    logger.info("\n" + "=" * 60)


def main():
    """Run all examples"""

    # Example 1: Basic integration
    # asyncio.run(run_trading_system_with_learning())

    # Example 2: ML-enhanced analysis
    asyncio.run(enhanced_analyzer_example())

    # Example 3: ML-adjusted position sizing
    asyncio.run(risk_adjusted_position_sizing())

    logger.info("\n✓ All examples complete!")


if __name__ == "__main__":
    main()

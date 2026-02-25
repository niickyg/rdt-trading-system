"""
Test and demonstrate the Learning Agent functionality
"""

import asyncio
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.learning_agent import LearningAgent
from agents.events import EventBus, EventType
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")


async def simulate_trading_session():
    """Simulate a trading session to test the learning agent"""

    # Initialize event bus
    event_bus = EventBus()
    await event_bus.start()

    # Create learning agent
    learning_agent = LearningAgent(
        event_bus=event_bus,
        model_dir=Path.home() / ".rdt-trading" / "test_models",
        retrain_threshold=10,  # Retrain after 10 trades
        retrain_interval_days=1,  # Or daily for testing
        lookback_months=6,
        min_improvement=0.01  # 1% improvement threshold
    )

    # Start the agent
    await learning_agent.start()

    logger.info("Learning Agent started - simulating trades...")

    # Simulate some signals and outcomes
    test_scenarios = [
        # Scenario 1: Strong RS signal that succeeds
        {
            "signal": {
                "symbol": "AAPL",
                "direction": "long",
                "rrs": 3.5,
                "price": 180.0,
                "atr": 3.5,
                "daily_strong": True,
                "daily_weak": False,
                "ema3": 178.0,
                "ema8": 175.0,
                "spy_pct_change": 0.5,
                "stock_pct_change": 2.0,
                "volume": 80000000
            },
            "position": {
                "symbol": "AAPL",
                "direction": "long",
                "entry_price": 180.0,
                "stop_price": 176.5,  # 1 ATR stop
                "target_price": 187.0,  # 2R target
                "shares": 100
            },
            "outcome": {
                "symbol": "AAPL",
                "direction": "long",
                "entry_price": 180.0,
                "exit_price": 187.5,  # Reached 2R+
                "shares": 100,
                "pnl": 750.0,
                "reason": "target_hit"
            }
        },
        # Scenario 2: Weak signal that fails
        {
            "signal": {
                "symbol": "TSLA",
                "direction": "long",
                "rrs": 2.2,  # Weaker RRS
                "price": 250.0,
                "atr": 8.0,
                "daily_strong": False,  # No daily confirmation
                "daily_weak": False,
                "ema3": 248.0,
                "ema8": 246.0,
                "spy_pct_change": -0.3,
                "stock_pct_change": 1.5,
                "volume": 120000000
            },
            "position": {
                "symbol": "TSLA",
                "direction": "long",
                "entry_price": 250.0,
                "stop_price": 242.0,
                "target_price": 266.0,
                "shares": 50
            },
            "outcome": {
                "symbol": "TSLA",
                "direction": "long",
                "entry_price": 250.0,
                "exit_price": 242.0,  # Hit stop
                "shares": 50,
                "pnl": -400.0,
                "reason": "stop_loss"
            }
        },
        # Scenario 3: Strong RW signal that succeeds
        {
            "signal": {
                "symbol": "NVDA",
                "direction": "short",
                "rrs": -3.8,
                "price": 500.0,
                "atr": 12.0,
                "daily_strong": False,
                "daily_weak": True,
                "ema3": 505.0,
                "ema8": 510.0,
                "spy_pct_change": -0.8,
                "stock_pct_change": -2.5,
                "volume": 90000000
            },
            "position": {
                "symbol": "NVDA",
                "direction": "short",
                "entry_price": 500.0,
                "stop_price": 512.0,
                "target_price": 476.0,
                "shares": 30
            },
            "outcome": {
                "symbol": "NVDA",
                "direction": "short",
                "entry_price": 500.0,
                "exit_price": 474.0,  # Reached 2R+
                "shares": 30,
                "pnl": 780.0,
                "reason": "target_hit"
            }
        },
        # Scenario 4: Medium signal - marginal success
        {
            "signal": {
                "symbol": "MSFT",
                "direction": "long",
                "rrs": 2.8,
                "price": 400.0,
                "atr": 6.0,
                "daily_strong": True,
                "daily_weak": False,
                "ema3": 398.0,
                "ema8": 395.0,
                "spy_pct_change": 0.2,
                "stock_pct_change": 1.8,
                "volume": 70000000
            },
            "position": {
                "symbol": "MSFT",
                "direction": "long",
                "entry_price": 400.0,
                "stop_price": 394.0,
                "target_price": 412.0,
                "shares": 75
            },
            "outcome": {
                "symbol": "MSFT",
                "direction": "long",
                "entry_price": 400.0,
                "exit_price": 413.0,  # Slightly over 2R
                "shares": 75,
                "pnl": 975.0,
                "reason": "target_hit"
            }
        },
        # Scenario 5: High volatility failure
        {
            "signal": {
                "symbol": "AMD",
                "direction": "long",
                "rrs": 3.2,
                "price": 150.0,
                "atr": 12.0,  # High volatility (8%)
                "daily_strong": True,
                "daily_weak": False,
                "ema3": 148.0,
                "ema8": 145.0,
                "spy_pct_change": 0.6,
                "stock_pct_change": 2.8,
                "volume": 100000000
            },
            "position": {
                "symbol": "AMD",
                "direction": "long",
                "entry_price": 150.0,
                "stop_price": 138.0,
                "target_price": 174.0,
                "shares": 60
            },
            "outcome": {
                "symbol": "AMD",
                "direction": "long",
                "entry_price": 150.0,
                "exit_price": 138.0,  # Hit stop
                "shares": 60,
                "pnl": -720.0,
                "reason": "stop_loss"
            }
        }
    ]

    # Process each scenario
    for i, scenario in enumerate(test_scenarios, 1):
        logger.info(f"\n--- Scenario {i}: {scenario['signal']['symbol']} ---")

        # 1. Signal found
        await event_bus.publish({
            "event_type": EventType.SIGNAL_FOUND,
            "data": scenario["signal"],
            "source": "test"
        })
        await asyncio.sleep(0.1)

        # 2. Position opened
        await event_bus.publish({
            "event_type": EventType.POSITION_OPENED,
            "data": scenario["position"],
            "source": "test"
        })
        await asyncio.sleep(0.1)

        # 3. Position closed (outcome)
        await event_bus.publish({
            "event_type": EventType.POSITION_CLOSED,
            "data": scenario["outcome"],
            "source": "test"
        })
        await asyncio.sleep(0.5)

    # Add more trades to trigger retraining
    logger.info("\n\n=== Adding more trades to trigger retraining ===\n")

    for j in range(6):
        # Create variations of the scenarios
        base_scenario = test_scenarios[j % len(test_scenarios)]
        symbol = f"TEST{j}"

        signal_data = {**base_scenario["signal"], "symbol": symbol}
        position_data = {**base_scenario["position"], "symbol": symbol}
        outcome_data = {**base_scenario["outcome"], "symbol": symbol}

        # Emit events
        await event_bus.publish({
            "event_type": EventType.SIGNAL_FOUND,
            "data": signal_data,
            "source": "test"
        })
        await asyncio.sleep(0.05)

        await event_bus.publish({
            "event_type": EventType.POSITION_OPENED,
            "data": position_data,
            "source": "test"
        })
        await asyncio.sleep(0.05)

        await event_bus.publish({
            "event_type": EventType.POSITION_CLOSED,
            "data": outcome_data,
            "source": "test"
        })
        await asyncio.sleep(0.2)

    # Wait for processing
    await asyncio.sleep(2)

    # Get statistics
    logger.info("\n\n=== Learning Agent Statistics ===")
    stats = learning_agent.get_learning_stats()
    for key, value in stats.items():
        logger.info(f"{key}: {value}")

    # Test prediction
    logger.info("\n\n=== Testing Signal Quality Prediction ===")

    test_signal = {
        "symbol": "GOOGL",
        "direction": "long",
        "rrs": 3.5,
        "price": 140.0,
        "atr": 3.0,
        "daily_strong": True,
        "daily_weak": False,
        "spy_pct_change": 0.5,
        "stock_pct_change": 2.0,
        "volume": 50000000
    }

    quality_score = learning_agent.predict_signal_quality(test_signal)
    logger.info(f"Signal quality prediction for GOOGL: {quality_score:.3f} (probability of success)")

    # Get model info
    logger.info("\n\n=== Model Information ===")
    model_info = learning_agent.get_model_info()
    for key, value in model_info.items():
        logger.info(f"{key}: {value}")

    # Get agent metrics
    logger.info("\n\n=== Agent Metrics ===")
    metrics = learning_agent.get_metrics()
    for key, value in metrics.items():
        if key != "custom_metrics":
            logger.info(f"{key}: {value}")

    logger.info("\nCustom Metrics:")
    for key, value in metrics.get("custom_metrics", {}).items():
        logger.info(f"  {key}: {value}")

    # Stop the agent
    await learning_agent.stop()
    await event_bus.stop()

    logger.info("\n\n=== Test Complete ===")


async def test_event_handling():
    """Test basic event handling"""
    logger.info("\n\n=== Testing Event Handling ===\n")

    event_bus = EventBus()
    await event_bus.start()

    learning_agent = LearningAgent(
        event_bus=event_bus,
        model_dir=Path.home() / ".rdt-trading" / "test_models"
    )

    await learning_agent.start()

    # Test single signal -> position -> close cycle
    signal = {
        "symbol": "TEST",
        "direction": "long",
        "rrs": 3.0,
        "price": 100.0,
        "atr": 2.0,
        "daily_strong": True,
        "daily_weak": False,
        "volume": 1000000
    }

    logger.info("Publishing SIGNAL_FOUND event...")
    await event_bus.publish({
        "event_type": EventType.SIGNAL_FOUND,
        "data": signal,
        "source": "test"
    })

    await asyncio.sleep(0.5)

    logger.info("Publishing POSITION_OPENED event...")
    await event_bus.publish({
        "event_type": EventType.POSITION_OPENED,
        "data": {
            "symbol": "TEST",
            "direction": "long",
            "entry_price": 100.0,
            "stop_price": 98.0,
            "target_price": 104.0,
            "shares": 100
        },
        "source": "test"
    })

    await asyncio.sleep(0.5)

    logger.info("Publishing POSITION_CLOSED event...")
    await event_bus.publish({
        "event_type": EventType.POSITION_CLOSED,
        "data": {
            "symbol": "TEST",
            "direction": "long",
            "entry_price": 100.0,
            "exit_price": 104.5,
            "shares": 100,
            "pnl": 450.0,
            "reason": "target_hit"
        },
        "source": "test"
    })

    await asyncio.sleep(1)

    stats = learning_agent.get_learning_stats()
    logger.info(f"\nStats after test: {stats}")

    await learning_agent.stop()
    await event_bus.stop()


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Learning Agent Test Suite")
    logger.info("=" * 60)

    # Run event handling test
    asyncio.run(test_event_handling())

    # Run full trading simulation
    asyncio.run(simulate_trading_session())

    logger.info("\n" + "=" * 60)
    logger.info("All tests complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

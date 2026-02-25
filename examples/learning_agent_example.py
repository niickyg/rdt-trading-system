"""
Example: Learning Agent Usage

Demonstrates how the Learning Agent continuously improves the trading system.
"""

import asyncio
from datetime import datetime
from loguru import logger

from agents.events import get_event_bus, EventType
from agents.learning_agent import LearningAgent


async def main():
    """Demonstrate Learning Agent functionality"""

    logger.info("=== Learning Agent Example ===")

    # Initialize event bus
    event_bus = get_event_bus()
    await event_bus.start()

    # Create Learning Agent
    learning_agent = LearningAgent(
        event_bus=event_bus,
        retrain_threshold=10,  # Retrain after 10 new trades (for demo)
        retrain_interval_days=7,
        lookback_months=6,
        min_improvement=0.02
    )

    # Start agent
    await learning_agent.start()

    # Simulate some trading signals and outcomes
    logger.info("\n--- Simulating Trading Signals ---")

    # Signal 1: Strong signal that succeeds
    signal_1 = {
        "symbol": "AAPL",
        "direction": "long",
        "rrs": 3.5,
        "price": 150.0,
        "atr": 3.0,
        "volume": 1000000,
        "daily_strong": True,
        "daily_weak": False,
        "spy_pct_change": 0.5,
        "stock_pct_change": 1.2,
        "timestamp": datetime.now().isoformat()
    }

    await event_bus.publish(
        event_bus._event_queue if hasattr(event_bus, '_event_queue') else None,
        EventType.SIGNAL_FOUND,
        signal_1
    )

    # Simulate position opened
    await asyncio.sleep(0.1)
    await learning_agent.publish(EventType.POSITION_OPENED, {
        "symbol": "AAPL",
        "direction": "long",
        "shares": 100,
        "entry_price": 150.0,
        "stop_price": 147.0,
        "target_price": 156.0,
        "rrs": 3.5
    })

    # Simulate successful position close (reached 2R)
    await asyncio.sleep(0.1)
    await learning_agent.publish(EventType.POSITION_CLOSED, {
        "symbol": "AAPL",
        "direction": "long",
        "entry_price": 150.0,
        "exit_price": 156.0,  # Hit target
        "shares": 100,
        "pnl": 600.0,
        "reason": "take_profit"
    })

    # Signal 2: Weak signal that fails
    signal_2 = {
        "symbol": "TSLA",
        "direction": "short",
        "rrs": -2.1,
        "price": 200.0,
        "atr": 8.0,
        "volume": 2000000,
        "daily_strong": False,
        "daily_weak": True,
        "spy_pct_change": -0.3,
        "stock_pct_change": -0.8,
        "timestamp": datetime.now().isoformat()
    }

    await asyncio.sleep(0.1)
    await learning_agent.publish(EventType.SIGNAL_FOUND, signal_2)

    await asyncio.sleep(0.1)
    await learning_agent.publish(EventType.POSITION_OPENED, {
        "symbol": "TSLA",
        "direction": "short",
        "shares": 50,
        "entry_price": 200.0,
        "stop_price": 208.0,
        "target_price": 184.0,
        "rrs": -2.1
    })

    # Simulate failed position close (hit stop)
    await asyncio.sleep(0.1)
    await learning_agent.publish(EventType.POSITION_CLOSED, {
        "symbol": "TSLA",
        "direction": "short",
        "entry_price": 200.0,
        "exit_price": 208.0,  # Hit stop
        "shares": 50,
        "pnl": -400.0,
        "reason": "stop_loss"
    })

    # More signals to trigger retraining...
    logger.info("\n--- Generating more signals to trigger retraining ---")

    for i in range(8):
        await asyncio.sleep(0.05)

        # Alternate between success and failure
        success = i % 2 == 0

        symbol = f"SYM{i}"
        direction = "long" if i % 2 == 0 else "short"
        rrs = 3.0 if success else 2.2
        entry = 100.0
        stop = 97.0 if direction == "long" else 103.0
        target = 106.0 if direction == "long" else 94.0
        exit_price = target if success else stop

        # Signal
        signal = {
            "symbol": symbol,
            "direction": direction,
            "rrs": rrs if direction == "long" else -rrs,
            "price": entry,
            "atr": 3.0,
            "volume": 500000,
            "daily_strong": direction == "long",
            "daily_weak": direction == "short",
            "spy_pct_change": 0.5,
            "stock_pct_change": 0.8,
        }

        await learning_agent.publish(EventType.SIGNAL_FOUND, signal)
        await asyncio.sleep(0.05)

        await learning_agent.publish(EventType.POSITION_OPENED, {
            "symbol": symbol,
            "direction": direction,
            "shares": 100,
            "entry_price": entry,
            "stop_price": stop,
            "target_price": target,
            "rrs": rrs
        })
        await asyncio.sleep(0.05)

        pnl = 600.0 if success else -300.0
        await learning_agent.publish(EventType.POSITION_CLOSED, {
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry,
            "exit_price": exit_price,
            "shares": 100,
            "pnl": pnl,
            "reason": "take_profit" if success else "stop_loss"
        })

    # Wait for processing
    await asyncio.sleep(1.0)

    # Check learning stats
    logger.info("\n--- Learning Agent Statistics ---")
    stats = learning_agent.get_learning_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    # Test signal quality prediction
    logger.info("\n--- Testing Signal Quality Prediction ---")

    test_signal = {
        "symbol": "TEST",
        "direction": "long",
        "rrs": 3.8,
        "price": 100.0,
        "atr": 2.5,
        "volume": 1000000,
        "daily_strong": True,
        "daily_weak": False,
        "spy_pct_change": 0.7,
        "stock_pct_change": 1.5,
    }

    quality = learning_agent.predict_signal_quality(test_signal)
    logger.info(f"Signal quality prediction: {quality:.3f}")

    # Get model info
    logger.info("\n--- Model Information ---")
    model_info = learning_agent.get_model_info()
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")

    # Get agent metrics
    logger.info("\n--- Agent Metrics ---")
    metrics = learning_agent.get_metrics()
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")

    # Cleanup
    logger.info("\n--- Shutting Down ---")
    await learning_agent.stop()
    await event_bus.stop()

    logger.info("Example complete!")


if __name__ == "__main__":
    asyncio.run(main())

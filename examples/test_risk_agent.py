"""
Test Risk Agent
Demonstrates the Risk Agent functionality and event-driven risk management
"""

import asyncio
from datetime import datetime
from loguru import logger

from agents import (
    EventBus, EventType, RiskAgent
)
from risk.risk_manager import RiskManager
from risk.models import RiskLimits


async def test_risk_agent():
    """Test Risk Agent with various scenarios"""

    logger.info("=== Risk Agent Test ===")

    # Initialize components
    event_bus = EventBus()
    await event_bus.start()

    # Configure risk limits
    risk_limits = RiskLimits(
        max_risk_per_trade=0.01,      # 1% per trade
        max_position_size=0.10,        # 10% position size
        min_risk_reward=2.0,           # 2:1 R/R
        max_daily_loss=0.03,           # 3% daily loss limit
        max_open_positions=5,
        max_drawdown=0.10              # 10% max drawdown
    )

    # Create risk manager
    risk_manager = RiskManager(
        account_size=100000.0,
        risk_limits=risk_limits
    )

    # Create risk agent
    risk_agent = RiskAgent(
        risk_manager=risk_manager,
        event_bus=event_bus
    )

    # Start agent
    await risk_agent.start()

    logger.info(f"\n{'='*60}")
    logger.info("Test 1: Valid Trade - Should Pass")
    logger.info('='*60)

    # Test 1: Valid trade
    await event_bus.publish(Event(
        event_type=EventType.SETUP_VALID,
        data={
            "symbol": "AAPL",
            "direction": "long",
            "entry_price": 150.00,
            "atr": 3.50,
            "timestamp": datetime.now().isoformat()
        },
        source="test"
    ))

    await asyncio.sleep(0.5)

    logger.info(f"\n{'='*60}")
    logger.info("Test 2: Multiple Positions - Test Exposure Limits")
    logger.info('='*60)

    # Test 2: Multiple positions
    symbols = ["MSFT", "GOOGL", "TSLA"]
    for symbol in symbols:
        await event_bus.publish(Event(
            event_type=EventType.SETUP_VALID,
            data={
                "symbol": symbol,
                "direction": "long",
                "entry_price": 200.00,
                "atr": 5.00,
                "timestamp": datetime.now().isoformat()
            },
            source="test"
        ))
        await asyncio.sleep(0.3)

    logger.info(f"\n{'='*60}")
    logger.info("Test 3: Simulate Position Open")
    logger.info('='*60)

    # Test 3: Simulate position opening
    await event_bus.publish(Event(
        event_type=EventType.POSITION_OPENED,
        data={
            "symbol": "AAPL",
            "direction": "long",
            "shares": 100,
            "entry_price": 150.00,
            "stop_price": 145.00,
            "target_price": 160.00,
            "timestamp": datetime.now().isoformat()
        },
        source="test"
    ))

    await asyncio.sleep(0.5)

    # Check status
    status = risk_agent.get_portfolio_status()
    logger.info(f"\nPortfolio Status:")
    logger.info(f"  Balance: ${status['balance']:,.2f}")
    logger.info(f"  Open Positions: {status['open_positions']}")
    logger.info(f"  Exposure: {status['exposure_percent']:.2f}%")
    logger.info(f"  Daily P&L: ${status['daily_pnl']:,.2f}")

    logger.info(f"\n{'='*60}")
    logger.info("Test 4: Simulate Losing Trade - Test Daily Loss Limit")
    logger.info('='*60)

    # Test 4: Simulate losing trade
    await event_bus.publish(Event(
        event_type=EventType.POSITION_CLOSED,
        data={
            "symbol": "AAPL",
            "direction": "long",
            "shares": 100,
            "entry_price": 150.00,
            "exit_price": 145.00,
            "pnl": -500.00,
            "is_day_trade": False,
            "timestamp": datetime.now().isoformat()
        },
        source="test"
    ))

    await asyncio.sleep(0.5)

    status = risk_agent.get_portfolio_status()
    logger.info(f"\nPortfolio Status After Loss:")
    logger.info(f"  Daily P&L: ${status['daily_pnl']:,.2f} ({status['daily_pnl_percent']:.2f}%)")
    logger.info(f"  Balance: ${status['balance']:,.2f}")

    logger.info(f"\n{'='*60}")
    logger.info("Test 5: Trading Allowed Check")
    logger.info('='*60)

    # Test 5: Check if trading is allowed
    allowed, reason = risk_agent.is_trade_allowed()
    logger.info(f"\nTrading Allowed: {allowed}")
    logger.info(f"Reason: {reason}")

    logger.info(f"\n{'='*60}")
    logger.info("Test 6: Simulate Multiple Losses - Trigger Circuit Breaker")
    logger.info('='*60)

    # Test 6: Multiple losses to trigger circuit breaker
    for i in range(3):
        await event_bus.publish(Event(
            event_type=EventType.POSITION_OPENED,
            data={
                "symbol": f"TEST{i}",
                "direction": "long",
                "shares": 100,
                "entry_price": 100.00,
                "stop_price": 95.00,
                "target_price": 110.00,
                "timestamp": datetime.now().isoformat()
            },
            source="test"
        ))
        await asyncio.sleep(0.2)

        await event_bus.publish(Event(
            event_type=EventType.POSITION_CLOSED,
            data={
                "symbol": f"TEST{i}",
                "direction": "long",
                "shares": 100,
                "entry_price": 100.00,
                "exit_price": 95.00,
                "pnl": -500.00,
                "is_day_trade": False,
                "timestamp": datetime.now().isoformat()
            },
            source="test"
        ))
        await asyncio.sleep(0.2)

    status = risk_agent.get_portfolio_status()
    logger.info(f"\nPortfolio Status After Multiple Losses:")
    logger.info(f"  Daily P&L: ${status['daily_pnl']:,.2f} ({status['daily_pnl_percent']:.2f}%)")
    logger.info(f"  Circuit Breaker: {status['circuit_breaker']}")
    logger.info(f"  Trading Halted: {status['trading_halted']}")

    allowed, reason = risk_agent.is_trade_allowed()
    logger.info(f"\nTrading Allowed: {allowed}")
    logger.info(f"Reason: {reason}")

    logger.info(f"\n{'='*60}")
    logger.info("Test 7: Risk Metrics")
    logger.info('='*60)

    # Test 7: Get risk metrics
    metrics = risk_agent.get_risk_metrics()
    logger.info(f"\nRisk Metrics:")
    logger.info(f"  Daily Trades: {metrics.daily_trades}")
    logger.info(f"  Wins: {metrics.daily_wins} | Losses: {metrics.daily_losses}")
    logger.info(f"  Open Positions: {metrics.open_positions}")
    logger.info(f"  Total Exposure: ${metrics.total_exposure:,.2f} ({metrics.exposure_percent:.2f}%)")
    logger.info(f"  Current Drawdown: ${metrics.current_drawdown:,.2f} ({metrics.current_drawdown_percent:.2f}%)")
    logger.info(f"  Max Drawdown: ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_percent:.2f}%)")

    logger.info(f"\n{'='*60}")
    logger.info("Test 8: Agent Metrics")
    logger.info('='*60)

    # Test 8: Get agent metrics
    agent_metrics = risk_agent.get_metrics()
    logger.info(f"\nAgent Metrics:")
    logger.info(f"  State: {agent_metrics['state']}")
    logger.info(f"  Events Processed: {agent_metrics['events_processed']}")
    logger.info(f"  Events Published: {agent_metrics['events_published']}")
    logger.info(f"  Trades Approved: {agent_metrics['trades_approved']}")
    logger.info(f"  Trades Rejected: {agent_metrics['trades_rejected']}")
    logger.info(f"  Approval Rate: {agent_metrics['approval_rate']}%")
    logger.info(f"  Risk Alerts Sent: {agent_metrics['risk_alerts_sent']}")
    logger.info(f"  Circuit Breakers: {agent_metrics['circuit_breakers_triggered']}")

    logger.info(f"\n{'='*60}")
    logger.info("Test 9: Reset Circuit Breaker")
    logger.info('='*60)

    # Test 9: Reset circuit breaker
    await risk_agent.reset_circuit_breaker()
    allowed, reason = risk_agent.is_trade_allowed()
    logger.info(f"\nTrading Allowed: {allowed}")
    logger.info(f"Reason: {reason}")

    logger.info(f"\n{'='*60}")
    logger.info("Test 10: Market Close - Generate Report")
    logger.info('='*60)

    # Test 10: Market close
    await event_bus.publish(Event(
        event_type=EventType.MARKET_CLOSE,
        data={"timestamp": datetime.now().isoformat()},
        source="test"
    ))

    await asyncio.sleep(0.5)

    # Cleanup
    await risk_agent.stop()
    await event_bus.stop()

    logger.info(f"\n{'='*60}")
    logger.info("Risk Agent Test Complete")
    logger.info('='*60)


async def test_dynamic_position_sizing():
    """Test dynamic position sizing based on portfolio state"""

    logger.info("\n\n=== Dynamic Position Sizing Test ===")

    event_bus = EventBus()
    await event_bus.start()

    risk_manager = RiskManager(
        account_size=100000.0,
        risk_limits=RiskLimits()
    )

    risk_agent = RiskAgent(
        risk_manager=risk_manager,
        event_bus=event_bus
    )

    await risk_agent.start()

    logger.info("\n--- Scenario 1: Clean Portfolio (Full Position Size) ---")

    await event_bus.publish(Event(
        event_type=EventType.SETUP_VALID,
        data={
            "symbol": "AAPL",
            "direction": "long",
            "entry_price": 150.00,
            "atr": 3.50,
        },
        source="test"
    ))

    await asyncio.sleep(0.5)

    logger.info("\n--- Scenario 2: After Opening Multiple Positions (Reduced Size) ---")

    # Open 3 positions
    for i, symbol in enumerate(["MSFT", "GOOGL", "TSLA"]):
        await event_bus.publish(Event(
            event_type=EventType.POSITION_OPENED,
            data={
                "symbol": symbol,
                "direction": "long",
                "shares": 100,
                "entry_price": 200.00,
                "stop_price": 195.00,
                "target_price": 210.00,
            },
            source="test"
        ))
        await asyncio.sleep(0.2)

    # Now try to add another position
    await event_bus.publish(Event(
        event_type=EventType.SETUP_VALID,
        data={
            "symbol": "NVDA",
            "direction": "long",
            "entry_price": 500.00,
            "atr": 15.00,
        },
        source="test"
    ))

    await asyncio.sleep(0.5)

    logger.info("\n--- Scenario 3: After Losses (Further Reduced Size) ---")

    # Close one position with a loss
    await event_bus.publish(Event(
        event_type=EventType.POSITION_CLOSED,
        data={
            "symbol": "MSFT",
            "pnl": -1000.00,
            "exit_price": 190.00,
            "is_day_trade": False,
        },
        source="test"
    ))

    await asyncio.sleep(0.3)

    # Try another trade
    await event_bus.publish(Event(
        event_type=EventType.SETUP_VALID,
        data={
            "symbol": "AMD",
            "direction": "long",
            "entry_price": 120.00,
            "atr": 4.00,
        },
        source="test"
    ))

    await asyncio.sleep(0.5)

    status = risk_agent.get_portfolio_status()
    logger.info(f"\nFinal Portfolio Status:")
    logger.info(f"  Balance: ${status['balance']:,.2f}")
    logger.info(f"  Daily P&L: ${status['daily_pnl']:,.2f}")
    logger.info(f"  Open Positions: {status['open_positions']}")
    logger.info(f"  Exposure: {status['exposure_percent']:.2f}%")

    await risk_agent.stop()
    await event_bus.stop()

    logger.info("\nDynamic Position Sizing Test Complete")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # Run tests
    asyncio.run(test_risk_agent())
    asyncio.run(test_dynamic_position_sizing())

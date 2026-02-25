"""
Example: Using the Enhanced Risk Agent

Demonstrates the Risk Agent's capabilities including:
- Real-time risk monitoring
- Sector concentration tracking
- Position correlation limits
- Dynamic position sizing
- Circuit breakers
"""

import asyncio
from datetime import datetime

from agents import RiskAgent, EventBus, EventType
from risk.risk_manager import RiskManager
from risk.models import RiskLimits


async def main():
    """Demonstrate Risk Agent functionality"""

    # Initialize event bus
    event_bus = EventBus()
    await event_bus.start()

    # Configure risk limits
    risk_limits = RiskLimits(
        max_risk_per_trade=0.01,      # 1% per trade
        max_position_size=0.10,        # 10% max position
        max_daily_loss=0.03,           # 3% daily loss limit
        max_open_positions=5,          # Max 5 positions
        max_sector_exposure=0.25,      # 25% max per sector
        max_correlated_positions=3,    # Max 3 correlated positions
        min_risk_reward=2.0            # Min 2:1 R/R
    )

    # Create risk manager and agent
    risk_manager = RiskManager(
        account_size=100000,
        risk_limits=risk_limits
    )

    risk_agent = RiskAgent(
        risk_manager=risk_manager,
        event_bus=event_bus
    )

    # Start the agent
    await risk_agent.start()

    print("\n" + "="*60)
    print("RISK AGENT DEMONSTRATION")
    print("="*60)

    # Example 1: Validate a trade setup with sector info
    print("\n1. Validating trade setup with sector tracking...")

    setup = {
        "symbol": "AAPL",
        "direction": "long",
        "entry_price": 150.00,
        "atr": 3.50,
        "sector": "Technology",
        "correlated_symbols": ["MSFT", "GOOGL", "META"]
    }

    await risk_agent.validate_setup(setup)

    # Wait for processing
    await asyncio.sleep(0.1)

    # Example 2: Simulate opening a position
    print("\n2. Simulating position opened...")

    position = {
        "symbol": "AAPL",
        "direction": "long",
        "shares": 100,
        "entry_price": 150.00,
        "stop_price": 145.00,
        "target_price": 160.00,
        "sector": "Technology",
        "correlated_symbols": ["MSFT", "GOOGL", "META"]
    }

    await risk_agent.track_position_opened(position)

    # Example 3: Check portfolio status
    print("\n3. Portfolio Status:")
    status = risk_agent.get_portfolio_status()

    print(f"   Balance: ${status['balance']:,.2f}")
    print(f"   Open Positions: {status['open_positions']}")
    print(f"   Exposure: {status['exposure_percent']:.2f}%")
    print(f"   Portfolio VaR: ${status['portfolio_var']:,.2f}")
    print(f"   Trading Halted: {status['trading_halted']}")
    print(f"   Sectors: {status['total_sectors']}")

    if status['sector_exposure']:
        print("\n   Sector Exposure:")
        for sector, pct in status['sector_exposure'].items():
            print(f"     {sector}: {pct:.2f}%")

    # Example 4: Test sector concentration limit
    print("\n4. Testing sector concentration limit...")

    # Try to add another tech stock that would exceed sector limit
    setup2 = {
        "symbol": "MSFT",
        "direction": "long",
        "entry_price": 300.00,
        "atr": 5.00,
        "sector": "Technology",
        "correlated_symbols": ["AAPL", "GOOGL", "META"]
    }

    await risk_agent.validate_setup(setup2)

    # Wait for processing
    await asyncio.sleep(0.1)

    # Example 5: Test correlation limit
    print("\n5. Testing correlation limit...")

    setup3 = {
        "symbol": "GOOGL",
        "direction": "long",
        "entry_price": 140.00,
        "atr": 4.00,
        "sector": "Technology",
        "correlated_symbols": ["AAPL", "MSFT", "META"]
    }

    await risk_agent.validate_setup(setup3)

    # Wait for processing
    await asyncio.sleep(0.1)

    # Example 6: Get risk metrics
    print("\n6. Risk Metrics:")
    metrics = risk_agent.get_metrics()

    print(f"   Trades Approved: {metrics['trades_approved']}")
    print(f"   Trades Rejected: {metrics['trades_rejected']}")
    print(f"   Approval Rate: {metrics['approval_rate']:.1f}%")
    print(f"   Risk Alerts: {metrics['risk_alerts_sent']}")
    print(f"   Sector Violations: {metrics['sector_violations']}")
    print(f"   Correlation Violations: {metrics['correlation_violations']}")

    # Example 7: Simulate position close
    print("\n7. Closing position...")

    close_position = {
        "symbol": "AAPL",
        "pnl": 500.00,
        "exit_price": 155.00,
        "is_day_trade": False
    }

    await risk_agent.track_position_closed(close_position)

    # Final status
    print("\n8. Final Portfolio Status:")
    final_status = risk_agent.get_portfolio_status()

    print(f"   Balance: ${final_status['balance']:,.2f}")
    print(f"   Daily P&L: ${final_status['daily_pnl']:,.2f} ({final_status['daily_pnl_percent']:.2f}%)")
    print(f"   Open Positions: {final_status['open_positions']}")
    print(f"   Sectors: {final_status['total_sectors']}")

    # Stop the agent
    await risk_agent.stop()
    await event_bus.stop()

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60 + "\n")

    print("Key Features Demonstrated:")
    print("  ✓ Sector concentration tracking and limits")
    print("  ✓ Position correlation monitoring")
    print("  ✓ Dynamic position sizing based on portfolio state")
    print("  ✓ Real-time risk metrics and alerts")
    print("  ✓ Comprehensive trade validation")
    print("  ✓ Portfolio exposure management")


if __name__ == "__main__":
    asyncio.run(main())

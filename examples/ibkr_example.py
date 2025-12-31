#!/usr/bin/env python3
"""
Example: Using IBKR broker with RDT Trading System
Shows how to auto-trade scanner signals using Interactive Brokers
"""

from brokers import get_broker, OrderSide, OrderType
from loguru import logger


def example_manual_trading():
    """Example 1: Manual trading through IBKR paper account"""
    print("\n" + "="*60)
    print("Example 1: Manual Trading")
    print("="*60)

    # Create IBKR broker (paper trading)
    broker = get_broker("ibkr", port=7497, paper_trading=True)

    # Connect
    if not broker.connect():
        print("Failed to connect. Make sure TWS is running!")
        return

    # Get account info
    account = broker.get_account()
    print(f"\nAccount: {account.account_id}")
    print(f"Cash: ${account.cash_available:,.2f}")
    print(f"Buying Power: ${account.buying_power:,.2f}")

    # Get current quote
    quote = broker.get_quote("T")
    print(f"\nT Quote: ${quote.last:.2f}")

    # Place a market order
    order = broker.place_order(
        symbol="T",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET
    )
    print(f"\nOrder placed: {order.order_id}")
    print(f"Status: {order.status}")

    # Get positions
    positions = broker.get_positions()
    for symbol, pos in positions.items():
        print(f"\n{symbol}: {pos.quantity} shares @ ${pos.avg_cost:.2f}")
        print(f"  P&L: ${pos.unrealized_pnl:.2f} ({pos.unrealized_pnl_pct:.1f}%)")

    broker.disconnect()


def example_scanner_integration():
    """Example 2: Auto-trade scanner signals"""
    print("\n" + "="*60)
    print("Example 2: Scanner Integration")
    print("="*60)

    # Scanner signals (from your RDT scanner)
    signals = [
        {"symbol": "T", "price": 24.78, "rrs": 2.42, "atr": 0.37},
        {"symbol": "DVN", "price": 36.19, "rrs": 1.84, "atr": 0.99},
        {"symbol": "OXY", "price": 40.40, "rrs": 1.99, "atr": 0.87}
    ]

    broker = get_broker("ibkr", port=7497, paper_trading=True)
    broker.connect()

    print("\nProcessing scanner signals...")

    for signal in signals:
        symbol = signal["symbol"]
        rrs = signal["rrs"]
        price = signal["price"]

        # Only trade strong signals (RRS >= 1.75)
        if rrs >= 1.75:
            # Calculate position size (3% risk)
            account = broker.get_account()
            risk_amount = account.total_value * 0.03  # 3% aggressive risk
            atr = signal["atr"]
            stop_distance = 0.75 * atr  # Tight stop
            shares = int(risk_amount / stop_distance)

            # Place order
            print(f"\n🎯 {symbol}: RRS {rrs:.2f} - BUY {shares} shares @ ${price:.2f}")

            order = broker.place_order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=shares,
                order_type=OrderType.MARKET
            )

            print(f"  Order {order.order_id}: {order.status}")

    broker.disconnect()


def example_options_trading():
    """Example 3: Trading options (requires ib_insync Option contract)"""
    print("\n" + "="*60)
    print("Example 3: Options Trading")
    print("="*60)
    print("Note: Options require using ib_insync directly")
    print("See: /home/user0/rdt-trading-system/brokers/ibkr/client.py")

    # This would require extending the AbstractBroker interface
    # For now, use ib_insync directly for options:
    from ib_insync import IB, Option

    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)

    # Buy T Jan 17 2025 $25 calls
    call = Option('T', '20250117', 25, 'C', 'SMART')
    ib.qualifyContracts(call)

    from ib_insync import MarketOrder
    order = MarketOrder('BUY', 35)
    trade = ib.placeOrder(call, order)

    print(f"\n✅ Options order placed: {trade.order.orderId}")

    ib.disconnect()


if __name__ == "__main__":
    print("\nIBKR Integration Examples")
    print("Make sure TWS is running before executing!")

    # Run examples
    # example_manual_trading()
    # example_scanner_integration()
    # example_options_trading()

    print("\n" + "="*60)
    print("Usage:")
    print("  Uncomment the example you want to run")
    print("  Make sure TWS/Gateway is running on port 7497")
    print("="*60)

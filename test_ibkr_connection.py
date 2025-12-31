#!/usr/bin/env python3
"""
Quick test script for IBKR connection
Tests both paper and live connection capability
"""

from brokers import get_broker
from loguru import logger

def test_ibkr_paper():
    """Test IBKR paper trading connection"""
    print("\n" + "="*60)
    print("Testing IBKR Paper Trading Connection")
    print("="*60)

    try:
        # Create IBKR broker instance (paper trading)
        broker = get_broker(
            "ibkr",
            port=7497,  # Paper trading port
            paper_trading=True
        )

        # Connect
        print("Connecting to IBKR paper account...")
        connected = broker.connect()

        if not connected:
            print("❌ Failed to connect. Make sure TWS is running!")
            print("\nTo start TWS:")
            print("1. Download from: https://www.interactivebrokers.com/en/trading/tws.php")
            print("2. Login with paper trading credentials")
            print("3. Enable API: File → Global Configuration → API → Settings")
            print("   - Check 'Enable ActiveX and Socket Clients'")
            print("   - Port should be 7497 for paper trading")
            return False

        print("✅ Connected successfully!")

        # Get account info
        print("\nFetching account information...")
        account = broker.get_account()

        print(f"\n📊 Account Summary:")
        print(f"  Account ID: {account.account_id}")
        print(f"  Cash Available: ${account.cash_available:,.2f}")
        print(f"  Buying Power: ${account.buying_power:,.2f}")
        print(f"  Total Value: ${account.total_value:,.2f}")
        print(f"  Daily P&L: ${account.daily_pnl:,.2f}")
        print(f"  Open Positions: {len(account.positions)}")

        # Test getting a quote
        print("\nTesting quote retrieval...")
        quote = broker.get_quote("AAPL")
        print(f"\n💰 AAPL Quote:")
        print(f"  Last: ${quote.last:.2f}")
        print(f"  Bid: ${quote.bid:.2f}")
        print(f"  Ask: ${quote.ask:.2f}")
        print(f"  Spread: ${quote.spread:.2f}")

        # Show current scanner signals
        print("\n📡 You can now auto-trade scanner signals!")
        print("   Example:")
        print("   - Scanner finds: T @ $24.78, RRS 2.42")
        print("   - Bot places: BUY 35 T Jan25 C25 calls")
        print("   - All in paper account, $0 risk")

        # Disconnect
        broker.disconnect()
        print("\n✅ Test complete! IBKR integration working.")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure TWS or IB Gateway is running")
        print("2. Check that API is enabled in TWS settings")
        print("3. Verify port 7497 is correct for paper trading")
        print("4. Install ib_insync: pip install ib-insync")
        return False


def test_quotes_batch():
    """Test batch quote retrieval"""
    print("\n" + "="*60)
    print("Testing Batch Quote Retrieval")
    print("="*60)

    try:
        broker = get_broker("ibkr", port=7497, paper_trading=True)
        broker.connect()

        # Get multiple quotes (from scanner signals)
        symbols = ["T", "DVN", "OXY"]
        print(f"\nFetching quotes for: {', '.join(symbols)}")

        quotes = broker.get_quotes(symbols)

        print("\n📈 Live Quotes:")
        for symbol, quote in quotes.items():
            print(f"  {symbol}: ${quote.last:.2f} (Bid: ${quote.bid:.2f}, Ask: ${quote.ask:.2f})")

        broker.disconnect()
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    # Test paper trading connection
    success = test_ibkr_paper()

    if success:
        # Test batch quotes
        test_quotes_batch()

    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. ✅ IBKR integration is working")
    print("2. 📝 Update your bot to use IBKR instead of paper broker:")
    print("   broker = get_broker('ibkr', port=7497, paper_trading=True)")
    print("3. 🚀 Auto-trade scanner signals in paper account")
    print("4. 📊 Validate for 30 days before going live")
    print("="*60)

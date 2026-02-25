#!/usr/bin/env python3
"""
IBKR Connection Test Script

This script tests connectivity and functionality of the Interactive Brokers
integration. Run this to verify your IBKR TWS/Gateway setup before using
the trading system.

Usage:
    python -m brokers.ibkr.test_connection
    # or
    python brokers/ibkr/test_connection.py

    # Options:
    python -m brokers.ibkr.test_connection --port 7497  # Paper trading
    python -m brokers.ibkr.test_connection --port 7496  # Live trading
    python -m brokers.ibkr.test_connection --test-order # Test order placement

Requirements:
    1. TWS or IB Gateway running with API enabled
    2. API port configured (default 7497 for paper, 7496 for live)
    3. Socket client enabled in TWS/Gateway settings
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger


class IBKRConnectionTester:
    """Test IBKR API connectivity and functionality."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 99,  # Use high client ID for testing
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.results = {}
        self.client = None

    def print_header(self, title: str):
        """Print a formatted header."""
        print("\n" + "=" * 60)
        print(f"  {title}")
        print("=" * 60)

    def print_result(self, test_name: str, passed: bool, message: str = ""):
        """Print a test result."""
        status = "[PASS]" if passed else "[FAIL]"
        color_code = "\033[92m" if passed else "\033[91m"
        reset = "\033[0m"
        print(f"  {color_code}{status}{reset} {test_name}")
        if message:
            print(f"         {message}")
        self.results[test_name] = passed

    def test_ib_insync_import(self) -> Tuple[bool, str]:
        """Test if ib_insync library is installed."""
        self.print_header("Library Check")

        try:
            import ib_insync
            version = getattr(ib_insync, '__version__', 'unknown')
            self.print_result(
                "ib_insync installed",
                True,
                f"Version: {version}"
            )
            return True, f"Version {version}"
        except ImportError:
            self.print_result(
                "ib_insync installed",
                False,
                "Install with: pip install ib_insync"
            )
            return False, "ib_insync not installed"

    def test_environment_variables(self) -> Tuple[bool, str]:
        """Test if environment variables are set (optional)."""
        self.print_header("Environment Variables Check")

        host = os.environ.get("IBKR_HOST", "127.0.0.1")
        port = os.environ.get("IBKR_PORT", str(self.port))
        client_id = os.environ.get("IBKR_CLIENT_ID", str(self.client_id))
        paper = os.environ.get("IBKR_PAPER_TRADING", "true")

        self.print_result("IBKR_HOST", True, host)
        self.print_result("IBKR_PORT", True, port)
        self.print_result("IBKR_CLIENT_ID", True, client_id)
        self.print_result("IBKR_PAPER_TRADING", True, paper)

        # Update instance variables if env vars are set
        self.host = host
        self.port = int(port)
        self.client_id = int(client_id)

        return True, "Environment configured"

    def test_connection(self) -> Tuple[bool, str]:
        """Test connection to TWS/Gateway."""
        self.print_header("Connection Test")

        try:
            from brokers.ibkr.client import IBKRClient

            print(f"  Connecting to {self.host}:{self.port}...")

            self.client = IBKRClient(
                host=self.host,
                port=self.port,
                client_id=self.client_id,
                paper_trading=(self.port == 7497 or self.port == 4002),
                timeout=15,
                auto_reconnect=False,
            )

            if self.client.connect():
                self.print_result(
                    "Connection established",
                    True,
                    f"Connected to {self.client._config.connection_type}"
                )
                return True, "Connected"
            else:
                self.print_result(
                    "Connection established",
                    False,
                    "connect() returned False"
                )
                return False, "Connection failed"

        except Exception as e:
            self.print_result(
                "Connection established",
                False,
                str(e)
            )
            print("\n  Troubleshooting tips:")
            print("  1. Ensure TWS or IB Gateway is running")
            print("  2. Enable API connections in TWS: Edit > Global Configuration > API > Settings")
            print("  3. Check 'Enable ActiveX and Socket Clients'")
            print(f"  4. Verify Socket port is {self.port}")
            print("  5. Add 127.0.0.1 to 'Trusted IPs' or disable 'Reject connections from other hosts'")
            return False, str(e)

    def test_account_info(self) -> Tuple[bool, str]:
        """Test account information retrieval."""
        self.print_header("Account Information Test")

        if not self.client or not self.client.is_connected:
            self.print_result("Account info", False, "Not connected")
            return False, "Not connected"

        try:
            account = self.client.get_account()

            self.print_result("Get account info", True)
            print(f"\n  Account Details:")
            print(f"    Account ID: {account.account_id}")
            print(f"    Buying Power: ${account.buying_power:,.2f}")
            print(f"    Cash: ${account.cash:,.2f}")
            print(f"    Equity: ${account.equity:,.2f}")
            print(f"    Day Trades Remaining: {account.day_trades_remaining}")
            print(f"    Positions Value: ${account.positions_value:,.2f}")
            print(f"    Daily P&L: ${account.daily_pnl:,.2f}")

            return True, "Account info retrieved"

        except Exception as e:
            self.print_result("Get account info", False, str(e))
            return False, str(e)

    def test_positions(self) -> Tuple[bool, str]:
        """Test position retrieval."""
        self.print_header("Positions Test")

        if not self.client or not self.client.is_connected:
            self.print_result("Positions", False, "Not connected")
            return False, "Not connected"

        try:
            positions = self.client.get_positions()

            self.print_result(
                "Get positions",
                True,
                f"Found {len(positions)} position(s)"
            )

            if positions:
                print(f"\n  Current Positions:")
                for symbol, pos in positions.items():
                    print(f"    {symbol}: {pos.quantity} shares @ ${pos.avg_cost:.2f}")
                    print(f"      Current: ${pos.current_price:.2f}")
                    print(f"      P&L: ${pos.unrealized_pnl:.2f} ({pos.unrealized_pnl_pct:.2f}%)")
            else:
                print("\n  No open positions")

            return True, "Positions retrieved"

        except Exception as e:
            self.print_result("Positions", False, str(e))
            return False, str(e)

    def test_quote(self, symbol: str = "SPY") -> Tuple[bool, str]:
        """Test market data quote retrieval."""
        self.print_header("Market Data Test")

        if not self.client or not self.client.is_connected:
            self.print_result("Quote", False, "Not connected")
            return False, "Not connected"

        try:
            print(f"  Requesting quote for {symbol}...")
            quote = self.client.get_quote(symbol)

            self.print_result("Get quote", True, symbol)
            print(f"\n  Quote for {symbol}:")
            print(f"    Last: ${quote.last:.2f}")
            print(f"    Bid: ${quote.bid:.2f} x {quote.bid_size}")
            print(f"    Ask: ${quote.ask:.2f} x {quote.ask_size}")
            print(f"    Spread: ${quote.spread:.2f} ({quote.spread_pct:.3f}%)")
            print(f"    Volume: {quote.volume:,}")
            print(f"    High: ${quote.high:.2f}")
            print(f"    Low: ${quote.low:.2f}")
            print(f"    Timestamp: {quote.timestamp}")

            return True, "Quote retrieved"

        except Exception as e:
            self.print_result("Quote", False, str(e))
            return False, str(e)

    def test_multiple_quotes(self, symbols: list = None) -> Tuple[bool, str]:
        """Test batch quote retrieval."""
        self.print_header("Batch Quote Test")

        if not self.client or not self.client.is_connected:
            self.print_result("Batch quotes", False, "Not connected")
            return False, "Not connected"

        symbols = symbols or ["AAPL", "MSFT", "GOOGL"]

        try:
            print(f"  Requesting quotes for {', '.join(symbols)}...")
            quotes = self.client.get_quotes(symbols)

            self.print_result(
                "Get batch quotes",
                True,
                f"Retrieved {len(quotes)}/{len(symbols)} quotes"
            )

            print(f"\n  Quotes:")
            for symbol, quote in quotes.items():
                print(f"    {symbol}: ${quote.last:.2f} (bid: ${quote.bid:.2f}, ask: ${quote.ask:.2f})")

            return True, "Batch quotes retrieved"

        except Exception as e:
            self.print_result("Batch quotes", False, str(e))
            return False, str(e)

    def test_order_placement(self, symbol: str = "SPY") -> Tuple[bool, str]:
        """Test order placement (PAPER TRADING ONLY)."""
        self.print_header("Order Placement Test")

        if not self.client or not self.client.is_connected:
            self.print_result("Order placement", False, "Not connected")
            return False, "Not connected"

        # Safety check - only allow on paper trading
        if not self.client.paper_trading:
            self.print_result(
                "Order placement",
                False,
                "SAFETY: Order test only allowed on paper trading accounts"
            )
            print("\n  For safety, order placement tests are disabled for live accounts.")
            print("  Use --port 7497 to test with paper trading.")
            return False, "Live trading - test skipped"

        try:
            from brokers.broker_interface import OrderSide, OrderType

            # Get current price for limit order
            quote = self.client.get_quote(symbol)
            limit_price = round(quote.bid - 1.00, 2)  # $1 below bid (won't fill)

            print(f"  Placing test limit order:")
            print(f"    Symbol: {symbol}")
            print(f"    Side: BUY")
            print(f"    Quantity: 1")
            print(f"    Limit: ${limit_price}")

            # Place a limit order that won't fill (below market)
            order = self.client.place_order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=1,
                order_type=OrderType.LIMIT,
                price=limit_price,
                time_in_force="DAY"
            )

            self.print_result("Place order", True, f"Order ID: {order.order_id}")
            print(f"\n  Order Details:")
            print(f"    Order ID: {order.order_id}")
            print(f"    Status: {order.status.value}")

            # Wait briefly and check status
            time.sleep(1)
            updated_order = self.client.get_order_status(order.order_id)
            if updated_order:
                print(f"    Updated Status: {updated_order.status.value}")

            # Cancel the test order
            print("\n  Cancelling test order...")
            cancelled = self.client.cancel_order(order.order_id)

            if cancelled:
                self.print_result("Cancel order", True, f"Order {order.order_id} cancelled")
            else:
                self.print_result("Cancel order", False, "Failed to cancel")

            return True, "Order test completed"

        except Exception as e:
            self.print_result("Order placement", False, str(e))
            return False, str(e)

    def test_open_orders(self) -> Tuple[bool, str]:
        """Test open orders retrieval."""
        self.print_header("Open Orders Test")

        if not self.client or not self.client.is_connected:
            self.print_result("Open orders", False, "Not connected")
            return False, "Not connected"

        try:
            orders = self.client.get_open_orders()

            self.print_result(
                "Get open orders",
                True,
                f"Found {len(orders)} open order(s)"
            )

            if orders:
                print(f"\n  Open Orders:")
                for order in orders:
                    print(f"    {order.order_id}: {order.side.value} {order.quantity} {order.symbol}")
                    print(f"      Type: {order.order_type.value}, Status: {order.status.value}")
            else:
                print("\n  No open orders")

            return True, "Open orders retrieved"

        except Exception as e:
            self.print_result("Open orders", False, str(e))
            return False, str(e)

    def run_all_tests(self, test_order: bool = False) -> bool:
        """Run all connectivity tests."""
        print("\n")
        print("=" * 60)
        print("  IBKR API CONNECTION TEST")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Target: {self.host}:{self.port}")
        print("=" * 60)

        # Test library installation
        lib_ok, _ = self.test_ib_insync_import()
        if not lib_ok:
            self.print_summary()
            return False

        # Test environment
        self.test_environment_variables()

        # Test connection
        conn_ok, _ = self.test_connection()
        if not conn_ok:
            self.print_summary()
            return False

        # Test account info
        self.test_account_info()

        # Test positions
        self.test_positions()

        # Test open orders
        self.test_open_orders()

        # Test quotes
        self.test_quote()
        self.test_multiple_quotes()

        # Test order placement (optional, paper only)
        if test_order:
            self.test_order_placement()

        # Disconnect
        if self.client:
            print("\n  Disconnecting...")
            self.client.disconnect()

        # Summary
        self.print_summary()

        return all(self.results.values())

    def print_summary(self):
        """Print test summary."""
        self.print_header("Test Summary")

        passed = sum(1 for v in self.results.values() if v)
        total = len(self.results)

        print(f"\n  Tests Passed: {passed}/{total}")

        if passed == total:
            print("\n  \033[92mAll tests passed! IBKR integration is ready.\033[0m")
        else:
            failed = [k for k, v in self.results.items() if not v]
            print(f"\n  \033[91mFailed tests: {', '.join(failed)}\033[0m")
            print("\n  Troubleshooting:")
            print("  1. Ensure TWS or IB Gateway is running")
            print("  2. Verify API connections are enabled")
            print("  3. Check firewall settings")
            print("  4. Verify port number (7497=paper, 7496=live)")
            print("  5. Make sure no other clients are using the same client_id")

        print()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test IBKR TWS/Gateway connectivity"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="TWS/Gateway host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7497,
        help="TWS/Gateway port (default: 7497 for paper)"
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=99,
        help="Client ID (default: 99)"
    )
    parser.add_argument(
        "--test-order",
        action="store_true",
        help="Test order placement (paper trading only)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Load environment variables from .env if present
    env_file = project_root / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            pass

    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    # Run tests
    tester = IBKRConnectionTester(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
    )

    success = tester.run_all_tests(test_order=args.test_order)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

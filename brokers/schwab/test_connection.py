#!/usr/bin/env python3
"""
Schwab API Connection Test Script

This script tests connectivity and functionality of the Schwab broker integration.
Run this to verify your Schwab API setup before using the trading system.

Usage:
    python -m brokers.schwab.test_connection
    # or
    python brokers/schwab/test_connection.py

Requirements:
    1. Schwab developer account (https://developer.schwab.com/)
    2. App credentials (SCHWAB_APP_KEY, SCHWAB_APP_SECRET)
    3. Complete OAuth flow at least once
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger


class SchwabConnectionTester:
    """Test Schwab API connectivity and functionality."""

    def __init__(self):
        self.results = {}
        self.client = None
        self.auth = None

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

    def test_environment_variables(self) -> Tuple[bool, str]:
        """Test if required environment variables are set."""
        self.print_header("Environment Variables Check")

        app_key = os.environ.get("SCHWAB_APP_KEY", "")
        app_secret = os.environ.get("SCHWAB_APP_SECRET", "")
        callback_url = os.environ.get("SCHWAB_CALLBACK_URL", "https://localhost:8080")

        # Check if using placeholder values
        if app_key in ("", "your_app_key_here"):
            self.print_result(
                "SCHWAB_APP_KEY",
                False,
                "Not set. Get from https://developer.schwab.com/"
            )
            return False, "App key not configured"

        if app_secret in ("", "your_app_secret_here"):
            self.print_result(
                "SCHWAB_APP_SECRET",
                False,
                "Not set. Get from https://developer.schwab.com/"
            )
            return False, "App secret not configured"

        self.print_result("SCHWAB_APP_KEY", True, f"Set ({app_key[:8]}...)")
        self.print_result("SCHWAB_APP_SECRET", True, "Set (hidden)")
        self.print_result("SCHWAB_CALLBACK_URL", True, callback_url)

        return True, "Environment variables configured"

    def test_token_file(self) -> Tuple[bool, str]:
        """Test if OAuth token file exists and is valid."""
        self.print_header("OAuth Token Check")

        from brokers.schwab.auth import SchwabAuth

        app_key = os.environ.get("SCHWAB_APP_KEY", "")
        app_secret = os.environ.get("SCHWAB_APP_SECRET", "")

        if not app_key or not app_secret:
            self.print_result("Token file", False, "Credentials not set")
            return False, "Credentials required first"

        self.auth = SchwabAuth(app_key, app_secret)

        if not self.auth.token_path.exists():
            self.print_result(
                "Token file exists",
                False,
                f"Not found at {self.auth.token_path}"
            )
            print("\n  To authenticate, run:")
            print("    from brokers.schwab import SchwabAuth")
            print("    auth = SchwabAuth(app_key, app_secret)")
            print("    auth.authorize_interactive()")
            return False, "Token file not found - OAuth required"

        self.print_result("Token file exists", True, str(self.auth.token_path))

        # Check token validity
        if self.auth._token is None:
            self.print_result("Token loaded", False, "Failed to parse token file")
            return False, "Token file corrupt"

        self.print_result("Token loaded", True)

        # Check expiration
        if self.auth._token.is_expired():
            expires_at = self.auth._token.expires_at
            self.print_result(
                "Token valid",
                False,
                f"Expired at {expires_at.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            return False, "Token expired - will attempt refresh"
        else:
            expires_at = self.auth._token.expires_at
            remaining = expires_at - datetime.now()
            self.print_result(
                "Token valid",
                True,
                f"Expires in {remaining.seconds // 60} minutes"
            )

        return True, "Token valid"

    def test_token_refresh(self) -> Tuple[bool, str]:
        """Test token refresh functionality."""
        self.print_header("Token Refresh Test")

        if not self.auth:
            self.print_result("Token refresh", False, "Auth not initialized")
            return False, "Auth required"

        if self.auth._token is None:
            self.print_result("Token refresh", False, "No token to refresh")
            return False, "No token"

        # Only refresh if expired or close to expiry
        if self.auth._token.is_expired():
            print("  Attempting token refresh...")
            try:
                success = self.auth.refresh_access_token()
                if success:
                    self.print_result(
                        "Token refresh",
                        True,
                        "Successfully refreshed access token"
                    )
                    return True, "Token refreshed"
                else:
                    self.print_result(
                        "Token refresh",
                        False,
                        "Refresh failed - re-authorization required"
                    )
                    return False, "Refresh failed"
            except Exception as e:
                self.print_result("Token refresh", False, str(e))
                return False, str(e)
        else:
            self.print_result(
                "Token refresh",
                True,
                "Token still valid, no refresh needed"
            )
            return True, "Token still valid"

    def test_api_connectivity(self) -> Tuple[bool, str]:
        """Test basic API connectivity."""
        self.print_header("API Connectivity Test")

        if not self.auth or not self.auth.is_authenticated:
            self.print_result(
                "API connection",
                False,
                "Not authenticated"
            )
            return False, "Not authenticated"

        from brokers.schwab.client import SchwabClient

        app_key = os.environ.get("SCHWAB_APP_KEY", "")
        app_secret = os.environ.get("SCHWAB_APP_SECRET", "")

        try:
            self.client = SchwabClient(app_key, app_secret)

            # Test connection without interactive OAuth
            if self.client.auth.is_authenticated:
                self.print_result("Authentication", True, "Token valid")
            else:
                self.print_result(
                    "Authentication",
                    False,
                    "Not authenticated"
                )
                return False, "Not authenticated"

            return True, "Connected"

        except Exception as e:
            self.print_result("API connection", False, str(e))
            return False, str(e)

    def test_account_info(self) -> Tuple[bool, str]:
        """Test account information retrieval."""
        self.print_header("Account Information Test")

        if not self.client:
            self.print_result("Account info", False, "Client not initialized")
            return False, "Client required"

        try:
            # Fetch accounts first
            accounts = self.client._fetch_accounts()

            if not accounts:
                self.print_result(
                    "Fetch accounts",
                    False,
                    "No accounts returned - check API permissions"
                )
                return False, "No accounts"

            self.print_result(
                "Fetch accounts",
                True,
                f"Found {len(accounts)} account(s)"
            )

            # Set account number
            if accounts:
                self.client._accounts = accounts
                self.client._account_number = accounts[0].get("accountNumber")

            # Get account info
            account_info = self.client.get_account_info()

            if account_info:
                self.print_result("Get account info", True)
                print(f"\n  Account Details:")
                print(f"    Account ID: {account_info.account_id}")
                print(f"    Buying Power: ${account_info.buying_power:,.2f}")
                print(f"    Cash: ${account_info.cash:,.2f}")
                print(f"    Equity: ${account_info.equity:,.2f}")
                print(f"    Day Trades Remaining: {account_info.day_trades_remaining}")
                print(f"    Pattern Day Trader: {account_info.pattern_day_trader}")
                return True, "Account info retrieved"
            else:
                self.print_result(
                    "Get account info",
                    False,
                    "No account info returned"
                )
                return False, "No account info"

        except Exception as e:
            self.print_result("Account info", False, str(e))
            return False, str(e)

    def test_positions(self) -> Tuple[bool, str]:
        """Test position retrieval."""
        self.print_header("Positions Test")

        if not self.client:
            self.print_result("Positions", False, "Client not initialized")
            return False, "Client required"

        try:
            positions = self.client.get_positions()

            if positions is None:
                self.print_result(
                    "Get positions",
                    False,
                    "Failed to retrieve positions"
                )
                return False, "Failed"

            self.print_result(
                "Get positions",
                True,
                f"Found {len(positions)} position(s)"
            )

            if positions:
                print(f"\n  Current Positions:")
                for symbol, pos in positions.items():
                    print(f"    {symbol}: {pos.quantity} shares @ ${pos.avg_price:.2f}")
                    print(f"      Current: ${pos.current_price:.2f}")
                    print(f"      P&L: ${pos.unrealized_pnl:.2f} ({pos.unrealized_pnl_pct:.2f}%)")

            return True, "Positions retrieved"

        except Exception as e:
            self.print_result("Positions", False, str(e))
            return False, str(e)

    def test_quote(self) -> Tuple[bool, str]:
        """Test market data quote retrieval."""
        self.print_header("Market Data Test")

        if not self.client:
            self.print_result("Quote", False, "Client not initialized")
            return False, "Client required"

        test_symbol = "SPY"

        try:
            quote = self.client.get_quote(test_symbol)

            if quote:
                self.print_result("Get quote", True, f"{test_symbol}")
                print(f"\n  Quote for {test_symbol}:")
                print(f"    Last: ${quote.last:.2f}")
                print(f"    Bid: ${quote.bid:.2f}")
                print(f"    Ask: ${quote.ask:.2f}")
                print(f"    Volume: {quote.volume:,}")
                print(f"    Timestamp: {quote.timestamp}")
                return True, "Quote retrieved"
            else:
                self.print_result(
                    "Get quote",
                    False,
                    f"No quote for {test_symbol}"
                )
                return False, "No quote"

        except Exception as e:
            self.print_result("Quote", False, str(e))
            return False, str(e)

    def run_all_tests(self) -> bool:
        """Run all connectivity tests."""
        print("\n")
        print("=" * 60)
        print("  SCHWAB API CONNECTION TEST")
        print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 60)

        # Test environment
        env_ok, _ = self.test_environment_variables()

        if not env_ok:
            self.print_summary()
            return False

        # Test token
        token_ok, token_msg = self.test_token_file()

        if not token_ok:
            if "expired" in token_msg.lower():
                # Try refresh
                refresh_ok, _ = self.test_token_refresh()
                if not refresh_ok:
                    self.print_summary()
                    return False
            else:
                self.print_summary()
                return False

        # Test API connectivity
        api_ok, _ = self.test_api_connectivity()

        if not api_ok:
            self.print_summary()
            return False

        # Test account info
        self.test_account_info()

        # Test positions
        self.test_positions()

        # Test quotes
        self.test_quote()

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
            print("\n  \033[92mAll tests passed! Schwab integration is ready.\033[0m")
        else:
            failed = [k for k, v in self.results.items() if not v]
            print(f"\n  \033[91mFailed tests: {', '.join(failed)}\033[0m")
            print("\n  Troubleshooting:")
            print("  1. Ensure SCHWAB_APP_KEY and SCHWAB_APP_SECRET are set in .env")
            print("  2. Complete OAuth authorization if token is missing")
            print("  3. Check Schwab developer console for API errors")
            print("  4. Verify your app has required API permissions")

        print()


def run_interactive_auth():
    """Run interactive OAuth authorization."""
    print("\n" + "=" * 60)
    print("  SCHWAB OAUTH AUTHORIZATION")
    print("=" * 60)

    app_key = os.environ.get("SCHWAB_APP_KEY", "")
    app_secret = os.environ.get("SCHWAB_APP_SECRET", "")

    if not app_key or not app_secret or \
       app_key == "your_app_key_here" or app_secret == "your_app_secret_here":
        print("\n  ERROR: SCHWAB_APP_KEY and SCHWAB_APP_SECRET must be set.")
        print("  Get credentials from: https://developer.schwab.com/")
        print("\n  Set them in your .env file:")
        print("    SCHWAB_APP_KEY=your_actual_key")
        print("    SCHWAB_APP_SECRET=your_actual_secret")
        return False

    from brokers.schwab.auth import SchwabAuth

    auth = SchwabAuth(app_key, app_secret)
    return auth.authorize_interactive()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Schwab API connectivity"
    )
    parser.add_argument(
        "--auth",
        action="store_true",
        help="Run interactive OAuth authorization"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Load environment variables from .env
    env_file = project_root / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)

    if args.auth:
        success = run_interactive_auth()
        sys.exit(0 if success else 1)

    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    # Run tests
    tester = SchwabConnectionTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

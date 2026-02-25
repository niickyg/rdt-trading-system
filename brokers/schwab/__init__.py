"""
Schwab broker integration for the RDT Trading System.

This module provides integration with Charles Schwab's Trader API
for live trading with a Schwab brokerage account.

Requirements:
    1. Schwab developer account (https://developer.schwab.com/)
    2. Registered app with API credentials
    3. thinkorswim enabled on your Schwab account

Usage:
    from brokers.schwab import SchwabClient, SchwabAuth

    # Initialize client
    client = SchwabClient(
        app_key="your_app_key",
        app_secret="your_app_secret"
    )

    # Connect (will prompt for OAuth if needed)
    if client.connect():
        account = client.get_account()
        print(f"Account: {account.account_id}")
        print(f"Buying Power: ${account.buying_power:,.2f}")

Test Connection:
    python -m brokers.schwab.test_connection

Interactive Auth:
    python -m brokers.schwab.test_connection --auth
"""

from brokers.schwab.auth import SchwabAuth, TokenData
from brokers.schwab.client import SchwabClient

__all__ = [
    "SchwabAuth",
    "SchwabClient",
    "TokenData",
]

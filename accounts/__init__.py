"""
Multiple Trading Account Support for the RDT Trading System.

Provides:
- AccountManager: Manage multiple trading accounts across different brokers
- TradingAccount: Database model for account storage
- PortfolioAggregator: Aggregate positions and performance across accounts
- Credential encryption for secure storage

Usage:
    from accounts import AccountManager, PortfolioAggregator

    # Create account manager
    manager = AccountManager(user_id="user123")

    # Add accounts
    manager.add_account(
        name="Main Trading",
        broker_type="schwab",
        credentials={"app_key": "...", "app_secret": "..."}
    )
    manager.add_account(
        name="Paper Trading",
        broker_type="paper",
        credentials={"initial_balance": 50000}
    )

    # Get accounts
    accounts = manager.get_all_accounts()
    default = manager.get_default_account()

    # Aggregate portfolio
    aggregator = PortfolioAggregator(manager)
    total_value = aggregator.get_total_value()
    positions = aggregator.get_combined_positions()
"""

from accounts.models import TradingAccount, BrokerType
from accounts.account_manager import AccountManager
from accounts.portfolio_aggregator import PortfolioAggregator

__all__ = [
    "TradingAccount",
    "BrokerType",
    "AccountManager",
    "PortfolioAggregator",
]

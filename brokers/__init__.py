"""
Broker Module
Provides unified interface for trading across different brokers
"""

from typing import Optional
from loguru import logger

from brokers.base import (
    AbstractBroker, Order, Position, Quote, AccountInfo,
    OrderSide, OrderType, OrderStatus
)
from brokers.paper.simulator import PaperBroker


def get_broker(
    broker_type: str = "paper",
    **kwargs
) -> AbstractBroker:
    """
    Factory function to create broker instances

    Args:
        broker_type: "paper", "schwab", or "ibkr"
        **kwargs: Broker-specific configuration
            Paper broker:
                - initial_balance: Starting balance (default 25000)
                - slippage_pct: Slippage percentage (default 0.001)
            Schwab broker:
                - app_key: Schwab API app key
                - app_secret: Schwab API app secret
                - callback_url: OAuth callback URL
                - account_number: Specific account to use
                - token_path: Path to store OAuth tokens
            IBKR broker:
                - host: TWS/Gateway host (default: 127.0.0.1)
                - port: TWS/Gateway port (7497 for paper, 7496 for live)
                - client_id: Unique client ID (default: 1)
                - paper_trading: Whether using paper account (default: True)

    Returns:
        AbstractBroker instance

    Example:
        # Paper trading
        broker = get_broker("paper", initial_balance=50000)

        # Live trading with Schwab
        broker = get_broker(
            "schwab",
            app_key="your_key",
            app_secret="your_secret"
        )

        # IBKR paper trading
        broker = get_broker("ibkr", port=7497, paper_trading=True)

        # IBKR live trading
        broker = get_broker("ibkr", port=7496, paper_trading=False)
    """
    broker_type = broker_type.lower()

    if broker_type == "paper":
        logger.info("Creating Paper trading broker")
        return PaperBroker(
            initial_balance=kwargs.get("initial_balance", 25000.0),
            slippage_pct=kwargs.get("slippage_pct", 0.001)
        )

    elif broker_type == "schwab":
        # Lazy import to avoid dependency issues
        from brokers.schwab.client import SchwabClient

        logger.info("Creating Schwab broker")

        app_key = kwargs.get("app_key")
        app_secret = kwargs.get("app_secret")

        if not app_key or not app_secret:
            raise ValueError("Schwab broker requires app_key and app_secret")

        return SchwabClient(
            app_key=app_key,
            app_secret=app_secret,
            callback_url=kwargs.get("callback_url", "https://localhost:8080"),
            account_number=kwargs.get("account_number"),
            token_path=kwargs.get("token_path")
        )

    elif broker_type == "ibkr":
        # Lazy import to avoid dependency issues
        from brokers.ibkr.client import IBKRClient

        logger.info("Creating IBKR broker")

        return IBKRClient(
            host=kwargs.get("host", "127.0.0.1"),
            port=kwargs.get("port", 7497),  # 7497 = paper, 7496 = live
            client_id=kwargs.get("client_id", 1),
            paper_trading=kwargs.get("paper_trading", True)
        )

    else:
        raise ValueError(f"Unknown broker type: {broker_type}. Use 'paper', 'schwab', or 'ibkr'")


__all__ = [
    "get_broker",
    "AbstractBroker",
    "PaperBroker",
    "Order",
    "Position",
    "Quote",
    "AccountInfo",
    "OrderSide",
    "OrderType",
    "OrderStatus"
]

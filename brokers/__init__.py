"""
Broker Module - RDT Trading System

Provides unified interface for trading across different brokers:
- Paper trading (simulation)
- Charles Schwab (live trading)
- Interactive Brokers (live trading)

Usage:
    from brokers import get_broker, OrderSide, OrderType

    # Paper trading (default)
    broker = get_broker("paper", initial_balance=50000)

    # Schwab live trading
    broker = get_broker("schwab", app_key="...", app_secret="...")

    # IBKR paper trading
    broker = get_broker("ibkr", port=7497, paper_trading=True)

    # Use unified interface
    broker.connect()
    account = broker.get_account()
    order = broker.place_order("AAPL", OrderSide.BUY, 10)
    positions = broker.get_positions()
    broker.disconnect()
"""

from typing import Optional
from loguru import logger

# Import broker interface and data classes
from brokers.broker_interface import (
    BrokerInterface, AbstractBroker,
    Order, Position, Quote, AccountInfo,
    OrderSide, OrderType, OrderStatus,
    BrokerError, AuthenticationError, ConnectionError,
    OrderError, InsufficientFundsError, PositionError
)

# Import paper broker
from brokers.paper_broker import PaperBroker

# Also export from paper submodule for backward compatibility
from brokers.paper.simulator import PaperBroker as PaperSimulator


def get_broker(
    broker_type: str = "paper",
    **kwargs
) -> BrokerInterface:
    """
    Factory function to create broker instances.

    Args:
        broker_type: Type of broker - "paper", "schwab", or "ibkr"
        **kwargs: Broker-specific configuration

    Broker-specific kwargs:
        Paper broker:
            initial_balance: Starting balance (default: 25000)
            slippage_pct: Slippage percentage (default: 0.001)
            commission_per_trade: Commission per trade (default: 0)
            realistic_fills: Enable slippage simulation (default: True)

        Schwab broker:
            app_key: Schwab API app key (required)
            app_secret: Schwab API app secret (required)
            callback_url: OAuth callback URL (default: https://localhost:8080)
            account_number: Specific account to use
            token_path: Path to store OAuth tokens

        IBKR broker:
            host: TWS/Gateway host (default: 127.0.0.1)
            port: TWS/Gateway port (7497=paper, 7496=live)
            client_id: Unique client ID (default: 1)
            paper_trading: Whether using paper account (default: True)
            timeout: Connection timeout in seconds (default: 20)
            auto_reconnect: Enable automatic reconnection (default: True)
            from_env: Load config from environment variables (default: False)

    Returns:
        BrokerInterface instance

    Raises:
        ValueError: If broker_type is unknown or required params missing

    Examples:
        # Paper trading with custom balance
        broker = get_broker("paper", initial_balance=100000)

        # Schwab live trading
        broker = get_broker(
            "schwab",
            app_key="your_app_key",
            app_secret="your_app_secret"
        )

        # IBKR paper trading
        broker = get_broker("ibkr", port=7497, paper_trading=True)

        # IBKR live trading
        broker = get_broker("ibkr", port=7496, paper_trading=False)
    """
    broker_type = broker_type.lower().strip()

    if broker_type == "paper":
        logger.info("Creating Paper trading broker")
        return PaperBroker(
            initial_balance=kwargs.get("initial_balance", 25000.0),
            slippage_pct=kwargs.get("slippage_pct", 0.001),
            commission_per_trade=kwargs.get("commission_per_trade", 0.0),
            realistic_fills=kwargs.get("realistic_fills", True),
            data_provider=kwargs.get("data_provider"),
        )

    elif broker_type == "schwab":
        # Lazy import to avoid dependency issues
        from brokers.schwab.client import SchwabClient

        logger.info("Creating Schwab broker")

        app_key = kwargs.get("app_key")
        app_secret = kwargs.get("app_secret")

        if not app_key or not app_secret:
            raise ValueError(
                "Schwab broker requires app_key and app_secret. "
                "Get credentials from https://developer.schwab.com/"
            )

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
        from brokers.ibkr.config import get_ibkr_config

        logger.info("Creating IBKR broker")

        # Check if using config from environment
        if kwargs.get("from_env", False):
            config = get_ibkr_config(
                host=kwargs.get("host"),
                port=kwargs.get("port"),
                client_id=kwargs.get("client_id"),
                paper_trading=kwargs.get("paper_trading"),
            )
            return IBKRClient(config=config)

        return IBKRClient(
            host=kwargs.get("host", "127.0.0.1"),
            port=kwargs.get("port", 7497),  # 7497 = paper, 7496 = live
            client_id=kwargs.get("client_id", 1),
            paper_trading=kwargs.get("paper_trading", True),
            timeout=kwargs.get("timeout", 20),
            auto_reconnect=kwargs.get("auto_reconnect", True),
        )

    else:
        raise ValueError(
            f"Unknown broker type: '{broker_type}'. "
            "Valid options: 'paper', 'schwab', 'ibkr'"
        )


def get_broker_from_config(config: dict) -> BrokerInterface:
    """
    Create broker from configuration dictionary.

    Args:
        config: Configuration with 'broker_type' and broker-specific settings

    Returns:
        BrokerInterface instance

    Example:
        config = {
            'broker_type': 'schwab',
            'app_key': 'your_key',
            'app_secret': 'your_secret',
            'paper_trading': False
        }
        broker = get_broker_from_config(config)
    """
    broker_type = config.get('broker_type', config.get('broker', 'paper'))

    # Check for paper trading override
    if config.get('paper_trading', True):
        if broker_type != 'paper':
            logger.warning(
                f"paper_trading=True overrides broker_type='{broker_type}'. "
                "Using paper broker instead."
            )
            broker_type = 'paper'

    return get_broker(broker_type, **config)


# Export all public symbols
__all__ = [
    # Factory functions
    "get_broker",
    "get_broker_from_config",

    # Base classes
    "BrokerInterface",
    "AbstractBroker",  # Alias for backward compatibility

    # Broker implementations
    "PaperBroker",
    "PaperSimulator",  # Backward compatibility

    # Data classes
    "Order",
    "Position",
    "Quote",
    "AccountInfo",

    # Enums
    "OrderSide",
    "OrderType",
    "OrderStatus",

    # Exceptions
    "BrokerError",
    "AuthenticationError",
    "ConnectionError",
    "OrderError",
    "InsufficientFundsError",
    "PositionError",
]

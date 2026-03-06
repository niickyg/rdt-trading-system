"""
IBKR Connection Configuration.

Handles configuration for Interactive Brokers TWS/Gateway connections.
Supports both environment variables and programmatic configuration.

Environment Variables:
    IBKR_HOST: TWS/Gateway host address (default: 127.0.0.1)
    IBKR_PORT: TWS/Gateway port (default: 4000)
        - 7497: TWS Paper Trading
        - 7496: TWS Live Trading
        - 4002: IB Gateway Paper Trading
        - 4001: IB Gateway Live Trading
        - 4000: IBC-managed Gateway (custom OverrideTwsApiPort)
    IBKR_CLIENT_ID: Unique client ID for this connection (default: 1)
    IBKR_PAPER_TRADING: Whether using paper trading account (default: true)
    IBKR_TIMEOUT: Connection timeout in seconds (default: 20)
    IBKR_READONLY: Read-only mode, no order placement (default: false)
    IBKR_ACCOUNT: Specific account number to use (optional)
    IBKR_MARKET_DATA_TYPE: Market data type (default: 1)
        - 1: Live (requires market data subscription)
        - 2: Frozen (last available live price)
        - 3: Delayed (15-20 min delay, free)
        - 4: Delayed-Frozen (last available delayed price)
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


# Default port mappings
PORTS = {
    "tws_paper": 7497,
    "tws_live": 7496,
    "gateway_paper": 4002,
    "gateway_live": 4001,
}


@dataclass
class IBKRConfig:
    """
    IBKR connection configuration.

    Attributes:
        host: TWS/Gateway host address
        port: TWS/Gateway port number
        client_id: Unique client ID for this connection
        paper_trading: Whether using paper trading account
        timeout: Connection timeout in seconds
        readonly: Read-only mode (no order placement)
        account: Specific account number to use
        market_data_type: Market data type (1=live, 2=frozen, 3=delayed, 4=delayed-frozen)
        auto_reconnect: Enable automatic reconnection
        max_reconnect_attempts: Maximum reconnection attempts
        reconnect_delay: Delay between reconnection attempts (seconds)
    """

    host: str = "127.0.0.1"
    port: int = 4000
    client_id: int = 1
    paper_trading: bool = True
    timeout: int = 20
    readonly: bool = False
    account: Optional[str] = None

    # Market data type: 1=live, 2=frozen, 3=delayed, 4=delayed-frozen
    market_data_type: int = 1

    # Auto-reconnect settings
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 0  # 0 = infinite retries
    reconnect_delay: float = 5.0

    # Rate limiting
    request_throttle: float = 0.05  # 50ms between requests

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure port matches paper_trading setting
        if self.paper_trading and self.port == 7496:
            logger.warning(
                "paper_trading=True but port=7496 (live). "
                "Setting port to 7497 (paper)."
            )
            self.port = 7497
        elif not self.paper_trading and self.port == 7497:
            logger.warning(
                "paper_trading=False but port=7497 (paper). "
                "Setting port to 7496 (live)."
            )
            self.port = 7496

        # Validate client_id
        if self.client_id < 0:
            raise ValueError("client_id must be non-negative")

        # Validate timeout
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

        # Validate market_data_type
        if self.market_data_type not in (1, 2, 3, 4):
            logger.warning(f"Invalid market_data_type={self.market_data_type}, defaulting to 1 (live)")
            self.market_data_type = 1

    @property
    def is_tws(self) -> bool:
        """Check if connecting to TWS (vs Gateway)."""
        return self.port in (7496, 7497)

    @property
    def is_gateway(self) -> bool:
        """Check if connecting to IB Gateway (vs TWS)."""
        return self.port in (4001, 4002)

    @property
    def connection_type(self) -> str:
        """Get human-readable connection type."""
        if self.is_tws:
            return "TWS Paper" if self.paper_trading else "TWS Live"
        elif self.is_gateway:
            return "Gateway Paper" if self.paper_trading else "Gateway Live"
        else:
            return f"Custom Port {self.port}"

    @classmethod
    def from_env(cls) -> "IBKRConfig":
        """
        Create configuration from environment variables.

        Environment Variables:
            IBKR_HOST: Host address (default: 127.0.0.1)
            IBKR_PORT: Port number (default: 4000)
            IBKR_CLIENT_ID: Client ID (default: 1)
            IBKR_PAPER_TRADING: Paper trading mode (default: true)
            IBKR_TIMEOUT: Connection timeout (default: 20)
            IBKR_READONLY: Read-only mode (default: false)
            IBKR_ACCOUNT: Specific account number (optional)
            IBKR_MARKET_DATA_TYPE: Market data type 1-4 (default: 1 = live)
            IBKR_AUTO_RECONNECT: Enable auto-reconnect (default: true)

        Returns:
            IBKRConfig instance
        """
        def parse_bool(value: str) -> bool:
            return value.lower() in ("true", "1", "yes", "on")

        host = os.environ.get("IBKR_HOST", "127.0.0.1")
        port = int(os.environ.get("IBKR_PORT", "4000"))
        client_id = int(os.environ.get("IBKR_CLIENT_ID", "1"))
        paper_trading = parse_bool(os.environ.get("IBKR_PAPER_TRADING", "true"))
        timeout = int(os.environ.get("IBKR_TIMEOUT", "20"))
        readonly = parse_bool(os.environ.get("IBKR_READONLY", "false"))
        account = os.environ.get("IBKR_ACCOUNT")
        market_data_type = int(os.environ.get("IBKR_MARKET_DATA_TYPE", "1"))
        auto_reconnect = parse_bool(os.environ.get("IBKR_AUTO_RECONNECT", "true"))
        max_reconnect_attempts = int(os.environ.get("IBKR_MAX_RECONNECT", "0"))
        reconnect_delay = float(os.environ.get("IBKR_RECONNECT_DELAY", "5.0"))

        return cls(
            host=host,
            port=port,
            client_id=client_id,
            paper_trading=paper_trading,
            timeout=timeout,
            readonly=readonly,
            account=account if account else None,
            market_data_type=market_data_type,
            auto_reconnect=auto_reconnect,
            max_reconnect_attempts=max_reconnect_attempts,
            reconnect_delay=reconnect_delay,
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "client_id": self.client_id,
            "paper_trading": self.paper_trading,
            "timeout": self.timeout,
            "readonly": self.readonly,
            "account": self.account,
            "market_data_type": self.market_data_type,
            "auto_reconnect": self.auto_reconnect,
            "max_reconnect_attempts": self.max_reconnect_attempts,
            "reconnect_delay": self.reconnect_delay,
        }

    def __str__(self) -> str:
        """String representation."""
        mdt_labels = {1: "live", 2: "frozen", 3: "delayed", 4: "delayed-frozen"}
        mdt = mdt_labels.get(self.market_data_type, f"type-{self.market_data_type}")
        return (
            f"IBKRConfig({self.connection_type} @ {self.host}:{self.port}, "
            f"client_id={self.client_id}, data={mdt})"
        )


def get_ibkr_config(
    host: Optional[str] = None,
    port: Optional[int] = None,
    client_id: Optional[int] = None,
    paper_trading: Optional[bool] = None,
    **kwargs
) -> IBKRConfig:
    """
    Get IBKR configuration with optional overrides.

    First loads from environment variables, then applies any provided overrides.

    Args:
        host: Override host address
        port: Override port number
        client_id: Override client ID
        paper_trading: Override paper trading setting
        **kwargs: Additional configuration options

    Returns:
        IBKRConfig instance

    Example:
        # Use environment defaults
        config = get_ibkr_config()

        # Override paper trading
        config = get_ibkr_config(paper_trading=False, port=7496)
    """
    # Start with environment configuration
    config = IBKRConfig.from_env()

    # Apply overrides
    if host is not None:
        config.host = host
    if port is not None:
        config.port = port
    if client_id is not None:
        config.client_id = client_id
    if paper_trading is not None:
        config.paper_trading = paper_trading

    # Apply additional kwargs
    for key, value in kwargs.items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)

    # Re-validate after overrides
    config.__post_init__()

    return config


# Convenience functions for common configurations

def tws_paper_config(client_id: int = 1) -> IBKRConfig:
    """Get configuration for TWS Paper Trading."""
    return IBKRConfig(
        host="127.0.0.1",
        port=7497,
        client_id=client_id,
        paper_trading=True,
    )


def tws_live_config(client_id: int = 1) -> IBKRConfig:
    """Get configuration for TWS Live Trading."""
    return IBKRConfig(
        host="127.0.0.1",
        port=7496,
        client_id=client_id,
        paper_trading=False,
    )


def gateway_paper_config(client_id: int = 1) -> IBKRConfig:
    """Get configuration for IB Gateway Paper Trading."""
    return IBKRConfig(
        host="127.0.0.1",
        port=4002,
        client_id=client_id,
        paper_trading=True,
    )


def gateway_live_config(client_id: int = 1) -> IBKRConfig:
    """Get configuration for IB Gateway Live Trading."""
    return IBKRConfig(
        host="127.0.0.1",
        port=4001,
        client_id=client_id,
        paper_trading=False,
    )

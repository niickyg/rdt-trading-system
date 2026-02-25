"""
Interactive Brokers integration for RDT Trading System.

Supports both paper and live trading via TWS/Gateway API using ib_insync library.

Usage:
    from brokers.ibkr import IBKRClient

    # Paper trading (default)
    client = IBKRClient(port=7497, paper_trading=True)

    # Live trading
    client = IBKRClient(port=7496, paper_trading=False)

    # Connect and trade
    if client.connect():
        account = client.get_account()
        positions = client.get_positions()
        order = client.place_order("AAPL", OrderSide.BUY, 10)
        client.disconnect()

Environment Variables:
    IBKR_HOST: TWS/Gateway host (default: 127.0.0.1)
    IBKR_PORT: TWS/Gateway port (default: 7497 for paper, 7496 for live)
    IBKR_CLIENT_ID: Unique client ID (default: 1)
    IBKR_PAPER_TRADING: Whether using paper account (default: True)
"""

from brokers.ibkr.client import IBKRClient
from brokers.ibkr.config import IBKRConfig, get_ibkr_config
from brokers.ibkr.orders import (
    convert_order_type,
    convert_order_side,
    convert_time_in_force,
    map_ibkr_order_status,
    map_ibkr_order_type,
    create_bracket_order,
    create_oco_order,
)

__all__ = [
    # Main client
    "IBKRClient",

    # Configuration
    "IBKRConfig",
    "get_ibkr_config",

    # Order utilities
    "convert_order_type",
    "convert_order_side",
    "convert_time_in_force",
    "map_ibkr_order_status",
    "map_ibkr_order_type",
    "create_bracket_order",
    "create_oco_order",
]

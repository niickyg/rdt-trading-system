"""
Interactive Brokers integration for RDT Trading System
Supports both paper and live trading via TWS/Gateway API
"""

from brokers.ibkr.client import IBKRClient

__all__ = ["IBKRClient"]

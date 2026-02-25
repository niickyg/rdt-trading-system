"""
Abstract broker interface for the RDT Trading System.

This module re-exports from broker_interface for backward compatibility.
New code should import directly from brokers.broker_interface or brokers.
"""

# Re-export everything from broker_interface for backward compatibility
from brokers.broker_interface import (
    # Base class
    BrokerInterface as AbstractBroker,
    BrokerInterface,

    # Data classes
    Order,
    Position,
    Quote,
    AccountInfo,

    # Enums
    OrderSide,
    OrderType,
    OrderStatus,

    # Exceptions
    BrokerError,
    AuthenticationError,
    ConnectionError,
    OrderError,
    InsufficientFundsError,
    PositionError,
)

__all__ = [
    "AbstractBroker",
    "BrokerInterface",
    "Order",
    "Position",
    "Quote",
    "AccountInfo",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "BrokerError",
    "AuthenticationError",
    "ConnectionError",
    "OrderError",
    "InsufficientFundsError",
    "PositionError",
]

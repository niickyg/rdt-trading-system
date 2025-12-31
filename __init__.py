"""
RDT Trading System
==================

An autonomous trading system based on the r/RealDayTrading methodology.

Features:
- Real Relative Strength (RRS) scanning
- Multi-agent autonomous trading
- Risk management with ATR-based position sizing
- Paper trading and live trading via Schwab API
- Portfolio monitoring and backtesting

Quick Start:
    from rdt_trading import run_trading_system, get_broker
    from rdt_trading.risk import RiskManager
    from rdt_trading.config import get_settings

    # Get configuration
    settings = get_settings()

    # Create broker (paper or live)
    broker = get_broker("paper", initial_balance=25000)

    # Create risk manager
    risk_manager = RiskManager(account_size=25000)

    # Run the system
    await run_trading_system(
        broker=broker,
        risk_manager=risk_manager,
        watchlist=['AAPL', 'MSFT', 'NVDA'],
        auto_trade=False  # Set True for automatic execution
    )

Components:
    - agents: Multi-agent framework (Scanner, Analyzer, Executor, Orchestrator)
    - brokers: Broker abstraction (Paper trading, Schwab API)
    - risk: Risk management (Position sizing, Daily limits, Drawdown)
    - portfolio: Position tracking and management
    - scanner: RRS signal scanning
    - shared: Indicators and utilities
    - config: Configuration management
    - data: Database and data storage

Author: RDT Trading System
License: MIT
"""

__version__ = "1.0.0"
__author__ = "RDT Trading System"

# Core imports for convenience
from config import get_settings, Settings
from brokers import get_broker, AbstractBroker
from risk import RiskManager, PositionSizer, RiskLimits
from portfolio import PositionManager
from agents import (
    Orchestrator,
    ScannerAgent,
    AnalyzerAgent,
    ExecutorAgent,
    run_trading_system,
    EventType,
    EventBus,
    get_event_bus
)

__all__ = [
    # Version
    "__version__",
    # Config
    "get_settings",
    "Settings",
    # Broker
    "get_broker",
    "AbstractBroker",
    # Risk
    "RiskManager",
    "PositionSizer",
    "RiskLimits",
    # Portfolio
    "PositionManager",
    # Agents
    "Orchestrator",
    "ScannerAgent",
    "AnalyzerAgent",
    "ExecutorAgent",
    "run_trading_system",
    # Events
    "EventType",
    "EventBus",
    "get_event_bus",
]

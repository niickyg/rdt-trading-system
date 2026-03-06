"""
Database module for RDT Trading System.

Supports both PostgreSQL and SQLite, with optional TimescaleDB integration
for optimized time-series queries.

TimescaleDB Support:
    When using PostgreSQL with TimescaleDB extension, the system automatically
    detects and enables time-series optimizations. Check with:

        from data.database import get_db_manager
        manager = get_db_manager()
        if manager.is_timescale:
            # Use TimescaleDB features
            from data.timescale import get_signals_time_bucket
            signals = get_signals_time_bucket('1 hour')
"""

from .models import (
    Base, Trade, Position, Signal, DailyStats, WatchlistItem, APIUser, User,
    TradeDirection, TradeStatus, ExitReason, SignalStatus, SubscriptionTierEnum,
    RejectedSignal, EquitySnapshot, ParameterChange,
    IntradayBar, TechnicalIndicator, TradeSnapshot, MarketRegimeDaily,
    SectorData, OptionsGreeksHistory, EarningsCalendar, DailyBar,
)
from .connection import (
    DatabaseManager, get_db_manager, init_database,
    run_migrations, check_migrations, get_database_url,
    check_timescale_extension, get_timescale_version, health_check
)
from .trades_repository import TradesRepository, get_trades_repository
from .ml_repository import MLDataRepository, get_ml_repository

__all__ = [
    # Models
    'Base', 'Trade', 'Position', 'Signal', 'DailyStats', 'WatchlistItem',
    'APIUser', 'User',
    'RejectedSignal', 'EquitySnapshot', 'ParameterChange',
    # ML Training Models
    'IntradayBar', 'TechnicalIndicator', 'TradeSnapshot', 'MarketRegimeDaily',
    'SectorData', 'OptionsGreeksHistory', 'EarningsCalendar', 'DailyBar',
    # Enums
    'TradeDirection', 'TradeStatus', 'ExitReason', 'SignalStatus',
    'SubscriptionTierEnum',
    # Connection management
    'DatabaseManager', 'get_db_manager', 'init_database',
    'run_migrations', 'check_migrations', 'get_database_url',
    'check_timescale_extension', 'get_timescale_version', 'health_check',
    # Repositories
    'TradesRepository', 'get_trades_repository',
    'MLDataRepository', 'get_ml_repository',
]

"""
Backtesting Module
Historical simulation and strategy evaluation
"""

from backtesting.engine import BacktestEngine, BacktestTrade, BacktestResult
from backtesting.data_loader import DataLoader, load_default_watchlist

__all__ = [
    "BacktestEngine",
    "BacktestTrade",
    "BacktestResult",
    "DataLoader",
    "load_default_watchlist"
]

"""
Multi-Strategy Trading Framework

This module provides multiple trading strategies that can run in parallel
to achieve higher returns through diversification and capital efficiency.
"""

from strategies.base_strategy import BaseStrategy, StrategySignal, StrategyResult
from strategies.leveraged_etf import LeveragedETFStrategy
from strategies.kelly_sizer import KellyCriterionSizer
from strategies.multi_strategy_engine import MultiStrategyEngine

__all__ = [
    'BaseStrategy',
    'StrategySignal',
    'StrategyResult',
    'LeveragedETFStrategy',
    'KellyCriterionSizer',
    'MultiStrategyEngine',
]

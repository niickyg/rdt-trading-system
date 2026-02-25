"""
Options Trading Module for the RDT Trading System.

Provides options strategy selection, execution, and risk management
integrated with the existing signal pipeline.
"""

from options.models import (
    OptionContract,
    OptionGreeks,
    OptionLeg,
    OptionsStrategy,
    IVAnalysis,
    OptionsPositionSizeResult,
    OptionRight,
    OptionAction,
    StrategyDirection,
    IVRegime,
)
from options.config import OptionsConfig, OptionsMode

__all__ = [
    # Models
    "OptionContract",
    "OptionGreeks",
    "OptionLeg",
    "OptionsStrategy",
    "IVAnalysis",
    "OptionsPositionSizeResult",
    "OptionRight",
    "OptionAction",
    "StrategyDirection",
    "IVRegime",
    # Config
    "OptionsConfig",
    "OptionsMode",
]

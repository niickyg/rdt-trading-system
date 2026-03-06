"""
Strategy Registry

Central registry for all trading strategies. Strategies register themselves
on import, and the scanner/engine iterates over active strategies.
"""

import threading
from typing import Dict, List, Optional
from loguru import logger

from strategies.base_strategy import BaseStrategy


class StrategyRegistry:
    """
    Singleton registry of all available trading strategies.

    Usage:
        StrategyRegistry.register("rsi2_mean_reversion", RSI2MeanReversionStrategy())
        active = StrategyRegistry.get_active()
    """
    _strategies: Dict[str, BaseStrategy] = {}
    _lock = threading.RLock()

    @classmethod
    def register(cls, name: str, strategy: BaseStrategy) -> None:
        """Register a strategy by name."""
        with cls._lock:
            if name in cls._strategies:
                logger.warning(f"Strategy '{name}' already registered, replacing")
            cls._strategies[name] = strategy
            logger.info(f"Strategy registered: {name}")

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a strategy from the registry."""
        with cls._lock:
            if name in cls._strategies:
                del cls._strategies[name]
                logger.info(f"Strategy unregistered: {name}")

    @classmethod
    def get(cls, name: str) -> Optional[BaseStrategy]:
        """Get a strategy by name."""
        with cls._lock:
            return cls._strategies.get(name)

    @classmethod
    def get_all(cls) -> Dict[str, BaseStrategy]:
        """Get all registered strategies."""
        with cls._lock:
            return dict(cls._strategies)

    @classmethod
    def get_active(cls) -> List[BaseStrategy]:
        """Get all active (enabled) strategies."""
        with cls._lock:
            return [s for s in cls._strategies.values() if s.is_active]

    @classmethod
    def get_names(cls) -> List[str]:
        """Get names of all registered strategies."""
        with cls._lock:
            return list(cls._strategies.keys())

    @classmethod
    def clear(cls) -> None:
        """Remove all strategies (for testing)."""
        with cls._lock:
            cls._strategies.clear()

    @classmethod
    def count(cls) -> int:
        """Number of registered strategies."""
        with cls._lock:
            return len(cls._strategies)

"""
Base Data Provider
Abstract base class defining the interface for all data providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import pandas as pd


class ProviderError(Exception):
    """Base exception for data provider errors."""
    pass


class RateLimitError(ProviderError):
    """Raised when provider rate limit is exceeded."""
    pass


class AuthenticationError(ProviderError):
    """Raised when authentication fails."""
    pass


class DataNotFoundError(ProviderError):
    """Raised when requested data is not available."""
    pass


class ProviderStatus(str, Enum):
    """Provider health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class Quote:
    """
    Standard quote format for all providers.
    All providers must convert their data to this format.
    """
    symbol: str
    price: float
    open: float
    high: float
    low: float
    volume: int
    previous_close: float
    change: float
    change_percent: float
    timestamp: datetime
    provider: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any], provider: str) -> "Quote":
        """Create Quote from dictionary."""
        return cls(
            symbol=data["symbol"],
            price=data["price"],
            open=data.get("open", data["price"]),
            high=data.get("high", data["price"]),
            low=data.get("low", data["price"]),
            volume=data.get("volume", 0),
            previous_close=data.get("previous_close", data["price"]),
            change=data.get("change", 0.0),
            change_percent=data.get("change_percent", 0.0),
            timestamp=data.get("timestamp", datetime.now()),
            provider=provider,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Quote to dictionary."""
        return {
            "symbol": self.symbol,
            "price": self.price,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "volume": self.volume,
            "previous_close": self.previous_close,
            "change": self.change,
            "change_percent": self.change_percent,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "provider": self.provider,
        }


@dataclass
class HistoricalData:
    """
    Standard historical data format for all providers.
    Contains OHLCV data as a pandas DataFrame.
    """
    symbol: str
    data: pd.DataFrame  # Columns: open, high, low, close, volume (lowercase)
    period: str
    interval: str
    provider: str
    fetched_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Normalize column names to lowercase."""
        if not self.data.empty:
            self.data.columns = [c.lower() for c in self.data.columns]

    @property
    def is_empty(self) -> bool:
        """Check if data is empty."""
        return self.data.empty

    @property
    def latest_close(self) -> Optional[float]:
        """Get the most recent close price."""
        if self.is_empty:
            return None
        return float(self.data["close"].iloc[-1])

    @property
    def previous_close(self) -> Optional[float]:
        """Get the previous close price."""
        if self.is_empty or len(self.data) < 2:
            return None
        return float(self.data["close"].iloc[-2])


@dataclass
class ProviderHealth:
    """Provider health information."""
    name: str
    status: ProviderStatus
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    error_message: Optional[str] = None
    requests_today: int = 0
    requests_limit: Optional[int] = None


class DataProvider(ABC):
    """
    Abstract base class for all data providers.

    All concrete providers must implement these methods to ensure
    consistent behavior across different data sources.
    """

    def __init__(self, name: str, priority: int = 100):
        """
        Initialize the data provider.

        Args:
            name: Provider name for identification
            priority: Provider priority (lower = higher priority)
        """
        self.name = name
        self.priority = priority
        self._last_success: Optional[datetime] = None
        self._last_failure: Optional[datetime] = None
        self._consecutive_failures: int = 0
        self._error_message: Optional[str] = None
        self._requests_today: int = 0
        self._requests_limit: Optional[int] = None

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """
        Get current quote for a symbol.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")

        Returns:
            Quote object with current market data

        Raises:
            ProviderError: If quote cannot be retrieved
        """
        pass

    @abstractmethod
    def get_historical(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> HistoricalData:
        """
        Get historical OHLCV data for a symbol.

        Args:
            symbol: Stock ticker symbol
            period: Time period (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max")
            interval: Data interval (e.g., "1m", "5m", "15m", "1h", "1d", "1wk", "1mo")

        Returns:
            HistoricalData object with OHLCV DataFrame

        Raises:
            ProviderError: If historical data cannot be retrieved
        """
        pass

    @abstractmethod
    def get_batch_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """
        Get quotes for multiple symbols in a single request.

        Args:
            symbols: List of stock ticker symbols

        Returns:
            Dictionary mapping symbols to Quote objects

        Raises:
            ProviderError: If batch quotes cannot be retrieved
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider is currently available.

        Returns:
            True if provider is available and ready to serve requests
        """
        pass

    def get_health(self) -> ProviderHealth:
        """
        Get current health status of the provider.

        Returns:
            ProviderHealth object with status information
        """
        if self._consecutive_failures >= 5:
            status = ProviderStatus.UNAVAILABLE
        elif self._consecutive_failures >= 2:
            status = ProviderStatus.DEGRADED
        elif self._last_success is not None:
            status = ProviderStatus.HEALTHY
        else:
            status = ProviderStatus.UNKNOWN

        return ProviderHealth(
            name=self.name,
            status=status,
            last_success=self._last_success,
            last_failure=self._last_failure,
            consecutive_failures=self._consecutive_failures,
            error_message=self._error_message,
            requests_today=self._requests_today,
            requests_limit=self._requests_limit,
        )

    def record_success(self) -> None:
        """Record a successful request."""
        self._last_success = datetime.now()
        self._consecutive_failures = 0
        self._error_message = None
        self._requests_today += 1

    def record_failure(self, error: Optional[str] = None) -> None:
        """Record a failed request."""
        self._last_failure = datetime.now()
        self._consecutive_failures += 1
        self._error_message = error
        self._requests_today += 1

    def reset_daily_counter(self) -> None:
        """Reset daily request counter."""
        self._requests_today = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, priority={self.priority})"

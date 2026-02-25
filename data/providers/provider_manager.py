"""
Provider Manager
Manages multiple data providers with automatic fallback, health checking, and caching.
"""

import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Type, Any, Callable

from loguru import logger

from data.providers.base import (
    DataProvider,
    Quote,
    HistoricalData,
    ProviderError,
    RateLimitError,
    ProviderStatus,
    ProviderHealth,
)


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, not accepting requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for a provider.

    Prevents cascading failures by temporarily disabling failing providers.
    """
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    half_open_max_calls: int = 3

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    half_open_calls: int = 0

    def record_success(self) -> None:
        """Record a successful call."""
        self.failure_count = 0
        self.half_open_calls = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("Circuit breaker closed after successful recovery")

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker opened after half-open failure")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def can_execute(self) -> bool:
        """Check if a call can be executed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time is None:
                return True
            elapsed = (datetime.now() - self.last_failure_time).total_seconds()
            if elapsed >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                logger.info("Circuit breaker entering half-open state")
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False

        return False


@dataclass
class CacheEntry:
    """Cache entry with expiration."""
    data: Any
    expires_at: datetime
    provider: str

    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at


class LRUCache:
    """
    LRU cache with TTL expiration.

    Thread-safe implementation for caching quotes and historical data.
    """

    def __init__(self, max_size: int = 1000, default_ttl: float = 30.0):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            if entry.is_expired:
                del self._cache[key]
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return entry.data

    def set(self, key: str, value: Any, provider: str, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        with self._lock:
            ttl = ttl or self.default_ttl
            expires_at = datetime.now() + timedelta(seconds=ttl)

            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                # Evict oldest if at capacity
                while len(self._cache) >= self.max_size:
                    self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(data=value, expires_at=expires_at, provider=provider)

    def invalidate(self, key: str) -> None:
        """Remove entry from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]

    def invalidate_pattern(self, pattern: str) -> int:
        """Remove entries matching pattern."""
        with self._lock:
            keys_to_remove = [k for k in self._cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self._cache[key]
            return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()

    def cleanup_expired(self) -> int:
        """Remove all expired entries."""
        with self._lock:
            expired_keys = [k for k, v in self._cache.items() if v.is_expired]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    @property
    def size(self) -> int:
        """Current number of entries."""
        return len(self._cache)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            expired_count = sum(1 for v in self._cache.values() if v.is_expired)
            providers = {}
            for entry in self._cache.values():
                providers[entry.provider] = providers.get(entry.provider, 0) + 1

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "expired": expired_count,
                "by_provider": providers,
            }


class ProviderManager:
    """
    Manages multiple data providers with automatic fallback.

    Features:
    - Provider priority ordering
    - Automatic fallback on failure
    - Circuit breaker pattern
    - Health checking
    - Caching layer
    """

    def __init__(
        self,
        providers: Optional[List[DataProvider]] = None,
        cache_ttl: float = 30.0,
        cache_max_size: int = 1000,
        circuit_failure_threshold: int = 5,
        circuit_recovery_timeout: float = 60.0,
        health_check_interval: float = 300.0,
    ):
        """
        Initialize provider manager.

        Args:
            providers: List of data providers (or None to auto-configure)
            cache_ttl: Cache time-to-live in seconds
            cache_max_size: Maximum cache entries
            circuit_failure_threshold: Failures before circuit opens
            circuit_recovery_timeout: Seconds before testing failed provider
            health_check_interval: Seconds between health checks
        """
        self._providers: Dict[str, DataProvider] = {}
        self._circuits: Dict[str, CircuitBreaker] = {}
        self._provider_order: List[str] = []

        self._cache = LRUCache(max_size=cache_max_size, default_ttl=cache_ttl)
        self._cache_ttl = cache_ttl

        self._circuit_failure_threshold = circuit_failure_threshold
        self._circuit_recovery_timeout = circuit_recovery_timeout
        self._health_check_interval = health_check_interval

        self._last_health_check: Optional[datetime] = None
        self._lock = threading.RLock()

        # Auto-configure providers if none provided
        if providers is None:
            providers = self._create_default_providers()

        # Register providers
        for provider in providers:
            self.register_provider(provider)

        logger.info(f"ProviderManager initialized with {len(self._providers)} providers: {self._provider_order}")

    def _create_default_providers(self) -> List[DataProvider]:
        """
        Create default providers based on environment configuration.

        Returns:
            List of configured providers
        """
        from data.providers.yfinance_provider import YFinanceProvider
        from data.providers.alpha_vantage_provider import AlphaVantageProvider

        providers = []

        # Read provider priority from environment
        provider_priority = os.environ.get("DATA_PROVIDER_PRIORITY", "yfinance,alpha_vantage")
        priority_list = [p.strip().lower() for p in provider_priority.split(",")]

        # YFinance - always available, no API key needed
        yf_priority = priority_list.index("yfinance") * 10 if "yfinance" in priority_list else 10
        providers.append(YFinanceProvider(priority=yf_priority))

        # Alpha Vantage - requires API key
        av_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        if av_api_key:
            av_priority = priority_list.index("alpha_vantage") * 10 if "alpha_vantage" in priority_list else 20
            av_premium = os.environ.get("ALPHA_VANTAGE_PREMIUM", "false").lower() == "true"
            av_daily_limit = int(os.environ.get("ALPHA_VANTAGE_DAILY_LIMIT", "25"))

            providers.append(AlphaVantageProvider(
                api_key=av_api_key,
                priority=av_priority,
                premium=av_premium,
                daily_limit=av_daily_limit,
            ))
        else:
            logger.info("Alpha Vantage API key not configured, provider disabled")

        return providers

    def register_provider(self, provider: DataProvider) -> None:
        """
        Register a data provider.

        Args:
            provider: Provider to register
        """
        with self._lock:
            self._providers[provider.name] = provider
            self._circuits[provider.name] = CircuitBreaker(
                failure_threshold=self._circuit_failure_threshold,
                recovery_timeout=self._circuit_recovery_timeout,
            )

            # Rebuild priority order
            self._provider_order = sorted(
                self._providers.keys(),
                key=lambda n: self._providers[n].priority
            )

            logger.debug(f"Registered provider: {provider.name} (priority={provider.priority})")

    def unregister_provider(self, name: str) -> None:
        """
        Unregister a provider.

        Args:
            name: Provider name
        """
        with self._lock:
            if name in self._providers:
                del self._providers[name]
                del self._circuits[name]
                self._provider_order = [n for n in self._provider_order if n != name]
                logger.debug(f"Unregistered provider: {name}")

    def get_provider(self, name: str) -> Optional[DataProvider]:
        """
        Get a specific provider by name.

        Args:
            name: Provider name

        Returns:
            Provider or None if not found
        """
        return self._providers.get(name)

    def _get_available_providers(self) -> List[DataProvider]:
        """
        Get list of providers that are available and have open circuits.

        Returns:
            List of available providers in priority order
        """
        available = []
        for name in self._provider_order:
            provider = self._providers[name]
            circuit = self._circuits[name]

            if circuit.can_execute() and provider.is_available():
                available.append(provider)

        return available

    def _execute_with_fallback(
        self,
        operation: str,
        func: Callable[[DataProvider], Any],
        cache_key: Optional[str] = None,
        cache_ttl: Optional[float] = None,
    ) -> Any:
        """
        Execute an operation with fallback across providers.

        Args:
            operation: Operation name for logging
            func: Function to execute on provider
            cache_key: Optional cache key
            cache_ttl: Optional cache TTL

        Returns:
            Result from first successful provider

        Raises:
            ProviderError: If all providers fail
        """
        # Check cache first
        if cache_key:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached

        available = self._get_available_providers()
        if not available:
            raise ProviderError(f"No providers available for {operation}")

        errors = []

        for provider in available:
            circuit = self._circuits[provider.name]

            try:
                logger.debug(f"Trying {operation} with {provider.name}")
                result = func(provider)

                # Record success
                circuit.record_success()
                provider.record_success()

                # Cache result
                if cache_key:
                    self._cache.set(cache_key, result, provider.name, cache_ttl)

                return result

            except RateLimitError as e:
                logger.warning(f"{provider.name} rate limited: {e}")
                circuit.record_failure()
                provider.record_failure(str(e))
                errors.append(f"{provider.name}: rate limited")

            except ProviderError as e:
                logger.warning(f"{provider.name} error: {e}")
                circuit.record_failure()
                provider.record_failure(str(e))
                errors.append(f"{provider.name}: {e}")

            except Exception as e:
                logger.error(f"{provider.name} unexpected error: {e}")
                circuit.record_failure()
                provider.record_failure(str(e))
                errors.append(f"{provider.name}: {e}")

        raise ProviderError(f"All providers failed for {operation}: {'; '.join(errors)}")

    def get_quote(self, symbol: str, use_cache: bool = True) -> Quote:
        """
        Get current quote for a symbol with automatic fallback.

        Args:
            symbol: Stock ticker symbol
            use_cache: Whether to use cached data

        Returns:
            Quote object

        Raises:
            ProviderError: If all providers fail
        """
        cache_key = f"quote:{symbol}" if use_cache else None

        return self._execute_with_fallback(
            operation=f"get_quote({symbol})",
            func=lambda p: p.get_quote(symbol),
            cache_key=cache_key,
            cache_ttl=self._cache_ttl,
        )

    def get_historical(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
        use_cache: bool = True,
    ) -> HistoricalData:
        """
        Get historical data for a symbol with automatic fallback.

        Args:
            symbol: Stock ticker symbol
            period: Time period
            interval: Data interval
            use_cache: Whether to use cached data

        Returns:
            HistoricalData object

        Raises:
            ProviderError: If all providers fail
        """
        # Longer cache TTL for historical data
        historical_cache_ttl = max(self._cache_ttl * 10, 300.0)  # At least 5 minutes
        cache_key = f"historical:{symbol}:{period}:{interval}" if use_cache else None

        return self._execute_with_fallback(
            operation=f"get_historical({symbol}, {period}, {interval})",
            func=lambda p: p.get_historical(symbol, period, interval),
            cache_key=cache_key,
            cache_ttl=historical_cache_ttl,
        )

    def get_batch_quotes(
        self,
        symbols: List[str],
        use_cache: bool = True,
    ) -> Dict[str, Quote]:
        """
        Get quotes for multiple symbols with automatic fallback.

        Args:
            symbols: List of stock ticker symbols
            use_cache: Whether to use cached data

        Returns:
            Dictionary mapping symbols to Quote objects

        Raises:
            ProviderError: If all providers fail
        """
        # Check cache for individual symbols
        result = {}
        missing_symbols = []

        if use_cache:
            for symbol in symbols:
                cache_key = f"quote:{symbol}"
                cached = self._cache.get(cache_key)
                if cached is not None:
                    result[symbol] = cached
                else:
                    missing_symbols.append(symbol)
        else:
            missing_symbols = symbols

        if not missing_symbols:
            return result

        # Fetch missing symbols
        try:
            batch_result = self._execute_with_fallback(
                operation=f"get_batch_quotes({len(missing_symbols)} symbols)",
                func=lambda p: p.get_batch_quotes(missing_symbols),
            )

            # Cache individual quotes
            if use_cache:
                for symbol, quote in batch_result.items():
                    self._cache.set(f"quote:{symbol}", quote, quote.provider, self._cache_ttl)

            result.update(batch_result)
            return result

        except ProviderError:
            # If batch fails, try individual quotes
            logger.warning("Batch quote failed, falling back to individual requests")
            for symbol in missing_symbols:
                try:
                    result[symbol] = self.get_quote(symbol, use_cache=use_cache)
                except ProviderError:
                    logger.debug(f"Failed to get quote for {symbol}")
                    continue

            if not result:
                raise ProviderError("Failed to get any quotes")

            return result

    def get_batch_historical(
        self,
        symbols: List[str],
        period: str = "60d",
        interval: str = "1d",
        use_cache: bool = True,
    ) -> Dict[str, HistoricalData]:
        """
        Get historical data for multiple symbols.

        Args:
            symbols: List of stock ticker symbols
            period: Time period
            interval: Data interval
            use_cache: Whether to use cached data

        Returns:
            Dictionary mapping symbols to HistoricalData objects
        """
        from data.providers.yfinance_provider import YFinanceProvider

        # Check for YFinance provider which supports batch historical
        for name in self._provider_order:
            provider = self._providers[name]
            circuit = self._circuits[name]

            if isinstance(provider, YFinanceProvider) and circuit.can_execute():
                try:
                    result = provider.get_batch_historical(symbols, period, interval)

                    # Cache results
                    if use_cache:
                        cache_ttl = max(self._cache_ttl * 10, 300.0)
                        for symbol, data in result.items():
                            cache_key = f"historical:{symbol}:{period}:{interval}"
                            self._cache.set(cache_key, data, provider.name, cache_ttl)

                    circuit.record_success()
                    return result

                except Exception as e:
                    logger.warning(f"Batch historical failed: {e}")
                    circuit.record_failure()

        # Fall back to individual requests
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.get_historical(symbol, period, interval, use_cache)
            except ProviderError:
                continue

        return result

    def get_health(self) -> Dict[str, ProviderHealth]:
        """
        Get health status of all providers.

        Returns:
            Dictionary mapping provider names to health status
        """
        health = {}
        for name, provider in self._providers.items():
            provider_health = provider.get_health()

            # Add circuit breaker state
            circuit = self._circuits[name]
            if circuit.state == CircuitState.OPEN:
                provider_health.status = ProviderStatus.UNAVAILABLE
            elif circuit.state == CircuitState.HALF_OPEN:
                provider_health.status = ProviderStatus.DEGRADED

            health[name] = provider_health

        return health

    def run_health_check(self) -> Dict[str, bool]:
        """
        Run health check on all providers.

        Returns:
            Dictionary mapping provider names to health status
        """
        results = {}

        for name, provider in self._providers.items():
            try:
                is_healthy = provider.is_available()
                results[name] = is_healthy

                if is_healthy:
                    logger.debug(f"Health check passed: {name}")
                else:
                    logger.warning(f"Health check failed: {name}")

            except Exception as e:
                logger.error(f"Health check error for {name}: {e}")
                results[name] = False

        self._last_health_check = datetime.now()
        return results

    def check_health(self, test_symbol: str = "SPY") -> Dict[str, Any]:
        """
        Perform a live data fetch health check on all providers.

        Unlike ``run_health_check`` (which only calls ``is_available``),
        this method actually requests a small historical data slice from
        each provider so transient network problems and auth failures are
        detected immediately.

        Args:
            test_symbol: Symbol to use for the test fetch (default: SPY)

        Returns:
            Dictionary with keys:
                ``healthy`` – True if at least one provider is responsive
                ``providers`` – Per-provider result dicts (responsive, latency_ms,
                                error)
                ``primary`` – Name of the first responsive provider, or None
        """
        import time as _time

        results: Dict[str, Any] = {}
        primary: Optional[str] = None

        for name in self._provider_order:
            provider = self._providers[name]
            t0 = _time.monotonic()
            try:
                hist = provider.get_historical(test_symbol, period="5d", interval="1d")
                latency_ms = (_time.monotonic() - t0) * 1000
                responsive = hist is not None and not hist.data.empty
                results[name] = {
                    "responsive": responsive,
                    "latency_ms": round(latency_ms, 1),
                    "error": None,
                }
                if responsive and primary is None:
                    primary = name
                    logger.debug(
                        f"Health check: {name} is responsive "
                        f"({latency_ms:.0f} ms)"
                    )
            except Exception as exc:
                latency_ms = (_time.monotonic() - t0) * 1000
                results[name] = {
                    "responsive": False,
                    "latency_ms": round(latency_ms, 1),
                    "error": str(exc),
                }
                logger.warning(
                    f"Health check: {name} is NOT responsive "
                    f"({latency_ms:.0f} ms) – {exc}"
                )

        self._last_health_check = datetime.now()
        overall_healthy = any(v["responsive"] for v in results.values())

        if not overall_healthy:
            logger.error(
                "Health check: ALL providers are unresponsive – "
                "data fetches will fail until at least one provider recovers"
            )

        return {
            "healthy": overall_healthy,
            "providers": results,
            "primary": primary,
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Cache statistics dictionary
        """
        return self._cache.stats()

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        logger.info("Provider cache cleared")

    def invalidate_symbol(self, symbol: str) -> int:
        """
        Invalidate all cached data for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Number of entries invalidated
        """
        count = self._cache.invalidate_pattern(symbol)
        logger.debug(f"Invalidated {count} cache entries for {symbol}")
        return count

    def reset_circuit(self, provider_name: str) -> bool:
        """
        Reset circuit breaker for a provider.

        Args:
            provider_name: Provider name

        Returns:
            True if reset successful
        """
        if provider_name not in self._circuits:
            return False

        self._circuits[provider_name] = CircuitBreaker(
            failure_threshold=self._circuit_failure_threshold,
            recovery_timeout=self._circuit_recovery_timeout,
        )
        logger.info(f"Circuit breaker reset for {provider_name}")
        return True

    def get_provider_order(self) -> List[str]:
        """
        Get current provider priority order.

        Returns:
            List of provider names in priority order
        """
        return list(self._provider_order)

    def set_provider_priority(self, provider_name: str, priority: int) -> bool:
        """
        Update priority for a provider.

        Args:
            provider_name: Provider name
            priority: New priority value

        Returns:
            True if update successful
        """
        if provider_name not in self._providers:
            return False

        with self._lock:
            self._providers[provider_name].priority = priority
            self._provider_order = sorted(
                self._providers.keys(),
                key=lambda n: self._providers[n].priority
            )

        logger.info(f"Updated {provider_name} priority to {priority}")
        return True

    def status(self) -> Dict[str, Any]:
        """
        Get comprehensive status report.

        Returns:
            Status dictionary
        """
        health = self.get_health()
        cache_stats = self.get_cache_stats()

        return {
            "providers": {
                name: {
                    "status": h.status.value,
                    "priority": self._providers[name].priority,
                    "circuit_state": self._circuits[name].state.value,
                    "consecutive_failures": h.consecutive_failures,
                    "requests_today": h.requests_today,
                    "requests_limit": h.requests_limit,
                    "last_success": h.last_success.isoformat() if h.last_success else None,
                    "last_failure": h.last_failure.isoformat() if h.last_failure else None,
                    "error_message": h.error_message,
                }
                for name, h in health.items()
            },
            "provider_order": self._provider_order,
            "cache": cache_stats,
            "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None,
        }


# Global provider manager instance
_provider_manager: Optional[ProviderManager] = None


def get_provider_manager() -> ProviderManager:
    """
    Get the global provider manager instance.

    Creates one with default configuration if not exists.

    Returns:
        ProviderManager instance
    """
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = ProviderManager()
    return _provider_manager


def reset_provider_manager() -> None:
    """Reset the global provider manager instance."""
    global _provider_manager
    _provider_manager = None

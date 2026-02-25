"""
Retry Logic with Exponential Backoff for Alert System
Provides configurable retry behavior for failed alert operations.
"""

import time
import random
import functools
from typing import Optional, Callable, Any, Type, Tuple, Union
from dataclasses import dataclass, field
from loguru import logger


class AlertDeliveryError(Exception):
    """Base exception for alert delivery failures."""

    def __init__(self, message: str, retryable: bool = True, retry_after: Optional[float] = None):
        """
        Initialize alert delivery error.

        Args:
            message: Error message
            retryable: Whether the error is retryable
            retry_after: Suggested delay before retry (from rate limit headers)
        """
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.retry_after = retry_after


class RateLimitError(AlertDeliveryError):
    """Exception for rate limit responses from alert services."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message, retryable=True, retry_after=retry_after)


class AuthenticationError(AlertDeliveryError):
    """Exception for authentication failures - not retryable."""

    def __init__(self, message: str):
        super().__init__(message, retryable=False)


class ConfigurationError(AlertDeliveryError):
    """Exception for configuration issues - not retryable."""

    def __init__(self, message: str):
        super().__init__(message, retryable=False)


class NetworkError(AlertDeliveryError):
    """Exception for network-related failures - retryable."""

    def __init__(self, message: str):
        super().__init__(message, retryable=True)


class ServiceUnavailableError(AlertDeliveryError):
    """Exception for service unavailability - retryable."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message, retryable=True, retry_after=retry_after)


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior with exponential backoff.

    Attributes:
        max_attempts: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds before first retry (default: 1.0)
        max_delay: Maximum delay cap in seconds (default: 60.0)
        exponential_base: Base for exponential backoff calculation (default: 2.0)
        jitter: Whether to add random jitter to delays (default: True)
        jitter_factor: Maximum jitter as fraction of delay (default: 0.25)
        retryable_exceptions: Tuple of exception types to retry on
    """
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.25
    retryable_exceptions: Tuple[Type[Exception], ...] = field(
        default_factory=lambda: (
            AlertDeliveryError,
            ConnectionError,
            TimeoutError,
        )
    )

    def calculate_delay(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """
        Calculate delay before next retry attempt.

        Args:
            attempt: Current attempt number (0-indexed)
            retry_after: Optional server-suggested retry delay

        Returns:
            float: Delay in seconds before next retry
        """
        if retry_after is not None and retry_after > 0:
            base = min(retry_after, self.max_delay)
        else:
            base = self.base_delay * (self.exponential_base ** attempt)

        delay = min(base, self.max_delay)

        if self.jitter:
            jitter_amount = delay * self.jitter_factor * random.random()
            delay = delay + jitter_amount

        return delay


@dataclass
class RetryResult:
    """Result of a retry operation."""
    success: bool
    result: Any = None
    attempts: int = 0
    last_error: Optional[Exception] = None
    total_delay: float = 0.0

    @property
    def error_message(self) -> Optional[str]:
        """Get error message if failed."""
        if self.last_error:
            return str(self.last_error)
        return None


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
):
    """
    Decorator for retrying failed operations with exponential backoff.

    Args:
        config: RetryConfig instance (uses defaults if None)
        on_retry: Optional callback called on each retry with (attempt, exception, delay)

    Returns:
        Decorated function that will retry on failure

    Example:
        @retry_with_backoff(RetryConfig(max_attempts=5, base_delay=2.0))
        def send_notification():
            # ... send logic
            pass
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> RetryResult:
            last_error = None
            total_delay = 0.0

            for attempt in range(config.max_attempts):
                try:
                    result = func(*args, **kwargs)
                    return RetryResult(
                        success=True,
                        result=result,
                        attempts=attempt + 1,
                        total_delay=total_delay
                    )
                except Exception as e:
                    last_error = e

                    should_retry = False
                    retry_after = None

                    if isinstance(e, AlertDeliveryError):
                        should_retry = e.retryable
                        retry_after = e.retry_after
                    elif isinstance(e, config.retryable_exceptions):
                        should_retry = True

                    if not should_retry:
                        logger.warning(
                            f"Non-retryable error in {func.__name__}: {e}"
                        )
                        return RetryResult(
                            success=False,
                            attempts=attempt + 1,
                            last_error=e,
                            total_delay=total_delay
                        )

                    if attempt < config.max_attempts - 1:
                        delay = config.calculate_delay(attempt, retry_after)
                        total_delay += delay

                        logger.info(
                            f"Retry {attempt + 1}/{config.max_attempts} for {func.__name__} "
                            f"after {delay:.2f}s delay. Error: {e}"
                        )

                        if on_retry:
                            on_retry(attempt + 1, e, delay)

                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} retry attempts exhausted for "
                            f"{func.__name__}. Last error: {e}"
                        )

            return RetryResult(
                success=False,
                attempts=config.max_attempts,
                last_error=last_error,
                total_delay=total_delay
            )

        return wrapper

    return decorator


class RetryExecutor:
    """
    Class-based retry executor for more control over retry operations.

    Useful when you need to execute retry logic programmatically rather than
    using the decorator approach.
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry executor.

        Args:
            config: RetryConfig instance (uses defaults if None)
        """
        self.config = config or RetryConfig()

    def execute(
        self,
        func: Callable,
        *args,
        on_retry: Optional[Callable[[int, Exception, float], None]] = None,
        **kwargs
    ) -> RetryResult:
        """
        Execute a function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            on_retry: Optional callback on each retry
            **kwargs: Keyword arguments for the function

        Returns:
            RetryResult: Result of the operation
        """
        last_error = None
        total_delay = 0.0

        for attempt in range(self.config.max_attempts):
            try:
                result = func(*args, **kwargs)
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempt + 1,
                    total_delay=total_delay
                )
            except Exception as e:
                last_error = e

                should_retry = False
                retry_after = None

                if isinstance(e, AlertDeliveryError):
                    should_retry = e.retryable
                    retry_after = e.retry_after
                elif isinstance(e, self.config.retryable_exceptions):
                    should_retry = True

                if not should_retry:
                    logger.warning(f"Non-retryable error: {e}")
                    return RetryResult(
                        success=False,
                        attempts=attempt + 1,
                        last_error=e,
                        total_delay=total_delay
                    )

                if attempt < self.config.max_attempts - 1:
                    delay = self.config.calculate_delay(attempt, retry_after)
                    total_delay += delay

                    logger.info(
                        f"Retry {attempt + 1}/{self.config.max_attempts} "
                        f"after {delay:.2f}s delay. Error: {e}"
                    )

                    if on_retry:
                        on_retry(attempt + 1, e, delay)

                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {self.config.max_attempts} retry attempts exhausted. "
                        f"Last error: {e}"
                    )

        return RetryResult(
            success=False,
            attempts=self.config.max_attempts,
            last_error=last_error,
            total_delay=total_delay
        )

    async def execute_async(
        self,
        func: Callable,
        *args,
        on_retry: Optional[Callable[[int, Exception, float], None]] = None,
        **kwargs
    ) -> RetryResult:
        """
        Execute an async function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            on_retry: Optional callback on each retry
            **kwargs: Keyword arguments for the function

        Returns:
            RetryResult: Result of the operation
        """
        import asyncio

        last_error = None
        total_delay = 0.0

        for attempt in range(self.config.max_attempts):
            try:
                result = await func(*args, **kwargs)
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempt + 1,
                    total_delay=total_delay
                )
            except Exception as e:
                last_error = e

                should_retry = False
                retry_after = None

                if isinstance(e, AlertDeliveryError):
                    should_retry = e.retryable
                    retry_after = e.retry_after
                elif isinstance(e, self.config.retryable_exceptions):
                    should_retry = True

                if not should_retry:
                    logger.warning(f"Non-retryable error: {e}")
                    return RetryResult(
                        success=False,
                        attempts=attempt + 1,
                        last_error=e,
                        total_delay=total_delay
                    )

                if attempt < self.config.max_attempts - 1:
                    delay = self.config.calculate_delay(attempt, retry_after)
                    total_delay += delay

                    logger.info(
                        f"Retry {attempt + 1}/{self.config.max_attempts} "
                        f"after {delay:.2f}s delay. Error: {e}"
                    )

                    if on_retry:
                        on_retry(attempt + 1, e, delay)

                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"All {self.config.max_attempts} retry attempts exhausted. "
                        f"Last error: {e}"
                    )

        return RetryResult(
            success=False,
            attempts=self.config.max_attempts,
            last_error=last_error,
            total_delay=total_delay
        )

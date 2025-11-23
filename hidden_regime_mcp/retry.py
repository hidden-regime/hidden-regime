"""
Retry logic with exponential backoff for MCP tool resilience.

Implements configurable retry mechanisms for transient failures,
with exponential backoff and jitter to avoid thundering herd.
"""

import time
import asyncio
import logging
from typing import Callable, Any, TypeVar, Optional, Type, Tuple
from functools import wraps
from dataclasses import dataclass

from hidden_regime_mcp.errors import MCPToolError, NetworkError, ResourceError

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retriable_exceptions: Tuple[Type[Exception], ...] = (
        NetworkError,
        ResourceError,
        TimeoutError,
        ConnectionError,
        OSError,
    )

    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt number.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds with exponential backoff and optional jitter
        """
        # Exponential backoff: delay = initial_delay * (base ^ attempt)
        delay = self.initial_delay_seconds * (self.exponential_base ** attempt)

        # Cap at maximum delay
        delay = min(delay, self.max_delay_seconds)

        # Add jitter to avoid thundering herd
        if self.jitter:
            import random

            jitter = random.uniform(0, delay * 0.1)
            delay += jitter

        return delay

    def is_retriable(self, exception: Exception) -> bool:
        """Check if exception is retriable."""
        return isinstance(exception, self.retriable_exceptions)


class RetryManager:
    """Manages retry logic for function calls."""

    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry manager.

        Args:
            config: RetryConfig instance, uses defaults if None
        """
        self.config = config or RetryConfig()

    def retry_sync(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with retry logic (synchronous).

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception: Last exception if all retries exhausted
        """
        last_exception = None

        for attempt in range(self.config.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                # Check if retriable
                if not self.config.is_retriable(e):
                    raise

                # Check if last attempt
                if attempt >= self.config.max_attempts - 1:
                    raise

                # Calculate delay and log
                delay = self.config.get_delay(attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{self.config.max_attempts} failed "
                    f"({type(e).__name__}), retrying in {delay:.2f}s"
                )

                # Wait before retry
                time.sleep(delay)

        # Should never reach here, but just in case
        if last_exception:
            raise last_exception

    async def retry_async(self, func: Callable[..., Any], *args, **kwargs) -> T:
        """
        Execute async function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception: Last exception if all retries exhausted
        """
        last_exception = None

        for attempt in range(self.config.max_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                # Check if retriable
                if not self.config.is_retriable(e):
                    raise

                # Check if last attempt
                if attempt >= self.config.max_attempts - 1:
                    raise

                # Calculate delay and log
                delay = self.config.get_delay(attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{self.config.max_attempts} failed "
                    f"({type(e).__name__}), retrying in {delay:.2f}s"
                )

                # Wait before retry
                await asyncio.sleep(delay)

        # Should never reach here, but just in case
        if last_exception:
            raise last_exception


# Global retry manager instance
_global_retry_manager = RetryManager()


def get_retry_manager(config: Optional[RetryConfig] = None) -> RetryManager:
    """
    Get global retry manager or create with custom config.

    Args:
        config: Optional custom config

    Returns:
        RetryManager instance
    """
    if config is not None:
        return RetryManager(config)
    return _global_retry_manager


def retry(config: Optional[RetryConfig] = None) -> Callable:
    """
    Decorator for synchronous functions with retry logic.

    Args:
        config: Optional RetryConfig

    Usage:
        @retry()
        def my_function():
            ...

        @retry(RetryConfig(max_attempts=5))
        def my_function():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            manager = get_retry_manager(config)
            return manager.retry_sync(func, *args, **kwargs)

        return wrapper

    return decorator


def async_retry(config: Optional[RetryConfig] = None) -> Callable:
    """
    Decorator for async functions with retry logic.

    Args:
        config: Optional RetryConfig

    Usage:
        @async_retry()
        async def my_async_function():
            ...

        @async_retry(RetryConfig(max_attempts=5))
        async def my_async_function():
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            manager = get_retry_manager(config)
            return await manager.retry_async(func, *args, **kwargs)

        return wrapper

    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests blocked
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 60.0,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening
            recovery_timeout_seconds: Time to wait before trying to recover
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def record_success(self) -> None:
        """Record successful request."""
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self) -> None:
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

    def is_closed(self) -> bool:
        """Check if circuit is closed (requests allowed)."""
        if self.state == "CLOSED":
            return True

        if self.state == "OPEN":
            # Check if we should try to recover
            if (
                self.last_failure_time
                and time.time() - self.last_failure_time
                > self.recovery_timeout_seconds
            ):
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker entering HALF_OPEN state")
                return True
            return False

        # HALF_OPEN state
        return True

    def get_state(self) -> str:
        """Get current circuit breaker state."""
        return self.state


# Global circuit breaker for yfinance requests
_yfinance_circuit_breaker = CircuitBreaker(
    failure_threshold=5, recovery_timeout_seconds=60.0
)


def get_yfinance_circuit_breaker() -> CircuitBreaker:
    """Get the global circuit breaker for yfinance."""
    return _yfinance_circuit_breaker

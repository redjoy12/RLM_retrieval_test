"""Circuit breaker pattern for LLM provider failover."""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, TypeVar

import structlog

logger = structlog.get_logger()

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()  # Normal operation, requests pass through
    OPEN = auto()  # Failing, requests blocked
    HALF_OPEN = auto()  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    success_threshold: int = 2


class CircuitBreaker:
    """Circuit breaker for protecting against failing services."""

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        enabled: bool = True,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            name: Circuit breaker name (usually provider name)
            config: Circuit breaker configuration
            enabled: Whether circuit breaker is enabled
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.enabled = enabled

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

        self._lock = asyncio.Lock()

        logger.info(
            "circuit_breaker_initialized",
            name=name,
            enabled=enabled,
            failure_threshold=self.config.failure_threshold,
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    async def can_execute(self) -> bool:
        """Check if request can be executed.

        Returns:
            True if request can proceed
        """
        if not self.enabled:
            return True

        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._last_failure_time is not None:
                    elapsed = time.monotonic() - self._last_failure_time
                    if elapsed >= self.config.recovery_timeout:
                        logger.info(
                            "circuit_breaker_half_open",
                            name=self.name,
                            elapsed=elapsed,
                        )
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_calls = 0
                        return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                # Allow limited calls in half-open state
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    async def record_success(self) -> None:
        """Record a successful call."""
        if not self.enabled:
            return

        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    logger.info(
                        "circuit_breaker_closed",
                        name=self.name,
                        successes=self._success_count,
                    )
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._half_open_calls = 0
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success in closed state
                if self._failure_count > 0:
                    self._failure_count = 0

    async def record_failure(self) -> None:
        """Record a failed call."""
        if not self.enabled:
            return

        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning(
                    "circuit_breaker_opened_from_half_open",
                    name=self.name,
                    failure_count=self._failure_count,
                )
                self._state = CircuitState.OPEN
                self._half_open_calls = 0
                self._success_count = 0

            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self.config.failure_threshold
            ):
                logger.warning(
                    "circuit_breaker_opened",
                    name=self.name,
                    failure_count=self._failure_count,
                    threshold=self.config.failure_threshold,
                )
                self._state = CircuitState.OPEN

    async def execute(
        self, func: Callable[..., T], *args: Any, **kwargs: Any
    ) -> T:
        """Execute a function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpen: If circuit is open
            Exception: If function fails
        """
        if not await self.can_execute():
            raise CircuitBreakerOpen(f"Circuit breaker '{self.name}' is OPEN")

        try:
            result = await func(*args, **kwargs)
            await self.record_success()
            return result
        except Exception as e:
            await self.record_failure()
            raise

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status.

        Returns:
            Status dictionary
        """
        return {
            "name": self.name,
            "state": self._state.name,
            "enabled": self.enabled,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "failure_threshold": self.config.failure_threshold,
            "last_failure_time": self._last_failure_time,
            "recovery_timeout": self.config.recovery_timeout,
            "time_since_last_failure": (
                time.monotonic() - self._last_failure_time
                if self._last_failure_time
                else None
            ),
        }


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class CircuitBreakerManager:
    """Manages multiple circuit breakers for different providers."""

    def __init__(self) -> None:
        """Initialize circuit breaker manager."""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._fallback_chains: Dict[str, List[str]] = {}

    def register_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        enabled: bool = True,
    ) -> CircuitBreaker:
        """Register a circuit breaker.

        Args:
            name: Provider name
            config: Circuit breaker configuration
            enabled: Whether enabled

        Returns:
            CircuitBreaker instance
        """
        breaker = CircuitBreaker(name, config, enabled)
        self._breakers[name] = breaker
        return breaker

    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name.

        Args:
            name: Provider name

        Returns:
            CircuitBreaker or None
        """
        return self._breakers.get(name)

    def set_fallback_chain(
        self, primary: str, fallbacks: List[str]
    ) -> None:
        """Set fallback chain for a provider.

        Args:
            primary: Primary provider name
            fallbacks: List of fallback provider names in order
        """
        self._fallback_chains[primary] = fallbacks
        logger.info(
            "fallback_chain_set",
            primary=primary,
            fallbacks=fallbacks,
        )

    async def execute_with_fallback(
        self,
        primary: str,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute function with automatic fallback.

        Args:
            primary: Primary provider name
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If all providers fail
        """
        providers = [primary] + self._fallback_chains.get(primary, [])

        last_error: Optional[Exception] = None

        for provider in providers:
            breaker = self._breakers.get(provider)
            if breaker is None:
                # No circuit breaker, try direct execution
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    continue

            try:
                return await breaker.execute(func, *args, **kwargs)
            except CircuitBreakerOpen:
                logger.info(
                    "circuit_open_trying_fallback",
                    primary=primary,
                    current=provider,
                )
                continue
            except Exception as e:
                last_error = e
                # Circuit breaker recorded the failure, try next
                continue

        # All providers failed
        if last_error:
            raise last_error
        raise Exception(f"All providers failed for {primary}")

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers.

        Returns:
            Dictionary of provider names to status
        """
        return {
            name: breaker.get_status()
            for name, breaker in self._breakers.items()
        }


# Global manager instance
_global_manager: Optional[CircuitBreakerManager] = None


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get global circuit breaker manager.

    Returns:
        CircuitBreakerManager instance
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = CircuitBreakerManager()
    return _global_manager

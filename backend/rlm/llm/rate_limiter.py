"""Rate limiter using token bucket algorithm for LLM API calls."""

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Optional

import structlog

logger = structlog.get_logger()


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 60
    tokens_per_minute: int = 60000
    burst_size: int = 10


class TokenBucket:
    """Token bucket for rate limiting."""

    def __init__(
        self,
        requests_per_minute: int,
        tokens_per_minute: int,
        burst_size: int = 10,
    ) -> None:
        """Initialize token bucket.

        Args:
            requests_per_minute: Maximum requests per minute
            tokens_per_minute: Maximum tokens per minute
            burst_size: Maximum burst size for requests
        """
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.burst_size = burst_size

        # Request bucket
        self._request_tokens = burst_size
        self._last_request_update = time.monotonic()

        # Token bucket (for TPM)
        self._token_tokens = tokens_per_minute
        self._last_token_update = time.monotonic()

        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 0) -> bool:
        """Acquire permission to make a request.

        Args:
            tokens: Number of tokens this request will consume

        Returns:
            True if acquired, False if would block
        """
        async with self._lock:
            now = time.monotonic()

            # Replenish request tokens
            elapsed = now - self._last_request_update
            self._request_tokens = min(
                self.burst_size,
                self._request_tokens
                + (elapsed * self.requests_per_minute / 60),
            )
            self._last_request_update = now

            # Replenish token bucket
            token_elapsed = now - self._last_token_update
            self._token_tokens = min(
                self.tokens_per_minute,
                self._token_tokens
                + (token_elapsed * self.tokens_per_minute / 60),
            )
            self._last_token_update = now

            # Check if we have enough tokens
            if self._request_tokens < 1:
                return False

            if tokens > 0 and self._token_tokens < tokens:
                return False

            # Consume tokens
            self._request_tokens -= 1
            if tokens > 0:
                self._token_tokens -= tokens

            return True

    async def acquire_wait(
        self, tokens: int = 0, timeout: Optional[float] = None
    ) -> bool:
        """Acquire permission, waiting if necessary.

        Args:
            tokens: Number of tokens this request will consume
            timeout: Maximum time to wait (None for indefinite)

        Returns:
            True if acquired, False if timeout
        """
        start_time = time.monotonic()

        while True:
            if await self.acquire(tokens):
                return True

            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    return False

            # Wait a bit before retrying
            await asyncio.sleep(0.1)

    def get_wait_time(self, tokens: int = 0) -> float:
        """Calculate how long to wait before next request is allowed.

        Args:
            tokens: Number of tokens needed

        Returns:
            Seconds to wait (0 if ready now)
        """
        now = time.monotonic()

        # Calculate request wait time
        elapsed = now - self._last_request_update
        current_requests = min(
            self.burst_size,
            self._request_tokens + (elapsed * self.requests_per_minute / 60),
        )

        if current_requests < 1:
            request_wait = (1 - current_requests) * 60 / self.requests_per_minute
        else:
            request_wait = 0

        # Calculate token wait time
        token_elapsed = now - self._last_token_update
        current_tokens = min(
            self.tokens_per_minute,
            self._token_tokens
            + (token_elapsed * self.tokens_per_minute / 60),
        )

        if tokens > 0 and current_tokens < tokens:
            token_wait = (tokens - current_tokens) * 60 / self.tokens_per_minute
        else:
            token_wait = 0

        return max(request_wait, token_wait)


class RateLimiter:
    """Rate limiter for multiple LLM providers."""

    # Default rate limits for providers
    DEFAULT_LIMITS = {
        "openai": RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=60000,
            burst_size=10,
        ),
        "anthropic": RateLimitConfig(
            requests_per_minute=50,
            tokens_per_minute=40000,
            burst_size=8,
        ),
        "ollama": RateLimitConfig(
            requests_per_minute=1000,
            tokens_per_minute=1000000,
            burst_size=100,
        ),
        "local": RateLimitConfig(
            requests_per_minute=1000,
            tokens_per_minute=1000000,
            burst_size=100,
        ),
    }

    def __init__(
        self,
        custom_limits: Optional[Dict[str, RateLimitConfig]] = None,
        enabled: bool = True,
    ) -> None:
        """Initialize rate limiter.

        Args:
            custom_limits: Override default limits for providers
            enabled: Whether rate limiting is enabled
        """
        self.enabled = enabled
        self._buckets: Dict[str, TokenBucket] = {}
        self._limits = self.DEFAULT_LIMITS.copy()

        if custom_limits:
            self._limits.update(custom_limits)

        logger.info("rate_limiter_initialized", enabled=enabled)

    def _get_bucket(self, provider: str) -> TokenBucket:
        """Get or create token bucket for provider."""
        if provider not in self._buckets:
            config = self._limits.get(
                provider,
                RateLimitConfig(
                    requests_per_minute=60,
                    tokens_per_minute=60000,
                    burst_size=10,
                ),
            )
            self._buckets[provider] = TokenBucket(
                requests_per_minute=config.requests_per_minute,
                tokens_per_minute=config.tokens_per_minute,
                burst_size=config.burst_size,
            )
            logger.debug(
                "created_rate_limit_bucket",
                provider=provider,
                rpm=config.requests_per_minute,
                tpm=config.tokens_per_minute,
            )

        return self._buckets[provider]

    async def acquire(
        self,
        provider: str,
        tokens: int = 0,
        wait: bool = True,
        timeout: Optional[float] = None,
    ) -> bool:
        """Acquire permission to make a request.

        Args:
            provider: Provider name (openai, anthropic, ollama)
            tokens: Number of tokens this request will consume
            wait: Whether to wait if rate limited
            timeout: Maximum wait time (if wait=True)

        Returns:
            True if request can proceed, False if rate limited
        """
        if not self.enabled:
            return True

        bucket = self._get_bucket(provider)

        if wait:
            result = await bucket.acquire_wait(tokens, timeout)
        else:
            result = await bucket.acquire(tokens)

        if not result:
            logger.warning(
                "rate_limit_exceeded",
                provider=provider,
                tokens_requested=tokens,
            )

        return result

    def get_wait_time(self, provider: str, tokens: int = 0) -> float:
        """Get estimated wait time for a provider.

        Args:
            provider: Provider name
            tokens: Number of tokens needed

        Returns:
            Seconds to wait before request can be made
        """
        if not self.enabled or provider not in self._buckets:
            return 0.0

        return self._buckets[provider].get_wait_time(tokens)

    def update_limits(
        self, provider: str, rpm: int, tpm: int, burst: int = 10
    ) -> None:
        """Update rate limits for a provider.

        Args:
            provider: Provider name
            rpm: Requests per minute
            tpm: Tokens per minute
            burst: Burst size
        """
        self._limits[provider] = RateLimitConfig(
            requests_per_minute=rpm,
            tokens_per_minute=tpm,
            burst_size=burst,
        )

        # Recreate bucket with new limits
        if provider in self._buckets:
            del self._buckets[provider]

        logger.info(
            "updated_rate_limits",
            provider=provider,
            rpm=rpm,
            tpm=tpm,
            burst=burst,
        )

    def get_current_usage(self, provider: str) -> Dict[str, float]:
        """Get current rate limiter status for a provider.

        Args:
            provider: Provider name

        Returns:
            Dictionary with current token levels
        """
        if provider not in self._buckets:
            return {"request_tokens": 1.0, "token_tokens": 1.0}

        bucket = self._buckets[provider]

        # Replenish calculations
        now = time.monotonic()
        elapsed = now - bucket._last_request_update
        request_tokens = min(
            bucket.burst_size,
            bucket._request_tokens
            + (elapsed * bucket.requests_per_minute / 60),
        )

        token_elapsed = now - bucket._last_token_update
        token_tokens = min(
            bucket.tokens_per_minute,
            bucket._token_tokens
            + (token_elapsed * bucket.tokens_per_minute / 60),
        )

        return {
            "request_tokens": request_tokens,
            "token_tokens": token_tokens,
            "request_percentage": request_tokens / bucket.burst_size,
            "token_percentage": token_tokens / bucket.tokens_per_minute,
        }

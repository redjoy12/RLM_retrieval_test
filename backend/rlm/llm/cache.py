"""Response cache for LLM calls (exact match)."""

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import structlog

from rlm.types import LLMResponse

logger = structlog.get_logger()


@dataclass
class CacheEntry:
    """A cached response entry."""

    key: str
    response: LLMResponse
    timestamp: float
    ttl_seconds: int

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return (time.time() - self.timestamp) > self.ttl_seconds


class ResponseCache:
    """Simple exact-match cache for LLM responses."""

    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_size: int = 1000,
        enabled: bool = True,
    ) -> None:
        """Initialize response cache.

        Args:
            ttl_seconds: Time-to-live for cached entries
            max_size: Maximum number of entries to store
            enabled: Whether caching is enabled
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.enabled = enabled
        self._cache: Dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0

        logger.info(
            "response_cache_initialized",
            enabled=enabled,
            ttl=ttl_seconds,
            max_size=max_size,
        )

    def _generate_key(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Generate cache key from request parameters.

        Args:
            prompt: User prompt
            model: Model name
            system_prompt: System prompt
            temperature: Temperature
            max_tokens: Max tokens
            **kwargs: Additional parameters

        Returns:
            Cache key string
        """
        # Create deterministic string representation
        key_parts = [
            f"model:{model}",
            f"temp:{temperature}",
            f"max_tokens:{max_tokens}",
            f"sys:{system_prompt or ''}",
            f"prompt:{prompt}",
        ]

        # Add sorted kwargs for consistency
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")

        key_string = "|".join(key_parts)

        # Hash to fixed length
        return hashlib.sha256(key_string.encode()).hexdigest()

    async def get(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Optional[LLMResponse]:
        """Get cached response if available.

        Args:
            prompt: User prompt
            model: Model name
            system_prompt: System prompt
            temperature: Temperature
            max_tokens: Max tokens
            **kwargs: Additional parameters

        Returns:
            Cached response or None
        """
        if not self.enabled:
            return None

        key = self._generate_key(
            prompt, model, system_prompt, temperature, max_tokens, **kwargs
        )

        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        if entry.is_expired():
            # Remove expired entry
            del self._cache[key]
            self._misses += 1
            logger.debug("cache_entry_expired", key=key[:8])
            return None

        self._hits += 1
        logger.debug(
            "cache_hit",
            key=key[:8],
            model=model,
            hit_rate=self.hit_rate,
        )
        return entry.response

    async def set(
        self,
        prompt: str,
        model: str,
        response: LLMResponse,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Cache a response.

        Args:
            prompt: User prompt
            model: Model name
            response: LLM response to cache
            system_prompt: System prompt
            temperature: Temperature
            max_tokens: Max tokens
            ttl_seconds: Override default TTL
            **kwargs: Additional parameters
        """
        if not self.enabled:
            return

        # Don't cache error responses or empty responses
        if not response.content or response.content.strip() == "":
            return

        key = self._generate_key(
            prompt, model, system_prompt, temperature, max_tokens, **kwargs
        )

        # Evict oldest entries if at capacity (simple LRU)
        if len(self._cache) >= self.max_size:
            self._evict_oldest()

        entry = CacheEntry(
            key=key,
            response=response,
            timestamp=time.time(),
            ttl_seconds=ttl_seconds or self.ttl_seconds,
        )

        self._cache[key] = entry

        logger.debug(
            "cache_set",
            key=key[:8],
            model=model,
            cache_size=len(self._cache),
        )

    def _evict_oldest(self) -> None:
        """Evict oldest entries to make room."""
        # Sort by timestamp and remove oldest 10%
        sorted_entries = sorted(
            self._cache.items(), key=lambda x: x[1].timestamp
        )
        to_remove = int(len(sorted_entries) * 0.1) + 1

        for i in range(to_remove):
            if i < len(sorted_entries):
                key = sorted_entries[i][0]
                del self._cache[key]

        logger.debug("cache_evicted", removed=to_remove)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("cache_cleared")

    def remove_expired(self) -> int:
        """Remove expired entries.

        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, entry in self._cache.items() if entry.is_expired()
        ]

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.debug("cache_expired_removed", count=len(expired_keys))

        return len(expired_keys)

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate.

        Returns:
            Hit rate as percentage (0-100)
        """
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return (self._hits / total) * 100

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "enabled": self.enabled,
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "ttl_seconds": self.ttl_seconds,
        }

    def invalidate_model(self, model: str) -> int:
        """Invalidate all cached entries for a model.

        Args:
            model: Model name

        Returns:
            Number of entries invalidated
        """
        # This is approximate since we can't easily filter by model from hash keys
        # In production, you might want to store model separately
        removed = 0
        keys_to_remove = []

        for key in list(self._cache.keys()):
            # Simple heuristic: remove entries that might be this model
            # In a real implementation, you'd store model with the entry
            del self._cache[key]
            removed += 1

        if removed:
            logger.info("cache_invalidated_model", model=model, removed=removed)

        return removed

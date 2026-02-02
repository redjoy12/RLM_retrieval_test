"""Session-level cache for sub-LLM queries."""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

logger = structlog.get_logger()


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""

    key: str
    value: str
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = time.time()


class SessionCache:
    """In-memory cache for sub-LLM queries within a session.

    This cache stores sub-LLM query results to avoid redundant API calls
    when the same or similar queries are made within a session. Uses TTL
    (time-to-live) based expiration and LRU (least recently used) eviction
    when capacity is reached.

    Example:
        ```python
        cache = SessionCache(ttl_seconds=3600, max_size=1000)

        # Store a result
        cache.set("session-123", "What is AI?", "AI is artificial intelligence...")

        # Retrieve (returns None if not found or expired)
        result = cache.get("session-123", "What is AI?")

        # Get cache statistics
        stats = cache.get_stats()
        print(f"Hit rate: {stats['hit_rate']:.2%}")
        ```
    """

    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_size: int = 1000,
        enabled: bool = True,
    ) -> None:
        """Initialize the session cache.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
            max_size: Maximum number of entries before LRU eviction
            enabled: Whether caching is enabled
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.enabled = enabled

        # Storage: {session_id: {cache_key: CacheEntry}}
        self._cache: dict[str, dict[str, CacheEntry]] = {}

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

        logger.info(
            "session_cache_initialized",
            ttl_seconds=ttl_seconds,
            max_size=max_size,
            enabled=enabled,
        )

    def _generate_key(self, query: str, context: Optional[str] = None) -> str:
        """Generate a cache key from query and optional context.

        Args:
            query: The query string
            context: Optional context string

        Returns:
            SHA256 hash as hex string
        """
        # Normalize query
        normalized = query.strip().lower()

        # Include context if provided (limited to first 1000 chars for performance)
        if context:
            normalized += "|" + context[:1000].strip().lower()

        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(
        self,
        session_id: str,
        query: str,
        context: Optional[str] = None,
    ) -> Optional[str]:
        """Get a cached result if available and not expired.

        Args:
            session_id: Session identifier
            query: Query string
            context: Optional context string

        Returns:
            Cached result or None if not found/expired
        """
        if not self.enabled:
            return None

        cache_key = self._generate_key(query, context)

        # Check if session exists
        if session_id not in self._cache:
            self._misses += 1
            return None

        # Check if key exists
        session_cache = self._cache[session_id]
        if cache_key not in session_cache:
            self._misses += 1
            return None

        entry = session_cache[cache_key]

        # Check if expired
        if time.time() - entry.timestamp > self.ttl_seconds:
            # Remove expired entry
            del session_cache[cache_key]
            self._expirations += 1
            self._misses += 1
            logger.debug(
                "cache_entry_expired",
                session_id=session_id,
                key=cache_key[:8],
                age=time.time() - entry.timestamp,
            )
            return None

        # Cache hit
        entry.touch()
        self._hits += 1
        logger.debug(
            "cache_hit",
            session_id=session_id,
            key=cache_key[:8],
            access_count=entry.access_count,
        )

        return entry.value

    def set(
        self,
        session_id: str,
        query: str,
        value: str,
        context: Optional[str] = None,
    ) -> None:
        """Store a result in the cache.

        Args:
            session_id: Session identifier
            query: Query string
            value: Result value to cache
            context: Optional context string
        """
        if not self.enabled:
            return

        # Initialize session cache if needed
        if session_id not in self._cache:
            self._cache[session_id] = {}
            logger.debug("session_cache_created", session_id=session_id)

        session_cache = self._cache[session_id]
        cache_key = self._generate_key(query, context)

        # Check if we need to evict (LRU)
        if len(session_cache) >= self.max_size and cache_key not in session_cache:
            self._evict_lru(session_id)

        # Store entry
        session_cache[cache_key] = CacheEntry(
            key=cache_key,
            value=value,
        )

        logger.debug(
            "cache_entry_set",
            session_id=session_id,
            key=cache_key[:8],
            size=len(session_cache),
        )

    def _evict_lru(self, session_id: str) -> None:
        """Evict least recently used entry from session cache.

        Args:
            session_id: Session to evict from
        """
        session_cache = self._cache[session_id]

        # Find LRU entry
        lru_key = min(
            session_cache.keys(),
            key=lambda k: session_cache[k].last_accessed,
        )

        del session_cache[lru_key]
        self._evictions += 1

        logger.debug(
            "cache_entry_evicted",
            session_id=session_id,
            key=lru_key[:8],
            reason="lru",
        )

    def invalidate_session(self, session_id: str) -> int:
        """Invalidate all cached entries for a session.

        Args:
            session_id: Session to invalidate

        Returns:
            Number of entries removed
        """
        if session_id not in self._cache:
            return 0

        count = len(self._cache[session_id])
        del self._cache[session_id]

        logger.info(
            "session_cache_invalidated",
            session_id=session_id,
            entries_removed=count,
        )

        return count

    def invalidate_all(self) -> int:
        """Invalidate all cached entries across all sessions.

        Returns:
            Total number of entries removed
        """
        total = sum(len(cache) for cache in self._cache.values())
        self._cache.clear()

        logger.info("all_cache_invalidated", entries_removed=total)

        return total

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        total_entries = sum(len(cache) for cache in self._cache.values())

        return {
            "enabled": self.enabled,
            "ttl_seconds": self.ttl_seconds,
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
            "expirations": self._expirations,
            "total_entries": total_entries,
            "active_sessions": len(self._cache),
        }

    def get_session_stats(self, session_id: str) -> Optional[dict[str, Any]]:
        """Get statistics for a specific session.

        Args:
            session_id: Session identifier

        Returns:
            Session statistics or None if session not found
        """
        if session_id not in self._cache:
            return None

        session_cache = self._cache[session_id]
        entries = list(session_cache.values())

        if not entries:
            return {
                "session_id": session_id,
                "entries": 0,
                "avg_access_count": 0.0,
                "oldest_entry": None,
                "newest_entry": None,
            }

        return {
            "session_id": session_id,
            "entries": len(entries),
            "avg_access_count": sum(e.access_count for e in entries) / len(entries),
            "oldest_entry": min(e.timestamp for e in entries),
            "newest_entry": max(e.timestamp for e in entries),
        }

    def cleanup_expired(self) -> int:
        """Remove all expired entries across all sessions.

        Returns:
            Number of entries removed
        """
        current_time = time.time()
        removed = 0

        for session_id in list(self._cache.keys()):
            session_cache = self._cache[session_id]

            expired_keys = [
                key
                for key, entry in session_cache.items()
                if current_time - entry.timestamp > self.ttl_seconds
            ]

            for key in expired_keys:
                del session_cache[key]
                removed += 1

            # Remove empty sessions
            if not session_cache:
                del self._cache[session_id]

        if removed > 0:
            logger.info("expired_entries_cleaned", removed=removed)

        return removed


class SemanticSessionCache(SessionCache):
    """Extended cache with semantic similarity matching (Phase 2 feature).

    This cache extends the base SessionCache with the ability to find
    similar queries using embeddings. Not implemented in Phase 1.
    """

    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_size: int = 1000,
        enabled: bool = True,
        similarity_threshold: float = 0.95,
    ) -> None:
        """Initialize semantic cache (placeholder for Phase 2).

        Args:
            ttl_seconds: Time-to-live for cache entries
            max_size: Maximum number of entries
            enabled: Whether caching is enabled
            similarity_threshold: Minimum similarity score (0-1)
        """
        super().__init__(ttl_seconds, max_size, enabled)
        self.similarity_threshold = similarity_threshold

        logger.warning(
            "semantic_cache_not_implemented",
            message="Semantic similarity caching is a Phase 2 feature",
        )

    def find_similar(
        self,
        session_id: str,
        query: str,
    ) -> Optional[str]:
        """Find a semantically similar cached query (placeholder).

        Args:
            session_id: Session identifier
            query: Query to match

        Returns:
            None (not implemented in Phase 1)
        """
        # Phase 2: Implement embeddings-based similarity search
        return None

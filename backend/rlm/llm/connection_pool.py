"""Async connection pool for LLM API calls."""

import asyncio
from typing import Any, Awaitable, Callable, Optional, TypeVar

import httpx
import structlog

logger = structlog.get_logger()

T = TypeVar("T")


class AsyncConnectionPool:
    """Manages HTTP connections for LLM API calls with connection pooling.

    This class provides efficient connection reuse and concurrency control
    for making multiple LLM API calls. Uses HTTPX for async HTTP support
    with configurable connection limits and keep-alive settings.

    Example:
        ```python
        pool = AsyncConnectionPool(max_connections=100, max_keepalive=20)

        # Get client for making requests
        client = await pool.get_client()

        # Execute with automatic connection management
        result = await pool.execute(my_api_call_func, arg1, arg2)

        # Cleanup when done
        await pool.close()
        ```
    """

    def __init__(
        self,
        max_connections: int = 100,
        max_keepalive: int = 20,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the async connection pool.

        Args:
            max_connections: Maximum number of concurrent HTTP connections
            max_keepalive: Maximum keep-alive connections to maintain
            timeout: Default timeout for HTTP requests in seconds
        """
        self.max_connections = max_connections
        self.max_keepalive = max_keepalive
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(max_connections)
        self._closed = False

        logger.info(
            "connection_pool_initialized",
            max_connections=max_connections,
            max_keepalive=max_keepalive,
            timeout=timeout,
        )

    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTPX client with connection limits.

        Returns:
            Configured httpx.AsyncClient instance

        Raises:
            RuntimeError: If the pool has been closed
        """
        if self._closed:
            raise RuntimeError("Connection pool has been closed")

        if self._client is None:
            limits = httpx.Limits(
                max_connections=self.max_connections,
                max_keepalive_connections=self.max_keepalive,
            )
            timeout = httpx.Timeout(self.timeout, connect=10.0)
            self._client = httpx.AsyncClient(
                limits=limits,
                timeout=timeout,
                http2=True,  # Enable HTTP/2 for better performance
            )
            logger.debug(
                "httpx_client_created",
                max_connections=self.max_connections,
                max_keepalive=self.max_keepalive,
            )

        return self._client

    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute a function with connection pool management.

        Acquires a connection slot before executing the function and releases
        it afterward. This prevents overwhelming the API with too many
        concurrent connections.

        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result from the executed function

        Example:
            ```python
            async def make_api_call(client: httpx.AsyncClient, url: str):
                return await client.get(url)

            result = await pool.execute(make_api_call, client, "https://api.example.com")
            ```
        """
        async with self._semaphore:
            return await func(*args, **kwargs)

    async def execute_many(
        self,
        funcs: list[tuple[Callable[..., Awaitable[T]], tuple[Any, ...], dict[str, Any]]],
    ) -> list[T]:
        """Execute multiple functions concurrently with pool management.

        Args:
            funcs: List of (function, args, kwargs) tuples to execute

        Returns:
            List of results in the same order as input

        Example:
            ```python
            calls = [
                (api_call_1, ("arg1",), {"key": "value"}),
                (api_call_2, ("arg2",), {}),
            ]
            results = await pool.execute_many(calls)
            ```
        """
        semaphore = self._semaphore

        async def execute_with_semaphore(
            func: Callable[..., Awaitable[T]],
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> T:
            async with semaphore:
                return await func(*args, **kwargs)

        tasks = [
            asyncio.create_task(execute_with_semaphore(func, args, kwargs))
            for func, args, kwargs in funcs
        ]

        return await asyncio.gather(*tasks, return_exceptions=True)

    def get_active_connections(self) -> int:
        """Get the number of currently active connections.

        Returns:
            Number of connections currently in use
        """
        return self.max_connections - self._semaphore._value

    def get_available_connections(self) -> int:
        """Get the number of available connection slots.

        Returns:
            Number of connections available for use
        """
        return self._semaphore._value

    async def close(self) -> None:
        """Close all connections and cleanup resources.

        This should be called when the pool is no longer needed to prevent
        resource leaks.
        """
        if self._client and not self._closed:
            await self._client.aclose()
            self._client = None
            self._closed = True
            logger.info("connection_pool_closed")

    async def __aenter__(self) -> "AsyncConnectionPool":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - automatically closes pool."""
        await self.close()


class ConnectionPoolManager:
    """Manages multiple connection pools for different endpoints."""

    def __init__(self) -> None:
        """Initialize the connection pool manager."""
        self._pools: dict[str, AsyncConnectionPool] = {}

    def get_pool(
        self,
        endpoint: str,
        max_connections: int = 100,
        max_keepalive: int = 20,
    ) -> AsyncConnectionPool:
        """Get or create a connection pool for an endpoint.

        Args:
            endpoint: Unique identifier for the endpoint (e.g., "openai", "anthropic")
            max_connections: Maximum concurrent connections
            max_keepalive: Maximum keep-alive connections

        Returns:
            AsyncConnectionPool instance for the endpoint
        """
        if endpoint not in self._pools:
            self._pools[endpoint] = AsyncConnectionPool(
                max_connections=max_connections,
                max_keepalive=max_keepalive,
            )
            logger.info(
                "pool_created_for_endpoint",
                endpoint=endpoint,
                max_connections=max_connections,
            )

        return self._pools[endpoint]

    async def close_all(self) -> None:
        """Close all managed connection pools."""
        for endpoint, pool in self._pools.items():
            await pool.close()
            logger.info("pool_closed", endpoint=endpoint)

        self._pools.clear()

    def get_stats(self) -> dict[str, dict[str, int]]:
        """Get statistics for all pools.

        Returns:
            Dictionary mapping endpoint to pool statistics
        """
        return {
            endpoint: {
                "active": pool.get_active_connections(),
                "available": pool.get_available_connections(),
                "max": pool.max_connections,
            }
            for endpoint, pool in self._pools.items()
        }


# Global pool manager instance
_pool_manager: Optional[ConnectionPoolManager] = None


def get_pool_manager() -> ConnectionPoolManager:
    """Get the global connection pool manager.

    Returns:
        Global ConnectionPoolManager instance
    """
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = ConnectionPoolManager()
    return _pool_manager


def reset_pool_manager() -> None:
    """Reset the global pool manager (useful for testing)."""
    global _pool_manager
    _pool_manager = None

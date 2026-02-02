"""Async Sub-LLM Call Manager for RLM systems.

This module provides the SubLLMManager class which manages asynchronous
sub-LLM calls with features like connection pooling, caching, batching,
and resilience patterns.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Coroutine, Dict, List, Optional, TypeVar, Union

import structlog

from rlm.config import get_settings
from rlm.llm.connection_pool import AsyncConnectionPool
from rlm.llm.enhanced_client import EnhancedLLMClient
from rlm.llm.interface import LLMClientInterface
from rlm.llm.prompts import get_sub_llm_system_prompt
from rlm.llm.session_cache import SessionCache
from rlm.types import LLMResponse

logger = structlog.get_logger()

T = TypeVar("T")


@dataclass
class SubLLMCall:
    """A single sub-LLM call request."""

    query: str
    context: Optional[str] = None
    priority: int = 5  # 1-10, lower = higher priority
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class SubLLMResult:
    """Result from a sub-LLM call."""

    call_id: str
    query: str
    response: Optional[str] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    from_cache: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if the call was successful."""
        return self.error is None and self.response is not None


@dataclass
class StreamingSubLLMResult:
    """Streaming result from a sub-LLM call."""

    call_id: str
    query: str
    chunk: Optional[str] = None
    is_complete: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SubLLMManager:
    """Manages async sub-LLM calls with batching, caching, and resilience.

    The SubLLMManager coordinates multiple sub-LLM calls with features:
    - Connection pooling for efficient HTTP reuse
    - Session-level caching to avoid redundant calls
    - Parallel execution with configurable concurrency
    - Partial failure handling (some calls can fail)
    - Circuit breaker integration for resilience
    - Progress tracking for long-running operations

    Example:
        ```python
        # Basic usage
        manager = SubLLMManager(max_concurrent=20)

        # Single call
        result = await manager.call(
            query="What is AI?",
            context="AI is a broad field...",
            session_id="session-123",
        )

        # Batch calls
        calls = [
            SubLLMCall(query="Q1", context="C1"),
            SubLLMCall(query="Q2", context="C2"),
        ]
        results = await manager.call_batch(calls, session_id="session-123")

        # Cleanup
        await manager.close()
        ```
    """

    def __init__(
        self,
        llm_client: Optional[LLMClientInterface] = None,
        max_concurrent: int = 20,
        enable_caching: bool = True,
        enable_connection_pool: bool = True,
        cache_ttl: int = 3600,
        cache_max_size: int = 1000,
        timeout: float = 30.0,
        retry_attempts: int = 3,
        retry_backoff: float = 1.5,
    ) -> None:
        """Initialize the Sub-LLM Manager.

        Args:
            llm_client: LLM client to use (creates EnhancedLLMClient if None)
            max_concurrent: Maximum concurrent sub-LLM calls
            enable_caching: Whether to enable session-level caching
            enable_connection_pool: Whether to use connection pooling
            cache_ttl: Cache time-to-live in seconds
            cache_max_size: Maximum cache entries per session
            timeout: Timeout for individual calls in seconds
            retry_attempts: Number of retry attempts for failed calls
            retry_backoff: Backoff multiplier for retries
        """
        settings = get_settings()

        # Initialize LLM client
        self._llm_client = llm_client or EnhancedLLMClient()

        # Configuration
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff
        self.enable_caching = enable_caching

        # Connection management
        self._connection_pool: Optional[AsyncConnectionPool] = None
        if enable_connection_pool:
            self._connection_pool = AsyncConnectionPool(
                max_connections=max_concurrent * 2,
                max_keepalive=max_concurrent,
                timeout=timeout,
            )

        # Caching
        self._cache: Optional[SessionCache] = None
        if enable_caching:
            self._cache = SessionCache(
                ttl_seconds=cache_ttl,
                max_size=cache_max_size,
            )

        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Statistics
        self._total_calls = 0
        self._successful_calls = 0
        self._failed_calls = 0
        self._cached_calls = 0

        logger.info(
            "sub_llm_manager_initialized",
            max_concurrent=max_concurrent,
            caching_enabled=enable_caching,
            connection_pool_enabled=enable_connection_pool,
            timeout=timeout,
        )

    async def call(
        self,
        query: str,
        context: Optional[str] = None,
        session_id: str = "default",
        priority: int = 5,
        stream: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[str, AsyncIterator[str]]:
        """Execute a single sub-LLM call.

        Args:
            query: The query to send to the LLM
            context: Optional context to include with the query
            session_id: Session identifier for caching
            priority: Priority level (1-10, lower = higher priority)
            stream: Whether to stream the response
            metadata: Optional metadata to track with the call

        Returns:
            Response string (or AsyncIterator if streaming)

        Example:
            ```python
            result = await manager.call(
                query="What is the main topic?",
                context="Document section...",
                session_id="doc-analysis-123",
            )
            ```
        """
        call = SubLLMCall(
            query=query,
            context=context,
            priority=priority,
            metadata=metadata or {},
        )

        if stream:
            return self._call_streaming(call, session_id)

        result = await self._execute_single(call, session_id)
        return result.response if result.success else f"[Error: {result.error}]"

    async def call_batch(
        self,
        calls: List[SubLLMCall],
        session_id: str = "default",
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, SubLLMResult]:
        """Execute multiple sub-LLM calls in parallel.

        Args:
            calls: List of SubLLMCall objects to execute
            session_id: Session identifier for caching
            on_progress: Optional callback(progress, total) for updates

        Returns:
            Dictionary mapping call IDs to SubLLMResult objects

        Example:
            ```python
            calls = [
                SubLLMCall(query="What is X?", context="Section 1..."),
                SubLLMCall(query="What is Y?", context="Section 2..."),
            ]
            results = await manager.call_batch(calls, session_id="analysis")

            for call_id, result in results.items():
                if result.success:
                    print(f"{call_id}: {result.response}")
            ```
        """
        if not calls:
            return {}

        logger.info(
            "batch_execution_started",
            session_id=session_id,
            call_count=len(calls),
            max_concurrent=self.max_concurrent,
        )

        # Execute all calls with semaphore-controlled concurrency
        semaphore = self._semaphore
        completed = 0

        async def execute_with_progress(call: SubLLMCall) -> SubLLMResult:
            nonlocal completed
            async with semaphore:
                result = await self._execute_single(call, session_id)
                completed += 1
                if on_progress:
                    on_progress(completed, len(calls))
                return result

        # Create tasks for all calls
        tasks = [asyncio.create_task(execute_with_progress(call)) for call in calls]

        # Wait for all to complete
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Build results dictionary
        results: Dict[str, SubLLMResult] = {}
        for i, result in enumerate(results_list):
            call_id = calls[i].id
            if isinstance(result, Exception):
                results[call_id] = SubLLMResult(
                    call_id=call_id,
                    query=calls[i].query,
                    error=str(result),
                )
            else:
                results[call_id] = result

        # Update statistics
        successful = sum(1 for r in results.values() if r.success)
        self._successful_calls += successful
        self._failed_calls += len(results) - successful

        logger.info(
            "batch_execution_complete",
            session_id=session_id,
            total=len(calls),
            successful=successful,
            failed=len(calls) - successful,
        )

        return results

    async def call_batch_streaming(
        self,
        calls: List[SubLLMCall],
        session_id: str = "default",
    ) -> AsyncIterator[StreamingSubLLMResult]:
        """Execute multiple calls and stream results as they complete.

        Args:
            calls: List of SubLLMCall objects to execute
            session_id: Session identifier for caching

        Yields:
            StreamingSubLLMResult objects as they complete

        Example:
            ```python
            async for result in manager.call_batch_streaming(calls, session_id):
                if result.is_complete:
                    print(f"Complete: {result.call_id}")
                elif result.chunk:
                    print(f"Chunk: {result.chunk}")
                elif result.error:
                    print(f"Error: {result.error}")
            ```
        """
        if not calls:
            return

        logger.info(
            "streaming_batch_started",
            session_id=session_id,
            call_count=len(calls),
        )

        semaphore = self._semaphore

        async def execute_streaming(call: SubLLMCall) -> None:
            async with semaphore:
                async for chunk in self._call_streaming(call, session_id):
                    yield StreamingSubLLMResult(
                        call_id=call.id,
                        query=call.query,
                        chunk=chunk,
                    )

                # Yield completion marker
                yield StreamingSubLLMResult(
                    call_id=call.id,
                    query=call.query,
                    is_complete=True,
                )

        # Create streaming tasks
        async def stream_task(call: SubLLMCall) -> AsyncIterator[StreamingSubLLMResult]:
            async with semaphore:
                full_response = ""
                try:
                    async for chunk in self._call_streaming(call, session_id):
                        full_response += chunk
                        yield StreamingSubLLMResult(
                            call_id=call.id,
                            query=call.query,
                            chunk=chunk,
                        )

                    # Cache the complete response
                    if self._cache:
                        self._cache.set(session_id, call.query, full_response, call.context)

                    yield StreamingSubLLMResult(
                        call_id=call.id,
                        query=call.query,
                        is_complete=True,
                    )
                except Exception as e:
                    yield StreamingSubLLMResult(
                        call_id=call.id,
                        query=call.query,
                        error=str(e),
                        is_complete=True,
                    )

        # Create all streaming tasks
        tasks = {asyncio.create_task(anext_safe(stream_task(call))): call for call in calls}

        # Stream results as they arrive
        while tasks:
            done, pending = await asyncio.wait(
                tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                call = tasks.pop(task)
                try:
                    result = task.result()
                    if result:
                        yield result

                        # If not complete, create continuation task
                        if not result.is_complete and not result.error:
                            next_task = asyncio.create_task(anext_safe(stream_task(call)))
                            tasks[next_task] = call
                except Exception as e:
                    yield StreamingSubLLMResult(
                        call_id=call.id,
                        query=call.query,
                        error=str(e),
                        is_complete=True,
                    )

    async def _execute_single(
        self,
        call: SubLLMCall,
        session_id: str,
    ) -> SubLLMResult:
        """Execute a single sub-LLM call with retries and caching.

        Args:
            call: SubLLMCall to execute
            session_id: Session identifier

        Returns:
            SubLLMResult with response or error
        """
        self._total_calls += 1

        # Check cache first
        if self._cache:
            cached = self._cache.get(session_id, call.query, call.context)
            if cached:
                self._cached_calls += 1
                logger.debug("cache_hit", session_id=session_id, query=call.query[:50])
                return SubLLMResult(
                    call_id=call.id,
                    query=call.query,
                    response=cached,
                    from_cache=True,
                    execution_time_ms=0.0,
                )

        # Build prompt
        system_prompt = get_sub_llm_system_prompt()
        user_prompt = call.query
        if call.context:
            user_prompt = f"""Context chunk:
```
{call.context[:50000]}
```

Question: {call.query}

Please analyze this chunk and answer the question."""

        # Execute with retries
        start_time = time.time()
        last_error: Optional[Exception] = None

        for attempt in range(self.retry_attempts):
            try:
                if self._connection_pool:
                    # Use connection pool
                    response = await self._connection_pool.execute(
                        self._make_llm_call,
                        user_prompt,
                        system_prompt,
                    )
                else:
                    # Direct call
                    response = await self._make_llm_call(user_prompt, system_prompt)

                execution_time = (time.time() - start_time) * 1000

                # Cache successful result
                if self._cache:
                    self._cache.set(
                        session_id,
                        call.query,
                        response.content,
                        call.context,
                    )

                self._successful_calls += 1

                return SubLLMResult(
                    call_id=call.id,
                    query=call.query,
                    response=response.content,
                    execution_time_ms=execution_time,
                    metadata={"usage": response.usage, "model": response.model},
                )

            except Exception as e:
                last_error = e
                logger.warning(
                    "sub_llm_call_attempt_failed",
                    session_id=session_id,
                    call_id=call.id,
                    attempt=attempt + 1,
                    error=str(e),
                )

                if attempt < self.retry_attempts - 1:
                    # Exponential backoff
                    wait_time = self.retry_backoff**attempt
                    await asyncio.sleep(wait_time)

        # All retries exhausted
        self._failed_calls += 1
        execution_time = (time.time() - start_time) * 1000

        return SubLLMResult(
            call_id=call.id,
            query=call.query,
            error=f"Failed after {self.retry_attempts} attempts: {last_error}",
            execution_time_ms=execution_time,
        )

    async def _make_llm_call(
        self,
        user_prompt: str,
        system_prompt: str,
    ) -> LLMResponse:
        """Make the actual LLM call.

        Args:
            user_prompt: User prompt
            system_prompt: System prompt

        Returns:
            LLMResponse from the client
        """
        return await self._llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )

    async def _call_streaming(
        self,
        call: SubLLMCall,
        session_id: str,
    ) -> AsyncIterator[str]:
        """Execute a streaming sub-LLM call.

        Args:
            call: SubLLMCall to execute
            session_id: Session identifier

        Yields:
            Response chunks as they arrive
        """
        # Build prompt
        system_prompt = get_sub_llm_system_prompt()
        user_prompt = call.query
        if call.context:
            user_prompt = f"""Context chunk:
```
{call.context[:50000]}
```

Question: {call.query}

Please analyze this chunk and answer the question."""

        try:
            async with self._semaphore:
                async for chunk in self._llm_client.generate_stream(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                ):
                    yield chunk
        except Exception as e:
            logger.error(
                "streaming_call_failed",
                session_id=session_id,
                call_id=call.id,
                error=str(e),
            )
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics.

        Returns:
            Dictionary with usage statistics
        """
        stats = {
            "total_calls": self._total_calls,
            "successful_calls": self._successful_calls,
            "failed_calls": self._failed_calls,
            "cached_calls": self._cached_calls,
            "success_rate": (
                self._successful_calls / self._total_calls if self._total_calls > 0 else 0.0
            ),
            "cache_hit_rate": (
                self._cached_calls / self._total_calls if self._total_calls > 0 else 0.0
            ),
            "max_concurrent": self.max_concurrent,
            "caching_enabled": self.enable_caching,
        }

        if self._cache:
            stats["cache_stats"] = self._cache.get_stats()

        return stats

    def invalidate_session_cache(self, session_id: str) -> int:
        """Invalidate all cached entries for a session.

        Args:
            session_id: Session to invalidate

        Returns:
            Number of entries removed
        """
        if self._cache:
            return self._cache.invalidate_session(session_id)
        return 0

    async def close(self) -> None:
        """Cleanup resources and close connections."""
        if self._connection_pool:
            await self._connection_pool.close()

        logger.info(
            "sub_llm_manager_closed",
            total_calls=self._total_calls,
            successful=self._successful_calls,
            failed=self._failed_calls,
        )

    async def __aenter__(self) -> "SubLLMManager":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - automatically closes."""
        await self.close()


async def anext_safe(aiter: AsyncIterator[T]) -> Optional[T]:
    """Safely get next item from async iterator.

    Args:
        aiter: Async iterator

    Returns:
        Next item or None if exhausted
    """
    try:
        return await aiter.__anext__()
    except StopAsyncIteration:
        return None

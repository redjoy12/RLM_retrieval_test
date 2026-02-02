"""Batch manager for parallel LLM request processing."""

import asyncio
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import structlog

from rlm.types import LLMResponse

logger = structlog.get_logger()

T = TypeVar("T")


@dataclass
class BatchRequest:
    """A single request in a batch."""

    id: str
    coro: Coroutine[Any, Any, LLMResponse]
    callback: Optional[Callable[[str, LLMResponse], None]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result from a batch execution."""

    request_id: str
    response: Optional[LLMResponse] = None
    error: Optional[Exception] = None
    execution_time_ms: float = 0.0

    @property
    def success(self) -> bool:
        """Check if request succeeded."""
        return self.error is None and self.response is not None


class BatchManager:
    """Manages batch processing of LLM requests with concurrency control."""

    def __init__(
        self,
        max_concurrent: int = 10,
        batch_timeout: float = 30.0,
        retry_failed: bool = False,
        max_retries: int = 2,
    ) -> None:
        """Initialize batch manager.

        Args:
            max_concurrent: Maximum concurrent requests
            batch_timeout: Timeout per batch in seconds
            retry_failed: Whether to retry failed requests
            max_retries: Maximum retry attempts
        """
        self.max_concurrent = max_concurrent
        self.batch_timeout = batch_timeout
        self.retry_failed = retry_failed
        self.max_retries = max_retries

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._pending: List[BatchRequest] = []

        logger.info(
            "batch_manager_initialized",
            max_concurrent=max_concurrent,
            timeout=batch_timeout,
        )

    def add_request(
        self,
        request_id: str,
        coro: Coroutine[Any, Any, LLMResponse],
        callback: Optional[Callable[[str, LLMResponse], None]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a request to the batch.

        Args:
            request_id: Unique request identifier
            coro: Coroutine that makes the LLM call
            callback: Optional callback for successful responses
            metadata: Optional metadata
        """
        request = BatchRequest(
            id=request_id,
            coro=coro,
            callback=callback,
            metadata=metadata or {},
        )
        self._pending.append(request)

    async def _execute_single(self, request: BatchRequest, attempt: int = 0) -> BatchResult:
        """Execute a single request with semaphore control.

        Args:
            request: Batch request
            attempt: Retry attempt number

        Returns:
            BatchResult
        """
        import time

        start_time = time.time()

        async with self._semaphore:
            try:
                response = await asyncio.wait_for(request.coro, timeout=self.batch_timeout)

                execution_time = (time.time() - start_time) * 1000

                # Call callback if provided
                if request.callback:
                    try:
                        request.callback(request.id, response)
                    except Exception as e:
                        logger.error(
                            "batch_callback_error",
                            request_id=request.id,
                            error=str(e),
                        )

                return BatchResult(
                    request_id=request.id,
                    response=response,
                    execution_time_ms=execution_time,
                )

            except asyncio.TimeoutError:
                execution_time = (time.time() - start_time) * 1000

                if self.retry_failed and attempt < self.max_retries:
                    logger.warning(
                        "batch_request_retry",
                        request_id=request.id,
                        attempt=attempt + 1,
                    )
                    return await self._execute_single(request, attempt + 1)

                return BatchResult(
                    request_id=request.id,
                    error=asyncio.TimeoutError(f"Request timed out after {self.batch_timeout}s"),
                    execution_time_ms=execution_time,
                )

            except Exception as e:
                execution_time = (time.time() - start_time) * 1000

                if self.retry_failed and attempt < self.max_retries:
                    logger.warning(
                        "batch_request_retry",
                        request_id=request.id,
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    return await self._execute_single(request, attempt + 1)

                return BatchResult(
                    request_id=request.id,
                    error=e,
                    execution_time_ms=execution_time,
                )

    async def execute(self, clear_pending: bool = True) -> Dict[str, BatchResult]:
        """Execute all pending requests in parallel.

        Args:
            clear_pending: Whether to clear pending list after execution

        Returns:
            Dictionary mapping request IDs to results
        """
        if not self._pending:
            return {}

        requests = self._pending.copy()
        if clear_pending:
            self._pending.clear()

        logger.info(
            "batch_execution_start",
            request_count=len(requests),
            max_concurrent=self.max_concurrent,
        )

        # Create tasks for all requests
        tasks = [asyncio.create_task(self._execute_single(req)) for req in requests]

        # Execute all tasks
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Build results dictionary
        results: Dict[str, BatchResult] = {}
        for i, result in enumerate(results_list):
            request_id = requests[i].id

            if isinstance(result, Exception):
                # Task itself raised an exception
                results[request_id] = BatchResult(request_id=request_id, error=result)
            else:
                results[request_id] = cast(BatchResult, result)

        # Log summary
        successful = sum(1 for r in results.values() if r.success)
        failed = len(results) - successful

        logger.info(
            "batch_execution_complete",
            total=len(results),
            successful=successful,
            failed=failed,
        )

        return results

    async def execute_stream(
        self,
    ) -> AsyncGenerator[BatchResult, None]:
        """Execute pending requests and stream results as they complete.

        Yields:
            BatchResult objects as they complete
        """
        if not self._pending:
            return

        requests = self._pending.copy()
        self._pending.clear()

        logger.info(
            "batch_stream_start",
            request_count=len(requests),
        )

        # Create tasks
        tasks = {asyncio.create_task(self._execute_single(req)): req.id for req in requests}

        # Yield results as they complete
        while tasks:
            done, pending_tasks = await asyncio.wait(
                tasks.keys(), return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                request_id = tasks.pop(task)
                try:
                    result = task.result()
                    yield result
                except Exception as e:
                    yield BatchResult(request_id=request_id, error=e)

    def clear(self) -> None:
        """Clear all pending requests."""
        count = len(self._pending)
        self._pending.clear()
        logger.info("batch_cleared", count=count)

    def get_pending_count(self) -> int:
        """Get number of pending requests."""
        return len(self._pending)

    async def execute_single(
        self,
        request_id: str,
        coro: Coroutine[Any, Any, LLMResponse],
        timeout: Optional[float] = None,
    ) -> BatchResult:
        """Execute a single request immediately.

        Args:
            request_id: Request identifier
            coro: Coroutine to execute
            timeout: Override timeout for this request

        Returns:
            BatchResult
        """
        import time

        start_time = time.time()
        effective_timeout = timeout or self.batch_timeout

        async with self._semaphore:
            try:
                response = await asyncio.wait_for(coro, timeout=effective_timeout)
                execution_time = (time.time() - start_time) * 1000

                return BatchResult(
                    request_id=request_id,
                    response=response,
                    execution_time_ms=execution_time,
                )

            except asyncio.TimeoutError:
                execution_time = (time.time() - start_time) * 1000
                return BatchResult(
                    request_id=request_id,
                    error=asyncio.TimeoutError(f"Request timed out after {effective_timeout}s"),
                    execution_time_ms=execution_time,
                )

            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                return BatchResult(
                    request_id=request_id,
                    error=e,
                    execution_time_ms=execution_time,
                )

    @staticmethod
    def create_from_calls(
        calls: List[Tuple[str, Coroutine[Any, Any, LLMResponse]]],
        max_concurrent: int = 10,
    ) -> "BatchManager":
        """Create batch manager from list of (id, coroutine) tuples.

        Args:
            calls: List of (request_id, coroutine) tuples
            max_concurrent: Maximum concurrent requests

        Returns:
            BatchManager with requests added
        """
        manager = BatchManager(max_concurrent=max_concurrent)

        for request_id, coro in calls:
            manager.add_request(request_id, coro)

        return manager

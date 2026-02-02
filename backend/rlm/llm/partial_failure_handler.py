"""Partial failure handler for resilient sub-LLM call processing."""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar

import structlog

logger = structlog.get_logger()

T = TypeVar("T")


class FailureStrategy(Enum):
    """Strategies for handling call failures."""

    RETRY = "retry"  # Retry with exponential backoff
    FALLBACK = "fallback"  # Use fallback response
    SKIP = "skip"  # Skip this call and continue
    ABORT = "abort"  # Abort entire batch


@dataclass
class FailurePolicy:
    """Policy for handling failures in a call."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    strategy: FailureStrategy = FailureStrategy.RETRY
    fallback_response: Optional[str] = None
    timeout: float = 30.0


@dataclass
class CallResult:
    """Result from a single call with failure metadata."""

    call_id: str
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0
    execution_time_ms: float = 0.0
    was_retried: bool = False
    fallback_used: bool = False


@dataclass
class BatchFailureSummary:
    """Summary of failures in a batch execution."""

    total_calls: int
    successful_calls: int
    failed_calls: int
    retried_calls: int
    fallback_used: int
    total_execution_time_ms: float
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    @property
    def is_partial_failure(self) -> bool:
        """Check if this was a partial failure (some succeeded)."""
        return 0 < self.failed_calls < self.total_calls

    @property
    def is_complete_failure(self) -> bool:
        """Check if all calls failed."""
        return self.failed_calls == self.total_calls and self.total_calls > 0


class PartialFailureHandler:
    """Handles partial failures in sub-LLM call batches.

    This class provides resilience patterns for handling failures in
    concurrent sub-LLM calls, including:
    - Individual retry with exponential backoff
    - Fallback responses for failed calls
    - Error isolation (one failure doesn't stop others)
    - Comprehensive failure reporting

    Example:
        ```python
        handler = PartialFailureHandler()

        async def make_call(id: str) -> str:
            # Your API call here
            return await api_call(id)

        # Execute with resilience
        results = await handler.execute_batch(
            calls=[("id1", make_call), ("id2", make_call)],
            policy=FailurePolicy(max_retries=3),
        )

        # Check results
        for call_id, result in results.items():
            if result.success:
                print(f"Success: {result.response}")
            else:
                print(f"Failed: {result.error}")
        ```
    """

    def __init__(self) -> None:
        """Initialize the partial failure handler."""
        self._total_handled = 0
        self._successful_handled = 0
        self._retried_calls = 0
        self._fallback_used = 0

        logger.info("partial_failure_handler_initialized")

    async def execute(
        self,
        call_id: str,
        coro: Callable[[], Coroutine[Any, Any, T]],
        policy: FailurePolicy,
    ) -> CallResult:
        """Execute a single call with failure handling.

        Args:
            call_id: Unique identifier for this call
            coro: Coroutine function to execute
            policy: Failure policy to apply

        Returns:
            CallResult with success/failure details
        """
        self._total_handled += 1
        start_time = time.time()
        last_error: Optional[Exception] = None

        for attempt in range(policy.max_retries + 1):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(coro(), timeout=policy.timeout)

                execution_time = (time.time() - start_time) * 1000

                if attempt > 0:
                    self._retried_calls += 1

                self._successful_handled += 1

                return CallResult(
                    call_id=call_id,
                    success=True,
                    response=str(result) if result is not None else None,
                    attempts=attempt + 1,
                    execution_time_ms=execution_time,
                    was_retried=attempt > 0,
                )

            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(
                    "call_timeout",
                    call_id=call_id,
                    attempt=attempt + 1,
                    timeout=policy.timeout,
                )

            except Exception as e:
                last_error = e
                logger.warning(
                    "call_failed",
                    call_id=call_id,
                    attempt=attempt + 1,
                    error=str(e),
                    error_type=type(e).__name__,
                )

            # Check if we should retry
            if attempt < policy.max_retries and policy.strategy == FailureStrategy.RETRY:
                # Exponential backoff
                delay = min(
                    policy.base_delay * (2 ** attempt),
                    policy.max_delay,
                )
                logger.info("retrying_call", call_id=call_id, delay=delay, attempt=attempt + 1)
                await asyncio.sleep(delay)
            else:
                # No more retries
                break

        # All attempts failed
        execution_time = (time.time() - start_time) * 1000

        # Try fallback if available
        if policy.fallback_response and policy.strategy == FailureStrategy.FALLBACK:
            self._fallback_used += 1
            return CallResult(
                call_id=call_id,
                success=True,  # Mark as success with fallback
                response=policy.fallback_response,
                attempts=policy.max_retries + 1,
                execution_time_ms=execution_time,
                was_retried=True,
                fallback_used=True,
            )

        return CallResult(
            call_id=call_id,
            success=False,
            error=f"Failed after {policy.max_retries + 1} attempts: {last_error}",
            attempts=policy.max_retries + 1,
            execution_time_ms=execution_time,
            was_retried=True,
        )

    async def execute_batch(
        self,
        calls: List[tuple[str, Callable[[], Coroutine[Any, Any, T]]]],
        policy: FailurePolicy,
        max_concurrent: int = 10,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, CallResult]:
        """Execute multiple calls with failure handling.

        Args:
            calls: List of (call_id, coroutine) tuples
            policy: Failure policy for all calls
            max_concurrent: Maximum concurrent executions
            on_progress: Optional callback(completed, total)

        Returns:
            Dictionary mapping call_ids to CallResult objects
        """
        if not calls:
            return {}

        semaphore = asyncio.Semaphore(max_concurrent)
        completed = 0

        async def execute_with_semaphore(
            call_id: str,
            coro: Callable[[], Coroutine[Any, Any, T]],
        ) -> CallResult:
            nonlocal completed
            async with semaphore:
                result = await self.execute(call_id, coro, policy)
                completed += 1
                if on_progress:
                    on_progress(completed, len(calls))
                return result

        # Create tasks
        tasks = [
            asyncio.create_task(execute_with_semaphore(call_id, coro))
            for call_id, coro in calls
        ]

        # Execute all
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Build results dict
        results: Dict[str, CallResult] = {}
        for i, result in enumerate(results_list):
            call_id = calls[i][0]
            if isinstance(result, Exception):
                results[call_id] = CallResult(
                    call_id=call_id,
                    success=False,
                    error=f"Task exception: {result}",
                )
            else:
                results[call_id] = result

        return results

    def get_batch_summary(self, results: Dict[str, CallResult]) -> BatchFailureSummary:
        """Generate a summary of batch execution results.

        Args:
            results: Dictionary of CallResult objects

        Returns:
            BatchFailureSummary with statistics
        """
        total_time = sum(r.execution_time_ms for r in results.values())
        errors = [r.error for r in results.values() if r.error]

        return BatchFailureSummary(
            total_calls=len(results),
            successful_calls=sum(1 for r in results.values() if r.success),
            failed_calls=sum(1 for r in results.values() if not r.success),
            retried_calls=sum(1 for r in results.values() if r.was_retried),
            fallback_used=sum(1 for r in results.values() if r.fallback_used),
            total_execution_time_ms=total_time,
            errors=errors[:10],  # Limit to first 10 errors
        )

    def should_retry_batch(self, summary: BatchFailureSummary, max_attempts: int = 3) -> bool:
        """Determine if a batch should be retried based on failure summary.

        Args:
            summary: BatchFailureSummary to analyze
            max_attempts: Maximum retry attempts for the batch

        Returns:
            True if batch should be retried
        """
        # Don't retry if all succeeded
        if summary.success_rate >= 1.0:
            return False

        # Don't retry if completely failed (likely systemic issue)
        if summary.is_complete_failure:
            return False

        # Retry partial failures
        if summary.is_partial_failure:
            return True

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics.

        Returns:
            Dictionary with handler statistics
        """
        return {
            "total_handled": self._total_handled,
            "successful": self._successful_handled,
            "failed": self._total_handled - self._successful_handled,
            "success_rate": (
                self._successful_handled / self._total_handled
                if self._total_handled > 0
                else 0.0
            ),
            "retried_calls": self._retried_calls,
            "fallback_used": self._fallback_used,
        }


def create_default_policy() -> FailurePolicy:
    """Create a default failure policy.

    Returns:
        FailurePolicy with sensible defaults
    """
    return FailurePolicy(
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        strategy=FailureStrategy.RETRY,
        timeout=30.0,
    )


def create_aggressive_retry_policy() -> FailurePolicy:
    """Create a policy with aggressive retry settings.

    Returns:
        FailurePolicy with more retries and shorter delays
    """
    return FailurePolicy(
        max_retries=5,
        base_delay=0.5,
        max_delay=10.0,
        strategy=FailureStrategy.RETRY,
        timeout=60.0,
    )


def create_fallback_policy(fallback: str) -> FailurePolicy:
    """Create a policy that uses fallback on failure.

    Args:
        fallback: Fallback response to use

    Returns:
        FailurePolicy with fallback strategy
    """
    return FailurePolicy(
        max_retries=1,
        base_delay=1.0,
        max_delay=5.0,
        strategy=FailureStrategy.FALLBACK,
        fallback_response=fallback,
        timeout=15.0,
    )

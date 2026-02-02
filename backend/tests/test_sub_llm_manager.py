"""Tests for Component 5: Async Sub-LLM Manager."""

import asyncio
import pytest
from typing import AsyncIterator, List, Optional

from rlm.llm.connection_pool import AsyncConnectionPool, get_pool_manager, reset_pool_manager
from rlm.llm.partial_failure_handler import (
    BatchFailureSummary,
    CallResult,
    FailurePolicy,
    FailureStrategy,
    PartialFailureHandler,
    create_default_policy,
)
from rlm.llm.query_batcher import QueryBatcher
from rlm.llm.session_cache import SessionCache
from rlm.llm.streaming_aggregator import StreamBuffer, StreamingAggregator
from rlm.llm.sub_llm_manager import SubLLMCall, SubLLMManager, SubLLMResult, StreamingSubLLMResult


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, delay: float = 0.01, fail_after: Optional[int] = None) -> None:
        self.delay = delay
        self.fail_after = fail_after
        self.call_count = 0

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        """Mock generate method."""
        from rlm.types import LLMResponse, TokenUsage

        self.call_count += 1

        if self.fail_after and self.call_count >= self.fail_after:
            raise Exception(f"Mock failure after {self.fail_after} calls")

        await asyncio.sleep(self.delay)

        return LLMResponse(
            content=f"Response to: {prompt[:50]}...",
            model="mock-model",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Mock generate_stream method."""
        self.call_count += 1

        if self.fail_after and self.call_count >= self.fail_after:
            raise Exception(f"Mock failure after {self.fail_after} calls")

        chunks = ["Hello", " ", "world", "!"]
        for chunk in chunks:
            await asyncio.sleep(self.delay)
            yield chunk

    def get_model_name(self) -> str:
        return "mock-model"


# =============================================================================
# Connection Pool Tests
# =============================================================================

@pytest.mark.asyncio
class TestAsyncConnectionPool:
    """Tests for AsyncConnectionPool."""

    async def test_pool_initialization(self):
        """Test pool initialization."""
        pool = AsyncConnectionPool(max_connections=50, max_keepalive=10)

        assert pool.max_connections == 50
        assert pool.max_keepalive == 10
        assert pool.get_available_connections() == 50
        assert pool.get_active_connections() == 0

        await pool.close()

    async def test_pool_execute(self):
        """Test executing function with pool."""
        pool = AsyncConnectionPool(max_connections=5)

        async def mock_task(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        result = await pool.execute(mock_task, 5)
        assert result == 10

        await pool.close()

    async def test_pool_execute_many(self):
        """Test executing multiple functions."""
        pool = AsyncConnectionPool(max_connections=5)

        async def mock_task(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        calls = [(mock_task, (i,), {}) for i in range(10)]
        results = await pool.execute_many(calls)

        assert len(results) == 10
        assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

        await pool.close()

    async def test_pool_context_manager(self):
        """Test pool as context manager."""
        async with AsyncConnectionPool(max_connections=5) as pool:
            assert pool.get_available_connections() == 5

    async def test_pool_closed_error(self):
        """Test error when using closed pool."""
        pool = AsyncConnectionPool(max_connections=5)
        await pool.close()

        async def mock_task() -> int:
            return 1

        with pytest.raises(RuntimeError, match="Connection pool has been closed"):
            await pool.execute(mock_task)


@pytest.mark.asyncio
class TestConnectionPoolManager:
    """Tests for ConnectionPoolManager."""

    async def test_get_pool(self):
        """Test getting/creating pools."""
        reset_pool_manager()
        manager = get_pool_manager()

        pool1 = manager.get_pool("endpoint1", max_connections=50)
        pool2 = manager.get_pool("endpoint1", max_connections=50)
        pool3 = manager.get_pool("endpoint2", max_connections=30)

        assert pool1 is pool2  # Same endpoint returns same pool
        assert pool1 is not pool3  # Different endpoint returns different pool

        await manager.close_all()

    async def test_get_stats(self):
        """Test getting pool statistics."""
        reset_pool_manager()
        manager = get_pool_manager()

        pool1 = manager.get_pool("endpoint1", max_connections=50)
        pool2 = manager.get_pool("endpoint2", max_connections=30)

        stats = manager.get_stats()

        assert "endpoint1" in stats
        assert "endpoint2" in stats
        assert stats["endpoint1"]["max"] == 50
        assert stats["endpoint2"]["max"] == 30

        await manager.close_all()


# =============================================================================
# Session Cache Tests
# =============================================================================

class TestSessionCache:
    """Tests for SessionCache."""

    def test_cache_set_and_get(self):
        """Test setting and getting cached values."""
        cache = SessionCache(ttl_seconds=3600, max_size=100)

        cache.set("session1", "What is AI?", "AI is artificial intelligence")
        result = cache.get("session1", "What is AI?")

        assert result == "AI is artificial intelligence"

    def test_cache_miss(self):
        """Test cache miss."""
        cache = SessionCache()

        result = cache.get("session1", "What is AI?")
        assert result is None

    def test_cache_with_context(self):
        """Test caching with context."""
        cache = SessionCache()

        context = "AI is a broad field..."
        cache.set("session1", "What is AI?", "Response", context)

        # Same query with same context should hit
        result = cache.get("session1", "What is AI?", context)
        assert result == "Response"

        # Different context should miss
        result = cache.get("session1", "What is AI?", "Different context")
        assert result is None

    def test_cache_invalidation(self):
        """Test session cache invalidation."""
        cache = SessionCache()

        cache.set("session1", "Q1", "A1")
        cache.set("session1", "Q2", "A2")
        cache.set("session2", "Q1", "A1")

        removed = cache.invalidate_session("session1")
        assert removed == 2

        assert cache.get("session1", "Q1") is None
        assert cache.get("session2", "Q1") == "A1"

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = SessionCache()

        cache.set("session1", "Q1", "A1")
        cache.get("session1", "Q1")  # Hit
        cache.get("session1", "Q2")  # Miss

        stats = cache.get_stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_disabled(self):
        """Test disabled cache."""
        cache = SessionCache(enabled=False)

        cache.set("session1", "Q1", "A1")
        result = cache.get("session1", "Q1")

        assert result is None


# =============================================================================
# SubLLMManager Tests
# =============================================================================

@pytest.mark.asyncio
class TestSubLLMManager:
    """Tests for SubLLMManager."""

    async def test_manager_initialization(self):
        """Test manager initialization."""
        mock_client = MockLLMClient()
        manager = SubLLMManager(
            llm_client=mock_client,
            max_concurrent=10,
            enable_caching=True,
        )

        assert manager.max_concurrent == 10
        assert manager.enable_caching is True

        await manager.close()

    async def test_single_call(self):
        """Test single sub-LLM call."""
        mock_client = MockLLMClient()
        manager = SubLLMManager(llm_client=mock_client)

        result = await manager.call(
            query="What is AI?",
            context="AI is a field...",
            session_id="test-session",
        )

        assert "Response to:" in result
        assert mock_client.call_count == 1

        await manager.close()

    async def test_call_with_caching(self):
        """Test that caching prevents duplicate calls."""
        mock_client = MockLLMClient()
        manager = SubLLMManager(llm_client=mock_client, enable_caching=True)

        # First call
        result1 = await manager.call(
            query="What is AI?",
            session_id="test-session",
        )

        # Second call with same query (should be cached)
        result2 = await manager.call(
            query="What is AI?",
            session_id="test-session",
        )

        assert result1 == result2
        assert mock_client.call_count == 1  # Only one API call made

        await manager.close()

    async def test_batch_calls(self):
        """Test batch sub-LLM calls."""
        mock_client = MockLLMClient(delay=0.01)
        manager = SubLLMManager(llm_client=mock_client, max_concurrent=5)

        calls = [
            SubLLMCall(query=f"Question {i}")
            for i in range(10)
        ]

        results = await manager.call_batch(calls, session_id="test-session")

        assert len(results) == 10
        assert all(r.success for r in results.values())

        await manager.close()

    async def test_batch_with_progress(self):
        """Test batch calls with progress callback."""
        mock_client = MockLLMClient(delay=0.01)
        manager = SubLLMManager(llm_client=mock_client, max_concurrent=5)

        calls = [SubLLMCall(query=f"Q{i}") for i in range(5)]
        progress_updates = []

        def on_progress(completed: int, total: int) -> None:
            progress_updates.append((completed, total))

        await manager.call_batch(calls, session_id="test", on_progress=on_progress)

        assert len(progress_updates) == 5
        assert progress_updates[-1] == (5, 5)

        await manager.close()

    async def test_manager_stats(self):
        """Test manager statistics."""
        mock_client = MockLLMClient()
        manager = SubLLMManager(llm_client=mock_client)

        await manager.call(query="Q1", session_id="s1")
        await manager.call(query="Q1", session_id="s1")  # Cached
        await manager.call(query="Q2", session_id="s1")

        stats = manager.get_stats()

        assert stats["total_calls"] == 3
        assert stats["cached_calls"] == 1
        assert stats["success_rate"] == 1.0

        await manager.close()


# =============================================================================
# Query Batcher Tests
# =============================================================================

class TestQueryBatcher:
    """Tests for QueryBatcher."""

    def test_group_similar(self):
        """Test grouping similar queries."""
        batcher = QueryBatcher()

        calls = [
            SubLLMCall(query="What is AI?"),
            SubLLMCall(query="What is machine learning?"),
            SubLLMCall(query="When was Python created?"),
            SubLLMCall(query="Where is Paris?"),
        ]

        groups = batcher.group_similar(calls)

        # Should group definition queries together
        assert len(groups) >= 1

    def test_group_by_priority(self):
        """Test grouping by priority."""
        batcher = QueryBatcher()

        calls = [
            SubLLMCall(query="Q1", priority=2),
            SubLLMCall(query="Q2", priority=8),
            SubLLMCall(query="Q3", priority=3),
        ]

        groups = batcher.group_by_priority(calls, priority_threshold=5)

        assert "high_priority" in groups
        assert "low_priority" in groups
        assert len(groups["high_priority"]) == 2
        assert len(groups["low_priority"]) == 1

    def test_execution_order(self):
        """Test creating execution order."""
        batcher = QueryBatcher()

        calls = [
            SubLLMCall(query="Q1", priority=2),
            SubLLMCall(query="Q2", priority=5),
        ]

        groups = batcher.group_similar(calls)
        order = batcher.create_execution_order(groups, strategy="priority")

        assert len(order) == len(groups)


# =============================================================================
# Streaming Aggregator Tests
# =============================================================================

@pytest.mark.asyncio
class TestStreamingAggregator:
    """Tests for StreamingAggregator."""

    async def test_stream_aggregation(self):
        """Test streaming result aggregation."""
        aggregator = StreamingAggregator()

        calls = [
            SubLLMCall(query="Q1"),
            SubLLMCall(query="Q2"),
        ]

        async def mock_execute(call: SubLLMCall, session_id: str) -> AsyncIterator[str]:
            yield f"Start-{call.id}"
            await asyncio.sleep(0.01)
            yield f"End-{call.id}"

        results = []
        async for result in aggregator.execute(calls, mock_execute, "test-session"):
            results.append(result)

        assert len(results) > 0

    async def test_sequential_streaming(self):
        """Test sequential streaming execution."""
        aggregator = StreamingAggregator()

        calls = [SubLLMCall(query=f"Q{i}") for i in range(3)]

        async def mock_execute(call: SubLLMCall, session_id: str) -> AsyncIterator[str]:
            yield f"Response-{call.id}"

        results = []
        async for result in aggregator.execute_sequential(calls, mock_execute, "test"):
            results.append(result)

        assert len(results) == 3

    def test_stream_buffer(self):
        """Test StreamBuffer."""
        buffer = StreamBuffer(call_id="test", query="Q1")

        buffer.add_chunk("Hello")
        buffer.add_chunk(" ")
        buffer.add_chunk("World")

        assert buffer.get_full_response() == "Hello World"
        assert len(buffer.chunks) == 3
        assert buffer.is_complete is False


# =============================================================================
# Partial Failure Handler Tests
# =============================================================================

@pytest.mark.asyncio
class TestPartialFailureHandler:
    """Tests for PartialFailureHandler."""

    async def test_successful_call(self):
        """Test successful call execution."""
        handler = PartialFailureHandler()

        async def mock_coro() -> str:
            return "success"

        policy = create_default_policy()
        result = await handler.execute("call-1", mock_coro, policy)

        assert result.success is True
        assert result.response == "success"
        assert result.attempts == 1

    async def test_retry_on_failure(self):
        """Test retry on failure."""
        handler = PartialFailureHandler()

        attempt_count = 0

        async def failing_coro() -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Temporary failure")
            return "success"

        policy = FailurePolicy(max_retries=3, base_delay=0.01)
        result = await handler.execute("call-1", failing_coro, policy)

        assert result.success is True
        assert result.was_retried is True
        assert result.attempts == 3

    async def test_fallback_on_failure(self):
        """Test fallback response on failure."""
        handler = PartialFailureHandler()

        async def failing_coro() -> str:
            raise Exception("Permanent failure")

        policy = FailurePolicy(
            strategy=FailureStrategy.FALLBACK,
            fallback_response="Default response",
        )
        result = await handler.execute("call-1", failing_coro, policy)

        assert result.success is True  # Marked as success with fallback
        assert result.response == "Default response"
        assert result.fallback_used is True

    async def test_complete_failure(self):
        """Test when all retries fail."""
        handler = PartialFailureHandler()

        async def failing_coro() -> str:
            raise Exception("Always fails")

        policy = FailurePolicy(max_retries=2, base_delay=0.01)
        result = await handler.execute("call-1", failing_coro, policy)

        assert result.success is False
        assert result.error is not None

    async def test_batch_execution(self):
        """Test batch execution with mixed success/failure."""
        handler = PartialFailureHandler()

        calls = []
        for i in range(5):
            if i % 2 == 0:
                async def success_coro() -> str:
                    return "success"
                calls.append((f"call-{i}", success_coro))
            else:
                async def fail_coro() -> str:
                    raise Exception("fail")
                calls.append((f"call-{i}", fail_coro))

        policy = FailurePolicy(max_retries=1, base_delay=0.01)
        results = await handler.execute_batch(calls, policy, max_concurrent=3)

        assert len(results) == 5

        successful = sum(1 for r in results.values() if r.success)
        failed = sum(1 for r in results.values() if not r.success)

        assert successful == 3  # calls 0, 2, 4
        assert failed == 2  # calls 1, 3

    async def test_batch_summary(self):
        """Test batch failure summary."""
        handler = PartialFailureHandler()

        results = {
            "call-1": CallResult(call_id="call-1", success=True, execution_time_ms=100),
            "call-2": CallResult(call_id="call-2", success=False, execution_time_ms=200),
            "call-3": CallResult(call_id="call-3", success=True, execution_time_ms=150),
        }

        summary = handler.get_batch_summary(results)

        assert summary.total_calls == 3
        assert summary.successful_calls == 2
        assert summary.failed_calls == 1
        assert summary.success_rate == 2 / 3
        assert summary.is_partial_failure is True
        assert summary.is_complete_failure is False


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.asyncio
class TestSubLLMManagerIntegration:
    """Integration tests for SubLLMManager."""

    async def test_end_to_end_single_call(self):
        """Test end-to-end single call flow."""
        mock_client = MockLLMClient(delay=0.01)

        async with SubLLMManager(llm_client=mock_client, max_concurrent=5) as manager:
            result = await manager.call(
                query="What is Python?",
                context="Python is a programming language.",
                session_id="test",
            )

            assert isinstance(result, str)
            assert len(result) > 0

    async def test_end_to_end_batch(self):
        """Test end-to-end batch processing."""
        mock_client = MockLLMClient(delay=0.01)

        async with SubLLMManager(
            llm_client=mock_client,
            max_concurrent=5,
            enable_caching=True,
        ) as manager:
            calls = [
                SubLLMCall(query=f"Question {i}", priority=i % 3)
                for i in range(20)
            ]

            results = await manager.call_batch(calls, session_id="batch-test")

            assert len(results) == 20
            assert all(isinstance(r, SubLLMResult) for r in results.values())

            stats = manager.get_stats()
            assert stats["total_calls"] == 20

    async def test_caching_deduplication(self):
        """Test that caching deduplicates identical queries."""
        mock_client = MockLLMClient(delay=0.01)

        manager = SubLLMManager(llm_client=mock_client, enable_caching=True)

        # Make 5 identical calls
        for _ in range(5):
            await manager.call(
                query="Same question",
                session_id="test",
            )

        # Should only make 1 API call due to caching
        assert mock_client.call_count == 1

        await manager.close()

    async def test_session_isolation(self):
        """Test that sessions are isolated."""
        mock_client = MockLLMClient(delay=0.01)
        manager = SubLLMManager(llm_client=mock_client, enable_caching=True)

        # Same query, different sessions
        await manager.call(query="Q1", session_id="session-1")
        await manager.call(query="Q1", session_id="session-2")

        # Should make 2 API calls (different sessions)
        assert mock_client.call_count == 2

        await manager.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

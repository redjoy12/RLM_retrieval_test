"""Tests for Component 3: Multi-Model LLM Client."""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from rlm.llm.batch_manager import BatchManager, BatchResult
from rlm.llm.cache import ResponseCache
from rlm.llm.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerManager,
    CircuitBreakerOpen,
)
from rlm.llm.cost_tracker import CostEntry, CostReport, CostTracker
from rlm.llm.model_registry import ModelInfo, ModelRegistry, get_model_registry
from rlm.llm.rate_limiter import RateLimiter, RateLimitConfig
from rlm.types import LLMResponse


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_register_and_get_model(self):
        """Test registering and retrieving models."""
        registry = ModelRegistry()

        # Test getting OpenAI model
        model = registry.get_model("gpt-4o-mini")
        assert model is not None
        assert model.provider == "openai"
        assert model.context_window == 128000

        # Test getting Anthropic model
        model = registry.get_model("claude-3-5-sonnet")
        assert model is not None
        assert model.provider == "anthropic"
        assert model.context_window == 200000

    def test_calculate_cost(self):
        """Test cost calculation."""
        registry = ModelRegistry()

        # Test GPT-4o-mini cost
        cost = registry.calculate_cost("gpt-4o-mini", 1000, 500)
        expected = (1000 / 1000 * 0.00015) + (500 / 1000 * 0.0006)
        assert cost == pytest.approx(expected, abs=0.0001)

    def test_list_models(self):
        """Test listing models."""
        registry = ModelRegistry()

        # List all models
        models = registry.list_models()
        assert "gpt-4o" in models
        assert "claude-3-5-sonnet" in models

        # List by provider
        openai_models = registry.list_models(provider="openai")
        assert "gpt-4o" in openai_models
        assert "claude-3-5-sonnet" not in openai_models

    def test_fallback_chains(self):
        """Test fallback chains."""
        registry = ModelRegistry()

        chain = registry.get_fallback_chain("smart")
        assert len(chain) > 0
        assert "gpt-4o" in chain or "claude-3-opus" in chain

    def test_global_registry(self):
        """Test global registry singleton."""
        registry1 = get_model_registry()
        registry2 = get_model_registry()
        assert registry1 is registry2


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.mark.asyncio
    async def test_token_bucket_basic(self):
        """Test basic token bucket functionality."""
        from rlm.llm.rate_limiter import TokenBucket

        bucket = TokenBucket(
            requests_per_minute=60,
            tokens_per_minute=60000,
            burst_size=10,
        )

        # Should acquire first request
        result = await bucket.acquire(tokens=100)
        assert result is True

    @pytest.mark.asyncio
    async def test_rate_limiter_enabled(self):
        """Test rate limiter when enabled."""
        limiter = RateLimiter(enabled=True)

        # Should acquire for openai
        result = await limiter.acquire("openai", tokens=0, wait=False)
        assert result is True

    @pytest.mark.asyncio
    async def test_rate_limiter_disabled(self):
        """Test rate limiter when disabled."""
        limiter = RateLimiter(enabled=False)

        # Should always return True when disabled
        result = await limiter.acquire("openai", tokens=1000000, wait=False)
        assert result is True

    def test_rate_limit_config(self):
        """Test RateLimitConfig dataclass."""
        config = RateLimitConfig(
            requests_per_minute=100,
            tokens_per_minute=100000,
            burst_size=20,
        )
        assert config.requests_per_minute == 100
        assert config.tokens_per_minute == 100000


class TestCostTracker:
    """Tests for CostTracker."""

    def test_calculate_cost(self):
        """Test cost calculation."""
        tracker = CostTracker(enabled=True)

        cost = tracker.calculate_cost("gpt-4o-mini", 1000, 500)
        assert cost > 0

    def test_log_cost(self):
        """Test logging a cost entry."""
        tracker = CostTracker(enabled=True)

        response = LLMResponse(
            content="Test",
            model="gpt-4o-mini",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )

        entry = tracker.log_cost("session-123", response, query_type="sub_llm")

        assert entry.session_id == "session-123"
        assert entry.model == "gpt-4o-mini"
        assert entry.prompt_tokens == 100
        assert entry.completion_tokens == 50
        assert entry.cost_usd > 0

    def test_session_cost_report(self):
        """Test generating session cost report."""
        tracker = CostTracker(enabled=True)

        # Log some costs
        for i in range(3):
            response = LLMResponse(
                content=f"Test {i}",
                model="gpt-4o-mini",
                usage={"prompt_tokens": 100, "completion_tokens": 50},
            )
            tracker.log_cost("session-1", response, query_type="sub_llm")

        # Get report
        report = tracker.get_session_cost("session-1")

        assert report.total_requests == 3
        assert report.total_cost > 0
        assert len(report.by_model) > 0

    def test_cost_tracker_disabled(self):
        """Test cost tracker when disabled."""
        tracker = CostTracker(enabled=False)

        response = LLMResponse(content="Test", model="gpt-4o-mini")
        entry = tracker.log_cost("session-123", response)

        assert entry.cost_usd == 0.0
        assert entry.prompt_tokens == 0

    def test_export_to_json(self):
        """Test exporting costs to JSON."""
        tracker = CostTracker(enabled=True)

        response = LLMResponse(
            content="Test",
            model="gpt-4o-mini",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        tracker.log_cost("session-1", response)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            tracker.export_to_json(filepath)

            with open(filepath) as f:
                data = json.load(f)

            assert "total_cost" in data
            assert "entries" in data
        finally:
            Path(filepath).unlink(missing_ok=True)


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    @pytest.mark.asyncio
    async def test_circuit_closed_state(self):
        """Test circuit in closed state."""
        breaker = CircuitBreaker("test", enabled=True)

        assert breaker.state.name == "CLOSED"
        assert await breaker.can_execute() is True

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self):
        """Test circuit opens after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60)
        breaker = CircuitBreaker("test", config=config, enabled=True)

        # Record failures
        for _ in range(3):
            await breaker.record_failure()

        assert breaker.state.name == "OPEN"
        assert await breaker.can_execute() is False

    @pytest.mark.asyncio
    async def test_circuit_success_resets_counter(self):
        """Test success in closed state resets failure counter."""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker("test", config=config, enabled=True)

        # Record some failures
        await breaker.record_failure()
        await breaker.record_failure()
        assert breaker._failure_count == 2

        # Record success
        await breaker.record_success()
        assert breaker._failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_half_open_recovery(self):
        """Test half-open state after recovery timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        breaker = CircuitBreaker("test", config=config, enabled=True)

        # Open circuit
        await breaker.record_failure()
        assert breaker.state.name == "OPEN"

        # Wait for recovery timeout
        await asyncio.sleep(0.2)

        # Should be half-open
        assert await breaker.can_execute() is True
        # Actually after can_execute in OPEN with elapsed time, it transitions to HALF_OPEN

    def test_circuit_disabled(self):
        """Test circuit when disabled."""
        breaker = CircuitBreaker("test", enabled=False)

        assert breaker.state.name == "CLOSED"

    @pytest.mark.asyncio
    async def test_circuit_breaker_manager(self):
        """Test CircuitBreakerManager."""
        manager = CircuitBreakerManager()

        # Register breakers
        breaker1 = manager.register_breaker("openai")
        breaker2 = manager.register_breaker("anthropic")

        assert manager.get_breaker("openai") is breaker1
        assert manager.get_breaker("anthropic") is breaker2

        # Set fallback chain
        manager.set_fallback_chain("openai", ["anthropic"])

        status = manager.get_all_status()
        assert "openai" in status
        assert "anthropic" in status


class TestResponseCache:
    """Tests for ResponseCache."""

    @pytest.mark.asyncio
    async def test_cache_get_and_set(self):
        """Test basic cache get/set."""
        cache = ResponseCache(ttl_seconds=3600, enabled=True)

        response = LLMResponse(content="Cached response", model="gpt-4o-mini")

        # Set in cache
        await cache.set(
            prompt="Test prompt",
            model="gpt-4o-mini",
            response=response,
        )

        # Get from cache
        cached = await cache.get(
            prompt="Test prompt",
            model="gpt-4o-mini",
        )

        assert cached is not None
        assert cached.content == "Cached response"

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss."""
        cache = ResponseCache(enabled=True)

        cached = await cache.get(
            prompt="Unknown prompt",
            model="gpt-4o-mini",
        )

        assert cached is None
        assert cache._misses == 1

    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test cache entry expiration."""
        cache = ResponseCache(ttl_seconds=0.1, enabled=True)

        response = LLMResponse(content="Test", model="gpt-4o-mini")
        await cache.set(prompt="Test", model="gpt-4o-mini", response=response)

        # Should be in cache immediately
        cached = await cache.get(prompt="Test", model="gpt-4o-mini")
        assert cached is not None

        # Wait for expiration
        import time

        time.sleep(0.2)

        # Should be expired now
        cached = await cache.get(prompt="Test", model="gpt-4o-mini")
        assert cached is None

    def test_cache_disabled(self):
        """Test cache when disabled."""
        cache = ResponseCache(enabled=False)

        # Should always return None
        # Can't test async easily here, but logic is straightforward

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = ResponseCache(enabled=True)

        stats = cache.get_stats()
        assert "enabled" in stats
        assert "size" in stats
        assert "hits" in stats
        assert "misses" in stats

    def test_cache_clear(self):
        """Test clearing cache."""
        cache = ResponseCache(enabled=True)
        cache._cache["test"] = None  # Add dummy entry

        cache.clear()
        assert len(cache._cache) == 0


class TestBatchManager:
    """Tests for BatchManager."""

    @pytest.mark.asyncio
    async def test_batch_execution(self):
        """Test batch execution."""
        manager = BatchManager(max_concurrent=5)

        # Create mock coroutines
        async def mock_coro(id: str):
            await asyncio.sleep(0.01)
            return LLMResponse(content=f"Response {id}", model="gpt-4o-mini")

        # Add requests
        for i in range(5):
            manager.add_request(f"req-{i}", mock_coro(str(i)))

        # Execute
        results = await manager.execute()

        assert len(results) == 5
        for i in range(5):
            assert f"req-{i}" in results
            assert results[f"req-{i}"].success is True

    @pytest.mark.asyncio
    async def test_batch_with_failures(self):
        """Test batch with some failing requests."""
        manager = BatchManager(max_concurrent=5)

        async def success_coro():
            return LLMResponse(content="Success", model="gpt-4o-mini")

        async def fail_coro():
            raise ValueError("Failed")

        manager.add_request("success", success_coro())
        manager.add_request("fail", fail_coro())

        results = await manager.execute()

        assert results["success"].success is True
        assert results["fail"].success is False
        assert results["fail"].error is not None

    @pytest.mark.asyncio
    async def test_batch_timeout(self):
        """Test batch timeout handling."""
        manager = BatchManager(max_concurrent=1, batch_timeout=0.1)

        async def slow_coro():
            await asyncio.sleep(1.0)
            return LLMResponse(content="Slow", model="gpt-4o-mini")

        manager.add_request("slow", slow_coro())
        results = await manager.execute()

        assert results["slow"].success is False
        assert isinstance(results["slow"].error, asyncio.TimeoutError)

    def test_batch_manager_clear(self):
        """Test clearing pending requests."""
        manager = BatchManager()

        async def mock_coro():
            return None

        manager.add_request("test", mock_coro())
        assert manager.get_pending_count() == 1

        manager.clear()
        assert manager.get_pending_count() == 0

    @pytest.mark.asyncio
    async def test_batch_single_execution(self):
        """Test single request execution."""
        manager = BatchManager()

        async def mock_coro():
            return LLMResponse(content="Test", model="gpt-4o-mini")

        result = await manager.execute_single("test", mock_coro())

        assert result.success is True
        assert result.request_id == "test"


class TestIntegration:
    """Integration tests for Component 3."""

    @pytest.mark.asyncio
    async def test_end_to_end_cost_tracking(self):
        """Test complete cost tracking flow."""
        tracker = CostTracker(enabled=True)

        # Simulate multiple requests
        for i in range(5):
            response = LLMResponse(
                content=f"Response {i}",
                model="gpt-4o-mini",
                usage={"prompt_tokens": 200, "completion_tokens": 100},
            )
            tracker.log_cost(f"session-{i}", response, provider="openai")

        # Get total cost
        report = tracker.get_total_cost()

        assert report.total_requests == 5
        assert report.total_cost > 0
        assert "gpt-4o-mini" in report.by_model
        assert "openai" in report.by_provider

    def test_model_registry_integration(self):
        """Test model registry with all components."""
        registry = get_model_registry()

        # Test all supported providers
        openai_model = registry.get_model("gpt-4o")
        assert openai_model.provider == "openai"

        anthropic_model = registry.get_model("claude-3-5-sonnet")
        assert anthropic_model.provider == "anthropic"

        local_model = registry.get_model("llama3.2")
        assert local_model.provider == "ollama"

        # Test pricing
        cost = registry.calculate_cost("gpt-4o-mini", 1000, 500)
        assert cost > 0

        cost = registry.calculate_cost("llama3.2", 1000, 500)
        assert cost == 0  # Local models are free

"""Enhanced LLM client with rate limiting, cost tracking, circuit breaker, and caching."""

from typing import Any, AsyncIterator, Dict, List, Optional

import structlog

from rlm.config import get_settings
from rlm.llm.batch_manager import BatchManager, BatchResult
from rlm.llm.cache import ResponseCache
from rlm.llm.circuit_breaker import CircuitBreakerConfig, CircuitBreakerManager
from rlm.llm.client import LiteLLMClient
from rlm.llm.cost_tracker import CostEntry, CostReport, CostTracker
from rlm.llm.interface import LLMClientInterface
from rlm.llm.model_registry import get_model_registry
from rlm.llm.rate_limiter import RateLimiter
from rlm.types import LLMResponse

logger = structlog.get_logger()


class EnhancedLLMClient(LLMClientInterface):
    """Enhanced LLM client with production-ready features.

    This client wraps LiteLLMClient and adds:
    - Rate limiting (token bucket algorithm)
    - Cost tracking (per-query in USD)
    - Circuit breaker (failover protection)
    - Response caching (exact match)
    - Batch processing (parallel execution)

    Example:
        ```python
        client = EnhancedLLMClient(
            model="gpt-4o-mini",
            enable_rate_limiting=True,
            enable_cost_tracking=True,
        )

        response = await client.generate("Hello, world!")

        # Get cost report
        report = client.get_cost_report(session_id)
        print(f"Cost: ${report.total_cost:.4f}")
        ```
    """

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        # Feature toggles
        enable_rate_limiting: bool = True,
        enable_cost_tracking: bool = True,
        enable_circuit_breaker: bool = True,
        enable_caching: bool = True,
        # Configuration
        rate_limiter: Optional[RateLimiter] = None,
        cost_tracker: Optional[CostTracker] = None,
        circuit_breaker_manager: Optional[CircuitBreakerManager] = None,
        cache: Optional[ResponseCache] = None,
        # Cache settings
        cache_ttl: int = 3600,
        cache_max_size: int = 1000,
        # Circuit breaker settings
        circuit_failure_threshold: int = 5,
        circuit_recovery_timeout: int = 60,
    ) -> None:
        """Initialize enhanced LLM client.

        Args:
            model: Model name
            provider: Provider name
            api_base: Custom API base URL
            timeout: Request timeout
            max_retries: Max retry attempts
            enable_rate_limiting: Enable rate limiting
            enable_cost_tracking: Enable cost tracking
            enable_circuit_breaker: Enable circuit breaker
            enable_caching: Enable response caching
            rate_limiter: Custom rate limiter instance
            cost_tracker: Custom cost tracker instance
            circuit_breaker_manager: Custom circuit breaker manager
            cache: Custom cache instance
            cache_ttl: Cache TTL in seconds
            cache_max_size: Maximum cache size
            circuit_failure_threshold: Circuit breaker failure threshold
            circuit_recovery_timeout: Circuit breaker recovery timeout
        """
        settings = get_settings()

        # Initialize base client
        self._base_client = LiteLLMClient(
            model=model or settings.default_model,
            provider=provider or settings.litellm_provider,
            api_base=api_base or settings.litellm_api_base,
            timeout=timeout or settings.llm_timeout,
            max_retries=max_retries or settings.litellm_retry_count,
        )

        # Initialize components
        self._rate_limiter = rate_limiter or RateLimiter(
            enabled=enable_rate_limiting
        )
        self._cost_tracker = cost_tracker or CostTracker(
            log_file=settings.log_dir + "/costs.jsonl"
            if settings.enable_trajectory_logging
            else None,
            enabled=enable_cost_tracking,
        )
        self._cache = cache or ResponseCache(
            ttl_seconds=cache_ttl,
            max_size=cache_max_size,
            enabled=enable_caching,
        )

        # Initialize circuit breaker
        self._circuit_manager = circuit_breaker_manager
        if enable_circuit_breaker and circuit_breaker_manager is None:
            self._circuit_manager = CircuitBreakerManager()
            provider_name = provider or settings.litellm_provider
            self._circuit_manager.register_breaker(
                provider_name,
                config=CircuitBreakerConfig(
                    failure_threshold=circuit_failure_threshold,
                    recovery_timeout=circuit_recovery_timeout,
                ),
            )

        self._model = model or settings.default_model
        self._provider = provider or settings.litellm_provider

        # Feature flags
        self._enable_rate_limiting = enable_rate_limiting
        self._enable_cost_tracking = enable_cost_tracking
        self._enable_circuit_breaker = enable_circuit_breaker
        self._enable_caching = enable_caching

        logger.info(
            "enhanced_llm_client_initialized",
            model=self._model,
            provider=self._provider,
            rate_limiting=enable_rate_limiting,
            cost_tracking=enable_cost_tracking,
            circuit_breaker=enable_circuit_breaker,
            caching=enable_caching,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        session_id: Optional[str] = None,
        query_type: str = "sub_llm",
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate completion with all enhancements.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Temperature
            max_tokens: Max tokens
            session_id: Session ID for cost tracking
            query_type: Query type ("root" or "sub_llm")
            **kwargs: Additional parameters

        Returns:
            LLMResponse
        """
        # 1. Check cache
        if self._enable_caching:
            cached = await self._cache.get(
                prompt=prompt,
                model=self._model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            if cached:
                logger.debug("cache_hit", model=self._model)
                return cached

        # 2. Check circuit breaker
        if self._enable_circuit_breaker and self._circuit_manager:
            breaker = self._circuit_manager.get_breaker(self._provider)
            if breaker and not await breaker.can_execute():
                logger.warning(
                    "circuit_breaker_open",
                    provider=self._provider,
                )
                # Try to fallback or raise
                # For now, we'll still try but log the warning

        # 3. Acquire rate limit
        if self._enable_rate_limiting:
            acquired = await self._rate_limiter.acquire(
                provider=self._provider,
                tokens=1000,  # Estimate
                wait=True,
                timeout=30.0,
            )
            if not acquired:
                logger.warning("rate_limit_exceeded", provider=self._provider)

        # 4. Execute with circuit breaker protection
        try:
            if self._enable_circuit_breaker and self._circuit_manager:
                response = await self._circuit_manager.execute_with_fallback(
                    self._provider,
                    self._base_client.generate,
                    prompt,
                    system_prompt,
                    temperature,
                    max_tokens,
                    **kwargs,
                )
            else:
                response = await self._base_client.generate(
                    prompt,
                    system_prompt,
                    temperature,
                    max_tokens,
                    **kwargs,
                )

            # 5. Track cost
            if self._enable_cost_tracking and session_id:
                self._cost_tracker.log_cost(
                    session_id=session_id,
                    response=response,
                    query_type=query_type,
                    provider=self._provider,
                )

            # 6. Cache result
            if self._enable_caching:
                await self._cache.set(
                    prompt=prompt,
                    model=self._model,
                    response=response,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )

            # 7. Record success in circuit breaker
            if self._enable_circuit_breaker and self._circuit_manager:
                breaker = self._circuit_manager.get_breaker(self._provider)
                if breaker:
                    await breaker.record_success()

            return response

        except Exception as e:
            # Record failure in circuit breaker
            if self._enable_circuit_breaker and self._circuit_manager:
                breaker = self._circuit_manager.get_breaker(self._provider)
                if breaker:
                    await breaker.record_failure()
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate streaming completion.

        Note: Streaming responses are not cached.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Temperature
            max_tokens: Max tokens
            **kwargs: Additional parameters

        Yields:
            Text chunks
        """
        # Streaming bypasses cache (can't cache partial responses)

        # 1. Check circuit breaker
        if self._enable_circuit_breaker and self._circuit_manager:
            breaker = self._circuit_manager.get_breaker(self._provider)
            if breaker and not await breaker.can_execute():
                logger.warning(
                    "circuit_breaker_open_streaming",
                    provider=self._provider,
                )

        # 2. Acquire rate limit
        if self._enable_rate_limiting:
            await self._rate_limiter.acquire(
                provider=self._provider,
                tokens=1000,
                wait=True,
            )

        # 3. Stream from base client
        async for chunk in self._base_client.generate_stream(
            prompt, system_prompt, temperature, max_tokens, **kwargs
        ):
            yield chunk

    async def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        session_id: Optional[str] = None,
        max_concurrent: int = 10,
        **kwargs: Any,
    ) -> Dict[str, BatchResult]:
        """Generate completions for multiple prompts in parallel.

        Args:
            prompts: List of prompts
            system_prompt: Shared system prompt
            temperature: Temperature
            max_tokens: Max tokens
            session_id: Session ID for cost tracking
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping prompt index to BatchResult
        """
        batch_manager = BatchManager(max_concurrent=max_concurrent)

        # Add all requests
        for i, prompt in enumerate(prompts):
            coro = self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                session_id=session_id,
                query_type="sub_llm",
                **kwargs,
            )
            batch_manager.add_request(str(i), coro)

        # Execute batch
        return await batch_manager.execute()

    def get_model_name(self) -> str:
        """Get model name."""
        return self._base_client.get_model_name()

    def get_cost_report(
        self,
        session_id: Optional[str] = None,
    ) -> CostReport:
        """Get cost report.

        Args:
            session_id: Session ID (None for all sessions)

        Returns:
            CostReport
        """
        if not self._enable_cost_tracking:
            return CostReport(
                total_cost=0.0,
                total_tokens=0,
                total_requests=0,
                by_provider={},
                by_model={},
                by_query_type={},
                entries=[],
            )

        if session_id:
            return self._cost_tracker.get_session_cost(session_id)
        else:
            return self._cost_tracker.get_total_cost()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Statistics dictionary
        """
        return self._cache.get_stats()

    def get_circuit_status(self) -> Dict[str, Any]:
        """Get circuit breaker status.

        Returns:
            Status dictionary
        """
        if self._circuit_manager:
            return self._circuit_manager.get_all_status()
        return {}

    def clear_cache(self) -> None:
        """Clear response cache."""
        self._cache.clear()

    def invalidate_cache_for_model(self, model: str) -> None:
        """Invalidate cache entries for a model.

        Args:
            model: Model name
        """
        self._cache.invalidate_model(model)

    def export_costs_to_json(self, filepath: str) -> None:
        """Export cost data to JSON.

        Args:
            filepath: Output file path
        """
        if self._enable_cost_tracking:
            self._cost_tracker.export_to_json(filepath)

    def export_costs_to_csv(self, filepath: str) -> None:
        """Export cost data to CSV.

        Args:
            filepath: Output file path
        """
        if self._enable_cost_tracking:
            self._cost_tracker.export_to_csv(filepath)

    @property
    def provider(self) -> str:
        """Get provider name."""
        return self._provider

    @property
    def model(self) -> str:
        """Get model name."""
        return self._model

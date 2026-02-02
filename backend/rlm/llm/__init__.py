"""LLM client module."""

from rlm.llm.batch_manager import BatchManager, BatchResult
from rlm.llm.cache import ResponseCache
from rlm.llm.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerManager,
    CircuitBreakerOpen,
)
from rlm.llm.client import LiteLLMClient, MockLLMClient
from rlm.llm.connection_pool import (
    AsyncConnectionPool,
    ConnectionPoolManager,
    get_pool_manager,
)
from rlm.llm.cost_tracker import CostEntry, CostReport, CostTracker
from rlm.llm.enhanced_client import EnhancedLLMClient
from rlm.llm.interface import LLMClientInterface
from rlm.llm.model_registry import ModelInfo, ModelRegistry, get_model_registry
from rlm.llm.partial_failure_handler import (
    BatchFailureSummary,
    CallResult,
    FailurePolicy,
    FailureStrategy,
    PartialFailureHandler,
    create_aggressive_retry_policy,
    create_default_policy,
    create_fallback_policy,
)
from rlm.llm.prompts import (
    get_error_recovery_prompt,
    get_root_system_prompt,
    get_sub_llm_system_prompt,
)
from rlm.llm.query_batcher import QueryBatcher
from rlm.llm.rate_limiter import RateLimiter, RateLimitConfig
from rlm.llm.session_cache import SessionCache, SemanticSessionCache
from rlm.llm.streaming_aggregator import StreamBuffer, StreamingAggregator
from rlm.llm.sub_llm_manager import (
    SubLLMCall,
    SubLLMManager,
    SubLLMResult,
    StreamingSubLLMResult,
)

__all__ = [
    # Base clients
    "LLMClientInterface",
    "LiteLLMClient",
    "MockLLMClient",
    # Enhanced client (Component 3)
    "EnhancedLLMClient",
    # Supporting components
    "RateLimiter",
    "RateLimitConfig",
    "CostTracker",
    "CostEntry",
    "CostReport",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerManager",
    "CircuitBreakerOpen",
    "ResponseCache",
    "BatchManager",
    "BatchResult",
    "ModelRegistry",
    "ModelInfo",
    "get_model_registry",
    # Component 5: Async Sub-LLM Manager
    "SubLLMManager",
    "SubLLMCall",
    "SubLLMResult",
    "StreamingSubLLMResult",
    "AsyncConnectionPool",
    "ConnectionPoolManager",
    "get_pool_manager",
    "SessionCache",
    "SemanticSessionCache",
    "QueryBatcher",
    "StreamingAggregator",
    "StreamBuffer",
    "PartialFailureHandler",
    "FailurePolicy",
    "FailureStrategy",
    "CallResult",
    "BatchFailureSummary",
    "create_default_policy",
    "create_aggressive_retry_policy",
    "create_fallback_policy",
    # Prompts
    "get_root_system_prompt",
    "get_sub_llm_system_prompt",
    "get_error_recovery_prompt",
]

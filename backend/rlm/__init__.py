"""RLM Document Retrieval System."""

from rlm.core.orchestrator import RLMOrchestrator
from rlm.core.recursion import RecursionController
from rlm.llm.client import LiteLLMClient, MockLLMClient
from rlm.llm.sub_llm_manager import SubLLMCall, SubLLMManager, SubLLMResult
from rlm.routing.analyzer import QueryAnalyzer
from rlm.routing.cost_estimator import CostEstimator
from rlm.routing.models import (
    ExecutionStrategy,
    QueryComplexity,
    RoutingDecision,
    RoutingVisibility,
)
from rlm.routing.optimizer import QueryOptimizer
from rlm.routing.query_router import QueryRouter, RoutingService
from rlm.routing.rag_engine import RAGEngine, RAGSearcher
from rlm.sandbox.docker_repl import DockerREPLSandbox
from rlm.sandbox.factory import SandboxFactory, create_sandbox
from rlm.sandbox.local_repl import LocalREPLSandbox
from rlm.trajectory.logger import TrajectoryLogger
from rlm.types import ChunkedContext, RLMResult

__version__ = "0.1.0"

__all__ = [
    # Core components
    "RLMOrchestrator",
    "RecursionController",
    # LLM clients
    "LiteLLMClient",
    "MockLLMClient",
    # Sub-LLM Manager (Component 5)
    "SubLLMManager",
    "SubLLMCall",
    "SubLLMResult",
    # Query Router & Optimizer (Component 7)
    "QueryRouter",
    "RoutingService",
    "QueryAnalyzer",
    "QueryOptimizer",
    "CostEstimator",
    "RAGEngine",
    "RAGSearcher",
    "RoutingDecision",
    "RoutingVisibility",
    "ExecutionStrategy",
    "QueryComplexity",
    # Sandboxes
    "LocalREPLSandbox",
    "DockerREPLSandbox",
    "SandboxFactory",
    "create_sandbox",
    # Other
    "TrajectoryLogger",
    "ChunkedContext",
    "RLMResult",
]

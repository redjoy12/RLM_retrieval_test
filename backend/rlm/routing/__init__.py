"""Query Router and Optimizer - Component 7.

This package provides intelligent query routing, cost estimation,
and execution strategy selection for the RLM Document Retrieval System.
"""

from rlm.routing.analyzer import QueryAnalyzer
from rlm.routing.cost_estimator import CostEstimator
from rlm.routing.models import (
    CostEstimate,
    ExecutionStrategy,
    OptimizationResult,
    QueryAnalysis,
    QueryComplexity,
    QuerySuggestion,
    RAGChunk,
    RAGResult,
    RoutingDecision,
    RoutingError,
    RoutingVisibility,
    StrategyResult,
)
from rlm.routing.optimizer import QueryOptimizer, SimpleOptimizer
from rlm.routing.query_router import QueryRouter, RoutingService
from rlm.routing.rag_engine import RAGEngine, RAGSearcher
from rlm.routing.strategies import (
    DirectLLMStrategy,
    HybridStrategy,
    RAGStrategy,
    RLMStrategy,
    StrategyFactory,
)

__all__ = [
    # Main router
    "QueryRouter",
    "RoutingService",
    # Analysis
    "QueryAnalyzer",
    "QueryAnalysis",
    "QueryComplexity",
    # Cost estimation
    "CostEstimator",
    "CostEstimate",
    # Optimization
    "QueryOptimizer",
    "SimpleOptimizer",
    "OptimizationResult",
    "QuerySuggestion",
    # RAG
    "RAGEngine",
    "RAGSearcher",
    "RAGResult",
    "RAGChunk",
    # Strategies
    "ExecutionStrategy",
    "DirectLLMStrategy",
    "RAGStrategy",
    "RLMStrategy",
    "HybridStrategy",
    "StrategyFactory",
    "StrategyResult",
    # Models
    "RoutingDecision",
    "RoutingVisibility",
    "RoutingError",
]

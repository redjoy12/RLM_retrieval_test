"""Query Router and Optimizer - Component 7.

Main entry point for intelligent query routing that analyzes complexity,
estimates costs, and selects optimal execution strategies.
"""

from typing import Any, Dict, List, Optional

import structlog

from rlm.config import get_settings
from rlm.documents.storage import DocumentStorage
from rlm.routing.analyzer import QueryAnalyzer
from rlm.routing.cost_estimator import CostEstimator
from rlm.routing.models import (
    ExecutionStrategy,
    OptimizationResult,
    QueryAnalysis,
    RoutingDecision,
    RoutingError,
    RoutingVisibility,
    StrategyResult,
)
from rlm.routing.optimizer import QueryOptimizer
from rlm.routing.strategies import StrategyFactory

logger = structlog.get_logger()


class QueryRouter:
    """Main query router for intelligent execution strategy selection.
    
    Analyzes queries, estimates costs, and routes to appropriate execution
    strategies (Direct LLM, RAG, RLM, or Hybrid).
    
    Example:
        ```python
        router = QueryRouter()
        
        # Analyze query
        decision = await router.analyze_query(
            "Compare the revenue trends",
            document_ids=["doc-123"]
        )
        
        # Execute with selected strategy
        result = await router.execute(decision)
        ```
    """
    
    def __init__(
        self,
        analyzer: Optional[QueryAnalyzer] = None,
        cost_estimator: Optional[CostEstimator] = None,
        optimizer: Optional[QueryOptimizer] = None,
    ) -> None:
        """Initialize query router.
        
        Args:
            analyzer: Query complexity analyzer
            cost_estimator: Cost estimation component
            optimizer: Query optimizer for suggestions
        """
        self.settings = get_settings()
        self.analyzer = analyzer or QueryAnalyzer()
        self.cost_estimator = cost_estimator or CostEstimator()
        self.optimizer = optimizer or QueryOptimizer()
        
        # Thresholds from settings
        self.direct_llm_max = getattr(self.settings, 'direct_llm_max_context', 10000)
        self.rag_max = getattr(self.settings, 'rag_max_context', 500000)
        
        logger.info("query_router_initialized")
    
    async def analyze_query(
        self,
        query: str,
        document_ids: List[str],
    ) -> RoutingDecision:
        """Analyze a query and determine routing strategy.
        
        Args:
            query: User's query string
            document_ids: List of document IDs to query
            
        Returns:
            RoutingDecision with strategy and metadata
        """
        logger.info("analyzing_query", query=query[:50], doc_count=len(document_ids))
        
        # Step 1: Analyze query complexity
        analysis = self.analyzer.analyze(query)
        
        # Step 2: Get context size
        context_size = await self._get_context_size(document_ids)
        
        # Step 3: Determine strategy
        strategy, reasoning, confidence = self._select_strategy(
            analysis, context_size
        )
        
        # Step 4: Estimate costs
        cost_estimate = self._estimate_strategy_cost(
            query, context_size, strategy, analysis
        )
        
        # Step 5: Get optimization suggestions
        optimization = await self.optimizer.optimize(
            query, analysis.complexity, context_size
        )
        
        # Step 6: Calculate estimated chunks
        estimated_chunks = self._estimate_chunks(strategy, context_size, analysis)
        
        decision = RoutingDecision(
            query=query,
            strategy=strategy,
            confidence=confidence,
            reasoning=reasoning,
            query_analysis=analysis,
            cost_estimate=cost_estimate,
            document_ids=document_ids,
            context_size=context_size,
            estimated_chunks=estimated_chunks,
            rag_top_k=5 if strategy in [ExecutionStrategy.RAG, ExecutionStrategy.HYBRID] else None,
            hybrid_rag_filter_ratio=0.5 if strategy == ExecutionStrategy.HYBRID else None,
        )
        
        logger.info(
            "routing_decision_made",
            query=query[:50],
            strategy=strategy.value,
            confidence=confidence,
            estimated_cost=cost_estimate.estimated_cost_usd,
        )
        
        return decision
    
    async def execute(
        self,
        decision: RoutingDecision,
        use_optimized_query: bool = False,
    ) -> StrategyResult:
        """Execute query using selected strategy.
        
        Args:
            decision: RoutingDecision from analyze_query()
            use_optimized_query: Whether to use optimized query if available
            
        Returns:
            StrategyResult with answer and execution metadata
        """
        query = decision.query
        
        # Use optimized query if requested
        if use_optimized_query:
            optimization = await self.optimizer.optimize(
                decision.query,
                decision.query_analysis.complexity,
                decision.context_size,
            )
            if optimization.use_optimized and optimization.optimized_query:
                query = optimization.optimized_query
                logger.info("using_optimized_query", original=decision.query[:50])
        
        # Create strategy instance
        strategy_kwargs = {}
        
        if decision.strategy in [ExecutionStrategy.RAG, ExecutionStrategy.HYBRID]:
            strategy_kwargs["top_k"] = decision.rag_top_k
        
        if decision.strategy == ExecutionStrategy.HYBRID:
            strategy_kwargs["filter_ratio"] = decision.hybrid_rag_filter_ratio
        
        strategy = StrategyFactory.create_strategy(
            decision.strategy,
            **strategy_kwargs
        )
        
        # Execute
        logger.info(
            "executing_strategy",
            strategy=decision.strategy.value,
            query=query[:50],
        )
        
        result = await strategy.execute(
            query=query,
            document_ids=decision.document_ids,
            cost_estimate=decision.cost_estimate,
            **strategy_kwargs
        )
        
        logger.info(
            "strategy_execution_complete",
            strategy=decision.strategy.value,
            execution_time_ms=result.execution_time_ms,
        )
        
        return result
    
    async def route_and_execute(
        self,
        query: str,
        document_ids: List[str],
        return_routing_info: bool = True,
    ) -> Dict[str, Any]:
        """One-step route and execute with full visibility.
        
        Args:
            query: User's query
            document_ids: Documents to query
            return_routing_info: Include routing decision in response
            
        Returns:
            Dictionary with answer and routing metadata
        """
        # Analyze and route
        decision = await self.analyze_query(query, document_ids)
        
        # Execute
        result = await self.execute(decision)
        
        # Build response
        response = {
            "answer": result.answer,
            "execution_time_ms": result.execution_time_ms,
            "tokens_used": result.tokens_used,
            "cost_usd": result.cost_usd,
        }
        
        if return_routing_info:
            response["routing"] = self._create_visibility(decision).to_dict()
        
        return response
    
    def _select_strategy(
        self,
        analysis: QueryAnalysis,
        context_size: int,
    ) -> tuple:
        """Select execution strategy based on analysis.
        
        Returns:
            Tuple of (strategy, reasoning, confidence)
        """
        complexity = analysis.complexity
        
        # Decision logic from implementation plan
        if context_size < self.direct_llm_max and complexity.value == "simple":
            return (
                ExecutionStrategy.DIRECT_LLM,
                f"Small context ({context_size} chars) with simple query",
                0.90
            )
        
        elif complexity.value == "single_hop" and context_size < self.rag_max:
            return (
                ExecutionStrategy.RAG,
                f"Single-hop retrieval with manageable context ({context_size} chars)",
                0.85
            )
        
        elif complexity.value in ["multi_hop", "quadratic"]:
            # For very large contexts with complex queries, use hybrid
            if context_size > self.rag_max:
                return (
                    ExecutionStrategy.HYBRID,
                    f"Large context ({context_size} chars) with {complexity.value} analysis. "
                    "Using RAG pre-filter + RLM deep analysis",
                    0.88
                )
            else:
                return (
                    ExecutionStrategy.RLM,
                    f"{complexity.value} reasoning required for comprehensive analysis",
                    0.82
                )
        
        else:
            # Default to hybrid for edge cases
            return (
                ExecutionStrategy.HYBRID,
                f"Balanced approach for context size {context_size} with {complexity.value} complexity",
                0.75
            )
    
    async def _get_context_size(self, document_ids: List[str]) -> int:
        """Get total context size for documents."""
        storage = DocumentStorage()
        total_size = 0
        
        for doc_id in document_ids:
            try:
                doc = await storage.get_document(doc_id)
                if doc and doc.content:
                    content = doc.content.cleaned_text or doc.content.raw_text
                    total_size += len(content)
            except Exception:
                pass
        
        return total_size
    
    def _estimate_strategy_cost(
        self,
        query: str,
        context_size: int,
        strategy: ExecutionStrategy,
        analysis: QueryAnalysis,
    ) -> Any:
        """Estimate cost for selected strategy."""
        if strategy == ExecutionStrategy.DIRECT_LLM:
            dummy_context = "x" * min(context_size, self.direct_llm_max)
            return self.cost_estimator.estimate_cost(query, dummy_context)
        
        elif strategy == ExecutionStrategy.RAG:
            return self.cost_estimator.estimate_rag_cost(query, num_chunks=5)
        
        elif strategy == ExecutionStrategy.RLM:
            estimated_calls = analysis.estimated_chunk_access
            return self.cost_estimator.estimate_rlm_cost(
                query, context_size, estimated_calls
            )
        
        else:  # HYBRID
            # Estimate as RLM on filtered context
            filtered_size = context_size // 5
            estimated_calls = max(1, analysis.estimated_chunk_access // 2)
            return self.cost_estimator.estimate_rlm_cost(
                query, filtered_size, estimated_calls
            )
    
    def _estimate_chunks(
        self,
        strategy: ExecutionStrategy,
        context_size: int,
        analysis: QueryAnalysis,
    ) -> int:
        """Estimate number of chunks to process."""
        if strategy == ExecutionStrategy.DIRECT_LLM:
            return 1
        elif strategy == ExecutionStrategy.RAG:
            return 5  # Default top_k
        elif strategy == ExecutionStrategy.HYBRID:
            return max(1, int(10 * 0.5))  # top_k * filter_ratio
        else:  # RLM
            return analysis.estimated_chunk_access
    
    def _create_visibility(self, decision: RoutingDecision) -> RoutingVisibility:
        """Create user-facing routing visibility."""
        strategy_descriptions = {
            ExecutionStrategy.DIRECT_LLM: "Direct LLM - Simple Q&A with full context",
            ExecutionStrategy.RAG: "RAG - Vector retrieval with semantic search",
            ExecutionStrategy.RLM: "RLM - Recursive analysis with sub-LLM calls",
            ExecutionStrategy.HYBRID: "Hybrid - RAG pre-filter + RLM deep analysis",
        }
        
        return RoutingVisibility(
            strategy_name=decision.strategy.value.upper(),
            strategy_description=strategy_descriptions.get(
                decision.strategy, "Unknown strategy"
            ),
            reasoning=decision.reasoning,
            estimated_cost_usd=f"${decision.cost_estimate.estimated_cost_usd:.4f}",
            estimated_tokens=decision.cost_estimate.estimated_total_tokens,
            confidence_percent=int(decision.confidence * 100),
        )


class RoutingService:
    """High-level service interface for routing."""
    
    def __init__(self) -> None:
        """Initialize routing service."""
        self.router = QueryRouter()
    
    async def analyze(
        self,
        query: str,
        document_ids: List[str],
    ) -> Dict[str, Any]:
        """Analyze query and return routing decision (no execution)."""
        decision = await self.router.analyze_query(query, document_ids)
        
        return {
            "query": query,
            "routing_decision": decision.to_dict(),
            "visibility": self.router._create_visibility(decision).to_dict(),
        }
    
    async def execute(
        self,
        query: str,
        document_ids: List[str],
    ) -> Dict[str, Any]:
        """Execute query with routing."""
        return await self.router.route_and_execute(query, document_ids)

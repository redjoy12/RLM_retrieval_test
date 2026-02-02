"""Execution strategies for query routing."""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import structlog

from rlm.config import get_settings
from rlm.core.orchestrator import RLMOrchestrator
from rlm.documents.storage import DocumentStorage
from rlm.routing.models import (
    CostEstimate,
    ExecutionStrategy,
    RAGResult,
    StrategyResult,
)
from rlm.routing.rag_engine import RAGEngine

logger = structlog.get_logger()


class ExecutionStrategy(ABC):
    """Abstract base class for execution strategies."""
    
    def __init__(self) -> None:
        """Initialize the strategy."""
        self.settings = get_settings()
    
    @abstractmethod
    async def execute(
        self,
        query: str,
        document_ids: List[str],
        cost_estimate: CostEstimate,
        **kwargs: Any,
    ) -> StrategyResult:
        """Execute the strategy.
        
        Args:
            query: The user's query
            document_ids: List of document IDs to query
            cost_estimate: Pre-calculated cost estimate
            **kwargs: Strategy-specific parameters
            
        Returns:
            StrategyResult with answer and metadata
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        pass


class DirectLLMStrategy(ExecutionStrategy):
    """Direct LLM strategy for simple queries with small context."""
    
    def __init__(self, llm_client: Optional[Any] = None) -> None:
        """Initialize direct LLM strategy."""
        super().__init__()
        self.llm_client = llm_client
    
    @property
    def name(self) -> str:
        return "Direct LLM"
    
    @property
    def description(self) -> str:
        return "Simple question answering with direct LLM call"
    
    async def execute(
        self,
        query: str,
        document_ids: List[str],
        cost_estimate: CostEstimate,
        **kwargs: Any,
    ) -> StrategyResult:
        """Execute direct LLM query."""
        start_time = time.time()
        
        # Get document content (small context)
        storage = DocumentStorage()
        context_parts = []
        
        for doc_id in document_ids:
            doc = await storage.get_document(doc_id)
            if doc and doc.content:
                content = doc.content.cleaned_text or doc.content.raw_text
                # Limit context size
                context_parts.append(content[:10000])  # Max 10K chars
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build prompt
        prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {query}

Answer:"""
        
        # Call LLM
        from rlm.llm.client import LiteLLMClient
        client = self.llm_client or LiteLLMClient()
        
        response = await client.generate(
            prompt=prompt,
            system_prompt="You are a helpful assistant. Answer based only on the provided context.",
            max_tokens=1000,
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        # Calculate actual cost
        usage = response.usage or {}
        actual_cost = self._calculate_actual_cost(usage, cost_estimate.model_used)
        
        logger.info(
            "direct_llm_executed",
            query=query[:50],
            execution_time_ms=execution_time,
            tokens_used=usage.get("total_tokens", 0),
        )
        
        return StrategyResult(
            strategy=ExecutionStrategy,
            answer=response.content,
            execution_time_ms=execution_time,
            tokens_used=usage.get("total_tokens", 0),
            cost_usd=actual_cost,
        )
    
    def _calculate_actual_cost(self, usage: Dict[str, int], model: str) -> float:
        """Calculate actual cost from usage."""
        from rlm.routing.cost_estimator import CostEstimator
        estimator = CostEstimator()
        
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        return estimator._calculate_cost(input_tokens, output_tokens, model)


class RAGStrategy(ExecutionStrategy):
    """RAG strategy using vector retrieval."""
    
    def __init__(
        self,
        rag_engine: Optional[RAGEngine] = None,
        llm_client: Optional[Any] = None,
    ) -> None:
        """Initialize RAG strategy."""
        super().__init__()
        self.rag_engine = rag_engine
        self.llm_client = llm_client
    
    @property
    def name(self) -> str:
        return "RAG"
    
    @property
    def description(self) -> str:
        return "Retrieval-Augmented Generation with vector search"
    
    async def execute(
        self,
        query: str,
        document_ids: List[str],
        cost_estimate: CostEstimate,
        **kwargs: Any,
    ) -> StrategyResult:
        """Execute RAG query."""
        start_time = time.time()
        
        # Get top_k from kwargs
        top_k = kwargs.get("top_k", 5)
        
        # Initialize RAG engine if not provided
        rag = self.rag_engine or RAGEngine()
        
        # Retrieve relevant chunks
        rag_result = await rag.retrieve(
            query=query,
            document_ids=document_ids,
            top_k=top_k,
        )
        
        # Build context from retrieved chunks
        context = rag_result.get_context_string()
        
        # Call LLM with retrieved context
        from rlm.llm.client import LiteLLMClient
        client = self.llm_client or LiteLLMClient()
        
        prompt = f"""Based on the following retrieved information, answer the question:

Retrieved Information:
{context}

Question: {query}

Provide a comprehensive answer based on the retrieved information."""
        
        response = await client.generate(
            prompt=prompt,
            system_prompt="You are a helpful assistant. Answer based on the retrieved information.",
            max_tokens=1500,
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        # Calculate actual cost
        usage = response.usage or {}
        actual_cost = self._calculate_actual_cost(usage, cost_estimate.model_used)
        
        logger.info(
            "rag_executed",
            query=query[:50],
            chunks_retrieved=len(rag_result.chunks),
            execution_time_ms=execution_time,
        )
        
        return StrategyResult(
            strategy=ExecutionStrategy,
            answer=response.content,
            execution_time_ms=execution_time,
            tokens_used=usage.get("total_tokens", 0),
            cost_usd=actual_cost,
            rag_result=rag_result,
        )
    
    def _calculate_actual_cost(self, usage: Dict[str, int], model: str) -> float:
        """Calculate actual cost from usage."""
        from rlm.routing.cost_estimator import CostEstimator
        estimator = CostEstimator()
        
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        return estimator._calculate_cost(input_tokens, output_tokens, model)


class RLMStrategy(ExecutionStrategy):
    """Full RLM recursive strategy."""
    
    def __init__(
        self,
        orchestrator: Optional[RLMOrchestrator] = None,
    ) -> None:
        """Initialize RLM strategy."""
        super().__init__()
        self.orchestrator = orchestrator
    
    @property
    def name(self) -> str:
        return "RLM"
    
    @property
    def description(self) -> str:
        return "Recursive Language Model with deep analysis"
    
    async def execute(
        self,
        query: str,
        document_ids: List[str],
        cost_estimate: CostEstimate,
        **kwargs: Any,
    ) -> StrategyResult:
        """Execute RLM query."""
        start_time = time.time()
        
        # Get document content
        storage = DocumentStorage()
        context_parts = []
        
        for doc_id in document_ids:
            doc = await storage.get_document(doc_id)
            if doc and doc.content:
                content = doc.content.cleaned_text or doc.content.raw_text
                context_parts.append(content)
        
        full_context = "\n\n---\n\n".join(context_parts)
        
        # Initialize orchestrator if not provided
        orchestrator = self.orchestrator or RLMOrchestrator()
        
        # Execute RLM
        result = await orchestrator.execute(
            query=query,
            context=full_context,
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        logger.info(
            "rlm_executed",
            query=query[:50],
            sub_llm_calls=result.total_sub_llm_calls,
            execution_time_ms=execution_time,
        )
        
        return StrategyResult(
            strategy=ExecutionStrategy,
            answer=result.answer,
            execution_time_ms=execution_time,
            tokens_used=0,  # Would need to aggregate from trajectory
            cost_usd=cost_estimate.estimated_cost_usd,  # Use estimate
            sub_llm_calls=result.total_sub_llm_calls,
            trajectory=result.trajectory,
        )


class HybridStrategy(ExecutionStrategy):
    """Hybrid strategy: RAG pre-filter + RLM deep analysis."""
    
    def __init__(
        self,
        rag_engine: Optional[RAGEngine] = None,
        orchestrator: Optional[RLMOrchestrator] = None,
    ) -> None:
        """Initialize hybrid strategy."""
        super().__init__()
        self.rag_engine = rag_engine
        self.orchestrator = orchestrator
    
    @property
    def name(self) -> str:
        return "Hybrid RAG+RLM"
    
    @property
    def description(self) -> str:
        return "RAG pre-filtering followed by RLM deep analysis"
    
    async def execute(
        self,
        query: str,
        document_ids: List[str],
        cost_estimate: CostEstimate,
        **kwargs: Any,
    ) -> StrategyResult:
        """Execute hybrid query."""
        start_time = time.time()
        
        # Parameters
        rag_top_k = kwargs.get("rag_top_k", 10)
        filter_ratio = kwargs.get("filter_ratio", 0.5)
        
        # Step 1: RAG retrieval
        logger.info("hybrid_step_1_rag", query=query[:50])
        
        rag = self.rag_engine or RAGEngine()
        rag_result = await rag.retrieve(
            query=query,
            document_ids=document_ids,
            top_k=rag_top_k,
        )
        
        # Step 2: Filter to top chunks based on ratio
        num_filtered = max(1, int(len(rag_result.chunks) * filter_ratio))
        filtered_chunks = rag_result.chunks[:num_filtered]
        
        logger.info(
            "hybrid_step_2_filter",
            chunks_retrieved=len(rag_result.chunks),
            chunks_filtered=num_filtered,
        )
        
        # Step 3: RLM analysis on filtered chunks
        logger.info("hybrid_step_3_rlm", query=query[:50])
        
        filtered_context = "\n\n---\n\n".join(
            f"[Document: {c.document_id}, Score: {c.score:.3f}]\n{c.content}"
            for c in filtered_chunks
        )
        
        orchestrator = self.orchestrator or RLMOrchestrator()
        
        result = await orchestrator.execute(
            query=query,
            context=filtered_context,
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        logger.info(
            "hybrid_executed",
            query=query[:50],
            chunks_processed=num_filtered,
            sub_llm_calls=result.total_sub_llm_calls,
            execution_time_ms=execution_time,
        )
        
        return StrategyResult(
            strategy=ExecutionStrategy,
            answer=result.answer,
            execution_time_ms=execution_time,
            tokens_used=0,  # Would need aggregation
            cost_usd=cost_estimate.estimated_cost_usd,
            rag_result=RAGResult(
                query=query,
                chunks=filtered_chunks,
                total_chunks_searched=rag_result.total_chunks_searched,
                retrieval_time_ms=rag_result.retrieval_time_ms,
            ),
            sub_llm_calls=result.total_sub_llm_calls,
            trajectory=result.trajectory,
        )


class StrategyFactory:
    """Factory for creating strategy instances."""
    
    @staticmethod
    def create_strategy(
        strategy_type: ExecutionStrategy,
        **kwargs: Any,
    ) -> ExecutionStrategy:
        """Create a strategy instance.
        
        Args:
            strategy_type: Type of strategy to create
            **kwargs: Dependencies to inject
            
        Returns:
            Strategy instance
        """
        strategies = {
            ExecutionStrategy.DIRECT_LLM: DirectLLMStrategy,
            ExecutionStrategy.RAG: RAGStrategy,
            ExecutionStrategy.RLM: RLMStrategy,
            ExecutionStrategy.HYBRID: HybridStrategy,
        }
        
        strategy_class = strategies.get(strategy_type)
        if not strategy_class:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        return strategy_class(**kwargs)

"""Advanced Hybrid RAG Strategy with full pipeline.

This module implements the advanced hybrid strategy that combines:
1. Hybrid retrieval (semantic + keyword)
2. Cross-encoder reranking
3. Adaptive chunk selection
4. RLM deep analysis
5. Citation tracking and verification
"""

import time
from typing import Any, Dict, List, Optional

import structlog

from rlm.config import get_settings
from rlm.core.orchestrator import RLMOrchestrator
from rlm.routing.models import (
    CostEstimate,
    ExecutionStrategy,
    RAGChunk,
    RAGResult,
    StrategyResult,
)
from rlm.routing.strategies import ExecutionStrategy as BaseExecutionStrategy

# Import hybrid components
try:
    from rlm.hybrid.search_engines import HybridSearcher
    from rlm.hybrid.reranker import (
        CrossEncoderReranker,
        LLMReranker,
        MultiStageReranker,
        RerankerPipeline,
    )
    from rlm.hybrid.chunk_selector import (
        AdaptiveChunkSelector,
        SelectionStrategyFactory,
    )
    from rlm.hybrid.citation_manager import CitationManager
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

logger = structlog.get_logger()


class AdvancedHybridStrategy(BaseExecutionStrategy):
    """Advanced hybrid strategy with full retrieval pipeline.
    
    Implements a complete hybrid RAG-RLM pipeline:
    1. Hybrid retrieval (vector + BM25 with RRF fusion)
    2. Multi-stage reranking (cross-encoder + optional LLM)
    3. Adaptive chunk selection based on query complexity
    4. RLM deep analysis on selected chunks
    5. Citation-enhanced answer generation
    
    Example:
        ```python
        strategy = AdvancedHybridStrategy()
        result = await strategy.execute(
            query="What are the revenue trends?",
            document_ids=["doc-123"],
            enable_reranking=True,
            enable_citations=True,
        )
        ```
    """
    
    def __init__(
        self,
        rag_engine: Optional[Any] = None,
        orchestrator: Optional[RLMOrchestrator] = None,
        reranker: Optional[Any] = None,
        chunk_selector: Optional[Any] = None,
        citation_manager: Optional[Any] = None,
    ) -> None:
        """Initialize advanced hybrid strategy.
        
        Args:
            rag_engine: RAG engine for retrieval
            orchestrator: RLM orchestrator for deep analysis
            reranker: Reranker instance (creates default if None)
            chunk_selector: Chunk selector (creates default if None)
            citation_manager: Citation manager (creates default if None)
        """
        super().__init__()
        
        # Initialize or store components
        self.rag_engine = rag_engine
        self.orchestrator = orchestrator
        
        # Initialize hybrid components if available
        if HYBRID_AVAILABLE:
            self.reranker = reranker or CrossEncoderReranker()
            self.chunk_selector = chunk_selector or AdaptiveChunkSelector()
            self.citation_manager = citation_manager or CitationManager()
            
            # Initialize reranker pipeline
            self.reranker_pipeline = RerankerPipeline(
                reranker=self.reranker,
                rerank_top_k=20,
                final_top_k=10,
            )
        else:
            self.reranker = None
            self.chunk_selector = None
            self.citation_manager = None
            self.reranker_pipeline = None
        
        logger.info(
            "advanced_hybrid_strategy_initialized",
            hybrid_available=HYBRID_AVAILABLE,
        )
    
    @property
    def name(self) -> str:
        """Strategy name."""
        return "Advanced Hybrid RAG+RLM"
    
    @property
    def description(self) -> str:
        """Human-readable description."""
        return (
            "Hybrid retrieval (semantic + keyword) with cross-encoder reranking, "
            "adaptive chunk selection, and RLM deep analysis with citations"
        )
    
    async def execute(
        self,
        query: str,
        document_ids: List[str],
        cost_estimate: CostEstimate,
        **kwargs: Any,
    ) -> StrategyResult:
        """Execute advanced hybrid strategy.
        
        Args:
            query: User's query
            document_ids: List of document IDs to query
            cost_estimate: Pre-calculated cost estimate
            **kwargs: Strategy-specific parameters including:
                - enable_reranking: Whether to apply reranking (default True)
                - enable_citations: Whether to track citations (default True)
                - enable_adaptive: Whether to use adaptive selection (default True)
                - semantic_weight: Weight for semantic search (default 0.7)
                - keyword_weight: Weight for keyword search (default 0.3)
                - max_chunks: Maximum chunks to analyze (default 10)
                - reranker_model: Model for reranking
                
        Returns:
            StrategyResult with answer and metadata
        """
        start_time = time.time()
        
        # Extract parameters
        enable_reranking = kwargs.get("enable_reranking", True)
        enable_citations = kwargs.get("enable_citations", True)
        enable_adaptive = kwargs.get("enable_adaptive", True)
        semantic_weight = kwargs.get("semantic_weight", 0.7)
        keyword_weight = kwargs.get("keyword_weight", 0.3)
        max_chunks = kwargs.get("max_chunks", 10)
        
        logger.info(
            "advanced_hybrid_execution_started",
            query=query[:50],
            document_count=len(document_ids),
            enable_reranking=enable_reranking,
            enable_citations=enable_citations,
        )
        
        try:
            # Step 1: Hybrid Retrieval
            step1_start = time.time()
            chunks = await self._hybrid_retrieval(
                query=query,
                document_ids=document_ids,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
                top_k=20,  # Get more for reranking
            )
            step1_time = (time.time() - step1_start) * 1000
            
            logger.info(
                "hybrid_retrieval_complete",
                chunks_retrieved=len(chunks),
                step_time_ms=step1_time,
            )
            
            if not chunks:
                return StrategyResult(
                    strategy=ExecutionStrategy.HYBRID,
                    answer="No relevant information found in the documents.",
                    execution_time_ms=(time.time() - start_time) * 1000,
                    tokens_used=0,
                    cost_usd=0.0,
                    metadata={
                        "error": "No chunks retrieved",
                        "step_times": {"retrieval": step1_time},
                    },
                )
            
            # Step 2: Reranking (if enabled and available)
            step2_start = time.time()
            if enable_reranking and HYBRID_AVAILABLE and self.reranker_pipeline:
                chunks = await self._rerank_chunks(query, chunks, top_k=20)
                step2_time = (time.time() - step2_start) * 1000
                logger.info(
                    "reranking_complete",
                    chunks_after_reranking=len(chunks),
                    step_time_ms=step2_time,
                )
            else:
                step2_time = 0
            
            # Step 3: Adaptive Chunk Selection
            step3_start = time.time()
            if enable_adaptive and HYBRID_AVAILABLE and self.chunk_selector:
                chunks = self._select_chunks_adaptive(query, chunks, max_chunks)
                step3_time = (time.time() - step3_start) * 1000
                logger.info(
                    "adaptive_selection_complete",
                    chunks_selected=len(chunks),
                    step_time_ms=step3_time,
                )
            else:
                # Simple top-k selection
                chunks = chunks[:max_chunks]
                step3_time = 0
            
            # Step 4: Citation Tracking Setup
            if enable_citations and HYBRID_AVAILABLE and self.citation_manager:
                self.citation_manager.clear_citations()
                for chunk in chunks:
                    self.citation_manager.add_chunk_citation(
                        chunk_id=chunk.get("chunk_id", "unknown"),
                        document_id=chunk.get("document_id", "unknown"),
                        content=chunk.get("content", ""),
                        score=chunk.get("score", 0),
                    )
            
            # Step 5: Build Context for RLM
            context = self._build_context(chunks, enable_citations)
            
            # Step 6: RLM Deep Analysis
            step4_start = time.time()
            orchestrator = self.orchestrator or RLMOrchestrator()
            
            rlm_result = await orchestrator.execute(
                query=query,
                context=context,
            )
            step4_time = (time.time() - step4_start) * 1000
            
            logger.info(
                "rlm_analysis_complete",
                sub_llm_calls=rlm_result.total_sub_llm_calls,
                step_time_ms=step4_time,
            )
            
            # Step 7: Post-process answer with citations
            final_answer = rlm_result.answer
            if enable_citations and HYBRID_AVAILABLE and self.citation_manager:
                final_answer = self._add_citations_to_answer(
                    final_answer, chunks
                )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Build metadata
            metadata = {
                "step_times": {
                    "retrieval_ms": step1_time,
                    "reranking_ms": step2_time,
                    "selection_ms": step3_time,
                    "rlm_analysis_ms": step4_time,
                    "total_ms": execution_time,
                },
                "chunks": {
                    "retrieved": len(chunks) + (20 if enable_reranking else 0),
                    "after_reranking": len(chunks) if enable_reranking else None,
                    "final_selected": len(chunks),
                },
                "search_params": {
                    "semantic_weight": semantic_weight,
                    "keyword_weight": keyword_weight,
                    "enable_reranking": enable_reranking,
                    "enable_adaptive": enable_adaptive,
                },
            }
            
            logger.info(
                "advanced_hybrid_execution_complete",
                query=query[:50],
                execution_time_ms=execution_time,
                final_chunks=len(chunks),
            )
            
            return StrategyResult(
                strategy=ExecutionStrategy.HYBRID,
                answer=final_answer,
                execution_time_ms=execution_time,
                tokens_used=0,  # Would need aggregation from trajectory
                cost_usd=cost_estimate.estimated_cost_usd,
                sub_llm_calls=rlm_result.total_sub_llm_calls,
                trajectory=rlm_result.trajectory,
                metadata=metadata,
            )
            
        except Exception as e:
            logger.error(
                "advanced_hybrid_execution_failed",
                query=query[:50],
                error=str(e),
            )
            
            # Return error result
            return StrategyResult(
                strategy=ExecutionStrategy.HYBRID,
                answer=f"Error during hybrid analysis: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
                tokens_used=0,
                cost_usd=0.0,
                metadata={"error": str(e), "error_type": type(e).__name__},
            )
    
    async def _hybrid_retrieval(
        self,
        query: str,
        document_ids: List[str],
        semantic_weight: float,
        keyword_weight: float,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid retrieval.
        
        Args:
            query: Search query
            document_ids: Document IDs to search
            semantic_weight: Weight for semantic search
            keyword_weight: Weight for keyword search
            top_k: Number of results
            
        Returns:
            Retrieved chunks
        """
        if not HYBRID_AVAILABLE:
            # Fall back to standard RAG engine retrieval
            from rlm.routing.rag_engine import RAGEngine
            
            engine = self.rag_engine or RAGEngine()
            result = await engine.retrieve(
                query=query,
                document_ids=document_ids,
                top_k=top_k,
                use_hybrid=False,
            )
            
            return [
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "score": chunk.score,
                    "metadata": chunk.metadata,
                }
                for chunk in result.chunks
            ]
        
        # Use hybrid searcher
        from rlm.config import get_settings
        
        settings = get_settings()
        host = getattr(settings, 'qdrant_host', 'localhost')
        port = getattr(settings, 'qdrant_port', 6333)
        collection = getattr(settings, 'qdrant_collection', 'rlm_chunks')
        
        from qdrant_client import QdrantClient
        
        client = QdrantClient(host=host, port=port)
        
        searcher = HybridSearcher(
            qdrant_client=client,
            collection_name=collection,
        )
        
        results = await searcher.search(
            query=query,
            document_ids=document_ids,
            top_k=top_k,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            vector_top_k=top_k * 2,
            keyword_top_k=top_k * 2,
        )
        
        return results
    
    async def _rerank_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Rerank chunks using cross-encoder.
        
        Args:
            query: Search query
            chunks: Chunks to rerank
            top_k: Number of top results to keep
            
        Returns:
            Reranked chunks
        """
        if not self.reranker_pipeline:
            return chunks[:top_k]
        
        try:
            reranked = await self.reranker_pipeline.rerank(query, chunks)
            return reranked[:top_k]
        except Exception as e:
            logger.error("reranking_failed", error=str(e))
            return chunks[:top_k]
    
    def _select_chunks_adaptive(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        max_chunks: int,
    ) -> List[Dict[str, Any]]:
        """Adaptively select chunks.
        
        Args:
            query: Search query
            chunks: Available chunks
            max_chunks: Maximum to select
            
        Returns:
            Selected chunks
        """
        if not self.chunk_selector:
            return chunks[:max_chunks]
        
        try:
            selected = self.chunk_selector.select(
                chunks=chunks,
                query=query,
                max_chunks=max_chunks,
            )
            return selected
        except Exception as e:
            logger.error("adaptive_selection_failed", error=str(e))
            return chunks[:max_chunks]
    
    def _build_context(
        self,
        chunks: List[Dict[str, Any]],
        enable_citations: bool,
    ) -> str:
        """Build context string from chunks.
        
        Args:
            chunks: Selected chunks
            enable_citations: Whether to include citation markers
            
        Returns:
            Context string for RLM
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            doc_id = chunk.get("document_id", "unknown")
            content = chunk.get("content", "")
            score = chunk.get("score", 0)
            
            if enable_citations:
                # Add citation marker
                citation_ref = f"[C{i+1}]"
                part = f"{citation_ref} [Document: {doc_id}, Score: {score:.3f}]\n{content}"
            else:
                part = f"[Document: {doc_id}, Score: {score:.3f}]\n{content}"
            
            context_parts.append(part)
        
        return "\n\n---\n\n".join(context_parts)
    
    def _add_citations_to_answer(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
    ) -> str:
        """Add citation references to answer.
        
        Args:
            answer: RLM-generated answer
            chunks: Source chunks with citation info
            
        Returns:
            Answer with citations
        """
        if not self.citation_manager:
            return answer
        
        try:
            # Generate citation summary
            citation_summary = self.citation_manager.format_citation_summary()
            
            # Append to answer
            return f"{answer}\n\n---\n\n{citation_summary}"
        except Exception as e:
            logger.error("citation_addition_failed", error=str(e))
            return answer


class AdvancedHybridStrategyFactory:
    """Factory for creating advanced hybrid strategies.
    
    Example:
        ```python
        # Standard configuration
        strategy = AdvancedHybridStrategyFactory.create()
        
        # With LLM reranker for higher accuracy
        strategy = AdvancedHybridStrategyFactory.create(
            reranker_type="llm",
            llm_model="gpt-5-mini",
        )
        
        # Multi-stage reranking
        strategy = AdvancedHybridStrategyFactory.create(
            reranker_type="multi_stage",
        )
        ```
    """
    
    @staticmethod
    def create(
        reranker_type: str = "cross_encoder",
        llm_model: Optional[str] = None,
        enable_citations: bool = True,
        **kwargs: Any,
    ) -> AdvancedHybridStrategy:
        """Create an advanced hybrid strategy.
        
        Args:
            reranker_type: Type of reranker ("cross_encoder", "llm", "multi_stage")
            llm_model: LLM model for LLM reranker
            enable_citations: Whether to enable citations
            **kwargs: Additional parameters
            
        Returns:
            Configured AdvancedHybridStrategy
        """
        if not HYBRID_AVAILABLE:
            logger.warning("hybrid_components_not_available, using_fallback")
            return AdvancedHybridStrategy()
        
        # Create reranker based on type
        if reranker_type == "cross_encoder":
            reranker = CrossEncoderReranker()
        elif reranker_type == "llm":
            reranker = LLMReranker(model=llm_model or "gpt-5-mini")
        elif reranker_type == "multi_stage":
            # Multi-stage: cross-encoder then LLM
            cross_encoder = CrossEncoderReranker()
            llm_reranker = LLMReranker(
                model=llm_model or "gpt-5-mini",
                max_chunks=10,
            )
            reranker = MultiStageReranker([
                (cross_encoder, 20),
                (llm_reranker, 10),
            ])
        else:
            reranker = CrossEncoderReranker()
        
        # Create citation manager if enabled
        citation_manager = CitationManager() if enable_citations else None
        
        # Create chunk selector
        chunk_selector = AdaptiveChunkSelector(
            min_chunks=kwargs.get("min_chunks", 3),
            max_chunks=kwargs.get("max_chunks", 10),
            diversity_threshold=kwargs.get("diversity_threshold", 0.8),
        )
        
        return AdvancedHybridStrategy(
            reranker=reranker,
            chunk_selector=chunk_selector,
            citation_manager=citation_manager,
        )

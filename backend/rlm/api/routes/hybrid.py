"""API routes for hybrid RAG functionality.

Provides REST endpoints for:
- Hybrid search
- Reranked retrieval
- Advanced hybrid execution
- Citation management
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

import structlog

from rlm.hybrid import (
    AdvancedHybridStrategy,
    HybridSearcher,
    RerankerPipeline,
)
from rlm.routing.rag_engine import RAGEngine

logger = structlog.get_logger()

router = APIRouter(prefix="/hybrid", tags=["hybrid"])


# Request/Response Models
class HybridSearchRequest(BaseModel):
    """Request for hybrid search."""
    query: str = Field(..., description="Search query")
    document_ids: List[str] = Field(..., description="Document IDs to search")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results")
    semantic_weight: float = Field(default=0.7, ge=0, le=1, description="Semantic search weight")
    keyword_weight: float = Field(default=0.3, ge=0, le=1, description="Keyword search weight")


class HybridSearchResponse(BaseModel):
    """Response from hybrid search."""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    search_time_ms: float
    semantic_weight: float
    keyword_weight: float


class RerankedSearchRequest(BaseModel):
    """Request for reranked search."""
    query: str = Field(..., description="Search query")
    document_ids: List[str] = Field(..., description="Document IDs to search")
    top_k: int = Field(default=5, ge=1, le=20, description="Final number of results")
    rerank_top_k: int = Field(default=20, ge=5, le=50, description="Number of chunks to rerank")
    use_hybrid: bool = Field(default=True, description="Use hybrid search")
    enable_reranking: bool = Field(default=True, description="Enable cross-encoder reranking")


class RerankedSearchResponse(BaseModel):
    """Response from reranked search."""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    search_time_ms: float
    reranked: bool


class AdvancedHybridRequest(BaseModel):
    """Request for advanced hybrid execution."""
    query: str = Field(..., description="Query to answer")
    document_ids: List[str] = Field(..., description="Document IDs to analyze")
    max_chunks: int = Field(default=10, ge=1, le=30, description="Maximum chunks to analyze")
    enable_reranking: bool = Field(default=True, description="Enable reranking")
    enable_citations: bool = Field(default=True, description="Enable citations")
    enable_adaptive: bool = Field(default=True, description="Enable adaptive selection")
    semantic_weight: float = Field(default=0.7, ge=0, le=1)
    keyword_weight: float = Field(default=0.3, ge=0, le=1)


class AdvancedHybridResponse(BaseModel):
    """Response from advanced hybrid execution."""
    query: str
    answer: str
    execution_time_ms: float
    chunks_analyzed: int
    sub_llm_calls: int
    citations: Optional[List[Dict[str, Any]]]
    step_times: Dict[str, float]
    search_params: Dict[str, Any]


# Endpoints
@router.post("/search", response_model=HybridSearchResponse)
async def hybrid_search(request: HybridSearchRequest) -> HybridSearchResponse:
    """Perform hybrid search (semantic + keyword).
    
    Combines vector-based semantic search with BM25 keyword search
    using Reciprocal Rank Fusion (RRF) for optimal results.
    """
    import time
    
    start_time = time.time()
    
    try:
        # Initialize RAG engine
        engine = RAGEngine()
        
        # Perform hybrid retrieval
        result = await engine.retrieve(
            query=request.query,
            document_ids=request.document_ids,
            top_k=request.top_k,
            use_hybrid=True,
            semantic_weight=request.semantic_weight,
            keyword_weight=request.keyword_weight,
        )
        
        # Convert to response format
        results = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "score": chunk.score,
                "metadata": chunk.metadata,
            }
            for chunk in result.chunks
        ]
        
        search_time = (time.time() - start_time) * 1000
        
        logger.info(
            "hybrid_search_api_complete",
            query=request.query[:50],
            results=len(results),
            time_ms=search_time,
        )
        
        return HybridSearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_time_ms=search_time,
            semantic_weight=request.semantic_weight,
            keyword_weight=request.keyword_weight,
        )
        
    except Exception as e:
        logger.error("hybrid_search_api_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/reranked-search", response_model=RerankedSearchResponse)
async def reranked_search(request: RerankedSearchRequest) -> RerankedSearchResponse:
    """Perform search with cross-encoder reranking.
    
    First retrieves chunks using hybrid search, then reranks using
    a cross-encoder model for higher precision.
    """
    import time
    
    start_time = time.time()
    
    try:
        # Initialize RAG engine
        engine = RAGEngine()
        
        # Perform reranked retrieval
        result = await engine.retrieve_with_reranking(
            query=request.query,
            document_ids=request.document_ids,
            top_k=request.top_k,
            rerank_top_k=request.rerank_top_k,
            use_hybrid=request.use_hybrid,
            enable_reranking=request.enable_reranking,
        )
        
        # Convert to response format
        results = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "score": chunk.score,
                "metadata": chunk.metadata,
            }
            for chunk in result.chunks
        ]
        
        search_time = (time.time() - start_time) * 1000
        
        logger.info(
            "reranked_search_api_complete",
            query=request.query[:50],
            results=len(results),
            time_ms=search_time,
        )
        
        return RerankedSearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_time_ms=search_time,
            reranked=request.enable_reranking,
        )
        
    except Exception as e:
        logger.error("reranked_search_api_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reranked search failed: {str(e)}")


@router.post("/execute", response_model=AdvancedHybridResponse)
async def advanced_hybrid_execute(request: AdvancedHybridRequest) -> AdvancedHybridResponse:
    """Execute advanced hybrid strategy with full pipeline.
    
    Performs the complete hybrid RAG-RLM pipeline:
    1. Hybrid retrieval
    2. Cross-encoder reranking
    3. Adaptive chunk selection
    4. RLM deep analysis
    5. Citation generation
    """
    import time
    
    start_time = time.time()
    
    try:
        # Initialize strategy
        strategy = AdvancedHybridStrategy()
        
        # Create cost estimate
        from rlm.routing.models import CostEstimate
        
        cost_estimate = CostEstimate(
            estimated_input_tokens=8000,
            estimated_output_tokens=1000,
            estimated_total_tokens=9000,
            estimated_cost_usd=0.015,
            model_used="gpt-4o",
        )
        
        # Execute
        result = await strategy.execute(
            query=request.query,
            document_ids=request.document_ids,
            cost_estimate=cost_estimate,
            enable_reranking=request.enable_reranking,
            enable_citations=request.enable_citations,
            enable_adaptive=request.enable_adaptive,
            semantic_weight=request.semantic_weight,
            keyword_weight=request.keyword_weight,
            max_chunks=request.max_chunks,
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        # Extract citations if available
        citations = None
        if result.metadata and "citations" in result.metadata:
            citations = result.metadata["citations"]
        
        logger.info(
            "advanced_hybrid_execute_api_complete",
            query=request.query[:50],
            execution_time_ms=execution_time,
        )
        
        return AdvancedHybridResponse(
            query=request.query,
            answer=result.answer,
            execution_time_ms=execution_time,
            chunks_analyzed=result.metadata.get("chunks", {}).get("final_selected", 0),
            sub_llm_calls=result.sub_llm_calls,
            citations=citations,
            step_times=result.metadata.get("step_times", {}),
            search_params=result.metadata.get("search_params", {}),
        )
        
    except Exception as e:
        logger.error("advanced_hybrid_execute_api_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")


@router.get("/collection-stats")
async def get_collection_stats() -> Dict[str, Any]:
    """Get statistics about the hybrid search collection."""
    try:
        engine = RAGEngine()
        stats = engine.get_collection_stats()
        
        return {
            "status": "success",
            "stats": stats,
        }
        
    except Exception as e:
        logger.error("collection_stats_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/search-modes")
async def get_search_modes() -> Dict[str, Any]:
    """Get available search modes and their descriptions."""
    return {
        "search_modes": {
            "hybrid": {
                "description": "Combines semantic and keyword search with RRF fusion",
                "best_for": "General queries, balanced precision and recall",
                "parameters": ["semantic_weight", "keyword_weight"],
            },
            "semantic_only": {
                "description": "Pure vector-based semantic search",
                "best_for": "Conceptual queries, understanding meaning",
                "parameters": [],
            },
            "keyword_only": {
                "description": "BM25 keyword search",
                "best_for": "Specific terms, exact matches",
                "parameters": [],
            },
        },
        "reranking_options": {
            "cross_encoder": {
                "description": "Cross-encoder reranking for high precision",
                "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            },
            "llm": {
                "description": "LLM-based reranking (slower, more accurate)",
                "models": ["gpt-5-mini", "gpt-4o-mini"],
            },
            "multi_stage": {
                "description": "Cross-encoder followed by LLM reranking",
                "best_for": "Maximum accuracy at higher cost",
            },
        },
    }

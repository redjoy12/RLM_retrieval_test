"""API routes for query execution with routing and visibility."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from rlm.routing.query_router import QueryRouter, RoutingService
from rlm.routing.models import RoutingDecision

router = APIRouter(prefix="/queries", tags=["queries"])


class QueryRequest(BaseModel):
    """Request model for query execution."""
    
    query: str = Field(..., description="The user's query")
    document_ids: List[str] = Field(..., description="Document IDs to query")
    use_optimized_query: bool = Field(
        default=False,
        description="Use LLM-optimized query if available"
    )
    show_routing_info: bool = Field(
        default=True,
        description="Include routing decision in response"
    )


class QueryResponse(BaseModel):
    """Response model for query execution."""
    
    answer: str = Field(..., description="The answer to the query")
    execution_time_ms: float = Field(..., description="Total execution time")
    tokens_used: int = Field(..., description="Total tokens consumed")
    cost_usd: float = Field(..., description="Actual cost in USD")
    routing: Optional[dict] = Field(
        default=None,
        description="Routing decision and visibility"
    )


class QueryAnalysisRequest(BaseModel):
    """Request model for query analysis."""
    
    query: str = Field(..., description="The user's query")
    document_ids: List[str] = Field(..., description="Document IDs to query")


class QueryAnalysisResponse(BaseModel):
    """Response model for query analysis."""
    
    query: str = Field(..., description="The analyzed query")
    routing_decision: dict = Field(..., description="Complete routing decision")
    visibility: dict = Field(..., description="User-friendly routing info")
    suggestions: Optional[dict] = Field(
        default=None,
        description="Query optimization suggestions"
    )


@router.post("/execute", response_model=QueryResponse)
async def execute_query(request: QueryRequest) -> QueryResponse:
    """Execute a query with intelligent routing.
    
    This endpoint:
    1. Analyzes query complexity
    2. Estimates costs
    3. Selects optimal strategy (Direct LLM, RAG, RLM, or Hybrid)
    4. Executes with selected strategy
    5. Returns answer with routing visibility
    
    Example:
        ```json
        {
            "query": "What are the main findings?",
            "document_ids": ["doc-123"],
            "show_routing_info": true
        }
        ```
    """
    try:
        router = QueryRouter()
        
        result = await router.route_and_execute(
            query=request.query,
            document_ids=request.document_ids,
            return_routing_info=request.show_routing_info,
        )
        
        return QueryResponse(
            answer=result["answer"],
            execution_time_ms=result["execution_time_ms"],
            tokens_used=result["tokens_used"],
            cost_usd=result["cost_usd"],
            routing=result.get("routing"),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query execution failed: {str(e)}"
        )


@router.post("/analyze", response_model=QueryAnalysisResponse)
async def analyze_query(request: QueryAnalysisRequest) -> QueryAnalysisResponse:
    """Analyze a query without executing.
    
    Returns routing decision, cost estimates, and optimization
    suggestions without actually executing the query.
    
    Example:
        ```json
        {
            "query": "Compare revenue trends",
            "document_ids": ["doc-123"]
        }
        ```
    
    Response:
        ```json
        {
            "routing_decision": {
                "strategy": "hybrid",
                "estimated_cost": 0.015,
                "confidence": 0.88
            },
            "visibility": {
                "strategy": "HYBRID",
                "reasoning": "Large context with multi-hop analysis"
            }
        }
        ```
    """
    try:
        service = RoutingService()
        
        # Get routing analysis
        analysis = await service.analyze(
            query=request.query,
            document_ids=request.document_ids,
        )
        
        # Get optimization suggestions
        from rlm.routing.optimizer import QueryOptimizer
        optimizer = QueryOptimizer()
        
        from rlm.routing.analyzer import QueryAnalyzer
        analyzer = QueryAnalyzer()
        query_analysis = analyzer.analyze(request.query)
        
        optimization = await optimizer.optimize(
            request.query,
            query_analysis.complexity,
        )
        
        return QueryAnalysisResponse(
            query=request.query,
            routing_decision=analysis["routing_decision"],
            visibility=analysis["visibility"],
            suggestions=optimization.to_dict() if optimization.suggestions else None,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query analysis failed: {str(e)}"
        )


@router.get("/strategies")
async def list_strategies() -> dict:
    """List available execution strategies.
    
    Returns information about all available routing strategies
    and when to use each.
    """
    from rlm.routing.models import ExecutionStrategy
    
    strategies = {
        "direct_llm": {
            "name": "Direct LLM",
            "description": "Simple Q&A with direct LLM call",
            "best_for": "Simple questions with small context (<10K chars)",
            "cost": "Low",
            "latency": "Fast",
        },
        "rag": {
            "name": "RAG",
            "description": "Retrieval-Augmented Generation with vector search",
            "best_for": "Single-hop questions with medium context (<500K chars)",
            "cost": "Medium",
            "latency": "Medium",
        },
        "rlm": {
            "name": "RLM",
            "description": "Recursive Language Model with deep analysis",
            "best_for": "Multi-hop or complex reasoning questions",
            "cost": "Higher",
            "latency": "Slower",
        },
        "hybrid": {
            "name": "Hybrid RAG+RLM",
            "description": "RAG pre-filtering followed by RLM deep analysis",
            "best_for": "Large documents with complex questions",
            "cost": "Medium-High",
            "latency": "Medium",
        },
    }
    
    return {
        "strategies": strategies,
        "total": len(strategies),
    }


@router.get("/cost-estimate")
async def estimate_cost(
    query: str = Query(..., description="Query to estimate"),
    document_ids: List[str] = Query(..., description="Document IDs"),
    strategy: Optional[str] = Query(
        default=None,
        description="Specific strategy to estimate (auto-detect if None)"
    ),
) -> dict:
    """Estimate cost for a query without executing.
    
    Provides detailed cost breakdown for the query based on
    the selected or auto-detected strategy.
    
    Query Parameters:
        - query: The query string
        - document_ids: List of document IDs (comma-separated)
        - strategy: Optional specific strategy
    
    Example:
        GET /queries/cost-estimate?query=What+is+AI&document_ids=doc-123,doc-456
    """
    try:
        from rlm.routing.cost_estimator import CostEstimator
        from rlm.documents.storage import DocumentStorage
        
        # Get context size
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
        
        # Get cost estimates
        estimator = CostEstimator()
        comparisons = estimator.compare_strategies(query, total_size)
        
        # Format response
        response = {
            "query": query,
            "context_size_chars": total_size,
            "estimates": {
                name: {
                    "input_tokens": est.estimated_input_tokens,
                    "output_tokens": est.estimated_output_tokens,
                    "total_tokens": est.estimated_total_tokens,
                    "cost_usd": round(est.estimated_cost_usd, 6),
                    "cost_with_buffer": round(est.cost_with_buffer, 6),
                }
                for name, est in comparisons.items()
            },
        }
        
        # If strategy specified, highlight it
        if strategy:
            response["selected_strategy"] = strategy
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cost estimation failed: {str(e)}"
        )

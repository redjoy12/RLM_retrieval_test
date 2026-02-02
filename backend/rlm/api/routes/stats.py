"""Statistics API routes for RLM Document Retrieval System.

Provides endpoints for usage statistics, cost tracking, and system metrics.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

import structlog

from rlm.config import get_settings
from rlm.llm.cost_tracker import CostTracker
from rlm.session.manager import SessionManager

logger = structlog.get_logger()

router = APIRouter(prefix="/stats", tags=["statistics"])

# Global instances
_cost_tracker: Optional[CostTracker] = None
_session_manager: Optional[SessionManager] = None


def get_cost_tracker() -> CostTracker:
    """Get or create cost tracker instance."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker


def get_session_manager() -> SessionManager:
    """Get or create session manager instance."""
    global _session_manager
    if _session_manager is None:
        from rlm.config import get_session_settings
        settings = get_session_settings()
        _session_manager = SessionManager(
            db_path=settings.db_path,
            max_tokens=settings.max_tokens_per_session,
        )
    return _session_manager


# Response Models

class UsageStats(BaseModel):
    """Usage statistics response."""
    total_queries: int = Field(..., description="Total number of queries")
    total_sessions: int = Field(..., description="Total number of sessions")
    total_documents: int = Field(..., description="Total number of documents")
    total_tokens: int = Field(..., description="Total tokens consumed")
    period: str = Field(..., description="Time period for stats")


class CostStats(BaseModel):
    """Cost statistics response."""
    total_cost_usd: float = Field(..., description="Total cost in USD")
    cost_by_model: Dict[str, float] = Field(..., description="Cost breakdown by model")
    cost_by_day: Dict[str, float] = Field(..., description="Cost breakdown by day")
    average_cost_per_query: float = Field(..., description="Average cost per query")
    period: str = Field(..., description="Time period for stats")


class ModelUsageStats(BaseModel):
    """Model usage statistics."""
    model: str = Field(..., description="Model name")
    queries: int = Field(..., description="Number of queries")
    tokens: int = Field(..., description="Total tokens")
    cost_usd: float = Field(..., description="Total cost")


class SystemHealth(BaseModel):
    """System health status."""
    status: str = Field(..., description="Overall system status")
    version: str = Field(..., description="System version")
    uptime_seconds: int = Field(..., description="System uptime in seconds")
    active_sessions: int = Field(..., description="Number of active sessions")
    components: Dict[str, str] = Field(..., description="Component health statuses")


class QueryStats(BaseModel):
    """Query execution statistics."""
    total_queries: int = Field(..., description="Total queries executed")
    avg_execution_time_ms: float = Field(..., description="Average execution time")
    success_rate: float = Field(..., description="Success rate (0-1)")
    queries_by_strategy: Dict[str, int] = Field(..., description="Queries by routing strategy")


class DocumentStats(BaseModel):
    """Document processing statistics."""
    total_documents: int = Field(..., description="Total documents")
    total_chunks: int = Field(..., description="Total chunks")
    documents_by_format: Dict[str, int] = Field(..., description="Documents by format")
    avg_document_size_bytes: float = Field(..., description="Average document size")


# API Endpoints

@router.get("/usage", response_model=UsageStats)
async def get_usage_stats(
    days: int = Query(default=7, ge=1, le=90, description="Number of days to look back"),
) -> UsageStats:
    """Get usage statistics for the specified period.
    
    Args:
        days: Number of days to look back
        
    Returns:
        Usage statistics
    """
    logger.info("usage_stats_request", days=days)
    
    try:
        session_manager = get_session_manager()
        cost_tracker = get_cost_tracker()
        
        # Get all sessions
        sessions = session_manager.list_sessions()
        
        # Calculate stats
        total_sessions = len(sessions)
        total_queries = sum(len(s.messages) for s in sessions)
        total_tokens = cost_tracker.get_total_tokens()
        
        return UsageStats(
            total_queries=total_queries,
            total_sessions=total_sessions,
            total_documents=0,  # TODO: Implement document counting
            total_tokens=total_tokens,
            period=f"last_{days}_days",
        )
    except Exception as e:
        logger.error("usage_stats_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/costs", response_model=CostStats)
async def get_cost_stats(
    days: int = Query(default=7, ge=1, le=90, description="Number of days to look back"),
) -> CostStats:
    """Get cost statistics for the specified period.
    
    Args:
        days: Number of days to look back
        
    Returns:
        Cost statistics
    """
    logger.info("cost_stats_request", days=days)
    
    try:
        cost_tracker = get_cost_tracker()
        
        # Get cost summary
        summary = cost_tracker.get_summary()
        
        # Calculate cost by day (placeholder implementation)
        cost_by_day = {}
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            cost_by_day[date] = summary.total_cost_usd / days  # Even distribution for now
        
        return CostStats(
            total_cost_usd=summary.total_cost_usd,
            cost_by_model=summary.cost_by_model,
            cost_by_day=cost_by_day,
            average_cost_per_query=summary.avg_cost_per_query,
            period=f"last_{days}_days",
        )
    except Exception as e:
        logger.error("cost_stats_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=List[ModelUsageStats])
async def get_model_usage(
    days: int = Query(default=7, ge=1, le=90, description="Number of days to look back"),
) -> List[ModelUsageStats]:
    """Get model usage statistics.
    
    Args:
        days: Number of days to look back
        
    Returns:
        List of model usage statistics
    """
    logger.info("model_usage_request", days=days)
    
    try:
        cost_tracker = get_cost_tracker()
        summary = cost_tracker.get_summary()
        
        models = []
        for model, cost in summary.cost_by_model.items():
            models.append(ModelUsageStats(
                model=model,
                queries=summary.queries_by_model.get(model, 0),
                tokens=summary.tokens_by_model.get(model, 0),
                cost_usd=cost,
            ))
        
        return models
    except Exception as e:
        logger.error("model_usage_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/queries", response_model=QueryStats)
async def get_query_stats(
    days: int = Query(default=7, ge=1, le=90, description="Number of days to look back"),
) -> QueryStats:
    """Get query execution statistics.
    
    Args:
        days: Number of days to look back
        
    Returns:
        Query statistics
    """
    logger.info("query_stats_request", days=days)
    
    try:
        session_manager = get_session_manager()
        sessions = session_manager.list_sessions()
        
        total_queries = sum(len(s.messages) for s in sessions)
        
        # Placeholder values - would need to track in real implementation
        return QueryStats(
            total_queries=total_queries,
            avg_execution_time_ms=0.0,
            success_rate=1.0,
            queries_by_strategy={
                "direct_llm": 0,
                "rag": 0,
                "rlm": 0,
                "hybrid": 0,
            },
        )
    except Exception as e:
        logger.error("query_stats_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=DocumentStats)
async def get_document_stats() -> DocumentStats:
    """Get document processing statistics.
    
    Returns:
        Document statistics
    """
    logger.info("document_stats_request")
    
    try:
        # Placeholder - would need document storage integration
        return DocumentStats(
            total_documents=0,
            total_chunks=0,
            documents_by_format={},
            avg_document_size_bytes=0.0,
        )
    except Exception as e:
        logger.error("document_stats_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=SystemHealth)
async def get_system_health() -> SystemHealth:
    """Get system health status.
    
    Returns:
        System health information
    """
    logger.info("health_check_request")
    
    try:
        settings = get_settings()
        session_manager = get_session_manager()
        
        # Count active sessions
        sessions = session_manager.list_sessions()
        active_sessions = len([s for s in sessions if s.status.value == "active"])
        
        return SystemHealth(
            status="healthy",
            version="0.1.0",
            uptime_seconds=0,  # Would need to track actual uptime
            active_sessions=active_sessions,
            components={
                "api": "healthy",
                "llm_client": "healthy",
                "session_manager": "healthy",
                "document_processor": "healthy",
            },
        )
    except Exception as e:
        logger.error("health_check_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_summary_stats(
    days: int = Query(default=7, ge=1, le=90, description="Number of days to look back"),
) -> Dict:
    """Get a summary of all statistics.
    
    Args:
        days: Number of days to look back
        
    Returns:
        Combined statistics summary
    """
    logger.info("summary_stats_request", days=days)
    
    try:
        usage = await get_usage_stats(days)
        costs = await get_cost_stats(days)
        queries = await get_query_stats(days)
        health = await get_system_health()
        
        return {
            "period": f"last_{days}_days",
            "usage": usage.model_dump(),
            "costs": costs.model_dump(),
            "queries": queries.model_dump(),
            "health": health.model_dump(),
            "generated_at": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error("summary_stats_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

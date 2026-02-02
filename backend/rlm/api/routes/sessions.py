"""API routes for session management - Component 9.

Provides REST endpoints for:
- Session lifecycle (create, get, list, delete)
- Conversation management (add message, get context)
- Session search with Component 8 integration
- Citation management
- Session forking
- Conversation history search
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

import structlog

from rlm.config import get_session_settings
from rlm.session import SessionManager
from rlm.session.integration import (
    CitationTracker,
    SearchHistoryLogger,
    SessionSearchEnhancer,
)
from rlm.session.types import SessionStatus

logger = structlog.get_logger()

router = APIRouter(prefix="/sessions", tags=["sessions"])

# Global session manager instance (initialized on first use)
_session_manager: Optional[SessionManager] = None
_search_enhancer: Optional[SessionSearchEnhancer] = None
_citation_tracker: Optional[CitationTracker] = None
_history_logger: Optional[SearchHistoryLogger] = None


def get_session_manager() -> SessionManager:
    """Get or create session manager instance."""
    global _session_manager
    if _session_manager is None:
        settings = get_session_settings()
        _session_manager = SessionManager(
            db_path=settings.db_path,
            max_tokens=settings.max_tokens_per_session,
            ttl_hours=settings.session_ttl_hours,
        )
    return _session_manager


def get_search_enhancer() -> SessionSearchEnhancer:
    """Get or create search enhancer instance."""
    global _search_enhancer
    if _search_enhancer is None:
        manager = get_session_manager()
        _search_enhancer = SessionSearchEnhancer(
            session_manager=manager,
            context_history_size=get_session_settings().context_enhancement_history_size,
        )
    return _search_enhancer


def get_citation_tracker() -> CitationTracker:
    """Get or create citation tracker instance."""
    global _citation_tracker
    if _citation_tracker is None:
        manager = get_session_manager()
        _citation_tracker = CitationTracker(session_manager=manager)
    return _citation_tracker


def get_history_logger() -> SearchHistoryLogger:
    """Get or create history logger instance."""
    global _history_logger
    if _history_logger is None:
        manager = get_session_manager()
        _history_logger = SearchHistoryLogger(session_manager=manager)
    return _history_logger


# Request/Response Models

class CreateSessionRequest(BaseModel):
    """Request to create a new session."""
    title: Optional[str] = Field(None, description="Session title")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")
    # Component 8 preferences
    default_search_strategy: str = Field("hybrid", description="Default search strategy")
    semantic_weight: float = Field(0.7, ge=0, le=1, description="Semantic search weight")
    keyword_weight: float = Field(0.3, ge=0, le=1, description="Keyword search weight")
    enable_reranking: bool = Field(True, description="Enable reranking")
    enable_citations: bool = Field(True, description="Enable citations")


class SessionResponse(BaseModel):
    """Session response."""
    id: str
    title: str
    status: str
    created_at: str
    last_activity: str
    expires_at: str
    total_tokens_used: int
    context_window_used: int
    message_count: int
    search_preferences: Dict[str, Any]


class AddMessageRequest(BaseModel):
    """Request to add a message."""
    role: str = Field(..., description="Message role: user, assistant, system, tool")
    content: str = Field(..., description="Message content")
    trajectory_id: Optional[str] = Field(None, description="Link to trajectory log")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class MessageResponse(BaseModel):
    """Message response."""
    id: int
    session_id: str
    role: str
    content: str
    tokens: int
    message_type: str
    created_at: str
    trajectory_id: Optional[str] = None


class ContextResponse(BaseModel):
    """Conversation context response."""
    session_id: str
    messages: List[Dict[str, Any]]
    total_tokens: int
    usage_percentage: float
    is_compacted: bool


class SessionQueryRequest(BaseModel):
    """Request to execute a query within a session."""
    query: str = Field(..., description="User query")
    document_ids: List[str] = Field(..., description="Document IDs to search")
    use_session_context: bool = Field(True, description="Enhance with session context")
    strategy: str = Field("hybrid", description="Search strategy")
    top_k: int = Field(10, ge=1, le=50, description="Number of results")
    enable_citations: Optional[bool] = Field(None, description="Enable citations (session default if None)")


class SessionQueryResponse(BaseModel):
    """Response from session query."""
    session_id: str
    query: str
    enhanced_query: Optional[str] = None
    answer: str
    execution_time_ms: float
    tokens_used: int
    citations: Optional[List[Dict[str, Any]]] = None
    search_strategy: str


class SearchHistoryResponse(BaseModel):
    """Search history response."""
    history: List[Dict[str, Any]]
    total: int


class CitationListResponse(BaseModel):
    """Citation list response."""
    citations: List[Dict[str, Any]]
    total: int


class SessionStatsResponse(BaseModel):
    """Session statistics response."""
    session_id: str
    title: str
    status: str
    message_count: int
    total_tokens_used: int
    usage_percentage: float
    is_expired: bool
    search_preferences: Dict[str, Any]


# Endpoints

@router.post("", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest) -> SessionResponse:
    """Create a new session.
    
    Creates a new conversation session with optional Component 8
    search preferences.
    """
    try:
        manager = get_session_manager()
        
        session = await manager.create_session(
            title=request.title,
            metadata=request.metadata,
            default_search_strategy=request.default_search_strategy,
            semantic_weight=request.semantic_weight,
            keyword_weight=request.keyword_weight,
            enable_reranking=request.enable_reranking,
            enable_citations=request.enable_citations,
        )
        
        return SessionResponse(
            id=session.id,
            title=session.title,
            status=session.status,
            created_at=session.created_at.isoformat() if session.created_at else "",
            last_activity=session.last_activity.isoformat() if session.last_activity else "",
            expires_at=session.expires_at.isoformat() if session.expires_at else "",
            total_tokens_used=session.total_tokens_used,
            context_window_used=session.context_window_used,
            message_count=0,
            search_preferences={
                "strategy": session.default_search_strategy,
                "semantic_weight": session.semantic_weight,
                "keyword_weight": session.keyword_weight,
                "enable_reranking": session.enable_reranking,
                "enable_citations": session.enable_citations,
            },
        )
        
    except Exception as e:
        logger.error("create_session_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str) -> SessionResponse:
    """Get session by ID."""
    try:
        manager = get_session_manager()
        session = await manager.get_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get message count
        message_count = await manager.get_message_count(session_id)
        
        return SessionResponse(
            id=session.id,
            title=session.title,
            status=session.status,
            created_at=session.created_at.isoformat() if session.created_at else "",
            last_activity=session.last_activity.isoformat() if session.last_activity else "",
            expires_at=session.expires_at.isoformat() if session.expires_at else "",
            total_tokens_used=session.total_tokens_used,
            context_window_used=session.context_window_used,
            message_count=message_count,
            search_preferences={
                "strategy": session.default_search_strategy,
                "semantic_weight": session.semantic_weight,
                "keyword_weight": session.keyword_weight,
                "enable_reranking": session.enable_reranking,
                "enable_citations": session.enable_citations,
            },
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_session_failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")


@router.get("", response_model=List[SessionResponse])
async def list_sessions(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> List[SessionResponse]:
    """List sessions with optional filtering."""
    try:
        manager = get_session_manager()
        
        status_filter = None
        if status:
            try:
                status_filter = SessionStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        sessions = await manager.list_sessions(
            status=status_filter,
            limit=limit,
            offset=offset,
        )
        
        response = []
        for session in sessions:
            message_count = await manager.get_message_count(session.id)
            response.append(SessionResponse(
                id=session.id,
                title=session.title,
                status=session.status,
                created_at=session.created_at.isoformat() if session.created_at else "",
                last_activity=session.last_activity.isoformat() if session.last_activity else "",
                expires_at=session.expires_at.isoformat() if session.expires_at else "",
                total_tokens_used=session.total_tokens_used,
                context_window_used=session.context_window_used,
                message_count=message_count,
                search_preferences={
                    "strategy": session.default_search_strategy,
                    "semantic_weight": session.semantic_weight,
                    "keyword_weight": session.keyword_weight,
                    "enable_reranking": session.enable_reranking,
                    "enable_citations": session.enable_citations,
                },
            ))
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("list_sessions_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@router.delete("/{session_id}")
async def delete_session(session_id: str) -> Dict[str, str]:
    """Delete a session and all associated data."""
    try:
        manager = get_session_manager()
        deleted = await manager.delete_session(session_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"status": "success", "message": f"Session {session_id} deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("delete_session_failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


@router.post("/{session_id}/messages", response_model=MessageResponse)
async def add_message(session_id: str, request: AddMessageRequest) -> MessageResponse:
    """Add a message to a session."""
    try:
        manager = get_session_manager()
        
        # Verify session exists
        session = await manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Add message
        message = await manager.add_message(
            session_id=session_id,
            role=request.role,
            content=request.content,
            trajectory_id=request.trajectory_id,
            metadata=request.metadata,
        )
        
        return MessageResponse(
            id=message.id,
            session_id=message.session_id,
            role=message.role,
            content=message.content,
            tokens=message.tokens,
            message_type=message.message_type,
            created_at=message.created_at.isoformat() if message.created_at else "",
            trajectory_id=message.trajectory_id,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("add_message_failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to add message: {str(e)}")


@router.get("/{session_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    session_id: str,
    limit: Optional[int] = Query(None, description="Maximum messages"),
) -> List[MessageResponse]:
    """Get messages for a session."""
    try:
        manager = get_session_manager()
        
        # Verify session exists
        session = await manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        messages = await manager.get_messages(session_id, limit=limit)
        
        return [
            MessageResponse(
                id=msg.id,
                session_id=msg.session_id,
                role=msg.role,
                content=msg.content,
                tokens=msg.tokens,
                message_type=msg.message_type,
                created_at=msg.created_at.isoformat() if msg.created_at else "",
                trajectory_id=msg.trajectory_id,
            )
            for msg in messages
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_messages_failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")


@router.get("/{session_id}/context", response_model=ContextResponse)
async def get_context(session_id: str) -> ContextResponse:
    """Get conversation context for LLM."""
    try:
        manager = get_session_manager()
        
        # Verify session exists
        session = await manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        context = await manager.get_context(session_id)
        compaction_stats = await manager.get_compaction_stats(session_id)
        
        return ContextResponse(
            session_id=session_id,
            messages=[msg.to_dict() for msg in context],
            total_tokens=sum(msg.tokens for msg in context),
            usage_percentage=compaction_stats.get("usage_percentage", 0),
            is_compacted=any(msg.message_type.value == "summary" for msg in context),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_context_failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get context: {str(e)}")


@router.get("/{session_id}/stats", response_model=SessionStatsResponse)
async def get_session_stats(session_id: str) -> SessionStatsResponse:
    """Get comprehensive session statistics."""
    try:
        manager = get_session_manager()
        stats = await manager.get_session_stats(session_id)
        
        if not stats:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionStatsResponse(
            session_id=stats["session_id"],
            title=stats["title"],
            status=stats["status"],
            message_count=stats["message_count"],
            total_tokens_used=stats["total_tokens_used"],
            usage_percentage=stats["usage_percentage"],
            is_expired=stats["is_expired"],
            search_preferences=stats["search_preferences"],
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_session_stats_failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/{session_id}/search")
async def search_conversation(
    session_id: str,
    query: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100),
) -> Dict[str, Any]:
    """Search within a conversation using FTS5."""
    try:
        manager = get_session_manager()
        
        # Verify session exists
        session = await manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        results = await manager.search_conversation(session_id, query, limit)
        
        return {
            "query": query,
            "results": [msg.to_dict() for msg in results],
            "total": len(results),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("search_conversation_failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/{session_id}/fork", response_model=SessionResponse)
async def fork_session(
    session_id: str,
    title: Optional[str] = Query(None, description="New session title"),
) -> SessionResponse:
    """Fork a session, copying all messages and settings."""
    try:
        manager = get_session_manager()
        
        new_session = await manager.fork_session(session_id, title)
        message_count = await manager.get_message_count(new_session.id)
        
        return SessionResponse(
            id=new_session.id,
            title=new_session.title,
            status=new_session.status,
            created_at=new_session.created_at.isoformat() if new_session.created_at else "",
            last_activity=new_session.last_activity.isoformat() if new_session.last_activity else "",
            expires_at=new_session.expires_at.isoformat() if new_session.expires_at else "",
            total_tokens_used=new_session.total_tokens_used,
            context_window_used=new_session.context_window_used,
            message_count=message_count,
            search_preferences={
                "strategy": new_session.default_search_strategy,
                "semantic_weight": new_session.semantic_weight,
                "keyword_weight": new_session.keyword_weight,
                "enable_reranking": new_session.enable_reranking,
                "enable_citations": new_session.enable_citations,
            },
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("fork_session_failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fork session: {str(e)}")


@router.get("/{session_id}/search-history", response_model=SearchHistoryResponse)
async def get_search_history(
    session_id: str,
    limit: int = Query(50, ge=1, le=200),
) -> SearchHistoryResponse:
    """Get Component 8 search history for a session."""
    try:
        manager = get_session_manager()
        
        # Verify session exists
        session = await manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        history = await manager.get_search_history(session_id, limit)
        
        return SearchHistoryResponse(
            history=[entry.to_dict() for entry in history],
            total=len(history),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_search_history_failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.get("/{session_id}/citations", response_model=CitationListResponse)
async def get_citations(
    session_id: str,
    message_id: Optional[int] = Query(None, description="Filter by message ID"),
) -> CitationListResponse:
    """Get citations for a session or specific message."""
    try:
        manager = get_session_manager()
        
        # Verify session exists
        session = await manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        citations = await manager.get_citations(session_id, message_id)
        
        return CitationListResponse(
            citations=[cite.to_dict() for cite in citations],
            total=len(citations),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_citations_failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get citations: {str(e)}")


@router.get("/{session_id}/export")
async def export_session(
    session_id: str,
    format: str = Query("json", description="Export format: json, csv, markdown, html, txt"),
    include_search_history: bool = Query(True, description="Include search history"),
    include_citations: bool = Query(True, description="Include citations"),
) -> Dict[str, Any]:
    """Export session data in various formats."""
    try:
        manager = get_session_manager()
        
        # Verify session exists
        session = await manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        from rlm.session.export import ConversationExporter
        
        exporter = ConversationExporter(manager)
        
        # Export session
        data = await exporter.export_session(
            session_id=session_id,
            format=format,
            include_search_history=include_search_history,
            include_citations=include_citations,
        )
        
        return {
            "session_id": session_id,
            "format": format,
            "data": data,
            "filename": f"{session.title.replace(' ', '_')}.{format}",
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("export_session_failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/{session_id}/export/summary")
async def get_export_summary(session_id: str) -> Dict[str, Any]:
    """Get summary of what would be exported for a session."""
    try:
        manager = get_session_manager()
        
        # Verify session exists
        session = await manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        from rlm.session.export import ConversationExporter
        
        exporter = ConversationExporter(manager)
        summary = await exporter.get_export_summary(session_id)
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("export_summary_failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")

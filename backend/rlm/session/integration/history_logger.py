"""Search history logger for Component 8 integration.

Logs all search operations with full parameters for analytics
and session context understanding.
"""

from typing import Any, Dict, List, Optional

import structlog

from rlm.session.manager import SessionManager
from rlm.session.types import SearchHistoryEntry

logger = structlog.get_logger()


class SearchHistoryLogger:
    """Logs search operations for Component 8 integration.
    
    Provides comprehensive logging of hybrid search operations
    including parameters, results, and performance metrics.
    
    Example:
        ```python
        logger = SearchHistoryLogger(session_manager)
        
        # Log a search operation
        await logger.log_hybrid_search(
            session_id="session-123",
            query="machine learning",
            document_ids=["doc-1"],
            results=[...],
            params={"semantic_weight": 0.7}
        )
        
        # Get search analytics
        analytics = await logger.get_search_analytics(session_id)
        ```
    """
    
    def __init__(self, session_manager: SessionManager) -> None:
        """Initialize search history logger.
        
        Args:
            session_manager: Session manager instance
        """
        self.session_manager = session_manager
        
        logger.info("search_history_logger_initialized")
    
    async def log_hybrid_search(
        self,
        session_id: str,
        query: str,
        document_ids: List[str],
        results: List[Dict[str, Any]],
        execution_time_ms: float,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        reranked: bool = False,
        adaptive: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SearchHistoryEntry:
        """Log a hybrid search operation.
        
        Args:
            session_id: Session ID
            query: Search query
            document_ids: Documents searched
            results: Search results
            execution_time_ms: Execution time
            semantic_weight: Semantic search weight
            keyword_weight: Keyword search weight
            reranked: Whether reranking was applied
            adaptive: Whether adaptive selection was used
            metadata: Additional metadata
            
        Returns:
            SearchHistoryEntry record
        """
        # Determine strategy
        if adaptive:
            strategy = "adaptive"
        elif reranked:
            strategy = "reranked"
        else:
            strategy = "hybrid"
        
        # Create metadata
        full_metadata = {
            "document_ids": document_ids,
            "result_chunk_ids": [r.get("chunk_id") for r in results],
            "result_scores": [r.get("score") for r in results],
            **(metadata or {}),
        }
        
        # Log via session manager
        record = await self.session_manager.log_search(
            session_id=session_id,
            query=query,
            strategy=strategy,
            results_count=len(results),
            execution_time_ms=execution_time_ms,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            metadata=full_metadata,
        )
        
        logger.debug(
            "hybrid_search_logged",
            session_id=session_id,
            strategy=strategy,
            results=len(results),
        )
        
        return SearchHistoryEntry(
            id=record.id,
            session_id=record.session_id,
            query=record.query,
            strategy=record.strategy,
            results_count=record.results_count,
            execution_time_ms=record.execution_time_ms,
            semantic_weight=record.semantic_weight,
            keyword_weight=record.keyword_weight,
            created_at=record.created_at,
        )
    
    async def get_search_analytics(
        self,
        session_id: str,
    ) -> Dict[str, Any]:
        """Get search analytics for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Analytics dictionary
        """
        history = await self.session_manager.get_search_history(session_id)
        
        if not history:
            return {
                "total_searches": 0,
                "avg_execution_time_ms": 0,
                "strategy_breakdown": {},
                "avg_results_count": 0,
            }
        
        # Calculate metrics
        total = len(history)
        avg_time = sum(h.execution_time_ms for h in history) / total
        avg_results = sum(h.results_count for h in history) / total
        
        # Strategy breakdown
        strategies = {}
        for h in history:
            strategies[h.strategy] = strategies.get(h.strategy, 0) + 1
        
        # Recent trends (last 10 searches)
        recent = history[:10]
        recent_avg_time = sum(h.execution_time_ms for h in recent) / len(recent)
        
        return {
            "total_searches": total,
            "avg_execution_time_ms": avg_time,
            "recent_avg_time_ms": recent_avg_time,
            "strategy_breakdown": strategies,
            "avg_results_count": avg_results,
            "most_common_strategy": max(strategies.items(), key=lambda x: x[1])[0],
        }
    
    async def get_similar_searches(
        self,
        session_id: str,
        query: str,
        limit: int = 5,
    ) -> List[SearchHistoryEntry]:
        """Find similar past searches in the session.
        
        Args:
            session_id: Session ID
            query: Query to compare
            limit: Maximum results
            
        Returns:
            List of similar SearchHistoryEntry objects
        """
        history = await self.session_manager.get_search_history(session_id)
        
        # Simple keyword-based similarity
        query_words = set(query.lower().split())
        
        scored = []
        for entry in history:
            entry_words = set(entry.query.lower().split())
            # Jaccard similarity
            intersection = len(query_words & entry_words)
            union = len(query_words | entry_words)
            similarity = intersection / union if union > 0 else 0
            
            if similarity > 0.3:  # Threshold
                scored.append((similarity, entry))
        
        # Sort by similarity and return top
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:limit]]
    
    async def get_search_timeline(
        self,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """Get chronological search timeline for visualization.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of search events with timestamps
        """
        history = await self.session_manager.get_search_history(
            session_id, limit=100
        )
        
        # Reverse to get chronological order
        history = list(reversed(history))
        
        timeline = []
        for i, entry in enumerate(history, 1):
            timeline.append({
                "sequence": i,
                "timestamp": entry.created_at.isoformat(),
                "query": entry.query,
                "strategy": entry.strategy,
                "results_count": entry.results_count,
                "execution_time_ms": entry.execution_time_ms,
            })
        
        return timeline
    
    async def export_search_history(
        self,
        session_id: str,
        format: str = "json",
    ) -> str:
        """Export search history to various formats.
        
        Args:
            session_id: Session ID
            format: Export format (json, csv, markdown)
            
        Returns:
            Exported data as string
        """
        history = await self.session_manager.get_search_history(
            session_id, limit=1000
        )
        
        if format == "json":
            import json
            return json.dumps(
                [h.to_dict() for h in history],
                indent=2
            )
        
        elif format == "csv":
            lines = ["timestamp,query,strategy,results_count,execution_time_ms"]
            for h in history:
                # Escape quotes in query
                query = h.query.replace('"', '""')
                lines.append(
                    f'"{h.created_at.isoformat()}",'
                    f'"{query}",'
                    f'"{h.strategy}",'
                    f'{h.results_count},'
                    f'{h.execution_time_ms}'
                )
            return "\n".join(lines)
        
        elif format == "markdown":
            lines = ["# Search History", ""]
            for h in history:
                lines.extend([
                    f"## {h.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
                    f"- **Query**: {h.query}",
                    f"- **Strategy**: {h.strategy}",
                    f"- **Results**: {h.results_count}",
                    f"- **Time**: {h.execution_time_ms:.2f}ms",
                    "",
                ])
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported format: {format}")

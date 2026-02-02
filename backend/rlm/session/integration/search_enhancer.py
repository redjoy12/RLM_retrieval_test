"""Session-aware search enhancement for Component 8.

Enhances Hybrid RAG search using conversation context from sessions.
"""

from typing import Any, Dict, List, Optional

import structlog

from rlm.hybrid import HybridSearcher
from rlm.session.manager import SessionManager
from rlm.session.types import MessageRole

logger = structlog.get_logger()


class SessionSearchEnhancer:
    """Enhances Component 8 hybrid search using session context.

    Automatically uses conversation history to improve search queries,
    handling pronouns and references from previous exchanges.

    Example:
        ```python
        enhancer = SessionSearchEnhancer(session_manager, hybrid_searcher)

        # Search with automatic context enhancement
        result = await enhancer.contextual_search(
            session_id="session-123",
            query="How does it work?",  # "it" resolved from context
            document_ids=["doc-1"]
        )
        ```
    """

    def __init__(
        self,
        session_manager: SessionManager,
        hybrid_searcher: Optional[HybridSearcher] = None,
        context_history_size: int = 3,
    ) -> None:
        """Initialize search enhancer.

        Args:
            session_manager: Session manager instance
            hybrid_searcher: Hybrid searcher instance (created if None)
            context_history_size: Number of previous queries to include
        """
        self.session_manager = session_manager
        self.hybrid_searcher = hybrid_searcher or HybridSearcher()
        self.context_history_size = context_history_size

        logger.info(
            "search_enhancer_initialized",
            context_history_size=context_history_size,
        )

    async def contextual_search(
        self,
        session_id: str,
        query: str,
        document_ids: List[str],
        top_k: int = 10,
        semantic_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
        use_context: bool = True,
    ) -> Dict[str, Any]:
        """Perform search enhanced with session context.

        Args:
            session_id: Session ID for context
            query: User query
            document_ids: Documents to search
            top_k: Number of results
            semantic_weight: Semantic search weight (uses session default if None)
            keyword_weight: Keyword search weight (uses session default if None)
            use_context: Whether to enhance query with session context

        Returns:
            Search results with metadata
        """
        # Get session for search preferences
        session = await self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Use session preferences if not specified
        semantic_weight = semantic_weight or session.semantic_weight
        keyword_weight = keyword_weight or session.keyword_weight

        # Enhance query with context if enabled
        enhanced_query = query
        if use_context:
            enhanced_query = await self._enhance_query(session_id, query)

        # Perform hybrid search
        import time

        start_time = time.time()

        try:
            search_result = await self.hybrid_searcher.search(
                query=enhanced_query,
                document_ids=document_ids,
                top_k=top_k,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
            )

            execution_time_ms = (time.time() - start_time) * 1000

            # Log search to session history
            await self.session_manager.log_search(
                session_id=session_id,
                query=query,
                strategy="hybrid_enhanced" if use_context else "hybrid",
                results_count=len(search_result),
                execution_time_ms=execution_time_ms,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
                metadata={
                    "enhanced_query": enhanced_query if use_context else None,
                    "original_query": query,
                },
            )

            logger.info(
                "contextual_search_complete",
                session_id=session_id,
                results=len(search_result),
                enhanced=use_context,
            )

            return {
                "query": query,
                "enhanced_query": enhanced_query if use_context else None,
                "results": search_result,
                "total_results": len(search_result),
                "execution_time_ms": execution_time_ms,
                "semantic_weight": semantic_weight,
                "keyword_weight": keyword_weight,
                "used_context": use_context,
            }

        except Exception as e:
            logger.error(
                "contextual_search_failed",
                session_id=session_id,
                error=str(e),
            )
            raise

    async def _enhance_query(self, session_id: str, query: str) -> str:
        """Enhance query using session context.

        Args:
            session_id: Session ID
            query: Original query

        Returns:
            Enhanced query with context
        """
        # Get recent conversation context
        recent_queries = await self.session_manager.get_recent_queries(
            session_id, n=self.context_history_size
        )

        if not recent_queries:
            return query

        # Check if query contains pronouns or ambiguous references
        pronouns = ["it", "its", "they", "them", "their", "this", "that", "these", "those"]
        query_lower = query.lower()

        needs_context = any(pronoun in query_lower.split() for pronoun in pronouns)

        if not needs_context:
            # Check for short queries that might benefit from context
            if len(query.split()) < 5:
                needs_context = True

        if not needs_context:
            return query

        # Build context string
        context_parts = []
        for i, prev_query in enumerate(recent_queries, 1):
            context_parts.append(f"Q{i}: {prev_query}")

        context_str = " | ".join(context_parts)

        # Enhance query with context
        enhanced = f"{query} [Context: {context_str}]"

        logger.debug(
            "query_enhanced",
            session_id=session_id,
            original=query,
            enhanced=enhanced,
        )

        return enhanced

    async def get_search_context(self, session_id: str) -> Dict[str, Any]:
        """Get search context summary for a session.

        Args:
            session_id: Session ID

        Returns:
            Context information
        """
        session = await self.session_manager.get_session(session_id)
        if not session:
            return {}

        recent_queries = await self.session_manager.get_recent_queries(
            session_id, n=self.context_history_size
        )

        search_history = await self.session_manager.get_search_history(session_id, limit=10)

        return {
            "session_id": session_id,
            "recent_queries": recent_queries,
            "search_count": len(search_history),
            "avg_execution_time_ms": (
                sum(h.execution_time_ms for h in search_history) / len(search_history)
                if search_history
                else 0
            ),
            "preferred_strategy": session.default_search_strategy,
            "semantic_weight": session.semantic_weight,
            "keyword_weight": session.keyword_weight,
        }

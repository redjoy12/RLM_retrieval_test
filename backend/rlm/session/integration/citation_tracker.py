"""Citation tracker for Component 8 integration.

Manages citations from hybrid RAG search results, storing them
inline with messages and providing retrieval capabilities.
"""

from typing import Any, Dict, List, Optional

import structlog

from rlm.hybrid import Citation as HybridCitation
from rlm.session.manager import SessionManager
from rlm.session.types import CitationEntry

logger = structlog.get_logger()


class CitationTracker:
    """Tracks citations from Component 8 hybrid search.
    
    Stores citations inline with conversation messages and provides
    retrieval capabilities for session context.
    
    Example:
        ```python
        tracker = CitationTracker(session_manager)
        
        # Add citations from Component 8 results
        citations = [Citation(chunk_id="abc", content="...", score=0.95)]
        await tracker.add_citations(session_id, message_id, citations)
        
        # Retrieve citations for a message
        cites = await tracker.get_citations_for_message(session_id, message_id)
        ```
    """
    
    def __init__(self, session_manager: SessionManager) -> None:
        """Initialize citation tracker.
        
        Args:
            session_manager: Session manager instance
        """
        self.session_manager = session_manager
        
        logger.info("citation_tracker_initialized")
    
    async def add_citations(
        self,
        session_id: str,
        message_id: int,
        citations: List[HybridCitation],
    ) -> List[CitationEntry]:
        """Add citations from Component 8 to a message.
        
        Args:
            session_id: Session ID
            message_id: Message ID that contains the citations
            citations: List of Component 8 Citation objects
            
        Returns:
            List of stored CitationEntry objects
        """
        entries = []
        
        for citation in citations:
            entry = await self.session_manager.add_citation(
                session_id=session_id,
                message_id=message_id,
                chunk_id=citation.chunk_id,
                document_id=citation.document_id,
                content_snippet=citation.content_snippet,
                score=citation.score,
            )
            entries.append(entry)
        
        logger.debug(
            "citations_added",
            session_id=session_id,
            message_id=message_id,
            count=len(entries),
        )
        
        return entries
    
    async def get_citations_for_message(
        self,
        session_id: str,
        message_id: int,
    ) -> List[CitationEntry]:
        """Get citations for a specific message.
        
        Args:
            session_id: Session ID
            message_id: Message ID
            
        Returns:
            List of CitationEntry objects
        """
        return await self.session_manager.get_citations(session_id, message_id)
    
    async def get_session_citations(
        self,
        session_id: str,
        document_id: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[CitationEntry]:
        """Get all citations for a session with optional filtering.
        
        Args:
            session_id: Session ID
            document_id: Optional filter by document
            min_score: Optional minimum score filter
            
        Returns:
            List of CitationEntry objects
        """
        citations = await self.session_manager.get_citations(session_id)
        
        # Apply filters
        if document_id:
            citations = [c for c in citations if c.document_id == document_id]
        
        if min_score is not None:
            citations = [c for c in citations if c.score >= min_score]
        
        return citations
    
    async def format_citations_inline(
        self,
        citations: List[CitationEntry],
        format_type: str = "markdown",
    ) -> str:
        """Format citations for inline display in messages.
        
        Args:
            citations: Citation entries to format
            format_type: Output format (markdown, html, plain)
            
        Returns:
            Formatted citation string
        """
        if not citations:
            return ""
        
        if format_type == "markdown":
            lines = ["\n\n**Sources:**"]
            for i, cite in enumerate(citations, 1):
                lines.append(
                    f"[{i}] Doc: {cite.document_id[:8]}... "
                    f"\"{cite.content_snippet[:100]}...\" "
                    f"(score: {cite.score:.2f})"
                )
            return "\n".join(lines)
        
        elif format_type == "html":
            lines = ['<div class="citations"><h4>Sources:</h4><ul>']
            for cite in citations:
                lines.append(
                    f'<li>Doc: {cite.document_id[:8]}... '
                    f'"{cite.content_snippet[:100]}..." '
                    f'(score: {cite.score:.2f})</li>'
                )
            lines.append('</ul></div>')
            return "\n".join(lines)
        
        else:  # plain
            lines = ["\nSources:"]
            for i, cite in enumerate(citations, 1):
                lines.append(
                    f"{i}. Document {cite.document_id[:8]}: "
                    f"{cite.content_snippet[:80]}..."
                )
            return "\n".join(lines)
    
    async def get_citation_stats(self, session_id: str) -> Dict[str, Any]:
        """Get citation statistics for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Statistics dictionary
        """
        citations = await self.session_manager.get_citations(session_id)
        
        if not citations:
            return {
                "total_citations": 0,
                "unique_documents": 0,
                "avg_score": 0,
                "top_documents": [],
            }
        
        # Calculate stats
        unique_docs = set(c.document_id for c in citations)
        avg_score = sum(c.score for c in citations) / len(citations)
        
        # Count citations per document
        doc_counts = {}
        for c in citations:
            doc_counts[c.document_id] = doc_counts.get(c.document_id, 0) + 1
        
        top_docs = sorted(
            doc_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "total_citations": len(citations),
            "unique_documents": len(unique_docs),
            "avg_score": avg_score,
            "top_documents": [
                {"document_id": doc_id, "citation_count": count}
                for doc_id, count in top_docs
            ],
        }

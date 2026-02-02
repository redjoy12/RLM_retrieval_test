"""Component 8 integration layer for session management.

Provides integration between Component 9 (Session Management) and
Component 8 (Hybrid RAG) for enhanced search capabilities.
"""

from rlm.session.integration.search_enhancer import SessionSearchEnhancer
from rlm.session.integration.citation_tracker import CitationTracker
from rlm.session.integration.history_logger import SearchHistoryLogger

__all__ = [
    "SessionSearchEnhancer",
    "CitationTracker", 
    "SearchHistoryLogger",
]

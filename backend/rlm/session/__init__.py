"""Session Management System - Component 9.

Provides persistent conversation management with:
- Session lifecycle management (create, resume, fork, delete)
- Conversation history with context compaction
- Token usage tracking for context window management
- FTS5-powered conversation search
- Component 8 integration (hybrid search, citations)
- Session-aware search enhancement

Example:
    ```python
    from rlm.session import SessionManager
    
    # Create or resume session
    manager = SessionManager()
    session = await manager.create_session("My Research")
    
    # Add messages
    await manager.add_message(session.id, "user", "What is AI?")
    await manager.add_message(session.id, "assistant", "AI is...")
    
    # Get context for LLM
    context = await manager.get_context(session.id)
    ```
"""

from rlm.session.manager import SessionManager
from rlm.session.models import (
    Message,
    SearchHistory,
    Session,
    SessionCitation,
    Summary,
)
from rlm.session.token_manager import TokenManager
from rlm.session.compactor import ContextCompactor
from rlm.session.cleanup import (
    SessionCleanupJob,
    CleanupScheduler,
    run_session_cleanup_now,
    start_background_cleanup,
)
from rlm.session.websocket import (
    SessionWebSocketManager,
    handle_session_websocket,
    SessionEventBroadcaster,
)
from rlm.session.export import (
    ConversationExporter,
    export_session_to_file,
)
from rlm.session.types import (
    SessionStatus,
    MessageType,
    SessionContext,
    MessageContext,
    SearchHistoryEntry,
    CitationEntry,
)

__all__ = [
    # Main manager
    "SessionManager",
    # Models
    "Session",
    "Message",
    "Summary",
    "SearchHistory",
    "SessionCitation",
    # Core components
    "TokenManager",
    "ContextCompactor",
    # Cleanup
    "SessionCleanupJob",
    "CleanupScheduler",
    "run_session_cleanup_now",
    "start_background_cleanup",
    # WebSocket
    "SessionWebSocketManager",
    "handle_session_websocket",
    "SessionEventBroadcaster",
    # Export
    "ConversationExporter",
    "export_session_to_file",
    # Types
    "SessionStatus",
    "MessageType",
    "SessionContext",
    "MessageContext",
    "SearchHistoryEntry",
    "CitationEntry",
]

__version__ = "1.0.0"

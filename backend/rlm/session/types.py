"""Type definitions for session management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SessionStatus(str, Enum):
    """Session lifecycle status."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    COMPACTED = "compacted"
    EXPIRED = "expired"


class MessageType(str, Enum):
    """Types of messages in conversation."""

    STANDARD = "standard"
    SUMMARY = "summary"
    COMPACTED = "compacted"
    SYSTEM = "system"
    TOOL = "tool"


class MessageRole(str, Enum):
    """Roles for conversation messages."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class SessionContext:
    """Context information for a session."""

    session_id: str
    title: str
    status: SessionStatus
    created_at: datetime
    last_activity: datetime
    total_tokens_used: int = 0
    context_window_used: int = 0
    message_count: int = 0
    is_compacted: bool = False
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "title": self.title,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "total_tokens_used": self.total_tokens_used,
            "context_window_used": self.context_window_used,
            "message_count": self.message_count,
            "is_compacted": self.is_compacted,
            "custom_metadata": self.custom_metadata,
        }


@dataclass
class MessageContext:
    """A message in conversation context."""

    id: int
    session_id: str
    role: MessageRole
    content: str
    tokens: int
    message_type: MessageType
    parent_message_id: Optional[int] = None
    trajectory_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role.value,
            "content": self.content,
            "tokens": self.tokens,
            "message_type": self.message_type.value,
            "parent_message_id": self.parent_message_id,
            "trajectory_id": self.trajectory_id,
            "created_at": self.created_at.isoformat(),
            "custom_metadata": self.custom_metadata,
        }


@dataclass
class SearchHistoryEntry:
    """Entry in session search history."""

    id: int
    session_id: str
    query: str
    strategy: str
    results_count: int
    execution_time_ms: float
    semantic_weight: float
    keyword_weight: float
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "query": self.query,
            "strategy": self.strategy,
            "results_count": self.results_count,
            "execution_time_ms": self.execution_time_ms,
            "semantic_weight": self.semantic_weight,
            "keyword_weight": self.keyword_weight,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class CitationEntry:
    """Citation entry for a session."""

    id: int
    session_id: str
    message_id: int
    chunk_id: str
    document_id: str
    content_snippet: str
    score: float
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "message_id": self.message_id,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content_snippet": self.content_snippet,
            "score": self.score,
            "created_at": self.created_at.isoformat(),
        }

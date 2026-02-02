"""SQLAlchemy models for session management with FTS5 support."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    event,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from rlm.session.types import MessageRole, MessageType, SessionStatus


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class Session(Base):
    """Session model for conversation management."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default=SessionStatus.ACTIVE.value)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_activity: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.utcnow() + timedelta(days=3)
    )

    # Hierarchy
    parent_session_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("sessions.id"), nullable=True
    )

    # Token tracking
    total_tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    context_window_used: Mapped[int] = mapped_column(Integer, default=0)

    # Component 8 integration - search preferences
    default_search_strategy: Mapped[str] = mapped_column(String(20), default="hybrid")
    semantic_weight: Mapped[float] = mapped_column(Float, default=0.7)
    keyword_weight: Mapped[float] = mapped_column(Float, default=0.3)
    enable_reranking: Mapped[bool] = mapped_column(Boolean, default=True)
    enable_citations: Mapped[bool] = mapped_column(Boolean, default=True)

    # Custom metadata (renamed from 'metadata' to avoid SQLAlchemy reserved name conflict)
    custom_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    # Relationships
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
    summaries: Mapped[List["Summary"]] = relationship(
        "Summary", back_populates="session", cascade="all, delete-orphan"
    )
    search_history: Mapped[List["SearchHistory"]] = relationship(
        "SearchHistory", back_populates="session", cascade="all, delete-orphan"
    )
    citations: Mapped[List["SessionCitation"]] = relationship(
        "SessionCitation", back_populates="session", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Session(id={self.id}, title={self.title}, status={self.status})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "parent_session_id": self.parent_session_id,
            "total_tokens_used": self.total_tokens_used,
            "context_window_used": self.context_window_used,
            "default_search_strategy": self.default_search_strategy,
            "semantic_weight": self.semantic_weight,
            "keyword_weight": self.keyword_weight,
            "enable_reranking": self.enable_reranking,
            "enable_citations": self.enable_citations,
            "custom_metadata": self.custom_metadata,
        }


class Message(Base):
    """Message model for conversation history."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sessions.id"), index=True)

    role: Mapped[str] = mapped_column(String(20))  # user, assistant, system, tool
    content: Mapped[str] = mapped_column(Text)
    tokens: Mapped[int] = mapped_column(Integer, default=0)

    message_type: Mapped[str] = mapped_column(String(20), default=MessageType.STANDARD.value)

    # Threading support
    parent_message_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("messages.id"), nullable=True
    )

    # Link to trajectory logs
    trajectory_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Custom metadata for tool calls, function results, etc. (renamed to avoid SQLAlchemy reserved name)
    custom_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    # Relationships
    session: Mapped["Session"] = relationship("Session", back_populates="messages")
    citations: Mapped[List["SessionCitation"]] = relationship(
        "SessionCitation", back_populates="message", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Message(id={self.id}, session={self.session_id}, role={self.role})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "tokens": self.tokens,
            "message_type": self.message_type,
            "parent_message_id": self.parent_message_id,
            "trajectory_id": self.trajectory_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "custom_metadata": self.custom_metadata,
        }


class Summary(Base):
    """Summary model for conversation compaction records."""

    __tablename__ = "summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sessions.id"), index=True)

    # Message range that was compacted
    start_message_id: Mapped[int] = mapped_column(Integer)
    end_message_id: Mapped[int] = mapped_column(Integer)

    # Summary content
    summary_content: Mapped[str] = mapped_column(Text)
    tokens_saved: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    session: Mapped["Session"] = relationship("Session", back_populates="summaries")

    def __repr__(self) -> str:
        return f"<Summary(id={self.id}, session={self.session_id}, messages={self.start_message_id}-{self.end_message_id})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "start_message_id": self.start_message_id,
            "end_message_id": self.end_message_id,
            "summary_content": self.summary_content,
            "tokens_saved": self.tokens_saved,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SearchHistory(Base):
    """Search history model for Component 8 integration."""

    __tablename__ = "search_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sessions.id"), index=True)

    query: Mapped[str] = mapped_column(Text)
    strategy: Mapped[str] = mapped_column(String(20))  # hybrid, reranked, adaptive

    results_count: Mapped[int] = mapped_column(Integer, default=0)
    execution_time_ms: Mapped[float] = mapped_column(Float, default=0.0)

    # Component 8 parameters
    semantic_weight: Mapped[float] = mapped_column(Float, default=0.7)
    keyword_weight: Mapped[float] = mapped_column(Float, default=0.3)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Additional custom metadata (renamed to avoid SQLAlchemy reserved name)
    custom_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    # Relationships
    session: Mapped["Session"] = relationship("Session", back_populates="search_history")

    def __repr__(self) -> str:
        return f"<SearchHistory(id={self.id}, session={self.session_id}, query={self.query[:50]})>"

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
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "custom_metadata": self.custom_metadata,
        }


class SessionCitation(Base):
    """Citation model for Component 8 integration."""

    __tablename__ = "session_citations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sessions.id"), index=True)
    message_id: Mapped[int] = mapped_column(Integer, ForeignKey("messages.id"), index=True)

    chunk_id: Mapped[str] = mapped_column(String(36))
    document_id: Mapped[str] = mapped_column(String(36))
    content_snippet: Mapped[str] = mapped_column(Text)
    score: Mapped[float] = mapped_column(Float, default=0.0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    session: Mapped["Session"] = relationship("Session", back_populates="citations")
    message: Mapped["Message"] = relationship("Message", back_populates="citations")

    def __repr__(self) -> str:
        return f"<SessionCitation(id={self.id}, message={self.message_id}, chunk={self.chunk_id})>"

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
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# FTS5 Virtual Table Model for full-text search
class MessageFTS(Base):
    """FTS5 virtual table for message content search."""

    __tablename__ = "messages_fts"

    # FTS5 virtual tables don't use standard column definitions
    rowid: Mapped[int] = mapped_column(Integer, primary_key=True)
    content: Mapped[str] = mapped_column(Text)
    session_id: Mapped[str] = mapped_column(String(36))

    def __repr__(self) -> str:
        return f"<MessageFTS(rowid={self.rowid}, session={self.session_id})>"


async def create_fts_tables(session: AsyncSession) -> None:
    """Create FTS5 virtual tables and triggers."""

    # Create FTS5 virtual table
    await session.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            content,
            session_id UNINDEXED,
            content='messages',
            content_rowid='id'
        )
        """
    )

    # Create triggers to keep FTS index in sync
    await session.execute(
        """
        CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, content, session_id)
            VALUES (new.id, new.content, new.session_id);
        END
        """
    )

    await session.execute(
        """
        CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, content, session_id)
            VALUES ('delete', old.id, old.content, old.session_id);
        END
        """
    )

    await session.execute(
        """
        CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, content, session_id)
            VALUES ('delete', old.id, old.content, old.session_id);
            INSERT INTO messages_fts(rowid, content, session_id)
            VALUES (new.id, new.content, new.session_id);
        END
        """
    )

    await session.commit()


async def init_database(session: AsyncSession) -> None:
    """Initialize the database with all tables and FTS5."""
    # Create all tables
    from sqlalchemy import MetaData

    async with session.bind.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Create FTS5 tables
    await create_fts_tables(session)

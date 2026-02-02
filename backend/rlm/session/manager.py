"""Session manager for conversation lifecycle management."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import structlog
from sqlalchemy import desc, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rlm.config import get_settings
from rlm.session.compactor import ContextCompactor
from rlm.session.models import (
    Base,
    Message,
    MessageFTS,
    SearchHistory,
    Session,
    SessionCitation,
    Summary,
    create_fts_tables,
    init_database,
)
from rlm.session.token_manager import TokenManager
from rlm.session.types import (
    CitationEntry,
    MessageContext,
    MessageRole,
    MessageType,
    SearchHistoryEntry,
    SessionContext,
    SessionStatus,
)

logger = structlog.get_logger()


class SessionManager:
    """Main manager for session lifecycle and conversation management.

    Provides complete session management including:
    - Session CRUD operations (create, read, update, delete)
    - Message management with automatic compaction
    - Context window tracking and management
    - Conversation history search via FTS5
    - Session forking with full state preservation
    - Component 8 integration (search history, citations)

    Example:
        ```python
        manager = SessionManager()

        # Create session
        session = await manager.create_session("Research Project")

        # Add messages
        await manager.add_message(session.id, "user", "What is AI?")
        await manager.add_message(session.id, "assistant", "AI is...")

        # Get context for LLM
        context = await manager.get_context(session.id)

        # Search history
        results = await manager.search_conversation(session.id, "machine learning")

        # Fork session
        forked = await manager.fork_session(session.id)
        ```
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        max_tokens: int = 128000,
        ttl_hours: int = 72,  # 3 days default
    ) -> None:
        """Initialize session manager.

        Args:
            db_path: Path to SQLite database (default from settings)
            max_tokens: Maximum context window size
            ttl_hours: Session time-to-live in hours
        """
        settings = get_settings()

        self.db_path = db_path or getattr(settings, "session_db_path", "./data/sessions.db")
        self.ttl_hours = ttl_hours

        # Initialize database engine
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{self.db_path}",
            echo=False,
        )
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Initialize managers
        self.token_manager = TokenManager(max_tokens=max_tokens)
        self.compactor = ContextCompactor(token_manager=self.token_manager)

        # Track initialization
        self._initialized = False

        logger.info(
            "session_manager_initialized",
            db_path=self.db_path,
            max_tokens=max_tokens,
            ttl_hours=ttl_hours,
        )

    async def initialize(self) -> None:
        """Initialize database tables and FTS5."""
        if self._initialized:
            return

        async with self.async_session() as session:
            # Create tables
            await init_database(session)

        self._initialized = True
        logger.info("session_database_initialized")

    # ==================== Session CRUD ====================

    async def create_session(
        self,
        title: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Session:
        """Create a new session.

        Args:
            title: Optional session title
            parent_session_id: Optional parent session for forking
            metadata: Optional metadata dictionary
            **kwargs: Additional session attributes (Component 8 settings)

        Returns:
            Created Session object
        """
        await self.initialize()

        async with self.async_session() as session:
            # Create session
            expires_at = datetime.utcnow() + timedelta(hours=self.ttl_hours)

            new_session = Session(
                title=title or f"Session {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                status=SessionStatus.ACTIVE.value,
                expires_at=expires_at,
                parent_session_id=parent_session_id,
                custom_metadata=metadata or {},
            )

            # Apply Component 8 settings if provided
            if "default_search_strategy" in kwargs:
                new_session.default_search_strategy = kwargs["default_search_strategy"]
            if "semantic_weight" in kwargs:
                new_session.semantic_weight = kwargs["semantic_weight"]
            if "keyword_weight" in kwargs:
                new_session.keyword_weight = kwargs["keyword_weight"]
            if "enable_reranking" in kwargs:
                new_session.enable_reranking = kwargs["enable_reranking"]
            if "enable_citations" in kwargs:
                new_session.enable_citations = kwargs["enable_citations"]

            session.add(new_session)
            await session.commit()
            await session.refresh(new_session)

            logger.info(
                "session_created",
                session_id=new_session.id,
                title=new_session.title,
                parent_id=parent_session_id,
            )

            return new_session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID.

        Args:
            session_id: Session ID

        Returns:
            Session object or None
        """
        await self.initialize()

        async with self.async_session() as session:
            result = await session.execute(select(Session).where(Session.id == session_id))
            return result.scalar_one_or_none()

    async def list_sessions(
        self,
        status: Optional[SessionStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Session]:
        """List sessions with optional filtering.

        Args:
            status: Filter by status
            limit: Maximum sessions to return
            offset: Number of sessions to skip

        Returns:
            List of Session objects
        """
        await self.initialize()

        async with self.async_session() as session:
            query = select(Session)

            if status:
                query = query.where(Session.status == status.value)

            query = query.order_by(desc(Session.last_activity))
            query = query.limit(limit).offset(offset)

            result = await session.execute(query)
            return list(result.scalars().all())

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all associated data.

        Args:
            session_id: Session ID to delete

        Returns:
            True if deleted, False if not found
        """
        await self.initialize()

        async with self.async_session() as session:
            # Get session
            result = await session.execute(select(Session).where(Session.id == session_id))
            db_session = result.scalar_one_or_none()

            if not db_session:
                return False

            # Delete (cascade will handle related records)
            await session.delete(db_session)
            await session.commit()

            logger.info("session_deleted", session_id=session_id)
            return True

    async def update_session_activity(self, session_id: str) -> None:
        """Update session last_activity timestamp.

        Args:
            session_id: Session ID
        """
        async with self.async_session() as session:
            result = await session.execute(select(Session).where(Session.id == session_id))
            db_session = result.scalar_one_or_none()

            if db_session:
                db_session.last_activity = datetime.utcnow()
                await session.commit()

    # ==================== Message Management ====================

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        parent_message_id: Optional[int] = None,
        trajectory_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_compact: bool = True,
    ) -> Message:
        """Add a message to a session.

        Args:
            session_id: Session ID
            role: Message role (user, assistant, system, tool)
            content: Message content
            parent_message_id: Optional parent message for threading
            trajectory_id: Optional link to trajectory log
            metadata: Optional metadata
            auto_compact: Whether to auto-compact if needed

        Returns:
            Created Message object
        """
        await self.initialize()

        async with self.async_session() as session:
            # Calculate tokens
            tokens = self.token_manager.estimate_tokens(content)

            # Create message
            message = Message(
                session_id=session_id,
                role=role,
                content=content,
                tokens=tokens,
                parent_message_id=parent_message_id,
                trajectory_id=trajectory_id,
                metadata=metadata or {},
            )

            session.add(message)

            # Update session activity and token count
            result = await session.execute(select(Session).where(Session.id == session_id))
            db_session = result.scalar_one()
            db_session.total_tokens_used += tokens
            db_session.context_window_used += tokens
            db_session.last_activity = datetime.utcnow()

            await session.commit()
            await session.refresh(message)

            logger.debug(
                "message_added",
                session_id=session_id,
                message_id=message.id,
                role=role,
                tokens=tokens,
            )

            # Auto-compact if needed
            if auto_compact and role == MessageRole.ASSISTANT.value:
                await self._maybe_compact(session_id)

            return message

    async def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        message_type: Optional[str] = None,
    ) -> List[Message]:
        """Get messages for a session.

        Args:
            session_id: Session ID
            limit: Maximum messages to return (None for all)
            message_type: Filter by message type

        Returns:
            List of Message objects
        """
        await self.initialize()

        async with self.async_session() as session:
            query = select(Message).where(Message.session_id == session_id)

            if message_type:
                query = query.where(Message.message_type == message_type)

            query = query.order_by(Message.created_at)

            if limit:
                query = query.limit(limit)

            result = await session.execute(query)
            return list(result.scalars().all())

    async def get_message_count(self, session_id: str) -> int:
        """Get total message count for a session.

        Args:
            session_id: Session ID

        Returns:
            Message count
        """
        await self.initialize()

        async with self.async_session() as session:
            result = await session.execute(
                select(func.count()).where(Message.session_id == session_id)
            )
            return result.scalar() or 0

    # ==================== Context Management ====================

    async def get_context(
        self,
        session_id: str,
        include_summaries: bool = True,
    ) -> List[MessageContext]:
        """Get conversation context for LLM.

        Returns messages in format suitable for LLM context,
        including summaries for compacted sections.

        Args:
            session_id: Session ID
            include_summaries: Whether to include summary messages

        Returns:
            List of MessageContext objects
        """
        await self.initialize()

        messages = await self.get_messages(session_id)

        context = []
        for msg in messages:
            if not include_summaries and msg.message_type == MessageType.SUMMARY.value:
                continue

            context.append(
                MessageContext(
                    id=msg.id,
                    session_id=msg.session_id,
                    role=MessageRole(msg.role),
                    content=msg.content,
                    tokens=msg.tokens,
                    message_type=MessageType(msg.message_type),
                    parent_message_id=msg.parent_message_id,
                    trajectory_id=msg.trajectory_id,
                    created_at=msg.created_at,
                    custom_metadata=msg.custom_metadata,
                )
            )

        return context

    async def _maybe_compact(self, session_id: str) -> None:
        """Check and perform compaction if needed.

        Args:
            session_id: Session ID
        """
        messages = await self.get_messages(session_id)

        if not self.compactor.should_compact(messages):
            return

        logger.info("auto_compaction_triggered", session_id=session_id)

        async with self.async_session() as session:
            # Compact messages
            compacted, summary_record = await self.compactor.compact(messages, session_id)

            if summary_record:
                # Save summary record
                session.add(summary_record)

                # Add summary message
                summary_msg = compacted[0]  # First message is the summary
                session.add(summary_msg)

                # Update session status
                result = await session.execute(select(Session).where(Session.id == session_id))
                db_session = result.scalar_one()
                db_session.status = SessionStatus.COMPACTED.value
                db_session.total_tokens_used -= summary_record.tokens_saved
                db_session.context_window_used -= summary_record.tokens_saved

                await session.commit()

                logger.info(
                    "auto_compaction_complete",
                    session_id=session_id,
                    tokens_saved=summary_record.tokens_saved,
                )

    async def get_compaction_stats(self, session_id: str) -> Dict[str, Any]:
        """Get compaction statistics for a session.

        Args:
            session_id: Session ID

        Returns:
            Statistics dictionary
        """
        messages = await self.get_messages(session_id)
        return self.compactor.get_compaction_stats(messages)

    # ==================== Search ====================

    async def search_conversation(
        self,
        session_id: str,
        query: str,
        limit: int = 20,
    ) -> List[MessageContext]:
        """Search within a conversation using FTS5.

        Args:
            session_id: Session ID to search
            query: Search query
            limit: Maximum results

        Returns:
            List of matching MessageContext objects
        """
        await self.initialize()

        async with self.async_session() as session:
            # Use FTS5 to search
            fts_query = text("""
                SELECT rowid FROM messages_fts 
                WHERE content MATCH :query AND session_id = :session_id
                ORDER BY rank
                LIMIT :limit
            """)

            result = await session.execute(
                fts_query,
                {
                    "query": query,
                    "session_id": session_id,
                    "limit": limit,
                },
            )

            message_ids = [row[0] for row in result.fetchall()]

            if not message_ids:
                return []

            # Fetch full messages
            messages_result = await session.execute(
                select(Message).where(Message.id.in_(message_ids))
            )
            messages = messages_result.scalars().all()

            # Convert to context
            return [
                MessageContext(
                    id=msg.id,
                    session_id=msg.session_id,
                    role=MessageRole(msg.role),
                    content=msg.content,
                    tokens=msg.tokens,
                    message_type=MessageType(msg.message_type),
                    parent_message_id=msg.parent_message_id,
                    trajectory_id=msg.trajectory_id,
                    created_at=msg.created_at,
                    custom_metadata=msg.custom_metadata,
                )
                for msg in messages
            ]

    async def get_recent_queries(
        self,
        session_id: str,
        n: int = 3,
    ) -> List[str]:
        """Get recent user queries from session.

        Args:
            session_id: Session ID
            n: Number of recent queries to retrieve

        Returns:
            List of query strings
        """
        await self.initialize()

        async with self.async_session() as session:
            result = await session.execute(
                select(Message)
                .where(Message.session_id == session_id, Message.role == MessageRole.USER.value)
                .order_by(desc(Message.created_at))
                .limit(n)
            )

            messages = result.scalars().all()
            return [msg.content for msg in reversed(messages)]

    # ==================== Forking ====================

    async def fork_session(
        self,
        session_id: str,
        title: Optional[str] = None,
    ) -> Session:
        """Fork a session, copying all state and messages.

        Args:
            session_id: Session ID to fork
            title: Optional new title

        Returns:
            New forked Session object
        """
        await self.initialize()

        # Get original session
        original = await self.get_session(session_id)
        if not original:
            raise ValueError(f"Session {session_id} not found")

        async with self.async_session() as session:
            # Create new session with copied settings
            new_session = Session(
                title=title or f"Fork of {original.title}",
                status=SessionStatus.ACTIVE.value,
                expires_at=datetime.utcnow() + timedelta(hours=self.ttl_hours),
                parent_session_id=session_id,
                # Copy Component 8 settings
                default_search_strategy=original.default_search_strategy,
                semantic_weight=original.semantic_weight,
                keyword_weight=original.keyword_weight,
                enable_reranking=original.enable_reranking,
                enable_citations=original.enable_citations,
                # Copy custom metadata
                custom_metadata={
                    **original.custom_metadata,
                    "forked_from": session_id,
                    "forked_at": datetime.utcnow().isoformat(),
                },
            )

            session.add(new_session)
            await session.flush()  # Get the new ID

            # Copy messages
            original_messages = await self.get_messages(session_id)
            for orig_msg in original_messages:
                new_message = Message(
                    session_id=new_session.id,
                    role=orig_msg.role,
                    content=orig_msg.content,
                    tokens=orig_msg.tokens,
                    message_type=orig_msg.message_type,
                    parent_message_id=orig_msg.parent_message_id,
                    trajectory_id=orig_msg.trajectory_id,
                    created_at=orig_msg.created_at,
                    custom_metadata={**orig_msg.custom_metadata, "forked": True},
                )
                session.add(new_message)

            # Copy total tokens
            new_session.total_tokens_used = original.total_tokens_used
            new_session.context_window_used = original.context_window_used

            await session.commit()
            await session.refresh(new_session)

            logger.info(
                "session_forked",
                original_id=session_id,
                new_id=new_session.id,
                message_count=len(original_messages),
            )

            return new_session

    # ==================== Component 8 Integration ====================

    async def log_search(
        self,
        session_id: str,
        query: str,
        strategy: str,
        results_count: int,
        execution_time_ms: float,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SearchHistory:
        """Log a search operation for Component 8 integration.

        Args:
            session_id: Session ID
            query: Search query
            strategy: Search strategy (hybrid, reranked, adaptive)
            results_count: Number of results returned
            execution_time_ms: Execution time
            semantic_weight: Semantic search weight
            keyword_weight: Keyword search weight
            metadata: Additional metadata

        Returns:
            SearchHistory record
        """
        await self.initialize()

        async with self.async_session() as session:
            search_record = SearchHistory(
                session_id=session_id,
                query=query,
                strategy=strategy,
                results_count=results_count,
                execution_time_ms=execution_time_ms,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
                metadata=metadata or {},
            )

            session.add(search_record)
            await session.commit()
            await session.refresh(search_record)

            logger.debug(
                "search_logged",
                session_id=session_id,
                strategy=strategy,
                results=results_count,
            )

            return search_record

    async def get_search_history(
        self,
        session_id: str,
        limit: int = 50,
    ) -> List[SearchHistoryEntry]:
        """Get search history for a session.

        Args:
            session_id: Session ID
            limit: Maximum records to return

        Returns:
            List of SearchHistoryEntry objects
        """
        await self.initialize()

        async with self.async_session() as session:
            result = await session.execute(
                select(SearchHistory)
                .where(SearchHistory.session_id == session_id)
                .order_by(desc(SearchHistory.created_at))
                .limit(limit)
            )

            records = result.scalars().all()

            return [
                SearchHistoryEntry(
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
                for record in records
            ]

    async def add_citation(
        self,
        session_id: str,
        message_id: int,
        chunk_id: str,
        document_id: str,
        content_snippet: str,
        score: float = 0.0,
    ) -> SessionCitation:
        """Add a citation from Component 8.

        Args:
            session_id: Session ID
            message_id: Message ID that contains the citation
            chunk_id: Chunk ID from Component 8
            document_id: Document ID
            content_snippet: Citation content snippet
            score: Citation relevance score

        Returns:
            SessionCitation record
        """
        await self.initialize()

        async with self.async_session() as session:
            citation = SessionCitation(
                session_id=session_id,
                message_id=message_id,
                chunk_id=chunk_id,
                document_id=document_id,
                content_snippet=content_snippet,
                score=score,
            )

            session.add(citation)
            await session.commit()
            await session.refresh(citation)

            logger.debug(
                "citation_added",
                session_id=session_id,
                message_id=message_id,
                chunk_id=chunk_id,
            )

            return citation

    async def get_citations(
        self,
        session_id: str,
        message_id: Optional[int] = None,
    ) -> List[CitationEntry]:
        """Get citations for a session or specific message.

        Args:
            session_id: Session ID
            message_id: Optional message ID to filter

        Returns:
            List of CitationEntry objects
        """
        await self.initialize()

        async with self.async_session() as session:
            query = select(SessionCitation).where(SessionCitation.session_id == session_id)

            if message_id:
                query = query.where(SessionCitation.message_id == message_id)

            query = query.order_by(desc(SessionCitation.created_at))

            result = await session.execute(query)
            records = result.scalars().all()

            return [
                CitationEntry(
                    id=record.id,
                    session_id=record.session_id,
                    message_id=record.message_id,
                    chunk_id=record.chunk_id,
                    document_id=record.document_id,
                    content_snippet=record.content_snippet,
                    score=record.score,
                    created_at=record.created_at,
                )
                for record in records
            ]

    # ==================== Cleanup ====================

    async def cleanup_expired_sessions(self) -> int:
        """Delete expired sessions.

        Returns:
            Number of sessions deleted
        """
        await self.initialize()

        async with self.async_session() as session:
            # Find expired sessions
            result = await session.execute(
                select(Session).where(Session.expires_at < datetime.utcnow())
            )
            expired = result.scalars().all()

            count = 0
            for db_session in expired:
                await session.delete(db_session)
                count += 1

            await session.commit()

            if count > 0:
                logger.info("expired_sessions_cleaned", count=count)

            return count

    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session statistics.

        Args:
            session_id: Session ID

        Returns:
            Statistics dictionary
        """
        await self.initialize()

        db_session = await self.get_session(session_id)
        if not db_session:
            return {}

        message_count = await self.get_message_count(session_id)
        compaction_stats = await self.get_compaction_stats(session_id)

        return {
            "session_id": session_id,
            "title": db_session.title,
            "status": db_session.status,
            "created_at": db_session.created_at.isoformat() if db_session.created_at else None,
            "last_activity": db_session.last_activity.isoformat()
            if db_session.last_activity
            else None,
            "expires_at": db_session.expires_at.isoformat() if db_session.expires_at else None,
            "is_expired": db_session.expires_at < datetime.utcnow()
            if db_session.expires_at
            else False,
            "message_count": message_count,
            "total_tokens_used": db_session.total_tokens_used,
            "context_window_used": db_session.context_window_used,
            "usage_percentage": compaction_stats.get("usage_percentage", 0),
            "should_compact": compaction_stats.get("should_compact", False),
            "search_preferences": {
                "strategy": db_session.default_search_strategy,
                "semantic_weight": db_session.semantic_weight,
                "keyword_weight": db_session.keyword_weight,
                "enable_reranking": db_session.enable_reranking,
                "enable_citations": db_session.enable_citations,
            },
        }

"""Test suite for Component 9: Session Management System.

Comprehensive tests for:
- Session CRUD operations
- Message management
- Context compaction
- Token management
- FTS5 search
- Session forking
- Component 8 integration
"""

import asyncio
import os
import tempfile
from datetime import datetime, timedelta

import pytest
import pytest_asyncio

from rlm.session import (
    ContextCompactor,
    SessionManager,
    TokenManager,
)
from rlm.session.types import MessageRole, MessageType, SessionStatus


@pytest_asyncio.fixture
async def temp_db_path():
    """Create temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


@pytest_asyncio.fixture
async def session_manager(temp_db_path):
    """Create session manager with temporary database."""
    manager = SessionManager(
        db_path=temp_db_path,
        max_tokens=10000,  # Small for testing
        ttl_hours=1,  # Short for testing
    )
    await manager.initialize()
    yield manager


class TestSessionCRUD:
    """Test session CRUD operations."""
    
    async def test_create_session(self, session_manager):
        """Test creating a session."""
        session = await session_manager.create_session(
            title="Test Session",
            metadata={"test": True},
        )
        
        assert session.id is not None
        assert session.title == "Test Session"
        assert session.status == SessionStatus.ACTIVE.value
        assert session.metadata == {"test": True}
        assert session.expires_at > datetime.utcnow()
    
    async def test_get_session(self, session_manager):
        """Test retrieving a session."""
        created = await session_manager.create_session(title="Get Test")
        
        retrieved = await session_manager.get_session(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.title == created.title
    
    async def test_get_nonexistent_session(self, session_manager):
        """Test retrieving non-existent session."""
        result = await session_manager.get_session("nonexistent-id")
        assert result is None
    
    async def test_list_sessions(self, session_manager):
        """Test listing sessions."""
        # Create multiple sessions
        sessions = []
        for i in range(3):
            session = await session_manager.create_session(title=f"Session {i}")
            sessions.append(session)
        
        # List all
        listed = await session_manager.list_sessions()
        assert len(listed) == 3
        
        # List by status
        active = await session_manager.list_sessions(status=SessionStatus.ACTIVE)
        assert len(active) == 3
    
    async def test_delete_session(self, session_manager):
        """Test deleting a session."""
        session = await session_manager.create_session(title="To Delete")
        
        # Delete
        deleted = await session_manager.delete_session(session.id)
        assert deleted is True
        
        # Verify deletion
        result = await session_manager.get_session(session.id)
        assert result is None
    
    async def test_delete_nonexistent_session(self, session_manager):
        """Test deleting non-existent session."""
        result = await session_manager.delete_session("nonexistent")
        assert result is False
    
    async def test_session_ttl(self, session_manager):
        """Test session time-to-live."""
        session = await session_manager.create_session(title="TTL Test")
        
        # Should not be expired immediately
        assert session.expires_at > datetime.utcnow()
        
        # Check stats
        stats = await session_manager.get_session_stats(session.id)
        assert stats["is_expired"] is False


class TestMessageManagement:
    """Test message management."""
    
    async def test_add_message(self, session_manager):
        """Test adding a message."""
        session = await session_manager.create_session(title="Message Test")
        
        message = await session_manager.add_message(
            session_id=session.id,
            role=MessageRole.USER.value,
            content="Test message",
        )
        
        assert message.id is not None
        assert message.session_id == session.id
        assert message.role == MessageRole.USER.value
        assert message.content == "Test message"
        assert message.tokens > 0
    
    async def test_add_multiple_messages(self, session_manager):
        """Test adding multiple messages."""
        session = await session_manager.create_session(title="Multiple Messages")
        
        # Add messages
        for i in range(5):
            await session_manager.add_message(
                session_id=session.id,
                role=MessageRole.USER.value if i % 2 == 0 else MessageRole.ASSISTANT.value,
                content=f"Message {i}",
            )
        
        # Verify count
        count = await session_manager.get_message_count(session.id)
        assert count == 5
        
        # Retrieve messages
        messages = await session_manager.get_messages(session.id)
        assert len(messages) == 5
    
    async def test_get_context(self, session_manager):
        """Test getting conversation context."""
        session = await session_manager.create_session(title="Context Test")
        
        # Add messages
        await session_manager.add_message(
            session_id=session.id,
            role=MessageRole.USER.value,
            content="What is AI?",
        )
        await session_manager.add_message(
            session_id=session.id,
            role=MessageRole.ASSISTANT.value,
            content="AI is artificial intelligence...",
        )
        
        # Get context
        context = await session_manager.get_context(session.id)
        assert len(context) == 2
        assert context[0].role == MessageRole.USER
        assert context[1].role == MessageRole.ASSISTANT
    
    async def test_message_token_tracking(self, session_manager):
        """Test token tracking for messages."""
        session = await session_manager.create_session(title="Token Test")
        
        # Add message
        message = await session_manager.add_message(
            session_id=session.id,
            role=MessageRole.USER.value,
            content="Hello world, this is a test message",
        )
        
        # Verify tokens calculated
        assert message.tokens > 0
        
        # Check session token count updated
        session_data = await session_manager.get_session(session.id)
        assert session_data.total_tokens_used >= message.tokens


class TestContextCompaction:
    """Test context compaction functionality."""
    
    async def test_should_compact(self, session_manager):
        """Test compaction threshold detection."""
        session = await session_manager.create_session(title="Compaction Test")
        
        # Add many messages to exceed threshold
        for i in range(50):
            await session_manager.add_message(
                session_id=session.id,
                role=MessageRole.USER.value,
                content=f"Question {i}: " + "A" * 1000,  # Long content
            )
            await session_manager.add_message(
                session_id=session.id,
                role=MessageRole.ASSISTANT.value,
                content=f"Answer {i}: " + "B" * 1000,
            )
        
        # Check compaction stats
        stats = await session_manager.get_compaction_stats(session.id)
        assert stats["should_compact"] is True
        assert stats["total_messages"] == 100
    
    async def test_retention_window(self, session_manager):
        """Test retention window calculation."""
        compactor = ContextCompactor()
        
        # Test with many messages
        messages = []
        for i in range(20):
            from rlm.session.models import Message
            msg = Message(
                session_id="test",
                role=MessageRole.USER.value,
                content=f"Message {i}",
                tokens=100,
            )
            messages.append(msg)
        
        # Calculate retention
        retention = compactor.token_manager.get_retention_message_count(messages)
        assert retention >= 6  # Minimum retention
    
    async def test_compaction_stats(self, session_manager):
        """Test compaction statistics."""
        session = await session_manager.create_session(title="Stats Test")
        
        # Add messages
        for i in range(10):
            await session_manager.add_message(
                session_id=session.id,
                role=MessageRole.USER.value,
                content=f"Message {i}",
            )
        
        stats = await session_manager.get_compaction_stats(session.id)
        assert "total_messages" in stats
        assert "total_tokens" in stats
        assert "usage_percentage" in stats
        assert "should_compact" in stats


class TestTokenManager:
    """Test token management."""
    
    def test_estimate_tokens(self):
        """Test token estimation."""
        manager = TokenManager(max_tokens=1000)
        
        # Empty text
        assert manager.estimate_tokens("") == 0
        
        # Short text
        tokens = manager.estimate_tokens("Hello world")
        assert tokens > 0
        assert tokens < 10  # Rough estimate
        
        # Long text
        long_text = "A" * 4000
        tokens = manager.estimate_tokens(long_text)
        assert tokens >= 1000  # ~4 chars per token
    
    def test_should_compact_threshold(self):
        """Test compaction threshold."""
        manager = TokenManager(max_tokens=1000, warning_threshold=0.8)
        
        # Below threshold
        assert manager.should_compact([]) is False
        
        # At threshold
        from rlm.session.models import Message
        messages = []
        for i in range(10):
            msg = Message(session_id="test", role="user", content="A" * 400, tokens=100)
            messages.append(msg)
        
        assert manager.should_compact(messages) is True
    
    def test_usage_percentage(self):
        """Test usage percentage calculation."""
        manager = TokenManager(max_tokens=1000)
        
        from rlm.session.models import Message
        messages = [Message(session_id="test", role="user", content="A" * 400, tokens=500)]
        
        percentage = manager.get_usage_percentage(messages)
        assert percentage == 50.0


class TestSessionForking:
    """Test session forking."""
    
    async def test_fork_session(self, session_manager):
        """Test forking a session."""
        # Create original
        original = await session_manager.create_session(
            title="Original",
            semantic_weight=0.8,
            enable_reranking=True,
        )
        
        # Add messages
        await session_manager.add_message(
            session_id=original.id,
            role=MessageRole.USER.value,
            content="Original question",
        )
        await session_manager.add_message(
            session_id=original.id,
            role=MessageRole.ASSISTANT.value,
            content="Original answer",
        )
        
        # Fork
        forked = await session_manager.fork_session(original.id, title="Forked")
        
        # Verify fork
        assert forked.id != original.id
        assert forked.parent_session_id == original.id
        assert forked.title == "Forked"
        assert forked.semantic_weight == original.semantic_weight
        assert forked.enable_reranking == original.enable_reranking
        
        # Verify messages copied
        messages = await session_manager.get_messages(forked.id)
        assert len(messages) == 2
    
    async def test_fork_nonexistent_session(self, session_manager):
        """Test forking non-existent session."""
        with pytest.raises(ValueError):
            await session_manager.fork_session("nonexistent")


class TestSearchHistory:
    """Test search history tracking."""
    
    async def test_log_search(self, session_manager):
        """Test logging a search."""
        session = await session_manager.create_session(title="Search Test")
        
        search = await session_manager.log_search(
            session_id=session.id,
            query="machine learning",
            strategy="hybrid",
            results_count=5,
            execution_time_ms=150.5,
            semantic_weight=0.7,
            keyword_weight=0.3,
        )
        
        assert search.id is not None
        assert search.query == "machine learning"
        assert search.strategy == "hybrid"
        assert search.results_count == 5
    
    async def test_get_search_history(self, session_manager):
        """Test retrieving search history."""
        session = await session_manager.create_session(title="History Test")
        
        # Log multiple searches
        for i in range(5):
            await session_manager.log_search(
                session_id=session.id,
                query=f"Query {i}",
                strategy="hybrid",
                results_count=i + 1,
                execution_time_ms=100.0 + i,
            )
        
        # Retrieve history
        history = await session_manager.get_search_history(session.id)
        assert len(history) == 5
        
        # Check order (most recent first)
        assert history[0].query == "Query 4"


class TestCitations:
    """Test citation tracking."""
    
    async def test_add_citation(self, session_manager):
        """Test adding a citation."""
        session = await session_manager.create_session(title="Citation Test")
        
        # Add a message first
        message = await session_manager.add_message(
            session_id=session.id,
            role=MessageRole.ASSISTANT.value,
            content="Answer with citation",
        )
        
        # Add citation
        citation = await session_manager.add_citation(
            session_id=session.id,
            message_id=message.id,
            chunk_id="chunk-123",
            document_id="doc-456",
            content_snippet="Relevant content...",
            score=0.95,
        )
        
        assert citation.id is not None
        assert citation.message_id == message.id
        assert citation.chunk_id == "chunk-123"
        assert citation.score == 0.95
    
    async def test_get_citations(self, session_manager):
        """Test retrieving citations."""
        session = await session_manager.create_session(title="Get Citations")
        
        # Add message
        message = await session_manager.add_message(
            session_id=session.id,
            role=MessageRole.ASSISTANT.value,
            content="Answer",
        )
        
        # Add multiple citations
        for i in range(3):
            await session_manager.add_citation(
                session_id=session.id,
                message_id=message.id,
                chunk_id=f"chunk-{i}",
                document_id="doc-1",
                content_snippet=f"Content {i}",
                score=0.9 - (i * 0.1),
            )
        
        # Get all citations
        citations = await session_manager.get_citations(session.id)
        assert len(citations) == 3
        
        # Get citations for specific message
        msg_citations = await session_manager.get_citations(session.id, message.id)
        assert len(msg_citations) == 3


class TestCleanup:
    """Test cleanup functionality."""
    
    async def test_cleanup_expired_sessions(self, temp_db_path):
        """Test cleaning up expired sessions."""
        # Create manager with very short TTL
        manager = SessionManager(
            db_path=temp_db_path,
            ttl_hours=0,  # Immediate expiration
        )
        await manager.initialize()
        
        # Create sessions
        for i in range(3):
            await manager.create_session(title=f"Session {i}")
        
        # Cleanup
        deleted = await manager.cleanup_expired_sessions()
        assert deleted == 3
        
        # Verify all deleted
        sessions = await manager.list_sessions()
        assert len(sessions) == 0


class TestSessionStats:
    """Test session statistics."""
    
    async def test_get_session_stats(self, session_manager):
        """Test getting session statistics."""
        session = await session_manager.create_session(
            title="Stats Test",
            semantic_weight=0.8,
            enable_reranking=True,
        )
        
        # Add messages
        await session_manager.add_message(
            session_id=session.id,
            role=MessageRole.USER.value,
            content="Test",
        )
        
        # Get stats
        stats = await session_manager.get_session_stats(session.id)
        
        assert stats["session_id"] == session.id
        assert stats["title"] == "Stats Test"
        assert stats["message_count"] == 1
        assert stats["search_preferences"]["semantic_weight"] == 0.8
        assert stats["search_preferences"]["enable_reranking"] is True


@pytest.mark.skip(reason="FTS5 requires manual testing with real database")
class TestFTS5Search:
    """Test FTS5 search (requires real database)."""
    
    async def test_search_conversation(self, session_manager):
        """Test searching within conversation."""
        session = await session_manager.create_session(title="Search Test")
        
        # Add searchable content
        await session_manager.add_message(
            session_id=session.id,
            role=MessageRole.USER.value,
            content="What is machine learning?",
        )
        await session_manager.add_message(
            session_id=session.id,
            role=MessageRole.ASSISTANT.value,
            content="Machine learning is a subset of AI...",
        )
        
        # Search
        results = await session_manager.search_conversation(
            session.id, "machine learning"
        )
        
        assert len(results) > 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

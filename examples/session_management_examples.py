"""Example usage of Component 9: Session Management System.

This example demonstrates all key features of the Session Management System:
- Creating and managing sessions
- Adding messages with automatic compaction
- Session-aware search with Component 8 integration
- Citation tracking
- Session forking
- Conversation history search
"""

import asyncio

from rlm.session import SessionManager, TokenManager, ContextCompactor
from rlm.session.integration import (
    SessionSearchEnhancer,
    CitationTracker,
    SearchHistoryLogger,
)
from rlm.session.types import MessageRole


async def example_1_basic_session_management():
    """Example 1: Basic session creation and message management."""
    print("\n=== Example 1: Basic Session Management ===\n")
    
    # Initialize session manager
    manager = SessionManager(
        db_path="./data/example_sessions.db",
        max_tokens=128000,
        ttl_hours=72,
    )
    
    # Create a new session
    session = await manager.create_session(
        title="My Research Session",
        metadata={"project": "AI Research", "tags": ["ml", "nlp"]},
    )
    print(f"Created session: {session.id}")
    print(f"Title: {session.title}")
    print(f"Expires at: {session.expires_at}\n")
    
    # Add messages to the session
    msg1 = await manager.add_message(
        session_id=session.id,
        role=MessageRole.USER.value,
        content="What is machine learning?",
    )
    print(f"Added user message (ID: {msg1.id})")
    
    msg2 = await manager.add_message(
        session_id=session.id,
        role=MessageRole.ASSISTANT.value,
        content="Machine learning is a subset of AI that enables systems to learn from data...",
    )
    print(f"Added assistant message (ID: {msg2.id}, tokens: {msg2.tokens})\n")
    
    # Get all messages
    messages = await manager.get_messages(session.id)
    print(f"Total messages: {len(messages)}")
    for msg in messages:
        print(f"  [{msg.role}]: {msg.content[:50]}...")
    
    # Get conversation context for LLM
    context = await manager.get_context(session.id)
    print(f"\nContext ready for LLM: {len(context)} messages")
    
    # Get session stats
    stats = await manager.get_session_stats(session.id)
    print(f"\nSession stats:")
    print(f"  Tokens used: {stats['total_tokens_used']}")
    print(f"  Usage: {stats['usage_percentage']:.1f}%")
    
    return session.id


async def example_2_context_compaction():
    """Example 2: Automatic context compaction."""
    print("\n=== Example 2: Context Compaction ===\n")
    
    manager = SessionManager()
    
    # Create session with small token limit to trigger compaction
    session = await manager.create_session(title="Compaction Demo")
    
    # Add many messages to trigger compaction
    print("Adding messages...")
    for i in range(20):
        await manager.add_message(
            session_id=session.id,
            role=MessageRole.USER.value,
            content=f"Question {i+1}: What about this topic?",
        )
        await manager.add_message(
            session_id=session.id,
            role=MessageRole.ASSISTANT.value,
            content=f"Answer {i+1}: " + "This is a detailed response. " * 50,
        )
    
    # Check compaction stats
    stats = await manager.get_compaction_stats(session.id)
    print(f"\nCompaction stats:")
    print(f"  Total messages: {stats['total_messages']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Usage: {stats['usage_percentage']:.1f}%")
    print(f"  Should compact: {stats['should_compact']}")
    
    # Get context (will include summary if compacted)
    context = await manager.get_context(session.id)
    summary_count = sum(1 for msg in context if msg.message_type.value == "summary")
    print(f"\nContext includes {summary_count} summary messages")
    
    return session.id


async def example_3_session_search():
    """Example 3: Searching within conversation history."""
    print("\n=== Example 3: Conversation Search ===\n")
    
    manager = SessionManager()
    
    # Create session with searchable content
    session = await manager.create_session(title="Search Demo")
    
    # Add messages with specific keywords
    keywords = ["machine learning", "deep learning", "neural networks", "AI"]
    for i, keyword in enumerate(keywords):
        await manager.add_message(
            session_id=session.id,
            role=MessageRole.USER.value,
            content=f"Tell me about {keyword}",
        )
        await manager.add_message(
            session_id=session.id,
            role=MessageRole.ASSISTANT.value,
            content=f"{keyword.capitalize()} is an important concept in AI...",
        )
    
    # Search conversation
    print("Searching for 'neural'...")
    results = await manager.search_conversation(session.id, "neural")
    print(f"Found {len(results)} matches:")
    for msg in results:
        print(f"  [{msg.role.value}]: {msg.content[:60]}...")
    
    return session.id


async def example_4_session_forking():
    """Example 4: Forking a session."""
    print("\n=== Example 4: Session Forking ===\n")
    
    manager = SessionManager()
    
    # Create original session
    original = await manager.create_session(
        title="Original Session",
        default_search_strategy="hybrid",
        semantic_weight=0.8,
        enable_reranking=True,
    )
    
    # Add messages
    await manager.add_message(
        session_id=original.id,
        role=MessageRole.USER.value,
        content="What is the first topic?",
    )
    await manager.add_message(
        session_id=original.id,
        role=MessageRole.ASSISTANT.value,
        content="The first topic is...",
    )
    
    # Fork the session
    print(f"Original session: {original.id}")
    forked = await manager.fork_session(original.id, title="Forked Session")
    print(f"Forked session: {forked.id}")
    print(f"Parent ID: {forked.parent_session_id}\n")
    
    # Verify settings were copied
    print("Forked session settings:")
    print(f"  Strategy: {forked.default_search_strategy}")
    print(f"  Semantic weight: {forked.semantic_weight}")
    print(f"  Enable reranking: {forked.enable_reranking}")
    
    # Verify messages were copied
    messages = await manager.get_messages(forked.id)
    print(f"\nForked session has {len(messages)} messages")
    
    return original.id, forked.id


async def example_5_component_8_integration():
    """Example 5: Integration with Component 8 (Hybrid RAG)."""
    print("\n=== Example 5: Component 8 Integration ===\n")
    
    # This example shows how Component 9 integrates with Component 8
    # Note: Requires Component 8 to be fully set up
    
    try:
        from rlm.hybrid import HybridSearcher
        from rlm.hybrid import Citation
        
        manager = SessionManager()
        
        # Create session with Component 8 preferences
        session = await manager.create_session(
            title="Hybrid Search Session",
            default_search_strategy="hybrid",
            semantic_weight=0.7,
            keyword_weight=0.3,
            enable_reranking=True,
            enable_citations=True,
        )
        
        # Initialize Component 8 integrations
        search_enhancer = SessionSearchEnhancer(manager)
        citation_tracker = CitationTracker(manager)
        history_logger = SearchHistoryLogger(manager)
        
        print(f"Session created: {session.id}")
        print("Component 8 integrations initialized")
        
        # Simulate a search (would work with actual documents in production)
        print("\nSimulating search flow:")
        print("1. User asks: 'What is machine learning?'")
        print("2. System searches with session context enhancement")
        print("3. Results are returned with citations")
        print("4. Search is logged to session history")
        
        # Log a search manually (in real use, this happens automatically)
        await manager.log_search(
            session_id=session.id,
            query="What is machine learning?",
            strategy="hybrid",
            results_count=5,
            execution_time_ms=150.5,
            semantic_weight=0.7,
            keyword_weight=0.3,
        )
        
        # Get search history
        history = await manager.get_search_history(session.id)
        print(f"\nSearch history: {len(history)} searches logged")
        for entry in history:
            print(f"  - {entry.query} ({entry.strategy})")
        
        return session.id
        
    except ImportError:
        print("Component 8 not available - skipping integration demo")
        return None


async def example_6_session_cleanup():
    """Example 6: Session cleanup and management."""
    print("\n=== Example 6: Session Cleanup ===\n")
    
    manager = SessionManager(ttl_hours=1)  # Short TTL for demo
    
    # Create some sessions
    sessions = []
    for i in range(3):
        session = await manager.create_session(title=f"Session {i+1}")
        sessions.append(session.id)
        print(f"Created: {session.id}")
    
    # List all sessions
    all_sessions = await manager.list_sessions()
    print(f"\nTotal sessions: {len(all_sessions)}")
    
    # Delete one session
    deleted = await manager.delete_session(sessions[0])
    print(f"\nDeleted session {sessions[0]}: {deleted}")
    
    # List again
    remaining = await manager.list_sessions()
    print(f"Remaining sessions: {len(remaining)}")
    
    return sessions


async def main():
    """Run all examples."""
    print("=" * 70)
    print("Component 9: Session Management System - Examples")
    print("=" * 70)
    
    try:
        # Run examples
        await example_1_basic_session_management()
        await example_2_context_compaction()
        await example_3_session_search()
        await example_4_session_forking()
        await example_5_component_8_integration()
        await example_6_session_cleanup()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

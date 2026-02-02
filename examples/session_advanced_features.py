"""Advanced examples for Component 9: Session Management System.

Demonstrates:
- Background cleanup jobs
- WebSocket real-time updates
- Export functionality
- Cleanup scheduling
"""

import asyncio
import tempfile

from rlm.session import (
    SessionManager,
    SessionCleanupJob,
    CleanupScheduler,
    start_background_cleanup,
    run_session_cleanup_now,
)
from rlm.session.websocket import SessionWebSocketManager, SessionEventBroadcaster
from rlm.session.export import ConversationExporter, export_session_to_file
from rlm.session.types import MessageRole


async def example_cleanup_job():
    """Example: Background cleanup job."""
    print("\n=== Example: Background Cleanup Job ===\n")
    
    # Create session manager with very short TTL for demo
    manager = SessionManager(ttl_hours=0)  # Immediate expiration
    
    # Create some sessions
    print("Creating sessions...")
    for i in range(5):
        session = await manager.create_session(title=f"Session {i+1}")
        print(f"  Created: {session.id[:8]}...")
    
    # List all sessions
    sessions = await manager.list_sessions()
    print(f"\nTotal sessions: {len(sessions)}")
    
    # Run cleanup manually
    print("\nRunning cleanup...")
    deleted = await run_session_cleanup_now(manager)
    print(f"Deleted {deleted} expired sessions")
    
    # Verify
    remaining = await manager.list_sessions()
    print(f"Remaining sessions: {len(remaining)}")


async def example_background_cleanup():
    """Example: Start background cleanup."""
    print("\n=== Example: Background Cleanup ===\n")
    
    manager = SessionManager()
    
    # Define callback
    def on_cleanup(count):
        print(f"Cleanup callback: {count} sessions deleted")
    
    # Start background cleanup (runs every hour by default)
    print("Starting background cleanup...")
    job = await start_background_cleanup(
        session_manager=manager,
        on_cleanup=on_cleanup,
    )
    
    print(f"Cleanup job running: {job.get_stats()['is_running']}")
    print(f"Interval: {job.interval_hours} hours")
    
    # Let it run for a moment, then stop
    await asyncio.sleep(2)
    
    print("\nStopping cleanup job...")
    await job.stop()
    print(f"Cleanup job running: {job.get_stats()['is_running']}")


async def example_cleanup_scheduler():
    """Example: Cleanup scheduler with multiple jobs."""
    print("\n=== Example: Cleanup Scheduler ===\n")
    
    scheduler = CleanupScheduler()
    
    # Create cleanup jobs
    session_cleanup = SessionCleanupJob(
        interval_hours=24,
    )
    
    # Add to scheduler
    scheduler.add_job("sessions", session_cleanup)
    
    print("Added jobs to scheduler:")
    print(f"  - sessions: {session_cleanup.interval_hours}h interval")
    
    # Get stats
    stats = scheduler.get_all_stats()
    print(f"\nScheduler stats: {stats}")
    
    # Start all
    print("\nStarting all jobs...")
    await scheduler.start_all()
    
    # Check running status
    print(f"Running: {session_cleanup.get_stats()['is_running']}")
    
    # Stop all
    await asyncio.sleep(1)
    print("\nStopping all jobs...")
    await scheduler.stop_all()
    print(f"Running: {session_cleanup.get_stats()['is_running']}")


async def example_export_formats():
    """Example: Export conversations in different formats."""
    print("\n=== Example: Export Functionality ===\n")
    
    # Create session with content
    manager = SessionManager()
    session = await manager.create_session(title="Export Demo Session")
    
    # Add messages
    print("Adding messages...")
    await manager.add_message(
        session_id=session.id,
        role=MessageRole.USER.value,
        content="What is machine learning?",
    )
    await manager.add_message(
        session_id=session.id,
        role=MessageRole.ASSISTANT.value,
        content="Machine learning is a subset of AI that enables systems to learn...",
    )
    
    # Add search history
    await manager.log_search(
        session_id=session.id,
        query="machine learning",
        strategy="hybrid",
        results_count=5,
        execution_time_ms=150.5,
    )
    
    # Create exporter
    exporter = ConversationExporter(manager)
    
    # Get export summary
    summary = await exporter.get_export_summary(session.id)
    print(f"\nExport Summary:")
    print(f"  Session: {summary['title']}")
    print(f"  Messages: {summary['message_count']}")
    print(f"  Search history: {summary['search_history_count']}")
    print(f"  Formats: {', '.join(summary['available_formats'])}")
    
    # Export to different formats
    formats = ["json", "csv", "markdown", "txt"]
    
    print("\nExporting to different formats:")
    for fmt in formats:
        try:
            data = await exporter.export_session(
                session_id=session.id,
                format=fmt,
                include_search_history=True,
                include_citations=True,
            )
            # Show preview
            preview = data[:150] if len(data) > 150 else data
            preview = preview.replace('\n', ' ')
            print(f"\n  {fmt.upper()}:")
            print(f"    {preview}...")
            print(f"    (total length: {len(data)} chars)")
        except Exception as e:
            print(f"  {fmt.upper()}: Error - {e}")
    
    # Export to file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    print(f"\nExporting to file: {temp_path}")
    await export_session_to_file(
        session_id=session.id,
        filepath=temp_path,
        format="json",
        session_manager=manager,
    )
    
    # Read and verify
    with open(temp_path, 'r') as f:
        content = f.read()
    print(f"File size: {len(content)} bytes")
    
    return session.id


async def example_websocket_manager():
    """Example: WebSocket manager for real-time updates."""
    print("\n=== Example: WebSocket Manager ===\n")
    
    manager = SessionManager()
    ws_manager = SessionWebSocketManager(manager)
    
    # Create a session
    session = await manager.create_session(title="WebSocket Demo")
    
    print(f"Created session: {session.id[:8]}...")
    print(f"WebSocket connections: {ws_manager.get_connection_stats()}")
    
    # Simulate events
    from rlm.session.types import MessageContext, MessageRole, MessageType
    
    # Add a message and notify
    message = await manager.add_message(
        session_id=session.id,
        role=MessageRole.USER.value,
        content="Test message for WebSocket",
    )
    
    context = MessageContext(
        id=message.id,
        session_id=message.session_id,
        role=MessageRole.USER,
        content=message.content,
        tokens=message.tokens,
        message_type=MessageType.STANDARD,
        created_at=message.created_at,
    )
    
    # This would normally broadcast to connected clients
    print("\nWebSocket events available:")
    print("  - notify_new_message()")
    print("  - notify_compaction()")
    print("  - notify_session_update()")
    print("  - notify_search_complete()")
    
    # Show connection stats
    stats = ws_manager.get_connection_stats()
    print(f"\nConnection stats: {stats}")


async def example_event_broadcaster():
    """Example: Automatic event broadcasting."""
    print("\n=== Example: Event Broadcaster ===\n")
    
    manager = SessionManager()
    ws_manager = SessionWebSocketManager(manager)
    broadcaster = SessionEventBroadcaster(ws_manager)
    
    print("Event broadcaster initialized")
    print("Can wrap functions to auto-broadcast:")
    print("  - add_message")
    print("  - compaction events")
    print("  - session updates")
    
    # Example of how to use (normally you'd wrap the manager method)
    print("\nUsage:")
    print("  broadcaster.wrap_add_message(session_manager.add_message)")


async def example_batch_export():
    """Example: Batch export multiple sessions."""
    print("\n=== Example: Batch Export ===\n")
    
    manager = SessionManager()
    
    # Create multiple sessions
    print("Creating sessions...")
    session_ids = []
    for i in range(3):
        session = await manager.create_session(title=f"Session {i+1}")
        
        # Add messages
        await manager.add_message(
            session_id=session.id,
            role=MessageRole.USER.value,
            content=f"Question {i+1}",
        )
        await manager.add_message(
            session_id=session.id,
            role=MessageRole.ASSISTANT.value,
            content=f"Answer {i+1}",
        )
        
        session_ids.append(session.id)
        print(f"  Created: {session.id[:8]}...")
    
    # Batch export
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nExporting to {tmpdir}...")
        
        exporter = ConversationExporter(manager)
        exported_files = await exporter.export_batch(
            session_ids=session_ids,
            format="json",
            output_dir=tmpdir,
        )
        
        print(f"Exported {len(exported_files)} files:")
        for filepath in exported_files:
            print(f"  - {filepath.split('/')[-1]}")


async def main():
    """Run all advanced examples."""
    print("=" * 70)
    print("Component 9: Advanced Features - Examples")
    print("=" * 70)
    
    try:
        # Cleanup examples
        await example_cleanup_job()
        await example_background_cleanup()
        await example_cleanup_scheduler()
        
        # Export examples
        await example_export_formats()
        await example_batch_export()
        
        # WebSocket examples
        await example_websocket_manager()
        await example_event_broadcaster()
        
        print("\n" + "=" * 70)
        print("All advanced examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

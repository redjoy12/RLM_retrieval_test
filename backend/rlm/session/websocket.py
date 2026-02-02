"""WebSocket integration for real-time session updates.

Provides WebSocket endpoints for real-time session monitoring,
including message updates, compaction events, and search results.
"""

import json
from typing import Any, Dict, List, Optional, Set
from datetime import datetime

import structlog
from fastapi import WebSocket, WebSocketDisconnect

from rlm.session.manager import SessionManager
from rlm.session.types import MessageContext, SessionStatus

logger = structlog.get_logger()


class SessionWebSocketManager:
    """Manages WebSocket connections for real-time session updates.

    Provides real-time updates for:
    - New messages added to sessions
    - Context compaction events
    - Session status changes
    - Search results

    Example:
        ```python
        manager = SessionWebSocketManager(session_manager)

        # In FastAPI route
        @app.websocket("/ws/sessions/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, session_id: str):
            await manager.connect(websocket, session_id)
        ```
    """

    def __init__(self, session_manager: Optional[SessionManager] = None) -> None:
        """Initialize WebSocket manager.

        Args:
            session_manager: Session manager instance
        """
        self.session_manager = session_manager or SessionManager()

        # Active connections: session_id -> set of websockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}

        # Global subscribers (receive all updates)
        self.global_subscribers: Set[WebSocket] = set()

        logger.info("websocket_manager_initialized")

    async def connect(
        self,
        websocket: WebSocket,
        session_id: Optional[str] = None,
        global_updates: bool = False,
    ) -> None:
        """Accept a WebSocket connection.

        Args:
            websocket: WebSocket object
            session_id: Optional session ID to subscribe to
            global_updates: Whether to receive all session updates
        """
        await websocket.accept()

        if session_id:
            if session_id not in self.active_connections:
                self.active_connections[session_id] = set()
            self.active_connections[session_id].add(websocket)
            logger.info("websocket_connected", session_id=session_id)

        if global_updates:
            self.global_subscribers.add(websocket)
            logger.info("websocket_global_subscriber_added")

    def disconnect(
        self,
        websocket: WebSocket,
        session_id: Optional[str] = None,
    ) -> None:
        """Disconnect a WebSocket.

        Args:
            websocket: WebSocket object
            session_id: Optional session ID to unsubscribe from
        """
        if session_id and session_id in self.active_connections:
            self.active_connections[session_id].discard(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]

        self.global_subscribers.discard(websocket)

        logger.info("websocket_disconnected", session_id=session_id)

    async def broadcast_to_session(
        self,
        session_id: str,
        message: Dict[str, Any],
    ) -> None:
        """Broadcast message to all connections for a session.

        Args:
            session_id: Session ID
            message: Message to broadcast
        """
        if session_id not in self.active_connections:
            return

        disconnected = set()

        for websocket in self.active_connections[session_id]:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.add(websocket)

        # Clean up disconnected clients
        for websocket in disconnected:
            self.active_connections[session_id].discard(websocket)

        if not self.active_connections[session_id]:
            del self.active_connections[session_id]

    async def broadcast_global(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all global subscribers.

        Args:
            message: Message to broadcast
        """
        disconnected = set()

        for websocket in self.global_subscribers:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.add(websocket)

        # Clean up disconnected clients
        for websocket in disconnected:
            self.global_subscribers.discard(websocket)

    async def notify_new_message(
        self,
        session_id: str,
        message: MessageContext,
    ) -> None:
        """Notify clients of a new message.

        Args:
            session_id: Session ID
            message: Message that was added
        """
        await self.broadcast_to_session(
            session_id,
            {
                "type": "new_message",
                "session_id": session_id,
                "data": message.to_dict(),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    async def notify_compaction(
        self,
        session_id: str,
        tokens_saved: int,
        summary_content: str,
    ) -> None:
        """Notify clients of context compaction.

        Args:
            session_id: Session ID
            tokens_saved: Number of tokens saved
            summary_content: Summary content
        """
        await self.broadcast_to_session(
            session_id,
            {
                "type": "compaction",
                "session_id": session_id,
                "data": {
                    "tokens_saved": tokens_saved,
                    "summary": summary_content[:200],  # Truncated
                },
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    async def notify_session_update(
        self,
        session_id: str,
        update_type: str,
        data: Dict[str, Any],
    ) -> None:
        """Notify clients of session updates.

        Args:
            session_id: Session ID
            update_type: Type of update (status, stats, etc.)
            data: Update data
        """
        await self.broadcast_to_session(
            session_id,
            {
                "type": f"session_{update_type}",
                "session_id": session_id,
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    async def notify_search_complete(
        self,
        session_id: str,
        query: str,
        results_count: int,
        execution_time_ms: float,
    ) -> None:
        """Notify clients of search completion.

        Args:
            session_id: Session ID
            query: Search query
            results_count: Number of results
            execution_time_ms: Execution time
        """
        await self.broadcast_to_session(
            session_id,
            {
                "type": "search_complete",
                "session_id": session_id,
                "data": {
                    "query": query,
                    "results_count": results_count,
                    "execution_time_ms": execution_time_ms,
                },
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "total_session_subscriptions": len(self.active_connections),
            "global_subscribers": len(self.global_subscribers),
            "sessions_with_connections": list(self.active_connections.keys()),
            "total_connections": sum(len(conns) for conns in self.active_connections.values())
            + len(self.global_subscribers),
        }


# WebSocket endpoint handler for FastAPI


async def handle_session_websocket(
    websocket: WebSocket,
    session_id: str,
    manager: SessionWebSocketManager,
) -> None:
    """Handle WebSocket connection for a session.

    Args:
        websocket: WebSocket object
        session_id: Session ID
        manager: WebSocket manager instance
    """
    await manager.connect(websocket, session_id)

    try:
        while True:
            # Receive and handle client messages
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                action = message.get("action")

                if action == "ping":
                    await websocket.send_json({"type": "pong"})

                elif action == "get_context":
                    # Send current context
                    context = await manager.session_manager.get_context(session_id)
                    await websocket.send_json(
                        {
                            "type": "context",
                            "data": [msg.to_dict() for msg in context],
                        }
                    )

                elif action == "get_stats":
                    # Send session stats
                    stats = await manager.session_manager.get_session_stats(session_id)
                    await websocket.send_json(
                        {
                            "type": "stats",
                            "data": stats,
                        }
                    )

                else:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"Unknown action: {action}",
                        }
                    )

            except json.JSONDecodeError:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Invalid JSON",
                    }
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)
    except Exception as e:
        logger.error("websocket_error", session_id=session_id, error=str(e))
        manager.disconnect(websocket, session_id)


# Decorator for auto-broadcasting session events


class SessionEventBroadcaster:
    """Decorator/utility to auto-broadcast session events via WebSocket."""

    def __init__(self, manager: SessionWebSocketManager) -> None:
        """Initialize broadcaster.

        Args:
            manager: WebSocket manager
        """
        self.manager = manager

    async def wrap_add_message(self, original_func):
        """Wrap add_message to broadcast updates.

        Args:
            original_func: Original add_message function

        Returns:
            Wrapped function
        """

        async def wrapper(*args, **kwargs):
            result = await original_func(*args, **kwargs)

            # Extract session_id from args or kwargs
            session_id = kwargs.get("session_id") or args[1] if len(args) > 1 else None

            if session_id:
                # Create message context from result
                from rlm.session.types import MessageContext, MessageRole, MessageType

                message_context = MessageContext(
                    id=result.id,
                    session_id=result.session_id,
                    role=MessageRole(result.role),
                    content=result.content,
                    tokens=result.tokens,
                    message_type=MessageType(result.message_type),
                    parent_message_id=result.parent_message_id,
                    trajectory_id=result.trajectory_id,
                    created_at=result.created_at,
                    metadata=result.metadata,
                )

                await self.manager.notify_new_message(session_id, message_context)

            return result

        return wrapper

"""WebSocket routes for real-time session updates.

Provides WebSocket endpoints for real-time session monitoring
and updates.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import structlog

from rlm.session import SessionManager, SessionWebSocketManager, handle_session_websocket
from rlm.config import get_session_settings

logger = structlog.get_logger()

router = APIRouter(prefix="/ws", tags=["websocket"])

# Global WebSocket manager instance
_websocket_manager: SessionWebSocketManager = None


def get_websocket_manager() -> SessionWebSocketManager:
    """Get or create WebSocket manager instance."""
    global _websocket_manager
    if _websocket_manager is None:
        _websocket_manager = SessionWebSocketManager()
    return _websocket_manager


@router.websocket("/sessions/{session_id}")
async def session_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time session updates.
    
    Connect to receive real-time updates for a session:
    - New messages
    - Compaction events
    - Session updates
    - Search completions
    
    Client messages:
    - `{"action": "ping"}` - Ping server
    - `{"action": "get_context"}` - Get current conversation context
    - `{"action": "get_stats"}` - Get session statistics
    
    Server messages:
    - `{"type": "new_message", "data": {...}}` - New message added
    - `{"type": "compaction", "data": {...}}` - Context compacted
    - `{"type": "session_update", "data": {...}}` - Session updated
    - `{"type": "search_complete", "data": {...}}` - Search completed
    """
    manager = get_websocket_manager()
    
    try:
        await handle_session_websocket(websocket, session_id, manager)
    except WebSocketDisconnect:
        logger.info("websocket_disconnected", session_id=session_id)
    except Exception as e:
        logger.error("websocket_error", session_id=session_id, error=str(e))


@router.websocket("/sessions")
async def global_sessions_websocket(websocket: WebSocket):
    """WebSocket endpoint for global session updates.
    
    Subscribe to all session events (admin/monitoring use case).
    """
    manager = get_websocket_manager()
    await manager.connect(websocket, global_updates=True)
    
    try:
        while True:
            data = await websocket.receive_text()
            # Handle ping or other global commands
            import json
            try:
                msg = json.loads(data)
                if msg.get("action") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif msg.get("action") == "stats":
                    stats = manager.get_connection_stats()
                    await websocket.send_json({"type": "stats", "data": stats})
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("global_websocket_disconnected")

"""Trajectory WebSocket Manager

Manages WebSocket connections for real-time trajectory streaming.
Provides live updates as RLM execution progresses.
"""

import asyncio
from typing import Dict, Set, Optional, Any
from datetime import datetime
import json

import structlog
from fastapi import WebSocket, WebSocketDisconnect

from rlm.trajectory.logger import TrajectoryLogger
from rlm.types import StreamEvent, StreamEventType

logger = structlog.get_logger()


class TrajectoryWebSocketManager:
    """Manages WebSocket connections for live trajectory updates.
    
    This manager handles multiple client connections and broadcasts
    trajectory events to all subscribers of a session.
    
    Example:
        >>> manager = TrajectoryWebSocketManager()
        >>> await manager.connect(websocket, "session-123")
        >>> await manager.broadcast("session-123", event)
    """
    
    def __init__(self) -> None:
        """Initialize the WebSocket manager."""
        # Map session_id to set of connected websockets
        self._connections: Dict[str, Set[WebSocket]] = {}
        # Map websocket to session_id for cleanup
        self._websocket_sessions: Dict[WebSocket, str] = {}
        
        logger.info("trajectory_websocket_manager_initialized")
    
    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Accept a new WebSocket connection.
        
        Args:
            websocket: The WebSocket connection
            session_id: Session to subscribe to
        """
        await websocket.accept()
        
        # Add to session connections
        if session_id not in self._connections:
            self._connections[session_id] = set()
        
        self._connections[session_id].add(websocket)
        self._websocket_sessions[websocket] = session_id
        
        logger.info(
            "websocket_connected",
            session_id=session_id,
            total_connections=len(self._connections[session_id]),
        )
        
        # Send connection confirmation
        await self._send_message(websocket, {
            "type": "connection_established",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection.
        
        Args:
            websocket: The WebSocket to remove
        """
        session_id = self._websocket_sessions.pop(websocket, None)
        
        if session_id and session_id in self._connections:
            self._connections[session_id].discard(websocket)
            
            # Clean up empty session
            if not self._connections[session_id]:
                del self._connections[session_id]
            
            logger.info(
                "websocket_disconnected",
                session_id=session_id,
                remaining_connections=len(self._connections.get(session_id, set())),
            )
    
    async def broadcast(
        self,
        session_id: str,
        message: Dict[str, Any],
    ) -> None:
        """Broadcast a message to all connections for a session.
        
        Args:
            session_id: Session to broadcast to
            message: Message to send
        """
        if session_id not in self._connections:
            return
        
        disconnected = []
        
        for websocket in self._connections[session_id]:
            try:
                await self._send_message(websocket, message)
            except Exception as e:
                logger.warning(
                    "websocket_send_failed",
                    session_id=session_id,
                    error=str(e),
                )
                disconnected.append(websocket)
        
        # Clean up failed connections
        for websocket in disconnected:
            self.disconnect(websocket)
    
    async def _send_message(
        self,
        websocket: WebSocket,
        message: Dict[str, Any],
    ) -> None:
        """Send a message to a specific websocket.
        
        Args:
            websocket: Target WebSocket
            message: Message to send
        """
        await websocket.send_json(message)
    
    def get_connection_count(self, session_id: Optional[str] = None) -> int:
        """Get number of active connections.
        
        Args:
            session_id: Specific session or None for all
            
        Returns:
            Number of connections
        """
        if session_id:
            return len(self._connections.get(session_id, set()))
        
        return len(self._websocket_sessions)
    
    def is_subscribed(self, session_id: str) -> bool:
        """Check if a session has any subscribers.
        
        Args:
            session_id: Session to check
            
        Returns:
            True if session has subscribers
        """
        return session_id in self._connections and bool(self._connections[session_id])


class TrajectoryStreamAdapter:
    """Adapter to convert StreamEvents to WebSocket messages.
    
    Bridges the existing streaming infrastructure with WebSocket
    broadcasting for the trajectory visualizer.
    """
    
    def __init__(self, manager: TrajectoryWebSocketManager) -> None:
        """Initialize the adapter.
        
        Args:
            manager: WebSocket manager instance
        """
        self.manager = manager
        
    def create_callback(self, session_id: str):
        """Create a callback function for streaming events.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Callback function for stream events
        """
        async def callback(event: StreamEvent) -> None:
            """Handle stream event and broadcast to WebSocket clients."""
            message = self._convert_event(event)
            await self.manager.broadcast(session_id, message)
        
        return callback
    
    def _convert_event(self, event: StreamEvent) -> Dict[str, Any]:
        """Convert StreamEvent to WebSocket message format.
        
        Args:
            event: StreamEvent from trajectory logger
            
        Returns:
            Dictionary suitable for WebSocket transmission
        """
        event_type_map = {
            StreamEventType.STEP_START: "step_start",
            StreamEventType.CODE_GENERATED: "code_generated",
            StreamEventType.CODE_OUTPUT: "code_output",
            StreamEventType.SUB_LLM_SPAWN: "sub_llm_spawn",
            StreamEventType.SUB_LLM_RESULT: "sub_llm_result",
            StreamEventType.FINAL_RESULT: "final_result",
            StreamEventType.ERROR: "error",
        }
        
        return {
            "type": event_type_map.get(event.type, "unknown"),
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": event.session_id,
            "data": event.data,
        }


# Global manager instance
websocket_manager = TrajectoryWebSocketManager()


async def handle_trajectory_websocket(websocket: WebSocket, session_id: str) -> None:
    """Handle WebSocket connection for trajectory streaming.
    
    This is the main entry point for WebSocket connections.
    
    Args:
        websocket: FastAPI WebSocket object
        session_id: Session to subscribe to
    """
    await websocket_manager.connect(websocket, session_id)
    
    try:
        while True:
            # Keep connection alive and handle client messages
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,
                )
                
                # Parse client message
                try:
                    message = json.loads(data)
                    msg_type = message.get("type", "unknown")
                    
                    if msg_type == "ping":
                        await websocket.send_json({"type": "pong"})
                    
                    elif msg_type == "request_full":
                        # Client requested full trajectory
                        # This would trigger sending the complete trajectory
                        logger.info(
                            "websocket_request_full_trajectory",
                            session_id=session_id,
                        )
                    
                except json.JSONDecodeError:
                    logger.warning(
                        "websocket_invalid_message",
                        session_id=session_id,
                        data=data,
                    )
            
            except asyncio.TimeoutError:
                # Send keepalive ping
                try:
                    await websocket.send_json({"type": "keepalive"})
                except:
                    break
    
    except WebSocketDisconnect:
        logger.info(
            "websocket_client_disconnected",
            session_id=session_id,
        )
    
    except Exception as e:
        logger.error(
            "websocket_error",
            session_id=session_id,
            error=str(e),
        )
    
    finally:
        websocket_manager.disconnect(websocket)

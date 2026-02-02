"""WebSocket routes for real-time trajectory streaming.

Provides WebSocket endpoints for live trajectory visualization.
"""

from fastapi import APIRouter, WebSocket

from rlm.trajectory import handle_trajectory_websocket

router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/trajectory/{session_id}")
async def trajectory_websocket(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for real-time trajectory updates.
    
    Connect to this endpoint to receive live updates as RLM executes.
    
    Args:
        websocket: WebSocket connection
        session_id: Session to subscribe to
        
    Example:
        ```javascript
        const ws = new WebSocket('ws://localhost:8000/ws/trajectory/session-123');
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log('Trajectory update:', data);
        };
        ```
    """
    await handle_trajectory_websocket(websocket, session_id)

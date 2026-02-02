/** WebSocket Hook for Real-time Updates */

import { useEffect, useRef, useState, useCallback } from "react";
import { useTrajectoryStore } from "@/stores/trajectoryStore";
import { WebSocketMessage } from "@/types/trajectory";

export function useWebSocket(sessionId: string | null) {
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const { addTimelineEvent, setLive } = useTrajectoryStore();

  const connect = useCallback(() => {
    if (!sessionId) return;

    const wsUrl = `ws://localhost:8000/ws/trajectory/${sessionId}`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setIsConnected(true);
      setLive(true);
      console.log("WebSocket connected");
    };

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);

        // Handle different message types
        switch (message.type) {
          case "connection_established":
            console.log("Connected to session:", message.session_id);
            break;

          case "step_start":
          case "step_complete":
          case "sub_llm_spawn":
          case "sub_llm_result":
          case "code_generated":
          case "code_output":
          case "error":
          case "final_result":
            // Convert to timeline event and add
            if (message.data) {
              addTimelineEvent({
                node_id: message.session_id,
                type: message.data.step_type || message.type,
                start_time: message.timestamp,
                end_time: null,
                duration_ms: null,
                depth: message.data.depth || 0,
                data: message.data,
              });
            }
            break;

          case "keepalive":
          case "pong":
            // Ignore keepalive messages
            break;

          default:
            console.log("Unknown message type:", message.type);
        }
      } catch (error) {
        console.error("Failed to parse WebSocket message:", error);
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      setLive(false);
      console.log("WebSocket disconnected");

      // Attempt to reconnect after 5 seconds
      setTimeout(() => {
        if (sessionId) {
          connect();
        }
      }, 5000);
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      setIsConnected(false);
      setLive(false);
    };

    wsRef.current = ws;
  }, [sessionId, addTimelineEvent, setLive]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
    setLive(false);
  }, [setLive]);

  const sendMessage = useCallback((message: object) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  useEffect(() => {
    if (sessionId) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [sessionId, connect, disconnect]);

  // Send ping every 30 seconds to keep connection alive
  useEffect(() => {
    if (!isConnected) return;

    const interval = setInterval(() => {
      sendMessage({ type: "ping" });
    }, 30000);

    return () => clearInterval(interval);
  }, [isConnected, sendMessage]);

  return {
    isConnected,
    connect,
    disconnect,
    sendMessage,
  };
}

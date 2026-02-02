/** API client for query endpoints */

import axios from "axios";

const API_BASE_URL = "/api/v1";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export interface QueryRequest {
  query: string;
  document_ids: string[];
  use_optimized_query?: boolean;
  show_routing_info?: boolean;
}

export interface QueryResponse {
  answer: string;
  execution_time_ms: number;
  tokens_used: number;
  cost_usd: number;
  routing?: {
    strategy: string;
    confidence: number;
    estimated_cost_usd: number;
  };
}

export interface QueryAnalysisResponse {
  query: string;
  routing_decision: {
    strategy: string;
    complexity: string;
    confidence: number;
  };
  visibility: {
    description: string;
    reason: string;
  };
  suggestions?: {
    optimized_query?: string;
    recommended_strategy?: string;
  };
}

export const queryApi = {
  /** Execute a query */
  async executeQuery(request: QueryRequest): Promise<QueryResponse> {
    const response = await api.post("/queries/execute", request);
    return response.data;
  },

  /** Analyze a query before execution */
  async analyzeQuery(
    query: string,
    documentIds: string[]
  ): Promise<QueryAnalysisResponse> {
    const response = await api.post("/queries/analyze", {
      query,
      document_ids: documentIds,
    });
    return response.data;
  },

  /** Get query status */
  async getQueryStatus(queryId: string): Promise<{
    status: string;
    progress?: number;
    result?: QueryResponse;
  }> {
    const response = await api.get(`/queries/${queryId}/status`);
    return response.data;
  },

  /** Stream query execution via WebSocket */
  streamQuery(
    request: QueryRequest,
    onMessage: (data: { type: string; content?: string; error?: string }) => void,
    onError?: (error: Event) => void,
    onClose?: () => void
  ): WebSocket {
    const sessionId = `query-${Date.now()}`;
    const ws = new WebSocket(`ws://localhost:8000/ws/trajectory/${sessionId}`);

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          type: "start_query",
          data: request,
        })
      );
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onMessage(data);
    };

    ws.onerror = (error) => {
      onError?.(error);
    };

    ws.onclose = () => {
      onClose?.();
    };

    return ws;
  },
};

export default api;

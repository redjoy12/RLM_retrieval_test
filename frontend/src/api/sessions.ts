/** API client for session endpoints */

import axios from "axios";

const API_BASE_URL = "/api/v1";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export interface Session {
  id: string;
  title: string;
  status: string;
  message_count: number;
  total_tokens: number;
  cost_usd: number;
  created_at: string;
  updated_at: string;
  document_ids: string[];
}

export interface SessionMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  tokens?: number;
  cost_usd?: number;
}

export interface CreateSessionRequest {
  title?: string;
  metadata?: Record<string, unknown>;
  default_search_strategy?: string;
  semantic_weight?: number;
  keyword_weight?: number;
  enable_reranking?: boolean;
  enable_citations?: boolean;
}

export const sessionApi = {
  /** Create a new session */
  async createSession(request: CreateSessionRequest = {}): Promise<Session> {
    const response = await api.post("/sessions", request);
    return response.data;
  },

  /** Get a session by ID */
  async getSession(sessionId: string): Promise<Session> {
    const response = await api.get(`/sessions/${sessionId}`);
    return response.data;
  },

  /** List all sessions */
  async listSessions(): Promise<Session[]> {
    const response = await api.get("/sessions");
    return response.data.sessions;
  },

  /** Continue a session with a new message */
  async continueSession(
    sessionId: string,
    message: string,
    documentIds?: string[]
  ): Promise<{ message: SessionMessage; response: SessionMessage }> {
    const response = await api.post(`/sessions/${sessionId}/continue`, {
      message,
      document_ids: documentIds,
    });
    return response.data;
  },

  /** Delete a session */
  async deleteSession(sessionId: string): Promise<void> {
    await api.delete(`/sessions/${sessionId}`);
  },

  /** Fork a session */
  async forkSession(
    sessionId: string,
    title?: string
  ): Promise<Session> {
    const response = await api.post(`/sessions/${sessionId}/fork`, { title });
    return response.data;
  },

  /** Search session history */
  async searchSessions(query: string): Promise<Session[]> {
    const response = await api.get("/sessions/search", {
      params: { query },
    });
    return response.data.results;
  },

  /** Export a session */
  async exportSession(
    sessionId: string,
    format: "json" | "markdown" = "json"
  ): Promise<Blob> {
    const response = await api.get(`/sessions/${sessionId}/export`, {
      params: { format },
      responseType: "blob",
    });
    return response.data;
  },
};

export default api;

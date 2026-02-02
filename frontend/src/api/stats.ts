/** API client for stats endpoints */

import axios from "axios";

const API_BASE_URL = "/api/v1";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export interface UsageStats {
  total_queries: number;
  total_sessions: number;
  total_documents: number;
  total_tokens: number;
  period: string;
}

export interface CostStats {
  total_cost_usd: number;
  cost_by_model: Record<string, number>;
  cost_by_day: Record<string, number>;
  average_cost_per_query: number;
  period: string;
}

export interface ModelUsageStats {
  model: string;
  queries: number;
  tokens: number;
  cost_usd: number;
}

export interface QueryStats {
  total_queries: number;
  avg_execution_time_ms: number;
  success_rate: number;
  queries_by_strategy: Record<string, number>;
}

export interface SystemHealth {
  status: string;
  version: string;
  uptime_seconds: number;
  active_sessions: number;
  components: Record<string, string>;
}

export interface SummaryStats {
  period: string;
  usage: UsageStats;
  costs: CostStats;
  queries: QueryStats;
  health: SystemHealth;
  generated_at: string;
}

export const statsApi = {
  /** Get usage statistics */
  async getUsageStats(days: number = 7): Promise<UsageStats> {
    const response = await api.get("/stats/usage", {
      params: { days },
    });
    return response.data;
  },

  /** Get cost statistics */
  async getCostStats(days: number = 7): Promise<CostStats> {
    const response = await api.get("/stats/costs", {
      params: { days },
    });
    return response.data;
  },

  /** Get model usage statistics */
  async getModelUsage(days: number = 7): Promise<ModelUsageStats[]> {
    const response = await api.get("/stats/models", {
      params: { days },
    });
    return response.data;
  },

  /** Get query statistics */
  async getQueryStats(days: number = 7): Promise<QueryStats> {
    const response = await api.get("/stats/queries", {
      params: { days },
    });
    return response.data;
  },

  /** Get document statistics */
  async getDocumentStats(): Promise<{
    total_documents: number;
    total_chunks: number;
    documents_by_format: Record<string, number>;
    avg_document_size_bytes: number;
  }> {
    const response = await api.get("/stats/documents");
    return response.data;
  },

  /** Get system health */
  async getSystemHealth(): Promise<SystemHealth> {
    const response = await api.get("/stats/health");
    return response.data;
  },

  /** Get summary of all statistics */
  async getSummary(days: number = 7): Promise<SummaryStats> {
    const response = await api.get("/stats/summary", {
      params: { days },
    });
    return response.data;
  },

  /** Health check */
  async healthCheck(): Promise<{ status: string; version: string; service: string }> {
    const response = await api.get("/health");
    return response.data;
  },
};

export default api;

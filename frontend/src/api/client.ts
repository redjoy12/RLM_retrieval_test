/** API client for trajectory endpoints */

import axios from "axios";
import {
  TrajectoryResponse,
  FlowTreeData,
  TimelineEvent,
  CostBreakdown,
  ExecutionStats,
  ExportFormat,
} from "@/types/trajectory";

const API_BASE_URL = "/api/v1";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export const trajectoryApi = {
  /** Get full trajectory data */
  async getTrajectory(sessionId: string): Promise<TrajectoryResponse> {
    const response = await api.get(`/trajectory/${sessionId}`);
    return response.data;
  },

  /** Get trajectory as tree structure (React Flow format) */
  async getTree(sessionId: string): Promise<FlowTreeData> {
    const response = await api.get(`/trajectory/${sessionId}/tree`);
    return response.data;
  },

  /** Get timeline data */
  async getTimeline(sessionId: string): Promise<TimelineEvent[]> {
    const response = await api.get(`/trajectory/${sessionId}/timeline`);
    return response.data.events;
  },

  /** Get cost breakdown */
  async getCosts(sessionId: string): Promise<CostBreakdown> {
    const response = await api.get(`/trajectory/${sessionId}/costs`);
    return response.data;
  },

  /** Get execution statistics */
  async getStatistics(sessionId: string): Promise<ExecutionStats> {
    const response = await api.get(`/trajectory/${sessionId}/statistics`);
    return response.data;
  },

  /** Export trajectory in specified format */
  async export(sessionId: string, format: ExportFormat): Promise<Blob> {
    const response = await api.get(
      `/trajectory/${sessionId}/export/${format}`,
      {
        responseType: "blob",
      }
    );
    return response.data;
  },
};

export default api;

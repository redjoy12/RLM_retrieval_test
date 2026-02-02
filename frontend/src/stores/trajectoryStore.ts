/** Zustand store for trajectory state management */

import { create } from "zustand";
import {
  TrajectoryState,
  TrajectoryResponse,
  FlowTreeData,
  TimelineEvent,
  ViewMode,
} from "@/types/trajectory";

export const useTrajectoryStore = create<TrajectoryState>((set) => ({
  // State
  sessionId: null,
  trajectory: null,
  treeData: null,
  selectedNodeId: null,
  viewMode: "tree",
  isLoading: false,
  isLive: false,
  error: null,

  // Actions
  setSessionId: (id) => set({ sessionId: id }),

  setTrajectory: (trajectory) =>
    set({
      trajectory,
      error: null,
    }),

  setTreeData: (treeData) =>
    set({
      treeData,
      error: null,
    }),

  selectNode: (nodeId) => set({ selectedNodeId: nodeId }),

  setViewMode: (mode) => set({ viewMode: mode }),

  setLoading: (loading) => set({ isLoading: loading }),

  setLive: (live) => set({ isLive: live }),

  setError: (error) => set({ error }),

  addTimelineEvent: (event) =>
    set((state) => {
      if (!state.trajectory) return state;

      const newTimeline = [...state.trajectory.timeline, event];
      return {
        trajectory: {
          ...state.trajectory,
          timeline: newTimeline,
        },
      };
    }),

  clear: () =>
    set({
      sessionId: null,
      trajectory: null,
      treeData: null,
      selectedNodeId: null,
      viewMode: "tree",
      isLoading: false,
      isLive: false,
      error: null,
    }),
}));

/** Zustand store for query state management */

import { create } from "zustand";

export interface QueryMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  tokens?: number;
  cost?: number;
}

export interface QueryState {
  // State
  messages: QueryMessage[];
  currentSessionId: string | null;
  isLoading: boolean;
  error: string | null;
  selectedStrategy: "rlm" | "rag" | "hybrid" | "direct";
  streamingContent: string;
  
  // Actions
  addMessage: (message: QueryMessage) => void;
  updateMessage: (messageId: string, updates: Partial<QueryMessage>) => void;
  clearMessages: () => void;
  setCurrentSession: (sessionId: string | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setStrategy: (strategy: QueryState["selectedStrategy"]) => void;
  appendStreamingContent: (content: string) => void;
  clearStreamingContent: () => void;
}

export const useQueryStore = create<QueryState>((set) => ({
  // Initial state
  messages: [],
  currentSessionId: null,
  isLoading: false,
  error: null,
  selectedStrategy: "rlm",
  streamingContent: "",
  
  // Actions
  addMessage: (message) =>
    set((state) => ({
      messages: [...state.messages, message],
    })),
  
  updateMessage: (messageId, updates) =>
    set((state) => ({
      messages: state.messages.map((m) =>
        m.id === messageId ? { ...m, ...updates } : m
      ),
    })),
  
  clearMessages: () => set({ messages: [], streamingContent: "" }),
  
  setCurrentSession: (sessionId) => set({ currentSessionId: sessionId }),
  
  setLoading: (loading) => set({ isLoading: loading }),
  
  setError: (error) => set({ error }),
  
  setStrategy: (strategy) => set({ selectedStrategy: strategy }),
  
  appendStreamingContent: (content) =>
    set((state) => ({
      streamingContent: state.streamingContent + content,
    })),
  
  clearStreamingContent: () => set({ streamingContent: "" }),
}));

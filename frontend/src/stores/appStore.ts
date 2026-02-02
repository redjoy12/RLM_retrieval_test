/** Zustand store for global app state management */

import { create } from "zustand";

export interface AppState {
  // UI State
  sidebarCollapsed: boolean;
  activeTab: string;
  
  // Settings
  defaultModel: string;
  temperature: number;
  maxTokens: number;
  
  // Actions
  toggleSidebar: () => void;
  setActiveTab: (tab: string) => void;
  setDefaultModel: (model: string) => void;
  setTemperature: (temp: number) => void;
  setMaxTokens: (tokens: number) => void;
}

export const useAppStore = create<AppState>((set) => ({
  // Initial state
  sidebarCollapsed: false,
  activeTab: "query",
  defaultModel: "gpt-5-mini",
  temperature: 0.7,
  maxTokens: 4000,
  
  // Actions
  toggleSidebar: () =>
    set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
  
  setActiveTab: (tab) => set({ activeTab: tab }),
  
  setDefaultModel: (model) => set({ defaultModel: model }),
  
  setTemperature: (temp) => set({ temperature: temp }),
  
  setMaxTokens: (tokens) => set({ maxTokens: tokens }),
}));

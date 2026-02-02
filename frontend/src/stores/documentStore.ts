/** Zustand store for document state management */

import { create } from "zustand";
import type { Document } from "@/api/documents";

export interface DocumentState {
  // State
  documents: Document[];
  selectedDocuments: string[];
  isLoading: boolean;
  error: string | null;
  uploadProgress: Record<string, number>;
  
  // Actions
  setDocuments: (documents: Document[]) => void;
  addDocument: (document: Document) => void;
  removeDocument: (documentId: string) => void;
  selectDocument: (documentId: string) => void;
  deselectDocument: (documentId: string) => void;
  clearSelection: () => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setUploadProgress: (documentId: string, progress: number) => void;
  clearUploadProgress: (documentId: string) => void;
}

export const useDocumentStore = create<DocumentState>((set) => ({
  // Initial state
  documents: [],
  selectedDocuments: [],
  isLoading: false,
  error: null,
  uploadProgress: {},
  
  // Actions
  setDocuments: (documents) => set({ documents }),
  
  addDocument: (document) =>
    set((state) => ({
      documents: [...state.documents, document],
    })),
  
  removeDocument: (documentId) =>
    set((state) => ({
      documents: state.documents.filter((d) => d.id !== documentId),
      selectedDocuments: state.selectedDocuments.filter((id) => id !== documentId),
    })),
  
  selectDocument: (documentId) =>
    set((state) => ({
      selectedDocuments: [...state.selectedDocuments, documentId],
    })),
  
  deselectDocument: (documentId) =>
    set((state) => ({
      selectedDocuments: state.selectedDocuments.filter((id) => id !== documentId),
    })),
  
  clearSelection: () => set({ selectedDocuments: [] }),
  
  setLoading: (loading) => set({ isLoading: loading }),
  
  setError: (error) => set({ error }),
  
  setUploadProgress: (documentId, progress) =>
    set((state) => ({
      uploadProgress: {
        ...state.uploadProgress,
        [documentId]: progress,
      },
    })),
  
  clearUploadProgress: (documentId) =>
    set((state) => {
      const { [documentId]: _, ...rest } = state.uploadProgress;
      return { uploadProgress: rest };
    }),
}));

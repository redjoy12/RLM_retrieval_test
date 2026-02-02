/** API client for document endpoints */

import axios from "axios";

const API_BASE_URL = "/api/v1";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export interface Document {
  id: string;
  filename: string;
  file_size: number;
  mime_type: string;
  format_type: string;
  status: string;
  created_at: string;
  updated_at: string;
  error_message?: string;
}

export interface DocumentListResponse {
  documents: Document[];
  total: number;
}

export interface UploadProgress {
  loaded: number;
  total: number;
  percentage: number;
}

export const documentApi = {
  /** Upload a document */
  async uploadDocument(
    file: File,
    onProgress?: (progress: UploadProgress) => void
  ): Promise<Document> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await api.post("/documents/upload", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          onProgress({
            loaded: progressEvent.loaded,
            total: progressEvent.total,
            percentage: Math.round((progressEvent.loaded * 100) / progressEvent.total),
          });
        }
      },
    });
    return response.data;
  },

  /** List all documents */
  async listDocuments(): Promise<DocumentListResponse> {
    const response = await api.get("/documents/list");
    return response.data;
  },

  /** Delete a document */
  async deleteDocument(documentId: string): Promise<void> {
    await api.delete(`/documents/${documentId}`);
  },

  /** Get document content */
  async getDocumentContent(documentId: string): Promise<{ content: string; word_count: number }> {
    const response = await api.get(`/documents/${documentId}/content`);
    return response.data;
  },
};

export default api;

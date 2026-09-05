export interface DocumentItem {
  id: string;
  filename: string;
  file_type: string;
  file_size_bytes: number;
  page_count: number;
  chunk_count: number;
  doc_metadata?: Record<string, any>;
  created_at: number;
}

export interface DocumentUploadResponse {
  success: boolean;
  document: DocumentItem;
  message: string;
}

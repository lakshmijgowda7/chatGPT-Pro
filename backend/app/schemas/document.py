"""
Pydantic Schemas for Document Ingestion & RAG
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    file_type: str
    file_size_bytes: int
    page_count: int
    chunk_count: int
    doc_metadata: Optional[Dict[str, Any]] = None
    created_at: float


class DocumentUploadResponse(BaseModel):
    success: bool
    document: DocumentRead
    message: str

"""
Documents & RAG REST Endpoints
Handles file uploads (PDF, DOCX, TXT), listing, and vector index registration.
"""

from typing import List
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.models.document import Document
from app.schemas.document import DocumentRead, DocumentUploadResponse
from app.services.document_service import DocumentService

router = APIRouter()


@router.get("", response_model=List[DocumentRead])
def list_documents(db: Session = Depends(get_db)):
    """Lists all uploaded documents."""
    return DocumentService.list_documents(db)


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Uploads and indexes a document for grounded RAG QA (supports PDF, DOCX, TXT, MD, CSV, JSON)."""
    allowed_exts = {".pdf", ".docx", ".txt", ".md", ".csv", ".json"}
    filename = file.filename or "uploaded_file.txt"
    import os
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Supported formats: PDF, DOCX, TXT, MD, CSV, JSON.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    doc = DocumentService.process_and_save_document(
        db=db,
        file_bytes=file_bytes,
        filename=filename,
    )

    return DocumentUploadResponse(
        success=True,
        document=doc,
        message=f"'{filename}' uploaded and parsed successfully ({doc.chunk_count} chunks indexed in FAISS vector store).",
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Deletes a document from database and vector index."""
    deleted = DocumentService.delete_document(db, document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return None


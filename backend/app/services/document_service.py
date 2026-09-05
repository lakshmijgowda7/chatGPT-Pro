"""
Document Service
Handles file uploads, text parsing, RAG chunk registration, vector index syncing,
and deletion across PostgreSQL / SQLite and persistent FAISS store.
"""

import os
import time
from typing import List, Optional
from sqlalchemy.orm import Session
from app.core.config import settings
from app.models.document import Document
from app.rag.loader import load_document
from app.rag.splitter import split_document
from app.rag.retriever import rag_retriever
from app.core.logging import logger


class DocumentService:
    @staticmethod
    def process_and_save_document(
        db: Session,
        file_bytes: bytes,
        filename: str,
        user_id: Optional[str] = None,
    ) -> Document:
        os.makedirs(settings.UPLOAD_DIRECTORY, exist_ok=True)
        file_path = os.path.join(settings.UPLOAD_DIRECTORY, filename)
        
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        extracted = load_document(file_bytes, filename)
        chunks = split_document(extracted)
        rag_retriever.add_chunks(chunks)

        # Check if existing record with same filename exists to update or create
        existing_doc = db.query(Document).filter(Document.filename == filename).first()
        if existing_doc:
            existing_doc.file_type = extracted.file_type
            existing_doc.file_path = file_path
            existing_doc.file_size_bytes = len(file_bytes)
            existing_doc.page_count = extracted.page_count
            existing_doc.chunk_count = len(chunks)
            existing_doc.doc_metadata = {"total_words": extracted.total_words}
            existing_doc.updated_at = time.time()
            db.commit()
            db.refresh(existing_doc)
            return existing_doc

        db_doc = Document(
            user_id=user_id,
            filename=filename,
            file_type=extracted.file_type,
            file_path=file_path,
            file_size_bytes=len(file_bytes),
            page_count=extracted.page_count,
            chunk_count=len(chunks),
            doc_metadata={"total_words": extracted.total_words},
            created_at=time.time(),
            updated_at=time.time(),
        )
        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)
        return db_doc

    @staticmethod
    def list_documents(db: Session) -> List[Document]:
        return db.query(Document).order_by(Document.created_at.desc()).all()

    @staticmethod
    def delete_document(db: Session, document_id: str) -> bool:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            return False

        filename = doc.filename
        file_path = doc.file_path

        # Delete from database
        db.delete(doc)
        db.commit()

        # Remove from persistent vector index
        rag_retriever.delete_document_chunks(filename)

        # Remove physical file if present
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove file '{file_path}': {e}")

        return True

    @staticmethod
    def sync_existing_documents_to_vector_store(db: Session) -> int:
        """
        Ensures that documents saved in database are loaded into the vector store
        if the vector store is empty on startup.
        """
        stats = rag_retriever.get_stats()
        if stats["total_chunks"] > 0:
            return stats["total_chunks"]

        docs = db.query(Document).all()
        indexed_count = 0
        for doc in docs:
            if doc.file_path and os.path.exists(doc.file_path):
                try:
                    with open(doc.file_path, "rb") as f:
                        file_bytes = f.read()
                    extracted = load_document(file_bytes, doc.filename)
                    chunks = split_document(extracted)
                    rag_retriever.add_chunks(chunks)
                    indexed_count += len(chunks)
                except Exception as e:
                    logger.warning(f"Could not re-index document '{doc.filename}': {e}")
        return indexed_count

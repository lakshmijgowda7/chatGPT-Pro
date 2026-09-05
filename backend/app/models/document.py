"""
SQLAlchemy Document Entity (for RAG index registry)
Stores uploaded document metadata, file properties, parsing statistics, and user association.
"""

import time
import uuid
from typing import Optional, Dict, Any, TYPE_CHECKING
from sqlalchemy import String, Float, Integer, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database.base import Base

if TYPE_CHECKING:
    from app.models.user import User


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: f"doc_{uuid.uuid4().hex[:12]}",
    )
    user_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(64), nullable=False)  # pdf, docx, txt, md
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    page_count: Mapped[int] = mapped_column(Integer, default=1)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    doc_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time, nullable=False)
    updated_at: Mapped[float] = mapped_column(Float, default=time.time, onupdate=time.time, nullable=False)

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="documents")

    @property
    def metadata_dict(self) -> Optional[Dict[str, Any]]:
        return self.doc_metadata

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename} chunks={self.chunk_count}>"

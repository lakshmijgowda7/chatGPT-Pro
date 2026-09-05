"""
SQLAlchemy Message Entity
Represents individual conversation turns with role, content, RAG sources, and metadata.
"""

import time
import uuid
from typing import TYPE_CHECKING, Optional, Dict, Any
from sqlalchemy import String, Float, Text, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database.base import Base

if TYPE_CHECKING:
    from app.models.conversation import Conversation


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: f"msg_{uuid.uuid4().hex[:12]}",
    )
    conversation_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)  # user, assistant, system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)  # RAG sources list
    msg_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)  # Token count, model
    created_at: Mapped[float] = mapped_column(Float, default=time.time, nullable=False)

    # Relationships
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")

    def __repr__(self) -> str:
        return f"<Message id={self.id} conv={self.conversation_id} role={self.role}>"

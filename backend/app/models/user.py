"""
SQLAlchemy User Model
Implements persistent user profile, role-based access, and secure hashed authentication.
Secrets are never stored in plain text.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional, TYPE_CHECKING, Dict, Any
from sqlalchemy import String, Boolean, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database.base import Base
from app.core.security import get_password_hash, verify_password

if TYPE_CHECKING:
    from app.models.conversation import Conversation
    from app.models.document import Document


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: f"usr_{uuid.uuid4().hex[:12]}",
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    profile: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    # Relationships
    conversations: Mapped[List["Conversation"]] = relationship(
        "Conversation",
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="Conversation.updated_at.desc()",
    )
    documents: Mapped[List["Document"]] = relationship(
        "Document",
        back_populates="user",
    )

    def set_password(self, plain_password: str) -> None:
        """Hash plain password with bcrypt before storage."""
        self.hashed_password = get_password_hash(plain_password)

    def check_password(self, plain_password: str) -> bool:
        """Verify plain password against stored bcrypt hash."""
        return verify_password(plain_password, self.hashed_password)

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email}>"

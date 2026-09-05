"""
SQLAlchemy Models Registry
Centralizes all entity models for easy imports and metadata collection.
"""

from app.database.base import Base
from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.document import Document

__all__ = ["Base", "User", "Conversation", "Message", "Document"]

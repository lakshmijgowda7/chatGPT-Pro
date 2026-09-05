"""
Pydantic Schemas for Conversation Management
"""

from typing import List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    role: str
    content: str
    sources: Optional[Any] = None
    created_at: float


class ConversationCreate(BaseModel):
    title: Optional[str] = Field("New Chat", max_length=255)
    system_prompt: Optional[str] = None


class ConversationUpdate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)


class ConversationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    message_count: int
    created_at: float
    updated_at: float


class ConversationDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    system_prompt: Optional[str]
    created_at: float
    updated_at: float
    messages: List[MessageRead]

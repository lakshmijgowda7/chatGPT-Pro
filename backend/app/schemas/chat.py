"""
Pydantic Schemas for Chat Requests & Responses
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChatCompletionRequest(BaseModel):
    conversation_id: Optional[str] = Field(None, description="Active conversation identifier")
    message: str = Field(..., min_length=1, description="User prompt text")
    mode: str = Field("chat", description="Chat mode: 'chat' or 'rag'")
    system_prompt: Optional[str] = Field(None, description="Optional override system prompt")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(1024, ge=1, le=4096)


class SourceReference(BaseModel):
    rank: int
    source: str
    page_number: Optional[int] = 1
    file_type: str = "txt"
    score: float
    score_pct: str
    text: str


class ChatCompletionResponse(BaseModel):
    conversation_id: str
    message_id: str
    role: str = "assistant"
    content: str
    sources: Optional[List[SourceReference]] = None
    created_at: float

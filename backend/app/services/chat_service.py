"""
Chat Service
Coordinates conversation lifecycle, message recording, and hosted LLM inference.
"""

import time
from typing import List, Dict, Any, Optional, Tuple, AsyncIterator
from sqlalchemy.orm import Session
from app.models.conversation import Conversation
from app.models.message import Message
from app.llm.client import llm_client
from app.llm.prompt import DEFAULT_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT
from app.rag.retriever import rag_retriever
from app.services.memory_service import MemoryService


class ChatService:
    @staticmethod
    def get_or_create_conversation(
        db: Session,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Conversation:
        if conversation_id:
            conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
            if conv:
                if not conv.user_id and user_id:
                    conv.user_id = user_id
                    db.commit()
                    db.refresh(conv)
                return conv

        conv = Conversation(
            user_id=user_id,
            title="New Chat",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
        return conv

    @staticmethod
    def save_user_message(db: Session, conversation_id: str, content: str) -> Message:
        msg = Message(
            conversation_id=conversation_id,
            role="user",
            content=content,
            created_at=time.time(),
        )
        db.add(msg)
        
        # Auto-update conversation title on first message
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conv and conv.title == "New Chat":
            clean_title = content.replace("\n", " ").strip()
            conv.title = (clean_title[:30] + "...") if len(clean_title) > 30 else clean_title
            conv.updated_at = time.time()

        db.commit()
        db.refresh(msg)
        return msg

    @staticmethod
    def save_assistant_message(
        db: Session,
        conversation_id: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Message:
        msg = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            sources={"items": sources} if sources else None,
            created_at=time.time(),
        )
        db.add(msg)
        
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conv:
            conv.updated_at = time.time()

        db.commit()
        db.refresh(msg)
        return msg

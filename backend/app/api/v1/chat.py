"""
Chat & Streaming REST / SSE Endpoints
Handles synchronous completions and real-time token streaming from the hosted LLM.
"""

import json
import time
from typing import AsyncIterator, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.api.deps import get_optional_current_user
from app.models.user import User
from app.schemas.chat import ChatCompletionRequest, ChatCompletionResponse, SourceReference
from app.services.chat_service import ChatService
from app.services.memory_service import MemoryService
from app.llm.client import llm_client
from app.llm.prompt import DEFAULT_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT
from app.rag.retriever import rag_retriever
from app.models.conversation import Conversation
from app.models.message import Message

router = APIRouter()

GUEST_MAX_CHATS = 5


def _check_guest_chat_limit(db: Session, current_user: Optional[User]) -> None:
    """
    Restricts guest/unauthenticated users to a maximum of 5 chats.
    Requires signing in with Google or Email for unlimited ChatGPT Pro chats.
    """
    is_guest = (
        current_user is None
        or current_user.id.startswith("guest_")
        or (current_user.profile and (current_user.profile.get("is_anonymous") or current_user.profile.get("provider") in ("anonymous", "guest_fallback")))
        or "guest" in current_user.email.lower()
    )
    if is_guest and current_user:
        guest_msg_count = (
            db.query(Message)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .filter(Conversation.user_id == current_user.id, Message.role == "user")
            .count()
        )
        if guest_msg_count >= GUEST_MAX_CHATS:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Guest chat limit reached ({guest_msg_count}/{GUEST_MAX_CHATS} chats used). Please sign in with Google or Email to unlock unlimited ChatGPT Pro chats!",
            )


@router.post("/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    payload: ChatCompletionRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """
    Synchronous response generation using hosted LLM.
    """
    _check_guest_chat_limit(db, current_user)
    conv = ChatService.get_or_create_conversation(
        db, payload.conversation_id, user_id=current_user.id if current_user else None
    )
    user_msg = ChatService.save_user_message(db, conv.id, payload.message)

    sources = None
    if payload.mode == "rag":
        raw_sources = rag_retriever.retrieve(payload.message, top_k=3)
        sources = [
            SourceReference(
                rank=s["rank"],
                source=s["source"],
                page_number=s.get("page_number", 1),
                file_type=s.get("file_type", "txt"),
                score=s["score"],
                score_pct=s["score_pct"],
                text=s["text"],
            )
            for s in raw_sources
        ]
        context_text = "\n\n".join([f"--- [{s.source} | Page {s.page_number}] ---\n{s.text}" for s in sources])
        prompt_with_context = f"CONTEXT:\n{context_text}\n\nUSER QUESTION: {payload.message}"
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_with_context},
        ]
    else:
        sys_prompt = payload.system_prompt or conv.system_prompt or DEFAULT_SYSTEM_PROMPT
        messages = MemoryService.format_history_for_llm(conv.messages, sys_prompt)

    try:
        res = await llm_client.generate_response(
            messages=messages,
            temperature=payload.temperature or 0.7,
            top_p=payload.top_p or 0.9,
            max_tokens=payload.max_tokens or 1024,
        )
        content = res["content"]
    except Exception as e:
        content = f"I apologize, but an error occurred while generating a response: {str(e)}"

    assistant_msg = ChatService.save_assistant_message(
        db,
        conv.id,
        content,
        sources=[s.model_dump() for s in sources] if sources else None,
    )

    return ChatCompletionResponse(
        conversation_id=conv.id,
        message_id=assistant_msg.id,
        role="assistant",
        content=content,
        sources=sources,
        created_at=assistant_msg.created_at,
    )


@router.post("/stream")
async def stream_chat_completion(
    payload: ChatCompletionRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """
    Real-time Server-Sent Events (SSE) streaming endpoint.
    """
    _check_guest_chat_limit(db, current_user)
    conv = ChatService.get_or_create_conversation(
        db, payload.conversation_id, user_id=current_user.id if current_user else None
    )
    user_msg = ChatService.save_user_message(db, conv.id, payload.message)

    sources = None
    if payload.mode == "rag":
        raw_sources = rag_retriever.retrieve(payload.message, top_k=3)
        sources = [
            SourceReference(
                rank=s["rank"],
                source=s["source"],
                page_number=s.get("page_number", 1),
                file_type=s.get("file_type", "txt"),
                score=s["score"],
                score_pct=s["score_pct"],
                text=s["text"],
            )
            for s in raw_sources
        ]
        context_text = "\n\n".join([f"--- [{s.source} | Page {s.page_number}] ---\n{s.text}" for s in sources])
        prompt_with_context = f"CONTEXT:\n{context_text}\n\nUSER QUESTION: {payload.message}"
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_with_context},
        ]
    else:
        sys_prompt = payload.system_prompt or conv.system_prompt or DEFAULT_SYSTEM_PROMPT
        messages = MemoryService.format_history_for_llm(conv.messages, sys_prompt)

    async def sse_event_generator() -> AsyncIterator[str]:
        full_text = ""
        # Yield metadata first (conversation_id & sources)
        yield f"data: {json.dumps({'type': 'start', 'conversation_id': conv.id, 'sources': [s.model_dump() for s in sources] if sources else None})}\n\n"

        try:
            async for token in llm_client.stream_response(
                messages=messages,
                temperature=payload.temperature or 0.7,
                top_p=payload.top_p or 0.9,
                max_tokens=payload.max_tokens or 1024,
            ):
                full_text += token
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
        except Exception as err:
            err_msg = f"\n\n[Streaming Error: {str(err)}]"
            full_text += err_msg
            yield f"data: {json.dumps({'type': 'error', 'error': err_msg})}\n\n"

        # Save assistant message upon stream completion
        saved_msg = ChatService.save_assistant_message(
            db,
            conv.id,
            full_text,
            sources=[s.model_dump() for s in sources] if sources else None,
        )
        yield f"data: {json.dumps({'type': 'done', 'message_id': saved_msg.id, 'conversation_id': conv.id, 'content': full_text})}\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
    )

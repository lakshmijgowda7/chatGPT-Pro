"""
Conversations REST Endpoints
Handles listing, fetching, updating title, and deleting conversation sessions with optional user scoping.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.api.deps import get_optional_current_user
from app.models.user import User
from app.models.conversation import Conversation
from app.schemas.conversation import (
    ConversationSummary,
    ConversationDetail,
    ConversationCreate,
    ConversationUpdate,
)

router = APIRouter()


@router.get("", response_model=List[ConversationSummary])
def list_conversations(
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """Lists conversations ordered by most recently updated, scoped to current user if authenticated."""
    query = db.query(Conversation)
    if current_user:
        # Show conversations owned by this user
        query = query.filter(Conversation.user_id == current_user.id)
    
    convs = query.order_by(Conversation.updated_at.desc()).all()
    summaries = []
    for c in convs:
        summaries.append(
            ConversationSummary(
                id=c.id,
                title=c.title,
                message_count=len(c.messages),
                created_at=c.created_at,
                updated_at=c.updated_at,
            )
        )
    return summaries


@router.post("", response_model=ConversationSummary, status_code=status.HTTP_201_CREATED)
def create_conversation(
    payload: ConversationCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """Creates a new conversation session associated with the authenticated user if logged in."""
    conv = Conversation(
        user_id=current_user.id if current_user else None,
        title=payload.title or "New Chat",
        system_prompt=payload.system_prompt,
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return ConversationSummary(
        id=conv.id,
        title=conv.title,
        message_count=0,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """Retrieves full conversation with message history."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # If owned by another user and not public/guest
    if current_user and conv.user_id and conv.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Access denied to this conversation")

    return conv


@router.patch("/{conversation_id}", response_model=ConversationSummary)
def update_conversation_title(
    conversation_id: str,
    payload: ConversationUpdate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """Renames a conversation."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if current_user and conv.user_id and conv.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Access denied to this conversation")

    conv.title = payload.title.strip()
    db.commit()
    db.refresh(conv)
    return ConversationSummary(
        id=conv.id,
        title=conv.title,
        message_count=len(conv.messages),
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """Permanently deletes a conversation."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if current_user and conv.user_id and conv.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Access denied to this conversation")

    db.delete(conv)
    db.commit()
    return None

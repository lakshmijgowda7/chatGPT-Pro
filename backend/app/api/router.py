"""
Main API Router Aggregator
Combines health, auth, chat, conversations, documents, and settings endpoints into /api/v1.
"""

from fastapi import APIRouter
from app.api.v1.health import router as health_router
from app.api.v1.auth import router as auth_router
from app.api.v1.chat import router as chat_router
from app.api.v1.conversations import router as conversations_router
from app.api.v1.documents import router as documents_router
from app.api.v1.settings import router as settings_router

api_router = APIRouter()

api_router.include_router(health_router, tags=["Health"])
api_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
api_router.include_router(chat_router, prefix="/chat", tags=["Chat & Streaming"])
api_router.include_router(conversations_router, prefix="/conversations", tags=["Conversations"])
api_router.include_router(documents_router, prefix="/documents", tags=["Documents & RAG"])
api_router.include_router(settings_router, prefix="/settings", tags=["Settings"])

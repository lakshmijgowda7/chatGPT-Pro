"""
FastAPI Application Entrypoint
Project 3: Production-Ready ChatGPT-Style AI Platform Backend Server
"""

import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.core.logging import logger
from app.core.errors import (
    AppException,
    app_exception_handler,
    http_exception_handler,
    validation_exception_handler,
    generic_exception_handler,
)
from app.database.base import Base
from app.database.session import engine, check_database_connection
from app.database.migration import run_migrations
from app.api.router import api_router

# Import all SQLAlchemy models to ensure metadata registration
from app.models import conversation, message, document, user


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle manager.
    Initializes storage directories and auto-executes database migrations.
    """
    logger.info(f"Starting {settings.PROJECT_NAME} (Environment: {settings.ENVIRONMENT})...")
    os.makedirs(settings.UPLOAD_DIRECTORY, exist_ok=True)
    
    # Initialize / migrate database
    try:
        run_migrations()
        logger.info("Database migrations applied successfully.")
    except Exception as e:
        logger.warning(f"Alembic auto-migration notice: {e}. Ensuring tables via metadata...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables verified.")

    db_healthy = check_database_connection()
    logger.info(f"Database connectivity status: {'HEALTHY' if db_healthy else 'UNAVAILABLE'}")

    # Synchronize persistent vector index with any documents in database
    try:
        from app.database.session import get_db_context
        from app.services.document_service import DocumentService
        with get_db_context() as db:
            indexed_chunks = DocumentService.sync_existing_documents_to_vector_store(db)
            logger.info(f"Vector store ready ({indexed_chunks} active chunks).")
    except Exception as e:
        logger.warning(f"Vector store startup synchronization notice: {e}")
    
    yield
    logger.info("Shutting down backend server...")



app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production-Ready FastAPI Backend for ChatGPT-Style AI Platform with Hosted LLM & Document RAG",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ------------------------------------------------------------------------------
# Request Timing & Logging Middleware
# ------------------------------------------------------------------------------
@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.2f}ms)"
    )
    return response


# ------------------------------------------------------------------------------
# Structured Error Handlers
# ------------------------------------------------------------------------------
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)


# ------------------------------------------------------------------------------
# CORS Middleware Configuration
# ------------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# Mount Versioned API Router (/api/v1/...)
# ------------------------------------------------------------------------------
app.include_router(api_router, prefix=settings.API_V1_STR)


# ------------------------------------------------------------------------------
# Root & Direct Health Check Endpoints
# ------------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": "1.0.0",
        "docs_url": "/docs",
        "api_v1": settings.API_V1_STR,
    }


@app.get("/health", tags=["Health"])
def health_check():
    """
    Direct health check endpoint.
    """
    db_healthy = check_database_connection()
    return {
        "status": "healthy" if db_healthy else "degraded",
        "project": settings.PROJECT_NAME,
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "database": "connected" if db_healthy else "disconnected",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.LLM_MODEL,
        "docs_url": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=settings.DEBUG,
    )

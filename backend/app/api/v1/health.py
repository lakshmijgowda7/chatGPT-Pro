"""
Health Check & System Info Endpoints
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.schemas.health import HealthResponse, SystemInfoResponse
from app.core.config import settings
from app.api.deps import get_db

router = APIRouter()


@router.api_route(
    "/health",
    methods=["GET", "HEAD"],
    response_model=HealthResponse,
    summary="System Health Check",
    description="Returns the operational status, version, and database connectivity of the backend.",
)
def check_health(db: Session = Depends(get_db)):
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    return HealthResponse(
        status="healthy" if "unhealthy" not in db_status else "degraded",
        project=settings.PROJECT_NAME,
        version="1.0.0",
        environment=settings.ENVIRONMENT,
        database=db_status,
        llm_provider=settings.LLM_PROVIDER,
        llm_model=settings.LLM_MODEL,
        docs_url="/docs",
    )


@router.get(
    "/info",
    response_model=SystemInfoResponse,
    summary="System Configuration Info",
    description="Returns public system configuration metadata.",
)
def get_system_info():
    cors_origins = (
        settings.CORS_ORIGINS
        if isinstance(settings.CORS_ORIGINS, list)
        else [settings.CORS_ORIGINS]
    )
    return SystemInfoResponse(
        project_name=settings.PROJECT_NAME,
        environment=settings.ENVIRONMENT,
        debug_mode=settings.DEBUG,
        api_prefix=settings.API_V1_STR,
        database_type="sqlite" if settings.DATABASE_URL.startswith("sqlite") else "postgresql",
        cors_allowed_origins=cors_origins,
    )

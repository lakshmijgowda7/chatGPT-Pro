"""
Health Check Schemas
"""

from typing import Optional, Dict
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="API operational status")
    project: str = Field(..., description="Project name")
    version: str = Field(default="1.0.0", description="API Version")
    environment: str = Field(default="development", description="Runtime environment")
    database: str = Field(default="connected", description="Database connection health")
    llm_provider: Optional[str] = Field(default=None, description="Active LLM inference provider")
    llm_model: Optional[str] = Field(default=None, description="Active LLM model")
    docs_url: str = Field(default="/docs", description="Swagger OpenAPI Docs URL")


class SystemInfoResponse(BaseModel):
    project_name: str
    environment: str
    debug_mode: bool
    api_prefix: str
    database_type: str
    cors_allowed_origins: list[str]

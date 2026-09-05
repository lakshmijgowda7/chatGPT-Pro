"""
Backend Core Configuration
Loads environment variables using Pydantic Settings with support for PostgreSQL,
connection pooling, SQLite fallback, CORS, security, and hosted LLM inference parameters.
"""

from typing import List, Union
import json
import os
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Project & Server
    PROJECT_NAME: str = "LocalGPT Cloud AI Platform"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    API_V1_STR: str = "/api/v1"

    # Database Configuration (PostgreSQL primary; SQLite dev/test fallback)
    DATABASE_URL: str = "sqlite:///./platform.db"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 1800
    DB_ECHO: bool = False

    # Security & Authentication (No plain text secrets)
    SECRET_KEY: str = "localgpt-cloud-production-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # Google OAuth 2.0 Credentials & Redirect URLs
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/v1/auth/google/callback"
    GOOGLE_FRONTEND_REDIRECT_URI: str = "http://localhost:3000/auth/callback"

    # Firebase Authentication & Project Configuration
    FIREBASE_PROJECT_ID: str = ""

    # CORS Allowed Origins
    FRONTEND_URL: str = "http://localhost:3000"
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3002",
        "http://localhost:3003",
        "http://127.0.0.1:3003",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, str) and v.startswith("["):
            return json.loads(v)
        elif isinstance(v, list):
            return v
        return [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
            "http://localhost:3002",
            "http://127.0.0.1:3002",
            "http://localhost:8003",
            "http://localhost:8000",
        ]

    # Hosted LLM Inference Configuration
    LLM_PROVIDER: str = "groq"
    LLM_API_KEY: str = "demo-key"
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"
    LLM_MODEL: str = "qwen/qwen3.8-27b"
    LLM_TEMPERATURE: float = 0.7
    LLM_TOP_P: float = 0.9
    LLM_MAX_TOKENS: int = 2048

    # Document & RAG Storage
    UPLOAD_DIRECTORY: str = "./data/documents"
    MAX_UPLOAD_SIZE_MB: int = 25

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()

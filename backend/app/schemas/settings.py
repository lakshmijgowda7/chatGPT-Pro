"""
Pydantic Schemas for Platform Configuration & Settings
"""

from typing import Optional
from pydantic import BaseModel


class PlatformSettingsRead(BaseModel):
    project_name: str
    llm_provider: str
    llm_model: str
    llm_base_url: str
    masked_api_key: str
    default_temperature: float
    default_top_p: float
    default_max_tokens: int
    debug_mode: bool


class PlatformSettingsUpdate(BaseModel):
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    default_temperature: Optional[float] = None
    default_top_p: Optional[float] = None
    default_max_tokens: Optional[int] = None


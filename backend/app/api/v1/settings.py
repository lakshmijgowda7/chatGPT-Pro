"""
Platform Settings Endpoint
Exposes non-sensitive runtime configurations for client UI.
"""

from fastapi import APIRouter
from app.core.config import settings
from app.core.security import mask_api_key
from app.schemas.settings import PlatformSettingsRead, PlatformSettingsUpdate
from app.llm.client import llm_client

router = APIRouter()


@router.get("", response_model=PlatformSettingsRead)
def get_settings():
    """Returns platform metadata and masked LLM configuration."""
    return PlatformSettingsRead(
        project_name=settings.PROJECT_NAME,
        llm_provider=settings.LLM_PROVIDER,
        llm_model=settings.LLM_MODEL,
        llm_base_url=settings.LLM_BASE_URL,
        masked_api_key=mask_api_key(settings.LLM_API_KEY),
        default_temperature=settings.LLM_TEMPERATURE,
        default_top_p=settings.LLM_TOP_P,
        default_max_tokens=settings.LLM_MAX_TOKENS,
        debug_mode=settings.DEBUG,
    )


@router.patch("", response_model=PlatformSettingsRead)
def update_settings(update_data: PlatformSettingsUpdate):
    """Updates runtime platform LLM parameters and switches hosted provider."""
    if update_data.llm_provider is not None:
        settings.LLM_PROVIDER = update_data.llm_provider
    if update_data.llm_model is not None:
        settings.LLM_MODEL = update_data.llm_model
    if update_data.llm_base_url is not None:
        settings.LLM_BASE_URL = update_data.llm_base_url
    if update_data.llm_api_key is not None and update_data.llm_api_key.strip():
        settings.LLM_API_KEY = update_data.llm_api_key.strip()
    if update_data.default_temperature is not None:
        settings.LLM_TEMPERATURE = update_data.default_temperature
    if update_data.default_top_p is not None:
        settings.LLM_TOP_P = update_data.default_top_p
    if update_data.default_max_tokens is not None:
        settings.LLM_MAX_TOKENS = update_data.default_max_tokens

    # Dynamically update the global LLM client
    llm_client.update_configuration(
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL,
    )

    return get_settings()


@router.post("", response_model=PlatformSettingsRead, include_in_schema=False)
def update_settings_post(update_data: PlatformSettingsUpdate):
    """Alias for PATCH."""
    return update_settings(update_data=update_data)


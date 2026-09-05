"""
Standard Error Response Schemas
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ValidationErrorItem(BaseModel):
    field: str = Field(..., description="Field path that failed validation")
    message: str = Field(..., description="Description of the error")
    type: str = Field(default="value_error", description="Validation error code")


class ErrorResponse(BaseModel):
    status: str = Field(default="error", description="Error status indicator")
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error explanation")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional context or validation details")

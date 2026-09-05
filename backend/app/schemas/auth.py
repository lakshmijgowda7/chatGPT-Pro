"""
Authentication & User Schemas
Defines request payloads, response models, and validation rules for registration, login, and Google OAuth.
"""

from typing import Optional, Dict, Any
from datetime import datetime
import re
from pydantic import BaseModel, Field, field_validator, ConfigDict

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class UserBase(BaseModel):
    email: str = Field(..., description="User email address")
    name: Optional[str] = Field(None, max_length=255, description="Display name")
    full_name: Optional[str] = Field(None, max_length=255, description="Full name")
    is_active: bool = True
    is_superuser: bool = False

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        if not EMAIL_REGEX.match((v or "").strip()):
            raise ValueError("Invalid email address format")
        return v.strip().lower()


class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=128, description="Plain password (min 8 chars)")
    profile: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom profile metadata")

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class UserLogin(BaseModel):
    email: str = Field(..., description="Registered email address")
    password: str = Field(..., min_length=1, description="Account password")

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        if not EMAIL_REGEX.match((v or "").strip()):
            raise ValueError("Invalid email address format")
        return v.strip().lower()


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    name: Optional[str] = None
    full_name: Optional[str] = None
    profile: Optional[Dict[str, Any]] = None
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int
    user: UserResponse


class UserUpdate(BaseModel):
    name: Optional[str] = None
    full_name: Optional[str] = None
    profile: Optional[Dict[str, Any]] = None


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(..., min_length=1, description="Current account password")
    new_password: str = Field(..., min_length=8, max_length=128, description="New account password (min 8 chars)")

    @field_validator("new_password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("New password must be at least 8 characters long")
        return v



class GoogleAuthUrlResponse(BaseModel):
    url: str = Field(..., description="Google OAuth 2.0 authorization URL")
    state: str = Field(..., description="CSRF anti-forgery state token")


class GoogleAuthCallbackRequest(BaseModel):
    code: str = Field(..., description="Google OAuth authorization code")
    state: Optional[str] = Field(None, description="OAuth state parameter")
    redirect_uri: Optional[str] = Field(None, description="Custom redirect URI if applicable")


class ForgotPasswordRequest(BaseModel):
    email: EmailStr = Field(..., description="Registered user email address")


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., description="Password reset verification token")
    new_password: str = Field(..., min_length=8, max_length=128, description="New account password (min 8 chars)")

    @field_validator("new_password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("New password must be at least 8 characters long")
        return v


"""
FastAPI Dependencies
Provides reusable dependency injectors for database sessions, settings, and JWT authentication.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.core.config import settings, Settings
from app.core.security import decode_access_token
from app.models.user import User

from datetime import datetime, timezone
from app.core.firebase import verify_firebase_id_token

# OAuth2 password bearer scheme for extracting Authorization header: "Bearer <token>"
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login",
    auto_error=False,
)


def get_current_settings() -> Settings:
    """Dependency returning current application settings."""
    return settings


def _resolve_user_from_token(db: Session, token: str) -> Optional[User]:
    """
    Resolves a user from either a Firebase ID token (with auto-provisioning)
    or an internal HMAC JWT token.
    """
    # 1. Try Firebase Token verification
    fb_data = verify_firebase_id_token(token)
    if fb_data:
        fb_uid = fb_data["uid"]
        user_id = f"fb_{fb_uid}"
        email = fb_data["email"]
        user = db.query(User).filter((User.id == user_id) | (User.email == email)).first()
        if not user:
            # Auto-provision user account from Firebase profile
            now = datetime.now(timezone.utc)
            user = User(
                id=user_id,
                email=email,
                hashed_password="firebase_external_auth",
                name=fb_data.get("name") or "Firebase User",
                full_name=fb_data.get("name"),
                profile={
                    "avatar_url": fb_data.get("picture"),
                    "provider": fb_data.get("sign_in_provider", "firebase"),
                    "is_anonymous": fb_data.get("is_anonymous", False),
                },
                is_active=True,
                created_at=now,
                updated_at=now,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        return user if user.is_active else None

    # 2. Try Local JWT
    payload = decode_access_token(token)
    if payload and "sub" in payload:
        user_id = payload.get("sub")
        user = db.query(User).filter(User.id == user_id).first()
        if user and user.is_active:
            return user

    return None


def get_current_user(
    db: Session = Depends(get_db),
    token: Optional[str] = Depends(oauth2_scheme),
) -> User:
    """
    Guards protected endpoints.
    Extracts, decodes, and validates JWT or Firebase Bearer token, returning the authenticated User.
    Raises 401 Unauthorized if missing, expired, or invalid.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = _resolve_user_from_token(db, token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials: Token is invalid, expired, or deactivated.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Verifies that the authenticated user is currently active.
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )
    return current_user


def get_optional_current_user(
    db: Session = Depends(get_db),
    token: Optional[str] = Depends(oauth2_scheme),
) -> Optional[User]:
    """
    Optional authentication dependency.
    Returns the User entity if a valid token (Firebase or local) is provided, or None for guest requests.
    """
    if not token:
        return None

    return _resolve_user_from_token(db, token)

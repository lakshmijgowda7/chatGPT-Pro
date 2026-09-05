"""
Authentication Service
Encapsulates user registration, password verification, Google OAuth 2.0 integration, and JWT session generation.
"""

import secrets
import urllib.parse
from typing import Optional, Dict, Any, Tuple
import httpx
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from app.models.user import User
from app.schemas.auth import UserCreate, UserResponse, TokenResponse, UserUpdate, PasswordChangeRequest
from app.core.security import create_access_token, get_password_hash

from app.core.config import settings
from app.core.logging import logger


class AuthService:
    @staticmethod
    def register_user(db: Session, user_in: UserCreate) -> User:
        """
        Registers a new user after verifying email uniqueness and hashing the password.
        """
        normalized_email = user_in.email.strip().lower()
        
        # Check for existing account
        existing_user = db.query(User).filter(User.email == normalized_email).first()
        if existing_user:
            logger.warning(f"Registration rejected: Email '{normalized_email}' is already registered.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="An account with this email address already exists.",
            )

        # Fallback names
        display_name = user_in.name or user_in.full_name or normalized_email.split("@")[0]
        full_name = user_in.full_name or user_in.name or display_name

        user = User(
            email=normalized_email,
            name=display_name,
            full_name=full_name,
            profile=user_in.profile or {},
            is_active=True,
            is_superuser=False,
        )
        user.set_password(user_in.password)

        db.add(user)
        db.commit()
        db.refresh(user)

        logger.info(f"New user registered: {user.email} (id: {user.id})")
        return user

    @staticmethod
    def authenticate_user(db: Session, email: str, password: str) -> User:
        """
        Validates user credentials against stored bcrypt password hashes.
        """
        normalized_email = email.strip().lower()
        user = db.query(User).filter(User.email == normalized_email).first()

        if not user or not user.check_password(password):
            logger.warning(f"Failed login attempt for: {normalized_email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not user.is_active:
            logger.warning(f"Login attempt by inactive account: {normalized_email}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is deactivated. Please contact support.",
            )

        logger.info(f"User authenticated successfully: {user.email} (id: {user.id})")
        return user

    @staticmethod
    def generate_google_auth_url(
        state: Optional[str] = None,
        redirect_uri: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Constructs the Google OAuth 2.0 authorization URL with anti-CSRF state token.
        """
        if not settings.GOOGLE_CLIENT_ID:
            logger.error("GOOGLE_CLIENT_ID is not configured in backend environment.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Google OAuth is not configured. Missing GOOGLE_CLIENT_ID.",
            )

        state_token = state or secrets.token_urlsafe(32)
        target_redirect_uri = redirect_uri or settings.GOOGLE_REDIRECT_URI

        params = {
            "client_id": settings.GOOGLE_CLIENT_ID,
            "redirect_uri": target_redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "access_type": "offline",
            "state": state_token,
            "prompt": "consent",
        }

        query_string = urllib.parse.urlencode(params)
        auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{query_string}"
        return auth_url, state_token

    @staticmethod
    async def exchange_google_code_for_userinfo(
        code: str,
        redirect_uri: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Exchanges Google authorization code for access token and fetches user profile.
        """
        if not settings.GOOGLE_CLIENT_ID or not settings.GOOGLE_CLIENT_SECRET:
            logger.error("Google OAuth credentials missing.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Google OAuth client credentials are not configured on the backend.",
            )

        target_redirect = redirect_uri or settings.GOOGLE_REDIRECT_URI
        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            "code": code,
            "client_id": settings.GOOGLE_CLIENT_ID,
            "client_secret": settings.GOOGLE_CLIENT_SECRET,
            "redirect_uri": target_redirect,
            "grant_type": "authorization_code",
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                token_res = await client.post(token_url, data=token_data)
                
                if token_res.status_code != 200:
                    logger.warning(f"Google Token exchange failed: {token_res.status_code} - {token_res.text}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to exchange Google OAuth code: {token_res.text}",
                    )

                token_json = token_res.json()
                access_token = token_json.get("access_token")
                if not access_token:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Google token response did not contain access_token.",
                    )

                # Fetch user profile using access token
                userinfo_url = "https://www.googleapis.com/oauth2/v3/userinfo"
                userinfo_res = await client.get(
                    userinfo_url,
                    headers={"Authorization": f"Bearer {access_token}"},
                )

                if userinfo_res.status_code != 200:
                    logger.warning(f"Failed to fetch Google userinfo: {userinfo_res.status_code} - {userinfo_res.text}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Failed to retrieve Google user profile.",
                    )

                return userinfo_res.json()

        except httpx.RequestError as exc:
            logger.error(f"HTTP communication error with Google OAuth servers: {exc}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Unable to connect to Google authentication service: {str(exc)}",
            )

    @staticmethod
    def authenticate_google_user(db: Session, google_info: Dict[str, Any]) -> User:
        """
        Connects Google profile to the User database, creating a new account or
        linking with an existing account to prevent duplicates.
        """
        email = (google_info.get("email") or "").strip().lower()
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Google OAuth profile does not contain an email address.",
            )

        google_id = str(google_info.get("sub") or "")
        name = google_info.get("name") or google_info.get("given_name") or email.split("@")[0]
        picture = google_info.get("picture")

        # Check if user already exists
        user = db.query(User).filter(User.email == email).first()

        if user:
            # Connect / update existing user profile
            current_profile = dict(user.profile or {})
            current_profile["google_id"] = google_id
            if picture:
                current_profile["avatar_url"] = picture
            current_profile["auth_provider"] = "google"
            user.profile = current_profile

            if not user.name and name:
                user.name = name
            if not user.is_active:
                user.is_active = True

            db.commit()
            db.refresh(user)
            logger.info(f"Existing user connected via Google OAuth: {user.email} (id: {user.id})")
        else:
            # Create new user record
            random_pwd = secrets.token_urlsafe(32)
            profile_data = {
                "google_id": google_id,
                "avatar_url": picture,
                "auth_provider": "google",
            }

            user = User(
                email=email,
                name=name,
                full_name=name,
                profile=profile_data,
                is_active=True,
                is_superuser=False,
            )
            # Store random hashed password so user cannot login with empty password
            user.set_password(random_pwd)

            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"New user created via Google OAuth: {user.email} (id: {user.id})")

        return user

    @staticmethod
    def create_user_token(user: User) -> TokenResponse:
        """
        Generates a signed JWT access token for the authenticated user.
        """
        token_payload = {
            "sub": user.id,
            "email": user.email,
            "name": user.name or "",
            "is_superuser": user.is_superuser,
        }
        access_token = create_access_token(data=token_payload)
        expires_seconds = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in_seconds=expires_seconds,
            user=UserResponse.model_validate(user),
        )

    @staticmethod
    def update_user_profile(db: Session, user: User, update_in: UserUpdate) -> User:
        """
        Updates profile fields (display name, full name, custom profile json) for an existing user.
        """
        if update_in.name is not None:
            user.name = update_in.name.strip()
        if update_in.full_name is not None:
            user.full_name = update_in.full_name.strip()
        if update_in.profile is not None:
            merged_profile = dict(user.profile or {})
            merged_profile.update(update_in.profile)
            user.profile = merged_profile

        db.commit()
        db.refresh(user)
        logger.info(f"User profile updated: {user.email} (id: {user.id})")
        return user

    @staticmethod
    def change_password(db: Session, user: User, change_req: PasswordChangeRequest) -> None:
        """
        Verifies current password and updates to a new hashed password.
        """
        if not user.check_password(change_req.current_password):
            logger.warning(f"Password change rejected: Incorrect current password for {user.email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password verification failed.",
            )

        user.set_password(change_req.new_password)
        db.commit()
        db.refresh(user)
        logger.info(f"Password changed successfully for user: {user.email}")

    @staticmethod
    def generate_password_reset_token(db: Session, email: str) -> str:
        """
        Generates a secure, signed password reset token valid for 30 minutes.
        """
        normalized_email = email.strip().lower()
        user = db.query(User).filter(User.email == normalized_email).first()
        if not user:
            logger.warning(f"Password reset requested for non-existent email: {normalized_email}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No account found with this email address.",
            )

        from datetime import timedelta
        token_payload = {
            "sub": user.id,
            "email": user.email,
            "purpose": "password_reset",
        }
        reset_token = create_access_token(data=token_payload, expires_delta=timedelta(minutes=30))
        logger.info(f"Password reset token issued for user: {user.email}")
        return reset_token

    @staticmethod
    def reset_password_with_token(db: Session, token: str, new_password: str) -> User:
        """
        Validates the password reset token and updates the user's password.
        """
        from app.core.security import decode_access_token
        payload = decode_access_token(token)
        if not payload or payload.get("purpose") != "password_reset":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired password reset token.",
            )

        user_id = payload.get("sub")
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User account no longer exists.",
            )

        user.set_password(new_password)
        db.commit()
        db.refresh(user)
        logger.info(f"Password successfully reset via token for user: {user.email}")
        return user



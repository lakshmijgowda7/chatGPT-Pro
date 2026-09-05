"""
Authentication REST Endpoints
Provides registration, login, profile retrieval, session termination, and Google OAuth 2.0 flows.
"""

from typing import Optional
import urllib.parse
from fastapi import APIRouter, Depends, Query, Request, HTTPException, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from app.schemas.auth import (
    UserCreate,
    UserLogin,
    UserResponse,
    TokenResponse,
    UserUpdate,
    PasswordChangeRequest,
    GoogleAuthUrlResponse,
    GoogleAuthCallbackRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
)
from app.services.auth_service import AuthService
from app.models.user import User
from app.api.deps import get_db, get_current_user
from app.core.config import settings
from app.core.logging import logger

router = APIRouter()


# ------------------------------------------------------------------------------
# Standard Password Authentication Endpoints
# ------------------------------------------------------------------------------

@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="User Registration",
    description="Registers a new user account with hashed password and returns an access token.",
)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    """Creates a new user and generates an initial JWT session token."""
    user = AuthService.register_user(db=db, user_in=user_in)
    return AuthService.create_user_token(user)


@router.post(
    "/signup",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="User Signup (Alias)",
    description="Alias endpoint for user registration.",
    include_in_schema=False,
)
def signup(user_in: UserCreate, db: Session = Depends(get_db)):
    """Alias for /register."""
    return register(user_in=user_in, db=db)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="User Login",
    description="Authenticates credentials and returns a signed JWT Bearer access token.",
)
def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Verifies email and password, returning JWT token and user profile."""
    user = AuthService.authenticate_user(
        db=db,
        email=credentials.email,
        password=credentials.password,
    )
    return AuthService.create_user_token(user)


# ------------------------------------------------------------------------------
# Google OAuth 2.0 Endpoints
# ------------------------------------------------------------------------------

@router.get(
    "/google/url",
    response_model=GoogleAuthUrlResponse,
    summary="Get Google OAuth URL",
    description="Generates the Google OAuth 2.0 authorization URL with anti-CSRF state token.",
)
def get_google_auth_url(
    state: Optional[str] = None,
    redirect_uri: Optional[str] = None,
):
    """Returns Google authorization URL for frontend clients."""
    auth_url, state_token = AuthService.generate_google_auth_url(
        state=state,
        redirect_uri=redirect_uri,
    )
    return GoogleAuthUrlResponse(url=auth_url, state=state_token)


@router.get(
    "/google/login",
    summary="Direct Google OAuth Redirect",
    description="Redirects browser directly to Google OAuth 2.0 authorization screen.",
)
def google_login(
    state: Optional[str] = None,
    redirect_uri: Optional[str] = None,
):
    """Direct 307 Redirect to Google consent screen."""
    auth_url, _ = AuthService.generate_google_auth_url(
        state=state,
        redirect_uri=redirect_uri,
    )
    return RedirectResponse(url=auth_url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@router.get(
    "/google/callback",
    summary="Google OAuth Browser Callback",
    description="Receives authorization code from Google, authenticates user, and redirects to frontend with token.",
)
async def google_callback_redirect(
    code: Optional[str] = Query(None, description="Google OAuth authorization code"),
    state: Optional[str] = Query(None, description="OAuth state parameter"),
    error: Optional[str] = Query(None, description="OAuth error code from Google"),
    error_description: Optional[str] = Query(None, description="OAuth error message"),
    db: Session = Depends(get_db),
):
    """Handles Google OAuth redirect flow and routes user back to frontend UI with session token."""
    frontend_base = settings.GOOGLE_FRONTEND_REDIRECT_URI.rstrip("/")

    if error:
        logger.warning(f"Google OAuth rejected by user or server: {error} - {error_description}")
        return RedirectResponse(
            url=f"{frontend_base}?error={urllib.parse.quote(error_description or error)}",
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
        )

    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Google authorization code in callback.",
        )

    google_info = await AuthService.exchange_google_code_for_userinfo(code=code)
    user = AuthService.authenticate_google_user(db=db, google_info=google_info)
    token_resp = AuthService.create_user_token(user)

    # Redirect user to frontend with token parameter
    redirect_target = f"{frontend_base}?token={token_resp.access_token}"
    return RedirectResponse(url=redirect_target, status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@router.post(
    "/google",
    response_model=TokenResponse,
    summary="Google OAuth Code Exchange (JSON API)",
    description="Exchanges Google authorization code for JWT token via JSON request body.",
)
async def google_auth_json(
    payload: GoogleAuthCallbackRequest,
    db: Session = Depends(get_db),
):
    """Direct API exchange for SPA / client-side Google auth."""
    if not payload.code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authorization code is required.",
        )

    google_info = await AuthService.exchange_google_code_for_userinfo(
        code=payload.code,
        redirect_uri=payload.redirect_uri,
    )
    user = AuthService.authenticate_google_user(db=db, google_info=google_info)
    return AuthService.create_user_token(user)


# ------------------------------------------------------------------------------
# Protected User & Session Endpoints
# ------------------------------------------------------------------------------

@router.get(
    "/me",
    response_model=UserResponse,
    summary="Current Authenticated User",
    description="Protected endpoint returning the profile of the currently logged-in user.",
)
def get_me(current_user: User = Depends(get_current_user)):
    """Returns currently authenticated user profile."""
    return current_user


@router.patch(
    "/me",
    response_model=UserResponse,
    summary="Update User Profile",
    description="Updates display name, full name, or custom profile data for the authenticated user.",
)
def update_me(
    update_in: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Updates authenticated user's profile information."""
    updated_user = AuthService.update_user_profile(
        db=db,
        user=current_user,
        update_in=update_in,
    )
    return updated_user


@router.put(
    "/me",
    response_model=UserResponse,
    summary="Update User Profile (Alias)",
    description="Alias for PATCH /me.",
    include_in_schema=False,
)
def update_me_put(
    update_in: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Alias for PATCH /me."""
    return update_me(update_in=update_in, db=db, current_user=current_user)


@router.post(
    "/change-password",
    summary="Change Account Password",
    description="Changes password for authenticated user after verifying current password.",
)
def change_password(
    change_req: PasswordChangeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Changes password after verifying existing credentials."""
    AuthService.change_password(
        db=db,
        user=current_user,
        change_req=change_req,
    )
    return {
        "success": True,
        "message": "Password updated successfully.",
    }


@router.post(
    "/forgot-password",
    summary="Request Password Reset",
    description="Generates a secure password reset token for the specified user email address.",
)
def forgot_password(
    req: ForgotPasswordRequest,
    db: Session = Depends(get_db),
):
    """Requests a password reset token for the given email address."""
    token = AuthService.generate_password_reset_token(db=db, email=req.email)
    return {
        "success": True,
        "message": "Password reset token generated successfully.",
        "reset_token": token,
    }


@router.post(
    "/reset-password",
    summary="Reset Password with Token",
    description="Resets the account password using a valid, signed reset token.",
)
def reset_password(
    req: ResetPasswordRequest,
    db: Session = Depends(get_db),
):
    """Resets password using verification token."""
    user = AuthService.reset_password_with_token(db=db, token=req.token, new_password=req.new_password)
    return {
        "success": True,
        "message": "Password reset completed successfully. You can now log in.",
        "email": user.email,
    }


@router.post(
    "/logout",
    summary="User Logout",
    description="Client session logout acknowledgement.",
)
def logout(current_user: User = Depends(get_current_user)):
    """Logs out authenticated session."""
    return {
        "success": True,
        "message": f"Session terminated for user {current_user.email}.",
    }



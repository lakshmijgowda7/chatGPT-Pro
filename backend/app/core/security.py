"""
Security & Password Hashing Module
Ensures secure hashing and verification of credentials. Secrets are NEVER stored in plain text.
Provides JWT access token creation and decoding with HS256 signature verification.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import bcrypt
import jwt
from app.core.config import settings
from app.core.logging import logger


def get_password_hash(password: str) -> str:
    """
    Hashes a plain text password using bcrypt with automated salt generation.
    Returns the decoded UTF-8 hash string suitable for database storage.
    """
    if not isinstance(password, str):
        password = str(password)
    # Truncate to bcrypt's 72 byte limit for safety
    pwd_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plain text password against a stored bcrypt hash string.
    Returns True if valid, False otherwise.
    """
    if not plain_password or not hashed_password:
        return False
    try:
        pwd_bytes = plain_password.encode("utf-8")[:72]
        hash_bytes = hashed_password.encode("utf-8")
        return bcrypt.checkpw(pwd_bytes, hash_bytes)
    except Exception:
        return False


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Generates a cryptographically signed JWT access token.
    """
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )
    return encoded_jwt


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decodes and validates a JWT token against secret key and algorithm.
    Returns payload dictionary if valid, or None if invalid/expired.
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("JWT Token has expired.")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT Token: {e}")
        return None
    except Exception as e:
        logger.error(f"JWT decode error: {e}")
        return None


def mask_api_key(api_key: str, visible_prefix: int = 4, visible_suffix: int = 4) -> str:
    """
    Masks a sensitive API key for safe public display (e.g. gsk_123...4567).
    """
    if not api_key or "placeholder" in api_key.lower():
        return "No key set"
    if len(api_key) <= (visible_prefix + visible_suffix):
        return "*" * len(api_key)
    return f"{api_key[:visible_prefix]}...{api_key[-visible_suffix:]}"

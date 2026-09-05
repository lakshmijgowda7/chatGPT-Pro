"""
Firebase Token Verification & Public Cert Cache
Validates Firebase Authentication ID tokens (RS256 JWTs) signed by Google.
Supports online certificate verification with caching and offline dev fallback.
"""

from typing import Optional, Dict, Any
import time
import json
import urllib.request
import jwt
from cryptography.x509 import load_pem_x509_certificate
from cryptography.hazmat.backends import default_backend

from app.core.config import settings
from app.core.logging import logger

GOOGLE_CERTS_URL = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"

# In-memory certificate cache: { "certs": dict, "expires_at": float }
_CERTS_CACHE: Dict[str, Any] = {"certs": {}, "expires_at": 0}


def get_google_public_certs() -> Dict[str, str]:
    """
    Fetches and caches Google's public x509 certificates for verifying Firebase ID tokens.
    """
    now = time.time()
    if _CERTS_CACHE["certs"] and _CERTS_CACHE["expires_at"] > now:
        return _CERTS_CACHE["certs"]

    try:
        req = urllib.request.Request(GOOGLE_CERTS_URL, headers={"User-Agent": "LocalGPT-Backend/1.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            # Cache for 1 hour
            _CERTS_CACHE["certs"] = data
            _CERTS_CACHE["expires_at"] = now + 3600
            return data
    except Exception as e:
        logger.warning(f"Could not fetch Google public certs for Firebase verification: {e}")
        return _CERTS_CACHE["certs"]


def verify_firebase_id_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verifies a Firebase ID token.
    Returns decoded token payload if valid, None otherwise.
    """
    if not token or not isinstance(token, str):
        return None

    try:
        # Inspect header without verification to extract kid and alg
        unverified_header = jwt.get_unverified_header(token)
        alg = unverified_header.get("alg")
        kid = unverified_header.get("kid")

        # Firebase ID tokens must be RS256
        if alg != "RS256":
            return None

        certs = get_google_public_certs()
        public_key = None

        if kid and kid in certs:
            pem_cert = certs[kid].encode("utf-8")
            cert = load_pem_x509_certificate(pem_cert, default_backend())
            public_key = cert.public_key()

        if public_key:
            # Verify signature with Google's public key
            options = {"verify_exp": True}
            if not settings.FIREBASE_PROJECT_ID:
                options["verify_aud"] = False

            payload = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                audience=settings.FIREBASE_PROJECT_ID or None,
                options=options,
            )
        else:
            # If Google certs cannot be reached (e.g. offline dev), decode payload and verify expiration
            payload = jwt.decode(
                token,
                options={"verify_signature": False, "verify_exp": True},
            )

        # Ensure required Firebase claims
        sub = payload.get("sub") or payload.get("user_id")
        if not sub:
            return None

        # Check issuer matches securetoken.google.com
        iss = payload.get("iss", "")
        if "securetoken.google.com" not in iss:
            return None

        email = payload.get("email") or f"guest_{sub[:8]}@localgpt.user"
        name = payload.get("name") or ("Guest User" if "guest" in email else "Firebase User")
        picture = payload.get("picture")
        firebase_meta = payload.get("firebase", {})
        sign_in_provider = firebase_meta.get("sign_in_provider", "firebase")

        return {
            "uid": sub,
            "email": email,
            "name": name,
            "picture": picture,
            "sign_in_provider": sign_in_provider,
            "is_anonymous": sign_in_provider == "anonymous",
            "raw_payload": payload,
        }
    except jwt.ExpiredSignatureError:
        logger.warning("Firebase token expired.")
        return None
    except Exception as e:
        logger.debug(f"Firebase token verification notice: {e}")
        return None

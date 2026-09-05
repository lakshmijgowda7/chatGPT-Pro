"""
Unit tests for Firebase Token Verification and User Auto-Provisioning
"""

import pytest
from datetime import datetime, timezone
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

from app.core.firebase import verify_firebase_id_token
from app.api.deps import _resolve_user_from_token
from app.database.session import SessionLocal
from app.models.user import User


@pytest.fixture
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_verify_firebase_token_invalid():
    assert verify_firebase_id_token("not-a-token") is None
    assert verify_firebase_id_token("") is None
    assert verify_firebase_id_token(None) is None


def test_firebase_user_resolution_and_autoprovision(db_session, monkeypatch):
    test_uid = "test_fb_user_12345"
    test_email = "testfirebase@localgpt.example.com"
    test_name = "Firebase Test User"

    # Mock verify_firebase_id_token
    monkeypatch.setattr(
        "app.api.deps.verify_firebase_id_token",
        lambda token: {
            "uid": test_uid,
            "email": test_email,
            "name": test_name,
            "picture": "https://example.com/avatar.png",
            "sign_in_provider": "google.com",
            "is_anonymous": False,
        } if token == "mock-valid-firebase-token" else None
    )

    # Clean up prior test user if exists
    existing = db_session.query(User).filter(User.id == f"fb_{test_uid}").first()
    if existing:
        db_session.delete(existing)
        db_session.commit()

    # 1. Resolve user for the first time -> should auto-provision
    user = _resolve_user_from_token(db_session, "mock-valid-firebase-token")
    assert user is not None
    assert user.id == f"fb_{test_uid}"
    assert user.email == test_email
    assert user.name == test_name
    assert user.is_active is True

    # 2. Resolve user again -> should find existing
    user_second = _resolve_user_from_token(db_session, "mock-valid-firebase-token")
    assert user_second is not None
    assert user_second.id == user.id

    # Clean up
    db_session.delete(user)
    db_session.commit()

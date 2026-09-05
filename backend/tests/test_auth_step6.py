"""
Comprehensive Test Suite for Step 6: User Authentication
Tests:
1. User Registration (POST /api/v1/auth/register) -> 201 Created + Token
2. Registration Alias (POST /api/v1/auth/signup) -> 201 Created
3. Duplicate Email Registration Rejection -> 400 Bad Request
4. Input Validation (invalid email, password < 8 chars) -> 422 Unprocessable Entity
5. User Login Success (POST /api/v1/auth/login) -> 200 OK + JWT Token
6. User Login Rejection with Wrong Password -> 401 Unauthorized
7. Protected Endpoint (GET /api/v1/auth/me) with Valid Token -> 200 OK
8. Protected Endpoint Rejection without Token -> 401 Unauthorized
9. Protected Endpoint Rejection with Invalid/Forged Token -> 401 Unauthorized
10. User Logout (POST /api/v1/auth/logout) -> 200 OK
11. Inaccessibility confirmation without active token
"""

import os
import sys
import time
from starlette.testclient import TestClient

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.database.session import get_db_context
from app.models.user import User

client = TestClient(app)


def test_user_registration_success():
    """Verify new user registration returns HTTP 201 and valid access token."""
    email = f"register_{int(time.time() * 1000)}@example.com"
    payload = {
        "email": email,
        "password": "SecurePassword123!",
        "name": "Jane Developer",
        "full_name": "Jane Developer",
        "profile": {"role": "Engineer", "theme": "dark"},
    }
    
    response = client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 201, f"Registration failed: {response.text}"
    
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == email
    assert data["user"]["name"] == "Jane Developer"
    assert data["user"]["id"].startswith("usr_")
    assert "password" not in data["user"]
    assert "hashed_password" not in data["user"]


def test_user_signup_alias():
    """Verify /signup alias works identical to /register."""
    email = f"signup_alias_{int(time.time() * 1000)}@example.com"
    payload = {
        "email": email,
        "password": "SecurePassword123!",
        "name": "Alias User",
    }
    
    response = client.post("/api/v1/auth/signup", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["user"]["email"] == email


def test_duplicate_registration_rejected():
    """Verify registering an existing email returns HTTP 400 Bad Request."""
    email = f"dup_{int(time.time() * 1000)}@example.com"
    payload = {
        "email": email,
        "password": "StrongPassword999!",
        "name": "First User",
    }
    
    # First registration
    res1 = client.post("/api/v1/auth/register", json=payload)
    assert res1.status_code == 201

    # Duplicate registration attempt
    res2 = client.post("/api/v1/auth/register", json=payload)
    assert res2.status_code == 400
    assert "already exists" in res2.json()["detail"]


def test_registration_validation_errors():
    """Verify input validation: invalid email or password < 8 characters -> HTTP 422."""
    # Short password
    res_short_pwd = client.post(
        "/api/v1/auth/register",
        json={"email": "valid@example.com", "password": "short"},
    )
    assert res_short_pwd.status_code == 422

    # Invalid email
    res_invalid_email = client.post(
        "/api/v1/auth/register",
        json={"email": "not-an-email", "password": "ValidPassword123!"},
    )
    assert res_invalid_email.status_code == 422


def test_login_success():
    """Verify user login with valid credentials returns JWT access token."""
    email = f"login_test_{int(time.time() * 1000)}@example.com"
    password = "CorrectPassword123!"
    
    # Register first
    client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password, "name": "Login User"},
    )

    # Login
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == email


def test_login_invalid_credentials():
    """Verify login with wrong password or unknown email returns HTTP 401."""
    email = f"wrong_pwd_{int(time.time() * 1000)}@example.com"
    client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "CorrectPassword123!"},
    )

    # Wrong password
    res_wrong_pwd = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "WrongPassword!"},
    )
    assert res_wrong_pwd.status_code == 401
    assert "Invalid email or password" in res_wrong_pwd.json()["detail"]

    # Unknown email
    res_unknown_email = client.post(
        "/api/v1/auth/login",
        json={"email": "nonexistent_9999@example.com", "password": "AnyPassword123!"},
    )
    assert res_unknown_email.status_code == 401


def test_protected_endpoint_me_with_valid_token():
    """Verify accessing protected endpoint GET /api/v1/auth/me with Bearer token succeeds."""
    email = f"me_test_{int(time.time() * 1000)}@example.com"
    password = "MePassword123!"
    
    reg_res = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password, "name": "Profile Owner"},
    )
    token = reg_res.json()["access_token"]

    # Call protected endpoint
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    user_data = response.json()
    assert user_data["email"] == email
    assert user_data["name"] == "Profile Owner"


def test_protected_endpoint_rejection_without_token():
    """Verify GET /api/v1/auth/me rejects unauthenticated requests with HTTP 401."""
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401
    assert "credentials were not provided" in response.json()["detail"]


def test_protected_endpoint_rejection_with_invalid_token():
    """Verify GET /api/v1/auth/me rejects forged/malformed tokens with HTTP 401."""
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.fake.payload"},
    )
    assert response.status_code == 401


def test_logout_flow_and_inaccessibility():
    """Verify logout endpoint and confirm protected endpoints require valid token."""
    email = f"logout_test_{int(time.time() * 1000)}@example.com"
    password = "LogoutPassword123!"
    
    reg_res = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password, "name": "Logout Tester"},
    )
    token = reg_res.json()["access_token"]

    # 1. Call protected endpoint before logout -> success
    me_before = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_before.status_code == 200

    # 2. Call logout endpoint -> success
    logout_res = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert logout_res.status_code == 200
    assert logout_res.json()["success"] is True

    # 3. Simulate client deleting token after logout -> request without token fails (401)
    me_after = client.get("/api/v1/auth/me")
    assert me_after.status_code == 401


if __name__ == "__main__":
    print("\n=======================================================")
    print(" Running Step 6: User Authentication Test Suite")
    print("=======================================================\n")

    print("[1/10] Testing User Registration...")
    test_user_registration_success()
    print("  -> PASSED: Registration returns 201 Created and JWT token.")

    print("[2/10] Testing Signup Alias...")
    test_user_signup_alias()
    print("  -> PASSED: /signup alias works properly.")

    print("[3/10] Testing Duplicate Registration Rejection...")
    test_duplicate_registration_rejected()
    print("  -> PASSED: Duplicate email returns 400 Bad Request.")

    print("[4/10] Testing Input Validation...")
    test_registration_validation_errors()
    print("  -> PASSED: Invalid email or short password rejected with 422.")

    print("[5/10] Testing User Login Success...")
    test_login_success()
    print("  -> PASSED: Valid login returns JWT token.")

    print("[6/10] Testing Login Invalid Credentials...")
    test_login_invalid_credentials()
    print("  -> PASSED: Wrong credentials return 401 Unauthorized.")

    print("[7/10] Testing Protected Endpoint /me with Token...")
    test_protected_endpoint_me_with_valid_token()
    print("  -> PASSED: Protected endpoint returns authenticated user profile.")

    print("[8/10] Testing Protected Endpoint Rejection without Token...")
    test_protected_endpoint_rejection_without_token()
    print("  -> PASSED: Missing token returns 401 Unauthorized.")

    print("[9/10] Testing Protected Endpoint Rejection with Forged Token...")
    test_protected_endpoint_rejection_with_invalid_token()
    print("  -> PASSED: Invalid token returns 401 Unauthorized.")

    print("[10/10] Testing Logout Flow & Protected Access Control...")
    test_logout_flow_and_inaccessibility()
    print("  -> PASSED: Logout succeeds and unauthorized access is blocked.")

    print("\n=======================================================")
    print(" ALL 10 AUTHENTICATION TESTS PASSED SUCCESSFULLY! ")
    print("=======================================================\n")

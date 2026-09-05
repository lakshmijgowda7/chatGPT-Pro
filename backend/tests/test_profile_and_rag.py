"""
Automated Integration Tests for Phase 3 Upgrades
Tests:
1. Profile update (PATCH /api/v1/auth/me)
2. Password change with verification (POST /api/v1/auth/change-password)
3. CSV and JSON document loaders
4. Persistent FAISS vector retriever search, deletion, and reload
5. End-to-end document upload & delete API endpoints
"""

import os
import uuid
import json
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database.base import Base
from app.database.session import get_db as session_get_db
from app.api.deps import get_db as deps_get_db
from app.rag.loader import extract_text_from_csv, extract_text_from_json, load_document
from app.rag.retriever import PersistentVectorRetriever
from app.rag.splitter import split_document

TEST_DB_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=test_engine)
    app.dependency_overrides[session_get_db] = override_get_db
    app.dependency_overrides[deps_get_db] = override_get_db
    yield
    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()


@pytest.fixture
def client():
    return TestClient(app)


def test_user_registration_profile_and_password_change(client):
    """Verifies registration, updating profile name, and changing password."""
    unique_id = uuid.uuid4().hex[:8]
    email = f"profile_user_{unique_id}@example.com"
    old_password = "InitialPassword123!"
    new_password = "UpdatedPassword456!"

    # 1. Register user
    reg_resp = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": old_password, "name": "Initial Name"},
    )
    assert reg_resp.status_code == 201
    token = reg_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Update profile name via PATCH /me
    patch_resp = client.patch(
        "/api/v1/auth/me",
        headers=headers,
        json={"name": "Jane Engineer", "full_name": "Jane Engineer, PhD"},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["name"] == "Jane Engineer"
    assert patch_resp.json()["full_name"] == "Jane Engineer, PhD"

    # 3. Change password with wrong current password (should fail)
    fail_pw_resp = client.post(
        "/api/v1/auth/change-password",
        headers=headers,
        json={"current_password": "WrongPassword123!", "new_password": new_password},
    )
    assert fail_pw_resp.status_code == 400

    # 4. Change password with correct current password (should succeed)
    succ_pw_resp = client.post(
        "/api/v1/auth/change-password",
        headers=headers,
        json={"current_password": old_password, "new_password": new_password},
    )
    assert succ_pw_resp.status_code == 200
    assert succ_pw_resp.json()["success"] is True

    # 5. Verify old password no longer works for login
    old_login = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": old_password},
    )
    assert old_login.status_code == 401

    # 6. Verify new password logs in successfully
    new_login = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": new_password},
    )
    assert new_login.status_code == 200
    assert "access_token" in new_login.json()


def test_csv_document_loader():
    """Verifies CSV document parsing with row and column structuring."""
    csv_content = b"Product,Category,Price\nLaptop,Electronics,999.99\nHeadphones,Audio,149.50"
    doc = extract_text_from_csv(csv_content, filename="products.csv")

    assert doc.is_valid is True
    assert doc.file_type == "csv"
    assert "Laptop" in doc.full_text
    assert "Price: 999.99" in doc.full_text
    assert doc.metadata["row_count"] == 2
    assert doc.metadata["col_count"] == 3


def test_json_document_loader():
    """Verifies JSON document parsing into structured text."""
    json_data = {
        "service": "LocalGPT",
        "features": ["Vector Search", "SSE Streaming", "OAuth"],
        "config": {"temperature": 0.7, "top_p": 0.9},
    }
    json_bytes = json.dumps(json_data).encode("utf-8")
    doc = extract_text_from_json(json_bytes, filename="config.json")

    assert doc.is_valid is True
    assert doc.file_type == "json"
    assert "Vector Search" in doc.full_text
    assert "temperature: 0.7" in doc.full_text


def test_persistent_vector_retriever_search_and_reload(tmp_path):
    """Verifies vector indexing, semantic search, deletion, and reload from disk."""
    store_dir = str(tmp_path / "vector_test")
    retriever = PersistentVectorRetriever(storage_dir=store_dir)
    retriever.clear()

    # Load and split sample documents
    txt_doc = load_document(b"Artificial Intelligence and Machine Learning are transforming software.", "ai_intro.txt")
    chunks = split_document(txt_doc)
    retriever.add_chunks(chunks)

    # Search query
    results = retriever.retrieve("Machine Learning software", top_k=2)
    assert len(results) > 0
    assert results[0]["source"] == "ai_intro.txt"
    assert "Machine Learning" in results[0]["text"]

    # Test Reloading into a new retriever instance from the same disk path
    reloaded_retriever = PersistentVectorRetriever(storage_dir=store_dir)
    stats = reloaded_retriever.get_stats()
    assert stats["total_chunks"] == len(chunks)
    assert "ai_intro.txt" in stats["sources"]

    # Test chunk deletion
    removed = reloaded_retriever.delete_document_chunks("ai_intro.txt")
    assert removed == len(chunks)
    assert reloaded_retriever.get_stats()["total_chunks"] == 0


def test_document_upload_and_delete_api(client, tmp_path):
    """Tests document upload of CSV and deletion via REST endpoints."""
    csv_bytes = b"City,Country,Population\nTokyo,Japan,37400000\nDelhi,India,29300000"
    
    upload_resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("cities.csv", csv_bytes, "text/csv")},
    )
    assert upload_resp.status_code == 201
    doc_data = upload_resp.json()["document"]
    doc_id = doc_data["id"]
    assert doc_data["filename"] == "cities.csv"
    assert doc_data["chunk_count"] >= 1

    # List documents
    list_resp = client.get("/api/v1/documents")
    assert list_resp.status_code == 200
    filenames = [d["filename"] for d in list_resp.json()]
    assert "cities.csv" in filenames

    # Delete document
    del_resp = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_resp.status_code == 204

    # Verify document is removed from list
    list_after = client.get("/api/v1/documents")
    assert list_after.status_code == 200
    assert doc_id not in [d["id"] for d in list_after.json()]


def test_conversation_rest_endpoints(client):
    """Verifies complete CRUD operations on /api/v1/conversations."""
    # 1. Create conversation
    create_resp = client.post(
        "/api/v1/conversations",
        json={"title": "Test Chat Workflow", "system_prompt": "You are a test assistant."},
    )
    assert create_resp.status_code == 201
    conv_data = create_resp.json()
    conv_id = conv_data["id"]
    assert conv_data["title"] == "Test Chat Workflow"

    # 2. Retrieve conversation details
    get_resp = client.get(f"/api/v1/conversations/{conv_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == conv_id
    assert get_resp.json()["system_prompt"] == "You are a test assistant."

    # 3. Rename conversation via PATCH
    patch_resp = client.patch(
        f"/api/v1/conversations/{conv_id}",
        json={"title": "Updated Chat Title"},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["title"] == "Updated Chat Title"

    # 4. List all conversations
    list_resp = client.get("/api/v1/conversations")
    assert list_resp.status_code == 200
    conv_ids = [c["id"] for c in list_resp.json()]
    assert conv_id in conv_ids

    # 5. Delete conversation
    del_resp = client.delete(f"/api/v1/conversations/{conv_id}")
    assert del_resp.status_code == 204

    # 6. Verify deleted conversation returns 404
    not_found_resp = client.get(f"/api/v1/conversations/{conv_id}")
    assert not_found_resp.status_code == 404


def test_chat_completions_sync_and_stream(client, monkeypatch):
    """Verifies synchronous and real-time SSE streaming chat completion endpoints."""
    from app.llm.client import llm_client

    # Mock LLM synchronous generator
    async def mock_generate_response(*args, **kwargs):
        return {"content": "This is a mocked synchronous LLM response.", "usage": {}}

    # Mock LLM streaming token generator
    async def mock_stream_response(*args, **kwargs):
        for token in ["Hello", " ", "from", " ", "mocked", " ", "stream", "!"]:
            yield token

    monkeypatch.setattr(llm_client, "generate_response", mock_generate_response)
    monkeypatch.setattr(llm_client, "stream_response", mock_stream_response)

    # 1. Test Synchronous Chat Completion
    sync_resp = client.post(
        "/api/v1/chat/completions",
        json={"message": "Hello AI!", "mode": "chat"},
    )
    assert sync_resp.status_code == 200
    sync_data = sync_resp.json()
    assert sync_data["role"] == "assistant"
    assert "This is a mocked synchronous LLM response." in sync_data["content"]
    conv_id = sync_data["conversation_id"]

    # 2. Test Real-time SSE Stream Completion
    stream_resp = client.post(
        "/api/v1/chat/stream",
        json={"conversation_id": conv_id, "message": "Stream to me", "mode": "chat"},
    )
    assert stream_resp.status_code == 200
    assert "text/event-stream" in stream_resp.headers["content-type"]
    stream_text = stream_resp.text
    assert "data: " in stream_text
    assert '"type": "start"' in stream_text
    assert '"type": "token"' in stream_text
    assert '"type": "done"' in stream_text
    assert "Hello from mocked stream!" in stream_text


def test_system_health_and_settings_endpoints(client):
    """Verifies root status, health probe, and platform settings inspection endpoints."""
    # 1. Root endpoint
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert root_resp.json()["status"] == "healthy"

    # 3. API v1 health endpoint
    api_health_resp = client.get("/api/v1/health")
    assert api_health_resp.status_code == 200
    assert api_health_resp.json()["status"] == "healthy"

    # 4. API v1 info endpoint
    info_resp = client.get("/api/v1/info")
    assert info_resp.status_code == 200
    assert "project_name" in info_resp.json()

    # 5. Settings endpoint
    settings_resp = client.get("/api/v1/settings")
    assert settings_resp.status_code == 200
    settings_data = settings_resp.json()
    assert "project_name" in settings_data
    assert "masked_api_key" in settings_data


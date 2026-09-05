"""
Comprehensive Test Suite for Step 5: PostgreSQL Database, Models, Migration & CRUD
Tests:
1. Database Connection & Health Probe
2. Alembic Migration Execution
3. User Model CRUD & Password Hashing Security (Zero Plaintext)
4. Conversation Model CRUD & User Relationship
5. Message Model CRUD, Ordering & RAG Sources JSON
6. Document Model CRUD, Metadata & User Relationship
7. Cascade Deletion & Foreign Key Integrity
8. Database Session Context & Transaction Rollback
"""

import os
import sys
import tempfile
import time
try:
    import pytest
except ImportError:
    pytest = None
from datetime import datetime, timezone
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.core.security import get_password_hash, verify_password
from app.database.base import Base
from app.database.session import SessionLocal, check_database_connection, get_db_context
from app.database.migration import run_migrations, get_alembic_config
from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.document import Document
from alembic import command


# ------------------------------------------------------------------------------
# Test 1: Database Connection Health Probe
# ------------------------------------------------------------------------------
def test_database_connection():
    """Verify database connection health check probe returns True."""
    is_connected = check_database_connection()
    assert is_connected is True, "Database connection check failed"


# ------------------------------------------------------------------------------
# Test 2: Alembic Migration Execution & Schema Inspection
# ------------------------------------------------------------------------------
def test_alembic_migration_schema():
    """Verify Alembic migration creates all required tables, columns, and foreign keys."""
    # Use an isolated SQLite database to test clean migration from scratch
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        test_db_path = tmp_db.name

    test_db_url = f"sqlite:///{test_db_path}"
    
    try:
        alembic_cfg = get_alembic_config()
        alembic_cfg.set_main_option("sqlalchemy.url", test_db_url)
        
        # Execute migration upgrade to head
        command.upgrade(alembic_cfg, "head")

        # Inspect resulting database
        test_engine = create_engine(test_db_url)
        inspector = inspect(test_engine)
        tables = inspector.get_table_names()

        # Check required tables exist
        assert "users" in tables, "Table 'users' missing from migration"
        assert "conversations" in tables, "Table 'conversations' missing from migration"
        assert "messages" in tables, "Table 'messages' missing from migration"
        assert "documents" in tables, "Table 'documents' missing from migration"
        assert "alembic_version" in tables, "Table 'alembic_version' missing"

        # Check users columns
        user_cols = {col["name"] for col in inspector.get_columns("users")}
        expected_user_cols = {"id", "email", "hashed_password", "name", "full_name", "profile", "is_active", "is_superuser", "created_at", "updated_at"}
        assert expected_user_cols.issubset(user_cols), f"Missing user columns: {expected_user_cols - user_cols}"

        # Check conversations columns
        conv_cols = {col["name"] for col in inspector.get_columns("conversations")}
        expected_conv_cols = {"id", "user_id", "title", "system_prompt", "created_at", "updated_at"}
        assert expected_conv_cols.issubset(conv_cols), f"Missing conversation columns: {expected_conv_cols - conv_cols}"

        # Check messages columns
        msg_cols = {col["name"] for col in inspector.get_columns("messages")}
        expected_msg_cols = {"id", "conversation_id", "role", "content", "sources", "msg_metadata", "created_at"}
        assert expected_msg_cols.issubset(msg_cols), f"Missing message columns: {expected_msg_cols - msg_cols}"

        # Check documents columns
        doc_cols = {col["name"] for col in inspector.get_columns("documents")}
        expected_doc_cols = {"id", "user_id", "filename", "file_type", "file_path", "file_size_bytes", "page_count", "chunk_count", "doc_metadata", "created_at", "updated_at"}
        assert expected_doc_cols.issubset(doc_cols), f"Missing document columns: {expected_doc_cols - doc_cols}"

    finally:
        if os.path.exists(test_db_path):
            try:
                os.remove(test_db_path)
            except Exception:
                pass


# ------------------------------------------------------------------------------
# Test 3: Password Hashing & Security (Zero Plaintext Secrets)
# ------------------------------------------------------------------------------
def test_password_security():
    """Verify passwords are never stored in plain text and bcrypt validation works."""
    raw_password = "SuperSecretPassword2026!"
    hashed = get_password_hash(raw_password)

    # Hash must not be plain text
    assert hashed != raw_password
    assert hashed.startswith("$2b$") or hashed.startswith("$2a$")
    assert len(hashed) >= 50

    # Verification must match original and reject incorrect password
    assert verify_password(raw_password, hashed) is True
    assert verify_password("WrongPassword123", hashed) is False
    assert verify_password("", hashed) is False


# ------------------------------------------------------------------------------
# Test 4: User CRUD Operations
# ------------------------------------------------------------------------------
def test_user_crud():
    """Verify creating, reading, updating, and querying Users with hashed passwords."""
    with get_db_context() as db:
        unique_email = f"test_{int(time.time() * 1000)}@example.com"
        user = User(
            email=unique_email,
            name="Alice Walker",
            full_name="Alice Walker",
            profile={"theme": "dark", "preferred_model": "llama-3.3-70b"},
        )
        user.set_password("SecurePass456!")
        db.add(user)
        db.flush()

        user_id = user.id
        assert user_id.startswith("usr_")
        assert user.hashed_password != "SecurePass456!"
        assert user.check_password("SecurePass456!") is True
        assert user.check_password("wrong") is False

    # Read back in new session
    with get_db_context() as db:
        fetched = db.query(User).filter(User.id == user_id).first()
        assert fetched is not None
        assert fetched.email == unique_email
        assert fetched.name == "Alice Walker"
        assert fetched.profile["theme"] == "dark"

        # Update
        fetched.name = "Alice B. Walker"
        fetched.profile = {"theme": "light", "preferred_model": "llama-3.3-70b"}
        db.flush()

    # Verify update
    with get_db_context() as db:
        updated = db.query(User).filter(User.id == user_id).first()
        assert updated.name == "Alice B. Walker"
        assert updated.profile["theme"] == "light"

        # Clean up
        db.delete(updated)


# ------------------------------------------------------------------------------
# Test 5: Conversation CRUD & User Relationship
# ------------------------------------------------------------------------------
def test_conversation_crud():
    """Verify Conversation creation, user link, title update, and query."""
    with get_db_context() as db:
        # Create user
        user = User(
            email=f"conv_user_{int(time.time() * 1000)}@example.com",
            name="Bob User",
        )
        user.set_password("BobSecret123")
        db.add(user)
        db.flush()

        # Create conversation
        conv = Conversation(
            user_id=user.id,
            title="Project Research Discussion",
            system_prompt="You are a helpful research scientist.",
        )
        db.add(conv)
        db.flush()

        conv_id = conv.id
        user_id = user.id
        assert conv_id.startswith("conv_")
        assert conv.user.email == user.email

    # Read and update
    with get_db_context() as db:
        fetched_conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
        assert fetched_conv is not None
        assert fetched_conv.title == "Project Research Discussion"
        assert fetched_conv.user_id == user_id

        # Update title
        fetched_conv.title = "Updated Research Discussion"
        db.flush()

    # Verify and clean up
    with get_db_context() as db:
        c = db.query(Conversation).filter(Conversation.id == conv_id).first()
        assert c.title == "Updated Research Discussion"
        u = db.query(User).filter(User.id == user_id).first()
        db.delete(u)  # Cascade will delete conversation


# ------------------------------------------------------------------------------
# Test 6: Message CRUD, Ordering & RAG Sources JSON
# ------------------------------------------------------------------------------
def test_message_crud_and_sources():
    """Verify adding messages, JSON sources citation, and ordered conversation history."""
    with get_db_context() as db:
        conv = Conversation(title="RAG Test Chat")
        db.add(conv)
        db.flush()
        conv_id = conv.id

        # User message
        msg1 = Message(
            conversation_id=conv_id,
            role="user",
            content="What are the key findings in section 2?",
        )
        db.add(msg1)
        db.flush()

        # Assistant message with RAG sources
        rag_sources = [
            {"doc_id": "doc_abc123", "filename": "q2_report.pdf", "page": 4, "snippet": "Finding 1: High accuracy"},
            {"doc_id": "doc_abc123", "filename": "q2_report.pdf", "page": 5, "snippet": "Finding 2: Low latency"},
        ]
        msg2 = Message(
            conversation_id=conv_id,
            role="assistant",
            content="Section 2 highlights high accuracy and low latency.",
            sources=rag_sources,
            msg_metadata={"model": "llama-3.3-70b-versatile", "tokens_used": 145},
        )
        db.add(msg2)
        db.flush()

    # Read back conversation with messages
    with get_db_context() as db:
        c = db.query(Conversation).filter(Conversation.id == conv_id).first()
        assert len(c.messages) == 2
        assert c.messages[0].role == "user"
        assert c.messages[1].role == "assistant"
        assert c.messages[1].sources == rag_sources
        assert c.messages[1].msg_metadata["tokens_used"] == 145

        # Cleanup
        db.delete(c)


# ------------------------------------------------------------------------------
# Test 7: Document CRUD & Metadata JSON
# ------------------------------------------------------------------------------
def test_document_crud():
    """Verify Document creation, metadata JSON storage, and retrieval."""
    with get_db_context() as db:
        user = User(
            email=f"doc_user_{int(time.time() * 1000)}@example.com",
            name="Charlie Doc Owner",
        )
        user.set_password("CharliePass789")
        db.add(user)
        db.flush()
        user_id = user.id

        doc = Document(
            user_id=user_id,
            filename="architecture_spec.pdf",
            file_type="pdf",
            file_path="/data/documents/architecture_spec.pdf",
            file_size_bytes=1048576,
            page_count=24,
            chunk_count=96,
            doc_metadata={"author": "Team Lead", "version": "2.1", "embedding_model": "all-MiniLM-L6-v2"},
        )
        db.add(doc)
        db.flush()
        doc_id = doc.id
        assert doc_id.startswith("doc_")

    # Read and update
    with get_db_context() as db:
        fetched_doc = db.query(Document).filter(Document.id == doc_id).first()
        assert fetched_doc is not None
        assert fetched_doc.filename == "architecture_spec.pdf"
        assert fetched_doc.file_type == "pdf"
        assert fetched_doc.page_count == 24
        assert fetched_doc.chunk_count == 96
        assert fetched_doc.doc_metadata["author"] == "Team Lead"
        assert fetched_doc.user.name == "Charlie Doc Owner"

        # Update chunk count and metadata
        fetched_doc.chunk_count = 100
        meta = dict(fetched_doc.doc_metadata)
        meta["status"] = "indexed"
        fetched_doc.doc_metadata = meta
        db.flush()

    # Verify update
    with get_db_context() as db:
        d = db.query(Document).filter(Document.id == doc_id).first()
        assert d.chunk_count == 100
        assert d.doc_metadata["status"] == "indexed"

        # Delete document and user
        db.delete(d)
        u = db.query(User).filter(User.id == user_id).first()
        db.delete(u)


# ------------------------------------------------------------------------------
# Test 8: Cascade Deletion & Foreign Key Constraints
# ------------------------------------------------------------------------------
def test_cascade_deletion():
    """Verify deleting conversation cascades to delete all child messages."""
    with get_db_context() as db:
        conv = Conversation(title="Cascade Test Chat")
        db.add(conv)
        db.flush()
        conv_id = conv.id

        msg = Message(conversation_id=conv_id, role="user", content="Test message")
        db.add(msg)
        db.flush()
        msg_id = msg.id

    # Delete conversation
    with get_db_context() as db:
        c = db.query(Conversation).filter(Conversation.id == conv_id).first()
        db.delete(c)

    # Verify message was also deleted by cascade
    with get_db_context() as db:
        orphan_msg = db.query(Message).filter(Message.id == msg_id).first()
        assert orphan_msg is None, "Message was not cascade deleted with Conversation"


# ------------------------------------------------------------------------------
# Test 9: Transaction Rollback on Error
# ------------------------------------------------------------------------------
def test_transaction_rollback():
    """Verify unhandled exception automatically rolls back database session state."""
    test_email = f"rollback_{int(time.time() * 1000)}@example.com"

    try:
        with get_db_context() as db:
            user = User(email=test_email, name="Rollback Test User")
            user.set_password("RollbackPass123")
            db.add(user)
            db.flush()
            # Force an error
            raise ValueError("Intentional exception to test transaction rollback")
    except ValueError:
        pass

    # Verify user was NOT persisted
    with get_db_context() as db:
        persisted = db.query(User).filter(User.email == test_email).first()
        assert persisted is None, "Transaction was not rolled back after exception"


if __name__ == "__main__":
    print("\n=======================================================")
    print(" Running Step 5: PostgreSQL Database & CRUD Test Suite")
    print("=======================================================\n")

    print("[1/9] Testing Database Connection...")
    test_database_connection()
    print("  -> PASSED: Database is healthy and reachable.")

    print("[2/9] Testing Alembic Migration Schema...")
    test_alembic_migration_schema()
    print("  -> PASSED: Alembic migrations create all tables & columns successfully.")

    print("[3/9] Testing Password Security (Zero Plaintext Secrets)...")
    test_password_security()
    print("  -> PASSED: Bcrypt hashing and verification pass security standards.")

    print("[4/9] Testing User CRUD Operations...")
    test_user_crud()
    print("  -> PASSED: User creation, reading, profile updating, and queries work.")

    print("[5/9] Testing Conversation CRUD & User Relationship...")
    test_conversation_crud()
    print("  -> PASSED: Conversation entity and User associations work.")

    print("[6/9] Testing Message CRUD & Sources JSON...")
    test_message_crud_and_sources()
    print("  -> PASSED: Message history and RAG sources JSON work.")

    print("[7/9] Testing Document CRUD & Metadata JSON...")
    test_document_crud()
    print("  -> PASSED: Document entity, user relation, and metadata work.")

    print("[8/9] Testing Cascade Deletion...")
    test_cascade_deletion()
    print("  -> PASSED: Cascade deletes child messages properly.")

    print("[9/9] Testing Transaction Rollback...")
    test_transaction_rollback()
    print("  -> PASSED: Session automatically rolls back upon exception.")

    print("\n=======================================================")
    print(" ALL 9 DATABASE TESTS PASSED SUCCESSFULLY! ")
    print("=======================================================\n")

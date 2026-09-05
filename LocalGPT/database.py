"""
LocalGPT: Local Database & Persistence (Step 14: Persistent Conversation Memory)
Uses standard Python SQLite (data/conversations/conversations.db) to persist
conversations locally with zero cloud dependencies.

Schema:
- chat_id (TEXT PRIMARY KEY)
- title (TEXT)
- messages (TEXT JSON)
- created_at (REAL)
- updated_at (REAL)
"""

import os
import json
import time
import sqlite3
from typing import List, Dict, Any, Optional

DEFAULT_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "conversations")
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "conversations.db")


def get_db_path(db_path: Optional[str] = None) -> str:
    """
    Returns the absolute path to the SQLite conversations database.
    """
    if db_path is not None:
        return db_path
    os.makedirs(DEFAULT_DB_DIR, exist_ok=True)
    return DEFAULT_DB_PATH


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Creates and initializes a thread-safe connection to the SQLite database.
    """
    target_path = get_db_path(db_path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    conn = sqlite3.connect(target_path, check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """
    Initializes the conversations table schema if not already present.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    chat_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    messages TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_updated_at 
                ON conversations(updated_at DESC)
                """
            )
    finally:
        conn.close()


def save_conversation(
    chat_id: str,
    title: str,
    messages: List[Dict[str, Any]],
    created_at: Optional[float] = None,
    updated_at: Optional[float] = None,
    db_path: Optional[str] = None,
) -> bool:
    """
    Inserts or updates a complete conversation record in the local SQLite database.
    
    Args:
        chat_id: Unique chat identifier.
        title: Conversation title.
        messages: List of message dictionaries.
        created_at: Creation timestamp (optional).
        updated_at: Update timestamp (optional).
        db_path: Optional database path override.
        
    Returns:
        True if saved successfully, False otherwise.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    now = time.time()
    
    try:
        with conn:
            # Check existing created_at if not provided
            if created_at is None:
                cursor = conn.execute("SELECT created_at FROM conversations WHERE chat_id = ?", (chat_id,))
                row = cursor.fetchone()
                if row:
                    created_at = float(row["created_at"])
                else:
                    created_at = now

            if updated_at is None:
                updated_at = now

            messages_json = json.dumps(messages, ensure_ascii=False)

            conn.execute(
                """
                INSERT INTO conversations (chat_id, title, messages, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    title = excluded.title,
                    messages = excluded.messages,
                    updated_at = excluded.updated_at
                """,
                (chat_id, title, messages_json, float(created_at), float(updated_at)),
            )
        return True
    except Exception as e:
        print(f"[database] Error saving conversation {chat_id}: {e}")
        return False
    finally:
        conn.close()


def load_conversation(
    chat_id: str,
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Loads a single conversation from the database by its chat_id.
    
    Args:
        chat_id: Conversation session identifier.
        db_path: Optional database path override.
        
    Returns:
        Dictionary with id, title, messages, created_at, updated_at, or None if not found.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT chat_id, title, messages, created_at, updated_at FROM conversations WHERE chat_id = ?",
            (chat_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        try:
            messages = json.loads(row["messages"])
        except Exception:
            messages = []

        return {
            "chat_id": row["chat_id"],
            "id": row["chat_id"],
            "session_id": row["chat_id"],
            "title": row["title"],
            "messages": messages,
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }
    except Exception as e:
        print(f"[database] Error loading conversation {chat_id}: {e}")
        return None
    finally:
        conn.close()


def load_all_conversations(
    db_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Loads all saved conversations from the database, ordered newest first.
    
    Returns:
        Dictionary mapping chat_id -> conversation dictionary.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    chats: Dict[str, Dict[str, Any]] = {}

    try:
        cursor = conn.execute(
            "SELECT chat_id, title, messages, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        )
        rows = cursor.fetchall()
        for row in rows:
            cid = row["chat_id"]
            try:
                msgs = json.loads(row["messages"])
            except Exception:
                msgs = []
            chats[cid] = {
                "chat_id": cid,
                "id": cid,
                "session_id": cid,
                "title": row["title"],
                "messages": msgs,
                "created_at": float(row["created_at"]),
                "updated_at": float(row["updated_at"]),
            }
    except Exception as e:
        print(f"[database] Error loading all conversations: {e}")
    finally:
        conn.close()

    return chats


def delete_conversation(
    chat_id: str,
    db_path: Optional[str] = None,
) -> bool:
    """
    Permanently removes a conversation from the local database.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.execute("DELETE FROM conversations WHERE chat_id = ?", (chat_id,))
        return cursor.rowcount > 0
    except Exception as e:
        print(f"[database] Error deleting conversation {chat_id}: {e}")
        return False
    finally:
        conn.close()


def rename_conversation(
    chat_id: str,
    new_title: str,
    db_path: Optional[str] = None,
) -> bool:
    """
    Renames a conversation and updates its modification timestamp.
    """
    clean_title = new_title.strip()
    if not clean_title:
        return False

    init_db(db_path)
    conn = get_connection(db_path)
    now = time.time()
    try:
        with conn:
            cursor = conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE chat_id = ?",
                (clean_title, now, chat_id),
            )
            return cursor.rowcount > 0
    except Exception as e:
        print(f"[database] Error renaming conversation {chat_id}: {e}")
        return False
    finally:
        conn.close()


# -------------------------------------------------------------
# BACKWARD COMPATIBILITY HELPERS
# -------------------------------------------------------------
def save_conversation_session(
    session_id: str,
    messages: List[Dict[str, Any]],
    persona: str = "General Assistant",
    metadata: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None,
) -> str:
    """
    Backward-compatible save wrapper updating the SQLite database.
    """
    title = (metadata or {}).get("title", "New Chat") if metadata else "New Chat"
    created_at = (metadata or {}).get("created_at") if metadata else None
    save_conversation(
        chat_id=session_id,
        title=title,
        messages=messages,
        created_at=created_at,
        db_path=db_path,
    )
    return get_db_path(db_path)


def load_conversation_session(
    session_id: str,
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Backward-compatible load wrapper.
    """
    return load_conversation(session_id, db_path=db_path)


def delete_conversation_session(
    session_id: str,
    db_path: Optional[str] = None,
) -> bool:
    """
    Backward-compatible delete wrapper.
    """
    return delete_conversation(session_id, db_path=db_path)


def list_saved_conversations(
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Lists metadata summaries of all saved conversations.
    """
    chats = load_all_conversations(db_path)
    summaries = []
    for c in chats.values():
        first_user_msg = next((m["content"] for m in c.get("messages", []) if m.get("role") == "user"), "Empty Conversation")
        preview = (first_user_msg[:60] + "...") if len(first_user_msg) > 60 else first_user_msg
        summaries.append({
            "session_id": c["id"],
            "title": c.get("title", "New Chat"),
            "timestamp": c.get("updated_at", 0),
            "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(c.get("updated_at", time.time()))),
            "preview": preview,
            "message_count": len(c.get("messages", [])),
        })
    return summaries

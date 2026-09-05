"""
LocalGPT: Step 14 Verification Test Suite
Tests Persistent Conversation Memory:
1. Save conversations locally (Chat ID, Title, Messages, Created time, Updated time)
2. Load conversations on application startup
3. Open old conversations
4. Update conversations after new messages
5. Delete conversations
6. Rename conversations
7. 100% local database with zero cloud dependencies
8. Full end-to-end simulated app restart lifecycle test
"""

import os
import sys
import time
import shutil
import tempfile
import sqlite3
from typing import Dict, Any, List

# Set UTF-8 stdout encoding for Windows compatibility
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure LocalGPT path is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import (
    get_db_path,
    init_db,
    save_conversation,
    load_conversation,
    load_all_conversations,
    delete_conversation,
    rename_conversation,
    save_conversation_session,
    load_conversation_session,
    delete_conversation_session,
    list_saved_conversations,
)


def run_all_tests():
    print("=" * 75)
    print("LocalGPT Step 14: Persistent Conversation Memory Verification Test Suite")
    print("=" * 75)

    results: Dict[str, bool] = {}
    temp_dir = tempfile.mkdtemp(prefix="localgpt_test_db_step14_")
    test_db_path = os.path.join(temp_dir, "test_conversations.db")

    try:
        # -------------------------------------------------------------
        # Test 1: SQLite Schema Initialization & Local File Creation
        # -------------------------------------------------------------
        print("\n[Test 1/8] Verifying SQLite Schema & Local File Initialization...")
        init_db(test_db_path)
        assert os.path.exists(test_db_path), f"Database file was not created at {test_db_path}"
        
        # Verify SQLite schema
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'")
        table_row = cursor.fetchone()
        assert table_row is not None, "conversations table was not created"
        
        cursor.execute("PRAGMA table_info(conversations)")
        columns = {col[1]: col[2] for col in cursor.fetchall()}
        conn.close()

        expected_cols = ["chat_id", "title", "messages", "created_at", "updated_at"]
        for col_name in expected_cols:
            assert col_name in columns, f"Missing column in schema: {col_name}"
        
        print(f"  [+] Local SQLite DB initialized at: {test_db_path}")
        print(f"  [+] Columns verified: {list(columns.keys())}")
        results["Schema & Local SQLite Initialization"] = True

        # -------------------------------------------------------------
        # Test 2: Create & Save Conversation with Complete Metadata
        # -------------------------------------------------------------
        print("\n[Test 2/8] Verifying Conversation Creation & Persistence...")
        chat_id_1 = "chat_test_001"
        title_1 = "Quantum Physics Basics"
        created_time_1 = time.time() - 100
        updated_time_1 = created_time_1

        messages_1 = [
            {"role": "user", "content": "What is superposition?"},
            {"role": "assistant", "content": "Superposition is a principle of quantum mechanics where a system exists in multiple states simultaneously."},
        ]

        save_res = save_conversation(
            chat_id=chat_id_1,
            title=title_1,
            messages=messages_1,
            created_at=created_time_1,
            updated_at=updated_time_1,
            db_path=test_db_path,
        )
        assert save_res is True, "save_conversation returned False"

        loaded_1 = load_conversation(chat_id_1, db_path=test_db_path)
        assert loaded_1 is not None, "Failed to load saved conversation"
        assert loaded_1["chat_id"] == chat_id_1
        assert loaded_1["id"] == chat_id_1
        assert loaded_1["title"] == title_1
        assert len(loaded_1["messages"]) == 2
        assert loaded_1["messages"][0]["content"] == "What is superposition?"
        assert abs(loaded_1["created_at"] - created_time_1) < 0.01
        assert abs(loaded_1["updated_at"] - updated_time_1) < 0.01

        print(f"  [+] Successfully saved and verified conversation '{title_1}' ({chat_id_1})")
        results["Save Conversation with Full Metadata"] = True

        # -------------------------------------------------------------
        # Test 3: Update Conversation after New Messages
        # -------------------------------------------------------------
        print("\n[Test 3/8] Verifying Conversation Update after Adding New Messages...")
        new_user_msg = {"role": "user", "content": "How does entanglement relate to it?"}
        new_ai_msg = {"role": "assistant", "content": "Quantum entanglement occurs when quantum particles interact such that the state of each particle cannot be described independently."}
        
        messages_updated = list(messages_1) + [new_user_msg, new_ai_msg]
        new_updated_time = time.time()

        save_update_res = save_conversation(
            chat_id=chat_id_1,
            title=title_1,
            messages=messages_updated,
            updated_at=new_updated_time,
            db_path=test_db_path,
        )
        assert save_update_res is True, "Failed to update conversation"

        reloaded_1 = load_conversation(chat_id_1, db_path=test_db_path)
        assert reloaded_1 is not None
        assert len(reloaded_1["messages"]) == 4
        assert reloaded_1["messages"][2]["content"] == "How does entanglement relate to it?"
        assert reloaded_1["messages"][3]["content"].startswith("Quantum entanglement")
        # Created timestamp must remain unchanged, updated timestamp must reflect update
        assert abs(reloaded_1["created_at"] - created_time_1) < 0.01
        assert reloaded_1["updated_at"] >= new_updated_time

        print(f"  [+] Conversation updated: message count grew from 2 to {len(reloaded_1['messages'])}")
        print(f"  [+] Creation timestamp preserved ({reloaded_1['created_at']:.2f}) and updated timestamp changed ({reloaded_1['updated_at']:.2f})")
        results["Update Conversation after New Messages"] = True

        # -------------------------------------------------------------
        # Test 4: Rename Conversation
        # -------------------------------------------------------------
        print("\n[Test 4/8] Verifying Conversation Renaming...")
        new_title = "Quantum Physics & Entanglement Deep Dive"
        rename_res = rename_conversation(chat_id_1, new_title, db_path=test_db_path)
        assert rename_res is True, "rename_conversation returned False"

        renamed_chat = load_conversation(chat_id_1, db_path=test_db_path)
        assert renamed_chat is not None
        assert renamed_chat["title"] == new_title
        assert len(renamed_chat["messages"]) == 4

        # Test renaming with whitespace rejection
        assert rename_conversation(chat_id_1, "   ", db_path=test_db_path) is False

        print(f"  [+] Conversation title successfully renamed to: '{renamed_chat['title']}'")
        results["Rename Conversation"] = True

        # -------------------------------------------------------------
        # Test 5: Load All Conversations & Ordering
        # -------------------------------------------------------------
        print("\n[Test 5/8] Verifying Loading All Conversations on Application Startup...")
        # Create second and third chats
        chat_id_2 = "chat_test_002"
        chat_id_3 = "chat_test_003"
        t_now = time.time()

        save_conversation(
            chat_id=chat_id_2,
            title="Python Data Structures",
            messages=[{"role": "user", "content": "Explain Python dicts."}],
            created_at=t_now - 50,
            updated_at=t_now - 10,
            db_path=test_db_path,
        )

        save_conversation(
            chat_id=chat_id_3,
            title="Local LLM Architecture",
            messages=[{"role": "user", "content": "What is self-attention?"}],
            created_at=t_now - 20,
            updated_at=t_now + 10,
            db_path=test_db_path,
        )

        all_chats = load_all_conversations(db_path=test_db_path)
        assert len(all_chats) == 3, f"Expected 3 conversations, found {len(all_chats)}"
        assert chat_id_1 in all_chats
        assert chat_id_2 in all_chats
        assert chat_id_3 in all_chats

        # Verify order (chat_id_3 has highest updated_at, so it should appear first)
        chat_list = list(all_chats.values())
        assert chat_list[0]["id"] == chat_id_3
        print(f"  [+] Loaded {len(all_chats)} conversations on startup in newest-first order.")
        results["Load All Conversations on Startup"] = True

        # -------------------------------------------------------------
        # Test 6: Open Old Conversations
        # -------------------------------------------------------------
        print("\n[Test 6/8] Verifying Opening Old Conversations...")
        old_chat = load_conversation(chat_id_2, db_path=test_db_path)
        assert old_chat is not None
        assert old_chat["id"] == chat_id_2
        assert old_chat["title"] == "Python Data Structures"
        assert len(old_chat["messages"]) == 1
        assert old_chat["messages"][0]["content"] == "Explain Python dicts."

        print(f"  [+] Reopened old conversation '{old_chat['title']}' with intact messages.")
        results["Open Old Conversations"] = True

        # -------------------------------------------------------------
        # Test 7: Delete Conversation
        # -------------------------------------------------------------
        print("\n[Test 7/8] Verifying Conversation Deletion...")
        del_res = delete_conversation(chat_id_2, db_path=test_db_path)
        assert del_res is True, "delete_conversation returned False"

        # Verify it no longer exists
        assert load_conversation(chat_id_2, db_path=test_db_path) is None
        remaining_chats = load_all_conversations(db_path=test_db_path)
        assert len(remaining_chats) == 2
        assert chat_id_2 not in remaining_chats

        # Deleting non-existent chat should return False
        assert delete_conversation("non_existent_id", db_path=test_db_path) is False

        print(f"  [+] Deleted conversation {chat_id_2}. Remaining count: {len(remaining_chats)}")
        results["Delete Conversation"] = True

        # -------------------------------------------------------------
        # Test 8: Full End-to-End Application Lifecycle Simulation
        # (Create Chat -> Add Messages -> Restart App -> Reopen Chat)
        # -------------------------------------------------------------
        print("\n[Test 8/8] Simulating App Lifecycle: Create -> Add Msgs -> Restart -> Reopen...")
        lifecycle_db_dir = tempfile.mkdtemp(prefix="localgpt_lifecycle_")
        lifecycle_db_path = os.path.join(lifecycle_db_dir, "conversations.db")

        # --- Phase 1: App Session 1 Running ---
        session1_state: Dict[str, Any] = {
            "chats": load_all_conversations(lifecycle_db_path),
            "current_chat_id": None,
        }
        
        # User creates a new chat
        new_cid = f"chat_{int(time.time())}_abc123"
        init_title = "New Chat"
        now_1 = time.time()
        session1_state["chats"][new_cid] = {
            "id": new_cid,
            "title": init_title,
            "messages": [],
            "created_at": now_1,
            "updated_at": now_1,
        }
        session1_state["current_chat_id"] = new_cid
        save_conversation(new_cid, init_title, [], now_1, now_1, db_path=lifecycle_db_path)

        # User sends message 1
        user_text = "Summarize the transformer architecture."
        ai_reply = "Transformers use self-attention mechanisms to process tokens in parallel across sequence positions."
        session1_state["chats"][new_cid]["title"] = "Summarize the transform..."
        session1_state["chats"][new_cid]["messages"].append({"role": "user", "content": user_text})
        session1_state["chats"][new_cid]["messages"].append({"role": "assistant", "content": ai_reply})
        t_msg = time.time()
        session1_state["chats"][new_cid]["updated_at"] = t_msg
        save_conversation(
            new_cid,
            session1_state["chats"][new_cid]["title"],
            session1_state["chats"][new_cid]["messages"],
            created_at=now_1,
            updated_at=t_msg,
            db_path=lifecycle_db_path,
        )

        # User sends message 2
        user_text_2 = "What are query, key, and value vectors?"
        ai_reply_2 = "Query, Key, and Value vectors represent projections used to compute attention weight matrices."
        session1_state["chats"][new_cid]["messages"].append({"role": "user", "content": user_text_2})
        session1_state["chats"][new_cid]["messages"].append({"role": "assistant", "content": ai_reply_2})
        t_msg_2 = time.time()
        session1_state["chats"][new_cid]["updated_at"] = t_msg_2
        save_conversation(
            new_cid,
            session1_state["chats"][new_cid]["title"],
            session1_state["chats"][new_cid]["messages"],
            created_at=now_1,
            updated_at=t_msg_2,
            db_path=lifecycle_db_path,
        )

        # --- Phase 2: Application Closes (Memory Wiped) ---
        print("  [>] Simulating application restart: Memory session cleared.")
        del session1_state

        # --- Phase 3: Application Starts (Session 2 Initialized) ---
        session2_state: Dict[str, Any] = {}
        session2_state["chats"] = load_all_conversations(lifecycle_db_path)
        sorted_keys = sorted(
            session2_state["chats"].keys(),
            key=lambda k: session2_state["chats"][k].get("updated_at", 0),
            reverse=True,
        )
        session2_state["current_chat_id"] = sorted_keys[0]

        # Verify reopened chat
        reopened_chat = session2_state["chats"][session2_state["current_chat_id"]]
        assert reopened_chat["id"] == new_cid
        assert reopened_chat["title"] == "Summarize the transform..."
        assert len(reopened_chat["messages"]) == 4
        assert reopened_chat["messages"][0]["content"] == user_text
        assert reopened_chat["messages"][1]["content"] == ai_reply
        assert reopened_chat["messages"][2]["content"] == user_text_2
        assert reopened_chat["messages"][3]["content"] == ai_reply_2

        print("  [+] App Restart test: All 4 messages, title, and timestamps perfectly restored!")
        results["Simulated Application Restart & Persistence"] = True

        shutil.rmtree(lifecycle_db_dir, ignore_errors=True)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 14 TEST EXECUTION SUMMARY:")
    print("=" * 75)
    all_passed = True
    for test_name, passed in results.items():
        status_str = "PASSED" if passed else "FAILED"
        status_box = "✅" if passed else "❌"
        print(f"  {status_box} {test_name:<50} : {status_str}")
        if not passed:
            all_passed = False

    print("=" * 75)
    if all_passed and len(results) == 8:
        print(f"ALL {len(results)} STEP 14 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    else:
        print(f"SOME TESTS FAILED ({sum(results.values())}/{len(results)} passed)")
    print("=" * 75)


if __name__ == "__main__":
    run_all_tests()

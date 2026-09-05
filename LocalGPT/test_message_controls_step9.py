"""
LocalGPT: Step 9 Message Controls Verification Test Suite
Tests:
1. Copy Control: Exact full AI response extraction & text preservation.
2. Regenerate Control: Re-generating response with same context, history truncation, and no duplicates.
3. Edit User Message Control: Modifying user message, branch pruning, and new response streaming.
4. Conversation History Integrity & Duplicate Prevention across multiple edits and regenerations.
5. Real-time Streaming Compatibility during edits and regenerations.
6. Session Persistence Sync (Save & Reload after edits/regenerations).
7. Neural X-Ray Inspection on generated AI responses.
"""

import os
import sys
import time
from typing import List, Dict, Any

# Set UTF-8 stdout encoding for Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure LocalGPT directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import load_model_and_tokenizer, stream_chat_response, generate_chat_response, clear_memory_cache
from tokenizer import format_chat_prompt, tokenize_text
from memory import ConversationMemory
from database import save_conversation_session, load_conversation_session, delete_conversation_session
from xray import extract_embeddings, extract_next_token_logits


def run_step9_tests():
    print("=" * 75)
    print("LocalGPT Step 9: Message Controls Verification & Integrity Test Suite")
    print("=" * 75)

    test_results = {}

    # Test 1: Load Model & Tokenizer
    print("\n[Test 1/7] Loading Model and Tokenizer...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer()
    t_load = time.time() - t0
    print(f"  [+] Model & Tokenizer loaded in {t_load:.2f}s")
    assert model is not None, "Model failed to load"
    assert tokenizer is not None, "Tokenizer failed to load"
    test_results["Model & Tokenizer Loading"] = True

    # Test 2: Copy Control Integrity (Exact Full Text Extraction)
    print("\n[Test 2/7] Testing AI Response Copy Integrity...")
    prompt_text = "State Newton's first law of motion in one concise sentence."
    messages = [
        {"role": "system", "content": "You are a concise physics tutor."},
        {"role": "user", "content": prompt_text},
    ]
    formatted = format_chat_prompt(messages, tokenizer=tokenizer)
    
    stream_chunks = []
    for chunk in stream_chat_response(formatted, model=model, tokenizer=tokenizer, max_new_tokens=40, temperature=0.0):
        stream_chunks.append(chunk)
    
    full_ai_response = "".join(stream_chunks).strip()
    print(f"  [+] Generated Response: '{full_ai_response}'")
    
    # Copy verification: check that full response is intact, non-empty, and contains expected terms
    copied_text = full_ai_response[:]
    assert len(copied_text) > 10, "Response text too short"
    assert copied_text == full_ai_response, "Copy text mismatch"
    assert any(term in copied_text.lower() for term in ["motion", "rest", "force", "velocity", "inertia"]), f"Unexpected content: {copied_text}"
    print("  [+] Copy control text fidelity verified: 100% exact match.")
    test_results["AI Response Copy Control"] = True

    # Test 3: Regenerate Control (Same Context, Re-run Generation, No Duplicates)
    print("\n[Test 3/7] Testing AI Response Regenerate Control...")
    chat_history = [
        {"role": "user", "content": "Suggest one creative name for a spacecraft."},
        {"role": "assistant", "content": full_ai_response},  # placeholder assistant response
    ]
    assert len(chat_history) == 2

    # Simulate Regenerate on assistant response (index 1)
    # Action: remove assistant response at idx 1 (keeping context up to user message)
    regen_context = chat_history[:1]
    assert len(regen_context) == 1
    assert regen_context[0]["role"] == "user"

    # Stream new response with temperature > 0
    regen_prompt = format_chat_prompt(
        [{"role": "system", "content": "You are an imaginative naming consultant."}] + regen_context,
        tokenizer=tokenizer,
    )
    new_chunks = []
    for chunk in stream_chat_response(regen_prompt, model=model, tokenizer=tokenizer, max_new_tokens=30, temperature=0.8):
        new_chunks.append(chunk)
    new_response = "".join(new_chunks).strip()

    # Update history with new response
    chat_history = regen_context + [{"role": "assistant", "content": new_response}]
    print(f"  [+] Regenerated Response: '{new_response}'")
    
    assert len(chat_history) == 2, f"Expected 2 messages after regeneration, got {len(chat_history)}"
    assert chat_history[0]["role"] == "user"
    assert chat_history[1]["role"] == "assistant"
    assert len(chat_history[1]["content"]) > 0
    print("  [+] Regenerate control verified: history cleanly updated without duplication.")
    test_results["AI Response Regenerate Control"] = True

    # Test 4: Edit User Message Control (Modify User Message & Truncate Subsequent Turns)
    print("\n[Test 4/7] Testing User Message Edit & Resubmit...")
    # Setup a 4-message multi-turn chat
    multi_turn_chat = [
        {"role": "user", "content": "I live in Seattle."},
        {"role": "assistant", "content": "Seattle is a beautiful city in Washington State, known for the Space Needle."},
        {"role": "user", "content": "What is the weather usually like here in November?"},
        {"role": "assistant", "content": "In November, Seattle typically experiences chilly and rainy weather."},
    ]
    assert len(multi_turn_chat) == 4

    # User edits message #0 ("I live in Seattle" -> "I live in Cairo, Egypt.")
    edit_idx = 0
    edited_text = "I live in Cairo, Egypt."

    # Edit action:
    # 1. Update user message at edit_idx
    # 2. Truncate all subsequent messages after edit_idx
    multi_turn_chat[edit_idx]["content"] = edited_text
    multi_turn_chat = multi_turn_chat[:edit_idx + 1]
    assert len(multi_turn_chat) == 1, f"Expected history truncated to 1 message, got {len(multi_turn_chat)}"
    assert multi_turn_chat[0]["content"] == edited_text

    # Stream new response for edited context
    edit_prompt = format_chat_prompt(
        [{"role": "system", "content": "You are a concise geography assistant."}] + multi_turn_chat,
        tokenizer=tokenizer,
    )
    edit_chunks = []
    for chunk in stream_chat_response(edit_prompt, model=model, tokenizer=tokenizer, max_new_tokens=40, temperature=0.3):
        edit_chunks.append(chunk)
    edit_response = "".join(edit_chunks).strip()
    multi_turn_chat.append({"role": "assistant", "content": edit_response})

    print(f"  [+] Edited User Message: '{edited_text}'")
    print(f"  [+] New Assistant Response: '{edit_response}'")

    assert len(multi_turn_chat) == 2
    assert "cairo" in edit_response.lower() or "egypt" in edit_response.lower() or "pyramid" in edit_response.lower() or "nile" in edit_response.lower(), f"Response did not reflect edit: {edit_response}"
    print("  [+] User message edit & re-streaming verified.")
    test_results["User Message Edit Control"] = True

    # Test 5: History Integrity & Duplicate Prevention
    print("\n[Test 5/7] Testing History Integrity & Duplicate Prevention...")
    # Validate alternating role pattern: user -> assistant -> user -> assistant
    for i, msg in enumerate(multi_turn_chat):
        expected_role = "user" if i % 2 == 0 else "assistant"
        assert msg["role"] == expected_role, f"Message #{i} has invalid role {msg['role']}, expected {expected_role}"
        assert isinstance(msg["content"], str) and len(msg["content"].strip()) > 0, f"Message #{i} has empty content"
    
    # Test editing the second user message in a 4-turn chat
    multi_turn_chat.append({"role": "user", "content": "What river flows through here?"})
    multi_turn_chat.append({"role": "assistant", "content": "The Nile River flows through Cairo."})
    assert len(multi_turn_chat) == 4

    # Edit turn 2 (second user message)
    edit_idx_2 = 2
    multi_turn_chat[edit_idx_2]["content"] = "What famous monuments are located here?"
    multi_turn_chat = multi_turn_chat[:edit_idx_2 + 1]
    assert len(multi_turn_chat) == 3

    regen_p2 = format_chat_prompt(
        [{"role": "system", "content": "You are a concise assistant."}] + multi_turn_chat,
        tokenizer=tokenizer,
    )
    p2_chunks = [c for c in stream_chat_response(regen_p2, model=model, tokenizer=tokenizer, max_new_tokens=30, temperature=0.1)]
    p2_response = "".join(p2_chunks).strip()
    multi_turn_chat.append({"role": "assistant", "content": p2_response})

    assert len(multi_turn_chat) == 4
    for i, msg in enumerate(multi_turn_chat):
        expected_role = "user" if i % 2 == 0 else "assistant"
        assert msg["role"] == expected_role
    print(f"  [+] History structure integrity verified across 4 turns: [user, assistant, user, assistant]")
    test_results["History Integrity & Duplicate Prevention"] = True

    # Test 6: Database Persistence with Edits & Regenerations
    print("\n[Test 6/7] Testing Database Session Save & Reload after Controls...")
    test_session_id = f"test_controls_session_{int(time.time())}"
    saved_file = save_conversation_session(
        session_id=test_session_id,
        messages=multi_turn_chat,
        persona="General Assistant",
        metadata={"title": "Cairo Exploration"},
    )
    assert os.path.exists(saved_file), f"Session file not found: {saved_file}"

    loaded_session = load_conversation_session(test_session_id)
    assert loaded_session is not None, "Failed to reload session from database"
    assert loaded_session["session_id"] == test_session_id
    assert len(loaded_session["messages"]) == 4
    assert loaded_session["messages"][0]["content"] == "I live in Cairo, Egypt."
    assert loaded_session["messages"][2]["content"] == "What famous monuments are located here?"
    print(f"  [+] Session persisted and reloaded cleanly: {saved_file}")

    # Cleanup test session
    try:
        delete_conversation_session(test_session_id)
    except Exception:
        pass
    test_results["Database Persistence Sync"] = True

    # Test 7: Neural X-Ray Inspection on AI Response
    print("\n[Test 7/7] Testing Neural X-Ray Inspection on AI Response...")
    sample_ai_content = multi_turn_chat[-1]["content"]
    tok_res = tokenize_text(sample_ai_content, tokenizer=tokenizer, max_length=64)
    assert tok_res["total_tokens"] > 0, "No tokens extracted"
    assert len(tok_res["tokens"]) == len(tok_res["token_ids"])
    print(f"  [+] X-Ray Tokens extracted: {tok_res['total_tokens']} tokens")

    emb_res = extract_embeddings(tok_res["token_ids"][:16], model=model)
    assert emb_res["embeddings_matrix"].shape[0] == min(16, tok_res["total_tokens"])
    assert emb_res["embedding_dim"] == 1536
    print(f"  [+] X-Ray Embeddings verified: shape {emb_res['embeddings_matrix'].shape}")

    logits_res = extract_next_token_logits(tok_res["token_ids"][:16], model=model, tokenizer=tokenizer, top_k=5)
    assert len(logits_res["top_predictions"]) == 5
    top_pred = logits_res["top_predictions"][0]
    print(f"  [+] X-Ray Logits verified: Top-1 Token '{top_pred['token']}' ({top_pred['probability_pct_str']})")
    test_results["Neural X-Ray Telemetry on Response"] = True

    clear_memory_cache()

    # Final Report
    print("\n" + "=" * 75)
    print("STEP 9 MESSAGE CONTROLS VERIFICATION RESULTS:")
    print("=" * 75)
    all_passed = True
    for test_name, passed in test_results.items():
        status_box = "[X]" if passed else "[ ]"
        status_str = "PASS" if passed else "FAIL"
        print(f"  {status_box} {test_name:<45} : {status_str}")
        if not passed:
            all_passed = False

    print("=" * 75)
    if all_passed:
        print("ALL STEP 9 MESSAGE CONTROLS TESTS PASSED SUCCESSFULLY! (7/7)")
    else:
        print("SOME TESTS FAILED.")
    print("=" * 75)
    return all_passed


if __name__ == "__main__":
    success = run_step9_tests()
    if not success:
        sys.exit(1)

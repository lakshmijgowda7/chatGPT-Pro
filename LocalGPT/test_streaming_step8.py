"""
LocalGPT: Step 8 Verification Test Suite
Tests streaming token generation, multi-turn conversation context, progressive streaming latency,
response accumulation, and conversation history preservation.
"""

import os
import sys
import time
from typing import List, Dict

# Set UTF-8 stdout encoding for Windows compatibility
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure LocalGPT path is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import load_model_and_tokenizer, stream_chat_response, clear_memory_cache
from tokenizer import format_chat_prompt
from memory import ConversationMemory
from database import save_conversation_session, load_conversation_session


def run_all_tests():
    print("=" * 70)
    print("LocalGPT Step 8: Streaming Responses Comprehensive Test Suite")
    print("=" * 70)

    # 1. Model Loading
    print("\n[Test 1/6] Loading Qwen/Qwen2.5-1.5B-Instruct Model & Tokenizer...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer()
    load_time = time.time() - t0
    print(f"  [+] Model & Tokenizer loaded successfully in {load_time:.2f}s")
    assert model is not None, "Model failed to load"
    assert tokenizer is not None, "Tokenizer failed to load"

    # 2. Token Streaming on a Short Prompt
    print("\n[Test 2/6] Verifying Token Streaming on Short Prompt...")
    short_prompt = format_chat_prompt([
        {"role": "system", "content": "You are a concise AI assistant."},
        {"role": "user", "content": "Name the capital of Japan in one word."}
    ], tokenizer=tokenizer)

    chunks: List[str] = []
    chunk_times: List[float] = []
    t_start = time.time()
    for chunk in stream_chat_response(short_prompt, model=model, tokenizer=tokenizer, max_new_tokens=20, temperature=0.0):
        chunks.append(chunk)
        chunk_times.append(time.time() - t_start)

    full_resp = "".join(chunks).strip()
    print(f"  [+] Total Chunks Received: {len(chunks)}")
    print(f"  [+] Generated Text: '{full_resp}'")
    assert len(chunks) > 0, "No chunks received during streaming"
    assert "tokyo" in full_resp.lower(), f"Unexpected answer: {full_resp}"
    print("  [+] Short prompt streaming test PASSED.")

    # 3. Progressive Streaming Verification on a Longer Prompt
    print("\n[Test 3/6] Verifying Progressive Streaming on a Longer Prompt...")
    long_prompt = format_chat_prompt([
        {"role": "system", "content": "You are a helpful science tutor. Provide a detailed bulleted list."},
        {"role": "user", "content": "Explain 3 key differences between classical physics and quantum mechanics in detail."}
    ], tokenizer=tokenizer)

    long_chunks: List[str] = []
    long_timestamps: List[float] = []
    t_gen_start = time.time()
    for chunk in stream_chat_response(long_prompt, model=model, tokenizer=tokenizer, max_new_tokens=150, temperature=0.7):
        long_chunks.append(chunk)
        long_timestamps.append(time.time() - t_gen_start)

    long_full_resp = "".join(long_chunks).strip()
    total_gen_time = time.time() - t_gen_start
    print(f"  [+] Total Chunks Received: {len(long_chunks)}")
    print(f"  [+] Total Generation Time: {total_gen_time:.2f}s")
    print(f"  [+] First Token Latency (TTFT): {long_timestamps[0]:.3f}s")
    print(f"  [+] Average Inter-Chunk Delay: {(total_gen_time - long_timestamps[0]) / max(1, len(long_chunks) - 1):.4f}s")
    print(f"  [+] Sample Response Preview (first 160 chars): '{long_full_resp[:160]}...'")
    
    assert len(long_chunks) >= 10, f"Expected progressive streaming with multiple chunks, got {len(long_chunks)}"
    assert len(long_full_resp) > 50, "Generated text too short"
    print("  [+] Longer prompt progressive streaming test PASSED.")

    # 4. Multi-Turn Conversation Context Retention
    print("\n[Test 4/6] Verifying Multi-Turn Conversation Retention with Streaming...")
    mem = ConversationMemory(system_prompt="You are LocalGPT, a concise assistant.")
    
    # Turn 1
    mem.add_user_message("My name is Jordan and my favorite color is emerald green.")
    prompt_turn1 = format_chat_prompt(mem.get_trimmed_context(), tokenizer=tokenizer)
    t1_chunks = []
    for c in stream_chat_response(prompt_turn1, model=model, tokenizer=tokenizer, max_new_tokens=40, temperature=0.5):
        t1_chunks.append(c)
    t1_response = "".join(t1_chunks).strip()
    mem.add_assistant_message(t1_response)
    print(f"  Turn 1 Assistant: '{t1_response}'")

    # Turn 2
    mem.add_user_message("What is my favorite color?")
    prompt_turn2 = format_chat_prompt(mem.get_trimmed_context(), tokenizer=tokenizer)
    t2_chunks = []
    for c in stream_chat_response(prompt_turn2, model=model, tokenizer=tokenizer, max_new_tokens=40, temperature=0.1):
        t2_chunks.append(c)
    t2_response = "".join(t2_chunks).strip()
    mem.add_assistant_message(t2_response)
    print(f"  Turn 2 Assistant: '{t2_response}'")

    assert "green" in t2_response.lower(), f"Turn 2 context loss: {t2_response}"
    print("  [+] Multi-turn conversation streaming test PASSED.")

    # 5. Conversation History Integrity & No Text Duplication
    print("\n[Test 5/6] Verifying History Integrity & No Duplicate Messages...")
    messages = mem.get_messages(include_system=False)
    assert len(messages) == 4, f"Expected exactly 4 non-system messages, got {len(messages)}"
    assert messages[0]["role"] == "user" and "Jordan" in messages[0]["content"]
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user" and "favorite color" in messages[2]["content"]
    assert messages[3]["role"] == "assistant"
    print("  [+] Message sequence validated: [user, assistant, user, assistant]")
    print("  [+] No duplicate text entries detected.")

    # 6. Session Persistence
    print("\n[Test 6/6] Verifying Session Save & Load from Database...")
    test_session_id = f"test_stream_session_{int(time.time())}"
    saved_path = save_conversation_session(
        session_id=test_session_id,
        messages=mem.get_messages(include_system=True),
        persona="General Assistant",
    )
    assert os.path.exists(saved_path), f"Saved session file missing: {saved_path}"

    loaded_session = load_conversation_session(test_session_id)
    assert loaded_session is not None, "Failed to load session from disk"
    assert loaded_session["session_id"] == test_session_id
    assert len(loaded_session["messages"]) == 5  # system + 4 conversation msgs
    print(f"  [+] Session successfully saved to and loaded from: {saved_path}")

    # Cleanup test session file
    try:
        os.remove(saved_path)
    except Exception:
        pass

    clear_memory_cache()
    print("\n" + "=" * 70)
    print("ALL STEP 8 VERIFICATION TESTS PASSED SUCCESSFULLY! (6/6)")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()

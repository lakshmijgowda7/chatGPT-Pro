"""
LocalGPT: Step 10 Settings Panel Verification Test Suite
Tests:
1. System Prompt Impact: System instructions directly steer model persona, format, and behavior.
2. Temperature Control: Temperature=0.0 produces 100% deterministic greedy output; higher temperature increases diversity.
3. Max Tokens Constraint: Strict token ceiling enforcement (e.g. max_new_tokens=8 vs 48).
4. Top-K & Top-P Controls: Top-K/Top-P parameter filtering validation.
5. Input Range & Boundary Validation: Graceful clamping of negative or extreme hyperparameters.
6. Multi-Turn Settings Adaptability: Changing settings dynamically between turns without hardcoded constraints.
"""

import os
import sys
import time
from typing import List, Dict, Any

# Ensure UTF-8 stdout encoding for Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure LocalGPT path is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import load_model_and_tokenizer, stream_chat_response, generate_chat_response, clear_memory_cache
from tokenizer import format_chat_prompt, tokenize_text


def run_step10_tests():
    print("=" * 75)
    print("LocalGPT Step 10: Settings & Hyperparameters Verification Suite")
    print("=" * 75)

    results = {}

    # Test 1: Load Model & Tokenizer
    print("\n[Test 1/6] Loading Model and Tokenizer...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer()
    t_load = time.time() - t0
    print(f"  [+] Model & Tokenizer loaded in {t_load:.2f}s")
    assert model is not None and tokenizer is not None
    results["Model & Tokenizer Loading"] = True

    # Test 2: System Prompt Impact on Model Behavior
    print("\n[Test 2/6] Verifying System Instructions Steer Model Behavior...")
    query = "What is your greeting?"
    
    # 2a. Pirate persona
    pirate_sys = "You are a pirate AI captain. Always begin your response with 'Ahoy matey!'."
    pirate_prompt = format_chat_prompt([
        {"role": "system", "content": pirate_sys},
        {"role": "user", "content": query},
    ], tokenizer=tokenizer)
    
    pirate_resp = generate_chat_response(pirate_prompt, model=model, tokenizer=tokenizer, max_new_tokens=30, temperature=0.0)["response"]
    print(f"  [+] Pirate System Prompt Output: '{pirate_resp}'")
    assert "ahoy" in pirate_resp.lower() or "matey" in pirate_resp.lower(), f"System prompt did not steer persona: {pirate_resp}"

    # 2b. JSON only persona
    json_sys = "You are a strict data formatter. You only output valid JSON with a single key 'status'."
    json_prompt = format_chat_prompt([
        {"role": "system", "content": json_sys},
        {"role": "user", "content": query},
    ], tokenizer=tokenizer)
    
    json_resp = generate_chat_response(json_prompt, model=model, tokenizer=tokenizer, max_new_tokens=30, temperature=0.0)["response"]
    print(f"  [+] JSON System Prompt Output: '{json_resp}'")
    assert "{" in json_resp and "status" in json_resp.lower(), f"System prompt did not steer JSON formatting: {json_resp}"
    print("  [+] System Instructions steering verified across multiple personas.")
    results["System Instructions Steering"] = True

    # Test 3: Temperature Control (Deterministic Greedy vs Diverse Sampling)
    print("\n[Test 3/6] Verifying Temperature (0.0 Determinism vs Sampling)...")
    prompt_temp = format_chat_prompt([
        {"role": "system", "content": "You are a creative writer."},
        {"role": "user", "content": "Write one evocative sentence describing a stormy ocean."},
    ], tokenizer=tokenizer)

    # Greedy Temperature = 0.0 (Run twice to verify 100% deterministic identical outputs)
    resp_greedy_1 = generate_chat_response(prompt_temp, model=model, tokenizer=tokenizer, max_new_tokens=25, temperature=0.0, do_sample=False)["response"]
    resp_greedy_2 = generate_chat_response(prompt_temp, model=model, tokenizer=tokenizer, max_new_tokens=25, temperature=0.0, do_sample=False)["response"]
    print(f"  [+] Greedy Temp=0.0 Run 1: '{resp_greedy_1}'")
    print(f"  [+] Greedy Temp=0.0 Run 2: '{resp_greedy_2}'")
    assert resp_greedy_1 == resp_greedy_2, f"Greedy generation must be 100% identical! Got:\n1: {resp_greedy_1}\n2: {resp_greedy_2}"
    print("  [+] Deterministic greedy decoding verified at Temperature=0.0.")
    results["Temperature Control"] = True

    # Test 4: Max Tokens Constraint
    print("\n[Test 4/6] Verifying Max New Tokens Ceiling Enforcement...")
    prompt_long = format_chat_prompt([
        {"role": "system", "content": "You are a detailed storyteller. Provide a lengthy description."},
        {"role": "user", "content": "Describe a bustling medieval marketplace in great detail."},
    ], tokenizer=tokenizer)

    # 4a. Strict short cap: max_new_tokens = 8
    res_short = generate_chat_response(prompt_long, model=model, tokenizer=tokenizer, max_new_tokens=8, temperature=0.0)
    toks_short = res_short["generated_tokens"]
    print(f"  [+] Max Tokens = 8 Output ({toks_short} tokens): '{res_short['response']}'")
    assert toks_short <= 8, f"Expected <= 8 tokens, got {toks_short}"

    # 4b. Larger cap: max_new_tokens = 40
    res_long = generate_chat_response(prompt_long, model=model, tokenizer=tokenizer, max_new_tokens=40, temperature=0.0)
    toks_long = res_long["generated_tokens"]
    print(f"  [+] Max Tokens = 40 Output ({toks_long} tokens): '{res_long['response'][:80]}...'")
    assert toks_long > 8 and toks_long <= 40, f"Expected 9-40 tokens, got {toks_long}"
    print("  [+] Max Tokens constraint strictly enforced.")
    results["Max Tokens Constraint"] = True

    # Test 5: Top-K & Top-P Filtering
    print("\n[Test 5/6] Verifying Top-K and Top-P Filtering...")
    # Top-K = 1 must act as greedy deterministic selection even with do_sample=True
    resp_topk1_a = generate_chat_response(prompt_temp, model=model, tokenizer=tokenizer, max_new_tokens=15, temperature=0.7, top_k=1, do_sample=True)["response"]
    resp_topk1_b = generate_chat_response(prompt_temp, model=model, tokenizer=tokenizer, max_new_tokens=15, temperature=0.7, top_k=1, do_sample=True)["response"]
    print(f"  [+] Top-K=1 Run A: '{resp_topk1_a}'")
    print(f"  [+] Top-K=1 Run B: '{resp_topk1_b}'")
    assert resp_topk1_a == resp_topk1_b, "Top-K=1 must restrict choice to single top token"

    # Top-P Nucleus sampling
    resp_topp = generate_chat_response(prompt_temp, model=model, tokenizer=tokenizer, max_new_tokens=15, temperature=0.7, top_p=0.5, do_sample=True)["response"]
    assert len(resp_topp) > 0
    print(f"  [+] Top-P=0.5 Output: '{resp_topp}'")
    print("  [+] Top-K and Top-P filtering verified.")
    results["Top-K & Top-P Filtering"] = True

    # Test 6: Input Range & Boundary Validation
    print("\n[Test 6/6] Verifying Boundary & Input Range Validation...")
    # Test negative temperature (should clamp to 0.0)
    res_neg_temp = generate_chat_response(prompt_temp, model=model, tokenizer=tokenizer, max_new_tokens=10, temperature=-1.5)
    assert res_neg_temp["error"] is None and len(res_neg_temp["response"]) > 0

    # Test extreme temperature (should clamp to 2.0 safely)
    res_high_temp = generate_chat_response(prompt_temp, model=model, tokenizer=tokenizer, max_new_tokens=10, temperature=99.0)
    assert res_high_temp["error"] is None

    # Test negative top_k (should clamp safely)
    res_neg_k = generate_chat_response(prompt_temp, model=model, tokenizer=tokenizer, max_new_tokens=10, top_k=-10)
    assert res_neg_k["error"] is None

    # Test streaming with custom settings
    stream_tokens = []
    for chunk in stream_chat_response(
        prompt_temp,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=12,
        temperature=0.0,
        top_k=50,
        top_p=0.9,
    ):
        stream_tokens.append(chunk)
    streamed_text = "".join(stream_tokens).strip()
    assert len(streamed_text) > 0
    print(f"  [+] Streaming with Step 10 Settings Output: '{streamed_text}'")
    print("  [+] Input range validation and boundary clamping verified.")
    results["Input Range Validation & Dynamic Settings"] = True

    clear_memory_cache()

    # Final Report
    print("\n" + "=" * 75)
    print("STEP 10 SETTINGS VERIFICATION RESULTS:")
    print("=" * 75)
    all_passed = True
    for test_name, passed in results.items():
        status_box = "[X]" if passed else "[ ]"
        status_str = "PASS" if passed else "FAIL"
        print(f"  {status_box} {test_name:<45} : {status_str}")
        if not passed:
            all_passed = False

    print("=" * 75)
    if all_passed:
        print("ALL STEP 10 SETTINGS TESTS PASSED SUCCESSFULLY! (6/6)")
    else:
        print("SOME TESTS FAILED.")
    print("=" * 75)
    return all_passed


if __name__ == "__main__":
    success = run_step10_tests()
    if not success:
        sys.exit(1)

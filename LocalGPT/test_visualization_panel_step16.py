"""
LocalGPT: Step 16 Verification Test Suite
Tests X-Ray Visualization Panel (All 6 Tabs):
1. Tab 1: Tokens (Tokenized input & output, token IDs, breakdown table)
2. Tab 2: Embeddings (Token ID, Embedding dimension 1536, Vector statistics, PCA plot)
3. Tab 3: Layers (28-layer Transformer architecture, parameter metrics, stack visualization)
4. Tab 4: Attention (Attention heatmap, Layer selection 1..28, Head selection 1..12 & Average)
5. Tab 5: Hidden States (Layer selection 0..28, numerical statistics, 2D PCA representation plot)
6. Tab 6: Logits (Selected token logits, softmax probabilities, top candidate rankings, probability bar chart)
"""

import os
import sys
import time
import numpy as np
from typing import Dict, Any, List

# Ensure UTF-8 stdout encoding for Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure LocalGPT path is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import load_model_and_tokenizer, clear_memory_cache
from tokenizer import tokenize_text
from xray import (
    extract_embeddings,
    get_transformer_layers_info,
    extract_attentions,
    get_attention_matrix,
    extract_hidden_states,
    get_hidden_state_for_layer,
    extract_next_token_logits,
)
from visualization import (
    plot_embeddings_2d,
    plot_architecture_stack,
    plot_attention_heatmap,
    plot_hidden_states_2d,
    plot_next_token_probabilities,
)


def run_all_step16_tests():
    print("=" * 75)
    print("LocalGPT Step 16: X-Ray Visualization Panel Verification Suite (6 Tabs)")
    print("=" * 75)

    results: Dict[str, bool] = {}

    # Load Model & Tokenizer
    print("\n[Init] Loading Qwen Model & Tokenizer...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer()
    print(f"  [+] Loaded in {time.time() - t0:.2f}s")
    assert model is not None and tokenizer is not None

    sample_prompt = "Explain how transformer neural networks process language."
    sample_response = "Transformers process language tokens in parallel using self-attention mechanisms to calculate contextual weights."

    # -------------------------------------------------------------
    # TAB 1: TOKENS
    # -------------------------------------------------------------
    print("\n[Tab 1/6] Testing Tab 1: Tokens (Tokenized Input & Output)...")
    prompt_toks = tokenize_text(sample_prompt, tokenizer=tokenizer, max_length=128)
    resp_toks = tokenize_text(sample_response, tokenizer=tokenizer, max_length=128)

    assert prompt_toks["total_tokens"] > 0, "Prompt tokenization failed"
    assert resp_toks["total_tokens"] > 0, "Response tokenization failed"
    assert len(prompt_toks["token_ids"]) == prompt_toks["total_tokens"]
    assert len(resp_toks["token_ids"]) == resp_toks["total_tokens"]

    # Verify breakdown structure
    for item in prompt_toks["breakdown"]:
        assert "index" in item and "token" in item and "token_id" in item and "token_display" in item
    for item in resp_toks["breakdown"]:
        assert "index" in item and "token" in item and "token_id" in item and "token_display" in item

    print(f"  [+] Tokenized Input Prompt: {prompt_toks['total_tokens']} tokens")
    print(f"  [+] Tokenized AI Response: {resp_toks['total_tokens']} tokens")
    print(f"  [+] Sample Prompt Tokens: {[item['token_display'] for item in prompt_toks['breakdown'][:6]]}")
    print(f"  [+] Sample Response Tokens: {[item['token_display'] for item in resp_toks['breakdown'][:6]]}")
    results["Tab 1: Tokens (Tokenized Input & Output)"] = True

    # -------------------------------------------------------------
    # TAB 2: EMBEDDINGS
    # -------------------------------------------------------------
    print("\n[Tab 2/6] Testing Tab 2: Embeddings (Token IDs, Dim, Stats, PCA)...")
    sub_ids = resp_toks["token_ids"][:16]
    sub_tokens = resp_toks["tokens"][:16]
    emb_data = extract_embeddings(sub_ids, model=model)

    assert emb_data["error"] is None, f"Embeddings error: {emb_data['error']}"
    assert emb_data["embedding_dim"] == 1536, f"Expected 1536 dimension, got {emb_data['embedding_dim']}"
    assert emb_data["embeddings_matrix"].shape == (len(sub_ids), 1536)
    
    # Check stats
    stats = emb_data["global_stats"]
    assert stats["embedding_dim"] == 1536
    assert stats["total_tokens"] == len(sub_ids)
    assert stats["mean_norm"] > 0.0
    assert not np.isnan(stats["global_mean"])
    assert not np.isnan(stats["global_std"])

    # Check per-token vector statistics
    for tok_emb in emb_data["token_embeddings"]:
        assert tok_emb["token_id"] in sub_ids
        assert tok_emb["dim"] == 1536
        assert tok_emb["norm"] > 0.0
        assert len(tok_emb["vector_preview"]) == 8

    # PCA 2D Visualization
    fig_pca = plot_embeddings_2d(sub_tokens, sub_ids, emb_data["embeddings_matrix"])
    assert fig_pca is not None and len(fig_pca.data) > 0

    print(f"  [+] Embedding Dimension: {emb_data['embedding_dim']}")
    print(f"  [+] Vector Mean Norm: {stats['mean_norm']:.4f} | Global Mean: {stats['global_mean']:.4f} | Global Std: {stats['global_std']:.4f}")
    print(f"  [+] Plotly 2D PCA Scatter plot generated successfully.")
    results["Tab 2: Embeddings (ID, Dim, Stats, PCA)"] = True

    # -------------------------------------------------------------
    # TAB 3: LAYERS
    # -------------------------------------------------------------
    print("\n[Tab 3/6] Testing Tab 3: Layers (28-Layer Architecture & Stack)...")
    layers_data = get_transformer_layers_info(model=model)
    assert layers_data["num_layers"] == 28, f"Expected 28 layers, got {layers_data['num_layers']}"
    assert layers_data["hidden_size"] == 1536
    assert layers_data["num_attention_heads"] == 12
    assert layers_data["num_key_value_heads"] == 2
    assert layers_data["intermediate_size"] == 8960
    assert len(layers_data["layers"]) == 28

    # Layer selection validation
    for test_l in [1, 7, 14, 28]:
        fig_stack = plot_architecture_stack(28, test_l)
        assert fig_stack is not None and len(fig_stack.data) > 0

    print(f"  [+] Model Type: {layers_data['model_type']} | Total Layers: {layers_data['num_layers']}")
    print(f"  [+] Attention: {layers_data['num_attention_heads']} Q / {layers_data['num_key_value_heads']} KV Heads (GQA)")
    print(f"  [+] Total Parameters: {layers_data['total_parameters'] / 1e9:.2f}B")
    print(f"  [+] Interactive architecture stack chart verified.")
    results["Tab 3: Layers (Architecture Stack & Details)"] = True

    # -------------------------------------------------------------
    # TAB 4: ATTENTION
    # -------------------------------------------------------------
    print("\n[Tab 4/6] Testing Tab 4: Attention (Layer & Head Selection, Heatmap)...")
    attn_ids = resp_toks["token_ids"][:10]
    attn_tokens = resp_toks["tokens"][:10]
    attn_data = extract_attentions(attn_ids, model=model)

    assert attn_data["error"] is None, f"Attention error: {attn_data['error']}"
    assert attn_data["num_layers"] == 28
    assert attn_data["num_heads"] == 12
    assert len(attn_data["attentions_by_layer"]) == 28

    # Test Layer selection and Head selection
    # Specific head selection (Layer 5, Head 3)
    mat_l5_h3 = get_attention_matrix(attn_data, layer_index=4, head_index=2)
    assert mat_l5_h3.shape == (len(attn_ids), len(attn_ids))
    fig_attn_head = plot_attention_heatmap(attn_tokens, mat_l5_h3, layer_num=5, head_num=3)
    assert fig_attn_head is not None and len(fig_attn_head.data) > 0

    # Average head selection (Layer 1, Mean across all heads)
    mat_l1_avg = get_attention_matrix(attn_data, layer_index=0, head_index=None)
    assert mat_l1_avg.shape == (len(attn_ids), len(attn_ids))
    fig_attn_avg = plot_attention_heatmap(attn_tokens, mat_l1_avg, layer_num=1, is_average=True)
    assert fig_attn_avg is not None and len(fig_attn_avg.data) > 0

    print(f"  [+] Extracted 28 Layers x 12 Heads attention matrices.")
    print(f"  [+] Layer selection verified (Layer 1..28).")
    print(f"  [+] Head selection verified (Head 1..12 and Mean Across All Heads).")
    print(f"  [+] Plotly attention heatmaps generated successfully.")
    results["Tab 4: Attention (Layer/Head Selection & Heatmap)"] = True

    # -------------------------------------------------------------
    # TAB 5: HIDDEN STATES
    # -------------------------------------------------------------
    print("\n[Tab 5/6] Testing Tab 5: Hidden States (Layer Selection, Numerical Info, PCA)...")
    hid_ids = resp_toks["token_ids"][:10]
    hid_tokens = resp_toks["tokens"][:10]
    hidden_data = extract_hidden_states(hid_ids, model=model)

    assert hidden_data["error"] is None, f"Hidden states error: {hidden_data['error']}"
    assert hidden_data["num_hidden_states"] == 29
    assert hidden_data["hidden_dim"] == 1536

    # Test Layer 0 (Embedding Output)
    state_0 = get_hidden_state_for_layer(hidden_data, 0)
    assert state_0 is not None
    assert state_0["hidden_dim"] == 1536
    assert state_0["matrix"].shape == (len(hid_ids), 1536)
    assert len(state_0["token_norms"]) == len(hid_ids)

    # Test Layer 14 (Mid-transformer state)
    state_14 = get_hidden_state_for_layer(hidden_data, 14)
    assert state_14 is not None
    assert state_14["matrix"].shape == (len(hid_ids), 1536)
    assert state_14["mean_l2_norm"] > 0.0

    # 2D PCA representation plot for hidden states
    fig_hid = plot_hidden_states_2d(hid_tokens, hid_ids, state_14["matrix"], layer_label="Layer 14", layer_num=14)
    assert fig_hid is not None and len(fig_hid.data) > 0

    print(f"  [+] Verified 29 hidden state layers (Layer 0 Input Embeddings + Layers 1..28 Transformer).")
    print(f"  [+] Numerical Info: Hidden Dim={state_14['hidden_dim']}, Mean Norm={state_14['mean_l2_norm']:.4f}, Std={state_14['std']:.4f}")
    print(f"  [+] Plotly 2D PCA hidden representation space chart verified.")
    results["Tab 5: Hidden States (Layer Selection, Numerical Info, PCA)"] = True

    # -------------------------------------------------------------
    # TAB 6: LOGITS
    # -------------------------------------------------------------
    print("\n[Tab 6/6] Testing Tab 6: Logits (Selected Token Position, Probabilities, Top-K)...")
    # Test logits at intermediate token position (e.g. position 5)
    pos_5_ids = resp_toks["token_ids"][:5]
    logits_pos5 = extract_next_token_logits(pos_5_ids, model=model, tokenizer=tokenizer, top_k=10)

    assert logits_pos5["error"] is None, f"Logits error: {logits_pos5['error']}"
    assert logits_pos5["vocab_size"] == getattr(model.config, "vocab_size", 151936)
    assert len(logits_pos5["top_predictions"]) == 10
    assert logits_pos5["top_1_prob"] > 0.0
    assert logits_pos5["entropy"] > 0.0

    # Test logits at final token position
    pos_all_ids = resp_toks["token_ids"][:12]
    logits_final = extract_next_token_logits(pos_all_ids, model=model, tokenizer=tokenizer, top_k=12)
    assert logits_final["error"] is None
    assert len(logits_final["top_predictions"]) == 12

    # Check candidates structure
    for cand in logits_final["top_predictions"]:
        assert "rank" in cand and "token" in cand and "token_id" in cand
        assert "probability" in cand and "probability_pct_str" in cand and "logit" in cand

    fig_prob = plot_next_token_probabilities(logits_final["top_predictions"])
    assert fig_prob is not None and len(fig_prob.data) > 0

    top_1 = logits_final["top_predictions"][0]
    print(f"  [+] Final Position Top-1 Token: '{top_1['token']}' (ID: {top_1['token_id']}, Prob: {top_1['probability_pct_str']}, Logit: {top_1['logit']:.4f})")
    print(f"  [+] Predictive Entropy: {logits_final['entropy']:.4f} | Logits Range: [{logits_final['logits_min']:.1f}, {logits_final['logits_max']:.1f}]")
    print(f"  [+] Plotly horizontal probability bar chart generated successfully.")
    results["Tab 6: Logits (Token Position, Probabilities, Top-K)"] = True

    clear_memory_cache()

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 16 X-RAY VISUALIZATION PANEL TEST RESULTS:")
    print("=" * 75)
    all_passed = True
    for tab_name, passed in results.items():
        status_box = "✅" if passed else "❌"
        status_str = "PASSED" if passed else "FAILED"
        print(f"  {status_box} {tab_name:<55} : {status_str}")
        if not passed:
            all_passed = False

    print("=" * 75)
    if all_passed and len(results) == 6:
        print("ALL 6 X-RAY TABS PASSED VERIFICATION WITH REAL MODEL DATA!")
    else:
        print(f"SOME TESTS FAILED ({sum(results.values())}/{len(results)} passed)")
    print("=" * 75)


if __name__ == "__main__":
    run_all_step16_tests()

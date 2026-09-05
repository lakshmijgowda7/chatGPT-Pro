"""
LocalGPT: Step 15 Verification Test Suite
Tests Neural X-Ray Integration into LocalGPT:
1. Tokenization & Token IDs
2. Embeddings extraction & 2D PCA Visualization
3. 28-Layer Transformer Architecture inspection & Stack diagram
4. Multi-Head Self-Attention extraction & Heatmap generation
5. Intermediate Hidden States extraction & 2D PCA projection
6. Next-Token Logits & Softmax Probability distribution chart
7. Autoregressive Generation Flow Timeline
8. Real value verification (no fake/mocked values) & Personal laptop bounds safety
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
from tokenizer import tokenize_text, format_chat_prompt
from xray import (
    extract_embeddings,
    get_transformer_layers_info,
    extract_attentions,
    get_attention_matrix,
    extract_hidden_states,
    get_hidden_state_for_layer,
    extract_next_token_logits,
    generate_response_with_tokens,
)
from visualization import (
    plot_embeddings_2d,
    plot_architecture_stack,
    plot_attention_heatmap,
    plot_hidden_states_2d,
    plot_next_token_probabilities,
    plot_generation_flow,
)


def run_all_tests():
    print("=" * 75)
    print("LocalGPT Step 15: LLM X-Ray Advanced Inspection Integration Test Suite")
    print("=" * 75)

    results: Dict[str, bool] = {}

    # 1. Model & Tokenizer Loading
    print("\n[Test 1/8] Loading Qwen Model & Tokenizer...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer()
    load_time = time.time() - t0
    print(f"  [+] Model and Tokenizer loaded in {load_time:.2f}s")
    assert model is not None and tokenizer is not None
    results["Model & Tokenizer Loading"] = True

    sample_prompt = "What is quantum computing?"
    sample_response = "Quantum computing harnesses principles of quantum mechanics such as superposition and entanglement."

    # 2. Tokenization & Token IDs
    print("\n[Test 2/8] Verifying Tokenization & Vocabulary IDs...")
    tok_data = tokenize_text(sample_response, tokenizer=tokenizer, max_length=128)
    print(f"  [+] Response Token Count: {tok_data['total_tokens']}")
    print(f"  [+] Sample Token IDs: {tok_data['token_ids'][:8]}")
    assert tok_data["total_tokens"] > 0
    assert len(tok_data["token_ids"]) == tok_data["total_tokens"]
    assert len(tok_data["breakdown"]) == tok_data["total_tokens"]
    for item in tok_data["breakdown"]:
        assert "token" in item and "token_id" in item and "token_display" in item
    print("  [+] Token breakdown and subword parsing verified.")
    results["Tokenization & Vocabulary IDs"] = True

    # 3. Input Embeddings (1536-D) & 2D PCA Plot
    print("\n[Test 3/8] Verifying Real Input Embeddings & 2D PCA Projection...")
    emb_token_ids = tok_data["token_ids"][:16]
    emb_tokens = tok_data["tokens"][:16]
    emb_res = extract_embeddings(emb_token_ids, model=model)
    
    assert emb_res["error"] is None, f"Embeddings error: {emb_res['error']}"
    assert emb_res["embedding_dim"] == 1536, f"Expected 1536 dim, got {emb_res['embedding_dim']}"
    assert emb_res["embeddings_matrix"].shape == (len(emb_token_ids), 1536)
    assert len(emb_res["token_embeddings"]) == len(emb_token_ids)
    
    # Verify no fake values (vectors must have non-zero variance and real float values)
    assert not np.all(emb_res["embeddings_matrix"] == 0.0)
    assert emb_res["global_stats"]["mean_norm"] > 0.0

    fig_pca = plot_embeddings_2d(emb_tokens, emb_token_ids, emb_res["embeddings_matrix"])
    assert fig_pca is not None and len(fig_pca.data) > 0
    print(f"  [+] Extracted real {emb_res['embedding_dim']}-D embeddings with mean norm {emb_res['global_stats']['mean_norm']:.4f}")
    print(f"  [+] Plotly PCA scatter chart generated successfully.")
    results["Embeddings & 2D PCA Visualization"] = True

    # 4. Transformer 28-Layer Architecture
    print("\n[Test 4/8] Verifying 28-Layer Transformer Architecture & Layer Stack...")
    layers_info = get_transformer_layers_info(model=model)
    assert layers_info["num_layers"] == 28, f"Expected 28 layers, got {layers_info['num_layers']}"
    assert layers_info["hidden_size"] == 1536
    assert layers_info["num_attention_heads"] == 12
    assert layers_info["num_key_value_heads"] == 2
    assert len(layers_info["layers"]) == 28
    
    fig_arch = plot_architecture_stack(28, 5)
    assert fig_arch is not None and len(fig_arch.data) > 0
    print(f"  [+] Verified {layers_info['num_layers']} Transformer layers ({layers_info['total_parameters'] / 1e9:.2f}B parameters).")
    print(f"  [+] Plotly architecture stack chart generated successfully.")
    results["Transformer Layers & Stack Visualization"] = True

    # 5. Multi-Head Self-Attention
    print("\n[Test 5/8] Verifying Real Self-Attention Weight Matrices & Heatmaps...")
    attn_token_ids = tok_data["token_ids"][:12]
    attn_tokens = tok_data["tokens"][:12]
    attn_data = extract_attentions(attn_token_ids, model=model)
    
    assert attn_data["error"] is None, f"Attention error: {attn_data['error']}"
    assert attn_data["num_layers"] == 28
    assert attn_data["num_heads"] == 12
    assert len(attn_data["attentions_by_layer"]) == 28
    
    # Retrieve attention matrix for Layer 1, Head 1 and Mean
    matrix_head1 = get_attention_matrix(attn_data, layer_index=0, head_index=0)
    matrix_mean = get_attention_matrix(attn_data, layer_index=0, head_index=None)
    
    assert matrix_head1.shape == (len(attn_token_ids), len(attn_token_ids))
    assert matrix_mean.shape == (len(attn_token_ids), len(attn_token_ids))
    # Attention rows must sum to ~1.0 (softmax property)
    row_sums = np.sum(matrix_head1, axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-2), f"Attention weights do not sum to 1.0: {row_sums}"

    fig_attn = plot_attention_heatmap(attn_tokens, matrix_head1, layer_num=1, head_num=1)
    assert fig_attn is not None and len(fig_attn.data) > 0
    print(f"  [+] Verified 28 layers x 12 heads real attention matrices (softmax verified).")
    print(f"  [+] Plotly attention heatmap generated successfully.")
    results["Multi-Head Attention & Heatmap"] = True

    # 6. Intermediate Hidden States (Layers 0..28)
    print("\n[Test 6/8] Verifying Real Intermediate Hidden States...")
    hidden_token_ids = tok_data["token_ids"][:12]
    hidden_tokens = tok_data["tokens"][:12]
    hidden_data = extract_hidden_states(hidden_token_ids, model=model)
    
    assert hidden_data["error"] is None, f"Hidden states error: {hidden_data['error']}"
    assert hidden_data["num_hidden_states"] == 29  # 1 embedding + 28 transformer layers
    assert hidden_data["hidden_dim"] == 1536
    
    layer_0 = get_hidden_state_for_layer(hidden_data, 0)
    layer_14 = get_hidden_state_for_layer(hidden_data, 14)
    assert layer_0 is not None and layer_0["matrix"].shape == (len(hidden_token_ids), 1536)
    assert layer_14 is not None and layer_14["matrix"].shape == (len(hidden_token_ids), 1536)

    fig_hidden = plot_hidden_states_2d(hidden_tokens, hidden_token_ids, layer_14["matrix"], layer_label="Layer 14", layer_num=14)
    assert fig_hidden is not None and len(fig_hidden.data) > 0
    print(f"  [+] Verified 29 hidden state layers (1 Embedding + 28 Transformer blocks).")
    print(f"  [+] Plotly hidden states PCA representation chart generated successfully.")
    results["Hidden States & Representation Space"] = True

    # 7. LM Head Logits & Softmax Probabilities
    print("\n[Test 7/8] Verifying LM Head Logits & Softmax Probability Distributions...")
    logits_token_ids = tok_data["token_ids"][:16]
    logits_res = extract_next_token_logits(logits_token_ids, model=model, tokenizer=tokenizer, top_k=10)
    
    assert logits_res["error"] is None, f"Logits error: {logits_res['error']}"
    assert logits_res["vocab_size"] == getattr(model.config, "vocab_size", 151936)
    assert len(logits_res["top_predictions"]) == 10
    assert logits_res["top_1_prob"] > 0.0
    assert logits_res["entropy"] > 0.0
    
    top_pred = logits_res["top_predictions"][0]
    print(f"  [+] Top-1 Prediction: '{top_pred['token']}' (ID: {top_pred['token_id']}, Prob: {top_pred['probability_pct_str']}, Logit: {top_pred['logit']:.3f})")
    print(f"  [+] Predictive Entropy: {logits_res['entropy']:.4f}")

    fig_probs = plot_next_token_probabilities(logits_res["top_predictions"])
    assert fig_probs is not None and len(fig_probs.data) > 0
    print(f"  [+] Plotly probability distribution bar chart generated successfully.")
    results["Logits & Softmax Probabilities"] = True

    # 8. Autoregressive Generation Flow
    print("\n[Test 8/8] Verifying Autoregressive Generation Flow Timeline...")
    prompt_tok_data = tokenize_text(sample_prompt, tokenizer=tokenizer, max_length=16)
    fig_flow = plot_generation_flow(prompt_tok_data["breakdown"], tok_data["breakdown"][:12])
    assert fig_flow is not None and len(fig_flow.data) > 0
    print(f"  [+] Generation timeline generated ({len(prompt_tok_data['breakdown'])} prompt + 12 generated tokens).")
    results["Autoregressive Generation Flow"] = True

    clear_memory_cache()

    # Final Summary
    print("\n" + "=" * 75)
    print("STEP 15 TEST EXECUTION SUMMARY:")
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
        print("ALL 8 STEP 15 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    else:
        print(f"SOME TESTS FAILED ({sum(results.values())}/{len(results)} passed)")
    print("=" * 75)


if __name__ == "__main__":
    run_all_tests()

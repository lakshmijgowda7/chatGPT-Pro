"""
LocalGPT: X-Ray Neural Inspection Engine
Reuses and modularizes all deep neural network inspection operations from Project 1:
- Embedding layer extraction
- 28-layer Transformer architecture inspection
- Multi-Head Attention weight extraction
- Intermediate hidden states tracking (Layers 0..28)
- LM Head logits & Softmax probability distributions
- Token-by-token autoregressive generation tracking
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from model import load_model, load_model_and_tokenizer


@torch.inference_mode()
def extract_embeddings(
    token_ids: List[int],
    model: Optional[PreTrainedModel] = None,
) -> Dict[str, Any]:
    """
    Extracts real input embedding vectors for token IDs from the model's embedding table.
    """
    if model is None:
        model = load_model()

    if not token_ids:
        return {
            "embedding_dim": 0,
            "embeddings_matrix": np.empty((0, 0), dtype=np.float32),
            "token_embeddings": [],
            "global_stats": {},
            "error": None,
        }

    device = next(model.parameters()).device
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    embedding_layer = model.get_input_embeddings()
    raw_embeddings = embedding_layer(input_tensor)

    embeddings_np = raw_embeddings[0].detach().cpu().to(torch.float32).numpy()
    seq_len, hidden_dim = embeddings_np.shape

    token_embeddings = []
    for idx, (tid, vec) in enumerate(zip(token_ids, embeddings_np)):
        token_embeddings.append({
            "index": idx,
            "token_id": tid,
            "vector": vec,
            "vector_preview": vec[:8].tolist(),
            "dim": hidden_dim,
            "mean": float(np.mean(vec)),
            "std": float(np.std(vec)),
            "min": float(np.min(vec)),
            "max": float(np.max(vec)),
            "norm": float(np.linalg.norm(vec)),
        })

    global_stats = {
        "embedding_dim": hidden_dim,
        "total_tokens": seq_len,
        "global_mean": float(np.mean(embeddings_np)),
        "global_std": float(np.std(embeddings_np)),
        "global_min": float(np.min(embeddings_np)),
        "global_max": float(np.max(embeddings_np)),
        "mean_norm": float(np.mean([item["norm"] for item in token_embeddings])) if token_embeddings else 0.0,
    }

    return {
        "embedding_dim": hidden_dim,
        "embeddings_matrix": embeddings_np,
        "token_embeddings": token_embeddings,
        "global_stats": global_stats,
        "error": None,
    }


def get_transformer_layers_info(model: Optional[PreTrainedModel] = None) -> Dict[str, Any]:
    """
    Extracts architectural structure and parameters across all 28 Transformer layers.
    """
    if model is None:
        model = load_model()

    config = model.config
    num_layers = getattr(config, "num_hidden_layers", 28)
    hidden_size = getattr(config, "hidden_size", 1536)
    num_heads = getattr(config, "num_attention_heads", 12)
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    intermediate_size = getattr(config, "intermediate_size", 8960)
    vocab_size = getattr(config, "vocab_size", 151936)
    head_dim = hidden_size // num_heads if num_heads > 0 else 128

    layers_modules = getattr(getattr(model, "model", model), "layers", [])
    actual_layer_count = len(layers_modules) if layers_modules else num_layers

    layer_details = []
    for i in range(actual_layer_count):
        param_count = 0
        if i < len(layers_modules):
            try:
                param_count = sum(p.numel() for p in layers_modules[i].parameters())
            except Exception:
                param_count = 0
        
        layer_details.append({
            "layer_index": i,
            "layer_number": i + 1,
            "param_count": param_count,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "hidden_size": hidden_size,
            "head_dim": head_dim,
            "intermediate_size": intermediate_size,
            "attn_type": "Grouped Query Attention (GQA)" if num_kv_heads < num_heads else "Multi-Head Attention (MHA)",
            "mlp_type": "SwiGLU (Gate, Up, Down Projections)",
            "norm_type": "RMSNorm",
        })

    return {
        "model_name": getattr(config, "_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct"),
        "model_type": getattr(config, "model_type", "qwen2"),
        "num_layers": actual_layer_count,
        "hidden_size": hidden_size,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "intermediate_size": intermediate_size,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "layers": layer_details,
    }


@torch.inference_mode()
def extract_attentions(
    token_ids: List[int],
    model: Optional[PreTrainedModel] = None,
) -> Dict[str, Any]:
    """
    Extracts real Multi-Head Attention matrices across all 28 layers.
    """
    if model is None:
        model = load_model()

    if not token_ids:
        return {
            "num_layers": 0,
            "num_heads": 0,
            "seq_len": 0,
            "attentions_by_layer": [],
            "error": "No token IDs provided.",
        }

    device = next(model.parameters()).device
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    outputs = model(input_ids=input_tensor, output_attentions=True)
    raw_attentions = outputs.attentions

    if raw_attentions is None or len(raw_attentions) == 0:
        return {
            "num_layers": 0,
            "num_heads": 0,
            "seq_len": len(token_ids),
            "attentions_by_layer": [],
            "error": "Model did not return attention weights.",
        }

    attentions_by_layer: List[np.ndarray] = []
    num_heads = 0
    seq_len = len(token_ids)

    for layer_tensor in raw_attentions:
        layer_np = layer_tensor.detach().cpu().to(torch.float32).numpy()
        if layer_np.ndim == 4:
            layer_attn = layer_np[0]
        elif layer_np.ndim == 3:
            layer_attn = layer_np
        else:
            layer_attn = np.zeros((1, seq_len, seq_len), dtype=np.float32)

        num_heads = layer_attn.shape[0]
        attentions_by_layer.append(layer_attn)

    return {
        "num_layers": len(attentions_by_layer),
        "num_heads": num_heads,
        "seq_len": seq_len,
        "attentions_by_layer": attentions_by_layer,
        "error": None,
    }


def get_attention_matrix(
    attentions_data: Dict[str, Any],
    layer_index: int,
    head_index: Optional[int] = None,
) -> np.ndarray:
    """
    Safely retrieves a 2D attention matrix for a specific layer and head.
    """
    layers_list = attentions_data.get("attentions_by_layer", [])
    seq_len = attentions_data.get("seq_len", 0)

    if not layers_list or seq_len == 0:
        return np.zeros((max(1, seq_len), max(1, seq_len)), dtype=np.float32)

    safe_layer_idx = max(0, min(layer_index, len(layers_list) - 1))
    layer_tensor = layers_list[safe_layer_idx]

    if head_index is None or head_index < 0:
        return np.mean(layer_tensor, axis=0)

    num_heads = layer_tensor.shape[0]
    safe_head = max(0, min(head_index, num_heads - 1))
    return layer_tensor[safe_head]


@torch.inference_mode()
def extract_hidden_states(
    token_ids: List[int],
    model: Optional[PreTrainedModel] = None,
) -> Dict[str, Any]:
    """
    Extracts intermediate hidden states (Layer 0 Embedding + Layers 1..28).
    """
    if model is None:
        model = load_model()

    if not token_ids:
        return {
            "num_hidden_states": 0,
            "num_layers": 0,
            "hidden_dim": 0,
            "seq_len": 0,
            "embedding_output": {},
            "layers": {},
            "error": "No token IDs provided.",
        }

    device = next(model.parameters()).device
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    outputs = model(input_ids=input_tensor, output_hidden_states=True)
    raw_hidden_states = outputs.hidden_states

    if raw_hidden_states is None or len(raw_hidden_states) == 0:
        return {
            "num_hidden_states": 0,
            "num_layers": 0,
            "hidden_dim": 0,
            "seq_len": len(token_ids),
            "embedding_output": {},
            "layers": {},
            "error": "Model did not return hidden states.",
        }

    seq_len = len(token_ids)
    num_total_states = len(raw_hidden_states)
    num_transformer_layers = max(0, num_total_states - 1)

    processed_states: List[Dict[str, Any]] = []

    for hf_idx, state_tensor in enumerate(raw_hidden_states):
        state_np = state_tensor.detach().cpu().to(torch.float32).numpy()
        matrix = state_np[0] if state_np.ndim == 3 else np.atleast_2d(state_np)
        hidden_dim = matrix.shape[1] if matrix.ndim >= 2 else 0

        token_norms = [float(np.linalg.norm(matrix[i])) for i in range(matrix.shape[0])] if matrix.size > 0 else []
        layer_label = "Embedding Output (Pre-Transformer)" if hf_idx == 0 else f"Transformer Layer {hf_idx}"

        processed_states.append({
            "name": layer_label,
            "layer_number": hf_idx,
            "hf_index": hf_idx,
            "is_embedding": (hf_idx == 0),
            "matrix": matrix,
            "hidden_dim": hidden_dim,
            "seq_len": matrix.shape[0],
            "mean": float(np.mean(matrix)) if matrix.size > 0 else 0.0,
            "std": float(np.std(matrix)) if matrix.size > 0 else 0.0,
            "min": float(np.min(matrix)) if matrix.size > 0 else 0.0,
            "max": float(np.max(matrix)) if matrix.size > 0 else 0.0,
            "mean_l2_norm": float(np.mean(token_norms)) if token_norms else 0.0,
            "token_norms": token_norms,
        })

    embedding_output = processed_states[0] if processed_states else {}
    layers_dict = {idx: processed_states[idx] for idx in range(1, num_total_states)}

    return {
        "num_hidden_states": num_total_states,
        "num_layers": num_transformer_layers,
        "hidden_dim": processed_states[0]["hidden_dim"] if processed_states else 0,
        "seq_len": seq_len,
        "embedding_output": embedding_output,
        "layers": layers_dict,
        "all_states_list": processed_states,
        "error": None,
    }


def get_hidden_state_for_layer(
    hidden_states_data: Dict[str, Any],
    layer_num: int,
) -> Optional[Dict[str, Any]]:
    """
    Retrieves hidden state for a layer with safety clamping.
    """
    if not hidden_states_data or hidden_states_data.get("error"):
        return None
    if layer_num <= 0:
        return hidden_states_data.get("embedding_output")
    layers_dict = hidden_states_data.get("layers", {})
    if layer_num in layers_dict:
        return layers_dict[layer_num]
    if layers_dict:
        valid_keys = sorted(layers_dict.keys())
        return layers_dict[max(valid_keys[0], min(layer_num, valid_keys[-1]))]
    return None


@torch.inference_mode()
def extract_next_token_logits(
    token_ids: List[int],
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Extracts LM Head logits and computes Softmax probabilities.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer()

    if not token_ids:
        return {
            "vocab_size": 0,
            "seq_len": 0,
            "top_predictions": [],
            "error": "No token IDs provided.",
        }

    device = next(model.parameters()).device
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    outputs = model(input_ids=input_tensor)
    raw_logits = outputs.logits

    last_position_logits = raw_logits[0, -1, :]
    vocab_size = last_position_logits.shape[0]

    probabilities = F.softmax(last_position_logits, dim=-1)

    safe_k = max(1, min(top_k, vocab_size))
    top_probs_tensor, top_indices_tensor = torch.topk(probabilities, k=safe_k)

    top_probs = top_probs_tensor.detach().cpu().tolist()
    top_indices = top_indices_tensor.detach().cpu().tolist()

    top_predictions = []
    for rank, (p, idx) in enumerate(zip(top_probs, top_indices), 1):
        tok_str = tokenizer.decode([idx])
        pct_val = p * 100.0
        pct_str = f"{pct_val:.2f}%" if pct_val >= 0.01 else f"{pct_val:.4f}%"

        top_predictions.append({
            "rank": rank,
            "token_id": idx,
            "token": tok_str,
            "token_display": tok_str.replace(" ", "␣").replace("\n", "↵"),
            "token_repr": repr(tok_str),
            "probability": float(p),
            "probability_pct": float(pct_val),
            "probability_pct_str": pct_str,
            "logit": float(last_position_logits[idx].item()),
        })

    entropy_val = -torch.sum(probabilities * torch.log(probabilities.clamp(min=1e-12))).item()

    top_1_tok = top_predictions[0]["token"] if top_predictions else ""
    top_1_p = top_predictions[0]["probability"] if top_predictions else 0.0
    top_1_p_pct = top_predictions[0]["probability_pct_str"] if top_predictions else "0.00%"

    return {
        "vocab_size": vocab_size,
        "seq_len": len(token_ids),
        "last_token_id": token_ids[-1] if token_ids else None,
        "top_predictions": top_predictions,
        "top_1_token": top_1_tok,
        "top_1_prob": float(top_1_p),
        "top_1_prob_pct": top_1_p_pct,
        "entropy": float(entropy_val),
        "logits_min": float(torch.min(last_position_logits).item()),
        "logits_max": float(torch.max(last_position_logits).item()),
        "logits_mean": float(torch.mean(last_position_logits).item()),
        "error": None,
    }


@torch.inference_mode()
def generate_response_with_tokens(
    prompt: str,
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
) -> Dict[str, Any]:
    """
    Generates text while tracking token-by-token progression for X-Ray flow visualization.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer()

    if not prompt or not prompt.strip():
        return {
            "prompt_text": "",
            "prompt_tokens": [],
            "generated_tokens": [],
            "final_response": "",
            "error": "Prompt cannot be empty.",
        }

    device = next(model.parameters()).device
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    input_seq_length = inputs.input_ids.shape[1]
    prompt_ids = inputs.input_ids[0].tolist()

    prompt_tokens = []
    for idx, tid in enumerate(prompt_ids):
        tok_str = tokenizer.decode([tid])
        prompt_tokens.append({
            "index": idx,
            "token_id": tid,
            "token": tok_str,
            "token_display": tok_str.replace(" ", "␣").replace("\n", "↵"),
            "token_repr": repr(tok_str),
            "is_generated": False,
        })

    safe_max_tokens = max(1, min(max_new_tokens, 512))
    safe_temp = max(0.0, float(temperature))
    safe_top_p = max(0.01, min(float(top_p), 1.0))
    safe_top_k = max(0, int(top_k)) if top_k is not None else 50

    gen_kwargs = {
        "max_new_tokens": safe_max_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample and safe_temp > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = safe_temp
        gen_kwargs["top_p"] = safe_top_p
        if safe_top_k > 0:
            gen_kwargs["top_k"] = safe_top_k
    else:
        gen_kwargs["do_sample"] = False

    output_ids = model.generate(**inputs, **gen_kwargs)
    generated_ids = output_ids[0][input_seq_length:].tolist()

    generated_tokens = []
    for step_idx, tid in enumerate(generated_ids, 1):
        tok_str = tokenizer.decode([tid])
        cum_text = tokenizer.decode(generated_ids[:step_idx], skip_special_tokens=True).strip()
        generated_tokens.append({
            "step": step_idx,
            "token_id": tid,
            "token": tok_str,
            "token_display": tok_str.replace(" ", "␣").replace("\n", "↵"),
            "token_repr": repr(tok_str),
            "cumulative_response": cum_text,
            "is_generated": True,
        })

    final_response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    is_truncated = (len(generated_ids) >= safe_max_tokens) and (tokenizer.eos_token_id not in generated_ids)

    return {
        "prompt_text": prompt,
        "prompt_tokens": prompt_tokens,
        "prompt_token_count": len(prompt_tokens),
        "generated_tokens": generated_tokens,
        "generated_token_count": len(generated_tokens),
        "total_token_count": len(prompt_tokens) + len(generated_tokens),
        "final_response": final_response_text,
        "is_truncated": is_truncated,
        "error": None,
    }

"""
LLM-XRay: Model Architecture & Local Inference Engine (STEP 15 Complete)
Provides robust, laptop-reliable local model and tokenizer operations for Qwen/Qwen2.5-1.5B-Instruct:
- Automatic CPU / GPU (CUDA / MPS) detection with optimal dtype selection
- Inference/no-gradient execution (@torch.inference_mode) with minimal memory footprint
- Resilient exception handling for model loading, inference, OOM, and shape variations
- Safe clamping and bounds checking for layer indices, attention heads, and hidden states
"""

import os
import gc
import traceback
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

# Target model identifier
DEFAULT_MODEL_NAME: str = "Qwen/Qwen2.5-1.5B-Instruct"

# Global in-memory caches
_MODEL_CACHE: Optional[PreTrainedModel] = None
_TOKENIZER_CACHE: Optional[PreTrainedTokenizerBase] = None
_CACHED_MODEL_NAME: Optional[str] = None


def clear_memory_cache() -> None:
    """
    Frees unreferenced memory and clears CUDA cache if available.
    """
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def get_device() -> torch.device:
    """
    Detects and returns the best available compute device (CUDA, MPS, or CPU).
    Guaranteed to never fail or throw an exception on personal laptops.
    """
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


def get_device_info() -> Dict[str, Any]:
    """
    Returns diagnostic information about the active compute device and memory status.
    """
    device = get_device()
    info: Dict[str, Any] = {
        "device_type": device.type,
        "device_str": str(device),
        "is_gpu": device.type in ("cuda", "mps"),
        "name": "CPU (Standard Processor)",
        "memory_details": "System RAM",
    }

    try:
        if device.type == "cuda":
            dev_idx = torch.cuda.current_device()
            info["name"] = torch.cuda.get_device_name(dev_idx)
            total_mem_gb = torch.cuda.get_device_properties(dev_idx).total_memory / (1024 ** 3)
            allocated_gb = torch.cuda.memory_allocated(dev_idx) / (1024 ** 3)
            info["memory_details"] = f"{allocated_gb:.2f} GB / {total_mem_gb:.2f} GB VRAM"
        elif device.type == "mps":
            info["name"] = "Apple Silicon GPU (MPS)"
            info["memory_details"] = "Unified Memory"
        else:
            info["name"] = f"CPU ({os.cpu_count() or 1} Cores)"
    except Exception as e:
        info["warning"] = str(e)

    return info


def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> PreTrainedTokenizerBase:
    """
    Loads and caches the Hugging Face tokenizer with robust error handling.
    
    Args:
        model_name: Hugging Face model repository ID.
        
    Returns:
        Loaded tokenizer instance.
    """
    global _TOKENIZER_CACHE, _CACHED_MODEL_NAME
    if _TOKENIZER_CACHE is not None and _CACHED_MODEL_NAME == model_name:
        return _TOKENIZER_CACHE

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        _TOKENIZER_CACHE = tokenizer
        return tokenizer
    except Exception as e:
        err_msg = (
            f"Failed to load tokenizer '{model_name}': {str(e)}.\n"
            f"Please verify your internet connection or check that the model repository exists on Hugging Face."
        )
        raise RuntimeError(err_msg) from e


def load_model(model_name: str = DEFAULT_MODEL_NAME) -> PreTrainedModel:
    """
    Loads and caches the autoregressive Causal LM on the available device (GPU or CPU)
    optimized for standard personal laptops.
    
    Memory and speed optimizations:
    - low_cpu_mem_usage=True to prevent RAM spikes during weight loading
    - float16 on CUDA/MPS for 50% memory reduction; float32 on CPU for numerical stability
    - output_attentions and output_hidden_states enabled for deep inspection
    - Model set to eval mode with gradients disabled
    
    Args:
        model_name: Hugging Face model repository ID.
        
    Returns:
        Loaded PyTorch model ready for inference.
    """
    global _MODEL_CACHE, _CACHED_MODEL_NAME
    if _MODEL_CACHE is not None and _CACHED_MODEL_NAME == model_name:
        return _MODEL_CACHE

    device = get_device()
    clear_memory_cache()

    # Select optimal dtype based on compute device
    if device.type == "cuda":
        torch_dtype = torch.float16
    elif device.type == "mps":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.config.output_attentions = True
        model.config.output_hidden_states = True
        model.to(device)
        model.eval()

        # Disable gradients globally for inference safety
        for param in model.parameters():
            param.requires_grad = False

        _MODEL_CACHE = model
        _CACHED_MODEL_NAME = model_name
        return model

    except torch.cuda.OutOfMemoryError as oom_err:
        clear_memory_cache()
        raise RuntimeError(
            f"GPU Out Of Memory (OOM) while loading '{model_name}'. "
            f"Please close other GPU applications or restart the application."
        ) from oom_err
    except Exception as e:
        clear_memory_cache()
        err_msg = (
            f"Failed to load model '{model_name}' on device '{device}': {str(e)}.\n"
            f"Troubleshooting tips:\n"
            f"1. Ensure you have an active internet connection on first run to download model weights (~3 GB).\n"
            f"2. Ensure you have at least 4 GB of available system RAM or disk space.\n"
            f"3. Verify required dependencies via 'pip install -r requirements.txt'."
        )
        raise RuntimeError(err_msg) from e


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Convenience function to load both model and tokenizer with centralized error reporting.
    
    Args:
        model_name: Hugging Face model repository ID.
        
    Returns:
        Tuple of (model, tokenizer).
    """
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)
    return model, tokenizer


@torch.inference_mode()
def extract_embeddings(
    token_ids: List[int],
    model: Optional[PreTrainedModel] = None,
) -> Dict[str, Any]:
    """
    Extracts real input embedding vectors for a list of token IDs directly from the model's
    embedding layer.
    
    Args:
        token_ids: List of integer token IDs.
        model: Optional preloaded model instance.
        
    Returns:
        Dictionary containing embedding dimension, raw matrix, and per-token statistics.
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

    try:
        device = next(model.parameters()).device
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

        embedding_layer = model.get_input_embeddings()
        raw_embeddings = embedding_layer(input_tensor)  # (1, seq_len, hidden_dim)

        # Convert to CPU numpy array immediately to release PyTorch tensors
        embeddings_np = raw_embeddings[0].detach().cpu().to(torch.float32).numpy()
        seq_len, hidden_dim = embeddings_np.shape

        token_embeddings = []
        for idx, (tid, vec) in enumerate(zip(token_ids, embeddings_np)):
            token_embeddings.append({
                "index": idx,
                "token_id": tid,
                "vector": vec,
                "vector_preview": vec[:8].tolist() if vec.size >= 8 else vec.tolist(),
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
    except Exception as e:
        return {
            "embedding_dim": 0,
            "embeddings_matrix": np.empty((0, 0), dtype=np.float32),
            "token_embeddings": [],
            "global_stats": {},
            "error": f"Error extracting embeddings: {str(e)}",
            "traceback": traceback.format_exc(),
        }


def get_transformer_layers_info(model: Optional[PreTrainedModel] = None) -> Dict[str, Any]:
    """
    Extracts real architectural structure and parameter details of the Transformer layers
    from the loaded model.
    
    Args:
        model: Optional preloaded model instance.
        
    Returns:
        Dictionary containing total layer count, architectural dimensions, and layer-by-layer info.
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

    total_params = 0
    try:
        total_params = sum(p.numel() for p in model.parameters())
    except Exception:
        total_params = 1540000000

    return {
        "model_name": getattr(config, "_name_or_path", DEFAULT_MODEL_NAME),
        "model_type": getattr(config, "model_type", "qwen2"),
        "num_layers": actual_layer_count,
        "hidden_size": hidden_size,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "intermediate_size": intermediate_size,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "total_parameters": total_params,
        "layers": layer_details,
    }


@torch.inference_mode()
def extract_attentions(
    token_ids: List[int],
    model: Optional[PreTrainedModel] = None,
) -> Dict[str, Any]:
    """
    Extracts real Multi-Head Attention weight matrices for a sequence of token IDs
    across all Transformer layers using a single forward pass with output_attentions=True.
    
    Args:
        token_ids: List of integer token IDs.
        model: Optional preloaded model instance.
        
    Returns:
        Dictionary containing attention matrices, dimensions, and error status.
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

    try:
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
                "error": "Model did not return attention weights. Ensure attn_implementation='eager' is configured.",
            }

        attentions_by_layer: List[np.ndarray] = []
        num_heads = 0
        seq_len = len(token_ids)

        for layer_tensor in raw_attentions:
            layer_np = layer_tensor.detach().cpu().to(torch.float32).numpy()

            # Handle dimensional variations safely
            if layer_np.ndim == 4:
                # Shape: (batch_size, num_heads, seq_len, seq_len)
                layer_attn = layer_np[0]
            elif layer_np.ndim == 3:
                # Shape: (num_heads, seq_len, seq_len)
                layer_attn = layer_np
            elif layer_np.ndim == 2:
                # Shape: (seq_len, seq_len)
                layer_attn = np.expand_dims(layer_np, axis=0)
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
    except torch.cuda.OutOfMemoryError:
        clear_memory_cache()
        return {
            "num_layers": 0,
            "num_heads": 0,
            "seq_len": len(token_ids),
            "attentions_by_layer": [],
            "error": "Out of memory while extracting attention weights. Try shortening your prompt.",
        }
    except Exception as e:
        return {
            "num_layers": 0,
            "num_heads": 0,
            "seq_len": len(token_ids),
            "attentions_by_layer": [],
            "error": f"Error extracting attention matrices: {str(e)}",
            "traceback": traceback.format_exc(),
        }


def get_attention_matrix(
    attentions_data: Dict[str, Any],
    layer_index: int,
    head_index: Optional[int] = None,
) -> np.ndarray:
    """
    Safely retrieves a 2D attention matrix for a specific layer and head.
    If head_index is None or -1, returns the mean attention matrix averaged across all heads.
    Guaranteed to never raise an IndexError.
    
    Args:
        attentions_data: Output dictionary from extract_attentions().
        layer_index: 0-indexed layer index (0 to num_layers - 1).
        head_index: Optional 0-indexed head index (0 to num_heads - 1), or None for average.
        
    Returns:
        2D numpy array of shape (seq_len, seq_len).
    """
    layers_list = attentions_data.get("attentions_by_layer", [])
    seq_len = attentions_data.get("seq_len", 0)

    if not layers_list or seq_len == 0:
        return np.zeros((max(1, seq_len), max(1, seq_len)), dtype=np.float32)

    # Safe layer clamping
    safe_layer_idx = max(0, min(layer_index, len(layers_list) - 1))
    layer_tensor = layers_list[safe_layer_idx]  # Shape: (num_heads, seq_len, seq_len)

    if layer_tensor.ndim < 3:
        return np.atleast_2d(layer_tensor)

    num_heads = layer_tensor.shape[0]

    if head_index is None or head_index < 0:
        # Compute mean across all heads
        return np.mean(layer_tensor, axis=0)

    # Safe head clamping
    safe_head = max(0, min(head_index, num_heads - 1))
    return layer_tensor[safe_head]


@torch.inference_mode()
def extract_hidden_states(
    token_ids: List[int],
    model: Optional[PreTrainedModel] = None,
) -> Dict[str, Any]:
    """
    Extracts real intermediate hidden-state representations from the Transformer layers
    using a forward pass with output_hidden_states=True.
    
    Args:
        token_ids: List of integer token IDs.
        model: Optional preloaded model instance.
        
    Returns:
        Dictionary containing hidden state matrices, statistics, and metadata.
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

    try:
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

            if state_np.ndim == 3:
                matrix = state_np[0]
                tensor_shape = state_np.shape
            elif state_np.ndim == 2:
                matrix = state_np
                tensor_shape = (1, state_np.shape[0], state_np.shape[1])
            else:
                matrix = np.atleast_2d(state_np)
                tensor_shape = (1, matrix.shape[0], matrix.shape[1])

            hidden_dim = matrix.shape[1] if matrix.ndim >= 2 else 0

            # Safe norm calculation
            token_norms = [float(np.linalg.norm(matrix[i])) for i in range(matrix.shape[0])] if matrix.size > 0 else []

            layer_label = "Embedding Output (Pre-Transformer)" if hf_idx == 0 else f"Transformer Layer {hf_idx}"
            layer_num = hf_idx

            state_dict = {
                "name": layer_label,
                "layer_number": layer_num,
                "hf_index": hf_idx,
                "is_embedding": (hf_idx == 0),
                "tensor_shape": tensor_shape,
                "matrix_shape": matrix.shape,
                "hidden_dim": hidden_dim,
                "seq_len": matrix.shape[0],
                "matrix": matrix,
                "mean": float(np.mean(matrix)) if matrix.size > 0 else 0.0,
                "std": float(np.std(matrix)) if matrix.size > 0 else 0.0,
                "min": float(np.min(matrix)) if matrix.size > 0 else 0.0,
                "max": float(np.max(matrix)) if matrix.size > 0 else 0.0,
                "mean_l2_norm": float(np.mean(token_norms)) if token_norms else 0.0,
                "token_norms": token_norms,
            }
            processed_states.append(state_dict)

        embedding_output = processed_states[0] if processed_states else {}

        layers_dict: Dict[int, Dict[str, Any]] = {}
        for idx in range(1, num_total_states):
            layers_dict[idx] = processed_states[idx]

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
    except torch.cuda.OutOfMemoryError:
        clear_memory_cache()
        return {
            "num_hidden_states": 0,
            "num_layers": 0,
            "hidden_dim": 0,
            "seq_len": len(token_ids),
            "embedding_output": {},
            "layers": {},
            "error": "Out of memory while extracting hidden states. Try shortening your prompt.",
        }
    except Exception as e:
        return {
            "num_hidden_states": 0,
            "num_layers": 0,
            "hidden_dim": 0,
            "seq_len": len(token_ids),
            "embedding_output": {},
            "layers": {},
            "error": f"Error extracting hidden states: {str(e)}",
            "traceback": traceback.format_exc(),
        }


def get_hidden_state_for_layer(
    hidden_states_data: Dict[str, Any],
    layer_num: int,
) -> Optional[Dict[str, Any]]:
    """
    Safely retrieves the hidden state metadata and matrix for a specific layer.
    Safely clamps out-of-bound layer numbers.
    
    Args:
        hidden_states_data: Output dictionary from extract_hidden_states().
        layer_num: 0 for Embedding output, 1..28 for Transformer layers 1 to 28.
        
    Returns:
        Dictionary containing layer metadata, shapes, statistics, and matrix.
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
        clamped_key = max(valid_keys[0], min(layer_num, valid_keys[-1]))
        return layers_dict[clamped_key]

    return None


@torch.inference_mode()
def extract_next_token_logits(
    token_ids: List[int],
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Extracts actual logits from the LM head at the final sequence position,
    computes exact softmax probabilities across the vocabulary,
    and returns the top-K predicted next tokens with full bounds safety.
    
    Args:
        token_ids: List of integer token IDs.
        model: Optional preloaded model instance.
        tokenizer: Optional preloaded tokenizer instance.
        top_k: Number of top candidate tokens to extract.
        
    Returns:
        Dictionary containing next-token predictions, entropy, and logit metrics.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer()

    if not token_ids:
        return {
            "vocab_size": 0,
            "seq_len": 0,
            "last_token_id": None,
            "last_token_str": "",
            "top_predictions": [],
            "top_1_token": "",
            "top_1_prob": 0.0,
            "top_1_prob_pct": "0.00%",
            "entropy": 0.0,
            "logits_min": 0.0,
            "logits_max": 0.0,
            "logits_mean": 0.0,
            "error": "No token IDs provided.",
        }

    try:
        device = next(model.parameters()).device
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

        outputs = model(input_ids=input_tensor)
        raw_logits = outputs.logits

        if raw_logits is None or raw_logits.numel() == 0:
            return {
                "vocab_size": 0,
                "seq_len": len(token_ids),
                "last_token_id": token_ids[-1] if token_ids else None,
                "last_token_str": "",
                "top_predictions": [],
                "top_1_token": "",
                "top_1_prob": 0.0,
                "top_1_prob_pct": "0.00%",
                "entropy": 0.0,
                "logits_min": 0.0,
                "logits_max": 0.0,
                "logits_mean": 0.0,
                "error": "Model did not return logits.",
            }

        last_position_logits = raw_logits[0, -1, :]
        vocab_size = last_position_logits.shape[0]

        probabilities = F.softmax(last_position_logits, dim=-1)

        safe_k = max(1, min(top_k, vocab_size))
        top_probs_tensor, top_indices_tensor = torch.topk(probabilities, k=safe_k)

        top_probs = top_probs_tensor.detach().cpu().tolist()
        top_indices = top_indices_tensor.detach().cpu().tolist()

        top_predictions: List[Dict[str, Any]] = []
        for rank, (p, idx) in enumerate(zip(top_probs, top_indices), 1):
            tok_str = tokenizer.decode([idx])
            display_str = tok_str.replace(" ", "␣").replace("\n", "↵")
            pct_val = p * 100.0
            pct_str = f"{pct_val:.2f}%" if pct_val >= 0.01 else f"{pct_val:.4f}%"

            top_predictions.append({
                "rank": rank,
                "token_id": idx,
                "token": tok_str,
                "token_display": display_str,
                "token_repr": repr(tok_str),
                "probability": float(p),
                "probability_pct": float(pct_val),
                "probability_pct_str": pct_str,
                "logit": float(last_position_logits[idx].item()),
            })

        entropy_val = -torch.sum(probabilities * torch.log(probabilities.clamp(min=1e-12))).item()
        last_tid = token_ids[-1]
        last_tok_str = tokenizer.decode([last_tid])

        top_1_tok = top_predictions[0]["token"] if top_predictions else ""
        top_1_p = top_predictions[0]["probability"] if top_predictions else 0.0
        top_1_p_pct = top_predictions[0]["probability_pct_str"] if top_predictions else "0.00%"

        return {
            "vocab_size": vocab_size,
            "seq_len": len(token_ids),
            "last_token_id": last_tid,
            "last_token_str": last_tok_str,
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
    except torch.cuda.OutOfMemoryError:
        clear_memory_cache()
        return {
            "vocab_size": 0,
            "seq_len": len(token_ids),
            "last_token_id": token_ids[-1] if token_ids else None,
            "last_token_str": "",
            "top_predictions": [],
            "top_1_token": "",
            "top_1_prob": 0.0,
            "top_1_prob_pct": "0.00%",
            "entropy": 0.0,
            "logits_min": 0.0,
            "logits_max": 0.0,
            "logits_mean": 0.0,
            "error": "Out of memory while computing next-token logits.",
        }
    except Exception as e:
        return {
            "vocab_size": 0,
            "seq_len": len(token_ids),
            "last_token_id": token_ids[-1] if token_ids else None,
            "last_token_str": "",
            "top_predictions": [],
            "top_1_token": "",
            "top_1_prob": 0.0,
            "top_1_prob_pct": "0.00%",
            "entropy": 0.0,
            "logits_min": 0.0,
            "logits_max": 0.0,
            "logits_mean": 0.0,
            "error": f"Error computing logits: {str(e)}",
            "traceback": traceback.format_exc(),
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
    Generates a response from a user prompt using local model inference,
    tracking every token step-by-step with complete exception and OOM safety.
    
    Args:
        prompt: User input string.
        model: Optional preloaded model instance.
        tokenizer: Optional preloaded tokenizer instance.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (> 0.0).
        top_p: Nucleus sampling probability threshold.
        top_k: Top-K filtering threshold.
        do_sample: Whether to use stochastic sampling or greedy decoding.
        
    Returns:
        Dictionary containing prompt tokens, generated tokens, text, and status.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer()

    if not prompt or not prompt.strip():
        return {
            "prompt_text": "",
            "formatted_prompt": "",
            "prompt_tokens": [],
            "prompt_token_count": 0,
            "generated_tokens": [],
            "generated_token_count": 0,
            "total_token_count": 0,
            "final_response": "",
            "is_truncated": False,
            "error": "Prompt cannot be empty. Please enter text to generate a response.",
        }

    try:
        device = next(model.parameters()).device

        messages = [{"role": "user", "content": prompt.strip()}]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            try:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                formatted_prompt = prompt.strip()
        else:
            formatted_prompt = prompt.strip()

        inputs = tokenizer([formatted_prompt], return_tensors="pt").to(device)
        input_seq_length = inputs.input_ids.shape[1]
        prompt_ids = inputs.input_ids[0].tolist()

        prompt_tokens: List[Dict[str, Any]] = []
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

        # Sanitize parameters
        safe_max_tokens = max(1, min(max_new_tokens, 512))
        safe_temp = max(0.0, float(temperature))
        safe_top_p = max(0.01, min(float(top_p), 1.0))
        safe_top_k = max(0, int(top_k)) if top_k is not None else 50

        generation_kwargs = {
            "max_new_tokens": safe_max_tokens,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        if do_sample and safe_temp > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = safe_temp
            generation_kwargs["top_p"] = safe_top_p
            if safe_top_k > 0:
                generation_kwargs["top_k"] = safe_top_k
        else:
            generation_kwargs["do_sample"] = False

        output_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )

        generated_ids = output_ids[0][input_seq_length:].tolist()
        generated_tokens: List[Dict[str, Any]] = []

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
            "formatted_prompt": formatted_prompt,
            "prompt_tokens": prompt_tokens,
            "prompt_token_count": len(prompt_tokens),
            "generated_tokens": generated_tokens,
            "generated_token_count": len(generated_tokens),
            "total_token_count": len(prompt_tokens) + len(generated_tokens),
            "final_response": final_response_text,
            "is_truncated": is_truncated,
            "error": None,
        }

    except torch.cuda.OutOfMemoryError:
        clear_memory_cache()
        return {
            "prompt_text": prompt,
            "formatted_prompt": "",
            "prompt_tokens": [],
            "prompt_token_count": 0,
            "generated_tokens": [],
            "generated_token_count": 0,
            "total_token_count": 0,
            "final_response": "",
            "is_truncated": False,
            "error": "GPU Out of Memory during generation. Lower 'Max New Tokens' or run in CPU mode.",
        }
    except Exception as e:
        return {
            "prompt_text": prompt,
            "formatted_prompt": "",
            "prompt_tokens": [],
            "prompt_token_count": 0,
            "generated_tokens": [],
            "generated_token_count": 0,
            "total_token_count": 0,
            "final_response": "",
            "is_truncated": False,
            "error": f"Generation error: {str(e)}",
            "traceback": traceback.format_exc(),
        }


@torch.inference_mode()
def generate_response(
    prompt: str,
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
) -> str:
    """
    Generates a response from a user prompt using local inference.
    Maintains backward compatibility.
    """
    res = generate_response_with_tokens(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
    )
    return res.get("final_response", "")


if __name__ == "__main__":
    print("=" * 60)
    print("LLM-XRay: Model Engine Diagnostics")
    print(f"Device Info: {get_device_info()}")
    print("=" * 60)

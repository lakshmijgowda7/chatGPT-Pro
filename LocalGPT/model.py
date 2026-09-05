"""
LocalGPT: Local Model Engine
Provides robust loading and local inference for Qwen/Qwen2.5-1.5B-Instruct:
- Device detection (CUDA, MPS, CPU)
- Memory management & cache clearing
- Multi-turn conversational generation & streaming
- 100% local execution with zero external APIs
"""

import os
import gc
import sys
import traceback
from typing import Tuple, Optional, Dict, Any, List, Iterator
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextIteratorStreamer,
)
from threading import Thread

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
    Returns diagnostic information about the compute device.
    """
    device = get_device()
    info: Dict[str, Any] = {
        "device": device,
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
    Loads and caches the tokenizer.
    """
    global _TOKENIZER_CACHE, _CACHED_MODEL_NAME
    if _TOKENIZER_CACHE is not None and _CACHED_MODEL_NAME == model_name:
        return _TOKENIZER_CACHE

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _TOKENIZER_CACHE = tokenizer
    _CACHED_MODEL_NAME = model_name
    return tokenizer


def load_model(model_name: str = DEFAULT_MODEL_NAME) -> PreTrainedModel:
    """
    Loads and caches the local Qwen-1.5B model.
    """
    global _MODEL_CACHE, _CACHED_MODEL_NAME
    if _MODEL_CACHE is not None and _CACHED_MODEL_NAME == model_name:
        return _MODEL_CACHE

    device = get_device()
    clear_memory_cache()

    if device.type == "cuda":
        torch_dtype = torch.float16
    elif device.type == "mps":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # Default to False during autoregressive generation for speed; xray.py enables on demand
    model.config.output_attentions = False
    model.config.output_hidden_states = False
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    _MODEL_CACHE = model
    _CACHED_MODEL_NAME = model_name
    return model


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Convenience function to load both model and tokenizer.
    """
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)
    return model, tokenizer


@torch.inference_mode()
def generate_chat_response(
    formatted_prompt: str,
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
) -> Dict[str, Any]:
    """
    Generates a response from a ChatML-formatted prompt.
    
    Args:
        formatted_prompt: ChatML formatted prompt string.
        model: Optional preloaded model.
        tokenizer: Optional preloaded tokenizer.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-P nucleus sampling threshold.
        top_k: Top-K filtering.
        do_sample: Whether to use sampling or greedy decoding.
        
    Returns:
        Dictionary with response text, generated token count, and status.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer()

    if not formatted_prompt.strip():
        return {
            "response": "",
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "error": "Prompt is empty.",
        }

    try:
        device = next(model.parameters()).device
        inputs = tokenizer([formatted_prompt], return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]

        safe_max_tokens = max(1, min(int(max_new_tokens), 2048))
        safe_temp = max(0.0, min(float(temperature), 2.0))
        safe_top_p = max(0.01, min(float(top_p), 1.0))
        safe_top_k = max(0, min(int(top_k), 200)) if top_k is not None else 50

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

        output_ids = model.generate(
            **inputs,
            **gen_kwargs,
        )

        generated_ids = output_ids[0][input_len:].tolist()
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return {
            "response": response_text,
            "prompt_tokens": input_len,
            "generated_tokens": len(generated_ids),
            "total_tokens": input_len + len(generated_ids),
            "error": None,
        }
    except torch.cuda.OutOfMemoryError:
        clear_memory_cache()
        return {
            "response": "",
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "error": "GPU Out Of Memory. Try shortening context or lowering max tokens.",
        }
    except Exception as e:
        return {
            "response": "",
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "error": f"Generation error: {str(e)}",
            "traceback": traceback.format_exc(),
        }


def stream_chat_response(
    formatted_prompt: str,
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
) -> Iterator[str]:
    """
    Streams generated text token-by-token using Hugging Face TextIteratorStreamer.
    
    Yields:
        Generated text pieces sequentially as they are produced.
    """
    if not formatted_prompt or not formatted_prompt.strip():
        return

    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer()

    device = next(model.parameters()).device
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    safe_max_tokens = max(1, min(int(max_new_tokens), 2048))
    safe_temp = max(0.0, min(float(temperature), 2.0))
    safe_top_p = max(0.01, min(float(top_p), 1.0))
    safe_top_k = max(0, min(int(top_k), 200)) if top_k is not None else 50

    gen_kwargs = {
        **inputs,
        "streamer": streamer,
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

    generation_error: Optional[Exception] = None

    def _generate_worker():
        nonlocal generation_error
        try:
            model.generate(**gen_kwargs)
        except Exception as e:
            generation_error = e

    thread = Thread(target=_generate_worker)
    thread.start()

    for new_text in streamer:
        yield new_text

    thread.join()

    if generation_error is not None:
        raise generation_error

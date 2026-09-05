"""
LocalGPT: Tokenizer Operations
Handles tokenization for Qwen/Qwen2.5-1.5B-Instruct, ChatML message formatting,
and subword decomposition.
"""

from typing import Dict, Any, List, Optional
import traceback
from transformers import AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_MODEL_NAME: str = "Qwen/Qwen2.5-1.5B-Instruct"

# In-memory cache
_TOKENIZER_CACHE: Optional[PreTrainedTokenizerBase] = None
_CACHED_MODEL_NAME: Optional[str] = None


def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> PreTrainedTokenizerBase:
    """
    Loads and caches the Hugging Face tokenizer.
    
    Args:
        model_name: Hugging Face model identifier.
        
    Returns:
        Loaded PreTrainedTokenizerBase instance.
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
        _CACHED_MODEL_NAME = model_name
        return tokenizer
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer for '{model_name}': {str(e)}. "
            f"Ensure an active internet connection on first run."
        ) from e


def format_chat_prompt(
    messages: List[Dict[str, str]],
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Formats a multi-turn conversation list into the official Qwen ChatML format.
    
    Format:
        <|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        {assistant_message}<|im_end|>
        <|im_start|>assistant
        
    Args:
        messages: List of message dicts with 'role' ('system', 'user', 'assistant') and 'content'.
        tokenizer: Optional loaded tokenizer instance.
        system_prompt: Optional default system prompt if not present in messages.
        
    Returns:
        Formatted ChatML string ready for model input.
    """
    if tokenizer is None:
        tokenizer = load_tokenizer()

    formatted_msgs = []

    # Insert default system prompt if specified and not already the first message
    if system_prompt and (not messages or messages[0].get("role") != "system"):
        formatted_msgs.append({"role": "system", "content": system_prompt})

    formatted_msgs.extend(messages)

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(
                formatted_msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    # Fallback ChatML manual reconstruction
    prompt_str = ""
    for msg in formatted_msgs:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt_str += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt_str += "<|im_start|>assistant\n"
    return prompt_str


def tokenize_text(
    text: str,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    max_length: Optional[int] = 512,
) -> Dict[str, Any]:
    """
    Tokenizes text into discrete subwords, vocabulary IDs, and character offsets.
    
    Args:
        text: Raw input string.
        tokenizer: Optional loaded tokenizer.
        max_length: Maximum token length limit.
        
    Returns:
        Dictionary containing tokens, token_ids, total_tokens, and breakdown.
    """
    if tokenizer is None:
        tokenizer = load_tokenizer()

    if not text:
        return {
            "original_input": "",
            "tokens": [],
            "token_ids": [],
            "total_tokens": 0,
            "breakdown": [],
            "is_truncated": False,
            "error": None,
        }

    try:
        token_ids: List[int] = tokenizer.encode(text, add_special_tokens=False)
        is_truncated = False

        if max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            is_truncated = True

        tokens: List[str] = [tokenizer.decode([tid]) for tid in token_ids]

        breakdown = [
            {
                "index": idx,
                "token": tok,
                "token_display": tok.replace(" ", "␣").replace("\n", "↵"),
                "token_repr": repr(tok)[1:-1],
                "token_id": tid,
            }
            for idx, (tok, tid) in enumerate(zip(tokens, token_ids))
        ]

        return {
            "original_input": text,
            "tokens": tokens,
            "token_ids": token_ids,
            "total_tokens": len(token_ids),
            "breakdown": breakdown,
            "is_truncated": is_truncated,
            "error": None,
        }
    except Exception as e:
        return {
            "original_input": text,
            "tokens": [],
            "token_ids": [],
            "total_tokens": 0,
            "breakdown": [],
            "is_truncated": False,
            "error": f"Tokenization error: {str(e)}",
            "traceback": traceback.format_exc(),
        }

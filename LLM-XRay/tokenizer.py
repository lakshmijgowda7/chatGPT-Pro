"""
LLM-XRay: Tokenizer Operations (STEP 15 Complete)
Handles robust tokenization for Qwen/Qwen2.5-1.5B-Instruct, token-to-ID mapping,
and sequence token breakdown with complete exception handling and whitespace resilience.
"""

from typing import Dict, Any, List, Optional
import traceback
from transformers import AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_MODEL_NAME: str = "Qwen/Qwen2.5-1.5B-Instruct"

# Global cache for loaded tokenizer
_TOKENIZER_CACHE: Optional[PreTrainedTokenizerBase] = None
_CACHED_MODEL_NAME: Optional[str] = None


def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> PreTrainedTokenizerBase:
    """
    Loads and caches the Hugging Face tokenizer with error handling.
    
    Args:
        model_name: Hugging Face model identifier.
        
    Returns:
        Loaded PreTrainedTokenizerBase.
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
            f"Please ensure internet access for first-time download."
        ) from e


def tokenize_text(
    text: str,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    max_length: Optional[int] = 512,
) -> Dict[str, Any]:
    """
    Tokenizes input text using the actual model tokenizer and extracts
    individual tokens, real token IDs, and token count.
    
    Safely handles empty text, whitespace, unicode characters, and long text.
    
    Args:
        text: Input string to tokenize.
        tokenizer: Optional loaded tokenizer instance.
        max_length: Optional token limit (default 512 for laptop responsiveness).
        
    Returns:
        Dictionary containing:
            - original_input (str)
            - tokens (List[str])
            - token_ids (List[int])
            - total_tokens (int)
            - breakdown (List[Dict[str, Any]])
            - is_truncated (bool)
            - error (Optional[str])
    """
    if tokenizer is None:
        try:
            tokenizer = load_tokenizer()
        except Exception as e:
            return {
                "original_input": text or "",
                "tokens": [],
                "token_ids": [],
                "total_tokens": 0,
                "breakdown": [],
                "is_truncated": False,
                "error": f"Tokenizer loading error: {str(e)}",
            }

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
        # Encode without special tokens to inspect pure input tokens
        token_ids: List[int] = tokenizer.encode(text, add_special_tokens=False)
        is_truncated = False

        # If sequence is exceptionally long, clamp to max_length for safety
        if max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            is_truncated = True
        
        # Decode each individual token ID back into its exact string piece
        tokens: List[str] = [tokenizer.decode([tid]) for tid in token_ids]
        
        # Create structured breakdown
        breakdown = [
            {
                "index": idx,
                "token": tok,
                "token_repr": repr(tok)[1:-1],  # Escaped representation for whitespace/newlines
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


if __name__ == "__main__":
    print("=" * 60)
    print("LLM-XRay: Tokenization Test (Qwen/Qwen2.5-1.5B-Instruct)")
    print("=" * 60)

    sample_text = "Hello, world! Welcome to LLM-XRay."
    tok = load_tokenizer()
    result = tokenize_text(sample_text, tok)
    print(f"Tokens: {result['tokens']}")
    print(f"Total: {result['total_tokens']}")

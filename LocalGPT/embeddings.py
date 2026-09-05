"""
LocalGPT: Embeddings Generation (Step 12: RAG with Sentence Transformers)
Provides dense semantic embeddings for text queries and document chunks
using Sentence Transformers (all-MiniLM-L6-v2) with L2 normalization.
"""
from typing import List, Optional
import numpy as np

# Embedding model cache
_SENTENCE_TRANSFORMER_CACHE = None
_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def get_embedding_model(model_name: str = _EMBEDDING_MODEL_NAME):
    """
    Loads and caches the SentenceTransformer embedding model.
    """
    global _SENTENCE_TRANSFORMER_CACHE, _EMBEDDING_MODEL_NAME
    if _SENTENCE_TRANSFORMER_CACHE is not None and _EMBEDDING_MODEL_NAME == model_name:
        return _SENTENCE_TRANSFORMER_CACHE

    try:
        from sentence_transformers import SentenceTransformer
        # Load local or cached model
        model = SentenceTransformer(model_name)
        _SENTENCE_TRANSFORMER_CACHE = model
        _EMBEDDING_MODEL_NAME = model_name
        return model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load embedding model '{model_name}': {str(e)}. "
            f"Ensure sentence-transformers is installed."
        ) from e


def compute_text_embedding(
    text: str,
    model=None,
) -> np.ndarray:
    """
    Computes a 384-dimensional L2-normalized semantic embedding vector for a single text query.
    
    Args:
        text: Input text string.
        model: Optional preloaded SentenceTransformer.
        
    Returns:
        1D numpy array of shape (384,) with L2 norm = 1.0 (float32).
    """
    if not text or not text.strip():
        return np.zeros((EMBEDDING_DIM,), dtype=np.float32)

    if model is None:
        model = get_embedding_model()

    # SentenceTransformer encode with normalize_embeddings=True
    vec = model.encode(
        text.strip(),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vec.astype(np.float32)


def compute_batch_embeddings(
    texts: List[str],
    model=None,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Computes L2-normalized semantic embedding vectors for a batch of text chunks.
    
    Args:
        texts: List of text strings.
        model: Optional preloaded SentenceTransformer.
        batch_size: Encoding batch size.
        
    Returns:
        2D numpy array of shape (len(texts), 384) with float32 dtype.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    if model is None:
        model = get_embedding_model()

    # Filter/clean texts
    cleaned_texts = [t if (t and t.strip()) else " " for t in texts]

    embeddings = model.encode(
        cleaned_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)

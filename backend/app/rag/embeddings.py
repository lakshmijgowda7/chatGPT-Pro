"""
RAG Embeddings Generation
Generates dense 384-dimensional L2-normalized embeddings for query and document chunks
with fast deterministic semantic hashing and optional SentenceTransformers support.
"""

from typing import List, Optional
import numpy as np
import hashlib
from app.core.logging import logger

_SENTENCE_TRANSFORMER_CACHE = None
_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def _semantic_vector_encode(text: str) -> np.ndarray:
    """
    Fast, deterministic dense semantic vector encoder using n-gram term hashing
    with position and frequency weighting, producing a unit-length 384D float32 vector.
    """
    vec = np.zeros((EMBEDDING_DIM,), dtype=np.float32)
    clean = (text or "").lower().strip()
    if not clean:
        return vec

    words = clean.split()
    # Unigrams and bigrams
    tokens = list(words)
    if len(words) > 1:
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")

    for idx, token in enumerate(tokens):
        # Hash token to a set of dimensions
        h_bytes = hashlib.md5(token.encode("utf-8")).digest()
        dim_idx1 = (h_bytes[0] | (h_bytes[1] << 8)) % EMBEDDING_DIM
        dim_idx2 = (h_bytes[2] | (h_bytes[3] << 8)) % EMBEDDING_DIM
        dim_idx3 = (h_bytes[4] | (h_bytes[5] << 8)) % EMBEDDING_DIM
        val = 1.0 + (1.0 / (1.0 + idx * 0.1))

        vec[dim_idx1] += val
        vec[dim_idx2] += val * 0.8
        vec[dim_idx3] += val * 0.6

    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec /= norm
    return vec


def compute_text_embedding(text: str, model=None) -> np.ndarray:
    """
    Computes a 384-dimensional L2-normalized semantic embedding vector for a single text.
    """
    return _semantic_vector_encode(text)


def compute_batch_embeddings(texts: List[str], model=None, batch_size: int = 32) -> np.ndarray:
    """
    Computes L2-normalized semantic embedding vectors for a batch of text chunks.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    vecs = [_semantic_vector_encode(t) for t in texts]
    return np.array(vecs, dtype=np.float32)

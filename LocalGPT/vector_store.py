"""
LocalGPT: Local FAISS Vector Store (Step 12: RAG with FAISS)
Provides an in-memory, highly efficient FAISS-based vector index
for document indexing and cosine similarity semantic search.
"""

import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss

from document_loader import DocumentChunk
from embeddings import compute_text_embedding, compute_batch_embeddings, EMBEDDING_DIM


class LocalVectorStore:
    """
    FAISS-powered local vector database for fast semantic retrieval.
    Uses Inner Product (IndexFlatIP) on L2-normalized embeddings for exact cosine similarity.
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.chunks: List[DocumentChunk] = []
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.embedding_dim)

    def clear(self) -> None:
        """
        Clears all indexed chunks and reinitializes the FAISS index.
        """
        self.chunks = []
        self.index = faiss.IndexFlatIP(self.embedding_dim)

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embedding_model=None,
    ) -> int:
        """
        Computes embeddings using SentenceTransformer and indexes chunks into FAISS.
        
        Args:
            chunks: List of DocumentChunk instances.
            embedding_model: Optional preloaded SentenceTransformer model.
            
        Returns:
            Number of newly added chunks.
        """
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = compute_batch_embeddings(texts, model=embedding_model)

        if embeddings.shape[0] == 0 or embeddings.shape[1] != self.embedding_dim:
            return 0

        # Add vectors to FAISS index
        self.index.add(embeddings)
        self.chunks.extend(chunks)

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.0,
        embedding_model=None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Searches for the top-K most semantically similar document chunks using FAISS.
        
        Args:
            query: User search query string.
            top_k: Number of relevant chunks to retrieve.
            similarity_threshold: Minimum cosine similarity score threshold.
            embedding_model: Optional preloaded SentenceTransformer.
            
        Returns:
            List of (DocumentChunk, score) tuples sorted by descending similarity.
        """
        if not self.chunks or self.index.ntotal == 0:
            return []

        if not query or not query.strip():
            return []

        query_vec = compute_text_embedding(query, model=embedding_model)
        query_mat = query_vec.reshape(1, -1).astype(np.float32)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_mat, k)

        results: List[Tuple[DocumentChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            cos_score = float(score)
            if cos_score >= similarity_threshold:
                results.append((self.chunks[idx], cos_score))

        return results

    def save_to_disk(self, directory: str = "data/vector_index") -> bool:
        """
        Persists the FAISS index and chunk metadata to disk.
        """
        try:
            os.makedirs(directory, exist_ok=True)
            index_file = os.path.join(directory, "faiss_index.bin")
            meta_file = os.path.join(directory, "chunks_meta.pkl")

            faiss.write_index(self.index, index_file)
            with open(meta_file, "wb") as f:
                pickle.dump([c.to_dict() for c in self.chunks], f)
            return True
        except Exception:
            return False

    def load_from_disk(self, directory: str = "data/vector_index") -> bool:
        """
        Loads a previously saved FAISS index and chunk metadata from disk.
        """
        index_file = os.path.join(directory, "faiss_index.bin")
        meta_file = os.path.join(directory, "chunks_meta.pkl")

        if not os.path.exists(index_file) or not os.path.exists(meta_file):
            return False

        try:
            self.index = faiss.read_index(index_file)
            with open(meta_file, "rb") as f:
                raw_chunks = pickle.load(f)
            
            self.chunks = []
            for item in raw_chunks:
                self.chunks.append(
                    DocumentChunk(
                        text=item["text"],
                        source=item["source"],
                        chunk_index=item["chunk_index"],
                        metadata=item.get("metadata", {}),
                    )
                )
            return True
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns summary statistics of the indexed vector store.
        """
        sources = set(c.source for c in self.chunks)
        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(sources),
            "sources": sorted(list(sources)),
            "embedding_dim": self.embedding_dim,
            "faiss_total_vectors": self.index.ntotal if self.index is not None else 0,
        }

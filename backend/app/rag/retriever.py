"""
RAG Semantic Vector Retriever
Manages persistent FAISS-based vector database with dense embeddings,
multi-source filtering, disk serialization, and reload after server restarts.
"""

import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
try:
    import faiss
except ImportError:
    faiss = None

from app.rag.splitter import DocumentChunk
from app.rag.embeddings import compute_text_embedding, compute_batch_embeddings, EMBEDDING_DIM
from app.core.config import settings
from app.core.logging import logger


class PersistentVectorRetriever:
    """
    FAISS-powered local vector database for fast semantic retrieval with disk persistence.
    Uses Inner Product (IndexFlatIP) on L2-normalized embeddings for exact cosine similarity.
    """

    def __init__(self, storage_dir: Optional[str] = None, embedding_dim: int = EMBEDDING_DIM):
        self.storage_dir = storage_dir or os.path.join(settings.UPLOAD_DIRECTORY, "vector_store")
        self.embedding_dim = embedding_dim
        self.chunks: List[DocumentChunk] = []
        self._init_faiss_index()
        self.load_from_disk()

    def _init_faiss_index(self):
        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            self.index = None
            self._numpy_vectors = np.empty((0, self.embedding_dim), dtype=np.float32)

    def clear(self) -> None:
        """Clears all indexed chunks and reinitializes the vector index."""
        self.chunks = []
        self._init_faiss_index()
        self.save_to_disk()

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Computes embeddings for chunks and adds them into the vector index and saves to disk.
        """
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = compute_batch_embeddings(texts)

        if embeddings.shape[0] == 0 or embeddings.shape[1] != self.embedding_dim:
            return 0

        if self.index is not None:
            self.index.add(embeddings)
        else:
            self._numpy_vectors = np.vstack([self._numpy_vectors, embeddings]) if len(self._numpy_vectors) else embeddings

        self.chunks.extend(chunks)
        self.save_to_disk()
        logger.info(f"Added {len(chunks)} chunks to vector retriever (Total chunks: {len(self.chunks)})")
        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top relevant chunks for a user query formatted for RAG.
        """
        if not self.chunks or not query.strip():
            return []

        query_vec = compute_text_embedding(query).reshape(1, -1).astype(np.float32)

        results: List[Dict[str, Any]] = []

        if self.index is not None and self.index.ntotal > 0:
            k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(query_vec, k)
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
                if idx < 0 or idx >= len(self.chunks):
                    continue
                cos_score = float(score)
                if cos_score >= similarity_threshold:
                    chunk = self.chunks[idx]
                    results.append({
                        "rank": rank,
                        "source": chunk.source,
                        "page_number": chunk.metadata.get("page_number", 1),
                        "file_type": chunk.metadata.get("file_type", "txt"),
                        "score": round(cos_score, 4),
                        "score_pct": f"{max(0.0, min(100.0, (cos_score + 1.0) / 2.0 * 100)):.1f}%",
                        "text": chunk.text,
                    })
        elif hasattr(self, "_numpy_vectors") and len(self._numpy_vectors) > 0:
            scores = np.dot(self._numpy_vectors, query_vec.T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            for rank, idx in enumerate(top_indices, start=1):
                cos_score = float(scores[idx])
                chunk = self.chunks[idx]
                results.append({
                    "rank": rank,
                    "source": chunk.source,
                    "page_number": chunk.metadata.get("page_number", 1),
                    "file_type": chunk.metadata.get("file_type", "txt"),
                    "score": round(cos_score, 4),
                    "score_pct": f"{max(0.0, min(100.0, (cos_score + 1.0) / 2.0 * 100)):.1f}%",
                    "text": chunk.text,
                })

        return results

    def delete_document_chunks(self, filename: str) -> int:
        """
        Removes all chunks belonging to a specific filename and rebuilds the vector index.
        """
        initial_count = len(self.chunks)
        remaining_chunks = [c for c in self.chunks if c.source != filename and c.metadata.get("filename") != filename]
        removed = initial_count - len(remaining_chunks)

        if removed > 0:
            self.chunks = []
            self._init_faiss_index()
            if remaining_chunks:
                self.add_chunks(remaining_chunks)
            else:
                self.save_to_disk()
            logger.info(f"Removed {removed} chunks for document '{filename}'. Remaining: {len(self.chunks)}")

        return removed

    def save_to_disk(self) -> bool:
        """
        Persists the vector index and chunks metadata to disk.
        """
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            meta_file = os.path.join(self.storage_dir, "chunks_meta.pkl")
            
            with open(meta_file, "wb") as f:
                serialized = [
                    {
                        "text": c.text,
                        "source": c.source,
                        "chunk_index": c.chunk_index,
                        "metadata": c.metadata,
                    }
                    for c in self.chunks
                ]
                pickle.dump(serialized, f)

            if self.index is not None and faiss is not None:
                index_file = os.path.join(self.storage_dir, "faiss_index.bin")
                faiss.write_index(self.index, index_file)
            elif hasattr(self, "_numpy_vectors"):
                vec_file = os.path.join(self.storage_dir, "vectors.npy")
                np.save(vec_file, self._numpy_vectors)

            logger.info(f"Saved vector index to {self.storage_dir} ({len(self.chunks)} chunks)")
            return True
        except Exception as e:
            logger.error(f"Failed to save vector index to disk: {e}")
            return False

    def load_from_disk(self) -> bool:
        """
        Loads a previously saved vector index and chunk metadata from disk upon server restart.
        """
        meta_file = os.path.join(self.storage_dir, "chunks_meta.pkl")
        index_file = os.path.join(self.storage_dir, "faiss_index.bin")
        vec_file = os.path.join(self.storage_dir, "vectors.npy")

        if not os.path.exists(meta_file):
            return False

        try:
            with open(meta_file, "rb") as f:
                raw_chunks = pickle.load(f)

            loaded_chunks = [
                DocumentChunk(
                    text=item["text"],
                    source=item["source"],
                    chunk_index=item.get("chunk_index", 0),
                    metadata=item.get("metadata", {}),
                )
                for item in raw_chunks
            ]

            if faiss is not None and os.path.exists(index_file):
                self.index = faiss.read_index(index_file)
                self.chunks = loaded_chunks
                logger.info(f"Loaded FAISS vector index from {self.storage_dir} ({len(self.chunks)} chunks)")
                return True
            elif os.path.exists(vec_file):
                self._numpy_vectors = np.load(vec_file)
                self.chunks = loaded_chunks
                logger.info(f"Loaded numpy vector index from {self.storage_dir} ({len(self.chunks)} chunks)")
                return True
            else:
                # Re-compute embeddings if vectors were missing but metadata exists
                self.chunks = []
                self._init_faiss_index()
                self.add_chunks(loaded_chunks)
                return True
        except Exception as e:
            logger.warning(f"Failed to load vector index from disk: {e}. Will reinitialize.")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns summary statistics of the indexed vector store.
        """
        sources = set(c.source for c in self.chunks)
        total_vectors = self.index.ntotal if self.index is not None else (len(self._numpy_vectors) if hasattr(self, "_numpy_vectors") else 0)
        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(sources),
            "sources": sorted(list(sources)),
            "embedding_dim": self.embedding_dim,
            "vector_count": total_vectors,
            "storage_dir": self.storage_dir,
        }


rag_retriever = PersistentVectorRetriever()
RAGRetriever = PersistentVectorRetriever


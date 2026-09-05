"""
RAG Text Splitter & Chunking Engine
Splits extracted documents into overlapping character chunks while preserving page numbers.
"""

from typing import List, Dict, Any
from app.rag.loader import ExtractedDocument


class DocumentChunk:
    def __init__(self, text: str, source: str, chunk_index: int, metadata: Dict[str, Any]):
        self.text = text
        self.source = source
        self.chunk_index = chunk_index
        self.metadata = metadata


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    if not text or not text.strip():
        return []
    clean = text.strip()
    if len(clean) <= chunk_size:
        return [clean]

    chunks = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        c = clean[start:end].strip()
        if c:
            chunks.append(c)
        if end == len(clean):
            break
        start += step
    return chunks


def split_document(doc: ExtractedDocument, chunk_size: int = 500, chunk_overlap: int = 50) -> List[DocumentChunk]:
    if not doc.is_valid:
        return []

    chunks: List[DocumentChunk] = []
    idx = 0

    if doc.file_type == "pdf" and doc.pages:
        for pg in doc.pages:
            pg_num = pg.get("page_number", 1)
            pg_text = pg.get("text", "").strip()
            if not pg_text:
                continue
            for piece in chunk_text(pg_text, chunk_size, chunk_overlap):
                chunks.append(DocumentChunk(
                    text=piece,
                    source=doc.filename,
                    chunk_index=idx,
                    metadata={"source": doc.filename, "file_type": doc.file_type, "page_number": pg_num},
                ))
                idx += 1
    else:
        for piece in chunk_text(doc.full_text, chunk_size, chunk_overlap):
            chunks.append(DocumentChunk(
                text=piece,
                source=doc.filename,
                chunk_index=idx,
                metadata={"source": doc.filename, "file_type": doc.file_type, "page_number": 1},
            ))
            idx += 1

    return chunks

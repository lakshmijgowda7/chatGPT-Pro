"""
LocalGPT: Retrieval-Augmented Generation (RAG) (Step 13: Document Source References)
Coordinates document retrieval with FAISS and Qwen local inference to provide
accurate, grounded, citation-backed answers with strict anti-hallucination guardrails
and clean, deduplicated markdown Sources sections.
"""
from typing import Dict, Any, List, Optional, Tuple, Iterator

from vector_store import LocalVectorStore
from document_loader import (
    load_and_chunk_all_documents,
    load_single_file,
    get_documents_directory,
    DocumentChunk,
)
from embeddings import get_embedding_model
from model import load_model_and_tokenizer, generate_chat_response, stream_chat_response
from tokenizer import format_chat_prompt


RAG_SYSTEM_PROMPT = (
    "You are LocalGPT RAG Assistant, a strictly grounded document question-answering assistant. "
    "Your task is to answer the user's question using ONLY the provided document context below.\n\n"
    "Guidelines:\n"
    "1. Rely exclusively on facts mentioned in the context. Do not extrapolate, assume, or invent details.\n"
    "2. Cite the source document name and page number (e.g., [Document: file.pdf, Page 1]) when providing facts.\n"
    "3. If the provided context does NOT contain information to answer the question, state clearly and concisely: "
    "'The provided document(s) do not contain information to answer this question.'\n"
    "4. Keep your answer factual, precise, and well-structured."
)


# -------------------------------------------------------------
# STEP 13 SOURCE FORMATTING UTILITIES
# -------------------------------------------------------------
def format_sources_section(sources: List[Dict[str, Any]]) -> str:
    """
    Formats the list of retrieved sources into a clean markdown Sources section:
    
    Sources:
    
    * research_paper.pdf — Page 4
    * research_paper.pdf — Page 7
    
    Args:
        sources: List of source dictionaries with 'source', 'page_number', and 'file_type'.
        
    Returns:
        Formatted markdown string, or empty string if no sources.
    """
    if not sources:
        return ""

    # Deduplicate citations while preserving retrieval order
    seen_citations = set()
    citation_lines = []

    for src in sources:
        doc_name = src.get("source", "document")
        file_type = src.get("file_type", "").lower()
        pg_num = src.get("page_number")
        
        # If PDF and page_number exists, format: filename — Page X
        if file_type == "pdf" and pg_num is not None:
            cite_str = f"{doc_name} — Page {pg_num}"
        elif pg_num is not None and pg_num > 1:
            cite_str = f"{doc_name} — Page {pg_num}"
        else:
            cite_str = f"{doc_name}"

        if cite_str not in seen_citations:
            seen_citations.add(cite_str)
            citation_lines.append(f"* {cite_str}")

    if not citation_lines:
        return ""

    return "Sources:\n\n" + "\n".join(citation_lines)


def format_answer_with_sources(answer: str, sources: List[Dict[str, Any]]) -> str:
    """
    Appends the formatted Sources section below the generated answer text.
    """
    sources_section = format_sources_section(sources)
    if not sources_section:
        return answer.strip()
    return f"{answer.strip()}\n\n{sources_section}"


class LocalRAG:
    """
    Manages FAISS document indexing and grounded QA inference.
    """

    def __init__(
        self,
        vector_store: Optional[LocalVectorStore] = None,
        model=None,
        tokenizer=None,
        embedding_model=None,
    ):
        self.vector_store = vector_store or LocalVectorStore()
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model

    def _ensure_models_loaded(self):
        if self.model is None or self.tokenizer is None:
            self.model, self.tokenizer = load_model_and_tokenizer()
        if self.embedding_model is None:
            self.embedding_model = get_embedding_model()

    def index_directory(
        self,
        directory_path: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 60,
    ) -> int:
        """
        Indexes all local documents in the directory into the FAISS vector store.
        """
        self._ensure_models_loaded()
        if directory_path is None:
            directory_path = get_documents_directory()

        chunks = load_and_chunk_all_documents(
            directory_path=directory_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.vector_store.clear()
        count = self.vector_store.add_chunks(chunks, embedding_model=self.embedding_model)
        return count

    def index_file(
        self,
        file_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 60,
    ) -> int:
        """
        Indexes a single file into the vector store.
        """
        self._ensure_models_loaded()
        chunks = load_single_file(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        count = self.vector_store.add_chunks(chunks, embedding_model=self.embedding_model)
        return count

    def retrieve_context(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the top-K relevant chunks with similarity scores and page numbers.
        """
        self._ensure_models_loaded()
        raw_results = self.vector_store.search(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            embedding_model=self.embedding_model,
        )

        sources = []
        for rank, (chunk, score) in enumerate(raw_results, 1):
            pg_num = chunk.metadata.get("page_number", 1)
            sources.append({
                "rank": rank,
                "source": chunk.source,
                "page_number": pg_num,
                "file_type": chunk.metadata.get("file_type", "txt"),
                "score": round(score, 4),
                "score_pct": f"{score * 100:.1f}%",
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
            })

        return sources

    def build_rag_prompt(
        self,
        query: str,
        sources: List[Dict[str, Any]],
    ) -> str:
        """
        Formats retrieved sources and question into a grounded ChatML prompt.
        """
        self._ensure_models_loaded()

        if not sources:
            context_text = "No relevant document excerpts found in the local knowledge base."
        else:
            context_blocks = []
            for src in sources:
                pg_str = f"Page {src['page_number']}" if src.get("page_number") else "Page 1"
                header = f"--- [Document: {src['source']} | {pg_str} | Relevance: {src['score_pct']}] ---"
                context_blocks.append(f"{header}\n{src['text']}")
            context_text = "\n\n".join(context_blocks)

        user_content = (
            f"DOCUMENT CONTEXT:\n"
            f"===================================================\n"
            f"{context_text}\n"
            f"===================================================\n\n"
            f"USER QUESTION: {query}\n\n"
            f"Please provide a factual answer based solely on the document context above:"
        )

        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        return format_chat_prompt(messages, tokenizer=self.tokenizer)

    def answer_query(
        self,
        query: str,
        top_k: int = 3,
        temperature: float = 0.1,
        max_new_tokens: int = 384,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end RAG QA: retrieval -> prompt building -> Qwen inference.
        """
        self._ensure_models_loaded()

        # 1. Retrieve most relevant chunks
        sources = self.retrieve_context(query, top_k=top_k)

        # 2. Build grounded prompt
        prompt = self.build_rag_prompt(query, sources)

        # 3. Generate response using Qwen
        gen_res = generate_chat_response(
            formatted_prompt=prompt,
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            do_sample=(temperature > 0.0),
        )

        raw_ans = gen_res.get("response", "")
        formatted_ans = format_answer_with_sources(raw_ans, sources)

        return {
            "query": query,
            "raw_answer": raw_ans,
            "answer": formatted_ans,
            "sources": sources,
            "sources_formatted": format_sources_section(sources),
            "num_sources": len(sources),
            "error": gen_res.get("error"),
            "prompt_tokens": gen_res.get("prompt_tokens", 0),
            "generated_tokens": gen_res.get("generated_tokens", 0),
        }

    def stream_answer_query(
        self,
        query: str,
        top_k: int = 3,
        temperature: float = 0.1,
        max_new_tokens: int = 384,
    ) -> Tuple[Iterator[str], List[Dict[str, Any]]]:
        """
        Streams RAG response tokens for real-time UI display along with retrieved sources.
        """
        self._ensure_models_loaded()

        sources = self.retrieve_context(query, top_k=top_k)
        prompt = self.build_rag_prompt(query, sources)

        stream = stream_chat_response(
            formatted_prompt=prompt,
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            do_sample=(temperature > 0.0),
        )

        return stream, sources

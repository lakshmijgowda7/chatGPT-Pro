"""
LocalGPT: Step 12 RAG Document Question Answering Verification Suite
Tests:
1. Embeddings: Sentence Transformers (all-MiniLM-L6-v2) 384-d normalized vector generation.
2. PDF Ingestion & Page Tracking: Multi-page document chunking with metadata preservation.
3. FAISS Vector Store: Accurate cosine similarity search and ranking via faiss.IndexFlatIP.
4. Grounded Document QA: Asking known factual questions from a multi-page test PDF.
5. Anti-Hallucination Guardrails: Correct refusal when information is not in the context.
6. Independence: Normal chat vs RAG separation.
"""

import os
import sys
import time
import shutil
import tempfile
import numpy as np

# Ensure UTF-8 stdout encoding for Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure LocalGPT path is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embeddings import get_embedding_model, compute_text_embedding, compute_batch_embeddings, EMBEDDING_DIM
from vector_store import LocalVectorStore
from document_loader import load_single_file, extract_text_from_pdf
from rag import LocalRAG
from model import load_model_and_tokenizer, generate_chat_response, clear_memory_cache
from tokenizer import format_chat_prompt


def create_known_knowledge_pdf(file_path: str):
    """
    Creates a multi-page PDF with verified facts for testing RAG QA.
    """
    p1_text = (
        "CONFIDENTIAL SPECIFICATION - PROJECT AURORA PRIME\n"
        "Project Code: NEXUS-42\n"
        "Lead System Architect: Dr. Elena Rostova\n"
        "Quantum encryption key length is exactly 4096 bits.\n"
        "Primary datacenter location: Reykjavik Subterranean Facility."
    )
    p2_text = (
        "REACTOR SUBSYSTEM TECHNICAL REPORT\n"
        "Operational resonance frequency: 1420.4 MHz\n"
        "Cooling liquid: Fluorinert FC-72\n"
        "Emergency containment protocol: Alpha-Nine\n"
        "Maximum thermal threshold: 850 Kelvin."
    )

    # Clean PDF bytes with 2 distinct pages
    p1_bytes = f"BT /F1 12 Tf 50 700 Td ({p1_text.replace(chr(10), ') Tj T* (')}) Tj ET".encode("ascii", errors="ignore")
    p2_bytes = f"BT /F1 12 Tf 50 700 Td ({p2_text.replace(chr(10), ') Tj T* (')}) Tj ET".encode("ascii", errors="ignore")

    pdf_body = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R /Resources << /Font << /F1 7 0 R >> >> >> endobj\n"
        b"4 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 6 0 R /Resources << /Font << /F1 7 0 R >> >> >> endobj\n"
        b"5 0 obj << /Length " + str(len(p1_bytes)).encode() + b" >> stream\n" + p1_bytes + b"\nendstream endobj\n"
        b"6 0 obj << /Length " + str(len(p2_bytes)).encode() + b" >> stream\n" + p2_bytes + b"\nendstream endobj\n"
        b"7 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"xref\n"
        b"0 8\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000236 00000 n \n"
        b"0000000357 00000 n \n"
        b"0000000500 00000 n \n"
        b"0000000650 00000 n \n"
        b"trailer << /Size 8 /Root 1 0 R >>\n"
        b"startxref\n"
        b"750\n"
        b"%%EOF\n"
    )

    with open(file_path, "wb") as f:
        f.write(pdf_body)


def run_step12_tests():
    print("=" * 75)
    print("LocalGPT Step 12: RAG Document Question Answering Verification Suite")
    print("=" * 75)

    test_dir = tempfile.mkdtemp(prefix="localgpt_step12_rag_")
    results = {}

    try:
        # 1. Create test PDF
        pdf_path = os.path.join(test_dir, "aurora_spec.pdf")
        create_known_knowledge_pdf(pdf_path)

        # -------------------------------------------------------------
        # Test 1: SentenceTransformers Embedding Generation
        # -------------------------------------------------------------
        print("\n[Test 1/6] Loading SentenceTransformers & Computing Embeddings...")
        t0 = time.time()
        emb_model = get_embedding_model()
        t_emb_load = time.time() - t0
        print(f"  [+] SentenceTransformer loaded in {t_emb_load:.2f}s")
        
        sample_query = "Who is the lead system architect for Project Aurora?"
        vec = compute_text_embedding(sample_query, model=emb_model)
        assert vec.shape == (EMBEDDING_DIM,), f"Expected shape ({EMBEDDING_DIM},), got {vec.shape}"
        norm = float(np.linalg.norm(vec))
        print(f"  [+] Vector shape: {vec.shape} | L2 Norm: {norm:.4f}")
        assert abs(norm - 1.0) < 1e-4, f"Vector should be L2 unit normalized, got {norm}"
        results["SentenceTransformers Embeddings"] = True

        # -------------------------------------------------------------
        # Test 2: Multi-Page PDF Ingestion & Page Tracking
        # -------------------------------------------------------------
        print("\n[Test 2/6] Ingesting & Chunking Multi-Page Test PDF...")
        chunks = load_single_file(pdf_path, chunk_size=300, chunk_overlap=40)
        print(f"  [+] Chunks created: {len(chunks)}")
        pages_seen = set()
        for c in chunks:
            pg = c.metadata.get("page_number")
            pages_seen.add(pg)
            print(f"      - Chunk #{c.chunk_index} | Page {pg} | {len(c.text)} chars: '{c.text[:60]}...'")
        
        assert len(chunks) >= 2, "Should create at least 2 chunks for 2 pages"
        assert 1 in pages_seen and 2 in pages_seen, f"Expected pages [1, 2], got {pages_seen}"
        results["PDF Ingestion & Page Preservation"] = True

        # -------------------------------------------------------------
        # Test 3: FAISS Vector Store Indexing & Similarity Search
        # -------------------------------------------------------------
        print("\n[Test 3/6] Indexing into FAISS Vector Store...")
        vstore = LocalVectorStore(embedding_dim=EMBEDDING_DIM)
        count = vstore.add_chunks(chunks, embedding_model=emb_model)
        print(f"  [+] FAISS Index total vectors: {vstore.index.ntotal}")
        assert count == len(chunks)
        assert vstore.index.ntotal == len(chunks)

        # Search Query 1 (from Page 1)
        res_q1 = vstore.search("lead system architect and project code", top_k=2, embedding_model=emb_model)
        assert len(res_q1) > 0
        top_chunk_q1, score_q1 = res_q1[0]
        print(f"  [+] Search Q1 Top Result: Page {top_chunk_q1.metadata.get('page_number')} (Score: {score_q1:.4f})")
        assert top_chunk_q1.metadata.get("page_number") == 1
        assert "Elena Rostova" in top_chunk_q1.text

        # Search Query 2 (from Page 2)
        res_q2 = vstore.search("reactor resonance frequency and cooling liquid", top_k=2, embedding_model=emb_model)
        assert len(res_q2) > 0
        top_chunk_q2, score_q2 = res_q2[0]
        print(f"  [+] Search Q2 Top Result: Page {top_chunk_q2.metadata.get('page_number')} (Score: {score_q2:.4f})")
        assert top_chunk_q2.metadata.get("page_number") == 2
        assert "1420.4 MHz" in top_chunk_q2.text or "Fluorinert" in top_chunk_q2.text
        results["FAISS Similarity Search"] = True

        # -------------------------------------------------------------
        # Test 4: End-to-End Grounded Qwen QA (Questions inside PDF)
        # -------------------------------------------------------------
        print("\n[Test 4/6] Testing Grounded Question Answering with Qwen LLM...")
        model, tokenizer = load_model_and_tokenizer()
        rag = LocalRAG(vector_store=vstore, model=model, tokenizer=tokenizer, embedding_model=emb_model)

        # Question A (from Page 1)
        q_a = "Who is the lead system architect and what is the project code?"
        ans_a = rag.answer_query(q_a, top_k=2, temperature=0.0)
        print(f"  [+] Q: '{q_a}'")
        print(f"  [+] Answer: '{ans_a['answer']}'")
        print(f"  [+] Sources: {[(s['source'], s['page_number'], s['score_pct']) for s in ans_a['sources']]}")
        
        ans_a_lower = ans_a["answer"].lower()
        assert "elena" in ans_a_lower or "rostova" in ans_a_lower, f"Answer missing architect: {ans_a['answer']}"
        assert "nexus-42" in ans_a_lower or "nexus" in ans_a_lower, f"Answer missing project code: {ans_a['answer']}"

        # Question B (from Page 2)
        q_b = "What is the operational resonance frequency and the cooling liquid?"
        ans_b = rag.answer_query(q_b, top_k=2, temperature=0.0)
        print(f"\n  [+] Q: '{q_b}'")
        print(f"  [+] Answer: '{ans_b['answer']}'")
        print(f"  [+] Sources: {[(s['source'], s['page_number'], s['score_pct']) for s in ans_b['sources']]}")
        
        ans_b_lower = ans_b["answer"].lower()
        assert "1420.4" in ans_b_lower or "mhz" in ans_b_lower, f"Answer missing frequency: {ans_b['answer']}"
        assert "fluorinert" in ans_b_lower or "fc-72" in ans_b_lower, f"Answer missing cooling liquid: {ans_b['answer']}"
        results["Grounded Qwen QA (Inside PDF)"] = True

        # -------------------------------------------------------------
        # Test 5: Anti-Hallucination Guardrail (Question NOT in PDF)
        # -------------------------------------------------------------
        print("\n[Test 5/6] Testing Anti-Hallucination for Unknown/Absent Facts...")
        q_c = "What is the flight speed of the supersonic fighter jet and the price of Bitcoin in 2050?"
        ans_c = rag.answer_query(q_c, top_k=2, temperature=0.0)
        print(f"  [+] Q: '{q_c}'")
        print(f"  [+] Answer: '{ans_c['answer']}'")
        
        ans_c_lower = ans_c["answer"].lower()
        # Should state information not present in the document
        not_found_keywords = ["not", "does not contain", "cannot be found", "no information", "not mentioned", "unmentioned"]
        assert any(kw in ans_c_lower for kw in not_found_keywords), f"Model hallucinated absent facts: {ans_c['answer']}"
        print("  [+] Anti-hallucination guardrail confirmed: model accurately stated absence of information.")
        results["Anti-Hallucination Guardrail"] = True

        # -------------------------------------------------------------
        # Test 6: Normal Chat Mode Independence
        # -------------------------------------------------------------
        print("\n[Test 6/6] Verifying Normal Chat Mode Operates Independently...")
        normal_prompt = format_chat_prompt([
            {"role": "system", "content": "You are a helpful conversational assistant."},
            {"role": "user", "content": "Say 'Normal conversation mode is online.' and nothing else."},
        ], tokenizer=tokenizer)
        normal_res = generate_chat_response(normal_prompt, model=model, tokenizer=tokenizer, max_new_tokens=20, temperature=0.0)
        print(f"  [+] Normal Chat Output: '{normal_res['response']}'")
        assert "normal conversation" in normal_res["response"].lower() or "online" in normal_res["response"].lower()
        results["Normal Chat Independence"] = True

        clear_memory_cache()

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)

    # Final Report
    print("\n" + "=" * 75)
    print("STEP 12 RAG DOCUMENT QA VERIFICATION RESULTS:")
    print("=" * 75)
    all_passed = True
    for test_name, passed in results.items():
        status_box = "[X]" if passed else "[ ]"
        status_str = "PASS" if passed else "FAIL"
        print(f"  {status_box} {test_name:<48} : {status_str}")
        if not passed:
            all_passed = False

    print("=" * 75)
    if all_passed:
        print("ALL STEP 12 RAG DOCUMENT QA TESTS PASSED SUCCESSFULLY! (6/6)")
    else:
        print("SOME TESTS FAILED.")
    print("=" * 75)
    return all_passed


if __name__ == "__main__":
    success = run_step12_tests()
    if not success:
        sys.exit(1)

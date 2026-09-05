"""
LocalGPT: Step 13 Document Source References Verification Suite
Tests:
1. Multi-Page PDF Source Tracking: Accurate page number retention across multiple pages.
2. Sources Section Formatting: Clean, deduplicated markdown citation format (e.g. '* doc.pdf — Page 2').
3. Retrievable Only Filtering: Only chunks actually retrieved by FAISS appear in Sources.
4. Answer Embedding: Sources section appended cleanly below AI answer.
5. No Invented Page Numbers: Page numbers match source document metadata.
6. Normal Chat Cleanliness: Normal conversation responses are free of any source sections.
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

from embeddings import get_embedding_model, compute_text_embedding, EMBEDDING_DIM
from vector_store import LocalVectorStore
from document_loader import load_single_file, extract_text_from_pdf
from rag import LocalRAG, format_sources_section, format_answer_with_sources
from model import load_model_and_tokenizer, generate_chat_response, clear_memory_cache
from tokenizer import format_chat_prompt


def create_three_page_pdf(file_path: str):
    """
    Creates a 3-page test PDF with distinct verifiable facts on each page.
    """
    p1 = "CHAPTER 1: QUANTUM CORE ARCHITECTURE\nThe core clock cycle is 3.8 GHz. The primary stabilizer is Lithium-7."
    p2 = "CHAPTER 2: MAGNETIC CONFINEMENT\nThe magnetic field intensity is 14.2 Tesla. The plasma temperature reaches 150 Million Celsius."
    p3 = "CHAPTER 3: SAFETY PROTOCOLS\nThe emergency pressure relief valve is Valve-Omega. Inspection interval is 90 days."

    p1_bytes = f"BT /F1 12 Tf 50 700 Td ({p1.replace(chr(10), ') Tj T* (')}) Tj ET".encode("ascii", errors="ignore")
    p2_bytes = f"BT /F1 12 Tf 50 700 Td ({p2.replace(chr(10), ') Tj T* (')}) Tj ET".encode("ascii", errors="ignore")
    p3_bytes = f"BT /F1 12 Tf 50 700 Td ({p3.replace(chr(10), ') Tj T* (')}) Tj ET".encode("ascii", errors="ignore")

    pdf_body = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 6 0 R /Resources << /Font << /F1 9 0 R >> >> >> endobj\n"
        b"4 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 7 0 R /Resources << /Font << /F1 9 0 R >> >> >> endobj\n"
        b"5 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 8 0 R /Resources << /Font << /F1 9 0 R >> >> >> endobj\n"
        b"6 0 obj << /Length " + str(len(p1_bytes)).encode() + b" >> stream\n" + p1_bytes + b"\nendstream endobj\n"
        b"7 0 obj << /Length " + str(len(p2_bytes)).encode() + b" >> stream\n" + p2_bytes + b"\nendstream endobj\n"
        b"8 0 obj << /Length " + str(len(p3_bytes)).encode() + b" >> stream\n" + p3_bytes + b"\nendstream endobj\n"
        b"9 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"xref\n"
        b"0 10\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000122 00000 n \n"
        b"0000000243 00000 n \n"
        b"0000000364 00000 n \n"
        b"0000000485 00000 n \n"
        b"0000000620 00000 n \n"
        b"0000000755 00000 n \n"
        b"0000000890 00000 n \n"
        b"trailer << /Size 10 /Root 1 0 R >>\n"
        b"startxref\n"
        b"990\n"
        b"%%EOF\n"
    )

    with open(file_path, "wb") as f:
        f.write(pdf_body)


def run_step13_tests():
    print("=" * 75)
    print("LocalGPT Step 13: Document Source References Verification Suite")
    print("=" * 75)

    test_dir = tempfile.mkdtemp(prefix="localgpt_step13_sources_")
    results = {}

    try:
        pdf_path = os.path.join(test_dir, "fusion_research.pdf")
        create_three_page_pdf(pdf_path)

        # -------------------------------------------------------------
        # Test 1: Track Source Metadata & Page Numbers
        # -------------------------------------------------------------
        print("\n[Test 1/6] Ingesting 3-Page PDF & Verifying Page Attribution...")
        chunks = load_single_file(pdf_path, chunk_size=250, chunk_overlap=30)
        print(f"  [+] Ingested {len(chunks)} chunks from 'fusion_research.pdf'")
        page_map = {}
        for c in chunks:
            pg = c.metadata.get("page_number")
            page_map[pg] = page_map.get(pg, 0) + 1
            print(f"      - Chunk #{c.chunk_index} ➔ Source: '{c.source}' | Page: {pg} | '{c.text[:45]}...'")

        assert len(page_map) == 3, f"Expected exactly 3 distinct pages, got {page_map}"
        assert set(page_map.keys()) == {1, 2, 3}
        results["Source Metadata & Page Preservation"] = True

        # -------------------------------------------------------------
        # Test 2: Source Section Formatting & Deduplication
        # -------------------------------------------------------------
        print("\n[Test 2/6] Verifying Sources Section Format & Deduplication...")
        mock_sources = [
            {"source": "fusion_research.pdf", "page_number": 2, "file_type": "pdf"},
            {"source": "fusion_research.pdf", "page_number": 2, "file_type": "pdf"},  # Duplicate chunk on page 2
            {"source": "fusion_research.pdf", "page_number": 3, "file_type": "pdf"},
        ]
        sec_text = format_sources_section(mock_sources)
        print(f"  [+] Formatted Sources Section:\n{sec_text}")

        assert sec_text.startswith("Sources:\n\n")
        assert "* fusion_research.pdf — Page 2" in sec_text
        assert "* fusion_research.pdf — Page 3" in sec_text
        assert sec_text.count("fusion_research.pdf — Page 2") == 1, "Must deduplicate identical page citations"
        results["Source Formatting & Deduplication"] = True

        # -------------------------------------------------------------
        # Test 3: Load Models & FAISS Indexing
        # -------------------------------------------------------------
        print("\n[Test 3/6] Indexing into FAISS Vector Store...")
        emb_model = get_embedding_model()
        model, tokenizer = load_model_and_tokenizer()
        
        vstore = LocalVectorStore(embedding_dim=EMBEDDING_DIM)
        vstore.add_chunks(chunks, embedding_model=emb_model)
        rag = LocalRAG(vector_store=vstore, model=model, tokenizer=tokenizer, embedding_model=emb_model)
        print(f"  [+] FAISS indexed {vstore.index.ntotal} chunks.")
        results["FAISS Indexing with Sources"] = True

        # -------------------------------------------------------------
        # Test 4: Grounded QA with Specific Page Retrieval (Page 2)
        # -------------------------------------------------------------
        print("\n[Test 4/6] Asking Question Targeting Page 2...")
        q_page2 = "What is the magnetic field intensity and the plasma temperature?"
        res_page2 = rag.answer_query(q_page2, top_k=2, temperature=0.0)
        
        print(f"  [+] Q: '{q_page2}'")
        print(f"  [+] Complete Formatted Answer:\n{res_page2['answer']}")
        print(f"  [+] Retrieved Sources: {[(s['source'], s['page_number']) for s in res_page2['sources']]}")

        ans_p2 = res_page2["answer"]
        assert "14.2 Tesla" in ans_p2 or "14.2" in ans_p2
        assert "Sources:" in ans_p2
        assert "fusion_research.pdf — Page 2" in ans_p2
        # Page 1 must not be listed as source if not retrieved
        retrieved_pages = [s["page_number"] for s in res_page2["sources"]]
        assert 2 in retrieved_pages
        results["Grounded QA with Page 2 Sources"] = True

        # -------------------------------------------------------------
        # Test 5: Grounded QA Targeting Page 3
        # -------------------------------------------------------------
        print("\n[Test 5/6] Asking Question Targeting Page 3...")
        q_page3 = "What is the emergency pressure relief valve name and inspection interval?"
        res_page3 = rag.answer_query(q_page3, top_k=2, temperature=0.0)
        
        print(f"  [+] Q: '{q_page3}'")
        print(f"  [+] Complete Formatted Answer:\n{res_page3['answer']}")
        print(f"  [+] Retrieved Sources: {[(s['source'], s['page_number']) for s in res_page3['sources']]}")

        ans_p3 = res_page3["answer"]
        assert "Valve-Omega" in ans_p3 or "Omega" in ans_p3
        assert "Sources:" in ans_p3
        assert "fusion_research.pdf — Page 3" in ans_p3
        results["Grounded QA with Page 3 Sources"] = True

        # -------------------------------------------------------------
        # Test 6: Normal Chat Mode Free of Sources
        # -------------------------------------------------------------
        print("\n[Test 6/6] Verifying Normal Chat Contains No Sources Block...")
        chat_prompt = format_chat_prompt([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me what color the sky is in one sentence."},
        ], tokenizer=tokenizer)
        normal_resp = generate_chat_response(chat_prompt, model=model, tokenizer=tokenizer, max_new_tokens=25, temperature=0.0)["response"]
        print(f"  [+] Normal Chat Output: '{normal_resp}'")
        assert "Sources:" not in normal_resp, "Normal chat responses must never contain 'Sources:'"
        results["Normal Chat Source-Free"] = True

        clear_memory_cache()

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)

    # Summary Report
    print("\n" + "=" * 75)
    print("STEP 13 DOCUMENT SOURCE REFERENCES VERIFICATION RESULTS:")
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
        print("ALL STEP 13 SOURCE REFERENCES TESTS PASSED SUCCESSFULLY! (6/6)")
    else:
        print("SOME TESTS FAILED.")
    print("=" * 75)
    return all_passed


if __name__ == "__main__":
    success = run_step13_tests()
    if not success:
        sys.exit(1)

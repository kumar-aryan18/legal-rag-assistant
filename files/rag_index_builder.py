"""Build a FAISS index from every PDF in ./docs.

Usage:
    python rag_index_builder.py                 # index all PDFs in ./docs
    python rag_index_builder.py path/to/file.pdf  # index one file
"""
import sys
import time
from pathlib import Path

import pymupdf
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR, INDEX_DIR, get_embeddings
from overview import save_overviews  # NEW
from pdf_tables import assemble_page_text  # NEW

# Gemini free tier allows ~100 embedding requests per minute.
BATCH_SIZE = 80
PAUSE_SECONDS = 62


def load_pdf_pages(pdf_path: Path) -> list[Document]:
    """One Document per page so every chunk keeps its source file and page number."""
    pages = []
    with pymupdf.open(pdf_path) as pdf:
        for number, page in enumerate(pdf, start=1):
            text = page.get_text().strip()
            try:  # NEW: keep table rows together ("Offence: Murder | IPC section: 302 | ...")
                tables = [(t.bbox, t.extract()) for t in page.find_tables().tables]
                if tables:
                    text = assemble_page_text(page.get_text("blocks"), tables).strip() or text
            except Exception:
                pass  # older PyMuPDF without find_tables(): keep the plain text
            if text:
                pages.append(
                    Document(
                        page_content=text,
                        metadata={"source": pdf_path.name, "page": number},
                    )
                )
    return pages


def split_documents(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )
    return splitter.split_documents(pages)


def _is_rate_limit(error: Exception) -> bool:
    text = str(error)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def build_store_throttled(chunks: list[Document], embeddings) -> FAISS:
    """Embed in small batches so the free-tier per-minute limit is not exceeded."""
    store = None
    total = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        for attempt in range(6):
            try:
                if store is None:
                    store = FAISS.from_documents(batch, embeddings)
                else:
                    store.add_documents(batch)
                break
            except Exception as e:
                if _is_rate_limit(e) and attempt < 5:
                    print("  Rate limited, waiting 65 seconds ...")
                    time.sleep(65)
                else:
                    raise
        done = min(start + BATCH_SIZE, total)
        print(f"  Embedded {done}/{total} chunks")
        if done < total:
            time.sleep(PAUSE_SECONDS)
    return store


def build_index(pdf_paths: list[Path], persist_dir: Path = INDEX_DIR) -> int:
    pages: list[Document] = []
    for pdf_path in pdf_paths:
        print(f"Reading {pdf_path.name} ...")
        pages.extend(load_pdf_pages(pdf_path))

    if not pages:
        raise RuntimeError(
            "No extractable text found. Scanned PDFs need OCR before indexing."
        )

    save_overviews(pages, persist_dir)  # NEW: per-document outlines for the overview tool

    chunks = split_documents(pages)
    print(f"Created {len(chunks)} chunks from {len(pages)} pages. Embedding ...")

    store = build_store_throttled(chunks, get_embeddings())
    store.save_local(str(persist_dir))
    print(f"Index saved to {persist_dir}")
    return len(chunks)


def build_index_from_docs_folder() -> int:
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDF files found in {DOCS_DIR}. Add some and retry.")
    return build_index(pdfs)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        build_index([Path(p) for p in sys.argv[1:]])
    else:
        build_index_from_docs_folder()
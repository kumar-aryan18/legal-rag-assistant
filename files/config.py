"""Shared configuration for the Legal RAG Assistant (local Ollama + FAISS)."""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

BASE_DIR = Path(__file__).parent

# Optional: loads ".env" and "_env" if present. Everything has a default,
# so no key or account is needed.
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / "_env")

DOCS_DIR = BASE_DIR / "docs"
INDEX_DIR = BASE_DIR / "rag_faiss_store"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
# Ollama's default context window is small; retrieved passages need more room.
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


def get_llm():
    return ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        num_ctx=NUM_CTX,
        temperature=0,
        keep_alive="30m",   # keep the model loaded between questions
        num_predict=400,    # cap answer length
    )
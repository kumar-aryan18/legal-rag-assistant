"""Shared configuration for the Legal RAG Assistant (Gemini + FAISS)."""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

BASE_DIR = Path(__file__).parent

# Loads ".env" and "_env" if present (local development).
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / "_env")

DOCS_DIR = BASE_DIR / "docs"
INDEX_DIR = BASE_DIR / "rag_faiss_store"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4


def _get_api_key() -> str | None:
    """Read the key from env/.env locally, or from Streamlit secrets when deployed."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


GEMINI_API_KEY = _get_api_key()
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "700"))  # raise for longer answers


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY,
    )


def get_llm():
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_budget=0,  # no hidden "thinking" tokens eating into the answer length
    )
    
def chunk_text(content) -> str:
    """Turn a model chunk or message content (string or list of blocks) into plain text."""
    content = getattr(content, "content", content)  # accept a message object too
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return ""
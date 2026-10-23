# Legal RAG Assistant (fully local)

Ask questions about your legal PDFs (contracts, acts, policies). Answers use only
the retrieved passages and cite `[file, p. N]`. Runs entirely on your computer
with Ollama: no API key, no account, documents never leave your machine.

## Setup
1. Install Ollama from https://ollama.com and make sure it is running.
2. Download the models (one time):
   ```
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```
   Low on RAM? Use a smaller chat model, e.g. `ollama pull llama3.2:3b`, and set
   `OLLAMA_CHAT_MODEL=llama3.2:3b` in a `.env` file.
3. `pip install -r requirements.txt`
4. Put PDFs in `docs/`.
5. Build the index: `python rag_index_builder.py`
6. Chat: `python cli.py` or `streamlit run app.py`

## Files
- `config.py` settings and Ollama clients
- `rag_index_builder.py` PDF -> chunks (with page numbers) -> FAISS index
- `legal_rag.py` retrieval + grounded answer
- `cli.py`, `app.py` terminal and web interfaces

## Notes
- Scanned (image-only) PDFs have no text layer and need OCR first.
- Re-run the index builder whenever you add or change documents.
- If you change the embedding model, rebuild the index.
- Small local models can miss details or phrase things loosely. Always verify
  against the cited page. Not legal advice.

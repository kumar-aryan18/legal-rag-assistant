# Legal RAG Assistant

Ask questions about your legal PDFs (contracts, acts, policies). Answers use only
the retrieved passages and cite `[file, p. N]`. It uses Google Gemini for
answers and embeddings, with a local FAISS index for search.

**Privacy note:** the question and the retrieved passages are sent to the Gemini
API to generate each answer, and document text is sent to create embeddings when
the index is built. Don't upload confidential or private documents.

## Setup
1. Get a free Gemini API key at https://aistudio.google.com (Get API key).
2. Copy `.env.template` to `.env` and set your key:
   ```
   GEMINI_API_KEY=your_key_here
   ```
   Never commit `.env` or share your key.
3. `pip install -r requirements.txt`
4. Put PDFs in `docs/`.
5. Build the index: `python rag_index_builder.py`
6. Chat: `python cli.py` or `streamlit run app.py`

## Deploying (Streamlit Community Cloud)
- Push the project to GitHub (without `.env` or `venv/`).
- Set the main file to `files/app.py`.
- Add `GEMINI_API_KEY =
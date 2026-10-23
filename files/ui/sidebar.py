"""Sidebar: upload, document list, index status, search settings, chat tools."""
import html
from datetime import datetime

import streamlit as st

from config import DOCS_DIR, INDEX_DIR, TOP_K
from rag_index_builder import build_index_from_docs_folder
from ui.components import chat_to_markdown


def _rebuild_or_clear() -> int:
    """Rebuild the index from the remaining PDFs, or remove it if none are left."""
    if any(DOCS_DIR.glob("*.pdf")):
        n = build_index_from_docs_folder()
    else:
        n = 0
        for f in INDEX_DIR.glob("index.*"):
            f.unlink()
    st.cache_resource.clear()
    return n


def render_sidebar() -> dict:
    """Draws the sidebar and returns the search settings for the current run."""
    with st.sidebar:
        st.header("⚖️ Documents")

        uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if st.button("Save & build index", disabled=not uploads, type="primary"):
            for f in uploads:
                (DOCS_DIR / f.name).write_bytes(f.getbuffer())
            with st.spinner("Building index..."):
                try:
                    n = build_index_from_docs_folder()
                    st.cache_resource.clear()
                    st.success(f"Indexed {n} chunks.")
                except Exception as e:
                    st.error(str(e))

        pdfs = sorted(DOCS_DIR.glob("*.pdf"))
        names = [p.name for p in pdfs]

        # --- index status ---
        index_file = INDEX_DIR / "index.faiss"
        if index_file.exists():
            built = index_file.stat().st_mtime
            st.caption(
                f"{len(pdfs)} document(s) · index built "
                f"{datetime.fromtimestamp(built).strftime('%d %b %Y, %H:%M')}"
            )
            if any(p.stat().st_mtime > built for p in pdfs):
                st.warning("Some PDFs were added after the last index build.")
        else:
            st.caption("No index yet. Upload PDFs and build the index.")

        # --- document list with delete ---
        deleted = None
        for p in pdfs:
            c1, c2 = st.columns([5, 1])
            c1.markdown(
                f"<div class='doc-name'>📄 {html.escape(p.name)}</div>",
                unsafe_allow_html=True,
            )
            if c2.button("🗑️", key=f"del_{p.name}", help=f"Delete {p.name} and rebuild index"):
                deleted = p
        if deleted is not None:
            rebuilt_ok = True
            with st.spinner("Removing document and rebuilding index..."):
                try:
                    deleted.unlink()
                    _rebuild_or_clear()
                except Exception as e:
                    rebuilt_ok = False
                    st.error(str(e))
            if rebuilt_ok:
                st.rerun()

        # --- search settings ---
        st.divider()
        st.subheader("Search settings")
        # Drop stale selections (e.g. a document that was just deleted).
        st.session_state["doc_filter"] = [
            d for d in st.session_state.get("doc_filter", []) if d in names
        ]
        st.multiselect(
            "Search only in",
            names,
            key="doc_filter",
            help="Leave empty to search all documents.",
        )
        top_k = st.slider("Passages to retrieve", 1, 10, TOP_K)

        # --- chat tools ---
        st.divider()
        msgs = st.session_state.get("messages", [])
        c1, c2 = st.columns(2)
        if c1.button("Clear chat", disabled=not msgs):
            st.session_state.messages = []
            st.rerun()
        c2.download_button(
            "Export",
            chat_to_markdown(msgs),
            file_name="legal_chat.md",
            mime="text/markdown",
            disabled=not msgs,
        )

    return {"top_k": top_k, "sources": st.session_state["doc_filter"] or None}

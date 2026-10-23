"""Streamlit web UI.  Run: streamlit run app.py"""
import streamlit as st

from config import DOCS_DIR, INDEX_DIR
from legal_rag import NOT_FOUND_MESSAGE, NOT_FOUND_PHRASE, LegalRAG
from ui.components import assistant_extras, quick_actions
from ui.sidebar import render_sidebar
from ui.styles import inject_css

st.set_page_config(page_title="Legal RAG Assistant", page_icon="⚖️", layout="wide")
inject_css()
DOCS_DIR.mkdir(exist_ok=True)

settings = render_sidebar()

st.title("⚖️ Legal RAG Assistant")
st.caption("Answers come only from your uploaded documents. This is not legal advice.")


@st.cache_resource
def get_rag():
    return LegalRAG()


if not (INDEX_DIR / "index.faiss").exists():
    st.info("No index yet. Upload PDFs in the sidebar and click **Save & build index**.")
    st.stop()

try:
    rag = get_rag()
except Exception as e:
    st.error(str(e))
    st.stop()

st.session_state.setdefault("messages", [])

# Replay the conversation.
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            assistant_extras(m.get("sources"), m.get("passages"))

# Starter questions on an empty chat.
clicked = quick_actions() if not st.session_state.messages else None

question = st.chat_input("Ask a question about your legal documents") or clicked
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Rebuild (user, assistant) pairs from earlier turns for follow-up context.
    msgs = st.session_state.messages[:-1]
    history = [
        (msgs[i]["content"], msgs[i + 1]["content"])
        for i in range(0, len(msgs) - 1, 2)
    ]

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents..."):
                docs, stream = rag.ask_stream(
                    question, history, k=settings["top_k"], sources=settings["sources"]
                )
            if docs:
                answer = st.write_stream(stream)
            else:
                answer = NOT_FOUND_MESSAGE
                st.markdown(answer)
        except Exception as e:
            st.session_state.messages.pop()  # keep user/assistant turns paired
            st.error(f"Could not generate an answer: {e}. Is Ollama running?")
            st.stop()

        not_found = NOT_FOUND_PHRASE in answer.lower()
        sources = [] if not_found else rag.collect_sources(docs)
        chunks = [
            {"source": d.metadata["source"], "page": d.metadata["page"], "text": d.page_content}
            for d in docs
        ]
        assistant_extras(sources, chunks)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "passages": chunks}
    )

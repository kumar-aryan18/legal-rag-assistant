"""Streamlit web UI.  Run: streamlit run app.py"""
from datetime import date

import streamlit as st

import overview
import router
import tools
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


def citation_warning(missing, stray):
    parts = []
    if missing:
        parts.append("section/article " + ", ".join(missing))
    if stray:
        parts.append("the number(s) " + ", ".join(stray))
    return (
        "Check before relying on this answer: " + " and ".join(parts)
        + " do not appear in the retrieved passages, so they may be wrong."
    )


def render_overview(docs):
    """One document -> short LLM summary + contents list. Several -> instant list, no LLM."""
    if len(docs) == 1:
        summary = st.write_stream(overview.stream_overview(docs[0]))
        outline = overview.outline_markdown(docs[0])
        if outline:
            st.markdown(outline)
        return f"{summary}\n\n{outline}".strip()
    text = overview.all_docs_markdown(docs)
    st.markdown(text)
    return text


# Deadline calculator form (same tool the router uses for typed questions).
with st.expander("📅 Deadline calculator"):
    kind = st.selectbox("Situation", list(tools.KINDS), format_func=tools.KINDS.get, key="dl_kind")
    start = st.date_input(tools.START_LABELS[kind], value=date.today(), key=f"dl_start_{kind}")
    if st.button("Calculate deadlines", key="dl_go"):
        st.markdown(tools.calculate_deadlines(kind, start))

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
            if m.get("label"):
                st.caption(m["label"])
            if m.get("warning"):
                st.warning(m["warning"])

# Starter questions on an empty chat. The slot is cleared as soon as a question is asked.
chips_slot = st.empty()
clicked = None
if not st.session_state.messages:
    with chips_slot.container():
        clicked = quick_actions()

question = st.chat_input("Ask a question about your legal documents") or clicked
if question:
    chips_slot.empty()
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
        sources, chunks, warning, label = [], [], None, None
        chosen = router.route(question, settings["sources"])
        try:
            if chosen.kind == "tool":  # exact answer, no LLM
                answer, label = chosen.text, chosen.label
                st.markdown(answer)
            elif chosen.kind == "overview":
                answer, label = render_overview(chosen.docs), chosen.label
            else:  # normal RAG
                with st.spinner("Searching documents..."):
                    docs, stream = rag.ask_stream(
                        question, history, k=settings["top_k"], sources=settings["sources"]
                    )
                if docs:
                    answer = st.write_stream(stream)
                else:  # relevance guard: nothing relevant enough, so skip the LLM entirely
                    answer = NOT_FOUND_MESSAGE
                    st.markdown(answer)
        except Exception as e:
            st.session_state.messages.pop()  # keep user/assistant turns paired
            st.error(
                f"Could not generate an answer: {e}. "
                "Check that GEMINI_API_KEY is set correctly, or wait a minute if you hit the free-tier rate limit."
            )
            st.stop()

        if chosen.kind == "rag":
            not_found = NOT_FOUND_PHRASE in answer.lower()
            sources = [] if not_found else rag.collect_sources(docs)
            chunks = [
                {"source": d.metadata["source"], "page": d.metadata["page"], "text": d.page_content}
                for d in docs
            ]
            assistant_extras(sources, chunks)
            # Citation check: flag section numbers the model used that are not in the context.
            if not not_found:
                missing = rag.check_citations(answer, docs)
                stray = rag.check_numbers(answer, docs)
                if missing or stray:
                    warning = citation_warning(missing, stray)
                    st.warning(warning)
        if label:
            st.caption(label)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "passages": chunks,
            "warning": warning,
            "label": label,
        }
    )
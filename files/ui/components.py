"""Reusable UI pieces: source chips, passage cards, quick actions, chat export."""
import html
 
import streamlit as st
 
# One starter per route, so the router is visible from the first screen.
QUICK_PROMPTS = [
    "Summarize the key points of these documents",
    "What is Section 420 IPC in BNS?",
    "My cheque bounced on 15 Nov 2026 - what are my deadlines?",
    "What are my rights if I am arrested?",
]
 
 
def source_chips(sources) -> None:
    """sources: iterable of (file, page). Pages are grouped per file."""
    by_file: dict[str, set] = {}
    for name, page in sources:
        by_file.setdefault(name, set()).add(page)
    chips = "".join(
        f"<span class='src-chip'>📄 {html.escape(str(name))} · p. "
        f"{', '.join(str(p) for p in sorted(pages))}</span>"
        for name, pages in by_file.items()
    )
    st.markdown(chips, unsafe_allow_html=True)
 
 
def passages(chunks) -> None:
    """chunks: list of dicts with keys source, page, text."""
    with st.expander(f"Retrieved passages ({len(chunks)})"):
        for c in chunks:
            body = html.escape(c["text"]).replace("\n", "<br>")
            st.markdown(
                f"<div class='passage'><div class='passage-head'>"
                f"{html.escape(str(c['source']))} · p. {c['page']}</div>{body}</div>",
                unsafe_allow_html=True,
            )
 
 
def assistant_extras(sources, chunks) -> None:
    if sources:
        source_chips(sources)
    if chunks:
        passages(chunks)
 
 
def quick_actions() -> str | None:
    """Starter questions shown on an empty chat. Returns the clicked prompt."""
    st.markdown("##### Try asking")
    cols = st.columns(2)
    for i, prompt in enumerate(QUICK_PROMPTS):
        if cols[i % 2].button(prompt, key=f"qa_{i}"):
            return prompt
    return None
 
 
def chat_to_markdown(messages) -> str:
    lines = ["# Legal RAG Assistant - chat export", ""]
    for m in messages:
        who = "You" if m["role"] == "user" else "Assistant"
        lines += [f"**{who}:** {m['content']}", ""]
        if m.get("sources"):
            cites = ", ".join(f"{s} (p. {p})" for s, p in m["sources"])
            lines += [f"*Sources: {cites}*", ""]
    return "\n".join(lines)
 

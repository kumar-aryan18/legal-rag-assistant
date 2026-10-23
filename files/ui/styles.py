"""Custom CSS for the Legal RAG UI."""
import streamlit as st

CSS = """
<style>
.block-container { padding-top: 2.2rem; max-width: 1100px; }
h1 { letter-spacing: .3px; }

/* source chips under an answer */
.src-chip {
    display: inline-block; padding: 2px 10px; margin: 2px 6px 2px 0;
    border: 1px solid rgba(201,162,39,.55); border-radius: 999px;
    font-size: .78rem; color: #E8D48B; background: rgba(201,162,39,.08);
}

/* retrieved passage cards */
.passage {
    border-left: 3px solid #C9A227; padding: .55rem .85rem; margin: .55rem 0;
    background: rgba(255,255,255,.03); border-radius: 0 6px 6px 0;
    font-size: .85rem; line-height: 1.45;
}
.passage-head { font-weight: 600; color: #E8D48B; font-size: .8rem; margin-bottom: .3rem; }

/* sidebar document names wrap cleanly instead of breaking mid-word */
.doc-name { font-size: .85rem; overflow-wrap: anywhere; padding-top: .35rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

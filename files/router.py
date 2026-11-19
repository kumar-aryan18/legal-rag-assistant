"""Question router: picks the right tool before falling back to normal RAG.

Order: exact section lookup -> deadline calculator -> document overview -> RAG.
It is rule-based on purpose: instant, predictable, and no extra LLM call on a slow local model.
"""
from dataclasses import dataclass, field

import overview
import tools


@dataclass
class Route:
    kind: str                                   # "tool" | "overview" | "rag"
    text: str = ""                              # ready-made answer when kind == "tool"
    label: str = ""                             # small caption shown under the answer
    docs: list = field(default_factory=list)    # file names when kind == "overview"


def route(question: str, sources=None) -> Route:
    q = question.strip()
    text = tools.lookup_section(q)
    if text:
        return Route("tool", text=text, label="🛠 Section lookup · exact table, no LLM")
    text = tools.mapping_table(q)
    if text:
        return Route("tool", text=text, label="🛠 Section mapping table · exact table, no LLM")
    text = tools.deadline_from_question(q)
    if text:
        return Route("tool", text=text, label="🛠 Deadline calculator · no LLM")
    docs = overview.overview_targets(q, sources)
    if docs:
        return Route("overview", docs=docs, label="📚 Document overview")
    return Route("rag")
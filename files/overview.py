"""Document overviews.

At index time `save_overviews()` stores each PDF's outline (its numbered headings) in
overview.json next to the FAISS index. At question time vague requests such as
"about the rule book" or "summarize these documents" are answered from that outline
instead of from whatever chunk happens to match best (usually the cover page).
"""
import json
import re
from pathlib import Path

from config import INDEX_DIR, chunk_text, get_llm

FILE_NAME = "overview.json"
HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+([A-Z][^\n]{2,90})$")
OVERVIEW_RE = re.compile(
    r"\b(about|overview|summar(?:y|ize|ise)|outline|contents|what (?:does|is in|is inside))\b", re.I)
DOC_NOUN_RE = re.compile(r"\b(documents?|docs?|files?|pdfs?|book|guide)s?\b", re.I)
_NAME_STOP = {"the", "and", "of", "pdf", "guide", "study", "introduction", "information", "group"}
_GENERIC = set(
    "a an the of on in to for me us please tell give show can could you i what whats does do is are "
    "about overview summary summarize summarise outline contents content table key main points point "
    "brief short quick whole entire complete detailed document documents doc docs file files pdf pdfs "
    "book books guide guides these this that those all my our uploaded and".split())


def _path(directory) -> Path:
    return Path(directory) / FILE_NAME


# ---------------------------------------------------------------- build time
def extract_outline(page_texts: list[str]) -> list[str]:
    seen, outline = set(), []
    for text in page_texts:
        for line in text.splitlines():
            line = line.strip()
            m = HEADING_RE.match(line)
            if not m or line.endswith((".", ";", ",")) or len(line.split()) > 14:
                continue
            key = line.lower()
            if key not in seen:
                seen.add(key)
                outline.append(line)
    if len(outline) < 3:  # no numbered headings: fall back to each page's first line
        outline, seen = [], set()
        for text in page_texts:
            first = next((l.strip() for l in text.splitlines() if l.strip()), "")
            if first and first.lower() not in seen:
                seen.add(first.lower())
                outline.append(first[:90])
    return outline[:80]


def save_overviews(pages, directory=INDEX_DIR) -> None:
    """pages: the per-page Documents made by the index builder."""
    by_source: dict[str, dict[int, str]] = {}
    for doc in pages:
        by_source.setdefault(doc.metadata["source"], {})[doc.metadata["page"]] = doc.page_content
    data = {}
    for name, page_map in by_source.items():
        ordered = [page_map[p] for p in sorted(page_map)]
        data[name] = {
            "pages": max(page_map),
            "outline": extract_outline(ordered),
            "intro": ordered[0][:800],
            "summary": None,  # filled in lazily the first time it is asked for
        }
    Path(directory).mkdir(parents=True, exist_ok=True)
    _path(directory).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load_overviews(directory=INDEX_DIR) -> dict:
    try:
        return json.loads(_path(directory).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# ------------------------------------------------------------- question time
def _name_tokens(name: str) -> set[str]:
    stem = Path(name).stem.lower()
    return {t for t in re.split(r"[^a-z0-9]+", stem) if len(t) >= 2 and t not in _NAME_STOP}


def overview_targets(question: str, sources=None, directory=INDEX_DIR) -> list[str]:
    """File names the question wants an overview of, or [] if it is not an overview request."""
    data = load_overviews(directory)
    names = [n for n in data if not sources or n in sources]
    if not names or not OVERVIEW_RE.search(question):
        return []
    words = set(re.findall(r"[a-z0-9]+", question.lower()))
    all_tokens = set().union(*(_name_tokens(n) for n in data))
    if words - _GENERIC - all_tokens:  # leftover topic words ("about arrest ...") -> normal search
        return []
    has_noun = bool(DOC_NOUN_RE.search(question))
    scored = [(len(_name_tokens(n) & words), n) for n in names]
    best = max(s for s, _ in scored)
    if best >= 2 or (best == 1 and has_noun):
        return [n for s, n in scored if s == best]
    if best == 0 and has_noun:  # "summarize these documents"
        return names
    return []


def outline_markdown(name: str, limit: int = 30, directory=INDEX_DIR) -> str:
    items = load_overviews(directory).get(name, {}).get("outline", [])[:limit]
    return "**Contents**\n\n" + "\n".join(f"- {h}" for h in items) if items else ""


def all_docs_markdown(names: list[str], directory=INDEX_DIR) -> str:
    data = load_overviews(directory)
    blocks = []
    for n in names:
        info = data.get(n, {})
        topics = "; ".join(re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", h) for h in info.get("outline", [])[:8])
        blocks.append(f"**{n}** · {info.get('pages', '?')} pages\n\nCovers: {topics or 'no outline found'}")
    return "\n\n".join(blocks) + "\n\n_Ask about one document by name for a fuller overview._"


def stream_overview(name: str, directory=INDEX_DIR):
    """Yields a short LLM summary written from the outline (cached after the first time)."""
    data = load_overviews(directory)
    info = data[name]
    if info.get("summary"):
        yield info["summary"]
        return
    outline = "\n".join(f"- {h}" for h in info["outline"][:60])
    prompt = (
        "Write a 3-4 sentence overview of the document below: what it covers and who it is useful for. "
        "Use only the outline and opening text. Ignore disclaimers and boilerplate. No preamble.\n\n"
        f"Document: {name}\n\nOutline:\n{outline}\n\nOpening text:\n{info['intro']}"
    )
    parts = []
    for chunk in get_llm().stream(prompt):
        text = chunk_text(chunk)   # Gemini may stream a list of blocks; flatten to text
        if text:
            parts.append(text)
            yield text
    info["summary"] = "".join(parts).strip()
    _path(directory).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

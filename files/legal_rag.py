"""Core Legal RAG logic: retrieve relevant passages, then answer with citations.

Upgrades over the first version:
  1. Relevance guard  - weak matches return no passages (app shows "not found").
  2. Hybrid search    - FAISS (meaning) + BM25 (exact words like "Section 138"),
                        merged with Reciprocal Rank Fusion. Falls back to FAISS only
                        if the optional `rank-bm25` package is not installed.
  3. Follow-up aware  - short follow-up questions are searched together with the
                        previous question, so "what is the punishment for that?" works.
  4. Citation check   - flags section/article numbers in the answer that do not
                        appear in the retrieved text.
"""
import re

from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import config
from config import INDEX_DIR, TOP_K, get_embeddings, get_llm

try:  # optional dependency: pip install rank-bm25
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None

# Relevance guard. FAISS returns an L2 *distance* by default (LOWER = more similar).
# If the best match is worse than this, we answer "not found" without calling the LLM.
# Add MAX_DISTANCE = <number> to config.py after tuning (see the [retrieval] lines
# printed in the terminal). None disables the guard.
MAX_DISTANCE = getattr(config, "MAX_DISTANCE", None)

# Keep these in sync with rule 3 of SYSTEM_PROMPT below. The UI uses the phrase
# to detect "no answer found" and hide the (irrelevant) source chips.
NOT_FOUND_MESSAGE = "I could not find this in the provided documents."
NOT_FOUND_PHRASE = "could not find this in the provided documents"

SYSTEM_PROMPT = """You are a Legal Research Assistant. You answer questions using ONLY \
the document excerpts provided in the context below.

Rules:
1. Base every statement on the context. Do not use outside legal knowledge.
2. Cite the supporting excerpt after each claim in the form [file name, p. N].
3. If the context does not contain the answer, say: "I could not find this in the \
provided documents." Do not guess.
4. Quote clause wording exactly when precision matters, and keep quotes short.
5. If the excerpts conflict or are ambiguous, point that out.
6. You provide information from documents, not legal advice. For decisions with legal \
consequences, remind the user to consult a qualified lawyer.
7. Be concise: answer in at most 3-4 sentences (or a short list) unless the user asks for detail. No preamble and no repeating the question.

Context:
{context}"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

_BRACKET_RE = re.compile(r"\[[^\]]*\]")          # [file.pdf, p. 4] markers
_NUM_RE = re.compile(r"(?<![\w.])(\d{2,4})(?![\w])")  # standalone 2-4 digit numbers
_CITE_RE = re.compile(r"\b(?:section|sec\.?|article|art\.?)\s*(\d+[A-Za-z]?)", re.IGNORECASE)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _key(doc) -> tuple:
    return (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:80])


class LegalRAG:
    def __init__(self):
        if not (INDEX_DIR / "index.faiss").exists():
            raise FileNotFoundError(
                "No index found. Put PDFs in ./docs and run: python rag_index_builder.py"
            )
        # The index is created locally by rag_index_builder.py, so loading is safe.
        self.store = FAISS.load_local(
            str(INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True
        )
        self.chain = PROMPT | get_llm()

        # Keyword index built from the same chunks that are in FAISS.
        self._all_docs = list(self.store.docstore._dict.values())
        self._bm25 = None
        if BM25Okapi is not None and self._all_docs:
            self._bm25 = BM25Okapi([_tokens(d.page_content) for d in self._all_docs])

    # ------------------------------------------------------------------ search
    @staticmethod
    def _search_query(question: str, history) -> str:
        """Short follow-ups ("and the punishment?") are searched with the last question."""
        if history and len(question.split()) <= 6:
            return f"{history[-1][0]} {question}"
        return question

    def retrieve(
        self,
        question: str,
        k: int = TOP_K,
        sources: list[str] | None = None,
        history=None,
    ):
        """Top-k passages (hybrid). If `sources` is given, only search those file names.
        Returns [] when nothing is relevant enough."""
        query = self._search_query(question, history)
        allowed = set(sources) if sources else None
        pool = max(k * 3, 12)

        # 1) Semantic search (FAISS) with distances.
        kwargs = {"k": pool, "fetch_k": max(50, pool * 10)}
        if allowed:
            kwargs["filter"] = lambda meta: meta.get("source") in allowed
        vec = self.store.similarity_search_with_score(query, **kwargs)

        best = min((score for _, score in vec), default=None)
        if best is not None:
            print(f"[retrieval] best distance {best:.3f} for: {question[:60]!r}")
        if not vec or (MAX_DISTANCE is not None and best > MAX_DISTANCE):
            return []
        vec_docs = [d for d, _ in vec]

        if self._bm25 is None:
            return vec_docs[:k]

        # 2) Keyword search (BM25) over the same chunks.
        scores = self._bm25.get_scores(_tokens(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        bm25_docs = []
        for i in order:
            if scores[i] <= 0 or len(bm25_docs) >= pool:
                break
            d = self._all_docs[i]
            if allowed and d.metadata.get("source") not in allowed:
                continue
            bm25_docs.append(d)

        # 3) Reciprocal Rank Fusion: documents ranked high in either list rise to the top.
        fused, by_key = {}, {}
        for ranking in (vec_docs, bm25_docs):
            for rank, d in enumerate(ranking):
                key = _key(d)
                by_key[key] = d
                fused[key] = fused.get(key, 0.0) + 1.0 / (60 + rank)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [by_key[key] for key in top]

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _format_context(docs) -> str:
        return "\n\n".join(
            f"[{d.metadata['source']}, p. {d.metadata['page']}]\n{d.page_content}"
            for d in docs
        )

    @staticmethod
    def collect_sources(docs) -> list[tuple[str, int]]:
        return sorted({(d.metadata["source"], d.metadata["page"]) for d in docs})

    @staticmethod
    def check_citations(answer: str, docs) -> list[str]:
        """Section/article numbers mentioned in the answer but absent from the retrieved text."""
        context = " ".join(d.page_content for d in docs)
        missing = []
        for num in dict.fromkeys(_CITE_RE.findall(answer)):
            if not re.search(rf"\b{re.escape(num)}\b", context, re.IGNORECASE):
                missing.append(num)
        return missing

    @staticmethod
    def check_numbers(answer: str, docs, limit: int = 6) -> list[str]:
        """Numbers in the answer (section numbers, table cells, days...) that appear nowhere in
        the retrieved passages. Years are ignored. Catches invented table rows."""
        context_numbers = set(re.findall(r"\d+", " ".join(d.page_content for d in docs)))
        missing = []
        for n in dict.fromkeys(_NUM_RE.findall(_BRACKET_RE.sub("", answer))):
            if re.fullmatch(r"(?:19|20)\d\d", n) or n in context_numbers:
                continue
            missing.append(n)
        return missing[:limit]

    def _prepare(self, question, history, k, sources):
        docs = self.retrieve(question, k, sources, history)
        messages = []
        for user_text, assistant_text in (history or [])[-2:]:
            messages.append(HumanMessage(user_text))
            messages.append(AIMessage(assistant_text))
        inputs = {
            "context": self._format_context(docs),
            "history": messages,
            "question": question,
        }
        return docs, inputs

    # --------------------------------------------------------------------- ask
    def ask(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None,
        k: int = TOP_K,
        sources: list[str] | None = None,
    ):
        """history is a list of (user_text, assistant_text) pairs, oldest first."""
        docs, inputs = self._prepare(question, history, k, sources)
        if not docs:
            return {"answer": NOT_FOUND_MESSAGE, "sources": [], "chunks": []}
        response = self.chain.invoke(inputs)
        return {
            "answer": response.content,
            "sources": self.collect_sources(docs),
            "chunks": docs,
        }

    def ask_stream(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None,
        k: int = TOP_K,
        sources: list[str] | None = None,
    ):
        """Retrieves first, then returns (docs, token_generator) for live streaming."""
        docs, inputs = self._prepare(question, history, k, sources)

        def tokens():
            for chunk in self.chain.stream(inputs):
                if chunk.content:
                    yield chunk.content

        return docs, tokens()
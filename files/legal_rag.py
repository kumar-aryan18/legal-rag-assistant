"""Core Legal RAG logic: retrieve relevant passages, then answer with citations."""
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import INDEX_DIR, TOP_K, get_embeddings, get_llm

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

    def retrieve(self, question: str, k: int = TOP_K, sources: list[str] | None = None):
        """Top-k passages. If `sources` is given, only search those file names."""
        if sources:
            allowed = set(sources)
            return self.store.similarity_search(
                question,
                k=k,
                filter=lambda meta: meta.get("source") in allowed,
                fetch_k=max(50, k * 10),  # look wider so the filter still yields k hits
            )
        return self.store.similarity_search(question, k=k)

    @staticmethod
    def _format_context(docs) -> str:
        return "\n\n".join(
            f"[{d.metadata['source']}, p. {d.metadata['page']}]\n{d.page_content}"
            for d in docs
        )

    @staticmethod
    def collect_sources(docs) -> list[tuple[str, int]]:
        return sorted({(d.metadata["source"], d.metadata["page"]) for d in docs})

    def _prepare(self, question, history, k, sources):
        docs = self.retrieve(question, k, sources)
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

    def ask(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None,
        k: int = TOP_K,
        sources: list[str] | None = None,
    ):
        """history is a list of (user_text, assistant_text) pairs, oldest first."""
        docs, inputs = self._prepare(question, history, k, sources)
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

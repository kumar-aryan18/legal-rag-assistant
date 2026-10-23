"""Terminal chat with the Legal RAG Assistant.  Run: python cli.py"""
from legal_rag import LegalRAG


def main():
    rag = LegalRAG()
    history: list[tuple[str, str]] = []
    print("Legal RAG Assistant ready. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue

        result = rag.ask(question, history)
        history.append((question, result["answer"]))

        print(f"\nAssistant: {result['answer']}\n")
        print("Sources:", ", ".join(f"{s} (p. {p})" for s, p in result["sources"]))
        print()


if __name__ == "__main__":
    main()

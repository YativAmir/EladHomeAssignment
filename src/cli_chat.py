import argparse
from typing import Any, Dict, List

from dotenv import load_dotenv

from src.graph import build_graph


def run_chat(thread_id: str, verbose: bool = False) -> None:
    """
    Simple command-line loop to chat with the LangGraph app.

    Usage (from project root):
        python -m src.cli_chat --thread-id user1
    """

    load_dotenv()
    app = build_graph()

    # In this CLI we keep the messages list in memory and also let the
    # LangGraph checkpointer persist it (per thread_id).
    messages: List[Dict[str, Any]] = []

    print("צ'אט פסיכולוגיה - מצב שורת פקודה")
    print("הקלד/י שאלה ולחץ/י Enter. ליציאה כתוב/כתבי: /exit")
    print("-" * 60)

    while True:
        try:
            user_input = input("את/ה: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nלהתראות 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in {"/exit", "exit", "quit"}:
            print("להתראות 👋")
            break

        # Append user message to local history.
        messages.append({"role": "user", "content": user_input})

        try:
            result = app.invoke(
                {
                    "messages": messages,
                    "retrieved_chunks": [],
                },
                config={"configurable": {"thread_id": thread_id}},
            )
        except Exception as e:
            print("בוט: (שגיאה)", str(e))
            messages.pop()  # remove the message we just added so user can retry
            continue

        # Normalize messages: graph may return dicts (role/content) or message objects (.type/.content)
        def _role(m: Any) -> str:
            if isinstance(m, dict):
                return str(m.get("role", ""))
            return str(getattr(m, "type", getattr(m, "role", "")))

        def _content(m: Any) -> str:
            if isinstance(m, dict):
                c = m.get("content", "")
                return c if isinstance(c, str) else str(c)
            c = getattr(m, "content", "")
            return c if isinstance(c, str) else str(c)

        norm_messages = [{"role": _role(m), "content": _content(m)} for m in result["messages"]]
        messages = norm_messages

        # Print the latest assistant message (LangGraph/LangChain may use "assistant" or "ai")
        last_assistant = next(
            (m for m in reversed(messages) if _role(m) in {"assistant", "ai"}),
            None,
        )
        if last_assistant:
            print("בוט:", last_assistant["content"])
        else:
            print("בוט: (לא התקבלה תשובה)")

        if verbose:
            chunks = result.get("retrieved_chunks") or []
            scores = [round(ch.get("score", 0), 3) for ch in chunks]
            print(f"  [verbose] Retrieved {len(chunks)} chunks, scores: {scores} (threshold=0.4)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Psychology RAG chatbot (CLI)")
    parser.add_argument(
        "--thread-id",
        type=str,
        default="default-cli-session",
        help="Session/thread ID for checkpointing (default: default-cli-session)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print retrieval debug info (chunk count and similarity scores)",
    )
    args = parser.parse_args()

    run_chat(thread_id=args.thread_id, verbose=args.verbose)


if __name__ == "__main__":
    main()


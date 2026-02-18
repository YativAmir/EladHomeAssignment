from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from src.config import get_settings


load_dotenv()
settings = get_settings()


class ChatState(TypedDict):
    """
    Conversation state flowing through the LangGraph.

    - messages: full chat history (trimmed to last 3 turns).
    - retrieved_chunks: list of retrieved knowledge base chunks
      that were used to answer the latest question. This will be useful
      later for the Streamlit UI to present citations / sources.
    """

    messages: List[AnyMessage]
    retrieved_chunks: List[Dict[str, Any]]


@dataclass
class Services:
    """
    Shared service clients (OpenAI, Pinecone, embeddings) for reuse across nodes.
    """

    openai_client: OpenAI = field(default_factory=lambda: OpenAI(api_key=settings.openai_api_key))
    pinecone_client: Pinecone = field(default_factory=lambda: Pinecone(api_key=settings.pinecone_api_key))
    embedding_model: SentenceTransformer = field(
        default_factory=lambda: SentenceTransformer(settings.embedding_model_name, device="cpu")
    )

    def get_index(self):
        if settings.pinecone_index_host:
            return self.pinecone_client.Index(host=settings.pinecone_index_host)
        return self.pinecone_client.Index(settings.pinecone_index_name)


services = Services()


def _latest_user_message(messages: List[AnyMessage]) -> Optional[str]:
    """
    Extract the text of the latest human/user message from the messages list.
    Handles both dicts (role/content) and message objects (.type/.content).
    """

    for msg in reversed(messages):
        if isinstance(msg, dict):
            if msg.get("role") in ("human", "user"):
                content = msg.get("content", "")
                return content if isinstance(content, str) else str(content)
        else:
            if getattr(msg, "type", None) in ("human", "user"):
                return getattr(msg, "content", "")
    return None


def truncate_history(state: ChatState, max_messages: int = 3) -> ChatState:
    """
    Keep only the last `max_messages` messages in the state.

    This ensures that the checkpointer stores only a short, recent history,
    as requested, and also keeps prompts small and efficient.
    """

    messages = state.get("messages", [])
    if len(messages) > max_messages:
        messages = messages[-max_messages:]
    return {
        **state,
        "messages": messages,
    }


def retrieve(state: ChatState) -> ChatState:
    """
    Retrieve relevant chunks from Pinecone based on the latest user message.
    """

    messages = state.get("messages", [])
    query = _latest_user_message(messages)
    if not query:
        return {**state, "retrieved_chunks": []}

    # Embed the query using the same embedding model as indexing (normalized).
    query_emb = services.embedding_model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()

    index = services.get_index()
    search_response = index.query(
        vector=query_emb,
        namespace=settings.pinecone_namespace,
        top_k=settings.dense_top_k,
        include_metadata=True,
        include_values=False,
    )

    retrieved: List[Dict[str, Any]] = []
    for match in search_response.matches or []:
        score = float(getattr(match, "score", 0.0))
        metadata = dict(getattr(match, "metadata", {}) or {})

        # Apply conservative similarity threshold. If all matches are below
        # threshold, we'll handle that in the generation step.
        retrieved.append(
            {
                "id": getattr(match, "id", metadata.get("id")),
                "score": score,
                "metadata": metadata,
            }
        )

    return {
        **state,
        "retrieved_chunks": retrieved,
    }


def _build_context_from_chunks(chunks: List[Dict[str, Any]]) -> str:
    """
    Turn retrieved Pinecone matches into a text context for the LLM.
    """

    if not chunks:
        return ""

    lines: List[str] = []
    for ch in chunks:
        meta = ch.get("metadata", {})
        text = meta.get("text", "")
        source = meta.get("source", "")
        doc_title = meta.get("doc_title", "")
        topic = meta.get("topic", "")
        score = ch.get("score", 0.0)
        header = f"[מסמך: {doc_title} | נושא: {topic} | קובץ: {source} | ציון דמיון: {score:.3f}]"
        lines.append(header)
        lines.append(text)
        lines.append("")  # blank line separator
    return "\n".join(lines).strip()


def generate_answer(state: ChatState) -> ChatState:
    """
    Use the LLM to generate a Hebrew answer based only on retrieved chunks.

    The assistant is:
    - Friendly, accessible, and non-clinical.
    - Focused strictly on psychology / mental health topics.
    - Avoids diagnoses, medical advice, or medication recommendations.
    """

    messages = state.get("messages", [])
    retrieved = state.get("retrieved_chunks", [])

    # Filter by similarity threshold.
    strong_chunks = [ch for ch in retrieved if ch.get("score", 0.0) >= settings.dense_score_threshold]

    if not strong_chunks:
        # No sufficiently similar knowledge found: answer with a safe, domain-limited message.
        assistant_content = (
            "אני לא מוצאת במאגר הידע שלי מידע מספיק רלוונטי לשאלה הזו.\n"
            "הצ'אטבוט הזה עוסק רק בנושאים של פסיכולוגיה ובריאות נפשית, "
            "ואינו נותן אבחנות רפואיות או המלצות טיפוליות אישיות.\n"
            "אם מדובר בשאלה חשובה או דחופה עבורך, מומלץ לפנות לאיש מקצוע מוסמך."
        )
    else:
        context_text = _build_context_from_chunks(strong_chunks)
        user_text = _latest_user_message(messages) or ""

        system_prompt = (
            "את/ה צ'אטבוט תומך, נגיש וידידותי בעברית, שמתמקד בפסיכולוגיה ובריאות נפשית בלבד.\n"
            "יש לך גישה רק למידע שמופיע בקטעי הטקסט הבאים ממסמכים מקצועיים, "
            "ואסור לך להמציא מידע שאיננו מופיע בהם.\n"
            "חשוב:\n"
            "- אל תספק/י אבחנות רפואיות או נפשיות.\n"
            "- אל תמליצ/י על תרופות, מינונים או שינויים בטיפול תרופתי.\n"
            "- אם השאלה אינה בתחום הפסיכולוגיה/בריאות נפשית, או אם אין מספיק מידע רלוונטי, "
            "הסבר/י זאת בעדינות.\n"
            "ענה/י תמיד בשפה ברורה, בגובה העיניים, וללא שימוש בז'רגון מקצועי מורכב.\n"
            "התבסס/י רק על המידע הבא:\n\n"
            f"{context_text}"
        )

        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"השאלה של המשתמש/ת:\n{user_text}",
            },
        ]

        completion = services.openai_client.chat.completions.create(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
            messages=prompt_messages,
        )

        assistant_content = completion.choices[0].message.content or ""

    # Append the assistant message to the conversation history, then truncate.
    new_state: ChatState = {
        **state,
        "messages": add_messages(
            state.get("messages", []),
            [{"role": "assistant", "content": assistant_content}],
        ),
        "retrieved_chunks": retrieved,
    }
    return truncate_history(new_state, max_messages=3)


def build_graph(checkpoint_path: str | None = None):
    """
    Construct and return a compiled LangGraph app.

    The graph has two main steps:
    1) retrieve  -> query Pinecone using the latest user question.
    2) generate  -> call OpenAI to produce a Hebrew answer.
    """

    graph = StateGraph(ChatState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate_answer)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")

    # Checkpointer for session-based conversations.
    checkpointer = None
    if checkpoint_path is None:
        checkpoint_path = str(settings.base_dir / "data" / "checkpoints.sqlite")

    # We try to use the SQLite checkpointer if the extra package / module is
    # available. If not, we fall back to an in-memory graph so the rest of
    # the system (and CLI) can still run without delaying you.
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore

        checkpointer = SqliteSaver.from_uri(f"sqlite:///{checkpoint_path}")
    except Exception:
        checkpointer = None

    if checkpointer is not None:
        app = graph.compile(checkpointer=checkpointer)
    else:
        app = graph.compile()
    return app


__all__ = [
    "ChatState",
    "build_graph",
]


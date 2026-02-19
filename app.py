"""
Psychology RAG Chatbot - Streamlit UI

Run from project root:
    streamlit run app.py
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

from src.graph import build_graph

load_dotenv()

MAX_MESSAGES_DISPLAY = 10  # Cap chat history to last N messages


# --- Session state helpers ---
def _init_session():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = []
    if "last_chunks" not in st.session_state:
        st.session_state.last_chunks: List[Dict[str, Any]] = []
    if "last_standalone_query" not in st.session_state:
        st.session_state.last_standalone_query: str = ""
    if "use_hyde" not in st.session_state:
        st.session_state.use_hyde = False
    if "last_hypothetical_document" not in st.session_state:
        st.session_state.last_hypothetical_document = ""


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


def _new_conversation():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.last_chunks = []
    st.session_state.last_standalone_query = ""
    st.session_state.last_hypothetical_document = ""


# --- RTL / Hebrew styling ---
RTL_CSS = """
<style>
    /* 0. Root: RTL for entire page */
    html, body {
        direction: rtl !important;
        text-align: right !important;
    }

    /* 1. Streamlit main content wrapper */
    main[data-testid="stAppViewContainer"],
    main[data-testid="stAppViewContainer"] > div,
    main[data-testid="stAppViewContainer"] > div > div,
    section[data-testid="stSidebar"] + div,
    .block-container {
        direction: rtl !important;
        text-align: right !important;
    }

    /* 3. Block container, vertical blocks */
    [data-testid="stAppViewBlockContainer"],
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    div[data-testid="stVerticalBlock"],
    div[data-testid="stVerticalBlock"] > div {
        direction: rtl !important;
        text-align: right !important;
    }

    /* 4. Elements: titles, captions, expanders, alerts */
    [data-testid="stAppViewBlockContainer"] h1,
    [data-testid="stAppViewBlockContainer"] h2,
    [data-testid="stAppViewBlockContainer"] h3,
    [data-testid="stAppViewBlockContainer"] p,
    [data-testid="stExpander"],
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] > div,
    [data-testid="stCaptionContainer"],
    .stCaption,
    [data-testid="stAlert"],
    .stAlert {
        direction: rtl !important;
        text-align: right !important;
    }

    /* 5. Chat messages: avatar on right, content RTL (flex row-reverse) */
    [data-testid="stChatMessage"] {
        display: flex !important;
        flex-direction: row-reverse !important;
        direction: rtl !important;
        text-align: right !important;
    }
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"] .stMarkdown {
        text-align: right !important;
        direction: rtl !important;
    }

    /* 6. Markdown: lists, paragraphs, RTL padding */
    [data-testid="stChatMessage"] .stMarkdown p,
    [data-testid="stChatMessage"] .stMarkdown li {
        text-align: right !important;
        direction: rtl !important;
    }
    [data-testid="stChatMessage"] .stMarkdown ul,
    [data-testid="stChatMessage"] .stMarkdown ol {
        padding-right: 1.5em !important;
        padding-left: 0 !important;
        margin-right: 0 !important;
        text-align: right !important;
        direction: rtl !important;
    }

    /* 7. Chat input: container + textarea RTL */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    .stChatInputContainer,
    .stChatInputContainer > div {
        direction: rtl !important;
        text-align: right !important;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        direction: rtl !important;
        text-align: right !important;
    }

    /* 8. Source cards (preserved) */
    .source-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        text-align: right !important;
        direction: rtl !important;
    }
    .source-header {
        font-weight: 600;
        color: #495057;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    .source-meta {
        font-size: 0.8rem;
        color: #6c757d;
        margin-bottom: 0.4rem;
    }
    .source-text {
        font-size: 0.9rem;
        line-height: 1.5;
        color: #212529;
    }
    .source-score {
        display: inline-block;
        background: #0d6efd;
        color: white;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
    }
</style>
"""


def _render_source_card(chunk: Dict[str, Any], idx: int) -> None:
    meta = chunk.get("metadata", {}) or {}
    text = meta.get("text", "(ללא טקסט)")
    doc_title = meta.get("doc_title", "")
    source = meta.get("source", "")
    topic = meta.get("topic", "")
    score = chunk.get("score", 0.0)
    score_pct = f"{score * 100:.0f}"

    with st.container():
        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-header">מקור {idx + 1}</div>
                <div class="source-meta">
                    {f'<strong>{doc_title}</strong>' if doc_title else ''}
                    {f' | נושא: {topic}' if topic else ''}
                    {f' | {source}' if source else ''}
                    <span class="source-score">ציון: {score_pct}</span>
                </div>
                <div class="source-text">{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="צ'אטבוט פסיכולוגיה",
        page_icon="🧠",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.markdown(RTL_CSS, unsafe_allow_html=True)
    _init_session()

    # Header
    st.title("🧠 צ'אטבוט פסיכולוגיה ובריאות נפשית")
    st.caption("שאל/י שאלה בתחום הפסיכולוגיה – התשובות מבוססות על מאגר הידע הפנימי בלבד.")

    # Sidebar: New conversation
    with st.sidebar:
        st.markdown("### הגדרות")
        st.session_state.use_hyde = st.toggle(
            "הפעל חיפוש מתקדם (HyDE)",
            value=st.session_state.use_hyde,
        )
        if st.button("🆕 שיחה חדשה", use_container_width=True):
            _new_conversation()
            st.rerun()

    # Chat history (show last N messages)
    for m in st.session_state.messages[-MAX_MESSAGES_DISPLAY:]:
        role = _role(m)
        content = _content(m)
        if role in ("user", "human"):
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        elif role in ("assistant", "ai"):
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)

    # Rephrased question (when different from user's raw message)
    sq = st.session_state.last_standalone_query
    last_user = next(
        (m for m in reversed(st.session_state.messages) if _role(m) in ("user", "human")),
        None,
    )
    last_user_text = _content(last_user) if last_user else ""
    if sq and sq.strip() != last_user_text.strip():
        st.caption(f"✏️ שאלה שעובדה לחיפוש: **{sq}**")

    # Hypothetical document (HyDE) from last response
    if st.session_state.last_hypothetical_document:
        with st.expander(
            "📄 מסמך היפותטי (HyDE) שנוצר עבור החיפוש",
            expanded=False,
        ):
            st.write(st.session_state.last_hypothetical_document)

    # Sources from last response (show after assistant message)
    chunks = st.session_state.last_chunks
    if chunks:
        with st.expander(f"📚 מקורות לתשובה האחרונה ({len(chunks)} קטעים)", expanded=False):
            for i, ch in enumerate(chunks):
                _render_source_card(ch, i)

    # Footer disclaimer
    st.divider()
    st.caption(
        "⚠️ זהו כלי מידע בלבד. אינו מהווה ייעוץ מקצועי, אבחון או המלצה טיפולית. "
        "לשאלות דחופות או אישיות – פנה/י לאיש מקצוע מוסמך."
    )

    # Chat input
    if prompt := st.chat_input("כתוב/כתבי את השאלה שלך..."):
        _init_session()
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            stream_placeholder = st.empty()
            stream_placeholder.markdown("מחפש מידע ומכין תשובה...")
            try:
                app = build_graph()
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                inputs = {
                    "messages": st.session_state.messages,
                    "retrieved_chunks": [],
                    "use_hyde": st.session_state.use_hyde,
                }
                full_content = ""
                last_state = None
                for event in app.stream(
                    inputs,
                    config=config,
                    stream_mode=["messages", "values"],
                ):
                    if event[0] == "messages":
                        msg_chunk, meta = event[1]
                        node_name = meta.get("langgraph_node", "")
                        content = getattr(msg_chunk, "content", None) or ""
                        if node_name == "generate" and content:
                            if not full_content:
                                stream_placeholder.empty()
                            full_content += content
                            stream_placeholder.markdown(full_content)
                    elif event[0] == "values":
                        last_state = event[1]

                if last_state is None:
                    st.error("שגיאה: לא התקבלה תוצאת גרף.")
                    st.stop()

                # Append only the new assistant message; do NOT replace with graph's
                # truncated history (graph keeps last 3 msgs, which drops older user Qs)
                last_assistant = next(
                    (
                        m
                        for m in reversed(last_state.get("messages", []))
                        if _role(m) in {"assistant", "ai"}
                    ),
                    None,
                )
                if last_assistant:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": _content(last_assistant)}
                    )
                if len(st.session_state.messages) > MAX_MESSAGES_DISPLAY:
                    st.session_state.messages = st.session_state.messages[-MAX_MESSAGES_DISPLAY:]
                st.session_state.last_chunks = last_state.get("retrieved_chunks") or []
                st.session_state.last_standalone_query = last_state.get("standalone_query") or ""
                st.session_state.last_hypothetical_document = last_state.get(
                    "hypothetical_document", ""
                )

                if not full_content and last_assistant:
                    stream_placeholder.markdown(_content(last_assistant))
                elif not full_content:
                    stream_placeholder.info("לא התקבלה תשובה.")
            except Exception as e:
                stream_placeholder.empty()
                st.error(f"שגיאה: {str(e)}")
                st.stop()

        st.rerun()


if __name__ == "__main__":
    main()

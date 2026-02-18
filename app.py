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


# --- Session state helpers ---
def _init_session():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = []
    if "last_chunks" not in st.session_state:
        st.session_state.last_chunks: List[Dict[str, Any]] = []


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


# --- RTL / Hebrew styling ---
RTL_CSS = """
<style>
    /* RTL layout for main content */
    [data-testid="stAppViewContainer"] {
        direction: rtl;
    }
    [data-testid="stChatMessage"] {
        direction: rtl;
        text-align: right;
    }
    .stChatMessage .stMarkdown {
        text-align: right;
    }
    /* Input area */
    [data-testid="stChatInput"] textarea {
        direction: rtl;
        text-align: right;
    }
    /* Source cards */
    .source-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        text-align: right;
        direction: rtl;
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
    score_pct = f"{score * 100:.0f}%"

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
        if st.button("🆕 שיחה חדשה", use_container_width=True):
            _new_conversation()
            st.rerun()

    # Chat history
    for m in st.session_state.messages:
        role = _role(m)
        content = _content(m)
        if role in ("user", "human"):
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        elif role in ("assistant", "ai"):
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)

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
            with st.spinner("מחפש מידע ומכין תשובה..."):
                try:
                    app = build_graph()
                    result = app.invoke(
                        {
                            "messages": st.session_state.messages,
                            "retrieved_chunks": [],
                        },
                        config={"configurable": {"thread_id": st.session_state.thread_id}},
                    )
                except Exception as e:
                    st.error(f"שגיאה: {str(e)}")
                    st.stop()

            # Normalize messages
            norm = [
                {"role": _role(m), "content": _content(m)}
                for m in result.get("messages", [])
            ]
            st.session_state.messages = norm
            st.session_state.last_chunks = result.get("retrieved_chunks") or []

            last = next(
                (m for m in reversed(norm) if _role(m) in {"assistant", "ai"}),
                None,
            )
            if last:
                st.markdown(last["content"])
            else:
                st.info("לא התקבלה תשובה.")

        st.rerun()


if __name__ == "__main__":
    main()

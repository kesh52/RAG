"""
Conversational RAG Chat Application
====================================
A dedicated ChatGPT-style chat interface for end users and engineers.
Launch with:
    streamlit run chat_app.py
"""

import os
import sys
import time
import re
import streamlit as st

# Ensure project root is on import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.db import chat_store
from src.pipeline import get_default_pipeline
from src.api.memory import ConversationalMemoryManager
from src.utils.config import config

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Internal Technical Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished Chat UI
st.markdown(
    """
    <style>
    /* Main chat container padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 950px;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }
    /* Citation badges */
    .source-pill {
        display: inline-block;
        background: rgba(66, 133, 244, 0.12);
        color: #1a73e8;
        border: 1px solid rgba(66, 133, 244, 0.3);
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.8rem;
        margin: 2px 4px 2px 0;
        text-decoration: none;
    }
    /* Attachment card */
    .attachment-chip {
        display: inline-flex;
        align-items: center;
        background: rgba(128, 128, 128, 0.1);
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 0.85rem;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar: User & Chat History
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("💬 RAG Assistant")

    # User Profile
    user_email = st.text_input("User Email", value="john.doe@mail.com", key="chat_user_email")
    user = chat_store.get_or_create_user(user_email)

    # New Chat Button
    if st.button("➕ **New Chat**", type="primary", use_container_width=True, key="btn_new_chat_sidebar"):
        new_session = chat_store.create_chat_session(user["id"], title="New Conversation")
        st.session_state["active_session_id"] = str(new_session["id"])
        st.rerun()

    st.markdown("---")
    st.subheader("Recent Conversations")

    # Fetch User Sessions
    sessions, _ = chat_store.get_user_chat_sessions(user_email, limit=30)

    # Initialize Active Session
    if "active_session_id" not in st.session_state or not any(str(s["id"]) == st.session_state["active_session_id"] for s in sessions):
        if sessions:
            st.session_state["active_session_id"] = str(sessions[0]["id"])
        else:
            new_session = chat_store.create_chat_session(user["id"], title="New Conversation")
            st.session_state["active_session_id"] = str(new_session["id"])
            st.rerun()

    active_id = st.session_state["active_session_id"]

    # Render Sessions List in Sidebar
    for s in sessions:
        sid_str = str(s["id"])
        is_active = (sid_str == active_id)
        btn_label = f"{'💬 ' if not is_active else '👉 '} {s['title'][:26]}"
        if st.button(
            btn_label,
            key=f"sess_btn_{sid_str}",
            use_container_width=True,
            type="secondary" if not is_active else "primary",
        ):
            st.session_state["active_session_id"] = sid_str
            st.rerun()

    st.markdown("---")
    with st.expander("⚙️ Pipeline Settings", expanded=False):
        use_hybrid = st.toggle("Hybrid Search (RRF)", value=True, key="sb_hybrid")
        use_reranker = st.toggle("Vertex Semantic Reranker", value=True, key="sb_reranker")
        model_name = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0, key="sb_model")


# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------
active_session = chat_store.get_chat_session(st.session_state["active_session_id"])

if not active_session:
    st.warning("Select or create a conversation from the sidebar.")
    st.stop()

# Header Toolbar
col_title, col_actions = st.columns([3, 1])
with col_title:
    st.header(f"💬 {active_session['title']}")
with col_actions:
    col_exp, col_del = st.columns(2)
    with col_exp:
        try:
            exp_data = chat_store.export_chat_session(active_session["id"], export_format="markdown")
            st.download_button(
                label="📥 Export",
                data=exp_data["content"],
                file_name=exp_data["filename"],
                mime="text/markdown",
                help="Download conversation as Markdown",
            )
        except Exception:
            pass
    with col_del:
        if st.button("🗑️", help="Archive chat", key="btn_del_chat"):
            chat_store.delete_chat_session(active_session["id"], hard_delete=False)
            st.session_state.pop("active_session_id", None)
            st.rerun()

# ---------------------------------------------------------------------------
# Render Message Stream
# ---------------------------------------------------------------------------
messages = chat_store.get_session_messages(active_session["id"])

if not messages:
    st.info("👋 **Welcome to your internal technical assistant!** Ask any question about standard operating procedures, incident remediation, or architecture.")

for msg in messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        # Render Attachment Chip if present
        if msg.get("attachment"):
            att = msg["attachment"]
            st.markdown(
                f"<div class='attachment-chip'>📎 <strong>Attached:</strong> {att['filename']} ({att['mime_type']}, {att['file_size_bytes']} bytes)</div>",
                unsafe_allow_html=True,
            )

        # Render Main Content
        st.markdown(msg["content"])

        # Render Citations & Feedback for Assistant Messages
        if msg["role"] == "assistant":
            # Sources Pills
            if msg.get("sources"):
                sources_html = "".join([f"<span class='source-pill'>📚 {src}</span>" for src in msg["sources"]])
                st.markdown(f"<div style='margin-top: 8px;'>{sources_html}</div>", unsafe_allow_html=True)

            # Debug Information Expander
            with st.expander("🛠️ Context & Retrieval Debug Details", expanded=False):
                if msg.get("condensed_query"):
                    st.caption(f"**Contextualized Query:** `{msg['condensed_query']}`")
                st.caption(f"⏱️ Latency: **{msg.get('latency_ms', 0)} ms** | 🤖 Model: `{msg.get('generation_model') or 'gemini-2.5-flash'}`")
                if msg.get("documentation_gaps"):
                    st.warning(f"⚠️ **Identified Documentation Gap:**\n\n{msg['documentation_gaps']}")
                if msg.get("retrieved_contexts"):
                    st.caption(f"**Retrieved Chunks ({len(msg['retrieved_contexts'])}):**")
                    for idx, chunk in enumerate(msg["retrieved_contexts"], 1):
                        st.code(chunk, language="text")

            # Feedback Thumbs Buttons
            fb = msg.get("feedback")
            col_u, col_d, col_st = st.columns([0.6, 0.6, 6])
            with col_u:
                if st.button("👍", key=f"app_up_{msg['id']}", help="Accurate & Grounded"):
                    chat_store.save_message_feedback(msg["id"], rating=1)
                    st.rerun()
            with col_d:
                if st.button("👎", key=f"app_down_{msg['id']}", help="Hallucination / Incomplete"):
                    chat_store.save_message_feedback(msg["id"], rating=-1)
                    st.rerun()
            with col_st:
                if fb:
                    st.caption(f"Feedback: {'👍 Positive' if fb['rating'] == 1 else '👎 Negative'}")

# ---------------------------------------------------------------------------
# Chat Input & File Attachment
# ---------------------------------------------------------------------------
with st.expander("📎 Attach File (Optional: PDF, Image, Log, Code, Text)", expanded=False):
    uploaded_file = st.file_uploader(
        "Upload document to include with your prompt",
        type=["pdf", "txt", "log", "json", "md", "png", "jpg", "jpeg"],
        key="app_file_uploader",
    )
    if uploaded_file:
        st.success(f"📎 Ready: `{uploaded_file.name}` ({uploaded_file.size / 1024:.1f} KB)")

user_prompt = st.chat_input("Ask a question, follow-up, or explain an error...")

if user_prompt:
    # 1. Read Attachment
    att_bytes = None
    att_filename = None
    att_mime = None
    if uploaded_file is not None:
        att_bytes = uploaded_file.getvalue()
        att_filename = uploaded_file.name
        att_mime = uploaded_file.type

    # 2. Record User Message in Database
    user_msg = chat_store.add_chat_message(
        session_id=active_session["id"],
        role="user",
        content=user_prompt,
    )

    if att_bytes and att_filename:
        chat_store.add_chat_attachment(
            message_id=user_msg["id"],
            filename=att_filename,
            mime_type=att_mime or "application/octet-stream",
            file_size_bytes=len(att_bytes),
            file_data=att_bytes,
        )

    # 3. Multi-Turn RAG Execution
    with st.spinner("🧠 Thinking & searching knowledge base..."):
        pipeline = get_default_pipeline()
        mem_mgr = ConversationalMemoryManager(
            genai_client=pipeline.generator_client,
            model_name=pipeline.generator_model,
        )

        prior_history = [m for m in messages if str(m["id"]) != str(user_msg["id"])]
        condensed_query = mem_mgr.condense_query(prior_history, user_prompt)
        search_query = condensed_query

        if att_bytes and att_mime and att_mime.startswith("text/"):
            try:
                snip = att_bytes.decode("utf-8", errors="ignore")[:600].strip()
                if snip:
                    search_query = f"{condensed_query}\n[Report Snippet: {snip}]"
            except Exception:
                pass

        t0 = time.time()
        q_vec = pipeline.embedding_service.get_dense_embedding(search_query)
        p_hybrid = st.session_state.get("sb_hybrid", True)
        p_rerank = st.session_state.get("sb_reranker", True)

        if p_hybrid:
            candidates = pipeline.retriever.hybrid_search_rrf(search_query, q_vec, limit=5)
        else:
            candidates = pipeline.retriever.vector_search(q_vec, limit=5)

        if p_rerank:
            retrieved_docs = pipeline.reranker.rank_candidates(search_query, candidates, top_n=2)
        else:
            retrieved_docs = candidates[:2]

        sources = []
        for doc in retrieved_docs:
            meta = doc.get("metadata") or {}
            url = meta.get("image_url") or meta.get("pdf_url") or meta.get("source_url")
            if url and url not in sources:
                sources.append(url)

        prompt_contents = mem_mgr.format_generation_contents(
            history=prior_history,
            current_query=user_prompt,
            retrieved_contexts=retrieved_docs,
            attached_file_bytes=att_bytes,
            attached_mime_type=att_mime,
        )

        gen_model = st.session_state.get("sb_model", "gemini-2.5-flash")
        gen_res = pipeline.generator_client.models.generate_content(
            model=gen_model,
            contents=prompt_contents,
        )
        raw_text = gen_res.text.strip()
        latency_ms = int((time.time() - t0) * 1000)

        # Parse documentation gaps
        doc_gaps = None
        gap_match = re.search(r"<!--\s*DOCUMENTATION_GAPS\s*-->([\s\S]*?)<!--\s*END_DOCUMENTATION_GAPS\s*-->", raw_text, re.IGNORECASE)
        if gap_match:
            gap_str = gap_match.group(1).strip()
            raw_text = re.sub(r"<!--\s*DOCUMENTATION_GAPS\s*-->([\s\S]*?)<!--\s*END_DOCUMENTATION_GAPS\s*-->", "", raw_text, flags=re.IGNORECASE).strip()
            if gap_str and not gap_str.lower().startswith("none"):
                doc_gaps = gap_str

        if sources and "Sources:" not in raw_text:
            raw_text += "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)

        # 4. Save Assistant Message in Database
        chat_store.add_chat_message(
            session_id=active_session["id"],
            role="assistant",
            content=raw_text,
            condensed_query=condensed_query,
            retrieved_contexts=[d["content"] for d in retrieved_docs],
            sources=sources,
            documentation_gaps=doc_gaps,
            generation_model=gen_model,
            latency_ms=latency_ms,
        )

        # Auto-title if first message
        if len(prior_history) == 0:
            try:
                new_title = mem_mgr.generate_chat_title(user_prompt, raw_text)
                chat_store.update_chat_session(active_session["id"], title=new_title)
            except Exception:
                pass

    st.rerun()


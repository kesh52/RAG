"""PostgreSQL persistence layer for Chat UI (users, sessions, messages, attachments, feedback)."""

import uuid
import logging
from contextlib import closing
from datetime import datetime
from typing import Any
import psycopg
from psycopg.rows import dict_row

from src.db import get_connection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Users Operations
# ---------------------------------------------------------------------------
def get_or_create_user(email: str, full_name: str | None = None) -> dict[str, Any]:
    """Retrieve an existing user by email or create a new one."""
    email_clean = email.strip().lower()
    with closing(get_connection()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO users (email, full_name, last_active_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (email) 
                DO UPDATE SET 
                    full_name = COALESCE(EXCLUDED.full_name, users.full_name),
                    last_active_at = NOW()
                RETURNING id, email, full_name, created_at, last_active_at;
                """,
                (email_clean, full_name),
            )
            user = cur.fetchone()
            conn.commit()
            return user


def get_user_by_email(email: str) -> dict[str, Any] | None:
    """Retrieve a user record by email."""
    with closing(get_connection()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT id, email, full_name, created_at, last_active_at FROM users WHERE email = %s;",
                (email.strip().lower(),),
            )
            return cur.fetchone()


# ---------------------------------------------------------------------------
# 2. Chat Sessions Operations
# ---------------------------------------------------------------------------
def create_chat_session(
    user_id: str | uuid.UUID,
    title: str = "New Conversation",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a new chat session for a user."""
    if metadata is None:
        metadata = {}

    with closing(get_connection()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO chat_sessions (user_id, title, metadata)
                VALUES (%s, %s, %s)
                RETURNING id, user_id, title, is_archived, metadata, created_at, updated_at;
                """,
                (str(user_id), title, psycopg.types.json.Jsonb(metadata)),
            )
            session = cur.fetchone()
            conn.commit()
            return session


def get_chat_session(session_id: str | uuid.UUID) -> dict[str, Any] | None:
    """Get a chat session by ID."""
    with closing(get_connection()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT s.id, s.user_id, u.email as user_email, s.title, s.is_archived, 
                       s.metadata, s.created_at, s.updated_at
                FROM chat_sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.id = %s;
                """,
                (str(session_id),),
            )
            return cur.fetchone()


def get_user_chat_sessions(
    user_email: str,
    limit: int = 50,
    offset: int = 0,
    include_archived: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Get all chat sessions for a user with preview snippet and message count."""
    archived_filter = "" if include_archived else "AND s.is_archived = FALSE"

    with closing(get_connection()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            # Count total
            cur.execute(
                f"""
                SELECT COUNT(*) as total
                FROM chat_sessions s
                JOIN users u ON s.user_id = u.id
                WHERE u.email = %s {archived_filter};
                """,
                (user_email.strip().lower(),),
            )
            total_count = cur.fetchone()["total"]

            # Query list with latest message snippet and count
            cur.execute(
                f"""
                SELECT 
                    s.id,
                    s.title,
                    s.is_archived,
                    s.metadata,
                    s.created_at,
                    s.updated_at,
                    COUNT(m.id) AS message_count,
                    (
                        SELECT m2.content 
                        FROM chat_messages m2 
                        WHERE m2.session_id = s.id 
                        ORDER BY m2.created_at DESC 
                        LIMIT 1
                    ) AS last_message_preview
                FROM chat_sessions s
                JOIN users u ON s.user_id = u.id
                LEFT JOIN chat_messages m ON s.id = m.session_id
                WHERE u.email = %s {archived_filter}
                GROUP BY s.id
                ORDER BY s.updated_at DESC
                LIMIT %s OFFSET %s;
                """,
                (user_email.strip().lower(), limit, offset),
            )
            sessions = cur.fetchall()
            return sessions, total_count


def update_chat_session(
    session_id: str | uuid.UUID,
    title: str | None = None,
    is_archived: bool | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Update title, archived flag, or metadata of a chat session."""
    updates = []
    params = []

    if title is not None:
        updates.append("title = %s")
        params.append(title)
    if is_archived is not None:
        updates.append("is_archived = %s")
        params.append(is_archived)
    if metadata is not None:
        updates.append("metadata = %s")
        params.append(psycopg.types.json.Jsonb(metadata))

    if not updates:
        return get_chat_session(session_id)

    updates.append("updated_at = NOW()")
    params.append(str(session_id))

    sql = f"""
        UPDATE chat_sessions
        SET {', '.join(updates)}
        WHERE id = %s
        RETURNING id, user_id, title, is_archived, metadata, created_at, updated_at;
    """

    with closing(get_connection()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, tuple(params))
            updated = cur.fetchone()
            conn.commit()
            return updated


def delete_chat_session(session_id: str | uuid.UUID, hard_delete: bool = False) -> bool:
    """Delete or soft-delete a chat session."""
    with closing(get_connection()) as conn:
        with conn.cursor() as cur:
            if hard_delete:
                cur.execute("DELETE FROM chat_sessions WHERE id = %s;", (str(session_id),))
            else:
                cur.execute(
                    "UPDATE chat_sessions SET is_archived = TRUE, updated_at = NOW() WHERE id = %s;",
                    (str(session_id),),
                )
            deleted = cur.rowcount > 0
            conn.commit()
            return deleted


# ---------------------------------------------------------------------------
# 3. Chat Messages & Attachments
# ---------------------------------------------------------------------------
def add_chat_message(
    session_id: str | uuid.UUID,
    role: str,
    content: str,
    condensed_query: str | None = None,
    retrieved_contexts: list[str] | None = None,
    sources: list[str] | None = None,
    documentation_gaps: str | None = None,
    generation_model: str | None = None,
    latency_ms: int | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> dict[str, Any]:
    """Insert a new chat message and update the session updated_at timestamp."""
    if retrieved_contexts is None:
        retrieved_contexts = []
    if sources is None:
        sources = []

    with closing(get_connection()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO chat_messages (
                    session_id, role, content, condensed_query, 
                    retrieved_contexts, sources, documentation_gaps, 
                    generation_model, latency_ms, prompt_tokens, completion_tokens
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s
                )
                RETURNING id, session_id, role, content, condensed_query, 
                          retrieved_contexts, sources, documentation_gaps, 
                          generation_model, latency_ms, prompt_tokens, 
                          completion_tokens, created_at;
                """,
                (
                    str(session_id),
                    role,
                    content,
                    condensed_query,
                    psycopg.types.json.Jsonb(retrieved_contexts),
                    psycopg.types.json.Jsonb(sources),
                    documentation_gaps,
                    generation_model,
                    latency_ms,
                    prompt_tokens,
                    completion_tokens,
                ),
            )
            msg = cur.fetchone()

            # Touch session timestamp
            cur.execute(
                "UPDATE chat_sessions SET updated_at = NOW() WHERE id = %s;",
                (str(session_id),),
            )
            conn.commit()
            return msg


def add_chat_attachment(
    message_id: str | uuid.UUID,
    filename: str,
    mime_type: str,
    file_size_bytes: int,
    file_data: bytes | None = None,
    storage_path: str | None = None,
) -> dict[str, Any]:
    """Store an attachment associated with a message."""
    with closing(get_connection()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO chat_attachments (
                    message_id, filename, mime_type, file_size_bytes, file_data, storage_path
                ) VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, message_id, filename, mime_type, file_size_bytes, storage_path, created_at;
                """,
                (str(message_id), filename, mime_type, file_size_bytes, file_data, storage_path),
            )
            attachment = cur.fetchone()
            conn.commit()
            return attachment


def get_attachment(attachment_id: str | uuid.UUID) -> dict[str, Any] | None:
    """Retrieve an attachment including its binary data."""
    with closing(get_connection()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, message_id, filename, mime_type, file_size_bytes, file_data, storage_path, created_at
                FROM chat_attachments
                WHERE id = %s;
                """,
                (str(attachment_id),),
            )
            return cur.fetchone()


def get_session_messages(session_id: str | uuid.UUID, limit: int = 100) -> list[dict[str, Any]]:
    """Retrieve full history of messages for a session including attachments and feedback."""
    with closing(get_connection()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT 
                    m.id,
                    m.session_id,
                    m.role,
                    m.content,
                    m.condensed_query,
                    m.retrieved_contexts,
                    m.sources,
                    m.documentation_gaps,
                    m.generation_model,
                    m.latency_ms,
                    m.prompt_tokens,
                    m.completion_tokens,
                    m.created_at,
                    -- Attachment details
                    att.id as attachment_id,
                    att.filename as attachment_filename,
                    att.mime_type as attachment_mime_type,
                    att.file_size_bytes as attachment_file_size,
                    -- Feedback details
                    fb.id as feedback_id,
                    fb.rating as feedback_rating,
                    fb.issue_tags as feedback_issue_tags,
                    fb.user_comment as feedback_user_comment,
                    fb.corrected_reference as feedback_corrected_reference
                FROM chat_messages m
                LEFT JOIN chat_attachments att ON m.id = att.message_id
                LEFT JOIN message_feedback fb ON m.id = fb.message_id
                WHERE m.session_id = %s
                ORDER BY m.created_at ASC
                LIMIT %s;
                """,
                (str(session_id), limit),
            )
            rows = cur.fetchall()

            messages = []
            for r in rows:
                attachment = None
                if r["attachment_id"]:
                    attachment = {
                        "id": r["attachment_id"],
                        "filename": r["attachment_filename"],
                        "mime_type": r["attachment_mime_type"],
                        "file_size_bytes": r["attachment_file_size"],
                    }

                feedback = None
                if r["feedback_id"]:
                    feedback = {
                        "id": r["feedback_id"],
                        "rating": r["feedback_rating"],
                        "issue_tags": r["feedback_issue_tags"] or [],
                        "user_comment": r["feedback_user_comment"],
                        "corrected_reference": r["feedback_corrected_reference"],
                    }

                messages.append({
                    "id": r["id"],
                    "session_id": r["session_id"],
                    "role": r["role"],
                    "content": r["content"],
                    "condensed_query": r["condensed_query"],
                    "retrieved_contexts": r["retrieved_contexts"] or [],
                    "sources": r["sources"] or [],
                    "documentation_gaps": r["documentation_gaps"],
                    "generation_model": r["generation_model"],
                    "latency_ms": r["latency_ms"],
                    "prompt_tokens": r["prompt_tokens"],
                    "completion_tokens": r["completion_tokens"],
                    "created_at": r["created_at"],
                    "attachment": attachment,
                    "feedback": feedback,
                })
            return messages


# ---------------------------------------------------------------------------
# 4. Message Feedback Operations
# ---------------------------------------------------------------------------
def save_message_feedback(
    message_id: str | uuid.UUID,
    rating: int,
    issue_tags: list[str] | None = None,
    user_comment: str | None = None,
    corrected_reference: str | None = None,
) -> dict[str, Any]:
    """Upsert feedback (thumbs up/down) for an assistant message."""
    if issue_tags is None:
        issue_tags = []

    with closing(get_connection()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO message_feedback (
                    message_id, rating, issue_tags, user_comment, corrected_reference, updated_at
                ) VALUES (%s, %s, %s, %s, %s, NOW())
                ON CONFLICT (message_id)
                DO UPDATE SET
                    rating = EXCLUDED.rating,
                    issue_tags = EXCLUDED.issue_tags,
                    user_comment = EXCLUDED.user_comment,
                    corrected_reference = EXCLUDED.corrected_reference,
                    updated_at = NOW()
                RETURNING id, message_id, rating, issue_tags, user_comment, corrected_reference, created_at, updated_at;
                """,
                (
                    str(message_id),
                    rating,
                    psycopg.types.json.Jsonb(issue_tags),
                    user_comment,
                    corrected_reference,
                ),
            )
            feedback = cur.fetchone()
            conn.commit()

            # Mirror into query_feedback for Admin UI Analytics & Benchmark Export
            try:
                # 1. Fetch assistant message details
                cur.execute(
                    """
                    SELECT session_id, content, condensed_query, retrieved_contexts, 
                           sources, documentation_gaps, generation_model, latency_ms, created_at
                    FROM chat_messages 
                    WHERE id = %s;
                    """,
                    (str(message_id),),
                )
                asst_row = cur.fetchone()
                if asst_row:
                    # 2. Find corresponding user query in the session
                    cur.execute(
                        """
                        SELECT id, content 
                        FROM chat_messages 
                        WHERE session_id = %s AND role = 'user' AND created_at <= %s 
                        ORDER BY created_at DESC 
                        LIMIT 1;
                        """,
                        (asst_row["session_id"], asst_row["created_at"]),
                    )
                    user_row = cur.fetchone()
                    user_query = user_row["content"] if user_row else (asst_row.get("condensed_query") or "Chat query")

                    # 3. Check for attachments on user message
                    att_fn = None
                    att_data = None
                    att_mime = None
                    if user_row:
                        cur.execute(
                            "SELECT filename, file_data, mime_type FROM chat_attachments WHERE message_id = %s LIMIT 1;",
                            (user_row["id"],),
                        )
                        att_row = cur.fetchone()
                        if att_row:
                            att_fn = att_row["filename"]
                            att_data = bytes(att_row["file_data"]) if att_row.get("file_data") else None
                            att_mime = att_row["mime_type"]

                    # 4. Save to query_feedback (map rating: +1 -> 5, -1 -> 1)
                    from src.feedback.feedback_store import save_feedback
                    save_feedback(
                        query=user_query,
                        response=asst_row["content"],
                        retrieved_contexts=asst_row.get("retrieved_contexts") or [],
                        sources=asst_row.get("sources") or [],
                        use_hybrid=True,
                        use_reranker=True,
                        pool_size=5,
                        final_top_k=len(asst_row.get("retrieved_contexts") or []) or 2,
                        generation_model=asst_row.get("generation_model") or "gemini-2.5-flash",
                        latency_ms=asst_row.get("latency_ms") or 0,
                        rating=5 if rating > 0 else 1,
                        issue_tags=issue_tags,
                        corrected_reference=corrected_reference,
                        user_comment=user_comment,
                        attached_filename=att_fn,
                        attached_file_data=att_data,
                        attached_file_mime=att_mime,
                        documentation_gaps=asst_row.get("documentation_gaps"),
                    )
                    logger.info(f"Successfully mirrored message feedback for message {message_id} to query_feedback.")
            except Exception as mirror_err:
                logger.warning(f"Could not mirror message feedback to query_feedback: {mirror_err}")

            return feedback


def get_message_feedback(message_id: str | uuid.UUID) -> dict[str, Any] | None:
    """Get feedback for a specific message."""
    with closing(get_connection()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, message_id, rating, issue_tags, user_comment, corrected_reference, created_at, updated_at
                FROM message_feedback
                WHERE message_id = %s;
                """,
                (str(message_id),),
            )
            return cur.fetchone()


# ---------------------------------------------------------------------------
# 5. Session Transcript Export
# ---------------------------------------------------------------------------
def export_chat_session(session_id: str | uuid.UUID, export_format: str = "markdown") -> dict[str, Any]:
    """Export the complete transcript of a chat session into Markdown or JSON format."""
    session = get_chat_session(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found.")

    messages = get_session_messages(session_id)

    if export_format.lower() == "json":
        return {
            "session": session,
            "messages": messages,
        }

    # Build Markdown format
    created_str = session["created_at"].strftime("%Y-%m-%d %H:%M:%S UTC") if isinstance(session["created_at"], datetime) else str(session["created_at"])
    lines = [
        f"# 💬 Chat Transcript: {session['title']}",
        f"**Session ID:** `{session['id']}`  ",
        f"**User:** `{session['user_email']}`  ",
        f"**Created:** {created_str}  ",
        "",
        "---",
        "",
    ]

    for msg in messages:
        role_label = "👤 **User**" if msg["role"] == "user" else "🤖 **Assistant**"
        msg_time = msg["created_at"].strftime("%H:%M:%S") if isinstance(msg["created_at"], datetime) else ""
        lines.append(f"### {role_label} *({msg_time})*")
        
        if msg.get("attachment"):
            att = msg["attachment"]
            lines.append(f"> 📎 **Attached File:** `{att['filename']}` ({att['mime_type']}, {att['file_size_bytes']} bytes)\n")

        lines.append(msg["content"])
        lines.append("")

        if msg.get("sources"):
            lines.append("**Sources Referenced:**")
            for src in msg["sources"]:
                lines.append(f"- {src}")
            lines.append("")

        if msg.get("feedback"):
            fb = msg["feedback"]
            icon = "👍 Thumbs Up" if fb["rating"] > 0 else "👎 Thumbs Down"
            lines.append(f"> 🏷️ **User Feedback:** {icon}")
            if fb.get("user_comment"):
                lines.append(f"> *Comment:* {fb['user_comment']}")
            lines.append("")

        lines.append("---")
        lines.append("")

    session_id_str = str(session["id"])
    return {
        "filename": f"chat_transcript_{session_id_str[:8]}.md",
        "content_type": "text/markdown",
        "content": "\n".join(lines),
    }


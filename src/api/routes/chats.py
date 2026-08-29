"""Chat session management, multi-turn messaging, attachments, streaming, and transcript export."""

import re
import time
import json
import logging
from typing import Optional, Literal
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    UploadFile,
    File,
    Form,
    Query,
    Request,
    BackgroundTasks,
    Response,
)
from fastapi.responses import StreamingResponse

from src.db import chat_store
from src.api.models import (
    CreateSessionRequest,
    UpdateSessionRequest,
    SessionResponse,
    ChatHistoryResponse,
    MessageResponse,
    AttachmentResponse,
    SendMessageResponse,
)
from src.api.dependencies import get_pipeline, get_memory_mgr
from src.pipeline.orchestrator import RAGPipeline
from src.api.memory import ConversationalMemoryManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chats", tags=["Chats"])


# ---------------------------------------------------------------------------
# Background Tasks
# ---------------------------------------------------------------------------
def _async_auto_title_session(
    session_id: str,
    query: str,
    response: str,
    memory_mgr: ConversationalMemoryManager,
):
    """Background worker to auto-generate a descriptive session title after first turn."""
    try:
        session = chat_store.get_chat_session(session_id)
        if session and (not session["title"] or session["title"] in ("New Conversation", "New Chat")):
            new_title = memory_mgr.generate_chat_title(query, response)
            chat_store.update_chat_session(session_id, title=new_title)
            logger.info(f"Auto-titled session {session_id} to '{new_title}'")
    except Exception as e:
        logger.warning(f"Failed to auto-title session {session_id}: {e}")


# ---------------------------------------------------------------------------
# 1. Session CRUD Endpoints
# ---------------------------------------------------------------------------
@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new chat session",
    description="Initiates a new conversation for a user email and returns a session ID for subsequent messages.",
)
async def create_chat(payload: CreateSessionRequest):
    try:
        user = chat_store.get_or_create_user(email=payload.user_email)
        title = payload.title or "New Conversation"
        session = chat_store.create_chat_session(
            user_id=user["id"],
            title=title,
            metadata=payload.metadata,
        )
        session["user_email"] = user["email"]
        return SessionResponse(**session)
    except Exception as e:
        logger.error(f"Error creating chat session for {payload.user_email}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create chat session: {str(e)}",
        )


@router.get(
    "/{chat_id}",
    response_model=ChatHistoryResponse,
    summary="Get chat session details and message history",
    description="Retrieves the session metadata and all previous turns (user messages, replies, attachments, sources, and feedback).",
)
async def get_chat(chat_id: str):
    session = chat_store.get_chat_session(chat_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{chat_id}' not found.",
        )

    messages = chat_store.get_session_messages(chat_id)
    return ChatHistoryResponse(
        session=SessionResponse(**session),
        messages=[MessageResponse(**m) for m in messages],
    )


@router.patch(
    "/{chat_id}",
    response_model=SessionResponse,
    summary="Update chat session (rename or archive)",
)
async def update_chat(chat_id: str, payload: UpdateSessionRequest):
    updated = chat_store.update_chat_session(
        session_id=chat_id,
        title=payload.title,
        is_archived=payload.is_archived,
        metadata=payload.metadata,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{chat_id}' not found.",
        )
    return SessionResponse(**updated)


@router.delete(
    "/{chat_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete / archive chat session",
)
async def delete_chat(
    chat_id: str,
    hard_delete: bool = Query(default=False, description="Permanently delete from database if true"),
):
    success = chat_store.delete_chat_session(session_id=chat_id, hard_delete=hard_delete)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{chat_id}' not found.",
        )
    return None


# ---------------------------------------------------------------------------
# 2. Messaging & Multimodal RAG Endpoint (Streaming + Non-Streaming)
# ---------------------------------------------------------------------------
@router.post(
    "/{chat_id}/messages",
    summary="Post a message to a chat session and receive RAG response",
    description=(
        "Accepts user message content and an optional file attachment (PDF, image, log, text). "
        "Supports standard JSON response or real-time Server-Sent Events (SSE) streaming."
    ),
)
async def send_message(
    chat_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    content: str = Form(..., description="User message prompt/query"),
    stream: bool = Form(default=False, description="Enable Server-Sent Events streaming"),
    attachment: Optional[UploadFile] = File(default=None, description="Optional document or image attachment"),
    pipeline: RAGPipeline = Depends(get_pipeline),
    memory_mgr: ConversationalMemoryManager = Depends(get_memory_mgr),
):
    # Verify session exists
    session = chat_store.get_chat_session(chat_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{chat_id}' not found.",
        )

    # 1. Read attachment if provided
    att_bytes = None
    att_filename = None
    att_mime = None
    att_size = 0

    if attachment and attachment.filename:
        att_filename = attachment.filename
        att_mime = attachment.content_type or "application/octet-stream"
        att_bytes = await attachment.read()
        att_size = len(att_bytes)

    # 2. Record User Message in DB
    user_msg = chat_store.add_chat_message(
        session_id=chat_id,
        role="user",
        content=content,
    )

    # Store attachment record if present
    att_record = None
    if att_bytes and att_filename:
        att_record = chat_store.add_chat_attachment(
            message_id=user_msg["id"],
            filename=att_filename,
            mime_type=att_mime,
            file_size_bytes=att_size,
            file_data=att_bytes,
        )

    # 3. Retrieve conversation history for memory & contextualization
    history = chat_store.get_session_messages(chat_id)
    # Exclude the newly inserted message from history so memory sees prior turns
    prior_history = [m for m in history if str(m["id"]) != str(user_msg["id"])]

    # 4. Contextualize query for RAG retrieval
    condensed_query = memory_mgr.condense_query(prior_history, content)
    search_query = condensed_query

    # Augment search query if plain text attachment
    if att_bytes and att_mime and att_mime.startswith("text/"):
        try:
            snippet = att_bytes.decode("utf-8", errors="ignore")[:600].strip()
            if snippet:
                search_query = f"{condensed_query}\n[Report Snippet: {snippet}]"
        except Exception:
            pass

    # 5. Core RAG Retrieval & Semantic Reranking
    start_time = time.time()
    query_vector = pipeline.embedding_service.get_dense_embedding(search_query)
    candidates = pipeline.retriever.hybrid_search_rrf(search_query, query_vector, limit=5)
    retrieved_contexts = pipeline.reranker.rank_candidates(search_query, candidates, top_n=2)

    # Extract unique source URLs
    sources = []
    for doc in retrieved_contexts:
        metadata = doc.get("metadata") or {}
        url = metadata.get("image_url") or metadata.get("pdf_url") or metadata.get("source_url")
        if url and url not in sources:
            sources.append(url)

    # 6. Format Generation Contents
    prompt_contents = memory_mgr.format_generation_contents(
        history=prior_history,
        current_query=content,
        retrieved_contexts=retrieved_contexts,
        attached_file_bytes=att_bytes,
        attached_mime_type=att_mime,
    )

    # Check if SSE stream requested
    accept_header = request.headers.get("accept", "")
    is_stream = stream or "text/event-stream" in accept_header

    # Helper function to parse doc gaps and sources footer
    def _finalize_response_text(raw_text: str) -> tuple[str, Optional[str]]:
        doc_gaps = None
        gaps_pattern = r"<!--\s*DOCUMENTATION_GAPS\s*-->([\s\S]*?)<!--\s*END_DOCUMENTATION_GAPS\s*-->"
        gaps_match = re.search(gaps_pattern, raw_text, re.IGNORECASE)
        if gaps_match:
            gaps_content = gaps_match.group(1).strip()
            raw_text = re.sub(gaps_pattern, "", raw_text, flags=re.IGNORECASE).strip()
            if gaps_content and not gaps_content.lower().startswith("none"):
                doc_gaps = gaps_content

        # Append source links footer if sources exist and not already in text
        if sources and "Sources:" not in raw_text:
            sources_footer = "\n\nSources:\n" + "\n".join(f"- {src}" for src in sources)
            raw_text += sources_footer

        return raw_text, doc_gaps

    # -----------------------------------------------------------------------
    # Streaming Response (SSE)
    # -----------------------------------------------------------------------
    if is_stream:
        async def event_generator():
            accumulated_chunks = []
            try:
                # Stream generation from Gemini
                stream_res = pipeline.generator_client.models.generate_content_stream(
                    model=pipeline.generator_model,
                    contents=prompt_contents,
                )

                for chunk in stream_res:
                    chunk_text = chunk.text or ""
                    if chunk_text:
                        accumulated_chunks.append(chunk_text)
                        payload = json.dumps({"type": "token", "content": chunk_text})
                        yield f"data: {payload}\n\n"

                # Finalize
                full_raw_text = "".join(accumulated_chunks)
                final_text, doc_gaps = _finalize_response_text(full_raw_text)
                latency_ms = int((time.time() - start_time) * 1000)

                # Persist assistant message in DB
                assistant_msg = chat_store.add_chat_message(
                    session_id=chat_id,
                    role="assistant",
                    content=final_text,
                    condensed_query=condensed_query,
                    retrieved_contexts=[doc["content"] for doc in retrieved_contexts],
                    sources=sources,
                    documentation_gaps=doc_gaps,
                    generation_model=pipeline.generator_model,
                    latency_ms=latency_ms,
                )

                # Trigger auto-title if first turn
                if len(prior_history) == 0:
                    background_tasks.add_task(
                        _async_auto_title_session,
                        chat_id,
                        content,
                        final_text,
                        memory_mgr,
                    )

                # Final metadata event
                final_payload = json.dumps({
                    "type": "done",
                    "assistant_message_id": str(assistant_msg["id"]),
                    "sources": sources,
                    "documentation_gaps": doc_gaps,
                    "latency_ms": latency_ms,
                })
                yield f"data: {final_payload}\n\n"

            except Exception as ex:
                logger.error(f"Error during streaming generation: {ex}", exc_info=True)
                error_payload = json.dumps({"type": "error", "error": str(ex)})
                yield f"data: {error_payload}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # -----------------------------------------------------------------------
    # Non-Streaming JSON Response
    # -----------------------------------------------------------------------
    gen_res = pipeline.generator_client.models.generate_content(
        model=pipeline.generator_model,
        contents=prompt_contents,
    )
    raw_text = gen_res.text.strip()
    final_text, doc_gaps = _finalize_response_text(raw_text)
    latency_ms = int((time.time() - start_time) * 1000)

    # Extract token metrics if available
    prompt_tokens = getattr(gen_res.usage_metadata, "prompt_token_count", None) if hasattr(gen_res, "usage_metadata") else None
    completion_tokens = getattr(gen_res.usage_metadata, "candidates_token_count", None) if hasattr(gen_res, "usage_metadata") else None

    # Persist assistant message in DB
    assistant_msg = chat_store.add_chat_message(
        session_id=chat_id,
        role="assistant",
        content=final_text,
        condensed_query=condensed_query,
        retrieved_contexts=[doc["content"] for doc in retrieved_contexts],
        sources=sources,
        documentation_gaps=doc_gaps,
        generation_model=pipeline.generator_model,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    # Auto-title background task if first turn
    if len(prior_history) == 0:
        background_tasks.add_task(
            _async_auto_title_session,
            chat_id,
            content,
            final_text,
            memory_mgr,
        )

    # Build response payload
    user_attachment_resp = None
    if att_record:
        user_attachment_resp = AttachmentResponse(**att_record)

    user_msg_resp = MessageResponse(
        id=user_msg["id"],
        session_id=user_msg["session_id"],
        role=user_msg["role"],
        content=user_msg["content"],
        created_at=user_msg["created_at"],
        attachment=user_attachment_resp,
    )

    assistant_msg_resp = MessageResponse(
        id=assistant_msg["id"],
        session_id=assistant_msg["session_id"],
        role=assistant_msg["role"],
        content=assistant_msg["content"],
        condensed_query=assistant_msg["condensed_query"],
        retrieved_contexts=assistant_msg["retrieved_contexts"],
        sources=assistant_msg["sources"],
        documentation_gaps=assistant_msg["documentation_gaps"],
        generation_model=assistant_msg["generation_model"],
        latency_ms=assistant_msg["latency_ms"],
        prompt_tokens=assistant_msg["prompt_tokens"],
        completion_tokens=assistant_msg["completion_tokens"],
        created_at=assistant_msg["created_at"],
    )

    return SendMessageResponse(
        user_message=user_msg_resp,
        assistant_message=assistant_msg_resp,
    )


# ---------------------------------------------------------------------------
# 3. Attachment Download & Transcript Export
# ---------------------------------------------------------------------------
@router.get(
    "/attachments/{attachment_id}",
    summary="Download or view an attached file binary",
)
async def download_attachment(attachment_id: str):
    att = chat_store.get_attachment(attachment_id)
    if not att or not att.get("file_data"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Attachment '{attachment_id}' not found.",
        )

    return Response(
        content=bytes(att["file_data"]),
        media_type=att["mime_type"],
        headers={"Content-Disposition": f'inline; filename="{att["filename"]}"'},
    )


@router.get(
    "/{chat_id}/export",
    summary="Export conversation transcript (Markdown or JSON)",
)
async def export_chat(
    chat_id: str,
    format: Literal["markdown", "json"] = Query(default="markdown", description="Export format"),
):
    try:
        export_data = chat_store.export_chat_session(chat_id, export_format=format)
        if format == "json":
            return export_data
        else:
            return Response(
                content=export_data["content"],
                media_type="text/markdown; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{export_data["filename"]}"'},
            )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except Exception as e:
        logger.error(f"Error exporting session {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to export transcript.")


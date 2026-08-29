"""Pydantic schemas for FastAPI request validation and response serialization."""

import uuid
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, EmailStr


# ---------------------------------------------------------------------------
# Session Schemas
# ---------------------------------------------------------------------------
class CreateSessionRequest(BaseModel):
    user_email: str = Field(..., description="Email of the user initiating the chat session", examples=["john.doe@mail.com"])
    title: str | None = Field(default=None, description="Optional custom chat title. If omitted, auto-generated from conversation.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary client metadata (e.g. client version, department)")


class UpdateSessionRequest(BaseModel):
    title: str | None = Field(default=None, description="Updated session title")
    is_archived: bool | None = Field(default=None, description="Archive or unarchive the session")
    metadata: dict[str, Any] | None = Field(default=None, description="Updated session metadata")


class SessionResponse(BaseModel):
    id: uuid.UUID | str
    user_id: uuid.UUID | str
    user_email: str | None = None
    title: str
    is_archived: bool
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    message_count: int | None = None
    last_message_preview: str | None = None


class SessionListResponse(BaseModel):
    user_email: str
    total_count: int
    limit: int
    offset: int
    chats: list[SessionResponse]


# ---------------------------------------------------------------------------
# Attachment Schemas
# ---------------------------------------------------------------------------
class AttachmentResponse(BaseModel):
    id: uuid.UUID | str
    filename: str
    mime_type: str
    file_size_bytes: int
    created_at: datetime | None = None


# ---------------------------------------------------------------------------
# Feedback Schemas
# ---------------------------------------------------------------------------
class FeedbackRequest(BaseModel):
    rating: int = Field(..., description="1 for Thumbs Up, -1 for Thumbs Down", ge=-1, le=1)
    issue_tags: list[str] = Field(
        default_factory=list,
        description="Optional issue categories (e.g. 'incorrect_fact', 'hallucination', 'outdated_runbook', 'missing_step')",
    )
    user_comment: str | None = Field(default=None, description="Optional text comment or explanation")
    corrected_reference: str | None = Field(default=None, description="Optional link or text to correct reference")


class FeedbackResponse(BaseModel):
    id: uuid.UUID | str
    message_id: uuid.UUID | str
    rating: int
    issue_tags: list[str] = Field(default_factory=list)
    user_comment: str | None = None
    corrected_reference: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


# ---------------------------------------------------------------------------
# Message Schemas
# ---------------------------------------------------------------------------
class MessageResponse(BaseModel):
    id: uuid.UUID | str
    session_id: uuid.UUID | str
    role: Literal["user", "assistant", "system"]
    content: str
    condensed_query: str | None = None
    retrieved_contexts: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    documentation_gaps: str | None = None
    generation_model: str | None = None
    latency_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    created_at: datetime
    attachment: AttachmentResponse | None = None
    feedback: FeedbackResponse | None = None


class ChatHistoryResponse(BaseModel):
    session: SessionResponse
    messages: list[MessageResponse]


class SendMessageResponse(BaseModel):
    user_message: MessageResponse
    assistant_message: MessageResponse


# ---------------------------------------------------------------------------
# Export Schema
# ---------------------------------------------------------------------------
class ExportResponse(BaseModel):
    session_id: uuid.UUID | str
    format: Literal["markdown", "json"]
    content: str | dict[str, Any]


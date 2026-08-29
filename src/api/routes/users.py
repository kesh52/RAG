"""User management and user-specific chat session list routes."""

import logging
from fastapi import APIRouter, Query, HTTPException, status
from src.db import chat_store
from src.api.models import SessionListResponse, SessionResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["Users"])


@router.get(
    "/{user_email}/chats",
    response_model=SessionListResponse,
    summary="List all chat sessions for a specific user",
    description="Returns a paginated list of chat sessions owned by the given user email, with message counts and preview snippets.",
)
async def list_user_chats(
    user_email: str,
    limit: int = Query(default=20, ge=1, le=100, description="Max chats to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    include_archived: bool = Query(default=False, description="Whether to include archived sessions"),
):
    try:
        sessions, total_count = chat_store.get_user_chat_sessions(
            user_email=user_email,
            limit=limit,
            offset=offset,
            include_archived=include_archived,
        )
        return SessionListResponse(
            user_email=user_email,
            total_count=total_count,
            limit=limit,
            offset=offset,
            chats=[SessionResponse(**s) for s in sessions],
        )
    except Exception as e:
        logger.error(f"Error fetching chats for user {user_email}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chats for user: {str(e)}",
        )


"""Thumbs up / down feedback routes for chat assistant messages."""

import logging
from fastapi import APIRouter, HTTPException, status
from src.db import chat_store
from src.api.models import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Feedback"])


@router.post(
    "/messages/{message_id}/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit thumbs up/down rating and comments for an assistant reply",
    description="Allows users to submit or update feedback (rating +1 or -1, issue tags, user comment, corrected reference) on any assistant message.",
)
async def submit_message_feedback(message_id: str, payload: FeedbackRequest):
    try:
        feedback = chat_store.save_message_feedback(
            message_id=message_id,
            rating=payload.rating,
            issue_tags=payload.issue_tags,
            user_comment=payload.user_comment,
            corrected_reference=payload.corrected_reference,
        )
        logger.info(f"Recorded feedback (rating={payload.rating}) for message {message_id}")
        return FeedbackResponse(**feedback)
    except Exception as e:
        logger.error(f"Error saving feedback for message {message_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record feedback: {str(e)}",
        )


@router.get(
    "/messages/{message_id}/feedback",
    response_model=FeedbackResponse,
    summary="Get feedback status for a message",
)
async def get_message_feedback(message_id: str):
    fb = chat_store.get_message_feedback(message_id)
    if not fb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No feedback found for message '{message_id}'.",
        )
    return FeedbackResponse(**fb)


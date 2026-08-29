"""Dependency injection providers for FastAPI routes."""

import logging
from typing import Optional
from google import genai

from src.pipeline.orchestrator import RAGPipeline, get_default_pipeline
from src.api.memory import ConversationalMemoryManager
from src.utils.config import config

logger = logging.getLogger(__name__)

# Global singletons
_RAG_PIPELINE: Optional[RAGPipeline] = None
_MEMORY_MANAGER: Optional[ConversationalMemoryManager] = None


def get_pipeline() -> RAGPipeline:
    """Provide the initialized RAGPipeline singleton."""
    global _RAG_PIPELINE
    if _RAG_PIPELINE is None:
        logger.info("Initializing RAGPipeline singleton for FastAPI...")
        _RAG_PIPELINE = get_default_pipeline()
    return _RAG_PIPELINE


def get_memory_mgr() -> ConversationalMemoryManager:
    """Provide the ConversationalMemoryManager singleton."""
    global _MEMORY_MANAGER
    if _MEMORY_MANAGER is None:
        pipeline = get_pipeline()
        _MEMORY_MANAGER = ConversationalMemoryManager(
            genai_client=pipeline.generator_client,
            model_name=pipeline.generator_model,
        )
    return _MEMORY_MANAGER


"""FastAPI application factory for RAG Chat UI Service."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import users_router, chats_router, feedback_router
from src.utils.config import config
from src.utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan setup and teardown."""
    logger.info("Starting up FastAPI RAG Chat Service...")
    yield
    logger.info("Shutting down FastAPI RAG Chat Service...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="RAG Chat UI Service",
        description=(
            "Conversational multi-turn RAG API with context persistence, "
            "multimodal attachments, streaming responses, and user feedback."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Enable CORS for frontend Chat UI web apps
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust for specific origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API v1 Routers
    api_v1_prefix = "/api/v1"
    app.include_router(chats_router, prefix=api_v1_prefix)
    app.include_router(users_router, prefix=api_v1_prefix)
    app.include_router(feedback_router, prefix=api_v1_prefix)

    # Health check & Root info
    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "healthy", "service": "rag-chat-api", "version": "1.0.0"}

    @app.get("/", tags=["Info"])
    async def root_info():
        return {
            "name": "RAG Chat API",
            "version": "1.0.0",
            "docs": "/docs",
            "openapi": "/openapi.json",
        }

    return app


app = create_app()

"""Unit and integration tests for FastAPI RAG Chat endpoints and memory management."""

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.memory import ConversationalMemoryManager
from src.api.dependencies import get_pipeline, get_memory_mgr
from src.db import chat_store


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Health & Root Tests
# ---------------------------------------------------------------------------
def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_info(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "RAG Chat UI API"


# ---------------------------------------------------------------------------
# 2. Conversational Memory Unit Tests
# ---------------------------------------------------------------------------
def test_conversational_memory_condense_query_empty_history():
    mock_client = MagicMock()
    mem_mgr = ConversationalMemoryManager(mock_client)
    
    query = "How to configure SSL?"
    result = mem_mgr.condense_query([], query)
    assert result == query
    mock_client.models.generate_content.assert_not_called()


def test_conversational_memory_condense_query_with_history():
    mock_client = MagicMock()
    mock_res = MagicMock()
    mock_res.text = "How to configure PostgreSQL SSL certificate"
    mock_client.models.generate_content.return_value = mock_res

    mem_mgr = ConversationalMemoryManager(mock_client)
    history = [
        {"role": "user", "content": "I am deploying PostgreSQL"},
        {"role": "assistant", "content": "PostgreSQL deployment instructions..."},
    ]
    result = mem_mgr.condense_query(history, "How do I configure SSL for it?")
    assert result == "How to configure PostgreSQL SSL certificate"
    mock_client.models.generate_content.assert_called_once()


def test_conversational_memory_generate_title():
    mock_client = MagicMock()
    mock_res = MagicMock()
    mock_res.text = "PostgreSQL SSL Setup"
    mock_client.models.generate_content.return_value = mock_res

    mem_mgr = ConversationalMemoryManager(mock_client)
    title = mem_mgr.generate_chat_title("How to set up SSL on PG?", "Here are the steps...")
    assert title == "PostgreSQL SSL Setup"


# ---------------------------------------------------------------------------
# 3. Session Endpoints Tests
# ---------------------------------------------------------------------------
@patch("src.db.chat_store.get_or_create_user")
@patch("src.db.chat_store.create_chat_session")
def test_create_chat_session(mock_create_session, mock_get_user, client):
    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    now = datetime.utcnow()

    mock_get_user.return_value = {"id": user_id, "email": "john.doe@mail.com"}
    mock_create_session.return_value = {
        "id": session_id,
        "user_id": user_id,
        "title": "Incident Investigation",
        "is_archived": False,
        "metadata": {"dept": "DevOps"},
        "created_at": now,
        "updated_at": now,
    }

    response = client.post(
        "/api/v1/chats",
        json={"user_email": "john.doe@mail.com", "title": "Incident Investigation", "metadata": {"dept": "DevOps"}},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["id"] == session_id
    assert data["user_email"] == "john.doe@mail.com"
    assert data["title"] == "Incident Investigation"


@patch("src.db.chat_store.get_user_chat_sessions")
def test_list_user_chats(mock_get_sessions, client):
    session_id = str(uuid.uuid4())
    now = datetime.utcnow()
    mock_get_sessions.return_value = (
        [
            {
                "id": session_id,
                "user_id": str(uuid.uuid4()),
                "user_email": "john.doe@mail.com",
                "title": "Payment Issue",
                "is_archived": False,
                "metadata": {},
                "created_at": now,
                "updated_at": now,
                "message_count": 4,
                "last_message_preview": "Check the ingress controller",
            }
        ],
        1,
    )

    response = client.get("/api/v1/users/john.doe@mail.com/chats?limit=10&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 1
    assert len(data["chats"]) == 1
    assert data["chats"][0]["title"] == "Payment Issue"
    assert data["chats"][0]["message_count"] == 4


@patch("src.db.chat_store.get_chat_session")
@patch("src.db.chat_store.get_session_messages")
def test_get_chat_history(mock_get_messages, mock_get_session, client):
    session_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    now = datetime.utcnow()

    mock_get_session.return_value = {
        "id": session_id,
        "user_id": user_id,
        "user_email": "john.doe@mail.com",
        "title": "Kafka Troubleshooting",
        "is_archived": False,
        "metadata": {},
        "created_at": now,
        "updated_at": now,
    }
    mock_get_messages.return_value = [
        {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "role": "user",
            "content": "Why is consumer lagging?",
            "condensed_query": None,
            "retrieved_contexts": [],
            "sources": [],
            "documentation_gaps": None,
            "generation_model": None,
            "latency_ms": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "created_at": now,
            "attachment": None,
            "feedback": None,
        }
    ]

    response = client.get(f"/api/v1/chats/{session_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["session"]["title"] == "Kafka Troubleshooting"
    assert len(data["messages"]) == 1
    assert data["messages"][0]["role"] == "user"


@patch("src.db.chat_store.update_chat_session")
def test_update_chat_session(mock_update, client):
    session_id = str(uuid.uuid4())
    now = datetime.utcnow()
    mock_update.return_value = {
        "id": session_id,
        "user_id": str(uuid.uuid4()),
        "title": "Renamed Incident",
        "is_archived": True,
        "metadata": {},
        "created_at": now,
        "updated_at": now,
    }

    response = client.patch(
        f"/api/v1/chats/{session_id}",
        json={"title": "Renamed Incident", "is_archived": True},
    )
    assert response.status_code == 200
    assert response.json()["title"] == "Renamed Incident"
    assert response.json()["is_archived"] is True


# ---------------------------------------------------------------------------
# 4. Message Posting & Multimodal RAG Tests
# ---------------------------------------------------------------------------
@patch("src.db.chat_store.get_chat_session")
@patch("src.db.chat_store.add_chat_message")
@patch("src.db.chat_store.add_chat_attachment")
@patch("src.db.chat_store.get_session_messages")
def test_send_message_with_attachment(
    mock_get_messages,
    mock_add_attachment,
    mock_add_msg,
    mock_get_session,
    client,
):
    session_id = str(uuid.uuid4())
    user_msg_id = str(uuid.uuid4())
    assistant_msg_id = str(uuid.uuid4())
    now = datetime.utcnow()

    mock_get_session.return_value = {
        "id": session_id,
        "user_id": str(uuid.uuid4()),
        "user_email": "john.doe@mail.com",
        "title": "New Conversation",
        "is_archived": False,
        "metadata": {},
        "created_at": now,
        "updated_at": now,
    }

    def side_effect_add_msg(session_id, role, content, **kwargs):
        msg_id = user_msg_id if role == "user" else assistant_msg_id
        return {
            "id": msg_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "condensed_query": kwargs.get("condensed_query"),
            "retrieved_contexts": kwargs.get("retrieved_contexts") or [],
            "sources": kwargs.get("sources") or [],
            "documentation_gaps": kwargs.get("documentation_gaps"),
            "generation_model": kwargs.get("generation_model"),
            "latency_ms": kwargs.get("latency_ms", 150),
            "prompt_tokens": kwargs.get("prompt_tokens"),
            "completion_tokens": kwargs.get("completion_tokens"),
            "created_at": now,
        }

    mock_add_msg.side_effect = side_effect_add_msg
    mock_get_messages.return_value = []
    mock_add_attachment.return_value = {
        "id": str(uuid.uuid4()),
        "message_id": user_msg_id,
        "filename": "error.log",
        "mime_type": "text/plain",
        "file_size_bytes": 45,
        "created_at": now,
    }

    # Mock RAG pipeline dependency
    mock_pipeline = MagicMock()
    mock_pipeline.embedding_service.get_dense_embedding.return_value = [0.1] * 768
    mock_pipeline.retriever.hybrid_search_rrf.return_value = [
        {"content": "Runbook step 1: restart server", "metadata": {"source_url": "https://wiki.corp/runbook1"}}
    ]
    mock_pipeline.reranker.rank_candidates.return_value = [
        {"content": "Runbook step 1: restart server", "metadata": {"source_url": "https://wiki.corp/runbook1"}}
    ]
    mock_pipeline.generator_model = "gemini-2.5-flash"
    mock_gen_res = MagicMock()
    mock_gen_res.text = "### 📚 Internal Documentation Guidance\nRestart the server.\n<!-- DOCUMENTATION_GAPS -->\nNONE\n<!-- END_DOCUMENTATION_GAPS -->"
    mock_gen_res.usage_metadata.prompt_token_count = 120
    mock_gen_res.usage_metadata.candidates_token_count = 50
    mock_pipeline.generator_client.models.generate_content.return_value = mock_gen_res

    # Mock memory manager
    mock_mem_mgr = MagicMock()
    mock_mem_mgr.condense_query.return_value = "Server error investigation"
    mock_mem_mgr.format_generation_contents.return_value = "Prompt text"
    mock_mem_mgr.generate_chat_title.return_value = "Server Error Investigation"

    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    app.dependency_overrides[get_memory_mgr] = lambda: mock_mem_mgr

    try:
        response = client.post(
            f"/api/v1/chats/{session_id}/messages",
            data={"content": "Please analyze this error log", "stream": "false"},
            files={"attachment": ("error.log", b"Error: Connection reset by peer at port 5432", "text/plain")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user_message"]["role"] == "user"
        assert data["user_message"]["attachment"]["filename"] == "error.log"
        assert data["assistant_message"]["role"] == "assistant"
        assert "https://wiki.corp/runbook1" in data["assistant_message"]["sources"]
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# 5. Feedback Endpoints Tests
# ---------------------------------------------------------------------------
@patch("src.db.chat_store.save_message_feedback")
def test_submit_feedback(mock_save_feedback, client):
    msg_id = str(uuid.uuid4())
    fb_id = str(uuid.uuid4())
    now = datetime.utcnow()

    mock_save_feedback.return_value = {
        "id": fb_id,
        "message_id": msg_id,
        "rating": 1,
        "issue_tags": ["helpful_runbook"],
        "user_comment": "Exact step that worked!",
        "corrected_reference": None,
        "created_at": now,
        "updated_at": now,
    }

    response = client.post(
        f"/api/v1/messages/{msg_id}/feedback",
        json={
            "rating": 1,
            "issue_tags": ["helpful_runbook"],
            "user_comment": "Exact step that worked!",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == fb_id
    assert data["rating"] == 1
    assert data["user_comment"] == "Exact step that worked!"


# ---------------------------------------------------------------------------
# 6. Transcript Export Tests
# ---------------------------------------------------------------------------
@patch("src.db.chat_store.export_chat_session")
def test_export_chat_markdown(mock_export, client):
    session_id = str(uuid.uuid4())
    mock_export.return_value = {
        "filename": "chat_transcript.md",
        "content_type": "text/markdown",
        "content": "# Chat Transcript\n\nUser: Hello\nAssistant: Hi there!",
    }

    response = client.get(f"/api/v1/chats/{session_id}/export?format=markdown")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "User: Hello" in response.text


@patch("src.db.chat_store.export_chat_session")
def test_export_chat_json(mock_export, client):
    session_id = str(uuid.uuid4())
    mock_export.return_value = {
        "session": {"id": session_id, "title": "Test Chat"},
        "messages": [],
    }

    response = client.get(f"/api/v1/chats/{session_id}/export?format=json")
    assert response.status_code == 200
    assert response.json()["session"]["title"] == "Test Chat"


# Parameterized RAG Pipeline, Conversational Chat Service & Evaluation Platform

This repository contains an enterprise-grade RAG (Retrieval-Augmented Generation) system integrated with **PostgreSQL (Cloud SQL + pgvector)** and **Google Cloud Vertex AI** (Embeddings, Semantic Ranker, Gemini 2.5 Flash / Pro).

It provides:
- 🚀 **FastAPI Chat Service**: Production REST & SSE streaming API with conversational session persistence, multi-turn memory, multimodal attachments, and user feedback.
- 💬 **ChatGPT-Style Web Chat UIs**: Standalone modern browser UI (`/chat`) and dedicated Streamlit chat app (`chat_app.py`).
- 🧠 **Conversational Memory Manager**: Automatic multi-turn query contextualization/rewriting for precise pgvector hybrid retrieval.
- 🔬 **Admin Operations Dashboard**: ETL crawling, database exploration, automated Ragas evaluations, and feedback dataset triage.
- 📎 **Multimodal Document Remediation**: Ingestion of PDFs, images, log files, and text reports alongside internal knowledge base runbooks.

---

## Directory Structure

```
AI/
├── .env                         # Environment variables (GCP, DB credentials)
├── requirements.txt             # Python project dependencies
├── config.yaml                  # Dynamic configuration parameters
├── pytest.ini                   # Pytest configuration file
├── architecture.md              # Pipeline architecture and design documentation
├── alembic.ini                  # Alembic migrations configuration
├── alembic/                     # Database version migration scripts
│   └── versions/                # Schema revisions (pgvector, feedback, chat_ui tables)
├── admin_app.py                 # Admin dashboard (ETL, DB Explorer, Evals, Playground)
├── chat_app.py                  # Standalone ChatGPT-style user chat application
├── run_api.py                   # FastAPI server entrypoint (Uvicorn launcher)
├── assets/                      # Assets storage (sample incident reports, crawler uploads)
├── docs/                        # Specifications and design plans
├── src/                         # Core Python package
│   ├── api/                     # FastAPI Chat Service
│   │   ├── app.py               # FastAPI application factory & CORS setup
│   │   ├── dependencies.py      # Dependency injection providers
│   │   ├── models.py            # Pydantic validation & response schemas
│   │   ├── memory.py            # Conversational memory & query contextualizer
│   │   ├── static/              # Web Chat UI (HTML5 + Tailwind CSS + SSE)
│   │   │   └── index.html       # Full-screen conversational client
│   │   └── routes/              # API v1 route handlers
│   │       ├── chats.py         # Sessions, multi-turn messages, attachments, export
│   │       ├── users.py         # User management & session listing
│   │       └── feedback.py      # Thumbs up/down feedback endpoints
│   ├── db/                      # Database connectivity & persistence layer
│   │   ├── pg_connector.py      # Cloud SQL connector & Postgres connection factory
│   │   └── chat_store.py        # PostgreSQL CRUD for users, sessions, messages, feedback
│   ├── embeddings/              # Dense vector embeddings (Vertex AI text-embedding-005)
│   ├── retrieval/               # Postgres pgvector hybrid retrieval (Dense + Sparse RRF)
│   ├── reranking/               # Semantic reranker (Vertex Discovery Engine)
│   ├── pipeline/                # End-to-end RAG pipeline orchestrator
│   ├── feedback/                # Feedback repository, analytics & benchmark curation
│   ├── etl/                     # Confluence scrapers, BFS crawlers & chunkers
│   └── utils/                   # Shared config and rotating logger utilities
├── evaluation/                  # Automated Ragas evaluation runner & datasets
├── scripts/                     # Utility scripts (migrations, DB seed, proxies)
└── tests/                       # Automated test suite (Pytest)
    ├── test_chat_api.py         # Unit & integration tests for FastAPI chat & memory
    ├── test_pipeline.py         # RAG pipeline mock unit tests
    ├── test_chunking.py         # Text chunking unit tests
    ├── test_etl.py              # ETL crawler unit tests
    └── test_config.py           # Configuration resolver tests
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Create a `.env` file in the root directory:
```ini
DB_PASSWORD=your_database_password
INSTANCE_CONNECTION_NAME=your_gcp_project:europe-west3:your_db_instance
GCP_PROJECT=your_gcp_project
GCP_LOCATION=europe-west3
GOOGLE_CLOUD_QUOTA_PROJECT=your_gcp_project
```

### 3. Connect to Cloud SQL & Run Migrations
```bash
# Terminal 1: Start Cloud SQL Auth Proxy
python3 scripts/start_proxy.py

# Terminal 2: Run Alembic Database Migrations
python3 -m scripts.run_migrations

# Seed initial knowledge base runbooks (Optional)
python3 -m scripts.seed_db
```

---

## 💬 Running the User Chat Interfaces

### Option A: FastAPI Web Chat (Recommended for Production & End Users)
Start the high-performance FastAPI server:
```bash
python3 run_api.py
```
- **Web Chat UI**: 👉 [http://localhost:8000/chat](http://localhost:8000/chat)
- **Interactive OpenAPI Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Option B: Standalone Streamlit Chat App
Launch the dedicated conversational Streamlit client:
```bash
streamlit run chat_app.py
```

---

## 🔬 Running the Admin Dashboard
Launch the operational management portal for ETL crawlers, database health checks, automated Ragas benchmarks, and feedback dataset triage:
```bash
streamlit run admin_app.py
```

---

## 📡 FastAPI REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/chats` | Initiates a new chat session for a user email (returns `session_id`). |
| `GET` | `/api/v1/users/{user_email}/chats` | Returns a paginated list of chat sessions with snippet previews and message counts. |
| `GET` | `/api/v1/chats/{chat_id}` | Retrieves full conversation history, attachments, cited sources, and feedback. |
| `PATCH` | `/api/v1/chats/{chat_id}` | Renames a session title or archives a conversation. |
| `DELETE` | `/api/v1/chats/{chat_id}` | Archives or deletes a chat session. |
| `POST` | `/api/v1/chats/{chat_id}/messages` | Sends a message with optional file attachment (`PDF`, `image`, `log`, `text`). Supports standard JSON or real-time Server-Sent Events (`Accept: text/event-stream`). |
| `GET` | `/api/v1/chats/{chat_id}/export` | Exports the full conversation transcript as formatted Markdown or JSON. |
| `GET` | `/api/v1/chats/attachments/{id}` | Downloads or streams an attached binary file. |
| `POST` | `/api/v1/messages/{id}/feedback` | Submits thumbs up (+1) or thumbs down (-1) feedback with issue tags and comments. |
| `GET` | `/health` | Service health check. |

---

## 🧠 Multi-Turn Memory & Conversational Retrieval

1. **Multi-Turn Query Contextualization**:
   Before querying the vector database, [`ConversationalMemoryManager`](src/api/memory.py) analyzes the recent conversation turns and rewrites ambiguous follow-up questions into standalone, entity-rich search queries.
2. **Multimodal Ingestion**:
   Uploaded PDFs, scan logs, and error traces are ingested alongside internal runbooks and cited in responses.
3. **Continuous Feedback Loop**:
   Thumbs up/down feedback submitted in any chat interface persists to `message_feedback` and automatically mirrors into `query_feedback`, making it immediately accessible in the Admin Dashboard for evaluation and ground-truth dataset curation.

---

## 🧪 Running Automated Tests

Run the complete test suite:
```bash
python3 -m pytest tests/ -v
```
To run only the Chat API and memory management tests:
```bash
python3 -m pytest tests/test_chat_api.py -v
```

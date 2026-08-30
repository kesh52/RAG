# Implemented Requirements & Architectural Specification

**System Name:** Modular Multi-Stage RAG Platform  
**Document Version:** 1.0.0  
**Target Environment:** Google Cloud Platform (Cloud Run, Cloud SQL PostgreSQL 16 + pgvector, Vertex AI)  
**Status:** Implemented / Reverse-Engineered from Source Code  

---

## Executive Summary & System Overview

This specification documents the architecture, functional capabilities, data contracts, and non-functional characteristics of the modular Retrieval-Augmented Generation (RAG) system. The platform orchestrates enterprise knowledge ingestion from Atlassian Confluence, multimodal document transcription (PDFs, images), hybrid retrieval (dense vector HNSW + sparse BM25 full-text search) merged with Reciprocal Rank Fusion (RRF), cross-encoder semantic reranking, dual-track LLM answer synthesis with silent knowledge gap detection, and continuous evaluation/annotation loops backed by Ragas.

---

## Architectural Component Breakdown

### 1. Configuration & Infrastructure Core

#### Responsibilities & Scope
The Configuration & Logging Core provides centralized, environment-aware configuration management and standardized rotating diagnostics across all pipeline subsystems. Its boundary covers loading dynamic settings from `config.yaml`, resolving runtime environment variables with fallback defaults, and initializing structured logging sinks.

#### Implemented Functional Requirements (FR)
- **`FR-CFG-01` Dynamic Variable Interpolation**: Parses YAML configuration files with nested `${VAR_NAME:default_value}` syntax, resolving sensitive credentials and environment-specific endpoints dynamically at runtime.
- **`FR-CFG-02` Dot-Notation Configuration Access**: Exposes a singleton `Config` class with hierarchical dot-separated key traversal (e.g., `config.get("database.host", "127.0.0.1")`).
- **`FR-LOG-01` Dual-Sink Rotating Logging**: Configures synchronous console logging and a size-managed `RotatingFileHandler` with automated log directory creation.

#### Technical & Non-Functional Decisions (NFR)
- **Libraries**: `pyyaml`, `python-dotenv`, `logging.handlers.RotatingFileHandler`.
- **Log Rotation Policy**: 5 MB maximum file size (`maxBytes=5*1024*1024`), retaining up to 3 backup archives (`backupCount=3`) with UTF-8 encoding.
- **Log Format**: `%(asctime)s [%(levelname)s] %(name)s: %(message)s`.

#### Interfaces & Data Contracts
- **Inputs**: Configuration path `config.yaml`, OS environment variables.
- **Outputs**: Dictionary/scalar values via `config.get(key_path, default=None)`.
- **External Dependencies**: Local filesystem.

#### Assumptions & Edge Case Handling
- If `config.yaml` is not found in the current working directory, the system attempts resolution relative to the project root directory before raising `FileNotFoundError`.
- Missing environment variables without specified defaults resolve to empty strings `""`.

---

### 2. Ingestion, Crawling & Chunking Engine (ETL)

#### Responsibilities & Scope
The ETL component extracts unstructured XHTML data and attachments from Atlassian Confluence, normalizes and cleans document structure, splits content into retrieval-optimized chunks, and transcribes visual/document assets. The boundary starts at the Confluence REST API and ends with chunked text structures passed to the embedding service.

#### Implemented Functional Requirements (FR)
- **`FR-ETL-01` Confluence REST Extraction & HTML Sanitization**: Fetches Confluence storage format XHTML via `APIConfluenceClient` and extracts structured text, inline links, image sources, and PDF attachment URLs using `ConfluenceHTMLParser`.
- **`FR-ETL-02` Breadth-First Search (BFS) Recursive Crawling**: Traverses page link graphs up to a configurable `max_depth` with cycle detection and domain restriction regex filtering (`RecursiveCrawler`).
- **`FR-ETL-03` Recursive Delimiter Chunking**: Slices text hierarchically through delimiters (`\n\n` -> `\n` -> `' '` -> `""`) with character overlap sliding windows (`RecursiveTextChunker`).
- **`FR-ETL-04` Semantic Boundary Chunking**: Segments text at sentence boundaries using statistical cosine distance shifts between adjacent sentence embedding windows (`SemanticChunker`).
- **`FR-ETL-05` Multimodal Asset Processing**: Transcribes `.png`/`.jpg`/`.bmp` diagrams into text descriptions and `.pdf` manuals into Markdown tables and content using `gemini-2.5-flash` before vectorization.
- **`FR-ETL-06` Asset Storage Routing**: Saves downloaded binary assets to local disk (`assets/uploaded/`) or Google Cloud Storage (`google-cloud-storage`) based on `crawler.asset_storage_type`.

#### Technical & Non-Functional Decisions (NFR)
- **Libraries**: `requests`, `html.parser`, `re`, `numpy`, `google-genai`, `google-cloud-storage`.
- **Recursive Chunking Constraints**: Default `chunk_size = 500` characters, `chunk_overlap = 50` characters.
- **Semantic Chunking Statistical Cutoffs**:
  - `percentile`: Cutoff at N-th percentile distance (default: `90.0`).
  - `standard_deviation`: Cutoff at mean + k * std_dev.
  - `interquartile`: Cutoff at Q3 + k * IQR.
  - `gradient`: Cutoff based on first-derivative anomaly rate-of-change spikes.
  - `fixed`: Absolute cosine distance threshold (0.0 to 1.0).
- **Semantic Size Guarantees**: Context buffering window (`buffer_size = 1`), minimum chunk merging threshold (`min_chunk_size = 50`), maximum chunk hard ceiling (`max_chunk_size = 2000`) with recursive fallback splitting.

#### Interfaces & Data Contracts
- **Inputs**: Confluence root page ID or URL (e.g., `123456` or `https://domain.atlassian.net/wiki/pages/viewpage.action?pageId=123456`).
- **Outputs**: Ingestion count (`int`), populated database records in table `documents`.
- **External Dependencies**: Confluence Cloud REST API `/wiki/rest/api/content/{id}`, Gemini Multimodal API (`gemini-2.5-flash`), Google Cloud Storage (optional).

#### Assumptions & Edge Case Handling
- Confluence page URLs with query parameters (`?pageId=...`) or path parameters (`/pages/...`) are automatically normalized to numeric page IDs to prevent circular duplicate crawling.
- If `SemanticChunker` runs without an active embedding service, it degrades gracefully to character-budgeted sentence grouping fallback.
- Images or PDFs that fail transcription are caught and logged with fallback placeholder text rather than aborting the transaction.

---

### 3. Embedding Generation & Vector Storage

#### Responsibilities & Scope
This component transforms raw textual chunks into dense vector representations and persists them alongside structured JSON metadata and full-text search tokens in a relational vector database. Its boundary spans the Vertex AI embedding model interface, database connection management, schema migrations, and index definitions.

#### Implemented Functional Requirements (FR)
- **`FR-VEC-01` Dense Vector Embedding Generation**: Vectorizes strings using Vertex AI `text-embedding-005` with 768 output dimensions and `RETRIEVAL_QUERY`/`RETRIEVAL_DOCUMENT` task types (`VertexEmbeddingService`).
- **`FR-VEC-02` Hybrid Document Schema with Auto-Generated FTS**: Maintains the `documents` table with generated English `tsvector` columns and JSONB metadata.
- **`FR-VEC-03` Dual Connection Engine (Cloud SQL & Direct Socket) with Connection Pooling**: Establishes pooled database connectivity via `psycopg_pool.ConnectionPool` integrated with `google-cloud-sql-connector` (IAM impersonation for `public`, `private`, or `psc` modes) or fallback direct TCP socket connection (`get_pool`, `get_connection`, `close_pool`).
- **`FR-VEC-04` Schema Versioning & Automated Repair**: Executes migration lifecycles via Alembic scripts and provides programmatic self-healing for missing tables/columns (`run_migrations.py`).

#### Technical & Non-Functional Decisions (NFR)
- **Libraries**: `psycopg` (v3), `psycopg_pool`, `pgvector`, `google-cloud-sql-connector`, `alembic`, `google-genai`.
- **Database Engine**: PostgreSQL 16+ with `vector` extension.
- **Index Definitions**:
  - Dense Vector: HNSW index on `embedding` using `vector_cosine_ops` (`idx_documents_embedding_hnsw`).
  - Sparse Keyword: GIN index on stored generated column `text_search_tsv` (`idx_documents_fts`).
- **Vector Dimensionality**: 768 float values.

#### Database Schema: `documents`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `BIGSERIAL` | `PRIMARY KEY` | Unique chunk record identifier |
| `content` | `TEXT` | `NOT NULL` | Text body of chunk |
| `metadata` | `JSONB` | | Source URL, chunk index, asset type (`confluence_scraped`, `confluence_image`, `confluence_pdf_page`) |
| `embedding` | `vector(768)` | | Dense cosine embedding vector |
| `text_search_tsv` | `tsvector` | `GENERATED ALWAYS AS (to_tsvector('english', COALESCE(content, ''))) STORED` | Stored tokenized document vector for BM25 full-text search |

#### Interfaces & Data Contracts
- **Inputs**: Plain text chunk strings, dictionary metadata.
- **Outputs**: `list[float]` (length 768), database insertion transactions.
- **External Dependencies**: Google Cloud SQL / PostgreSQL, Vertex AI GenAI Embeddings API.

#### Assumptions & Edge Case Handling
- Batch embedding failures automatically degrade to item-by-item generation to isolate malformed strings.
- Vector strings format into standard PostgreSQL bracket notation `"[0.1,0.2,...]"` before parameterized SQL execution.

---

### 4. Retrieval & Reranking Engine

#### Responsibilities & Scope
The Retrieval & Reranking component handles candidate discovery from the PostgreSQL database and applies cross-encoder reranking to score semantic relevance. Its boundary covers single-stage dense search, two-stage hybrid Reciprocal Rank Fusion (RRF), and Vertex AI Discovery Engine cross-encoder reranking.

#### Implemented Functional Requirements (FR)
- **`FR-RET-01` Pure Dense Vector Retrieval**: Queries pgvector using cosine distance operator `<=>` ordered ascending up to candidate pool limit (`vector_search`).
- **`FR-RET-02` Hybrid Search with Reciprocal Rank Fusion (RRF)**: Executes unified SQL combining Dense HNSW ranking and Sparse `ts_rank_cd` full-text search using `FULL OUTER JOIN` with RRF scoring formula:
  $$\text{RRF\_Score} = \frac{1.0}{k + \text{dense\_rank}} + \frac{1.0}{k + \text{sparse\_rank}} \quad (k = 60)$$
- **`FR-RET-03` Stage-2 Cross-Encoder Semantic Reranking**: Sends top candidate chunks along with the query to Vertex AI Semantic Ranker (`semantic-ranker-512@latest`) to eliminate false positives and produce final top-K contexts (`VertexReranker`).

#### Technical & Non-Functional Decisions (NFR)
- **Libraries**: `psycopg`, `google-cloud-discoveryengine` (`discoveryengine_v1.RankServiceClient`).
- **Retrieval Defaults**:
  - `pool_size = 5` (Candidate retrieval limit for Stage 1).
  - `final_top_k = 2` (Final context limit for Stage 2).
  - `rrf_k = 60` (Smoothing constant for Reciprocal Rank Fusion).
- **Reranker Model**: `semantic-ranker-512@latest` executed against `default_ranking_config` resource paths.

#### Interfaces & Data Contracts
- **Inputs**: `query: str`, `query_vector: list[float]`, `limit: int`, `top_n: int`.
- **Outputs**: `list[dict]` containing `{"content": str, "metadata": dict}`.
- **External Dependencies**: PostgreSQL `documents` table, Google Cloud Discovery Engine.

#### Assumptions & Edge Case Handling
- If semantic reranking is toggled off (`use_reranker=False`), the candidate pool is truncated directly via `candidates[:final_top_k]`.
- Empty candidate sets passed to `VertexReranker.rank_candidates` return an empty list immediately without issuing external API requests.

---

### 5. Synthesis & Pipeline Orchestration (Generation & Remediation)

#### Responsibilities & Scope
The Orchestration component coordinates end-to-end query processing: augmenting input queries with report snippets/attachments, executing retrieval and reranking stages, formatting dual-track grounded prompts, calling the generative LLM, and parsing hidden documentation gaps.

#### Implemented Functional Requirements (FR)
- **`FR-SYN-01` Dynamic Retrieval & Reranking Orchestration**: Orchestrates the full query lifecycle with feature toggles (`use_hybrid`, `use_reranker`, `pool_size`, `final_top_k`) (`RAGPipeline.retrieve_and_generate`).
- **`FR-SYN-02` Multimodal Document Query Augmentation**: Accepts raw bytes of uploaded diagnostic files (`.pdf`, `.png`, `.jpg`, `.txt`, `.log`), prepending plain text snippets (first 600 characters) to retrieval queries to optimize vector search.
- **`FR-SYN-03` Dual-Track Knowledge Attribution Prompting**: Constructs structured prompts strictly segregating:
  1. *Internal Documentation Guidance (Verified from Company Docs)* — strictly grounded in retrieved chunks with inline citations (`[Source: URL]`).
  2. *General Industry Best Practices (General LLM Knowledge)* — supplementary technical knowledge.
  3. *Issue Diagnosis & Summary* and *Verification & Prevention* (for attached diagnostic reports).
- **`FR-SYN-04` Silent Internal Documentation Gap Extraction**: Prompts the model to identify missing runbooks inside delimited comment tags (`<!-- DOCUMENTATION_GAPS -->...<!-- END_DOCUMENTATION_GAPS -->`), programmatically parses and extracts them, strips them from the user-facing text, and returns them in the metadata response.
- **`FR-SYN-05` Deduplicated Source Footer Generation**: Aggregates unique cited source URLs (prioritizing direct image/PDF links over parent Confluence page URLs) and appends a formatted reference list to the answer.

#### Technical & Non-Functional Decisions (NFR)
- **Libraries**: `google-genai` (SDK for Vertex AI).
- **Default LLM Model**: `gemini-2.5-flash`.
- **MIME Type Detection**: Automatic mapping for `.pdf`, `.png`, `.jpg`, `.bmp`, `.txt`, `.log`, `.json`, `.md`.

#### Interfaces & Data Contracts
- **Inputs**:
  - `query: str`
  - `use_hybrid: bool = True`
  - `use_reranker: bool = True`
  - `pool_size: int = 5`
  - `final_top_k: int = 2`
  - `attached_file_bytes: bytes | None = None`
  - `attached_filename: str | None = None`
  - `attached_mime_type: str | None = None`
- **Outputs**:
  - `user_input: str`
  - `retrieved_contexts: list[str]`
  - `sources: list[str]`
  - `response: str`
  - `attached_filename: str | None`
  - `documentation_gaps: str | None`
- **External Dependencies**: Google GenAI Client (`gemini-2.5-flash`), PostgreSQL, Discovery Engine.

#### Assumptions & Edge Case Handling
- If the model generates documentation gaps marked as `"NONE"`, `documentation_gaps` resolves to `None`.
- Supports regex extraction fallbacks for older legacy headers (`### ⚠️ Internal Documentation Gaps`).

---

### 6. Continuous Feedback, Triage & Annotation Store

#### Responsibilities & Scope
This component captures operational query telemetry, end-user ratings, root-cause failure taxonomy tags, corrected reference answers, and uploaded binary report files. It provides persistence in PostgreSQL and enables promotion of validated interactions into versioned JSON benchmark datasets.

#### Implemented Functional Requirements (FR)
- **`FR-FB-01` Feedback & Telemetry Persistence**: Records user ratings (1 for Negative, 5 for Positive), latency (ms), model configurations, retrieved chunks, issue tags, uploaded files, and parsed knowledge gaps in table `query_feedback` (`save_feedback`).
- **`FR-FB-02` Binary File Storage & Lazy Retrieval**: Persists attached incident reports as binary `BYTEA` data in PostgreSQL, exposing a dedicated lazy loader (`get_feedback_attachment`) to minimize query overhead during listing.
- **`FR-FB-03` Failure Taxonomy Aggregation**: Aggregates satisfaction rates, latency distributions, and root-cause failure tag distributions (`jsonb_array_elements_text`) via SQL (`get_feedback_analytics`).
- **`FR-FB-04` Ground-Truth Correction & Benchmark Dataset Promotion**: Allows domain experts to write ideal reference answers, update database records, and append/update `{"query": "...", "reference": "..."}` cases in `evaluation/datasets/{dataset_name}.json` (`promote_to_benchmark`).

#### Technical & Non-Functional Decisions (NFR)
- **Libraries**: `psycopg`, `json`, `datetime`.
- **Database Schema**: Table `query_feedback` with indexes on `rating` and `created_at DESC`.
- **Supported Issue Tags**:
  - `Hallucination / Ungrounded Claim`
  - `Missing Context (Recall Failure)`
  - `Poor Ranking (Precision Failure)`
  - `Vague / Incomplete Answer`
  - `Incorrect Remediation Guidance`
  - `Formatting / Tone Issue`

#### Database Table Schema: `query_feedback`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `BIGSERIAL` | `PRIMARY KEY` | Unique feedback identifier |
| `created_at` | `TIMESTAMPTZ` | `DEFAULT NOW()` | Execution timestamp |
| `query` | `TEXT` | `NOT NULL` | Input user query |
| `response` | `TEXT` | `NOT NULL` | Generated response text |
| `retrieved_contexts` | `JSONB` | `NOT NULL` | Array of retrieved chunk texts |
| `sources` | `JSONB` | `NOT NULL` | Array of cited URLs |
| `use_hybrid` / `use_reranker` | `BOOLEAN` | `NOT NULL` | Pipeline execution toggles |
| `pool_size` / `final_top_k` | `INTEGER` | `NOT NULL` | Parameter settings |
| `generation_model` | `TEXT` | `NOT NULL` | Model name identifier |
| `latency_ms` | `INTEGER` | | End-to-end execution duration |
| `rating` | `INTEGER` | `NOT NULL` | Rating value (5 = 👍 Good, 1 = 👎 Bad) |
| `issue_tags` | `JSONB` | `DEFAULT '[]'::jsonb` | Taxonomy tags |
| `corrected_reference` | `TEXT` | | Human-curated ground-truth answer |
| `user_comment` | `TEXT` | | Optional notes |
| `is_promoted_to_benchmark` | `BOOLEAN` | `DEFAULT FALSE` | Export status |
| `attached_filename` | `TEXT` | | Uploaded document filename |
| `attached_file_data` | `BYTEA` | | Raw binary bytes of uploaded report |
| `attached_file_mime` | `TEXT` | | Document MIME type |
| `documentation_gaps` | `TEXT` | | Extracted internal documentation gaps |

---

### 7. Evaluation & Quality Assurance (Ragas Benchmark)

#### Responsibilities & Scope
The Evaluation component automates regression testing and metric scoring of RAG pipeline variations against curated benchmark datasets. Its boundary covers batch query execution, Ragas metric computation via Vertex AI judge models, aggregated KPI calculation, and historical run persistence.

#### Implemented Functional Requirements (FR)
- **`FR-EVL-01` Automated Ragas Metric Evaluation**: Measures four standard RAG metrics across dataset test cases:
  1. `faithfulness` — Factual consistency of response against context.
  2. `answer_relevancy` — Directness and conciseness with respect to the query.
  3. `context_precision` — Ratio of relevant retrieved chunks ranked at the top.
  4. `context_recall` — Coverage of expected ground-truth facts within retrieved chunks.
- **`FR-EVL-02` Vertex AI LLM & Embedding Judge Wrappers**: Wraps `ChatVertexAI(model_name="gemini-2.5-flash")` and `VertexAIEmbeddings(model_name="text-embedding-005")` with LangChain and Ragas wrapper abstractions (`evaluate_ragas.py`).
- **`FR-EVL-03` Dataset Discovery & Loading**: Automatically discovers and parses JSON benchmark files from `evaluation/datasets/` with fallback to embedded default test cases.
- **`FR-EVL-04` Evaluation History Persistence & Comparison**: Stores full run parameters, aggregated averages, and per-question score breakdowns in table `evaluation_runs` (`save_evaluation_run`).

#### Technical & Non-Functional Decisions (NFR)
- **Libraries**: `ragas`, `datasets` (HuggingFace), `langchain-google-vertexai`, `pandas`, `psycopg`.
- **Judge Models**: `gemini-2.5-flash` (LLM Judge), `text-embedding-005` (Embedding Judge).
- **Target Metric Production Thresholds**:
  - `faithfulness`: Good >= 0.90, Acceptable >= 0.70, Poor < 0.70.
  - `answer_relevancy`: Good >= 0.85, Acceptable >= 0.70, Poor < 0.70.
  - `context_precision`: Good >= 0.80, Acceptable >= 0.60, Poor < 0.60.
  - `context_recall`: Good >= 0.80, Acceptable >= 0.60, Poor < 0.60.

#### Database Table Schema: `evaluation_runs`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `BIGSERIAL` | `PRIMARY KEY` | Unique evaluation run identifier |
| `run_name` | `TEXT` | | Human-readable run name |
| `created_at` | `TIMESTAMPTZ` | `DEFAULT NOW()` | Execution timestamp |
| `use_hybrid` / `use_reranker` | `BOOLEAN` | `NOT NULL` | Pipeline configuration snapshot |
| `pool_size` / `final_top_k` | `INTEGER` | `NOT NULL` | Retrieval depth parameters |
| `chunk_size` / `chunk_overlap` | `INTEGER` | `NOT NULL` | Chunking parameters snapshot |
| `embedding_model` / `rerank_model` / `generation_model` | `TEXT` | `NOT NULL` | Models used during execution |
| `dataset_name` | `TEXT` | `NOT NULL` | Target benchmark dataset identifier |
| `dataset_cases` | `JSONB` | `NOT NULL` | Benchmark questions & references snapshot |
| `avg_faithfulness` / `avg_answer_relevancy` / `avg_context_precision` / `avg_context_recall` | `FLOAT` | | Aggregated mean metric scores |
| `detailed_results` | `JSONB` | `NOT NULL` | Per-question outputs, contexts, and metric scores |
| `notes` | `TEXT` | | User execution notes |

---

### 8. User Interface & Administrative Dashboard

#### Responsibilities & Scope
The UI component provides an interactive Streamlit-based web control plane for managing ETL operations, visually debugging chunking strategies, exploring database tables, executing Ragas evaluations, and testing the live RAG playground. Its boundary is defined by `admin_app.py` and modular sub-tabs in `src/ui/tabs/`.

#### Implemented Functional Requirements (FR)
- **`FR-UI-01` Interactive Chunking Playground & Transition Visualizer**: Slices sample or pasted Confluence text with real-time parameter tuning, rendering side-by-side Recursive vs. Semantic comparisons and sentence-to-sentence cosine distance line charts (`tab_etl.py`).
- **`FR-UI-02` Database Table Inspector & Vector Dimensionality Reduction**: Browses any database table with dynamic column selection, text search, and 2D vector space scatter plots using PCA or t-SNE (`tab_db.py`).
- **`FR-UI-03` Interactive Evaluation Dashboard & Run Diff Engine**: Triggers Ragas benchmark runs with live progress feedback, metric quality icons (🟢/🟡/🔴), and side-by-side run delta comparisons (`tab_eval.py`).
- **`FR-UI-04` Playground with Embedded In-Browser Document Previews**: Supports diagnostic file uploads (`.pdf`, `.png`, `.jpg`, `.txt`, `.log`), renders in-browser Base64 PDF iframes and image viewers, captures feedback ratings and root-cause tags, and promotes corrected answers directly to JSON benchmarks (`tab_playground.py`).

#### Technical & Non-Functional Decisions (NFR)
- **Libraries**: `streamlit`, `pandas`, `scikit-learn` (`PCA`, `TSNE`), `base64`, `pypdf`.
- **UI Architecture**: Multi-tab layout (`⚙️ ETL Pipeline`, `🗄️ Database Explorer`, `📊 Evaluation`, `💬 Playground & Feedback`) with `st.session_state` management.

---

## 9. Technical Debt & Implementation Gaps

| Area | Current Implementation | Limitation / Technical Debt | Recommended Remediation |
| :--- | :--- | :--- | :--- |
| **Connection Pooling** | `psycopg_pool.ConnectionPool` managing dynamic pool sizes for Cloud SQL & TCP socket connections (`get_pool`, `get_connection`). | None (Implemented). | Parameterized in `config.yaml` (`min_pool_size`, `max_pool_size`, `pool_timeout`, etc.). |
| **Embedding Task Types** | `VertexEmbeddingService` hardcodes `task_type="RETRIEVAL_QUERY"`. | Ingestion chunks vectorized during ETL ideally require `RETRIEVAL_DOCUMENT` to optimize asymmetric cosine distance scoring in `text-embedding-005`. | Add a `task_type` parameter to `get_dense_embedding(text, task_type="RETRIEVAL_QUERY")`. |
| **Asynchronous I/O** | Confluence scraping, asset downloads, and Gemini API calls execute synchronously. | Ingestion of large Confluence spaces with many attachments blocks worker threads sequentially. | Refactor crawling and batch processing using `asyncio` and `httpx`/`aiohttp`. |
| **Streaming Generation** | Real-time token streaming via `client.models.generate_content_stream` and `st.write_stream` in Streamlit Playground and Chat UI (`tab_playground.py`, `chat_app.py`). | None (Implemented). | Perceived time-to-first-token latency minimized with dynamic stream rendering and asynchronous SSE API endpoints. |
| **Confluence Auth** | Confluence client uses Basic HTTP authentication (`username`, `api_token`). | Incompatible with enterprise OAuth 2.0 (3LO) or Atlassian Connect app authentication standards. | Add OAuth 2.0 Bearer token authorization support. |

---


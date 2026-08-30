# Confluence ETL Pipeline & Continuous Feedback Loop

This directory contains the high-performance **Extract, Transform, and Load (ETL)** pipeline that crawls Confluence spaces asynchronously using `asyncio` and `httpx`, parses multimodal content (structured text, images, PDF documents), performs batch vector embeddings with Vertex AI, and indexes them into PostgreSQL (`pgvector`).

It also defines how **user feedback** from the interactive Playground feeds back into tuning ETL chunking, crawler depth, and automated regression datasets.

---

## 1. ETL Architecture & Component Overview

| Module | Responsibility | Key Classes / Functions |
| :--- | :--- | :--- |
| [`confluence.py`](file:///Users/ilja/DEV/AI/src/etl/confluence.py) | Asynchronous Confluence REST client (`httpx.AsyncClient`) with connection pooling, authentication, HTML/XHTML parsing, and concurrent BFS crawler (`asyncio.gather`, `asyncio.Semaphore`). | `BaseConfluenceClient`, `APIConfluenceClient`, `RecursiveCrawler`, `ConfluenceHTMLParser` |
| [`chunking.py`](file:///Users/ilja/DEV/AI/src/etl/chunking.py) | Configurable text chunking strategies: **Recursive** delimiter splitting and **Semantic** embedding distance breakpoint detection. | `RecursiveTextChunker`, `SemanticChunker`, `get_chunker` |
| [`pipeline.py`](file:///Users/ilja/DEV/AI/src/etl/pipeline.py) | End-to-end async orchestrator: concurrent page crawling, concurrent image & PDF asset downloading/transcription, batch vector embedding, and transactional database ingestion. | `ConfluenceETLPipeline` |

### Multimodal Ingestion Flow

```mermaid
graph TD
    A["Confluence Storage XHTML<br/>(APIConfluenceClient with httpx)"] --> B["Concurrent Async BFS Crawler<br/>(max_depth=1..5, semaphore=N)"]
    B --> C1["Structured Text Chunks"]
    B --> C2["Attached Images (.png, .jpg, .bmp)<br/>(Async Download)"]
    B --> C3["Attached PDFs (.pdf)<br/>(Async Download)"]
    
    C1 --> D1{"Configurable Chunker<br/>(get_chunker)"}
    C2 --> D2["Gemini 2.5 Flash<br/>(Visual Layout & Diagram Transcription)"]
    C3 --> D3["Gemini 2.5 Flash<br/>(Markdown Document Transcription)"]
    
    D2 --> D1
    D3 --> D1
    
    D1 -->|Recursive Strategy| D1a["RecursiveTextChunker<br/>(chunk_size=500, overlap=50)"]
    D1 -->|Semantic Strategy| D1b["SemanticChunker<br/>(Embedding Cosine Breakpoints)"]
    
    D1a --> E["Batch Embedding Service<br/>(get_dense_embeddings, batch_size=20)"]
    D1b --> E
    E --> F[("PostgreSQL (pgvector + tsvector)<br/>'documents' table (executemany transaction)")]
```

---

## 2. Closing the Loop: Utilizing User Feedback for ETL & Retrieval Tuning

In production RAG systems, user feedback collected via the **Playground & Feedback** tab directly drives continuous optimization of the ETL and retrieval pipeline:

```mermaid
graph TD
    A[User Query & Output] --> B[User Rating & Failure Tagging]
    B --> C{Failure Classification}
    
    C -->|🔍 Wrong / Missing Context| D["ETL & Retrieval Tuning<br/>• Adjust chunk_size & overlap<br/>• Increase crawl depth<br/>• Tune Hybrid Search RRF constant<br/>• Adjust pool_size"]
    C -->|❓ Knowledge Gap| E["Content Backlog<br/>• Information missing from Confluence<br/>• Ingest additional root pages"]
    C -->|🤥 Hallucination / Factually Wrong| F["Prompt & Generator Tuning<br/>• Tighten system prompt<br/>• Lower temperature"]
    
    B --> G["Gold Benchmark Expansion<br/>(Promote user corrections to evaluation/datasets/)"]
    G --> H["Continuous CI/CD Evaluation<br/>(Ragas automated regression suite)"]
```

### Actionable Troubleshooting Matrix

| Failure Mode Tag | Root Cause | Action in ETL & Pipeline |
| :--- | :--- | :--- |
| **🔍 Missing / Irrelevant Context** | Chunks too small (fragmented ideas) or too large (diluted vector similarity). | • Adjust `pipeline.chunk_size` (e.g. 500 $\rightarrow$ 800) and `pipeline.chunk_overlap` in [`config.yaml`](file:///Users/ilja/DEV/AI/config.yaml).<br/>• Increase `pool_size` in Stage 1 retrieval.<br/>• Check if crawler missed subpages (increase `max_depth`). |
| **❓ Knowledge Gap** | Content does not exist in the database. | • Run ETL on the missing Confluence parent page ID.<br/>• Add the domain/URL to `crawler.allowed_domain_pattern`. |
| **🤥 Hallucination / Wrong Facts** | Retriever succeeded, but LLM ignored context or hallucinated. | • Refine prompt in [`orchestrator.py`](file:///Users/ilja/DEV/AI/src/pipeline/orchestrator.py) to mandate strict adherence to retrieved text.<br/>• Enable or tune Semantic Reranker to remove borderline distracting chunks. |
| **🗣️ Bad Formatting / Incomplete** | Generation parameters or context window limits. | • Increase `final_top_k` or update output formatting guidelines in system prompt. |

---

## 3. Continuous Test Dataset Curation

When domain experts test queries in the Playground and provide a **Corrected Reference Answer**:
1. The feedback is persisted in PostgreSQL (`query_feedback` table).
2. Clicking **"⭐ Promote to Benchmark Dataset"** exports the `(query, reference)` pair directly into `evaluation/datasets/default.json`.
3. Future automated evaluations run with `evaluate_ragas.py` test these exact queries to ensure previous regressions never reappear.

---

## 4. How to Run the ETL Pipeline

### Option A: Via Streamlit Admin UI (Recommended)
1. Launch the dashboard:
   ```bash
   python3 -m streamlit run admin_app.py
   ```
2. Navigate to the **⚙️ ETL Pipeline** tab.
3. Enter the Confluence Page ID or URL, adjust crawl depth and chunking parameters, and click **🚀 Run ETL Pipeline**. Live logs will stream in the expander.

### Option B: Via Command-Line Script
```bash
# Ingest with default strategy and concurrency (from config.yaml)
python3 scripts/run_etl.py 123456 --max-depth 2

# Ingest with customized concurrency and batch size
python3 scripts/run_etl.py 123456 --max-depth 2 --concurrency 10 --batch-size 50

# Ingest specifically using Semantic Chunking
python3 scripts/run_etl.py 123456 --max-depth 2 --strategy semantic
```

# RAG Pipeline Architecture

This document describes the modular object-oriented design of the RAG pipeline.

## Class Responsibility Table

| Class / Component | Responsibility | Key Methods / API |
| :--- | :--- | :--- |
| **`VertexEmbeddingService`** | Generates 768-dimensional dense vector representation of input text using Vertex AI `text-embedding-005`. | `get_dense_embedding(text)` |
| **`PostgresRetriever`** | Queries PostgreSQL for matching document chunks. Supports pure vector search (pgvector) and hybrid sparse/dense search merged using Reciprocal Rank Fusion (RRF). | `vector_search()`, `hybrid_search_rrf()` |
| **`VertexReranker`** | Reranks Stage 1 candidates using cross-encoder scoring (Vertex AI Semantic Ranker) to eliminate false positives. | `rank_candidates()` |
| **`RAGPipeline`** | Orchestrates the end-to-end RAG flow: embeds the query, retrieves candidates, reranks them, constructs the context prompt, and generates the final answer. | `retrieve_and_generate()` |
| **`db` (Module helper)** | Manages database connections dynamically, using the native `google-cloud-sql-connector` if configured or falling back to TCP socket connections. | `get_connection()` |

---

## Class Interaction Diagram

This sequence diagram illustrates how components interact during a single query lifecycle when `RAGPipeline.retrieve_and_generate(...)` is invoked:

```mermaid
sequenceDiagram
    autonumber
    actor User as Client / Script
    participant Pipeline as RAGPipeline
    participant Embedder as VertexEmbeddingService
    participant Retriever as PostgresRetriever
    participant DB as Postgres Database
    participant Reranker as VertexReranker
    participant Gemini as Gemini Client (Vertex AI)

    User->>Pipeline: retrieve_and_generate(query, use_hybrid, use_reranker)
    activate Pipeline
    
    Pipeline->>Embedder: get_dense_embedding(query)
    activate Embedder
    Embedder-->>Pipeline: query_vector (list[float])
    deactivate Embedder

    alt use_hybrid is True
        Pipeline->>Retriever: hybrid_search_rrf(query, query_vector, pool_size)
        activate Retriever
        Retriever->>DB: Execute hybrid SQL (FTS + pgvector)
        DB-->>Retriever: results
        Retriever-->>Pipeline: candidates (list[str])
        deactivate Retriever
    else use_hybrid is False
        Pipeline->>Retriever: vector_search(query_vector, pool_size)
        activate Retriever
        Retriever->>DB: SELECT ... ORDER BY embedding <=> vector
        DB-->>Retriever: results
        Retriever-->>Pipeline: candidates (list[str])
        deactivate Retriever
    end

    alt use_reranker is True
        Pipeline->>Reranker: rank_candidates(query, candidates, top_n)
        activate Reranker
        Reranker-->>Pipeline: retrieved_contexts (list[str])
        deactivate Reranker
    else use_reranker is False
        Pipeline-->>Pipeline: Use top N candidates directly
    end

    Pipeline->>Gemini: models.generate_content(model, prompt with contexts)
    activate Gemini
    Gemini-->>Pipeline: gen_res (text)
    deactivate Gemini

    Pipeline-->>User: result dict (user_input, retrieved_contexts, response)
    deactivate Pipeline
```


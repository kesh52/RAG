from unittest.mock import MagicMock
import pytest
from src.pipeline import (
    VertexEmbeddingService,
    PostgresRetriever,
    VertexReranker,
    RAGPipeline
)

def test_vertex_embedding_service():
    """Verify that VertexEmbeddingService successfully calls the Vertex AI API and processes the response."""
    mock_values = [0.1] * 768
    mock_emb = MagicMock()
    mock_emb.values = mock_values

    mock_res = MagicMock()
    mock_res.embeddings = [mock_emb]

    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = mock_res

    service = VertexEmbeddingService(mock_client)
    emb = service.get_dense_embedding("test query")
    
    assert len(emb) == 768
    assert emb[0] == 0.1
    mock_client.models.embed_content.assert_called_once()


def test_postgres_retriever_vector_search():
    """Verify that vector_search formats query vectors and issues the correct similarity search SQL statement."""
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        ("context_chunk_1", {"source_url": "url_1"}),
        ("context_chunk_2", {"source_url": "url_2"})
    ]

    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    mock_conn_factory = MagicMock()
    mock_conn_factory.return_value.__enter__.return_value = mock_conn

    retriever = PostgresRetriever(mock_conn_factory)
    results = retriever.vector_search([0.2] * 768, limit=2)

    assert results == [
        {"content": "context_chunk_1", "metadata": {"source_url": "url_1"}},
        {"content": "context_chunk_2", "metadata": {"source_url": "url_2"}}
    ]
    mock_cursor.execute.assert_called_once()
    executed_sql = mock_cursor.execute.call_args[0][0]
    assert "ORDER BY embedding <=> %s::vector" in executed_sql


def test_postgres_retriever_hybrid_search_rrf():
    """Verify that hybrid_search_rrf executes the combined sparse-dense Reciprocal Rank Fusion (RRF) SQL command."""
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        ("hybrid_chunk_1", {"source_url": "url_1"}),
        ("hybrid_chunk_2", {"source_url": "url_2"})
    ]

    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    mock_conn_factory = MagicMock()
    mock_conn_factory.return_value.__enter__.return_value = mock_conn

    retriever = PostgresRetriever(mock_conn_factory)
    results = retriever.hybrid_search_rrf("query_text", [0.3] * 768, limit=5, rrf_k=60)

    assert results == [
        {"content": "hybrid_chunk_1", "metadata": {"source_url": "url_1"}},
        {"content": "hybrid_chunk_2", "metadata": {"source_url": "url_2"}}
    ]
    mock_cursor.execute.assert_called_once()
    executed_sql = mock_cursor.execute.call_args[0][0]
    assert "COALESCE(1.0 / (%s + d.dense_rank)" in executed_sql


def test_vertex_reranker():
    """Verify that VertexReranker correctly prepares RankingRecord objects and handles successful API responses."""
    mock_record = MagicMock()
    mock_record.id = "0"
    mock_record.content = "candidate_1"

    mock_response = MagicMock()
    mock_response.records = [mock_record]

    mock_rank_client = MagicMock()
    mock_rank_client.ranking_config_path.return_value = "projects/mock-project/locations/global/rankingConfigs/default_ranking_config"
    mock_rank_client.rank.return_value = mock_response

    reranker = VertexReranker(mock_rank_client, project="mock-project", location="global")
    candidates = [
        {"content": "candidate_1", "metadata": {"source_url": "url_1"}},
        {"content": "candidate_2", "metadata": {"source_url": "url_2"}}
    ]
    results = reranker.rank_candidates("query", candidates, top_n=1)

    assert results == [
        {"content": "candidate_1", "metadata": {"source_url": "url_1"}}
    ]
    mock_rank_client.rank.assert_called_once()


def test_vertex_reranker_empty_candidates():
    """Verify that VertexReranker returns early without triggering an API request when candidate list is empty."""
    mock_rank_client = MagicMock()
    reranker = VertexReranker(mock_rank_client, project="mock-project")
    results = reranker.rank_candidates("query", [], top_n=5)
    
    assert results == []
    mock_rank_client.rank.assert_not_called()


def test_rag_pipeline_execution():
    """Verify end-to-end RAGPipeline orchestration workflows under different configuration feature flags."""
    mock_emb_service = MagicMock()
    mock_emb_service.get_dense_embedding.return_value = [0.4] * 768

    mock_retriever = MagicMock()
    mock_retriever.hybrid_search_rrf.return_value = [
        {"content": "candidate_a", "metadata": {"source_url": "url_a"}},
        {"content": "candidate_b", "metadata": {"source_url": "url_b"}}
    ]
    mock_retriever.vector_search.return_value = [
        {"content": "candidate_c", "metadata": {"source_url": "url_c"}}
    ]

    mock_reranker = MagicMock()
    mock_reranker.rank_candidates.return_value = [
        {"content": "candidate_a", "metadata": {"source_url": "url_a"}}
    ]

    mock_gen_res = MagicMock()
    mock_gen_res.text = "Mock pipeline response text"

    mock_gen_client = MagicMock()
    mock_gen_client.models.generate_content.return_value = mock_gen_res

    pipeline = RAGPipeline(
        embedding_service=mock_emb_service,
        retriever=mock_retriever,
        reranker=mock_reranker,
        generator_client=mock_gen_client,
        generator_model="mock-gemini"
    )

    # Path 1: Hybrid search and Semantic Reranker both enabled
    res = pipeline.retrieve_and_generate("user query", use_hybrid=True, use_reranker=True)
    assert res["response"] == "Mock pipeline response text\n\nSources:\n- url_a"
    assert res["retrieved_contexts"] == ["candidate_a"]
    assert res["sources"] == ["url_a"]
    
    mock_emb_service.get_dense_embedding.assert_called_once_with("user query")
    mock_retriever.hybrid_search_rrf.assert_called_once()
    mock_reranker.rank_candidates.assert_called_once()

    # Path 2: Standard vector search enabled, reranker disabled
    mock_emb_service.get_dense_embedding.reset_mock()
    mock_retriever.vector_search.reset_mock()
    mock_reranker.rank_candidates.reset_mock()

    res = pipeline.retrieve_and_generate("another query", use_hybrid=False, use_reranker=False, pool_size=5, final_top_k=1)
    assert res["response"] == "Mock pipeline response text\n\nSources:\n- url_c"
    assert res["retrieved_contexts"] == ["candidate_c"]
    assert res["sources"] == ["url_c"]
    
    mock_retriever.vector_search.assert_called_once()
    mock_reranker.rank_candidates.assert_not_called()


def test_rag_pipeline_source_link_priority():
    """Verify that retrieve_and_generate prioritizes image_url and pdf_url over source_url."""
    mock_emb_service = MagicMock()
    mock_emb_service.get_dense_embedding.return_value = [0.4] * 768

    mock_retriever = MagicMock()
    # Mock documents returned having image_url and pdf_url respectively
    mock_retriever.vector_search.return_value = [
        {"content": "chunk_img", "metadata": {"source_url": "page_url", "image_url": "file_url_img.png"}},
        {"content": "chunk_pdf", "metadata": {"source_url": "page_url", "pdf_url": "file_url_doc.pdf"}}
    ]

    mock_reranker = MagicMock()
    mock_gen_res = MagicMock()
    mock_gen_res.text = "Mock response"
    mock_gen_client = MagicMock()
    mock_gen_client.models.generate_content.return_value = mock_gen_res

    pipeline = RAGPipeline(
        embedding_service=mock_emb_service,
        retriever=mock_retriever,
        reranker=mock_reranker,
        generator_client=mock_gen_client,
        generator_model="mock-gemini"
    )

    res = pipeline.retrieve_and_generate("query", use_hybrid=False, use_reranker=False, final_top_k=2)
    
    # Assert that image_url and pdf_url were collected instead of page_url
    assert res["sources"] == ["file_url_img.png", "file_url_doc.pdf"]
    assert res["response"] == "Mock response\n\nSources:\n- file_url_img.png\n- file_url_doc.pdf"


def test_rag_pipeline_multimodal_file_attachment():
    """Verify that retrieve_and_generate properly passes attached file parts to Gemini for remediation workflows."""
    mock_emb_service = MagicMock()
    mock_emb_service.get_dense_embedding.return_value = [0.5] * 768

    mock_retriever = MagicMock()
    mock_retriever.hybrid_search_rrf.return_value = [
        {"content": "Internal Runbook: Fix Cloud SQL SSL errors", "metadata": {"source_url": "https://wiki/runbook-sql"}}
    ]

    mock_reranker = MagicMock()
    mock_reranker.rank_candidates.return_value = [
        {"content": "Internal Runbook: Fix Cloud SQL SSL errors", "metadata": {"source_url": "https://wiki/runbook-sql"}}
    ]

    mock_gen_res = MagicMock()
    mock_gen_res.text = "1. Diagnosis: SSL certificate failure.\n2. Remediation Plan: Update TLS certs."
    mock_gen_client = MagicMock()
    mock_gen_client.models.generate_content.return_value = mock_gen_res

    pipeline = RAGPipeline(
        embedding_service=mock_emb_service,
        retriever=mock_retriever,
        reranker=mock_reranker,
        generator_client=mock_gen_client,
        generator_model="gemini-2.5-flash"
    )

    pdf_mock_bytes = b"%PDF-1.4 mock audit report"
    res = pipeline.retrieve_and_generate(
        query="How to remediate findings in this report?",
        use_hybrid=True,
        use_reranker=True,
        attached_file_bytes=pdf_mock_bytes,
        attached_filename="security_audit_report.pdf",
    )

    assert "1. Diagnosis" in res["response"]
    assert res["attached_filename"] == "security_audit_report.pdf"
    assert res["sources"] == ["https://wiki/runbook-sql"]

    # Verify generate_content received multimodal contents list with Part
    mock_gen_client.models.generate_content.assert_called_once()
    call_args = mock_gen_client.models.generate_content.call_args[1]
    assert isinstance(call_args["contents"], list)
    assert len(call_args["contents"]) == 2


def test_rag_pipeline_documentation_gaps_extraction():
    """Verify that documentation gaps are silently extracted from the generated response."""
    mock_emb_service = MagicMock()
    mock_emb_service.get_dense_embedding.return_value = [0.1] * 768

    mock_retriever = MagicMock()
    mock_retriever.vector_search.return_value = [
        {"content": "Standard SOP for Redis.", "metadata": {"source_url": "https://wiki/redis"}}
    ]

    mock_gen_res = MagicMock()
    mock_gen_res.text = """### 1. 📌 Diagnosis
Redis cluster memory high.

### 2. 📚 Internal Runbook Guidance
Increase maxmemory policy to volatile-lru.

<!-- DOCUMENTATION_GAPS -->
Missing internal runbook for Redis Sentinel automated failover procedures.
<!-- END_DOCUMENTATION_GAPS -->"""

    mock_gen_client = MagicMock()
    mock_gen_client.models.generate_content.return_value = mock_gen_res

    pipeline = RAGPipeline(
        embedding_service=mock_emb_service,
        retriever=mock_retriever,
        reranker=MagicMock(),
        generator_client=mock_gen_client,
        generator_model="gemini-2.5-flash"
    )

    res = pipeline.retrieve_and_generate("How to configure Redis failover?", use_hybrid=False, use_reranker=False)

    # Verify that the user-facing response does NOT contain the gaps tag
    assert "<!-- DOCUMENTATION_GAPS -->" not in res["response"]
    assert "Missing internal runbook for Redis Sentinel" not in res["response"]
    # Verify that the gap was silently extracted into the documentation_gaps key
    assert res["documentation_gaps"] == "Missing internal runbook for Redis Sentinel automated failover procedures."
    assert "### 1. 📌 Diagnosis" in res["response"]


def test_rag_pipeline_retrieve_and_generate_stream():
    """Verify that retrieve_and_generate_stream invokes generate_content_stream and returns streaming iterator."""
    mock_emb_service = MagicMock()
    mock_emb_service.get_dense_embedding.return_value = [0.2] * 768

    mock_retriever = MagicMock()
    mock_retriever.hybrid_search_rrf.return_value = [
        {"content": "Streaming runbook chunk", "metadata": {"source_url": "https://wiki/stream"}}
    ]

    mock_reranker = MagicMock()
    mock_reranker.rank_candidates.return_value = [
        {"content": "Streaming runbook chunk", "metadata": {"source_url": "https://wiki/stream"}}
    ]

    # Mock chunk stream
    mock_chunk1 = MagicMock()
    mock_chunk1.text = "Token 1 "
    mock_chunk2 = MagicMock()
    mock_chunk2.text = "Token 2"
    mock_stream = iter([mock_chunk1, mock_chunk2])

    mock_gen_client = MagicMock()
    mock_gen_client.models.generate_content_stream.return_value = mock_stream

    pipeline = RAGPipeline(
        embedding_service=mock_emb_service,
        retriever=mock_retriever,
        reranker=mock_reranker,
        generator_client=mock_gen_client,
        generator_model="gemini-2.5-flash",
    )

    stream_res, retrieved_contexts, sources = pipeline.retrieve_and_generate_stream(
        query="Streaming test query",
        use_hybrid=True,
        use_reranker=True,
    )

    assert retrieved_contexts == [{"content": "Streaming runbook chunk", "metadata": {"source_url": "https://wiki/stream"}}]
    assert sources == ["https://wiki/stream"]
    mock_gen_client.models.generate_content_stream.assert_called_once()

    # Verify iteration over stream chunks
    tokens = [c.text for c in stream_res]
    assert tokens == ["Token 1 ", "Token 2"]

    # Verify response parsing
    full_text = "".join(tokens)
    final_text, gaps = pipeline.parse_response_text(full_text, sources)
    assert "Token 1 Token 2" in final_text
    assert "Sources:\n- https://wiki/stream" in final_text




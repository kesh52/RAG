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

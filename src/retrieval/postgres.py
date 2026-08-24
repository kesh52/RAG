import logging
from src.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)

class PostgresRetriever(BaseRetriever):
    """Service to retrieve candidate documents from PostgreSQL using pgvector and RRF."""
    
    def __init__(self, db_conn_factory):
        self.db_conn_factory = db_conn_factory

    def vector_search(self, query_vector: list[float], limit: int = 2) -> list[str]:
        """Stage 1: Pure Dense Vector Retrieval (pgvector)."""
        logger.debug(f"Executing vector similarity search (limit={limit})")
        vector_str = f"[{','.join(str(x) for x in query_vector)}]"
        with self.db_conn_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT content FROM documents
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (vector_str, limit),
                )
                results = [row[0] for row in cur.fetchall()]
                logger.debug(f"Vector similarity search retrieved {len(results)} candidate chunks")
                return results

    def hybrid_search_rrf(self, query: str, query_vector: list[float], limit: int = 10, rrf_k: int = 60) -> list[str]:
        """Stage 1: Dense Vector + Sparse FTS Keyword Retrieval merged using Reciprocal Rank Fusion (RRF)."""
        logger.debug(f"Executing sparse-dense hybrid search with RRF (limit={limit}, rrf_k={rrf_k})")
        vector_str = f"[{','.join(str(x) for x in query_vector)}]"
        hybrid_query = """
        WITH dense_search AS (
            SELECT id, content, ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS dense_rank
            FROM documents
            LIMIT %s
        ),
        sparse_search AS (
            SELECT id, content, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(text_search_tsv, plainto_tsquery('english', %s)) DESC) AS sparse_rank
            FROM documents
            WHERE text_search_tsv @@ plainto_tsquery('english', %s)
            LIMIT %s
        )
        SELECT 
            COALESCE(d.content, s.content) AS content,
            COALESCE(1.0 / (%s + d.dense_rank), 0.0) + 
            COALESCE(1.0 / (%s + s.sparse_rank), 0.0) AS rrf_score
        FROM dense_search d
        FULL OUTER JOIN sparse_search s ON d.id = s.id
        ORDER BY rrf_score DESC
        LIMIT %s;
        """
        with self.db_conn_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    hybrid_query,
                    (
                        vector_str, limit,
                        query, query, limit,
                        rrf_k, rrf_k,
                        limit
                    ),
                )
                results = [row[0] for row in cur.fetchall()]
                logger.debug(f"Hybrid search retrieved {len(results)} merged candidate chunks")
                return results


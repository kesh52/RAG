import os
import sys
from contextlib import closing
from google.genai import types as genai_types
import psycopg
from pgvector.psycopg import register_vector
from google import genai

# Ensure root directory is in system path to resolve src imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.db as db

def generate_and_insert_embeddings():
    # Initialize Vertex AI client using the project and location config
    GCP_PROJECT = os.getenv("GCP_PROJECT")
    GCP_LOCATION = os.getenv("GCP_LOCATION")
    
    if not GCP_PROJECT or not GCP_LOCATION:
        raise ValueError("GCP_PROJECT and GCP_LOCATION environment variables must be set in the .env file.")
        
    client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)

    # --- Sample Documents ---
    documents = [
        {
            "content": "Spring Batch uses chunk-oriented processing where an ItemReader reads items one by one, an ItemProcessor transforms them, and an ItemWriter writes items in configurable batch sizes within a transaction boundary.",
            "metadata": {"category": "backend", "framework": "spring_batch"}
        },
        {
            "content": "In Spring Boot, asynchronous task execution is enabled via @EnableAsync and configured with ThreadPoolTaskExecutor for controlling core pool size, max pool size, and queue capacity.",
            "metadata": {"category": "backend", "framework": "spring_boot"}
        },
        {
            "content": "PostgreSQL with pgvector provides HNSW (Hierarchical Navigable Small World) and IVFFlat indexes. HNSW offers faster search performance and higher recall at the expense of longer index build times and higher memory consumption.",
            "metadata": {"category": "database", "engine": "postgres", "feature": "pgvector"}
        },
        {
            "content": "PostgreSQL full-text search utilizes tsvector to represent parsed documents and tsquery to represent search terms, using GIN indexes for rapid keyword filtering and ts_rank_cd for relevance scoring.",
            "metadata": {"category": "database", "engine": "postgres", "feature": "full_text_search"}
        },
        {
            "content": "Reciprocal Rank Fusion (RRF) combines ranked lists from dense vector search and sparse BM25 keyword search using formula 1 / (60 + rank), ensuring robust ranking without manual score normalization.",
            "metadata": {"category": "search", "algorithm": "hybrid_search"}
        },
        {
            "content": "Google Cloud SQL Auth Proxy establishes an encrypted local TLS tunnel to Cloud SQL instances over port 5432 using IAM credentials, avoiding the need for public IP whitelisting.",
            "metadata": {"category": "cloud", "provider": "gcp", "service": "cloud_sql"}
        },
        {
            "content": "Vertex AI text-embedding-005 outputs 768-dimensional dense vectors and supports distinct task_type configurations such as RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, and SEMANTIC_SIMILARITY.",
            "metadata": {"category": "ai", "provider": "gcp", "service": "vertex_ai"}
        },
        {
            "content": "Vertex AI Semantic Ranker (Discovery Engine) is a cross-encoder model that scores query-document pairs simultaneously to rerank candidates and eliminate topic-drift false positives from vector search.",
            "metadata": {"category": "ai", "provider": "gcp", "service": "vertex_ai"}
        },
        {
            "content": "Ragas evaluates RAG systems using Faithfulness (groundedness in context), Answer Relevancy (conciseness and directness), Context Precision (ranking accuracy), and Context Recall (ground truth alignment).",
            "metadata": {"category": "evaluation", "framework": "ragas"}
        },
        {
            "content": "In microservices architecture, the Transactional Outbox pattern guarantees at-least-once message delivery to Apache Kafka by saving events to a database table within the same ACID transaction before publishing.",
            "metadata": {"category": "architecture", "pattern": "outbox"}
        }
    ]

    # Add source_url dynamically to each mock document metadata for pipeline compat
    for idx, doc in enumerate(documents):
        doc["metadata"]["source_url"] = f"https://confluence.example.com/wiki/pages/viewpage.action?pageId={1000 + idx}"

    texts = [doc["content"] for doc in documents]

    print("Generating embeddings using text-embedding-005...")
    response = client.models.embed_content(
        model="text-embedding-005",
        contents=texts,
        config=genai_types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768
        )
    )

    embeddings = [emb.values for emb in response.embeddings]

    print("Connecting to PostgreSQL and inserting document embeddings...")
    with closing(db.get_connection()) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            insert_query = """
                INSERT INTO documents (content, metadata, embedding)
                VALUES (%s, %s, %s);
            """
            for doc, emb in zip(documents, embeddings):
                cur.execute(
                    insert_query,
                    (doc["content"], psycopg.types.json.Jsonb(doc["metadata"]), emb)
                )
        conn.commit()

    print(f"Successfully inserted {len(documents)} embeddings into PostgreSQL.")

if __name__ == "__main__":
    generate_and_insert_embeddings()


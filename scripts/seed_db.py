import os
import sys
import logging
from contextlib import closing
from google.genai import types as genai_types
import psycopg
from pgvector.psycopg import register_vector
from google import genai

# Ensure root directory is in system path to resolve src imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.db as db

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("seed_db")

def create_minimal_pdf(filepath: str, text_content: str):
    """Creates a minimal valid PDF document containing the specified text."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    stream_content = f"BT\n/F1 12 Tf\n72 712 Td\n({text_content}) Tj\nET\n"
    pdf_data = (
        "%PDF-1.4\n"
        "1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
        "2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj\n"
        "3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources <</Font <</F1 <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>>>>> >> endobj\n"
        f"4 0 obj <</Length {len(stream_content)}>> stream\n"
        f"{stream_content}"
        "endstream\nendobj\n"
        "xref\n0 5\n0000000000 65535 f\n"
        "0000000009 00000 n\n"
        "0000000056 00000 n\n"
        "0000000111 00000 n\n"
        "0000000256 00000 n\n"
        "trailer <</Size 5 /Root 1 0 R>>\n"
        "startxref\n350\n%%EOF\n"
    )
    with open(filepath, "wb") as f:
        f.write(pdf_data.encode("ascii", errors="ignore"))
    logger.info(f"Created mock PDF file: {filepath}")

def create_mock_bmp(filepath: str):
    """Writes a minimal valid 2x2 BMP image file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # 2x2 Red pixel BMP bytes
    bmp_data = (
        b'BM\x3e\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00'
        b'\x02\x00\x00\x00\x02\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00'
        b'\x08\x00\x00\x00\x12\x0b\x00\x00\x12\x0b\x00\x00\x00\x00\x00\x00'
        b'\x00\x00\x00\x00\x00\x00\xff\x00\x00\xff\x00\x00\x00\x00\x00\x00'
        b'\xff\x00\x00\xff\x00\x00\x00\x00'
    )
    with open(filepath, "wb") as f:
        f.write(bmp_data)
    logger.info(f"Created mock BMP image file: {filepath}")


def generate_and_insert_embeddings():
    # Initialize Vertex AI client using the project and location config
    GCP_PROJECT = os.getenv("GCP_PROJECT") or "woven-spring-288610"
    GCP_LOCATION = os.getenv("GCP_LOCATION") or "europe-west3"
    
    client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)

    # 1. Generate local mock assets
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    pdf_path = os.path.join(assets_dir, "spring_batch_manual_mock.pdf")
    bmp_path = os.path.join(assets_dir, "postgres_pgvector_architecture.bmp")
    
    create_minimal_pdf(pdf_path, "Spring Batch is structured around Job, Step, ItemReader, and ItemWriter elements.")
    create_mock_bmp(bmp_path)

    # 2. Text documents definitions
    documents = [
        {
            "content": "Spring Batch uses chunk-oriented processing where an ItemReader reads items one by one, an ItemProcessor transforms them, and an ItemWriter writes items in configurable batch sizes within a transaction boundary.",
            "metadata": {"category": "backend", "framework": "spring_batch", "type": "confluence_scraped"}
        },
        {
            "content": "In Spring Boot, asynchronous task execution is enabled via @EnableAsync and configured with ThreadPoolTaskExecutor for controlling core pool size, max pool size, and queue capacity.",
            "metadata": {"category": "backend", "framework": "spring_boot", "type": "confluence_scraped"}
        },
        {
            "content": "PostgreSQL with pgvector provides HNSW (Hierarchical Navigable Small World) and IVFFlat indexes. HNSW offers faster search performance and higher recall at the expense of longer index build times and higher memory consumption.",
            "metadata": {"category": "database", "engine": "postgres", "feature": "pgvector", "type": "confluence_scraped"}
        },
        {
            "content": "PostgreSQL full-text search utilizes tsvector to represent parsed documents and tsquery to represent search terms, using GIN indexes for rapid keyword filtering and ts_rank_cd for relevance scoring.",
            "metadata": {"category": "database", "engine": "postgres", "feature": "full_text_search", "type": "confluence_scraped"}
        },
        {
            "content": "Reciprocal Rank Fusion (RRF) combines ranked lists from dense vector search and sparse BM25 keyword search using formula 1 / (60 + rank), ensuring robust ranking without manual score normalization.",
            "metadata": {"category": "search", "algorithm": "hybrid_search", "type": "confluence_scraped"}
        },
        {
            "content": "Google Cloud SQL Auth Proxy establishes an encrypted local TLS tunnel to Cloud SQL instances over port 5432 using IAM credentials, avoiding the need for public IP whitelisting.",
            "metadata": {"category": "cloud", "provider": "gcp", "service": "cloud_sql", "type": "confluence_scraped"}
        },
        {
            "content": "Vertex AI text-embedding-005 outputs 768-dimensional dense vectors and supports distinct task_type configurations such as RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, and SEMANTIC_SIMILARITY.",
            "metadata": {"category": "ai", "provider": "gcp", "service": "vertex_ai", "type": "confluence_scraped"}
        },
        {
            "content": "Vertex AI Semantic Ranker (Discovery Engine) is a cross-encoder model that scores query-document pairs simultaneously to rerank candidates and eliminate topic-drift false positives from vector search.",
            "metadata": {"category": "ai", "provider": "gcp", "service": "vertex_ai", "type": "confluence_scraped"}
        },
        {
            "content": "Ragas evaluates RAG systems using Faithfulness (groundedness in context), Answer Relevancy (conciseness and directness), Context Precision (ranking accuracy), and Context Recall (ground truth alignment).",
            "metadata": {"category": "evaluation", "framework": "ragas", "type": "confluence_scraped"}
        },
        {
            "content": "In microservices architecture, the Transactional Outbox pattern guarantees at-least-once message delivery to Apache Kafka by saving events to a database table within the same ACID transaction before publishing.",
            "metadata": {"category": "architecture", "pattern": "outbox", "type": "confluence_scraped"}
        }
    ]

    # Resolve absolute paths and mock URLs
    local_pdf_url = f"file://{pdf_path}"
    local_bmp_url = f"file://{bmp_path}"

    # 3. Process PDF file
    pdf_text = "PDF Content: Spring Batch is structured around Job, Step, ItemReader, and ItemWriter elements."
    try:
        logger.info("Transcribing seeded PDF file using Gemini...")
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                genai_types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                "Extract all text from this PDF."
            ]
        )
        if response.text:
            pdf_text = f"PDF Content: {response.text.strip()}"
    except Exception as e:
        logger.warning(f"Failed calling Gemini for PDF transcription, using local mock fallback: {e}")

    documents.append({
        "content": pdf_text,
        "metadata": {
            "category": "backend",
            "framework": "spring_batch",
            "pdf_url": local_pdf_url,
            "type": "confluence_pdf_page",
            "page_number": 1
        }
    })

    # 4. Process BMP Image
    image_description = "Image Description: Technical architectural flowchart diagram showing a 2x2 database cluster layout."
    try:
        logger.info("Transcribing seeded BMP image using Gemini...")
        with open(bmp_path, "rb") as f:
            img_bytes = f.read()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                genai_types.Part.from_bytes(data=img_bytes, mime_type="image/bmp"),
                "Describe this image/diagram layout in detail."
            ]
        )
        if response.text:
            image_description = f"Image Description: {response.text.strip()}"
    except Exception as e:
        logger.warning(f"Failed calling Gemini for Image transcription, using local mock fallback: {e}")

    documents.append({
        "content": image_description,
        "metadata": {
            "category": "database",
            "engine": "postgres",
            "image_url": local_bmp_url,
            "type": "confluence_image"
        }
    })

    # Apply generic source URLs
    for idx, doc in enumerate(documents):
        doc["metadata"]["source_url"] = f"https://confluence.example.com/wiki/pages/viewpage.action?pageId={1000 + idx}"

    texts = [doc["content"] for doc in documents]

    logger.info("Generating embeddings using text-embedding-005...")
    try:
        response = client.models.embed_content(
            model="text-embedding-005",
            contents=texts,
            config=genai_types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768
            )
        )
        embeddings = [emb.values for emb in response.embeddings]
    except Exception as e:
        logger.warning(f"Google GenAI API embedding call failed. Using mock dummy vectors (for sandbox runs): {e}")
        embeddings = [[0.1] * 768 for _ in documents]

    logger.info("Connecting to PostgreSQL and inserting document embeddings...")
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

    logger.info(f"Successfully inserted {len(documents)} documents (including PDF and Image representations) into PostgreSQL.")

if __name__ == "__main__":
    generate_and_insert_embeddings()

import os
import sys
import argparse
import logging
from google import genai
from contextlib import closing

# Ensure root directory is in system path to resolve src imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import config
from src.etl.confluence import APIConfluenceClient
from src.etl.chunking import get_chunker
from src.embeddings.vertex import VertexEmbeddingService
from src.etl.pipeline import ConfluenceETLPipeline

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_etl")

def main():
    parser = argparse.ArgumentParser(description="Run the Confluence ETL pipeline on a live page ID or URL.")
    parser.add_argument("root_page_id", type=str, help="The Confluence page ID or URL to start crawling from.")
    parser.add_argument("--max-depth", type=int, default=1, help="BFS crawl depth limit (default: 1).")
    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        default=config.get("pipeline.chunking_strategy", "recursive"),
        choices=["recursive", "semantic"],
        help="Chunking strategy to use: 'recursive' or 'semantic' (default from config: %(default)s).",
    )
    args = parser.parse_args()

    # Load credentials
    domain = os.getenv("CONFLUENCE_DOMAIN")
    username = os.getenv("CONFLUENCE_USERNAME")
    api_token = os.getenv("CONFLUENCE_API_TOKEN")

    if not domain or not username or not api_token:
        logger.error(
            "Missing Confluence credentials! Please ensure CONFLUENCE_DOMAIN, "
            "CONFLUENCE_USERNAME, and CONFLUENCE_API_TOKEN are configured in your env/.env file."
        )
        sys.exit(1)

    logger.info("Initializing API Confluence Client and AI Services...")
    
    # 1. Initialize Confluence Client
    client = APIConfluenceClient(domain=domain, username=username, api_token=api_token)
    
    # 2. Initialize Vertex AI client for multimodal embeddings/transcriptions
    gcp_project = config.get("gcp.project")
    gcp_location = config.get("gcp.location")
    ai_client = genai.Client(vertexai=True, project=gcp_project, location=gcp_location)
    embedding_service = VertexEmbeddingService(client=ai_client, model_name=config.get("models.embedding"))

    # 3. Initialize Chunker based on chosen strategy
    chunker = get_chunker(strategy=args.strategy, embedding_service=embedding_service)
    logger.info(f"Using chunking strategy: '{args.strategy}' ({chunker.__class__.__name__})")

    # 4. Instantiate and execute pipeline
    pipeline = ConfluenceETLPipeline(
        confluence_client=client,
        chunker=chunker,
        embedding_service=embedding_service
    )

    logger.info(f"Running ingestion starting from Confluence target: {args.root_page_id}...")
    try:
        chunks_inserted = pipeline.run(root_identifier=args.root_page_id, max_depth=args.max_depth)
        logger.info(f"SUCCESS: Ingestion finished! Inserted {chunks_inserted} vector chunks into the database.")
    except Exception as e:
        logger.error(f"Failed running ETL pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


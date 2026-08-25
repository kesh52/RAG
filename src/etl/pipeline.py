import logging
import json
import psycopg
from contextlib import closing
from src.db import get_connection
from src.embeddings.base import BaseEmbeddingService
from src.etl.confluence import BaseConfluenceClient, RecursiveCrawler
from src.etl.chunking import BaseChunker
from src.utils.config import config

logger = logging.getLogger(__name__)

class ConfluenceETLPipeline:
    """Orchestrates crawling Confluence pages, chunking text, generating embeddings, and storing to DB."""
    
    def __init__(
        self,
        confluence_client: BaseConfluenceClient,
        chunker: BaseChunker,
        embedding_service: BaseEmbeddingService,
        db_conn_factory=get_connection
    ):
        self.client = confluence_client
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.db_conn_factory = db_conn_factory

    def run(self, root_identifier: str, max_depth: int = 2) -> int:
        """Runs the ETL process starting from the root page.

        Returns:
            The total count of chunks successfully ingested into the database.
        """
        logger.info(f"Starting ETL pipeline for root page: {root_identifier} (max depth: {max_depth})...")
        
        # 1. Extract: crawl confluence pages with domain validation matching config
        allowed_pattern = config.get("crawler.allowed_domain_pattern")
        crawler = RecursiveCrawler(self.client, max_depth=max_depth, allowed_domain_pattern=allowed_pattern)
        crawled_data = crawler.crawl(root_identifier)
        
        total_ingested = 0
        
        # 2. Transform & Load: Process and insert each page
        logger.info(f"Completed crawl. Processing {len(crawled_data)} crawled pages...")
        
        try:
            with closing(self.db_conn_factory()) as conn:
                with conn.cursor() as cur:
                    for source_id, text in crawled_data.items():
                        if not text:
                            logger.warning(f"Skipping empty text content for source: {source_id}")
                            continue
                            
                        # Chunk the text content
                        chunks = self.chunker.split_text(text)
                        logger.info(f"Page '{source_id}' split into {len(chunks)} chunks.")
                        
                        for idx, chunk in enumerate(chunks):
                            # Generate vector embedding using the embedding service
                            embedding = self.embedding_service.get_dense_embedding(chunk)
                            
                            # Construct JSON metadata with source URL resolved
                            source_url = source_id
                            if source_id.isdigit():
                                base_url = getattr(self.client, "base_url", "https://confluence.localhost")
                                source_url = f"{base_url}/wiki/pages/viewpage.action?pageId={source_id}"

                            metadata = {
                                "source": source_id,
                                "source_url": source_url,
                                "chunk_index": idx,
                                "type": "confluence_scraped"
                            }
                            
                            # Execute INSERT
                            insert_query = """
                                INSERT INTO documents (content, metadata, embedding)
                                VALUES (%s, %s, %s);
                            """
                            cur.execute(
                                insert_query,
                                (chunk, psycopg.types.json.Jsonb(metadata), embedding)
                            )
                            total_ingested += 1
                            
                conn.commit()
                logger.info(f"Database transaction committed successfully! Ingested {total_ingested} chunks.")
                
        except Exception as e:
            logger.error(f"Failed executing ETL pipeline load transaction: {e}")
            raise e

        return total_ingested


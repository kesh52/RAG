import logging
import json
import psycopg
import os
from urllib.parse import urlparse
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
        """Runs the ETL process starting from the root page, parsing text, images, and PDFs.

        Returns:
            The total count of chunks successfully ingested into the database.
        """
        logger.info(f"Starting ETL pipeline for root page: {root_identifier} (max depth: {max_depth})...")
        
        # 1. Extract: crawl confluence pages with domain validation matching config
        allowed_pattern = config.get("crawler.allowed_domain_pattern")
        crawler = RecursiveCrawler(self.client, max_depth=max_depth, allowed_domain_pattern=allowed_pattern)
        crawled_data = crawler.crawl(root_identifier)
        
        total_ingested = 0
        
        # 2. Transform & Load: Process and insert page text and attachments
        logger.info(f"Completed crawl. Processing {len(crawled_data)} crawled pages...")
        
        try:
            with closing(self.db_conn_factory()) as conn:
                with conn.cursor() as cur:
                    for source_id, page_data in crawled_data.items():
                        page_text = page_data.get("text", "")
                        page_images = page_data.get("images", [])
                        page_pdfs = page_data.get("pdfs", [])

                        # 2a. Ingest normal text content
                        if page_text:
                            chunks = self.chunker.split_text(page_text)
                            logger.info(f"Page '{source_id}' split into {len(chunks)} text chunks.")
                            for idx, chunk in enumerate(chunks):
                                embedding = self.embedding_service.get_dense_embedding(chunk)
                                source_url = self._resolve_source_url(source_id)
                                metadata = {
                                    "source": source_id,
                                    "source_url": source_url,
                                    "chunk_index": idx,
                                    "type": "confluence_scraped"
                                }
                                cur.execute(
                                    """
                                    INSERT INTO documents (content, metadata, embedding)
                                    VALUES (%s, %s, %s);
                                    """,
                                    (chunk, psycopg.types.json.Jsonb(metadata), embedding)
                                )
                                total_ingested += 1

                        # 2b. Ingest images as descriptions
                        for image_url in page_images:
                            try:
                                logger.info(f"Processing image attachment: {image_url}")
                                image_bytes = self.client.download_attachment(image_url)
                                filename = os.path.basename(urlparse(image_url).path) or "image.png"
                                saved_url = self._save_asset(filename, image_bytes)
                                
                                # Transcribe image content
                                description = self._get_image_description(image_bytes, filename)
                                full_content = f"[Embedded Image: {filename} - Source URL: {saved_url}]\n{description}"
                                
                                embedding = self.embedding_service.get_dense_embedding(full_content)
                                source_url = self._resolve_source_url(source_id)
                                metadata = {
                                    "source": source_id,
                                    "source_url": source_url,
                                    "image_url": saved_url,
                                    "type": "confluence_image"
                                }
                                cur.execute(
                                    """
                                    INSERT INTO documents (content, metadata, embedding)
                                    VALUES (%s, %s, %s);
                                    """,
                                    (full_content, psycopg.types.json.Jsonb(metadata), embedding)
                                )
                                total_ingested += 1
                            except Exception as e:
                                logger.warning(f"Failed processing image '{image_url}': {e}")

                        # 2c. Ingest PDFs as chunked markdown
                        for pdf_url in page_pdfs:
                            try:
                                logger.info(f"Processing PDF attachment: {pdf_url}")
                                pdf_bytes = self.client.download_attachment(pdf_url)
                                filename = os.path.basename(urlparse(pdf_url).path) or "document.pdf"
                                saved_url = self._save_asset(filename, pdf_bytes)
                                
                                # Transcribe PDF content
                                markdown_text = self._get_pdf_markdown(pdf_bytes, filename)
                                full_pdf_content = f"[PDF Attachment: {filename} - Source URL: {saved_url}]\n{markdown_text}"
                                
                                chunks = self.chunker.split_text(full_pdf_content)
                                for idx, chunk in enumerate(chunks):
                                    embedding = self.embedding_service.get_dense_embedding(chunk)
                                    source_url = self._resolve_source_url(source_id)
                                    metadata = {
                                        "source": source_id,
                                        "source_url": source_url,
                                        "pdf_url": saved_url,
                                        "chunk_index": idx,
                                        "type": "confluence_pdf_page"
                                    }
                                    cur.execute(
                                        """
                                        INSERT INTO documents (content, metadata, embedding)
                                        VALUES (%s, %s, %s);
                                        """,
                                        (chunk, psycopg.types.json.Jsonb(metadata), embedding)
                                    )
                                    total_ingested += 1
                            except Exception as e:
                                logger.warning(f"Failed processing PDF '{pdf_url}': {e}")

                conn.commit()
                logger.info(f"Database transaction committed successfully! Ingested {total_ingested} chunks.")
                
        except Exception as e:
            logger.error(f"Failed executing ETL pipeline load transaction: {e}")
            raise e

        return total_ingested

    def _resolve_source_url(self, source_id: str) -> str:
        source_url = source_id
        if source_id.isdigit():
            base_url = getattr(self.client, "base_url", "https://confluence.localhost")
            source_url = f"{base_url}/wiki/pages/viewpage.action?pageId={source_id}"
        return source_url

    def _save_asset(self, filename: str, file_bytes: bytes) -> str:
        """Saves binary asset either to local filesystem or Google Cloud Storage based on config."""
        storage_type = config.get("crawler.asset_storage_type", "local").lower()
        filename = os.path.basename(filename)
        
        if storage_type == "gcs":
            try:
                from google.cloud import storage
                bucket_name = config.get("crawler.gcs_bucket_name", "my-rag-assets")
                logger.info(f"Uploading asset '{filename}' to GCS bucket '{bucket_name}'...")
                
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(f"confluence/{filename}")
                blob.upload_from_string(file_bytes)
                return blob.public_url
            except Exception as e:
                logger.warning(f"Failed GCS upload for '{filename}', falling back to local: {e}")
                storage_type = "local"
                
        if storage_type == "local":
            local_dir = config.get("crawler.local_assets_dir", "assets/uploaded")
            os.makedirs(local_dir, exist_ok=True)
            filepath = os.path.join(local_dir, filename)
            logger.info(f"Saving asset '{filename}' to local folder '{local_dir}'...")
            with open(filepath, "wb") as f:
                f.write(file_bytes)
            return f"file://{os.path.abspath(filepath)}"
            
        raise ValueError(f"Unknown asset_storage_type: {storage_type}")

    def _get_image_description(self, image_bytes: bytes, filename: str) -> str:
        """Queries Gemini to transcribe an image into detailed text description."""
        if not hasattr(self.embedding_service, "client") or self.embedding_service.client is None:
            logger.warning("Embedding service client not available. Skipping image description.")
            return f"[Image description skipped: {filename}]"
        try:
            from google.genai import types
            client = self.embedding_service.client
            logger.info(f"Calling Gemini 2.5 Flash to transcribe image: {filename}")
            
            # Detect mime type based on filename extension
            ext = filename.lower().split('.')[-1]
            mime_type = "image/png"
            if ext in ["jpg", "jpeg"]:
                mime_type = "image/jpeg"
            elif ext == "gif":
                mime_type = "image/gif"
                
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    "Describe this image/diagram in detail. List all labels and connections if it is a flowchart."
                ]
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed generating image description for '{filename}': {e}")
            return f"[Failed to transcribe image: {filename}]"

    def _get_pdf_markdown(self, pdf_bytes: bytes, filename: str) -> str:
        """Queries Gemini to transcribe a PDF into structured markdown."""
        if not hasattr(self.embedding_service, "client") or self.embedding_service.client is None:
            logger.warning("Embedding service client not available. Skipping PDF transcription.")
            return f"[PDF description skipped: {filename}]"
        try:
            from google.genai import types
            client = self.embedding_service.client
            logger.info(f"Calling Gemini 2.5 Flash to transcribe PDF: {filename}")
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                    "Extract all text, convert any tables into Markdown tables, and describe any diagrams in detail."
                ]
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed generating PDF markdown for '{filename}': {e}")
            return f"[Failed to transcribe PDF: {filename}]"

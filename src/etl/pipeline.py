import logging
import json
import psycopg
import os
import asyncio
import concurrent.futures
from urllib.parse import urlparse
from contextlib import closing
from src.db import get_connection
from src.embeddings.base import BaseEmbeddingService
from src.etl.confluence import BaseConfluenceClient, RecursiveCrawler, _run_sync, _resolve_coro_or_sync
from src.etl.chunking import BaseChunker
from src.utils.config import config

logger = logging.getLogger(__name__)


class ConfluenceETLPipeline:
    """Orchestrates asynchronous Confluence crawling, chunking, batch vector embeddings, and database loading."""
    
    def __init__(
        self,
        confluence_client: BaseConfluenceClient,
        chunker: BaseChunker,
        embedding_service: BaseEmbeddingService,
        db_conn_factory=get_connection,
        batch_size: int | None = None,
        max_concurrency: int | None = None,
    ):
        self.client = confluence_client
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.db_conn_factory = db_conn_factory
        
        cfg_batch = batch_size or int(config.get("crawler.batch_size", 20))
        self.batch_size = max(1, cfg_batch)
        
        cfg_concurrency = max_concurrency or int(config.get("crawler.max_concurrency", 5))
        self.max_concurrency = max(1, cfg_concurrency)

    async def run_async(self, root_identifier: str, max_depth: int = 2) -> int:
        """Asynchronously runs the ETL process starting from the root page, parsing text, images, and PDFs.

        Returns:
            The total count of chunks successfully ingested into the database.
        """
        logger.info(f"Starting async ETL pipeline for root page: {root_identifier} (max depth: {max_depth})...")
        
        # 1. Extract: Asynchronously crawl confluence pages with domain validation matching config
        allowed_pattern = config.get("crawler.allowed_domain_pattern")
        crawler = RecursiveCrawler(
            self.client,
            max_depth=max_depth,
            allowed_domain_pattern=allowed_pattern,
            max_concurrency=self.max_concurrency,
        )
        crawled_data = await crawler.crawl_async(root_identifier)
        
        if not crawled_data:
            logger.warning("No pages crawled. Exiting ETL pipeline with 0 chunks.")
            return 0
            
        logger.info(f"Completed crawl. Processing {len(crawled_data)} crawled pages in batch mode...")
        
        # 2. Transform & Ingest Multimodal Content
        pending_records: list[tuple[str, dict]] = []
        semaphore = asyncio.Semaphore(self.max_concurrency)

        # 2a. Process structured text for all pages
        for source_id, page_data in crawled_data.items():
            page_text = page_data.get("text", "")
            if page_text:
                chunks = self.chunker.split_text(page_text)
                logger.info(f"Page '{source_id}' split into {len(chunks)} text chunks.")
                source_url = self._resolve_source_url(source_id)
                for idx, chunk in enumerate(chunks):
                    metadata = {
                        "source": source_id,
                        "source_url": source_url,
                        "chunk_index": idx,
                        "type": "confluence_scraped"
                    }
                    pending_records.append((chunk, metadata))

        # 2b. Concurrently process image attachments across all pages
        image_tasks = []
        for source_id, page_data in crawled_data.items():
            source_url = self._resolve_source_url(source_id)
            for image_url in page_data.get("images", []):
                image_tasks.append(self._process_image_async(source_id, source_url, image_url, semaphore))

        if image_tasks:
            logger.info(f"Processing {len(image_tasks)} image attachments concurrently...")
            image_results = await asyncio.gather(*image_tasks, return_exceptions=True)
            for res in image_results:
                if isinstance(res, Exception):
                    logger.warning(f"Image processing error: {res}")
                elif res:
                    pending_records.append(res)

        # 2c. Concurrently process PDF attachments across all pages
        pdf_tasks = []
        for source_id, page_data in crawled_data.items():
            source_url = self._resolve_source_url(source_id)
            for pdf_url in page_data.get("pdfs", []):
                pdf_tasks.append(self._process_pdf_async(source_id, source_url, pdf_url, semaphore))

        if pdf_tasks:
            logger.info(f"Processing {len(pdf_tasks)} PDF attachments concurrently...")
            pdf_results = await asyncio.gather(*pdf_tasks, return_exceptions=True)
            for res in pdf_results:
                if isinstance(res, Exception):
                    logger.warning(f"PDF processing error: {res}")
                elif res:
                    pending_records.extend(res)

        if not pending_records:
            logger.warning("No text or attachment chunks generated from crawled pages.")
            return 0

        # 3. Batch Vector Embeddings
        logger.info(f"Generating dense vector embeddings for {len(pending_records)} total chunks (batch size: {self.batch_size})...")
        embedded_records: list[tuple[str, dict, list[float]]] = []

        for i in range(0, len(pending_records), self.batch_size):
            batch = pending_records[i : i + self.batch_size]
            batch_texts = [item[0] for item in batch]
            try:
                batch_embeddings = await asyncio.to_thread(
                    self.embedding_service.get_dense_embeddings, batch_texts
                )
            except Exception as e:
                logger.warning(f"Batch embedding failed ({e}), falling back to individual embeddings.")
                batch_embeddings = [
                    await asyncio.to_thread(self.embedding_service.get_dense_embedding, t)
                    for t in batch_texts
                ]

            for (content, meta), emb in zip(batch, batch_embeddings):
                embedded_records.append((content, meta, emb))

        # 4. Batch Database Ingestion
        total_ingested = len(embedded_records)
        logger.info(f"Loading {total_ingested} embedded chunks into PostgreSQL within an atomic transaction...")
        
        def _db_insert_batch():
            with closing(self.db_conn_factory()) as conn:
                with conn.cursor() as cur:
                    insert_query = """
                        INSERT INTO documents (content, metadata, embedding)
                        VALUES (%s, %s, %s);
                    """
                    data_tuples = [
                        (content, psycopg.types.json.Jsonb(meta), emb)
                        for content, meta, emb in embedded_records
                    ]
                    cur.executemany(insert_query, data_tuples)
                conn.commit()

        try:
            await asyncio.to_thread(_db_insert_batch)
            logger.info(f"Database transaction committed successfully! Ingested {total_ingested} total chunks.")
        except Exception as e:
            logger.error(f"Failed executing ETL pipeline load transaction: {e}")
            raise e

        return total_ingested

    def run(self, root_identifier: str, max_depth: int = 2) -> int:
        """Synchronous wrapper for run_async."""
        return _run_sync(self.run_async(root_identifier=root_identifier, max_depth=max_depth))

    async def _process_image_async(
        self, source_id: str, source_url: str, image_url: str, semaphore: asyncio.Semaphore
    ) -> tuple[str, dict] | None:
        """Downloads, transcribes, and formats a single image attachment."""
        async with semaphore:
            try:
                logger.info(f"Processing image attachment: {image_url}")
                image_bytes = await _resolve_coro_or_sync(self.client.download_attachment, image_url)
                filename = os.path.basename(urlparse(image_url).path) or "image.png"
                saved_url = self._save_asset(filename, image_bytes)
                
                # Transcribe image content
                description = await asyncio.to_thread(self._get_image_description, image_bytes, filename)
                full_content = f"[Embedded Image: {filename} - Source URL: {saved_url}]\n{description}"
                
                metadata = {
                    "source": source_id,
                    "source_url": source_url,
                    "image_url": saved_url,
                    "type": "confluence_image"
                }
                return full_content, metadata
            except Exception as e:
                logger.warning(f"Failed processing image '{image_url}': {e}")
                return None

    async def _process_pdf_async(
        self, source_id: str, source_url: str, pdf_url: str, semaphore: asyncio.Semaphore
    ) -> list[tuple[str, dict]]:
        """Downloads, transcribes, and splits a single PDF attachment into chunks."""
        async with semaphore:
            try:
                logger.info(f"Processing PDF attachment: {pdf_url}")
                pdf_bytes = await _resolve_coro_or_sync(self.client.download_attachment, pdf_url)
                filename = os.path.basename(urlparse(pdf_url).path) or "document.pdf"
                saved_url = self._save_asset(filename, pdf_bytes)
                
                # Transcribe PDF content
                markdown_text = await asyncio.to_thread(self._get_pdf_markdown, pdf_bytes, filename)
                full_pdf_content = f"[PDF Attachment: {filename} - Source URL: {saved_url}]\n{markdown_text}"
                
                chunks = self.chunker.split_text(full_pdf_content)
                records = []
                for idx, chunk in enumerate(chunks):
                    metadata = {
                        "source": source_id,
                        "source_url": source_url,
                        "pdf_url": saved_url,
                        "chunk_index": idx,
                        "type": "confluence_pdf_page"
                    }
                    records.append((chunk, metadata))
                return records
            except Exception as e:
                logger.warning(f"Failed processing PDF '{pdf_url}': {e}")
                return []

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

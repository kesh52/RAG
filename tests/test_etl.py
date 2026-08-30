import pytest
import asyncio
import httpx
from unittest.mock import MagicMock
from src.etl.confluence import BaseConfluenceClient, APIConfluenceClient, RecursiveCrawler
from src.etl.chunking import RecursiveTextChunker
from src.etl.pipeline import ConfluenceETLPipeline
from src.embeddings.base import BaseEmbeddingService

# ----------------- Mock Clients & Services -----------------

class MockConfluenceClient(BaseConfluenceClient):
    """Hermetic mock confluence client containing a predefined graph of pages and links."""
    
    def __init__(self, async_mode: bool = False):
        self.domain = "wiki.example.com"
        self.async_mode = async_mode
        # Directed graph layout representing page IDs and their content + links
        # Schema: page_id -> (text_content, linked_page_ids, image_links, pdf_links)
        self.site_graph = {
            # Root (Depth 0)
            "root_page": ("Welcome to our wiki root page", ["page_d1_a", "page_d1_b"], [], []),
            
            # Level 1 (Depth 1)
            "page_d1_a": ("This is depth 1 page A details", ["page_d2_a"], [], []),
            "page_d1_b": ("This is depth 1 page B details", ["page_d2_b", "root_page"], [], []), 
            
            # Level 2 (Depth 2)
            "page_d2_a": ("Deep info page at depth 2 A", ["page_d3_a"], [], []), 
            "page_d2_b": ("Deep info page at depth 2 B", [], [], []),
            
            # Level 3 (Depth 3 - must not be crawled when max_depth=2)
            "page_d3_a": ("This content should be unreachable during crawl", [], [], [])
        }

    async def get_page_content_and_links(self, page_id_or_url: str) -> tuple[str, list[str], list[str], list[str]]:
        if self.async_mode:
            await asyncio.sleep(0.01)
        if page_id_or_url in self.site_graph:
            val = self.site_graph[page_id_or_url]
            if len(val) == 2:
                return val[0], val[1], [], []
            return val
        raise KeyError(f"Page {page_id_or_url} not found in mock graph.")

    async def download_attachment(self, attachment_url: str) -> bytes:
        if self.async_mode:
            await asyncio.sleep(0.01)
        return b"mock binary content"


class MockEmbeddingService(BaseEmbeddingService):
    """Mock embedding generator returning dummy floats and recording batch calls."""
    
    def __init__(self):
        self.call_history = []

    def get_dense_embedding(self, text: str) -> list[float]:
        self.call_history.append([text])
        return [0.1] * 768

    def get_dense_embeddings(self, texts: list[str]) -> list[list[float]]:
        self.call_history.append(texts)
        return [[0.1] * 768 for _ in texts]


# ----------------- Unit Tests -----------------

def test_recursive_text_chunker():
    """Assert chunker correctly splits text recursively using delimiters and overlaps."""
    chunker = RecursiveTextChunker(chunk_size=15, chunk_overlap=5)
    
    text = "hello world this is a test"
    chunks = chunker.split_text(text)
    assert len(chunks) == 3
    assert chunks[0] == "hello world"
    assert chunks[1] == "world this is a"
    assert chunks[2] == "is a test"


def test_recursive_text_chunker_empty():
    """Verify chunker returns empty list for empty/null values."""
    chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.split_text("") == []
    assert chunker.split_text("   ") == []


def test_recursive_crawler_depth_and_cycles():
    """Assert crawler reaches depth boundary limits and skips circular loops."""
    client = MockConfluenceClient()
    crawler = RecursiveCrawler(client, max_depth=2, max_concurrency=4)
    
    crawled_data = crawler.crawl("root_page")
    
    # Expected page crawls:
    # Depth 0: root_page
    # Depth 1: page_d1_a, page_d1_b
    # Depth 2: page_d2_a, page_d2_b
    # Ignored: page_d3_a (depth 3)
    
    assert "root_page" in crawled_data
    assert "page_d1_a" in crawled_data
    assert "page_d1_b" in crawled_data
    assert "page_d2_a" in crawled_data
    assert "page_d2_b" in crawled_data
    assert "page_d3_a" not in crawled_data
    
    assert len(crawled_data) == 5
    assert crawled_data["root_page"]["text"] == "Welcome to our wiki root page"


def test_recursive_crawler_async_concurrency():
    """Assert async crawler runs concurrently across links with semaphore."""
    async def _run():
        client = MockConfluenceClient(async_mode=True)
        crawler = RecursiveCrawler(client, max_depth=2, max_concurrency=3)
        crawled_data = await crawler.crawl_async("root_page")
        assert len(crawled_data) == 5
        assert "root_page" in crawled_data
        assert "page_d2_a" in crawled_data

    asyncio.run(_run())


def test_recursive_crawler_domain_restrictions():
    """Assert crawler filters out links pointing to domains outside the allowed pattern."""
    client = MockConfluenceClient()
    # Add external links and internal absolute links to root_page mock content
    client.site_graph["root_page"] = (
        "Root page content",
        [
            "https://wiki.example.com/pages/viewpage.action?pageId=123", # Allowed (matches domain)
            "https://google.com/search",                                 # External (should be blocked)
            "https://malicious.network/docs",                           # External (should be blocked)
            "local_page_id",                                             # Allowed (no domain / relative)
            "https://mybestsubpage.wiki.example.com"
        ]
    )
    # Define contents for internal pages
    client.site_graph["https://wiki.example.com/pages/viewpage.action?pageId=123"] = ("Internal page 123", [])
    client.site_graph["local_page_id"] = ("Local relative page", [])
    
    # Run crawl using the default pattern derived from client.domain
    crawler = RecursiveCrawler(client, max_depth=1)
    crawled_data = crawler.crawl("root_page")
    
    assert "root_page" in crawled_data
    assert "https://wiki.example.com/pages/viewpage.action?pageId=123" in crawled_data
    assert "local_page_id" in crawled_data
    assert "https://google.com/search" not in crawled_data
    assert "https://malicious.network/docs" not in crawled_data


def test_api_confluence_client_httpx_mock():
    """Verify APIConfluenceClient makes correct HTTP requests using httpx AsyncClient."""
    async def _run():
        def custom_handler(request: httpx.Request):
            url_str = str(request.url)
            if "/wiki/rest/api/content/12345" in url_str:
                return httpx.Response(
                    200,
                    json={
                        "id": "12345",
                        "title": "Incident SOP",
                        "body": {
                            "storage": {
                                "value": "<p>This is test content with a <a href='/wiki/pages/viewpage.action?pageId=67890'>link</a> and <img src='/download/attachments/test.png'/> and <a href='/download/attachments/doc.pdf'>PDF</a>.</p>"
                            }
                        }
                    }
                )
            elif "/download/attachments/test.png" in url_str:
                return httpx.Response(200, content=b"\x89PNG\r\n\x1a\n...")
            return httpx.Response(404, text="Not Found")

        mock_transport = httpx.MockTransport(custom_handler)
        async with httpx.AsyncClient(transport=mock_transport) as async_http_client:
            client = APIConfluenceClient(
                domain="wiki.example.com",
                username="user",
                api_token="secret",
                client=async_http_client
            )
            
            details = await client.fetch_page_details("12345")
            assert details["title"] == "Incident SOP"
            assert "This is test content" in details["text"]
            assert len(details["links"]) == 1
            assert len(details["images"]) == 1
            assert len(details["pdfs"]) == 1

            img_bytes = await client.download_attachment("https://wiki.example.com/download/attachments/test.png")
            assert img_bytes.startswith(b"\x89PNG")

    asyncio.run(_run())


def test_api_confluence_client_sync_wrappers():
    """Verify synchronous wrappers on APIConfluenceClient function properly."""
    def custom_handler(request: httpx.Request):
        return httpx.Response(
            200,
            json={
                "id": "999",
                "title": "Sync Test",
                "body": {"storage": {"value": "<p>Sync text content</p>"}}
            }
        )

    mock_transport = httpx.MockTransport(custom_handler)
    async_http_client = httpx.AsyncClient(transport=mock_transport)
    client = APIConfluenceClient(
        domain="wiki.example.com",
        username="user",
        api_token="secret",
        client=async_http_client
    )
    
    details = client.fetch_page_details_sync("999")
    assert details["title"] == "Sync Test"
    assert "Sync text content" in details["text"]


def test_confluence_etl_pipeline_run_batch():
    """Assert pipeline crawls, chunks, batches embeddings, and executes batch DB insertion."""
    client = MockConfluenceClient()
    chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=10)
    emb_service = MockEmbeddingService()
    
    # Mock psycopg DB context managers
    mock_cursor = MagicMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    
    mock_db_factory = MagicMock()
    mock_db_factory.return_value = mock_conn

    pipeline = ConfluenceETLPipeline(
        confluence_client=client,
        chunker=chunker,
        embedding_service=emb_service,
        db_conn_factory=mock_db_factory,
        batch_size=2,
    )
    
    ingested_count = pipeline.run("root_page", max_depth=1)
    
    # max_depth=1: root_page (Welcome to our wiki root page) + page_d1_a + page_d1_b
    # Total 3 chunks.
    assert ingested_count == 3
    
    # Verify DB calls were triggered using executemany for batch insertion
    assert mock_db_factory.call_count == 1
    assert mock_cursor.executemany.call_count == 1
    # Check that 3 tuples were passed to executemany
    inserted_tuples = mock_cursor.executemany.call_args[0][1]
    assert len(inserted_tuples) == 3
    assert mock_conn.commit.call_count == 1
    
    # Verify batch embedding service was used with batch size 2 (first batch of 2, second of 1)
    assert len(emb_service.call_history) == 2
    assert len(emb_service.call_history[0]) == 2
    assert len(emb_service.call_history[1]) == 1


def test_confluence_etl_pipeline_multimodal_ingestion():
    """Verify that ConfluenceETLPipeline concurrently downloads, transcribes, and inserts images and PDFs."""
    client = MockConfluenceClient()
    # Configure root_page to have 1 image link and 1 PDF link
    client.site_graph["root_page"] = (
        "Welcome to our wiki root page",
        [],
        ["https://wiki.example.com/download/attachments/logo.png"],
        ["https://wiki.example.com/download/attachments/spec.pdf"]
    )
    
    chunker = RecursiveTextChunker(chunk_size=300, chunk_overlap=10)
    emb_service = MockEmbeddingService()
    
    mock_cursor = MagicMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    
    mock_db_factory = MagicMock()
    mock_db_factory.return_value = mock_conn

    pipeline = ConfluenceETLPipeline(
        confluence_client=client,
        chunker=chunker,
        embedding_service=emb_service,
        db_conn_factory=mock_db_factory
    )
    
    ingested_count = pipeline.run("root_page", max_depth=0)
    
    # Ingested chunks:
    # 1. Plain text page: "Welcome to our wiki root page" (1 chunk)
    # 2. Image: downloaded, saved, transcribed (1 chunk)
    # 3. PDF: downloaded, saved, transcribed (1 chunk)
    # Total = 3 chunks!
    assert ingested_count == 3
    assert mock_cursor.executemany.call_count == 1
    inserted_tuples = mock_cursor.executemany.call_args[0][1]
    assert len(inserted_tuples) == 3


def test_confluence_etl_pipeline_with_semantic_chunker():
    """Verify that ConfluenceETLPipeline functions properly with SemanticChunker."""
    from src.etl.chunking import SemanticChunker

    client = MockConfluenceClient()
    emb_service = MockEmbeddingService()
    chunker = SemanticChunker(
        embedding_service=emb_service,
        min_chunk_size=10,
        max_chunk_size=500,
    )

    mock_cursor = MagicMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_db_factory = MagicMock(return_value=mock_conn)

    pipeline = ConfluenceETLPipeline(
        confluence_client=client,
        chunker=chunker,
        embedding_service=emb_service,
        db_conn_factory=mock_db_factory,
    )

    ingested_count = pipeline.run("root_page", max_depth=1)
    assert ingested_count == 3
    assert mock_cursor.executemany.call_count == 1
    assert mock_conn.commit.call_count == 1

import pytest
from unittest.mock import MagicMock
from src.etl.confluence import BaseConfluenceClient, RecursiveCrawler
from src.etl.chunking import RecursiveTextChunker
from src.etl.pipeline import ConfluenceETLPipeline
from src.embeddings.base import BaseEmbeddingService

# ----------------- Mock Clients & Services -----------------

class MockConfluenceClient(BaseConfluenceClient):
    """Hermetic mock confluence client containing a predefined graph of pages and links."""
    
    def __init__(self):
        self.domain = "wiki.example.com"
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

    def get_page_content_and_links(self, page_id_or_url: str) -> tuple[str, list[str], list[str], list[str]]:
        if page_id_or_url in self.site_graph:
            val = self.site_graph[page_id_or_url]
            if len(val) == 2:
                return val[0], val[1], [], []
            return val
        raise KeyError(f"Page {page_id_or_url} not found in mock graph.")

    def download_attachment(self, attachment_url: str) -> bytes:
        return b"mock binary content"


class MockEmbeddingService(BaseEmbeddingService):
    """Mock embedding generator returning dummy floats."""
    
    def get_dense_embedding(self, text: str) -> list[float]:
        return [0.1] * 768


# ----------------- Unit Tests -----------------

def test_recursive_text_chunker():
    """Assert chunker correctly splits text recursively using delimiters and overlaps."""
    chunker = RecursiveTextChunker(chunk_size=15, chunk_overlap=5)
    
    text = "hello world this is a test"
    # splits by space: ['hello', 'world', 'this', 'is', 'a', 'test']
    # Merge splits:
    # 1: "hello world" (len: 11)
    # 2: "world this is a" (len: 15)
    # 3: "is a test" (len: 9)
    
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
    crawler = RecursiveCrawler(client, max_depth=2)
    
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


def test_confluence_etl_pipeline_run():
    """Assert pipeline crawls, chunks, calls embedding service, and commits database transactions."""
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
        db_conn_factory=mock_db_factory
    )
    
    ingested_count = pipeline.run("root_page", max_depth=1)
    
    # max_depth=1: root_page (Welcome to our wiki root page) + page_d1_a + page_d1_b
    # All are short, so each fits in 1 chunk. Total 3 chunks.
    assert ingested_count == 3
    
    # Verify DB calls were triggered
    assert mock_db_factory.call_count == 1
    assert mock_cursor.execute.call_count == 3
    assert mock_conn.commit.call_count == 1


def test_confluence_etl_pipeline_multimodal_ingestion():
    """Verify that ConfluenceETLPipeline downloads, transcribes, and inserts images and PDFs."""
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
    # 2. Image: downloaded, saved, transcribed (1 chunk) -> since Gemini client is mocked/None, gets fallback text
    # 3. PDF: downloaded, saved, transcribed (1 chunk)
    # Total = 3 chunks!
    assert ingested_count == 3
    assert mock_cursor.execute.call_count == 3


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
    assert mock_cursor.execute.call_count == 3
    assert mock_conn.commit.call_count == 1



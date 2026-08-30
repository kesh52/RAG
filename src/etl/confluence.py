import re
import logging
import asyncio
import inspect
import concurrent.futures
from abc import ABC, abstractmethod
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse, parse_qs
import httpx

from src.utils.config import config

logger = logging.getLogger(__name__)


def _run_sync(coro):
    """Safely executes an async coroutine from synchronous contexts, even if an event loop is already active."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


async def _resolve_coro_or_sync(func, *args, **kwargs):
    """Invokes a callable and awaits it if it returns a coroutine, supporting both sync and async clients."""
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    res = func(*args, **kwargs)
    if inspect.iscoroutine(res):
        return await res
    return res


class BaseConfluenceClient(ABC):
    """Abstract base class defining the Confluence API client interface."""
    
    @abstractmethod
    def get_page_content_and_links(self, page_id_or_url: str):
        """Fetch the text content, page links, image links, and PDF attachment links from a page.

        Can be implemented as either an async coroutine or a synchronous method.

        Returns:
            A tuple of (text_content, page_links, image_links, pdf_links)
        """
        pass

    @abstractmethod
    def download_attachment(self, attachment_url: str):
        """Download binary attachment content from Confluence.

        Can be implemented as either an async coroutine or a synchronous method.

        Returns:
            Binary bytes content of the downloaded file.
        """
        pass


class ConfluenceHTMLParser(HTMLParser):
    """HTML parser to extract structured text content, page links, images, and PDF attachments."""

    BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "div", "tr", "table", "section", "article", "blockquote", "pre"}
    INLINE_BREAK_TAGS = {"br", "li"}

    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.links = []
        self.images = []
        self.pdfs = []
        self.text_parts = []

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCK_TAGS:
            self.text_parts.append("\n\n")
        elif tag in self.INLINE_BREAK_TAGS:
            self.text_parts.append("\n")
        elif tag == "a":
            for attr, value in attrs:
                if attr == "href" and value:
                    absolute_url = urljoin(self.base_url, value)
                    # Check if link points to a PDF attachment
                    if absolute_url.lower().endswith(".pdf") or "/download/attachments/" in absolute_url.lower():
                        if absolute_url not in self.pdfs:
                            self.pdfs.append(absolute_url)
                    else:
                        if absolute_url not in self.links:
                            self.links.append(absolute_url)
        elif tag == "img":
            for attr, value in attrs:
                if attr == "src" and value:
                    absolute_url = urljoin(self.base_url, value)
                    if absolute_url not in self.images:
                        self.images.append(absolute_url)

    def handle_endtag(self, tag):
        if tag in self.BLOCK_TAGS:
            self.text_parts.append("\n\n")

    def handle_data(self, data):
        self.text_parts.append(data)

    def get_parsed_data(self) -> tuple[str, list[str], list[str], list[str]]:
        raw_text = "".join(self.text_parts)
        # Normalize whitespace while preserving paragraph double newlines
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw_text.splitlines()]
        cleaned_lines = []
        for line in lines:
            if line:
                cleaned_lines.append(line)
            elif cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
        text = "\n".join(cleaned_lines).strip()
        return text, self.links, self.images, self.pdfs


class APIConfluenceClient(BaseConfluenceClient):
    """Asynchronous Confluence client that makes non-blocking HTTP calls using httpx."""
    
    def __init__(
        self,
        domain: str,
        username: str,
        api_token: str,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ):
        self.domain = domain.strip().replace("https://", "").replace("http://", "")
        self.base_url = f"https://{self.domain}"
        self.username = username
        self.api_token = api_token
        self.timeout = timeout
        self._external_client = client
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Returns the active httpx.AsyncClient or creates a default one."""
        if self._external_client is not None:
            return self._external_client
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                auth=httpx.BasicAuth(self.username, self.api_token),
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
                headers={"Accept": "application/json", "User-Agent": "RAG-Confluence-Crawler/2.0"},
                follow_redirects=True,
            )
        return self._client

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    async def aclose(self):
        """Closes the underlying HTTP client session."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def close(self):
        """Synchronously closes the client session."""
        if self._client is not None and not self._client.is_closed:
            _run_sync(self.aclose())

    def _extract_page_id(self, page_id_or_url: str) -> str:
        """Helper to resolve a page ID from either a raw string ID or a Confluence URL."""
        if page_id_or_url.isdigit():
            return page_id_or_url
            
        parsed = urlparse(page_id_or_url)
        qs = parse_qs(parsed.query)
        if "pageId" in qs:
            return qs["pageId"][0]
            
        # Match path-based page ID: /wiki/spaces/.../pages/123456/...
        match = re.search(r"/pages/(\d+)", parsed.path)
        if match:
            return match.group(1)
            
        raise ValueError(f"Could not extract a valid page ID from: {page_id_or_url}")

    async def fetch_page_details(self, page_id_or_url: str) -> dict:
        """Asynchronously fetches page title, parsed structured text, raw HTML, links, images, and PDFs."""
        page_id = self._extract_page_id(page_id_or_url)
        api_url = f"{self.base_url}/wiki/rest/api/content/{page_id}?expand=body.storage,version"
        
        logger.info(f"Fetching Confluence page ID {page_id} from API...")
        client = self._get_client()
        response = await client.get(api_url)
        response.raise_for_status()
        
        data = response.json()
        title = data.get("title", f"Page {page_id}")
        html_content = data.get("body", {}).get("storage", {}).get("value", "")
        
        # Parse the HTML content for links, images, and PDFs
        parser = ConfluenceHTMLParser(self.base_url)
        parser.feed(html_content)
        text, links, images, pdfs = parser.get_parsed_data()
        
        full_text = f"# {title}\n\n{text}" if title and not text.startswith("#") else text

        return {
            "page_id": page_id,
            "title": title,
            "text": full_text,
            "html": html_content,
            "links": links,
            "images": images,
            "pdfs": pdfs,
        }

    def fetch_page_details_sync(self, page_id_or_url: str) -> dict:
        """Synchronous wrapper for fetch_page_details."""
        return _run_sync(self.fetch_page_details(page_id_or_url))

    async def get_page_content_and_links(self, page_id_or_url: str) -> tuple[str, list[str], list[str], list[str]]:
        """Asynchronously fetches text content, links, images, and PDFs."""
        details = await self.fetch_page_details(page_id_or_url)
        return details["text"], details["links"], details["images"], details["pdfs"]

    def get_page_content_and_links_sync(self, page_id_or_url: str) -> tuple[str, list[str], list[str], list[str]]:
        """Synchronous wrapper for get_page_content_and_links."""
        return _run_sync(self.get_page_content_and_links(page_id_or_url))

    async def download_attachment(self, attachment_url: str) -> bytes:
        """Asynchronously downloads binary attachment content using httpx."""
        logger.info(f"Downloading attachment from URL: {attachment_url}")
        client = self._get_client()
        response = await client.get(attachment_url)
        response.raise_for_status()
        return response.content

    def download_attachment_sync(self, attachment_url: str) -> bytes:
        """Synchronous wrapper for download_attachment."""
        return _run_sync(self.download_attachment(attachment_url))


class RecursiveCrawler:
    """Asynchronous crawler engine that traverses Confluence pages concurrently up to a specified depth using BFS."""
    
    def __init__(
        self,
        client: BaseConfluenceClient,
        max_depth: int = 2,
        allowed_domain_pattern: str | None = None,
        max_concurrency: int | None = None,
    ):
        self.client = client
        self.max_depth = max_depth
        
        # Concurrency limit (default from config or 5)
        concurrency = max_concurrency or int(config.get("crawler.max_concurrency", 5))
        self.max_concurrency = max(1, concurrency)
        
        self.allowed_domain_pattern = None
        if allowed_domain_pattern:
            self.allowed_domain_pattern = re.compile(allowed_domain_pattern, re.IGNORECASE)
        elif hasattr(client, "domain") and client.domain:
            escaped_domain = re.escape(client.domain)
            self.allowed_domain_pattern = re.compile(rf"^(.*\.)?{escaped_domain}$", re.IGNORECASE)

    def _normalize_id(self, identifier: str) -> str:
        """Normalizes page URL or ID to ensure proper deduplication."""
        if hasattr(self.client, "_extract_page_id"):
            try:
                return self.client._extract_page_id(identifier)
            except ValueError:
                return identifier
        return identifier

    def _is_allowed_link(self, link: str) -> bool:
        """Checks whether a link conforms to domain constraints."""
        if self.allowed_domain_pattern and (link.startswith("http://") or link.startswith("https://")):
            parsed_url = urlparse(link)
            if parsed_url.netloc and not self.allowed_domain_pattern.match(parsed_url.netloc):
                logger.warning(f"Skipping link '{link}' because it does not match the allowed domain pattern.")
                return False
        return True

    async def _fetch_single_page(self, current_id: str, semaphore: asyncio.Semaphore) -> tuple[str, dict | None, list[str]]:
        """Asynchronously fetches a single page protected by a concurrency semaphore."""
        async with semaphore:
            try:
                text, links, images, pdfs = await _resolve_coro_or_sync(
                    self.client.get_page_content_and_links, current_id
                )
                return current_id, {
                    "text": text,
                    "images": images,
                    "pdfs": pdfs,
                }, links
            except Exception as e:
                logger.warning(f"Skipping page {current_id} due to fetch error: {e}")
                return current_id, None, []

    async def crawl_async(self, root_identifier: str) -> dict[str, dict]:
        """Asynchronously crawls pages starting from the root page up to max_depth using concurrent BFS.

        Returns:
            A dictionary mapping page identifier -> {"text": str, "images": list, "pdfs": list}
        """
        visited = set()
        crawled_data = {}
        semaphore = asyncio.Semaphore(self.max_concurrency)

        # Queue contains list of URLs/identifiers for the current BFS level: [(identifier, depth)]
        current_level = [root_identifier]
        visited.add(self._normalize_id(root_identifier))

        for depth in range(self.max_depth + 1):
            if not current_level:
                break

            logger.info(f"Crawling BFS depth {depth}/{self.max_depth} ({len(current_level)} pages concurrently)...")
            
            # Fetch all pages in the current level concurrently
            tasks = [self._fetch_single_page(page_id, semaphore) for page_id in current_level]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            next_level = []
            for item in results:
                if isinstance(item, Exception):
                    logger.error(f"Unexpected error in page crawl task: {item}")
                    continue
                
                page_id, page_dict, links = item
                if page_dict is not None:
                    crawled_data[page_id] = page_dict

                # Discover new links if depth boundary has not been reached
                if depth < self.max_depth:
                    for link in links:
                        if not self._is_allowed_link(link):
                            continue
                        
                        norm_link = self._normalize_id(link)
                        if norm_link not in visited:
                            visited.add(norm_link)
                            next_level.append(link)

            current_level = next_level

        logger.info(f"Async crawl complete. Visited and collected {len(crawled_data)} total pages.")
        return crawled_data

    def crawl(self, root_identifier: str) -> dict[str, dict]:
        """Synchronous wrapper for crawl_async."""
        return _run_sync(self.crawl_async(root_identifier))

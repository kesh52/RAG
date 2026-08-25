import re
import logging
from abc import ABC, abstractmethod
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse, parse_qs
import requests

logger = logging.getLogger(__name__)

class BaseConfluenceClient(ABC):
    """Abstract base class defining the Confluence API client interface."""
    
    @abstractmethod
    def get_page_content_and_links(self, page_id_or_url: str) -> tuple[str, list[str], list[str], list[str]]:
        """Fetch the text content, page links, image links, and PDF attachment links from a page.

        Returns:
            A tuple of (text_content, page_links, image_links, pdf_links)
        """
        pass

    @abstractmethod
    def download_attachment(self, attachment_url: str) -> bytes:
        """Download binary attachment content from Confluence.

        Returns:
            Binary bytes content of the downloaded file.
        """
        pass


class ConfluenceHTMLParser(HTMLParser):
    """HTML parser to extract text content, page links, images, and PDF attachments from HTML body."""
    
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.links = []
        self.images = []
        self.pdfs = []
        self.text_parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
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

    def handle_data(self, data):
        self.text_parts.append(data)

    def get_parsed_data(self) -> tuple[str, list[str], list[str], list[str]]:
        text = " ".join(part.strip() for part in self.text_parts if part.strip())
        # Clean up double spacing
        text = re.sub(r"\s+", " ", text).strip()
        return text, self.links, self.images, self.pdfs


class APIConfluenceClient(BaseConfluenceClient):
    """Concrete Confluence client that makes live HTTP calls to the REST API."""
    
    def __init__(self, domain: str, username: str, api_token: str):
        self.domain = domain.strip().replace("https://", "").replace("http://", "")
        self.base_url = f"https://{self.domain}"
        self.auth = (username, api_token)

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

    def get_page_content_and_links(self, page_id_or_url: str) -> tuple[str, list[str], list[str], list[str]]:
        page_id = self._extract_page_id(page_id_or_url)
        api_url = f"{self.base_url}/wiki/rest/api/content/{page_id}?expand=body.storage"
        
        logger.info(f"Fetching Confluence page ID {page_id} from API...")
        response = requests.get(api_url, auth=self.auth, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        html_content = data.get("body", {}).get("storage", {}).get("value", "")
        
        # Parse the HTML content for links, images, and PDFs
        parser = ConfluenceHTMLParser(self.base_url)
        parser.feed(html_content)
        text, links, images, pdfs = parser.get_parsed_data()
        
        return text, links, images, pdfs

    def download_attachment(self, attachment_url: str) -> bytes:
        logger.info(f"Downloading attachment from URL: {attachment_url}")
        response = requests.get(attachment_url, auth=self.auth, timeout=30)
        response.raise_for_status()
        return response.content


class RecursiveCrawler:
    """Crawler engine that traverses Confluence pages up to a specified depth using BFS."""
    
    def __init__(self, client: BaseConfluenceClient, max_depth: int = 2, allowed_domain_pattern: str | None = None):
        self.client = client
        self.max_depth = max_depth
        
        self.allowed_domain_pattern = None
        if allowed_domain_pattern:
            self.allowed_domain_pattern = re.compile(allowed_domain_pattern, re.IGNORECASE)
        elif hasattr(client, "domain") and client.domain:
            escaped_domain = re.escape(client.domain)
            self.allowed_domain_pattern = re.compile(rf"^(.*\.)?{escaped_domain}$", re.IGNORECASE)

    def crawl(self, root_identifier: str) -> dict[str, str]:
        """Crawl pages starting from the root page up to max_depth.

        Returns:
            A dictionary mapping page identifier -> text_content
        """
        visited = set()
        queue = [(root_identifier, 0)]  # Queue stores tuples of (identifier, depth)
        crawled_data = {}

        while queue:
            current_id, depth = queue.pop(0)
            
            # Normalize/sanitize visited tracker to avoid duplicate pages under different formats
            # (e.g. page ID vs absolute URL matching the same page ID)
            if hasattr(self.client, "_extract_page_id"):
                try:
                    norm_id = self.client._extract_page_id(current_id)
                except ValueError:
                    norm_id = current_id
            else:
                norm_id = current_id

            if norm_id in visited:
                continue
            visited.add(norm_id)

            try:
                text, links, images, pdfs = self.client.get_page_content_and_links(current_id)
                crawled_data[current_id] = {
                    "text": text,
                    "images": images,
                    "pdfs": pdfs
                }
                
                # Expand links if we haven't reached the max depth boundary
                if depth < self.max_depth:
                    for link in links:
                        # Check domain restriction if pattern is set
                        if self.allowed_domain_pattern and (link.startswith("http://") or link.startswith("https://")):
                            parsed_url = urlparse(link)
                            if parsed_url.netloc and not self.allowed_domain_pattern.match(parsed_url.netloc):
                                logger.warning(f"Skipping link '{link}' because it does not match the allowed domain pattern.")
                                continue

                        # Extract normalized link ID to check before adding to queue
                        if hasattr(self.client, "_extract_page_id"):
                            try:
                                norm_link = self.client._extract_page_id(link)
                            except ValueError:
                                norm_link = link
                        else:
                            norm_link = link
                            
                        if norm_link not in visited:
                            queue.append((link, depth + 1))
                            
            except Exception as e:
                logger.warning(f"Skipping page {current_id} due to fetch error: {e}")
                continue

        return crawled_data


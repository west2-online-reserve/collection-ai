"""
HTTP client wrapper for the FZU announcement crawler.
"""
from typing import cast

import httpx
from lxml import html
from lxml.html import HtmlElement  # pyright: ignore[reportAny]

from .url_utils import BASE_URL, MAIN_PAGE_URL
from .parsers import extract_total_page


class CrawlerClient:
    """
    HTTP client wrapper for fetching and parsing web pages.
    
    Supports context manager protocol for proper resource cleanup.
    
    Example:
        with CrawlerClient() as client:
            tree = client.fetch_page("https://example.com")
            total = client.get_total_pages()
    """
    
    def __init__(self, base_url: str = BASE_URL):
        """
        Initialize the crawler client.
        
        Args:
            base_url: Base URL for the crawler (default: jwch.fzu.edu.cn)
        """
        self._client: httpx.Client = httpx.Client(verify=False)
        self.base_url: str = base_url
    
    def fetch_page(self, url: str) -> HtmlElement:  # pyright: ignore[reportAny]
        """
        Fetch a page and return the parsed HTML tree.
        
        Args:
            url: URL to fetch
            
        Returns:
            Parsed lxml HtmlElement tree
            
        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        response = self._client.get(url)
        _ = response.raise_for_status()
        tree = html.fromstring(response.text)  # pyright: ignore[reportUnknownMemberType]
        return cast(HtmlElement, tree)  # pyright: ignore[reportAny]
    
    def fetch_page_text(self, url: str) -> str:
        """
        Fetch a page and return the raw text.
        
        Args:
            url: URL to fetch
            
        Returns:
            Raw response text
            
        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        response = self._client.get(url)
        _ = response.raise_for_status()
        return response.text
    
    def get_total_pages(self) -> int:
        """
        Fetch the main page and extract the total page count.
        
        Returns:
            Total number of pages
        """
        tree: HtmlElement = self.fetch_page(MAIN_PAGE_URL)  # pyright: ignore[reportAny]
        return extract_total_page(tree)
    
    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
    
    def __enter__(self) -> "CrawlerClient":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object) -> None:
        """Context manager exit - ensures client is closed."""
        self.close()

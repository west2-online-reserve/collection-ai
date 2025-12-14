"""
URL utilities for constructing crawler URLs.
"""
from urllib.parse import urljoin


BASE_URL = "https://jwch.fzu.edu.cn"
URL_TPL = "https://jwch.fzu.edu.cn/jxtz/{}.htm"
MAIN_PAGE_URL = "https://jwch.fzu.edu.cn/jxtz.htm"


def construct_full_url(href: str, base_url: str = BASE_URL) -> str:
    """
    Construct a full URL from a possibly relative href.
    
    This function handles:
    - Relative URLs starting with '/' (e.g., '/info/1013/16745.htm')
    - Relative URLs without leading '/' (e.g., 'info/1013/16745.htm')
    - Already absolute URLs (e.g., 'https://example.com/page')
    - Fragment-only URLs (e.g., '#section')
    - Query-only URLs (e.g., '?param=value')
    
    Args:
        href: The URL or relative path to construct
        base_url: The base URL to use for relative URLs (default: BASE_URL)
    
    Returns:
        A fully qualified URL
    
    Examples:
        >>> construct_full_url('/info/1013/16745.htm')
        'https://jwch.fzu.edu.cn/info/1013/16745.htm'
        >>> construct_full_url('https://example.com/page')
        'https://example.com/page'
    """
    if not href:
        return base_url
    
    # urljoin properly handles all cases:
    # - Absolute URLs are returned as-is
    # - Relative URLs are joined with the base
    # - Handles edge cases like fragments, queries, etc.
    return urljoin(base_url, href)


def build_page_urls(max_page: int, start_page: int | None = None, end_page: int | None = None) -> list[str]:
    """
    Build a list of URLs to crawl, from latest to oldest.
    
    This is a pure function that does not make network requests.
    
    The url list includes (latest -> oldest):
        https://jwch.fzu.edu.cn/jxtz.htm  (newest page)
        https://jwch.fzu.edu.cn/jxtz/206.htm
        https://jwch.fzu.edu.cn/jxtz/205.htm
        ...
        https://jwch.fzu.edu.cn/jxtz/1.htm  (oldest page)
    
    Args:
        max_page: The maximum page number (typically total_pages - 1)
        start_page: Starting page number (inclusive). None means start from max_page.
        end_page: Ending page number (inclusive). None means end at page 1.
        
    Returns:
        List of URLs, sorted from newest to oldest.
        
    Examples:
        build_page_urls(206)  # Get all pages
        build_page_urls(206, start_page=206, end_page=200)  # Get pages 206-200
    """
    urls: list[str] = [MAIN_PAGE_URL]
    
    # Set default values
    actual_start = start_page if start_page is not None else max_page
    actual_end = end_page if end_page is not None else 1
    
    # Validate range
    actual_start = min(actual_start, max_page)
    actual_end = max(actual_end, 1)
    
    # Iterate from start_page to end_page in descending order (newest to oldest)
    for i in range(actual_start, actual_end - 1, -1):
        urls.append(URL_TPL.format(i))
    
    return urls

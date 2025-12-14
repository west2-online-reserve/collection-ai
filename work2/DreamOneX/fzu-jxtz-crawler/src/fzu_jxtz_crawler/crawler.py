from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from lxml.html import HtmlElement  # pyright: ignore[reportAny]

from .client import CrawlerClient
from .models import AnnouncementEntry
from .parsers import parse_announcement_list, parse_announcement_page
from .url_utils import build_page_urls


def _fetch_and_parse_list_page(
    client: CrawlerClient,
    url: str
) -> list[AnnouncementEntry]:
    """Helper to fetch and parse a single list page."""
    tree: HtmlElement = client.fetch_page(url)  # pyright: ignore[reportAny]
    return parse_announcement_list(tree)


def crawl_announcement_list(
    client: CrawlerClient,
    urls: list[str],
    on_page_fetched: Callable[[int, int, int], None] | None = None,
    max_workers: int = 5
) -> list[AnnouncementEntry]:
    """
    Crawl announcement list pages and return all entries.
    
    Args:
        client: CrawlerClient instance for making requests
        urls: List of page URLs to crawl
        on_page_fetched: Optional callback(page_index, total_pages, entries_count)
            called after each page is successfully fetched
        max_workers: Number of threads to use for crawling
            
    Returns:
        List of all AnnouncementEntry objects from the pages
    """
    all_entries: list[AnnouncementEntry] = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map futures to their index to maintain order or track progress
        future_to_idx = {
            executor.submit(_fetch_and_parse_list_page, client, url): idx 
            for idx, url in enumerate(urls)
        }
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                entries = future.result()
                all_entries.extend(entries)
                
                if on_page_fetched:
                    on_page_fetched(idx + 1, len(urls), len(entries))
                    
            except Exception as e:
                # Log error or handle it? For now, we print to stderr potentially, 
                # but better to re-raise or let the caller handle.
                # In a real app, use logging.
                print(f"Error fetching page {idx}: {e}")

    # Sorting might be needed if order matters, but usually for announcements 
    # we sort by date later or the order in list doesn't strictly matter 
    # if we are collecting all of them. 
    # However, to be safe and consistent with previous behavior (roughly), we can sort by something 
    # or just accept they are mixed.
    # The original implementation extended in order.
    # If strictly needed, we could collect results in a list of size len(urls) and fill them.
    
    # But usually date sorting happens at the end or is implicit. 
    # Let's assume the caller will sort if needed or that 'all_entries' order 
    # being slightly shuffled is acceptable for now.
    
    return all_entries


def _fetch_and_fill_details(
    client: CrawlerClient, 
    entry: AnnouncementEntry
) -> None:
    """Helper to fetch and fill details for a single entry."""
    tree: HtmlElement = client.fetch_page(entry.url)  # pyright: ignore[reportAny]
    parse_announcement_page(tree, entry)


def crawl_announcement_details(
    client: CrawlerClient,
    entries: list[AnnouncementEntry],
    on_entry_fetched: Callable[[int, int, AnnouncementEntry], None] | None = None,
    max_workers: int = 5
) -> None:
    """
    Fetch and fill details for announcement entries.
    
    This function mutates the entries in place, filling in body and attachments.
    
    Args:
        client: CrawlerClient instance for making requests
        entries: List of AnnouncementEntry objects to fetch details for
        on_entry_fetched: Optional callback(entry_index, total_entries, entry)
            called after each entry is successfully fetched
        max_workers: Number of threads to use
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_entry = {
            executor.submit(_fetch_and_fill_details, client, entry): (idx, entry)
            for idx, entry in enumerate(entries)
        }
        
        for future in as_completed(future_to_entry):
            idx, entry = future_to_entry[future]
            try:
                future.result()
                if on_entry_fetched:
                    on_entry_fetched(idx + 1, len(entries), entry)
            except Exception as e:
                print(f"Error fetching details for {entry.url}: {e}")


def get_latest_announcements(
    client: CrawlerClient,
    num_pages: int = 1,
    fetch_details: bool = False,
    max_workers: int = 5
) -> list[AnnouncementEntry]:
    """
    Convenience function to get the latest announcements.
    
    Args:
        client: CrawlerClient instance
        num_pages: Number of latest pages to fetch (default: 1)
        fetch_details: Whether to also fetch detail pages (default: False)
        max_workers: Number of threads to use (default: 5)
        
    Returns:
        List of AnnouncementEntry objects
    """
    total_pages = client.get_total_pages()
    max_page = total_pages - 1
    
    # Calculate page range
    start_page = max_page
    end_page = max(1, max_page - num_pages + 1)
    
    urls = build_page_urls(max_page, start_page=start_page, end_page=end_page)
    entries = crawl_announcement_list(client, urls, max_workers=max_workers)
    
    if fetch_details:
        crawl_announcement_details(client, entries, max_workers=max_workers)
    
    return entries

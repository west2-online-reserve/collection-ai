"""
FZU Academic Affairs Announcement Crawler

A modular crawler for fetching announcements from https://jwch.fzu.edu.cn
"""
from rich import pretty, traceback

# Setup rich for better error output
pretty.install()
_ = traceback.install(show_locals=True)

# Public API exports
from fzu_jxtz_crawler.models import Attachment, AnnouncementEntry
from fzu_jxtz_crawler.url_utils import BASE_URL, MAIN_PAGE_URL, construct_full_url, build_page_urls
from fzu_jxtz_crawler.parsers import (
    parse_announcement_list,
    parse_announcement_page,
    parse_entry_from_li,
    parse_attachment_from_li,
    extract_total_page,
)
from fzu_jxtz_crawler.client import CrawlerClient
from fzu_jxtz_crawler.crawler import (
    crawl_announcement_list,
    crawl_announcement_details,
    get_latest_announcements,
)

from fzu_jxtz_crawler.text_utils import (
    normalize_line_endings,
    clean_whitespace,
    clean_text,
)
from fzu_jxtz_crawler.export import entries_to_dataframe, export_to_csv

__all__ = [
    # Models
    "Attachment",
    "AnnouncementEntry",
    # URL utilities
    "BASE_URL",
    "MAIN_PAGE_URL",
    "construct_full_url",
    "build_page_urls",
    # Parsers
    "parse_announcement_list",
    "parse_announcement_page",
    "parse_entry_from_li",
    "parse_attachment_from_li",
    "extract_total_page",
    # Client
    "CrawlerClient",
    # Crawler
    "crawl_announcement_list",
    "crawl_announcement_details",
    "get_latest_announcements",
    # Text utils
    "normalize_line_endings",
    "clean_whitespace",
    "clean_text",
    # Export
    "entries_to_dataframe",
    "export_to_csv",
]
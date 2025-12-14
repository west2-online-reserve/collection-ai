"""
Test suite for URL utilities in src/url_utils.py
"""
import pytest
from fzu_jxtz_crawler import construct_full_url, build_page_urls, MAIN_PAGE_URL


class TestConstructFullUrl:
    """Test cases for the construct_full_url function"""
    
    def test_relative_url_with_leading_slash(self):
        """Test relative URL with leading /"""
        result = construct_full_url("/info/1013/16745.htm")
        assert result == "https://jwch.fzu.edu.cn/info/1013/16745.htm"
    
    def test_relative_url_without_leading_slash(self):
        """Test relative URL without leading /"""
        result = construct_full_url("info/1013/16745.htm")
        assert result == "https://jwch.fzu.edu.cn/info/1013/16745.htm"
    
    def test_absolute_url(self):
        """Test already absolute URL"""
        result = construct_full_url("https://example.com/page")
        assert result == "https://example.com/page"
    
    def test_relative_url_with_query_params(self):
        """Test relative URL with query parameters"""
        href = "/system/_content/download.jsp?urltype=news.DownloadAttachUrl&owner=1744984858&wbfileid=16738452"
        expected = "https://jwch.fzu.edu.cn/system/_content/download.jsp?urltype=news.DownloadAttachUrl&owner=1744984858&wbfileid=16738452"
        result = construct_full_url(href)
        assert result == expected
    
    def test_empty_href(self):
        """Test empty href returns base URL"""
        result = construct_full_url("")
        assert result == "https://jwch.fzu.edu.cn"
    
    def test_custom_base_url(self):
        """Test using a custom base URL"""
        result = construct_full_url("/page", base_url="https://custom.example.com")
        assert result == "https://custom.example.com/page"


class TestBuildPageUrls:
    """Test cases for the build_page_urls function"""
    
    def test_build_all_pages(self):
        """Test building URLs for all pages"""
        urls = build_page_urls(max_page=5)
        assert len(urls) == 6  # main page + 5 numbered pages
        assert urls[0] == MAIN_PAGE_URL
        assert urls[1] == "https://jwch.fzu.edu.cn/jxtz/5.htm"
        assert urls[-1] == "https://jwch.fzu.edu.cn/jxtz/1.htm"
    
    def test_build_range(self):
        """Test building URLs for a specific range"""
        urls = build_page_urls(max_page=10, start_page=5, end_page=3)
        assert len(urls) == 4  # main page + pages 5,4,3
        assert urls[0] == MAIN_PAGE_URL
        assert urls[1] == "https://jwch.fzu.edu.cn/jxtz/5.htm"
        assert urls[-1] == "https://jwch.fzu.edu.cn/jxtz/3.htm"
    
    def test_build_single_page(self):
        """Test building URL for a single page"""
        urls = build_page_urls(max_page=10, start_page=5, end_page=5)
        assert len(urls) == 2  # main page + page 5
        assert urls[1] == "https://jwch.fzu.edu.cn/jxtz/5.htm"
    
    def test_start_exceeds_max(self):
        """Test that start_page is clamped to max_page"""
        urls = build_page_urls(max_page=5, start_page=10, end_page=3)
        assert urls[1] == "https://jwch.fzu.edu.cn/jxtz/5.htm"
    
    def test_end_below_one(self):
        """Test that end_page is clamped to 1"""
        urls = build_page_urls(max_page=5, start_page=3, end_page=-5)
        assert urls[-1] == "https://jwch.fzu.edu.cn/jxtz/1.htm"


@pytest.mark.parametrize("href,expected,description", [
    ("/info/1013/16745.htm", "https://jwch.fzu.edu.cn/info/1013/16745.htm", "Relative URL with /"),
    ("info/1013/16745.htm", "https://jwch.fzu.edu.cn/info/1013/16745.htm", "Relative URL without /"),
    ("https://example.com/page", "https://example.com/page", "Absolute URL"),
    ("/system/_content/download.jsp?urltype=news.DownloadAttachUrl&owner=1744984858&wbfileid=16738452",
     "https://jwch.fzu.edu.cn/system/_content/download.jsp?urltype=news.DownloadAttachUrl&owner=1744984858&wbfileid=16738452",
     "URL with query params"),
    ("", "https://jwch.fzu.edu.cn", "Empty href"),
])
def test_construct_full_url_parametrized(href: str, expected: str, description: str) -> None:
    """Parametrized test for various URL construction scenarios"""
    result = construct_full_url(href)
    assert result == expected, f"Failed for: {description}"

"""
Test suite for HTML parsing functions in src/parsers.py
"""
from typing import cast

import pytest
from datetime import datetime
from lxml import html
from lxml.html import HtmlElement

from fzu_jxtz_crawler import (
    parse_announcement_list,
    parse_announcement_page,
    parse_entry_from_li,
    parse_attachment_from_li,
    extract_total_page,
)
from fzu_jxtz_crawler import AnnouncementEntry


class TestParseEntryFromLi:
    """Test cases for parsing announcement entries from li elements"""
    
    def test_parse_valid_entry(self, sample_announcement_li: HtmlElement) -> None:
        """Test parsing a valid announcement entry"""
        entry = parse_entry_from_li(sample_announcement_li)
        
        assert entry is not None
        assert entry.title == "关于公布2025-2026学年各学院转专业实施细则的通知"
        assert "【教学运行】" in entry.issuer
        assert entry.url == "https://jwch.fzu.edu.cn/info/1036/14352.htm"
        assert entry.pub_date.year == 2025
        assert entry.pub_date.month == 11
        assert entry.pub_date.day == 14
        assert entry.body is None
        assert entry.attachments is None
    
    def test_parse_entry_missing_link(self):
        """Test parsing entry with missing link returns None"""
        html_content = """
        <li>
            <span class="doclist_time">2025-11-14</span>
            【教学运行】
        </li>
        """
        tree = html.fromstring(html_content)
        li_element = cast(list[HtmlElement], tree.xpath("//li"))[0]
        
        entry = parse_entry_from_li(li_element)
        assert entry is None
    
    def test_parse_entry_with_absolute_url(self):
        """Test parsing entry with absolute URL"""
        html_content = """
        <li>
            <span class="doclist_time">2025-11-14</span>
            【教学运行】
            <a href="https://example.com/page" target="_blank" title="外部链接">外部链接</a>
        </li>
        """
        tree = html.fromstring(html_content)
        li_element = cast(list[HtmlElement], tree.xpath("//li"))[0]
        
        entry = parse_entry_from_li(li_element)
        
        assert entry is not None
        assert entry.url == "https://example.com/page"
    
    def test_parse_entry_missing_date(self):
        """Test parsing entry with missing date uses current date"""
        html_content = """
        <li>
            【教学运行】
            <a href="info/1036/14352.htm" target="_blank">测试通知</a>
        </li>
        """
        tree = html.fromstring(html_content)
        li_element = cast(list[HtmlElement], tree.xpath("//li"))[0]
        
        entry = parse_entry_from_li(li_element)
        
        assert entry is not None
        # Date should be close to now (within the same day)
        assert entry.pub_date.year == datetime.now().year
    
    def test_parse_entry_empty_href(self):
        """Test parsing entry with empty href returns None"""
        html_content = """
        <li>
            <span class="doclist_time">2025-11-14</span>
            【教学运行】
            <a href="" target="_blank">测试通知</a>
        </li>
        """
        tree = html.fromstring(html_content)
        li_element = cast(list[HtmlElement], tree.xpath("//li"))[0]
        
        entry = parse_entry_from_li(li_element)
        assert entry is None


class TestParseAttachmentFromLi:
    """Test cases for parsing attachments from li elements"""
    
    def test_parse_valid_attachment(self, sample_attachment_li: HtmlElement) -> None:
        """Test parsing a valid attachment"""
        attachment = parse_attachment_from_li(sample_attachment_li)
        
        assert attachment is not None
        assert attachment.name == "2501高等数学A（下）期末考-12.7上午.xlsx"
        assert attachment.download_times == 146
        assert attachment.owner_code == "1744984858"
        assert attachment.file_code == "16738452"
        assert "jwch.fzu.edu.cn" in attachment.url
        assert attachment.local_path is None
    
    def test_parse_attachment_no_link(self):
        """Test parsing attachment with no link returns None"""
        html_content = "<li>附件【无链接】已下载<span>0</span>次</li>"
        tree = html.fromstring(html_content)
        li_element = cast(list[HtmlElement], tree.xpath("//li"))[0]
        
        attachment = parse_attachment_from_li(li_element)
        assert attachment is None
    
    def test_parse_attachment_no_download_count(self):
        """Test parsing attachment without download count defaults to 0"""
        html_content = """
        <li>附件【<a href="/system/_content/download.jsp?owner=123&wbfileid=456">test.pdf</a>】已下载</li>
        """
        tree = html.fromstring(html_content)
        li_element = cast(list[HtmlElement], tree.xpath("//li"))[0]
        
        attachment = parse_attachment_from_li(li_element)
        
        assert attachment is not None
        assert attachment.download_times == 0
    
    def test_parse_attachment_empty_href(self):
        """Test parsing attachment with empty href returns None"""
        html_content = """
        <li>附件【<a href="">test.pdf</a>】已下载<span>10</span>次</li>
        """
        tree = html.fromstring(html_content)
        li_element = cast(list[HtmlElement], tree.xpath("//li"))[0]
        
        attachment = parse_attachment_from_li(li_element)
        assert attachment is None
    
    def test_parse_attachment_missing_query_params(self):
        """Test parsing attachment with missing query params"""
        html_content = """
        <li>附件【<a href="/system/_content/download.jsp">test.pdf</a>】已下载<span>5</span>次</li>
        """
        tree = html.fromstring(html_content)
        li_element = cast(list[HtmlElement], tree.xpath("//li"))[0]
        
        attachment = parse_attachment_from_li(li_element)
        
        assert attachment is not None
        assert attachment.owner_code == ""
        assert attachment.file_code == ""


class TestParseAnnouncementList:
    """Test cases for parsing full announcement lists"""
    
    def test_parse_announcement_list(self, sample_announcement_list_html: str) -> None:
        """Test parsing a full announcement list"""
        tree = html.fromstring(sample_announcement_list_html)
        
        entries = parse_announcement_list(tree)
        
        assert len(entries) == 2
        assert entries[0].title == "通知1"
        assert entries[1].title == "通知2"
        assert "【教学运行】" in entries[0].issuer
        assert "【考试通知】" in entries[1].issuer
    
    def test_parse_empty_list(self):
        """Test parsing page with no announcements returns empty list"""
        html_content = """
        <html><body><div></div></body></html>
        """
        tree = html.fromstring(html_content)
        
        entries = parse_announcement_list(tree)
        
        assert entries == []


class TestParseAnnouncementPage:
    """Test cases for parsing announcement detail pages"""
    
    def test_parse_announcement_page_with_attachments(self, sample_announcement_detail_html: str) -> None:
        """Test parsing announcement page with attachments"""
        tree = html.fromstring(sample_announcement_detail_html)
        entry = AnnouncementEntry(
            url="https://example.com/test",
            title="Test",
            issuer="Test",
            pub_date=datetime.now(),
            body=None,
            attachments=None
        )
        
        parse_announcement_page(tree, entry)
        
        assert entry.body is not None
        assert "公告正文内容" in entry.body
        assert entry.attachments is not None
        assert len(entry.attachments) == 1
        assert entry.attachments[0].name == "test.xlsx"
    
    def test_parse_announcement_page_no_attachments(self):
        """Test parsing announcement page without attachments"""
        html_content = """
        <html><body>
            <div id="vsb_content"><div>正文内容</div></div>
        </body></html>
        """
        tree = html.fromstring(html_content)
        entry = AnnouncementEntry(
            url="https://example.com/test",
            title="Test",
            issuer="Test",
            pub_date=datetime.now(),
            body=None,
            attachments=None
        )
        
        parse_announcement_page(tree, entry)
        
        assert entry.body == "正文内容"
        assert entry.attachments == []
    
    def test_parse_announcement_page_missing_body(self):
        """Test parsing announcement page with missing body raises error"""
        html_content = """<html><body><div>No body here</div></body></html>"""
        tree = html.fromstring(html_content)
        entry = AnnouncementEntry(
            url="https://example.com/test",
            title="Test",
            issuer="Test",
            pub_date=datetime.now(),
            body=None,
            attachments=None
        )
        
        with pytest.raises(ValueError, match="Could not find the announcement body"):
            parse_announcement_page(tree, entry)


class TestExtractTotalPage:
    """Test cases for extracting total page count"""
    
    def test_extract_total_page_missing_element(self):
        """Test extracting total page when element is missing raises error"""
        html_content = "<html><body><div>No page count</div></body></html>"
        tree = html.fromstring(html_content)
        
        with pytest.raises(ValueError, match="Could not find total page count"):
            extract_total_page(tree)

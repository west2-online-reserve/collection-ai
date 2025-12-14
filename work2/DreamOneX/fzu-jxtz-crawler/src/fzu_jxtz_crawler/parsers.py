"""
HTML parsing functions for the FZU announcement crawler.

All functions in this module are pure - they take HtmlElement inputs
and return parsed data without any network I/O.
"""
from datetime import datetime
from typing import cast
from urllib.parse import parse_qs, urlparse

from lxml.html import HtmlElement  # pyright: ignore[reportAny]

from .models import Attachment, AnnouncementEntry
from .url_utils import construct_full_url


# XPath constants
TOTAL_PAGE_XPATH = r"/html/body/div[1]/div[2]/div[2]/div/div/div[3]/div[2]/div[1]/div/span[1]/span[9]/a"
ANNOUNCEMENT_LIST_XPATH = "/html/body/div[1]/div[2]/div[2]/div/div/div[3]/div[1]/ul/li"
ANNOUNCEMENT_BODY_XPATH = "//*[@id='vsb_content']/div"
ATTACHMENT_LIST_XPATH = "/html/body/div/div[2]/div[2]/form/div/div[1]/div/ul"


def extract_total_page(tree: HtmlElement) -> int:  # pyright: ignore[reportAny]
    """
    Extract the total page count from a page listing HTML tree.
    
    Args:
        tree: Parsed HTML tree of the main announcement listing page
        
    Returns:
        Total page count as integer
        
    Raises:
        ValueError: If the total page count element is not found
    """
    elements = cast(list[HtmlElement], tree.xpath(TOTAL_PAGE_XPATH))  # pyright: ignore[reportAny]
    if not elements:
        raise ValueError("Could not find total page count on the page")
    
    element_text: str | None = elements[0].text  # pyright: ignore[reportAny]
    if element_text is None:
        raise ValueError("Could not find total page count on the page")
    
    return int(element_text)


def parse_attachment_from_li(li: HtmlElement) -> Attachment | None:  # pyright: ignore[reportAny]
    """
    Parse a single attachment from a <li> element.
    
    Expected HTML structure:
    <li>附件【<a href="/system/_content/download.jsp?...">filename.xlsx</a>】
        已下载<span id="nattach16738452">146</span>次</li>
    
    Args:
        li: The <li> HtmlElement containing attachment info
        
    Returns:
        Attachment object if valid, None otherwise
    """
    # Extract attachment link
    a_elements = cast(list[HtmlElement], li.xpath(".//a"))  # pyright: ignore[reportAny]
    if not a_elements:
        return None
    
    a_element: HtmlElement = a_elements[0]  # pyright: ignore[reportAny]
    name: str = cast(str, a_element.text_content()).strip()  # pyright: ignore[reportAny]
    href: str | None = cast(str | None, a_element.get("href"))  # pyright: ignore[reportAny]
    
    if not href:
        return None
    
    # Parse owner_code and file_code from URL
    # Example: /system/_content/download.jsp?urltype=news.DownloadAttachUrl&owner=1744984858&wbfileid=16738452
    parsed_url = urlparse(href)
    query_params = parse_qs(parsed_url.query)
    
    owner: str = query_params.get("owner", [""])[0]
    file_code: str = query_params.get("wbfileid", [""])[0]
    
    # Extract download times from span element
    download_times: int = 0
    span_elements = cast(list[HtmlElement], li.xpath(".//span"))  # pyright: ignore[reportAny]
    if span_elements:
        download_times_text: str = cast(str, span_elements[0].text_content()).strip()  # pyright: ignore[reportAny]
        try:
            download_times = int(download_times_text)
        except ValueError:
            download_times = 0
    
    # Construct full URL
    full_url = construct_full_url(href)
    
    return Attachment(
        name=name,
        download_times=download_times,
        url=full_url,
        owner_code=owner,
        file_code=file_code,
        local_path=None
    )


def parse_entry_from_li(li: HtmlElement) -> AnnouncementEntry | None:  # pyright: ignore[reportAny]
    """
    Parse a single announcement entry from a <li> element.
    
    Expected HTML structure:
    <li>
        <span class="doclist_time">2025-11-14</span>
        【教学运行】
        <a href="info/1036/14352.htm" target="_blank" title="...">标题</a>
    </li>
    
    Args:
        li: The <li> HtmlElement containing entry info
        
    Returns:
        AnnouncementEntry object if valid, None otherwise
    """
    # Extract date from span element
    span_elements = cast(list[HtmlElement], li.xpath(".//span[@class='doclist_time']"))  # pyright: ignore[reportAny]
    date_str: str = cast(str, span_elements[0].text_content()).strip() if span_elements else ""  # pyright: ignore[reportAny]
    
    # Parse date - format should be YYYY-MM-DD
    try:
        pub_date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        # Use current date as fallback if date parsing fails
        pub_date = datetime.now()
    
    # Extract URL and title from <a> tag
    a_elements = cast(list[HtmlElement], li.xpath(".//a"))  # pyright: ignore[reportAny]
    if not a_elements:
        return None
    
    a_element: HtmlElement = a_elements[0]  # pyright: ignore[reportAny]
    title: str = cast(str, a_element.text_content()).strip()  # pyright: ignore[reportAny]
    href: str | None = cast(str | None, a_element.get("href"))  # pyright: ignore[reportAny]
    
    if not href:
        return None
    
    # Construct full URL
    full_url = construct_full_url(href)
    
    # Extract issuer from text content between span and a tag
    # Get all text content from li, then remove date and title parts
    li_text: str = cast(str, li.text_content())  # pyright: ignore[reportAny]
    
    # Remove date and title to get issuer
    issuer: str = ""
    # Try to get issuer from span tail first (more robust)
    if span_elements and span_elements[0].tail:  # pyright: ignore[reportAny]
        issuer = span_elements[0].tail.strip()  # pyright: ignore[reportAny]
    
    # Fallback to text replacement if tail is empty
    if not issuer:
        issuer = li_text.replace(date_str, "").replace(title, "").strip()
    
    return AnnouncementEntry(
        url=full_url,
        title=title,
        issuer=issuer,
        pub_date=pub_date,
        body=None,  # Will be filled when parsing detail page
        attachments=None  # Will be filled when parsing detail page
    )


def parse_announcement_list(tree: HtmlElement) -> list[AnnouncementEntry]:  # pyright: ignore[reportAny]
    """
    Parse all announcement entries from a listing page.
    
    Args:
        tree: Parsed HTML tree of an announcement listing page
        
    Returns:
        List of AnnouncementEntry objects
    """
    li_elements: list[HtmlElement] = cast(list[HtmlElement], tree.xpath(ANNOUNCEMENT_LIST_XPATH))  # pyright: ignore[reportAny]
    
    result: list[AnnouncementEntry] = []
    for li in li_elements:  # pyright: ignore[reportAny]
        entry = parse_entry_from_li(li)
        if entry:
            result.append(entry)
    
    return result


def parse_announcement_page(tree: HtmlElement, entry: AnnouncementEntry) -> None:  # pyright: ignore[reportAny]
    """
    Parse the announcement detail page and fill the entry with body and attachments.
    
    This function mutates the entry in place.
    
    Args:
        tree: Parsed HTML tree of an announcement detail page
        entry: AnnouncementEntry to fill with details
        
    Raises:
        ValueError: If the announcement body is not found
    """
    # Parse body content
    body_nodes = cast(list[HtmlElement], tree.xpath(ANNOUNCEMENT_BODY_XPATH))  # pyright: ignore[reportAny]
    if not body_nodes:
        raise ValueError("Could not find the announcement body")
    body_node: HtmlElement = body_nodes[0]  # pyright: ignore[reportAny]
    
    # Remove script and style tags to prevent code/css from appearing in text
    for element in body_node.xpath(".//script | .//style"):  # pyright: ignore[reportAny]
        element.drop_tree()  # pyright: ignore[reportAny]
        
    entry.body = cast(str, body_node.text_content())  # pyright: ignore[reportAny]

    # Parse attachments
    ul_nodes = cast(list[HtmlElement], tree.xpath(ATTACHMENT_LIST_XPATH))  # pyright: ignore[reportAny]
    if not ul_nodes:
        entry.attachments = []
        return
    
    ul_element: HtmlElement = ul_nodes[0]  # pyright: ignore[reportAny]
    li_elements: list[HtmlElement] = cast(list[HtmlElement], ul_element.xpath(".//li"))  # pyright: ignore[reportAny]
    
    attachments: list[Attachment] = []
    for li in li_elements:  # pyright: ignore[reportAny]
        attachment = parse_attachment_from_li(li)
        if attachment:
            attachments.append(attachment)
    
    entry.attachments = attachments

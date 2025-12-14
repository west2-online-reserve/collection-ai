"""
Data models for the FZU announcement crawler.
"""
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Attachment:
    """
    Represents an attachment in an announcement.
    
    If local_path is not None, it means the attachment has been downloaded.
    """
    name: str
    download_times: int
    url: str
    owner_code: str
    file_code: str
    local_path: str | None


@dataclass
class AnnouncementEntry:
    """
    Represents an announcement entry.
    
    body and attachments could be None, which may indicate that the crawler
    has not fetched the detail page yet, or that there are simply no attachments.
    """
    url: str
    title: str
    issuer: str
    pub_date: datetime

    body: str | None
    attachments: list[Attachment] | None

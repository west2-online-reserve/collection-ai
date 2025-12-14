"""
CSV export functionality for the FZU announcement crawler.

This module provides functions to export announcement data to CSV format using pandas.
"""
from pathlib import Path

import pandas as pd

from .models import AnnouncementEntry
from .text_utils import clean_text


def entries_to_dataframe(
    entries: list[AnnouncementEntry],
    clean_text_fields: bool = True
) -> pd.DataFrame:
    """
    Convert announcement entries to a pandas DataFrame.
    
    Args:
        entries: List of AnnouncementEntry objects
        clean_text_fields: If True, apply text cleaning to all text fields
        
    Returns:
        DataFrame with columns: url, title, issuer, pub_date, body,
        attachment_count, attachment_names
    """
    data: list[dict[str, str | int | None]] = []
    
    for entry in entries:
        # Apply text cleaning if requested
        title = clean_text(entry.title) if clean_text_fields and entry.title else entry.title
        issuer = clean_text(entry.issuer) if clean_text_fields and entry.issuer else entry.issuer
        body = clean_text(entry.body) if clean_text_fields and entry.body else entry.body
        
        # Extract attachment information
        attachment_count = len(entry.attachments) if entry.attachments else 0
        attachment_names = ""
        if entry.attachments:
            names = [att.name for att in entry.attachments]
            attachment_names = "; ".join(names)
            if clean_text_fields:
                attachment_names = clean_text(attachment_names)
        
        data.append({
            "url": entry.url,
            "title": title,
            "issuer": issuer,
            "pub_date": entry.pub_date.strftime("%Y-%m-%d"),
            "body": body if body else "",
            "attachment_count": attachment_count,
            "attachment_names": attachment_names,
            "attachment_urls": "; ".join([att.url for att in entry.attachments]) if entry.attachments else ""
        })
    
    return pd.DataFrame(data)


def export_to_csv(
    entries: list[AnnouncementEntry],
    filepath: str | Path,
    clean_text_fields: bool = True
) -> None:
    """
    Export announcement entries to CSV file.
    
    Args:
        entries: List of AnnouncementEntry objects to export
        filepath: Path to the output CSV file
        clean_text_fields: If True, apply text cleaning to all text fields
    """
    df = entries_to_dataframe(entries, clean_text_fields=clean_text_fields)
    
    # Export to CSV with proper line ending handling
    df.to_csv(
        filepath,
        index=False,
        encoding="utf-8",
        lineterminator="\n"  # Force LF line endings
    )

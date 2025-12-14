"""
Text cleaning utilities for the FZU announcement crawler.

This module provides functions for normalizing line endings and cleaning whitespace.
"""
import re


def normalize_line_endings(text: str) -> str:
    """
    Normalize line endings to LF (\\n).
    
    Converts CRLF (\\r\\n) and CR (\\r) to LF (\\n).
    
    Args:
        text: Input text with potentially mixed line endings
        
    Returns:
        Text with normalized LF line endings
    """
    # Replace CRLF with LF first, then replace any remaining CR with LF
    text = text.replace('\r\n', '\n')
    text = text.replace('\r', '\n')
    return text


def clean_whitespace(text: str) -> str:
    """
    Clean meaningless whitespace from text.
    
    - Removes leading and trailing whitespace
    - Collapses multiple consecutive spaces into single space
    - Collapses multiple consecutive newlines into at most two newlines
    - Removes spaces around newlines
    
    Args:
        text: Input text with potentially excessive whitespace
        
    Returns:
        Text with cleaned whitespace
    """
    # Strip leading and trailing whitespace
    text = text.strip()
    
    # Collapse multiple spaces into single space (but preserve newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove spaces around newlines
    text = re.sub(r' *\n *', '\n', text)
    
    # Collapse multiple newlines into at most two (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text


def clean_text(text: str) -> str:
    """
    Apply all text cleaning operations.
    
    Combines normalize_line_endings() and clean_whitespace().
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text with normalized line endings and whitespace
    """
    text = normalize_line_endings(text)
    text = clean_whitespace(text)
    return text

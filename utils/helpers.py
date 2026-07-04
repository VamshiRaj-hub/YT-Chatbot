"""
helpers.py
----------
Utility / helper functions used across the application.
Covers URL validation, video-ID extraction, and miscellaneous helpers.
"""

import re
import logging
from urllib.parse import urlparse, parse_qs
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YouTube URL patterns
# ---------------------------------------------------------------------------
_YT_PATTERNS = [
    # Standard watch URL:  https://www.youtube.com/watch?v=VIDEO_ID
    r"(?:youtube\.com\/watch\?(?:.*&)?v=)([A-Za-z0-9_-]{11})",
    # Short URL:           https://youtu.be/VIDEO_ID
    r"(?:youtu\.be\/)([A-Za-z0-9_-]{11})",
    # Embed URL:           https://www.youtube.com/embed/VIDEO_ID
    r"(?:youtube\.com\/embed\/)([A-Za-z0-9_-]{11})",
    # Shorts URL:          https://www.youtube.com/shorts/VIDEO_ID
    r"(?:youtube\.com\/shorts\/)([A-Za-z0-9_-]{11})",
]


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract the 11-character YouTube video ID from any standard YouTube URL.

    Args:
        url: Raw URL string entered by the user.

    Returns:
        11-character video ID string, or None if not found.
    """
    for pattern in _YT_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def is_valid_youtube_url(url: str) -> bool:
    """
    Validate whether a given string is a recognisable YouTube URL.

    Args:
        url: Raw URL string.

    Returns:
        True if valid, False otherwise.
    """
    if not url or not url.strip():
        return False
    video_id = extract_video_id(url.strip())
    return video_id is not None


def sanitize_url(url: str) -> str:
    """
    Strip leading/trailing whitespace from a URL string.

    Args:
        url: Raw URL string.

    Returns:
        Cleaned URL string.
    """
    return url.strip()


def format_docs(docs: list) -> str:
    """
    Join a list of LangChain Document objects into a single string.

    Args:
        docs: List of Document objects.

    Returns:
        Concatenated page_content strings separated by double newlines.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def truncate_text(text: str, max_chars: int = 300) -> str:
    """
    Truncate a string to max_chars characters, appending '…' if cut.

    Args:
        text:      Input string.
        max_chars: Maximum character length.

    Returns:
        Truncated (or original) string.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def build_video_url(video_id: str) -> str:
    """
    Construct a canonical YouTube watch URL from a video ID.

    Args:
        video_id: 11-character YouTube video ID.

    Returns:
        Full watch URL string.
    """
    return f"https://www.youtube.com/watch?v={video_id}"

"""
loader.py
---------
Handles loading YouTube video transcripts via LangChain's YoutubeLoader,
and fetches video metadata (title, author, publish date, views, description)
using pytubefix — the actively maintained pytube fork.

The metadata is prepended as a dedicated Document so the RAG chain can
answer questions like "when was this posted?", "who is the host?", etc.
"""

import logging
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain_core.documents import Document

from utils.helpers import extract_video_id, build_video_url

logger = logging.getLogger(__name__)


class TranscriptLoadError(Exception):
    """Raised when a transcript cannot be loaded for any reason."""


def _fetch_metadata(video_id: str) -> dict:
    """
    Fetch video metadata using pytubefix.

    Args:
        video_id: YouTube video ID.

    Returns:
        Dict with keys: title, author, publish_date, view_count,
        length, description. Falls back to "Unknown" on any error.
    """
    result = {
        "title": "Unknown",
        "author": "Unknown",
        "publish_date": "Unknown",
        "view_count": "Unknown",
        "length": "Unknown",
        "description": "",
    }
    try:
        from pytubefix import YouTube
        yt = YouTube(build_video_url(video_id))

        result["title"]        = yt.title or "Unknown"
        result["author"]       = yt.author or "Unknown"
        result["view_count"]   = str(yt.views) if yt.views else "Unknown"
        result["length"]       = str(yt.length) + " seconds" if yt.length else "Unknown"
        result["description"]  = (yt.description or "")[:600]

        # publish_date is a datetime object or None
        if yt.publish_date:
            result["publish_date"] = yt.publish_date.strftime("%B %d, %Y")

        logger.info(
            "Metadata fetched: title='%s', author='%s', published='%s'",
            result["title"], result["author"], result["publish_date"],
        )
    except Exception as exc:
        logger.warning("Could not fetch video metadata via pytubefix: %s", exc)

    return result


def _build_metadata_doc(meta: dict, video_id: str) -> Document:
    """
    Build a Document whose content is a human-readable metadata summary.
    This gets embedded into ChromaDB so the retriever can surface it
    for metadata questions.
    """
    desc = meta["description"]
    lines = [
        "=== VIDEO METADATA ===",
        f"Title        : {meta['title']}",
        f"Channel/Host : {meta['author']}",
        f"Published    : {meta['publish_date']}",
        f"View Count   : {meta['view_count']}",
        f"Duration     : {meta['length']}",
        f"Video ID     : {video_id}",
        f"URL          : https://www.youtube.com/watch?v={video_id}",
    ]
    if desc:
        lines += ["", "Description  :", desc]

    return Document(
        page_content="\n".join(lines),
        metadata={"source": video_id, "video_id": video_id, "type": "metadata"},
    )


def load_transcript(url: str) -> tuple[list[Document], str]:
    """
    Load the transcript of a YouTube video as a list of LangChain Documents.

    The first document in the returned list is always a metadata summary.
    The remaining documents contain the transcript text.

    Args:
        url: A valid YouTube video URL.

    Returns:
        A tuple of:
            - List of Document objects (metadata doc first, then transcript).
            - The canonical video ID string.

    Raises:
        TranscriptLoadError: If the transcript cannot be fetched.
    """
    video_id = extract_video_id(url)
    if not video_id:
        raise TranscriptLoadError(
            "Could not extract a valid video ID from the provided URL. "
            "Please check the URL and try again."
        )

    canonical_url = build_video_url(video_id)
    logger.info("Loading transcript for video ID: %s", video_id)

    # ------------------------------------------------------------------
    # Load transcript (without add_video_info — pytube is broken)
    # ------------------------------------------------------------------
    try:
        loader = YoutubeLoader.from_youtube_url(
            canonical_url,
            add_video_info=False,
            transcript_format=TranscriptFormat.TEXT,
            language=["en", "en-US", "en-GB"],
        )
        docs = loader.load()
    except Exception as exc:
        error_msg = str(exc).lower()

        if "no transcript" in error_msg or "transcripts disabled" in error_msg:
            raise TranscriptLoadError(
                "This video does not have an available transcript. "
                "Please try a video with subtitles or auto-generated captions enabled."
            ) from exc
        if "private" in error_msg or "unavailable" in error_msg:
            raise TranscriptLoadError(
                "This video is private or unavailable. "
                "Please use a publicly accessible video."
            ) from exc
        if "could not retrieve" in error_msg:
            raise TranscriptLoadError(
                "Could not retrieve the transcript. "
                "The video may not have captions in any supported language."
            ) from exc

        logger.exception("Unexpected error while loading transcript")
        raise TranscriptLoadError(
            f"An unexpected error occurred while loading the transcript: {exc}"
        ) from exc

    if not docs:
        raise TranscriptLoadError(
            "The transcript was loaded but appears to be empty. "
            "Please try a different video."
        )

    # ------------------------------------------------------------------
    # Fetch metadata separately via pytubefix and prepend as a Document
    # ------------------------------------------------------------------
    meta = _fetch_metadata(video_id)
    metadata_doc = _build_metadata_doc(meta, video_id)
    all_docs = [metadata_doc] + docs

    logger.info(
        "Loaded %d transcript doc(s) + 1 metadata doc for video %s",
        len(docs), video_id,
    )
    return all_docs, video_id

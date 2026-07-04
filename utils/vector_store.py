"""
vector_store.py
---------------
Manages the persistent ChromaDB vector store.

Responsibilities:
  - Split raw transcript Documents into chunks.
  - Embed chunks and upsert them into ChromaDB.
  - Load an existing ChromaDB collection.
  - Clear / delete an existing collection.
  - Expose a retriever for downstream RAG usage.
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from utils.config import (
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVER_K,
    RETRIEVER_FETCH_K,
)
from utils.embeddings import get_embeddings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------

def split_documents(docs: list[Document]) -> list[Document]:
    """
    Split a list of Documents into smaller chunks using
    RecursiveCharacterTextSplitter.

    Args:
        docs: Raw transcript Documents (usually one large document).

    Returns:
        List of chunked Document objects with preserved metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],  # try larger separators first
    )
    chunks = splitter.split_documents(docs)
    logger.info(
        "Split %d raw document(s) into %d chunk(s) "
        "(chunk_size=%d, overlap=%d)",
        len(docs),
        len(chunks),
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )
    return chunks


# ---------------------------------------------------------------------------
# Vector store helpers
# ---------------------------------------------------------------------------

def build_vector_store(chunks: list[Document], video_id: str) -> Chroma:
    """
    Embed the provided chunks and persist them in ChromaDB.

    Clears the existing collection first to prevent duplicate embeddings
    when the same (or a different) video is re-processed.

    Args:
        chunks:   List of chunked Document objects.
        video_id: YouTube video ID used to tag metadata.

    Returns:
        Chroma vector store instance loaded with the embedded chunks.
    """
    # Tag every chunk with the video_id for traceability
    for chunk in chunks:
        chunk.metadata["video_id"] = video_id

    embeddings = get_embeddings()

    # Always clear the collection before inserting new chunks so that
    # re-processing a video never doubles the document count.
    try:
        existing = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_DIR,
        )
        existing.delete_collection()
        logger.info("Cleared existing ChromaDB collection before re-embedding.")
    except Exception:
        pass  # collection may not exist yet on first run — that's fine

    logger.info(
        "Embedding %d chunks and persisting to ChromaDB (collection: %s)…",
        len(chunks),
        CHROMA_COLLECTION_NAME,
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_DB_DIR,
    )

    logger.info("ChromaDB vector store built and persisted successfully.")
    return vector_store


def load_vector_store() -> Optional[Chroma]:
    """
    Load an existing ChromaDB collection from disk.

    Returns:
        Chroma instance if the collection exists and contains documents,
        otherwise None.
    """
    embeddings = get_embeddings()
    try:
        vector_store = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_DIR,
        )
        # Check that the collection is non-empty
        count = vector_store._collection.count()
        if count == 0:
            logger.info("ChromaDB collection exists but is empty.")
            return None
        logger.info("Loaded ChromaDB collection with %d document(s).", count)
        return vector_store
    except Exception as exc:
        logger.warning("Could not load ChromaDB collection: %s", exc)
        return None


def clear_vector_store() -> bool:
    """
    Delete all documents from the ChromaDB collection.

    Returns:
        True if cleared successfully, False if an error occurred.
    """
    embeddings = get_embeddings()
    try:
        vector_store = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_DIR,
        )
        vector_store.delete_collection()
        logger.info("ChromaDB collection '%s' deleted.", CHROMA_COLLECTION_NAME)
        return True
    except Exception as exc:
        logger.error("Failed to clear ChromaDB collection: %s", exc)
        return False


def get_retriever(vector_store: Chroma):
    """
    Create a retriever from the given Chroma vector store.

    Args:
        vector_store: A loaded / built Chroma instance.

    Returns:
        LangChain VectorStoreRetriever configured for similarity search.
    """
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_FETCH_K},
    )


def get_document_count(vector_store: Chroma) -> int:
    """
    Return the total number of embedded documents in the collection.

    Args:
        vector_store: Chroma instance.

    Returns:
        Integer document count.
    """
    try:
        return vector_store._collection.count()
    except Exception:
        return 0

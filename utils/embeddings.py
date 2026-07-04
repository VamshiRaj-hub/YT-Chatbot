"""
embeddings.py
-------------
Initialises and caches the HuggingFace sentence-transformer embedding model.
The model is downloaded once and reused across the application session.
"""

import logging
from langchain_huggingface import HuggingFaceEmbeddings

from utils.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

# Module-level singleton so the model is loaded only once per process.
_embeddings_instance: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a (cached) HuggingFaceEmbeddings instance.

    The model is loaded from HuggingFace Hub on first call and reused
    on all subsequent calls within the same Python process.

    Returns:
        HuggingFaceEmbeddings object configured with all-MiniLM-L6-v2.
    """
    global _embeddings_instance

    if _embeddings_instance is None:
        logger.info("Loading HuggingFace embedding model: %s", EMBEDDING_MODEL_NAME)
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},   # use CPU; swap to "cuda" if GPU available
            encode_kwargs={"normalize_embeddings": True},  # cosine-similarity ready
        )
        logger.info("Embedding model loaded successfully.")

    return _embeddings_instance

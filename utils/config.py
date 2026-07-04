"""
config.py
---------
Central configuration module for the YouTube Chatbot application.
Loads environment variables and defines application-wide constants.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env file (if present)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
CHROMA_DB_DIR: str = str(BASE_DIR / "chroma_db")

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# ChromaDB settings
# ---------------------------------------------------------------------------
CHROMA_COLLECTION_NAME: str = "youtube_transcripts"

# ---------------------------------------------------------------------------
# Text splitting settings
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = 1500        # characters per chunk
CHUNK_OVERLAP: int = 150      # overlap between consecutive chunks

# ---------------------------------------------------------------------------
# Retriever settings
# ---------------------------------------------------------------------------
RETRIEVER_K: int = 5          # number of chunks to retrieve per query
RETRIEVER_FETCH_K: int = 30   # candidate pool for MMR diversity selection

# ---------------------------------------------------------------------------
# Gemini model settings
# ---------------------------------------------------------------------------
GEMINI_MODEL: str = "gemini-2.5-flash"
GEMINI_TEMPERATURE: float = 0.2
GEMINI_MAX_OUTPUT_TOKENS: int = 1024

# ---------------------------------------------------------------------------
# Google API Key (can be overridden at runtime via the UI)
# ---------------------------------------------------------------------------
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

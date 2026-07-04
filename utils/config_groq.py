"""
config_groq.py
--------------
Groq-specific configuration constants.
All other app settings (embeddings, ChromaDB, chunking) remain in config.py.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Groq model settings
# ---------------------------------------------------------------------------
GROQ_MODEL: str = "llama-3.3-70b-versatile"   # best quality on free tier
GROQ_TEMPERATURE: float = 0.2
GROQ_MAX_OUTPUT_TOKENS: int = 1024

# ---------------------------------------------------------------------------
# Groq API Key (can be overridden at runtime via the UI)
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

"""
Central configuration for the Internal Knowledge Navigator.
All paths / secrets are read from environment variables (.env file).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _get(name: str, default=None, required: bool = False):
    val = os.getenv(name, default)
    if required and not val:
        raise RuntimeError(f"Missing required environment variable '{name}'.")
    return val


# ---- Corpus / index -------------------------------------------------
CORPUS_DIR = _get("CORPUS_DIR", str(ROOT_DIR / "data" / "corpus"))
CHROMA_PATH = _get("CHROMA_PATH", str(ROOT_DIR / "chroma_db"))
CHROMA_COLLECTION = _get("CHROMA_COLLECTION", "nimbus_wiki")
MANIFEST_PATH = _get("MANIFEST_PATH", str(Path(CORPUS_DIR) / "manifest.json"))

# "tfidf"  -> zero-dependency, matches the original notebook, decent for small corpora
# "sbert"  -> sentence-transformers embeddings, much better semantic recall, needs a model download
EMBEDDING_BACKEND = _get("EMBEDDING_BACKEND", "tfidf").lower()
SBERT_MODEL_NAME = _get("SBERT_MODEL_NAME", "all-MiniLM-L6-v2")
TFIDF_VECTORIZER_PATH = _get(
    "TFIDF_VECTORIZER_PATH", str(ROOT_DIR / "chroma_db" / "tfidf_vectorizer.pkl")
)

TOP_K = int(_get("TOP_K", "5"))

# ---- LLM (Anthropic) --------------------------------------------------


# ---- AWS Bedrock ------------------------------------------------------
AWS_ACCESS_KEY_ID = _get("AWS_ACCESS_KEY_ID", required=True)
AWS_SECRET_ACCESS_KEY = _get("AWS_SECRET_ACCESS_KEY", required=True)
AWS_REGION = _get("AWS_REGION", "us-east-1")

GENERATION_MODEL = _get("GENERATION_MODEL", "amazon.nova-micro-v1:0")
MAX_TOKENS = int(_get("MAX_TOKENS", "1024"))
# ---- MCP server ---------------------------------------------------------
# Path to the python interpreter + script used to launch our custom MCP server
MCP_SERVER_SCRIPT = _get(
    "MCP_SERVER_SCRIPT", str(ROOT_DIR / "mcp_server" / "server.py")
)


# How many days back counts as "recent" for the What's New sidebar feature
# (app/whats_new.py), which calls the MCP list_recent_updates tool.
RECENT_UPDATES_DAYS = int(_get("RECENT_UPDATES_DAYS", "7"))

# ---- Langfuse observability ------------------------------------------
LANGFUSE_PUBLIC_KEY = _get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = _get("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = _get("LANGFUSE_HOST", "https://cloud.langfuse.com")

# ---- Guardrails --------------------------------------------------------
# Minimum fraction of answer content-words that must be traceable back to the
# retrieved context before we trust the answer as "grounded".
GROUNDEDNESS_THRESHOLD = float(_get("GROUNDEDNESS_THRESHOLD", "0.25"))

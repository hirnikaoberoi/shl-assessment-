import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

CATALOG_PATH = ROOT_DIR / "data" / "processed" / "catalog.json"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

MAX_CLARIFICATION_ROUNDS = 2
MAX_RECOMMENDATIONS = 10
DEFAULT_RECOMMENDATIONS = 5
RETRIEVAL_TOP_K = 15
LLM_TIMEOUT_SECONDS = 20

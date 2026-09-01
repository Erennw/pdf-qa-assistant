"""Central configuration.

Every tunable value lives here so experiments only touch one file.
Nothing in this module talks to a model or a database -- it only
describes *where* things are and *how* they are configured.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Loads the .env file sitting next to this project into os.environ.
# Call it once, at import time, before anything reads a key.
load_dotenv()

# --- Paths ---------------------------------------------------------------
# config.py lives in rag/, so parent.parent is the project root.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# --- Embedding model -----------------------------------------------------
# Runs locally on CPU. First run downloads ~1.1 GB into the HuggingFace cache.
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# The e5 family is trained with asymmetric prefixes: a question and a
# document are encoded differently on purpose. Dropping these prefixes
# measurably degrades retrieval.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# --- Chunking ------------------------------------------------------------
# Characters, not tokens. Start here, then let the eval set decide.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_CHUNK_SIZE = 100  # fragments shorter than this are dropped as noise

# --- Retrieval -----------------------------------------------------------
COLLECTION_NAME = "pdf_chunks"
TOP_K = 4

# Cosine distance = 1 - cosine similarity, so LOWER is more similar.
# This value is deliberately loose. Calibrate it against the eval set
# before trusting it: run eval/run.py, look at the distances of
# correct vs. wrong hits, and set the threshold between the two bands.
MAX_DISTANCE = 0.30

# --- Answering LLM -------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_MAX_TOKENS = 800

# There is deliberately no temperature setting. Sampling parameters
# (temperature, top_p, top_k) were removed from the Messages API in 2026
# and are rejected if sent. Grounding is enforced by the system prompt in
# answer.py, which is the more robust place for it in any case.

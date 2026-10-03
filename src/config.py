"""Central settings for CyberShield SME. Change values here, not inside other files."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"            # put ACSC PDFs / saved web pages here
PROCESSED_DIR = ROOT / "data" / "processed"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
SOURCES_FILE = ROOT / "data" / "sources.csv"  # filename -> title, url, date accessed
EMB_FILE = PROCESSED_DIR / "embeddings.npy"
RESULTS_DIR = ROOT / "results"

# Chunking (words, not tokens: simple and good enough for a small corpus)
CHUNK_SIZE = 220
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 5
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, free, runs on CPU

# Generation (Ollama runs locally and is free)
OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "llama3.1:8b"   # on a weak laptop use "llama3.2:3b"
JUDGE_MODEL = "llama3.1:8b" # LLM used to score answers in evaluation
TEMPERATURE = 0.0

# Phrase the assistant must use when the sources don't cover the question.
# Evaluation counts this as "unanswered" (the Walert metric).
NO_ANSWER = "I can't find official ACSC guidance on that in my sources."

INCIDENT_TYPES = ["phishing", "ransomware", "bec", "account_compromise"]

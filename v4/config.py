"""Runtime settings for v4. Keep model and storage choices in one place."""

MODEL = "qwen2.5:1.5b"
EMBED_MODEL = "nomic-embed-text"
MEMORY_FILE = "memory.json"
REPORTS_DIR = "reports"
CACHE_FILE = "scrape_cache.sqlite3"
PAGE_CHARS = 5000
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 8
TOP_K_ROUND2 = 6
MAX_PREVIOUS_CHARS = 2000
SCRAPE_CACHE_TTL_SECONDS = 24 * 60 * 60
MAX_SCRAPE_WORKERS = 8

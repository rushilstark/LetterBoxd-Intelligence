"""
Central configuration for Letterboxd Intelligence Engine.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
EMBEDDINGS_DIR  = BASE_DIR / "embeddings"
LETTERBOXD_DIR  = Path("/Users/rushilreddy/letterboxd")

# Letterboxd CSV files
DIARY_CSV    = LETTERBOXD_DIR / "diary.csv"
RATINGS_CSV  = LETTERBOXD_DIR / "ratings.csv"
REVIEWS_CSV  = LETTERBOXD_DIR / "reviews.csv"
WATCHLIST_CSV = LETTERBOXD_DIR / "watchlist.csv"

# SQLite cache for TMDB responses
TMDB_CACHE_DB = DATA_DIR / "tmdb_cache.db"

# ChromaDB collections
CHROMA_COMMUNITY_COLLECTION = "community_reviews"
CHROMA_MOVIE_DNA_COLLECTION  = "movie_dna"
CHROMA_MY_LENS_COLLECTION    = "my_lens"

# ── Ollama ─────────────────────────────────────────────────────────────────────
# Best free local embedding model. Falls back to nomic-embed-text if not found.
OLLAMA_EMBED_MODEL = "mxbai-embed-large"  # 334MB, 1024-dim, excellent quality
OLLAMA_HOST        = "http://localhost:11434"

# ── TMDB ───────────────────────────────────────────────────────────────────────
# ── TMDB ───────────────────────────────────────────────────────────────────────
TMDB_API_KEY      = os.environ.get("TMDB_API_KEY", "your_tmdb_api_key_here")
TMDB_BASE_URL     = "https://api.themoviedb.org/3"
TMDB_RATE_LIMIT   = 0.25   # seconds between requests (4 req/s, well within 40/10s)

# ── Embedding ──────────────────────────────────────────────────────────────────
EMBED_BATCH_SIZE  = 8       # process N texts before persisting to ChromaDB
MAX_TEXT_CHARS    = 1800    # truncate text before embedding

# ── Predictor ──────────────────────────────────────────────────────────────────
TOP_K_SIMILAR     = 20      # how many nearest neighbours to use for prediction
WEIGHTS = {
    "my_lens":   0.50,      # your own lens is most personal
    "movie_dna": 0.30,      # movie DNA captures objective similarity
    "community": 0.20,      # community signal as tiebreaker
}

# ── UI ─────────────────────────────────────────────────────────────────────────
APP_TITLE = "🎬 Letterboxd Intelligence"
ACCENT    = "#E9A84C"       # warm amber – letterboxd-ish

# ── Cross-domain APIs ─────────────────────────────────────────────────────────
LASTFM_API_KEY        = os.environ.get('LASTFM_API_KEY', '')
SPOTIFY_CLIENT_ID     = os.environ.get('SPOTIFY_CLIENT_ID', '')
SPOTIFY_CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET', '')
GOOGLE_BOOKS_API_KEY  = os.environ.get('GOOGLE_BOOKS_API_KEY', '')

# ── Cross-domain ChromaDB ─────────────────────────────────────────────────────
CHROMA_CULTURAL_COLLECTION = 'cultural_fingerprint'

# ── Cross-domain cache files ──────────────────────────────────────────────────
LASTFM_CACHE   = DATA_DIR / 'lastfm_cache.json'
BOOKS_CACHE    = DATA_DIR / 'books_cache.json'
SPOTIFY_CACHE  = DATA_DIR / 'spotify_cache.json'

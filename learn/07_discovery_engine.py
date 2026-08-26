"""
LESSON 7: HONEST DISCOVERY — Finding Movies You Don't Know Yet
Without using your watchlist. That's cheating.

Real discovery: go into the world (TMDB popular/top-rated/similar chains)
and predict which unknown movies you'd love.

Run: python learn/07_discovery_engine.py
"""
import sys; sys.path.insert(0, ".")
import requests
import json
import time
import numpy as np
from src.config import TMDB_API_KEY, TMDB_BASE_URL, TMDB_CACHE_DB
from src.embedder import embed_text
from src.vector_store import query_similar, collection_count
from src.config import CHROMA_MY_LENS_COLLECTION, CHROMA_MOVIE_DNA_COLLECTION
from src.ingest import load_letterboxd_data

print("\n" + "="*60)
print("  LESSON 7: HONEST DISCOVERY ENGINE")
print("="*60)

if collection_count(CHROMA_MY_LENS_COLLECTION) == 0:
    print("⚠  Run pipeline.py --limit 50 first"); exit(1)

# ─── Load what you've already seen ────────────────────────────
df = load_letterboxd_data()
already_seen = set(df["title"].str.lower().str.strip())
print(f"\n  Movies you've already seen: {len(already_seen)}")
print(f"  These will be excluded from recommendations.\n")

# ─── Fetch candidates from TMDB ───────────────────────────────
def tmdb_get(endpoint, params={}):
    params["api_key"] = TMDB_API_KEY
    r = requests.get(f"{TMDB_BASE_URL}/{endpoint}", params=params, timeout=10)
    time.sleep(0.25)
    return r.json() if r.status_code == 200 else {}

def get_candidates(n_pages=3):
    """
    Get candidate movies from multiple TMDB sources.
    No watchlist. Real discovery.
    """
    candidates = {}

    sources = [
        ("Top Rated",  "movie/top_rated"),
        ("Popular",    "movie/popular"),
        ("Now Playing","movie/now_playing"),
    ]

    for source_name, endpoint in sources:
        for page in range(1, n_pages + 1):
            data = tmdb_get(endpoint, {"language": "en-US", "page": page})
            for m in data.get("results", []):
                title = m.get("title", "")
                year  = (m.get("release_date", "")[:4]) or None
                tid   = m.get("id")
                if title and tid:
                    candidates[tid] = {
                        "title":   title,
                        "year":    year,
                        "source":  source_name,
                        "tmdb_id": tid,
                        "overview": m.get("overview", ""),
                        "genres":  m.get("genre_ids", []),
                        "tmdb_rating": m.get("vote_average", 0),
                    }

    return list(candidates.values())

print("[1] Fetching candidates from TMDB (top rated + popular)...")
candidates = get_candidates(n_pages=2)
print(f"  Found {len(candidates)} candidate films from TMDB\n")

# ─── Filter out already seen ───────────────────────────────────
unseen = [c for c in candidates
          if c["title"].lower().strip() not in already_seen]
print(f"[2] After removing movies you've seen: {len(unseen)} unseen films\n")

# ─── Score each unseen film ────────────────────────────────────
print("[3] Scoring each film against your taste embedding...\n")
print(f"  {'Title':<40} {'TMDB':>5}  {'Your Pred':>9}  {'Source'}")
print(f"  {'─'*40} {'─'*5}  {'─'*9}  {'─'*12}")

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

results = []
for film in unseen[:60]:  # process first 60 for speed in lesson mode
    # Build text from TMDB overview (no review since we haven't seen it)
    text = f"Title: {film['title']}\nYear: {film['year']}\nPlot: {film['overview']}"
    emb  = embed_text(text)
    if not emb:
        continue

    # Query my_lens — what movies in your history feel similar?
    qr = query_similar(CHROMA_MY_LENS_COLLECTION, emb, n_results=10)
    if not qr["ids"][0]:
        continue

    ratings, sims = [], []
    for i in range(len(qr["ids"][0])):
        meta = qr["metadatas"][0][i]
        dist = qr["distances"][0][i]
        r    = meta.get("rating")
        if r is not None:
            try:
                ratings.append(float(r))
                sims.append(1 - dist)
            except (ValueError, TypeError):
                pass

    if not ratings:
        continue

    sims_arr    = np.array(sims)
    ratings_arr = np.array(ratings)
    pred        = np.dot(sims_arr, ratings_arr) / sims_arr.sum()
    pred_r      = round(pred * 2) / 2

    results.append({**film, "predicted": pred, "predicted_r": pred_r})

# Sort by predicted rating
results.sort(key=lambda x: x["predicted"], reverse=True)

# Print top 20
for r in results[:20]:
    title = r["title"][:38]
    print(f"  {title:<40} {r['tmdb_rating']:>5.1f}  {r['predicted_r']:>6.1f}★    {r['source']}")

# ─── The anti-watchlist philosophy ────────────────────────────
print(f"""
─────────────────────────────────────────────────────────────
[4] WHY WATCHLIST IS CHEATING

  Your watchlist = movies you've already decided you want to see.
  That's selection bias — you're predicting ratings for a sample
  you CHOSE because you suspected you'd like them.

  REAL discovery asks: out of ALL films ever made,
  which ones would you have loved but never discovered?

  The sources above (TMDB top_rated, popular) are the "universe" —
  unbiased candidates you might or might not know about.

  EVEN BETTER sources for discovery (harder to implement):
    - Letterboxd's popular this week (requires their API/scraping)
    - TMDB similar_movies chains: start from a 5★ film,
      follow similar_movies links 3 hops → massive candidate set
    - Films from directors/actors you consistently rate high
    - Award nominees you haven't seen (Oscars, Cannes, etc.)

─────────────────────────────────────────────────────────────
[5] THE EXPLORATION vs EXPLOITATION TRADEOFF

  This is a core concept in RL and recommenders.

  EXPLOITATION: recommend what the model thinks you'll love
  → Safe. Predictable. Gets boring. Filter bubble.

  EXPLORATION: recommend some things the model ISN'T sure about
  → Risky. Occasionally wrong. But this is how you discover new taste.

  Industry solution: Epsilon-Greedy
    90% of the time: recommend top prediction (exploit)
    10% of the time: recommend a random candidate (explore)

  Better: Thompson Sampling
    Model uncertainty on each prediction.
    Recommend films where you're uncertain — that's where you learn most.

  Your system currently: pure exploitation (always top predicted rating)
  Next step: add an uncertainty score and surface some "wildcards"
""")

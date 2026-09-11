"""
TMDB API client with SQLite caching.
Fetches movie metadata and community reviews.
"""
import sqlite3
import json
import time
import requests
from pathlib import Path
from typing import Optional

from src.config import TMDB_BASE_URL, TMDB_RATE_LIMIT, TMDB_CACHE_DB, DATA_DIR
from src.config_store import get_key
import os

def get_tmdb_key():
    k = get_key("TMDB_API_KEY", "")
    if not k:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            k = os.environ.get("TMDB_API_KEY", "")
        except: pass
    if not k:
        try:
            from src.config import TMDB_API_KEY
            k = TMDB_API_KEY
        except: pass
    return k

# ── DB setup ──────────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(TMDB_CACHE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS movie_cache (
            title TEXT, year INTEGER, data TEXT,
            PRIMARY KEY (title, year)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reviews_cache (
            tmdb_id INTEGER PRIMARY KEY, data TEXT
        )
    """)
    conn.commit()
    return conn


# ── TMDB Helpers ──────────────────────────────────────────────────────────────

def _tmdb_get(endpoint: str, params: dict = {}) -> Optional[dict]:
    """Raw GET with rate limiting."""
    url = f"{TMDB_BASE_URL}/{endpoint}"
    params["api_key"] = get_tmdb_key()
    try:
        resp = requests.get(url, params=params, timeout=10)
        time.sleep(TMDB_RATE_LIMIT)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"[tmdb] Error {endpoint}: {e}")
    return None


def _search_movie(title: str, year: Optional[int] = None) -> Optional[dict]:
    """Search TMDB for a movie. Returns best match."""
    params = {"query": title, "include_adult": False}
    if year and not (isinstance(year, float) and str(year) == "nan"):
        params["year"] = int(year)
    data = _tmdb_get("search/movie", params)
    if data and data.get("results"):
        return data["results"][0]
    # Retry without year
    if year:
        data = _tmdb_get("search/movie", {"query": title, "include_adult": False})
        if data and data.get("results"):
            return data["results"][0]
    return None


def _get_details(tmdb_id: int) -> Optional[dict]:
    """Get full movie details with credits and keywords."""
    return _tmdb_get(f"movie/{tmdb_id}", {"append_to_response": "credits,keywords"})


def _get_reviews(tmdb_id: int) -> list[str]:
    """Get up to 8 community reviews from TMDB."""
    conn = _get_conn()
    row = conn.execute("SELECT data FROM reviews_cache WHERE tmdb_id=?", (tmdb_id,)).fetchone()
    if row:
        conn.close()
        return json.loads(row[0])

    data = _tmdb_get(f"movie/{tmdb_id}/reviews", {"language": "en-US"})
    texts = []
    if data and data.get("results"):
        for r in data["results"][:8]:
            content = r.get("content", "").strip()
            if content and len(content) > 50:
                texts.append(content[:1500])   # cap individual review length

    conn.execute("INSERT OR REPLACE INTO reviews_cache VALUES (?,?)",
                 (tmdb_id, json.dumps(texts)))
    conn.commit()
    conn.close()
    return texts


# ── Public API ────────────────────────────────────────────────────────────────

def get_movie_info(title: str, year=None) -> dict:
    """
    Returns a dict with:
        tmdb_id, title, year, overview, genres, director, cast,
        keywords, runtime, origin_country, vote_average,
        community_reviews (list[str]), found (bool)
    """
    conn = _get_conn()

    # Cache lookup
    safe_year = int(year) if year and str(year) not in ("nan", "None", "<NA>") else 0
    row = conn.execute(
        "SELECT data FROM movie_cache WHERE title=? AND year=?",
        (title, safe_year)
    ).fetchone()
    conn.close()

    if row:
        data = json.loads(row[0])
        # Backfill poster_url for entries cached before poster support was added
        if "poster_url" not in data:
            pp = data.get("poster_path", "")
            data["poster_url"] = f"https://image.tmdb.org/t/p/w300{pp}" if pp else ""
        return data

    # Fetch fresh
    result = {
        "tmdb_id": None, "title": title, "year": safe_year or None,
        "overview": "", "genres": [], "director": "", "cast": [],
        "keywords": [], "runtime": None, "origin_country": [],
        "vote_average": None, "poster_path": "", "poster_url": "",
        "community_reviews": [], "found": False,
    }

    search = _search_movie(title, year if safe_year else None)
    if not search:
        _cache_result(title, safe_year, result)
        return result

    tmdb_id = search["id"]
    details = _get_details(tmdb_id)
    if not details:
        _cache_result(title, safe_year, result)
        return result

    # Parse credits
    crew   = details.get("credits", {}).get("crew", [])
    cast   = details.get("credits", {}).get("cast", [])
    kws    = details.get("keywords", {}).get("keywords", [])
    director = next((p["name"] for p in crew if p.get("job") == "Director"), "")

    poster_path = details.get("poster_path") or search.get("poster_path", "")
    result.update({
        "tmdb_id":        tmdb_id,
        "title":          details.get("title", title),
        "year":           safe_year or (details.get("release_date", "")[:4] or None),
        "overview":       details.get("overview", ""),
        "genres":         [g["name"] for g in details.get("genres", [])],
        "director":       director,
        "cast":           [c["name"] for c in cast[:6]],
        "keywords":       [k["name"] for k in kws[:15]],
        "runtime":        details.get("runtime"),
        "origin_country": details.get("origin_country", []),
        "vote_average":   details.get("vote_average"),
        "poster_path":    poster_path,
        "poster_url":     f"https://image.tmdb.org/t/p/w300{poster_path}" if poster_path else "",
        "community_reviews": _get_reviews(tmdb_id),
        "found":          True,
    })

    _cache_result(title, safe_year, result)
    return result


def _cache_result(title: str, year: int, data: dict):
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO movie_cache VALUES (?,?,?)",
        (title, year, json.dumps(data))
    )
    conn.commit()
    conn.close()


# ── Text builders ─────────────────────────────────────────────────────────────

def build_movie_dna_text(info: dict) -> str:
    """Build a rich text representation of the movie's objective DNA."""
    parts = []
    if info.get("title"):
        parts.append(f"Title: {info['title']}")
    if info.get("year"):
        parts.append(f"Year: {info['year']}")
    if info.get("director"):
        parts.append(f"Director: {info['director']}")
    if info.get("cast"):
        parts.append(f"Cast: {', '.join(info['cast'])}")
    if info.get("genres"):
        parts.append(f"Genres: {', '.join(info['genres'])}")
    if info.get("overview"):
        parts.append(f"Plot: {info['overview']}")
    if info.get("keywords"):
        parts.append(f"Themes: {', '.join(info['keywords'])}")
    if info.get("runtime"):
        parts.append(f"Runtime: {info['runtime']} minutes")
    if info.get("origin_country"):
        parts.append(f"Country: {', '.join(info['origin_country'])}")
    if info.get("vote_average"):
        parts.append(f"TMDB Rating: {info['vote_average']:.1f}/10")
    return "\n".join(parts)


def build_community_text(info: dict) -> str:
    """Concatenate TMDB community reviews."""
    reviews = info.get("community_reviews", [])
    if not reviews:
        # Fall back to overview + genres so the collection isn't empty
        return build_movie_dna_text(info)
    return "\n\n---\n\n".join(reviews)


def build_my_lens_text(title: str, rating, review: str,
                        is_rewatch: bool, tags: str, watch_date=None) -> str:
    """Build the personal experience text for this movie."""
    parts = [f"Movie: {title}"]
    if rating and str(rating) not in ("nan", "None"):
        parts.append(f"My Rating: {rating}/5")
    if is_rewatch:
        parts.append("I rewatched this film.")
    if watch_date:
        parts.append(f"Watched: {str(watch_date)[:10]}")
    if tags and str(tags).strip():
        parts.append(f"Tags: {tags}")
    if review and len(str(review).strip()) > 10:
        parts.append(f"My Review:\n{review.strip()}")
    return "\n".join(parts)

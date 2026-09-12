"""
persona_profile.py

Builds a rich, data-driven profile of Rushil's cinematic taste from ALL available
Letterboxd data. This is computed ONCE at startup and injected into every persona
chat prompt as hard facts — preventing the LLM from hallucinating opinions.

Key AI Engineering principle: "Grounding". Give the LLM facts it can cite,
not vague style instructions it will ignore.
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import pandas as pd

from src.config import TMDB_CACHE_DB
from src.ingest import load_letterboxd_data

# Cache so we only compute once per app session
_profile_cache: Dict = None


def _query_tmdb_cache(titles_years: List[Tuple[str, int]]) -> Dict[str, Dict]:
    """
    Batch-query the local SQLite TMDB cache for genres and directors.
    Returns {title_lower: {genres: [...], director: str}}
    """
    if not TMDB_CACHE_DB.exists():
        return {}

    result = {}
    try:
        conn = sqlite3.connect(str(TMDB_CACHE_DB))
        cur = conn.cursor()
        cur.execute("SELECT title, year, data FROM movie_cache")
        rows = cur.fetchall()
        conn.close()

        for title, year, data_json in rows:
            try:
                data = json.loads(data_json)
                key = title.lower().strip()
                result[key] = {
                    "genres":   data.get("genres", []),
                    "director": data.get("director", ""),
                    "cast":     data.get("cast", [])[:3],
                }
            except Exception:
                pass
    except Exception:
        pass

    return result


def build_profile(df: pd.DataFrame = None) -> Dict:
    """
    Build a comprehensive taste profile from all Letterboxd data.
    Called once, cached for the session.
    """
    global _profile_cache
    if _profile_cache is not None:
        return _profile_cache

    if df is None:
        df = load_letterboxd_data()

    if len(df) == 0:
        _profile_cache = {
            "total_movies": 0, "total_rated": 0, "total_reviews": 0,
            "rewatches": 0, "avg_rating": 0, "five_star_count": 0,
            "one_star_count": 0, "peak_year": 2024, "dist_str": "",
            "north_stars": [], "top_genres": [], "low_genres": [],
            "top_directors": [], "top_decade": None,
            "best_reviews": [], "_df": df, "_reviewed": pd.DataFrame(),
        }
        return _profile_cache

    rated = df[df["final_rating"].notna()].copy()
    if "my_review" not in df.columns:
        df["my_review"] = ""
    df["my_review"] = df["my_review"].fillna("").astype(str)
    reviewed = df[df["my_review"].str.len() > 20].copy()
    reviewed["rev_len"] = reviewed["my_review"].str.len()

    # ── Basic stats ──────────────────────────────────────────────────────────
    avg_rating = float(rated["final_rating"].mean()) if len(rated) else 0.0
    total = len(df)
    total_reviews = len(reviewed)
    total_rated = len(rated)
    if "is_rewatch" not in df.columns:
        df["is_rewatch"] = False
    rewatches = int(df["is_rewatch"].fillna(False).sum())
    five_star_count = int((rated["final_rating"] == 5.0).sum())
    one_star_count  = int((rated["final_rating"] <= 1.0).sum())

    # Watch timeline
    if "watch_date" in df.columns:
        dated = df.dropna(subset=["watch_date"]).copy()
        try:
            dated["watch_date"] = pd.to_datetime(dated["watch_date"], errors="coerce")
            dated = dated.dropna(subset=["watch_date"])
            dated["year_watched"] = dated["watch_date"].dt.year
            peak_year_s = dated.groupby("year_watched").size()
            peak_year = int(peak_year_s.idxmax()) if len(peak_year_s) > 0 else 2024
        except Exception:
            peak_year = 2024
    else:
        peak_year = 2024

    # ── North star films (rewatched AND rated 5★) ────────────────────────────
    if "final_rating" in df.columns and "is_rewatch" in df.columns:
        north_stars = df[(df["is_rewatch"] == True) & (df["final_rating"] == 5.0)]["title"].tolist()
    else:
        north_stars = []
    north_stars_display = north_stars[:8] if north_stars else []

    # ── Films rated 5★ (for director/genre extraction) ───────────────────────
    loved = rated[rated["final_rating"] == 5.0]["title"].str.lower().tolist()
    hated = rated[rated["final_rating"] <= 1.0]["title"].str.lower().tolist()

    # ── TMDB cache lookup for genres/directors ───────────────────────────────
    tmdb = _query_tmdb_cache([])  # Load all

    loved_genres: Counter = Counter()
    hated_genres: Counter = Counter()
    loved_directors: Counter = Counter()

    for title_lower in loved:
        info = tmdb.get(title_lower, {})
        for g in info.get("genres", []):
            loved_genres[g] += 1
        d = info.get("director", "")
        if d:
            loved_directors[d] += 1

    for title_lower in hated:
        info = tmdb.get(title_lower, {})
        for g in info.get("genres", []):
            hated_genres[g] += 1

    top_genres = [g for g, _ in loved_genres.most_common(6)]
    low_genres  = [g for g, _ in hated_genres.most_common(4)]
    top_directors = [d for d, c in loved_directors.most_common(8) if c >= 2]

    # ── Best long reviews (for writing style anchors) ────────────────────────
    best_reviews = reviewed.nlargest(20, "rev_len")[["title", "final_rating", "my_review"]].to_dict("records")

    # ── Rating distribution summary ───────────────────────────────────────────
    dist = rated["final_rating"].value_counts().sort_index()
    dist_str = " | ".join([f"{k}★:{int(v)}" for k, v in dist.items()])

    # ── Decades you love ─────────────────────────────────────────────────────
    df_dec = rated.copy()
    df_dec["decade"] = (pd.to_numeric(df_dec["year"], errors="coerce") // 10 * 10)
    decade_avg = df_dec.groupby("decade")["final_rating"].agg(["mean", "count"]).dropna()
    decade_avg = decade_avg[decade_avg["count"] >= 10]
    top_decade_row = decade_avg["mean"].idxmax() if len(decade_avg) > 0 else None
    top_decade = int(top_decade_row) if top_decade_row is not None else None

    _profile_cache = {
        # Raw stats
        "total_movies": total,
        "total_rated": total_rated,
        "total_reviews": total_reviews,
        "rewatches": rewatches,
        "avg_rating": round(avg_rating, 2),
        "five_star_count": five_star_count,
        "one_star_count": one_star_count,
        "peak_year": peak_year,
        "dist_str": dist_str,

        # Taste DNA
        "north_stars": north_stars_display,
        "top_genres": top_genres,
        "low_genres": low_genres,
        "top_directors": top_directors,
        "top_decade": top_decade,

        # Writing samples
        "best_reviews": best_reviews,

        # Raw df reference for dynamic lookups
        "_df": df,
        "_reviewed": reviewed,
    }

    return _profile_cache


def format_profile_card(profile: Dict) -> str:
    """
    Formats the profile as a structured text block for injection into the LLM prompt.
    Written like a data brief, not a vague description.
    """
    ns = ", ".join(profile["north_stars"]) if profile["north_stars"] else "Tamasha, Fight Club, Matrix"
    tg = ", ".join(profile["top_genres"]) if profile["top_genres"] else "Drama, Thriller"
    lg = ", ".join(profile["low_genres"]) if profile["low_genres"] else "(varies)"
    td = ", ".join(profile["top_directors"]) if profile["top_directors"] else "(see reviews)"

    return f"""RUSHIL'S CINEMATIC IDENTITY — BUILT FROM ACTUAL DATA (3507 FILMS):

STATS:
- Watched {profile['total_movies']} films, rated {profile['total_rated']}, written reviews for {profile['total_reviews']}
- Average rating: {profile['avg_rating']}★ (generous — {profile['five_star_count']} films got 5★, only {profile['one_star_count']} got 0.5★ or 1★)
- Rating spread: {profile['dist_str']}
- Rewatched {profile['rewatches']} films. Rewatching = love.
- Peak watching year: {profile['peak_year']} (when you were most obsessed with cinema)
- Favourite decade of filmmaking: {profile['top_decade']}s (highest avg rating)

NORTH STAR FILMS (5★ AND rewatched — these define you):
{ns}

GENRES YOU CONSISTENTLY RATE HIGHEST:
{tg}

GENRES YOU CONSISTENTLY RATE LOWEST:
{lg}

DIRECTORS YOU'VE GIVEN MULTIPLE 5-STARS TO:
{td}

USE THESE FACTS. When someone asks what you love, cite specific films. When asked about a genre, use the data above. Don't make up opinions — you already have 3507 of them recorded."""


def get_writing_samples(profile: Dict, n: int = 15) -> str:
    """
    Returns the n longest/richest reviews as style anchors.
    These show the LLM HOW Rushil writes, not just what he thinks.
    """
    samples = profile["best_reviews"][:n]
    parts = []
    for r in samples:
        rating_str = f"{r['final_rating']}★" if r.get("final_rating") else "★"
        # Take first 400 chars — enough to capture the voice, not exhaust the context window
        snippet = str(r["my_review"])[:400].replace("\r\n", "\n").strip()
        parts.append(f'[{r["title"]} — {rating_str}]\n"{snippet}"')
    return "\n\n".join(parts)


def get_direct_opinion(title: str, profile: Dict) -> str:
    """
    If Rushil has reviewed or rated this specific film, return it verbatim.
    This is the most important context — his actual opinion.
    """
    df = profile.get("_df")
    if df is None:
        return ""

    # Case-insensitive title match
    matches = df[df["title"].str.lower() == title.lower().strip()]
    if len(matches) == 0:
        # Fuzzy: title contains
        matches = df[df["title"].str.lower().str.contains(title.lower().strip(), regex=False, na=False)]

    if len(matches) == 0:
        return ""

    row = matches.iloc[0]
    review = str(row.get("my_review", "")).strip()
    rating = row.get("final_rating")
    rewatch = row.get("is_rewatch", False)

    parts = [f"YOUR ACTUAL OPINION ON '{row['title']}':"]
    parts.append(f"Rating: {rating}★" + (" (rewatched — you loved it enough to revisit)" if rewatch else ""))
    if review and len(review) > 10:
        parts.append(f"Your full review:\n{review}")
    else:
        parts.append("(You rated this but didn't write a review)")

    return "\n".join(parts)

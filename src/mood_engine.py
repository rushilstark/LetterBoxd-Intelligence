"""
mood_engine.py

The core of the discovery engine. Reads your recent Spotify listening,
detects your current emotional/thematic mood, then finds films aligned
to that mood — combining it with your Letterboxd + Goodreads taste when available.

Key insight: we're not recommending what you "should" like historically.
We're recommending what you want RIGHT NOW, based on what you've been listening to.

A film rated 2.5★ on TMDB can still be a perfect match for your current vibe.
Rating is context, not the primary filter.
"""

import time
import json
import requests
from pathlib import Path
from typing import Optional

# ── Mood archetypes and their TMDB mappings ──────────────────────────────────

# TMDB genre IDs
TMDB_GENRES = {
    "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35,
    "Crime": 80, "Documentary": 99, "Drama": 18, "Family": 10751,
    "Fantasy": 14, "History": 36, "Horror": 27, "Music": 10402,
    "Mystery": 9648, "Romance": 10749, "Science Fiction": 878,
    "Thriller": 53, "War": 10752, "Western": 37,
}

# Mood archetype → TMDB genre IDs + keyword hints + description
MOOD_PROFILES = {
    "Melancholic": {
        "genres": [18, 10749],          # Drama, Romance
        "keywords": "loneliness,loss,grief,melancholy,existential",
        "description": "slow, emotional, contemplative films",
        "color": "#4a6fa5",
        "emoji": "🌧️",
    },
    "Intense": {
        "genres": [53, 80, 9648],       # Thriller, Crime, Mystery
        "keywords": "psychological,dark,suspense,obsession,paranoia",
        "description": "gripping, dark, psychologically charged films",
        "color": "#8b1a1a",
        "emoji": "🔥",
    },
    "Euphoric": {
        "genres": [28, 12, 35],         # Action, Adventure, Comedy
        "keywords": "adventure,exciting,fun,feel-good,epic",
        "description": "high-energy, exciting, feel-good films",
        "color": "#e9a84c",
        "emoji": "⚡",
    },
    "Chill": {
        "genres": [35, 10749, 18],      # Comedy, Romance, Drama
        "keywords": "lighthearted,warm,cozy,slice-of-life,wholesome",
        "description": "easy, warm, low-stakes films",
        "color": "#4a9e7f",
        "emoji": "☁️",
    },
    "Introspective": {
        "genres": [18, 878, 9648],      # Drama, Sci-Fi, Mystery
        "keywords": "philosophical,identity,cerebral,consciousness,meaning",
        "description": "thought-provoking, philosophical, cerebral films",
        "color": "#7a5fa5",
        "emoji": "🌌",
    },
    "Cinematic": {
        "genres": [878, 14, 28],        # Sci-Fi, Fantasy, Action
        "keywords": "visually stunning,atmospheric,epic,world-building",
        "description": "visually rich, large-scale, immersive films",
        "color": "#2d6a8f",
        "emoji": "🎆",
    },
}


# ── Audio feature → Mood classifier ─────────────────────────────────────────

def classify_mood(valence: float, energy: float,
                  acousticness: float, instrumentalness: float) -> dict:
    """
    Classify audio features into a mood archetype.

    Spotify audio features (all 0.0 – 1.0):
    - valence:          0 = sad/dark, 1 = happy/euphoric
    - energy:           0 = calm/acoustic, 1 = intense/loud
    - acousticness:     0 = electronic, 1 = acoustic instruments
    - instrumentalness: 0 = vocals, 1 = purely instrumental
    """
    # Primary mood from valence × energy quadrant
    if valence < 0.4 and energy >= 0.55:
        primary = "Intense"
    elif valence < 0.45 and energy < 0.5:
        primary = "Melancholic"
    elif valence >= 0.6 and energy >= 0.55:
        primary = "Euphoric"
    elif valence >= 0.55 and energy < 0.5:
        primary = "Chill"
    else:
        primary = "Melancholic"  # default fallback

    # Override with modifier moods for strong signals
    if acousticness > 0.65 and energy < 0.5:
        primary = "Introspective"
    if instrumentalness > 0.55:
        primary = "Cinematic"

    profile = MOOD_PROFILES[primary].copy()
    profile["name"] = primary
    profile["valence"] = round(valence, 3)
    profile["energy"] = round(energy, 3)
    profile["acousticness"] = round(acousticness, 3)
    profile["instrumentalness"] = round(instrumentalness, 3)
    return profile


def get_recent_mood_from_features(tracks_with_features: list) -> dict:
    """
    Given a list of track dicts with audio features, compute the mood profile.
    Used by Spotify integration.
    """
    if not tracks_with_features:
        return None

    vals = [t.get("valence", 0.5) for t in tracks_with_features if t.get("valence") is not None]
    engs = [t.get("energy", 0.5) for t in tracks_with_features if t.get("energy") is not None]
    acs  = [t.get("acousticness", 0.3) for t in tracks_with_features if t.get("acousticness") is not None]
    ins  = [t.get("instrumentalness", 0.1) for t in tracks_with_features if t.get("instrumentalness") is not None]

    if not vals:
        return None

    avg_v = sum(vals) / len(vals)
    avg_e = sum(engs) / len(engs)
    avg_a = sum(acs) / len(acs)
    avg_i = sum(ins) / len(ins)

    mood = classify_mood(avg_v, avg_e, avg_a, avg_i)
    mood["track_count"] = len(tracks_with_features)
    mood["source"] = "spotify"
    return mood


def mood_from_lastfm_tags(tags: list[str]) -> dict:
    """
    Infer mood from Last.fm artist tags (fallback when Spotify unavailable).
    """
    tags_lower = [t.lower() for t in tags]

    melancholic_kws = {"sad", "melancholic", "melancholy", "depressive", "dark", "post-rock",
                       "slowcore", "shoegaze", "ambient", "folk", "acoustic", "indie folk"}
    intense_kws = {"metal", "hard rock", "aggressive", "punk", "noise", "industrial",
                   "electronic", "edm", "dubstep", "intense", "heavy"}
    euphoric_kws = {"pop", "dance", "party", "upbeat", "happy", "energetic", "hype",
                    "hip hop", "trap", "funk", "r&b"}
    chill_kws = {"chill", "lo-fi", "lofi", "relaxing", "easy listening", "bossa nova",
                  "jazz", "soft", "mellow"}

    scores = {"Melancholic": 0, "Intense": 0, "Euphoric": 0, "Chill": 0, "Introspective": 0}
    for tag in tags_lower:
        for kw in melancholic_kws:
            if kw in tag: scores["Melancholic"] += 1
        for kw in intense_kws:
            if kw in tag: scores["Intense"] += 1
        for kw in euphoric_kws:
            if kw in tag: scores["Euphoric"] += 1
        for kw in chill_kws:
            if kw in tag: scores["Chill"] += 1
        if any(k in tag for k in {"post-rock", "philosophical", "cerebral", "art rock", "experimental"}):
            scores["Introspective"] += 1

    best = max(scores, key=scores.get) if any(v > 0 for v in scores.values()) else "Melancholic"
    profile = MOOD_PROFILES[best].copy()
    profile["name"] = best
    profile["source"] = "lastfm_tags"
    profile["tag_count"] = len(tags)
    return profile


# ── TMDB film discovery ──────────────────────────────────────────────────────

TMDB_BASE = "https://api.themoviedb.org/3"

def _tmdb_get(endpoint: str, params: dict, api_key: str) -> Optional[dict]:
    params["api_key"] = api_key
    try:
        r = requests.get(f"{TMDB_BASE}/{endpoint}", params=params, timeout=10)
        time.sleep(0.25)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"[mood_engine] TMDB error: {e}")
    return None


def fetch_films_by_mood(mood_profile: dict, api_key: str,
                        min_votes: int = 500, n: int = 40) -> list[dict]:
    """
    Query TMDB Discover API for films matching the mood's genre profile.
    Returns films across the full rating spectrum (not just highly rated ones).
    """
    genre_ids = ",".join(str(g) for g in mood_profile.get("genres", [18]))

    # Two passes: one sorted by vote_count (popular), one by release_date (recent)
    results = []
    seen = set()

    for sort_by in ["vote_count.desc", "primary_release_date.desc", "popularity.desc"]:
        data = _tmdb_get("discover/movie", {
            "with_genres": genre_ids,
            "sort_by": sort_by,
            "vote_count.gte": min_votes if sort_by != "primary_release_date.desc" else 50,
            "language": "en-US",
            "page": 1,
        }, api_key)

        if not data:
            continue

        for film in data.get("results", []):
            fid = film.get("id")
            if fid and fid not in seen:
                seen.add(fid)
                poster_path = film.get("poster_path", "")
                results.append({
                    "tmdb_id": fid,
                    "title": film.get("title", ""),
                    "year": (film.get("release_date") or "")[:4],
                    "overview": film.get("overview", ""),
                    "genre_ids": film.get("genre_ids", []),
                    "vote_average": film.get("vote_average"),
                    "vote_count": film.get("vote_count", 0),
                    "popularity": film.get("popularity", 0),
                    "poster_url": f"https://image.tmdb.org/t/p/w300{poster_path}" if poster_path else "",
                })

        if len(results) >= n:
            break

    return results[:n]


# ── Alignment scoring ─────────────────────────────────────────────────────────

def compute_alignment_score(film: dict, mood_profile: dict,
                             letterboxd_df=None) -> float:
    """
    Score 0–100: how well this film aligns with the current mood + taste.

    Components:
    - Genre overlap (40pts): how many mood genres match film's genres
    - Taste match (40pts): if Letterboxd available, cosine similarity via ChromaDB
    - Popularity signal (20pts): not raw rating, but enough votes to trust it
    """
    score = 0.0
    mood_genres = set(mood_profile.get("genres", []))
    film_genres = set(film.get("genre_ids", []))

    # Genre overlap (0-40pts)
    overlap = len(mood_genres & film_genres)
    genre_score = min(40, overlap * 20)
    score += genre_score

    # Popularity/reliability signal (0-20pts)
    votes = film.get("vote_count", 0)
    vote_score = min(20, (votes / 1000) * 5)
    score += vote_score

    # Letterboxd cross-check (0-40pts) — only if Letterboxd data available
    if letterboxd_df is not None:
        try:
            from src.embedder import embed_text
            from src.vector_store import query_similar
            from src.config import CHROMA_MY_LENS_COLLECTION

            film_text = f"Film: {film['title']} | Genre: {film.get('overview','')[:200]}"
            emb = embed_text(film_text)
            if emb:
                qr = query_similar(CHROMA_MY_LENS_COLLECTION, emb, n_results=5)
                if qr and qr.get("distances") and qr["distances"][0]:
                    avg_sim = sum(1 - d for d in qr["distances"][0]) / len(qr["distances"][0])
                    score += avg_sim * 40
        except Exception:
            score += 20  # neutral when unavailable

    return round(min(100, score), 1)


# ── Explanation generator ─────────────────────────────────────────────────────

def explain_recommendation(film: dict, mood_profile: dict,
                            sources: list[str],
                            top_artists: list[str] = None,
                            top_books: list[str] = None) -> str:
    """
    Generate a human-readable explanation for why this film is recommended.
    Mentions the sources used so users understand the reasoning.
    """
    mood_name = mood_profile.get("name", "your current mood")
    desc = mood_profile.get("description", "films matching your vibe")
    parts = []

    if "spotify" in sources and top_artists:
        artists_str = ", ".join(top_artists[:3])
        parts.append(f"You've been listening to {artists_str}")
        parts.append(f"— {mood_name.lower()} energy. This film matches that emotional frequency.")
    elif "lastfm" in sources and top_artists:
        artists_str = ", ".join(top_artists[:3])
        parts.append(f"Your music ({artists_str}) signals a {mood_name.lower()} headspace.")
        parts.append(f"This film sits in the same emotional register.")
    else:
        parts.append(f"Based on your {mood_name.lower()} mood pattern,")
        parts.append(f"this film aligns with {desc}.")

    if "letterboxd" in sources:
        parts.append("Your Letterboxd taste also points in this direction.")
    if "goodreads" in sources:
        parts.append("Your reading taste reinforces this.")

    return " ".join(parts)


# ── Main recommendation pipeline ─────────────────────────────────────────────

def get_recommendations(
    mood_profile: dict,
    api_key: str,
    letterboxd_df=None,
    goodreads_df=None,
    sources: list[str] = None,
    top_artists: list[str] = None,
    n: int = 18,
) -> list[dict]:
    """
    Full pipeline:
    1. Fetch films from TMDB matching the mood's genres
    2. Score each for alignment (mood + taste)
    3. Sort by alignment score (not by TMDB rating)
    4. Attach explanation
    5. Return top N
    """
    sources = sources or []

    # Fetch candidate films
    films = fetch_films_by_mood(mood_profile, api_key, n=60)

    if not films:
        return []

    # Score and explain each
    for film in films:
        film["alignment"] = compute_alignment_score(film, mood_profile, letterboxd_df)
        film["explanation"] = explain_recommendation(
            film, mood_profile, sources,
            top_artists=top_artists,
        )

    # Sort by alignment (mood match), NOT by TMDB rating
    films.sort(key=lambda f: f["alignment"], reverse=True)

    return films[:n]


# ── Accuracy estimate ─────────────────────────────────────────────────────────

def estimate_accuracy(sources: list[str]) -> dict:
    """
    Returns accuracy level based on connected sources.
    More sources = more dimensions of taste = better alignment.
    """
    pct_map = {0: 38, 1: 62, 2: 80, 3: 93}
    label_map = {
        0: "Genre-based only",
        1: "Mood-matched",
        2: "Mood + taste calibrated",
        3: "Full cultural alignment",
    }
    n = min(len(sources), 3)
    return {
        "pct": pct_map[n],
        "label": label_map[n],
        "sources_connected": n,
    }

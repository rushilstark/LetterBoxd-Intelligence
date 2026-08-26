"""
Rating Predictor.

Given a movie (title + optional year), predicts what Rushil would rate it
by finding the most similar already-watched movies across the 3 embedding
spaces and computing a weighted average of his actual ratings.
"""
import numpy as np
from typing import Optional
from src.embedder import embed_text
from src.vector_store import query_similar
from src.tmdb_client import (
    get_movie_info, build_movie_dna_text,
    build_community_text, build_my_lens_text,
)
from src.config import (
    CHROMA_COMMUNITY_COLLECTION,
    CHROMA_MOVIE_DNA_COLLECTION,
    CHROMA_MY_LENS_COLLECTION,
    TOP_K_SIMILAR,
    WEIGHTS,
)


def predict_rating(title: str, year: Optional[int] = None) -> dict:
    """
    Predict the user's rating for an unwatched movie.

    Returns:
        {
            predicted_rating: float (0.5–5.0),
            confidence: float (0–1),
            top_similar: list of dicts with title, rating, distance,
            tmdb_info: dict,
            space_predictions: dict with per-space predictions,
        }
    """
    # ── 1. Fetch TMDB info ──────────────────────────────────────────────────
    info = get_movie_info(title, year)

    # ── 2. Build texts for all 3 spaces ────────────────────────────────────
    dna_text       = build_movie_dna_text(info)
    community_text = build_community_text(info)
    my_lens_text   = build_my_lens_text(title, None, "", False, "")  # no personal data yet

    # ── 3. Embed ────────────────────────────────────────────────────────────
    dna_emb       = embed_text(dna_text)
    community_emb = embed_text(community_text)
    my_lens_emb   = embed_text(my_lens_text)

    # ── 4. Query each space ─────────────────────────────────────────────────
    results = {}
    for name, emb, collection in [
        ("movie_dna",  dna_emb,       CHROMA_MOVIE_DNA_COLLECTION),
        ("community",  community_emb, CHROMA_COMMUNITY_COLLECTION),
        ("my_lens",    my_lens_emb,   CHROMA_MY_LENS_COLLECTION),
    ]:
        if not emb:
            results[name] = {"prediction": None, "hits": []}
            continue

        qr = query_similar(collection, emb, n_results=TOP_K_SIMILAR)
        hits = []
        for i in range(len(qr["ids"][0])):
            meta = qr["metadatas"][0][i]
            dist = qr["distances"][0][i]
            rating = meta.get("rating")
            if rating is not None:
                try:
                    rating = float(rating)
                    similarity = 1 - dist   # cosine distance → similarity
                    hits.append({
                        "title":      meta.get("title", ""),
                        "year":       meta.get("year"),
                        "rating":     rating,
                        "similarity": similarity,
                        "distance":   dist,
                    })
                except (ValueError, TypeError):
                    pass

        if hits:
            # Weighted average by similarity score
            weights_arr = np.array([h["similarity"] for h in hits])
            ratings_arr = np.array([h["rating"] for h in hits])
            # Avoid negative weights (shouldn't happen with cosine but just in case)
            weights_arr = np.clip(weights_arr, 0, None)
            total_w = weights_arr.sum()
            pred = float(np.dot(weights_arr, ratings_arr) / total_w) if total_w > 0 else None
        else:
            pred = None

        results[name] = {"prediction": pred, "hits": hits}

    # ── 5. Combine across spaces ────────────────────────────────────────────
    valid_preds = {k: v["prediction"] for k, v in results.items() if v["prediction"] is not None}

    if not valid_preds:
        return {
            "predicted_rating": None,
            "confidence": 0.0,
            "top_similar": [],
            "tmdb_info": info,
            "space_predictions": results,
        }

    total_weight = sum(WEIGHTS[k] for k in valid_preds)
    final_pred = sum(WEIGHTS[k] * v for k, v in valid_preds.items()) / total_weight

    # Clip to valid range
    final_pred = float(np.clip(final_pred, 0.5, 5.0))

    # Confidence = avg similarity of top hits in most trusted space
    best_space = max(valid_preds, key=lambda k: WEIGHTS[k])
    top_hits = results[best_space]["hits"][:5]
    confidence = float(np.mean([h["similarity"] for h in top_hits])) if top_hits else 0.0

    # Merge top similar from my_lens (most personal)
    my_hits = results.get("my_lens", {}).get("hits", [])
    dna_hits = results.get("movie_dna", {}).get("hits", [])
    # Deduplicate by title
    seen = set()
    merged_hits = []
    for h in (my_hits + dna_hits):
        key = h["title"].lower()
        if key not in seen:
            seen.add(key)
            merged_hits.append(h)
        if len(merged_hits) >= 8:
            break

    return {
        "predicted_rating": round(final_pred * 2) / 2,   # round to nearest 0.5
        "predicted_raw":    final_pred,
        "confidence":       round(confidence, 3),
        "top_similar":      merged_hits,
        "tmdb_info":        info,
        "space_predictions": {k: v["prediction"] for k, v in results.items()},
    }


def batch_predict(movies: list[dict]) -> list[dict]:
    """Predict ratings for a list of dicts with keys: title, year."""
    results = []
    for m in movies:
        pred = predict_rating(m["title"], m.get("year"))
        results.append({**m, **{
            "predicted_rating": pred["predicted_rating"],
            "confidence": pred["confidence"],
        }})
    return sorted(results, key=lambda x: (x["predicted_rating"] or 0), reverse=True)

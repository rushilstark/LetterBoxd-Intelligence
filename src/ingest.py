"""
Ingest and merge Letterboxd CSV exports into a unified DataFrame.
"""
import pandas as pd
from pathlib import Path
from src.config import DIARY_CSV, RATINGS_CSV, REVIEWS_CSV, WATCHLIST_CSV


def load_letterboxd_data() -> pd.DataFrame:
    """
    Merge diary, ratings, and reviews into one clean DataFrame.
    Each row = one unique (movie + my experience with it).

    Returns columns:
        name, year, rating, rewatch, review, tags, watched_date,
        letterboxd_uri
    """
    # ── 1. Ratings (broadest: every movie you rated) ─────────────────────────
    ratings = pd.read_csv(RATINGS_CSV, encoding="utf-8-sig")
    ratings.columns = [c.strip().lower().replace(" ", "_") for c in ratings.columns]
    ratings = ratings.rename(columns={"letterboxd_uri": "uri", "name": "name"})
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")

    # ── 2. Reviews (rich text) ────────────────────────────────────────────────
    reviews = pd.read_csv(REVIEWS_CSV, encoding="utf-8-sig")
    reviews.columns = [c.strip().lower().replace(" ", "_") for c in reviews.columns]
    reviews = reviews.rename(columns={
        "letterboxd_uri": "uri_review",
        "rating": "rating_review",
        "review": "review_text",
        "rewatch": "rewatch",
        "tags": "tags",
        "watched_date": "watched_date",
    })
    reviews["rating_review"] = pd.to_numeric(reviews["rating_review"], errors="coerce")

    # Collapse multiple reviews for same film → keep longest
    reviews["text_len"] = reviews["review_text"].fillna("").str.len()
    reviews = (
        reviews.sort_values("text_len", ascending=False)
        .drop_duplicates(subset=["name", "year"], keep="first")
        .drop(columns=["text_len"])
    )

    # ── 3. Diary (watch dates + rewatch flag) ─────────────────────────────────
    diary = pd.read_csv(DIARY_CSV, encoding="utf-8-sig")
    diary.columns = [c.strip().lower().replace(" ", "_") for c in diary.columns]
    diary = diary.rename(columns={
        "letterboxd_uri": "uri_diary",
        "rating": "rating_diary",
        "rewatch": "rewatch_diary",
        "tags": "tags_diary",
        "watched_date": "watched_date_diary",
    })
    diary["rating_diary"] = pd.to_numeric(diary["rating_diary"], errors="coerce")

    # Take the most recent diary entry per film
    diary["watched_date_diary"] = pd.to_datetime(diary["watched_date_diary"], errors="coerce")
    diary = (
        diary.sort_values("watched_date_diary", ascending=False)
        .drop_duplicates(subset=["name", "year"], keep="first")
    )

    # ── 4. Merge ──────────────────────────────────────────────────────────────
    # Start from ratings as base
    merged = ratings[["name", "year", "rating", "uri", "date"]].copy()
    merged["year"] = pd.to_numeric(merged["year"], errors="coerce").astype("Int64")

    # Join reviews
    merged = merged.merge(
        reviews[["name", "year", "rating_review", "review_text", "rewatch", "tags", "watched_date"]],
        on=["name", "year"],
        how="left",
    )

    # Join diary for watch date
    merged = merged.merge(
        diary[["name", "year", "rating_diary", "rewatch_diary", "watched_date_diary"]],
        on=["name", "year"],
        how="left",
    )

    # Resolve best rating (review > diary > ratings)
    merged["final_rating"] = merged["rating_review"].fillna(
        merged["rating_diary"].fillna(merged["rating"])
    )

    # Resolve best rewatch flag
    merged["is_rewatch"] = (
        merged["rewatch"].fillna(merged["rewatch_diary"]).str.lower() == "yes"
    )

    # Resolve best watch date
    merged["watch_date"] = pd.to_datetime(
        merged["watched_date"].fillna(merged["watched_date_diary"]), errors="coerce"
    )

    # Clean up
    merged = merged.rename(columns={"name": "title", "review_text": "my_review"})
    merged["my_review"] = merged["my_review"].fillna("")
    merged["tags"] = merged["tags"].fillna("")
    merged["year"] = merged["year"].astype("Int64")

    # Drop rows with no usable data
    merged = merged.dropna(subset=["title"])
    merged = merged[merged["title"].str.strip() != ""]

    # Final column selection
    keep = ["title", "year", "final_rating", "my_review", "tags", "is_rewatch",
            "watch_date", "uri"]
    merged = merged[keep].copy()
    merged = merged.drop_duplicates(subset=["title", "year"])
    merged = merged.reset_index(drop=True)

    print(f"[ingest] Loaded {len(merged)} unique movies | "
          f"{merged['my_review'].str.len().gt(10).sum()} with reviews | "
          f"{merged['final_rating'].notna().sum()} rated")

    return merged


def load_watchlist() -> pd.DataFrame:
    """Load watchlist CSVs for discovery predictions."""
    try:
        wl = pd.read_csv(WATCHLIST_CSV, encoding="utf-8-sig")
        wl.columns = [c.strip().lower().replace(" ", "_") for c in wl.columns]
        wl = wl.rename(columns={"name": "title", "letterboxd_uri": "uri"})
        wl["year"] = pd.to_numeric(wl.get("year", pd.Series(dtype=float)), errors="coerce").astype("Int64")
        return wl[["title", "year"]].dropna(subset=["title"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["title", "year"])

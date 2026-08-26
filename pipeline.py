"""
End-to-end pipeline: ingest → TMDB enrich → embed → store.
Incremental: skips already-embedded movies. Shows live progress.
Run this once to build the vector store. Takes ~30-60 min for full history.
"""
import sys
import time
from tqdm import tqdm

from src.config import (
    CHROMA_COMMUNITY_COLLECTION,
    CHROMA_MOVIE_DNA_COLLECTION,
    CHROMA_MY_LENS_COLLECTION,
    DATA_DIR,
    EMBEDDINGS_DIR,
)
from src.ingest import load_letterboxd_data
from src.tmdb_client import get_movie_info, build_movie_dna_text, build_community_text, build_my_lens_text
from src.embedder import embed_text, check_ollama_ready
from src.vector_store import upsert_embedding, doc_exists, collection_count


def make_doc_id(title: str, year) -> str:
    safe_year = str(year) if str(year) not in ("nan", "None", "<NA>") else "unknown"
    safe_title = title.lower().replace(" ", "_").replace("/", "_")[:50]
    return f"{safe_title}_{safe_year}"


def run_pipeline(limit: int = None, force_reembed: bool = False):
    """
    Main pipeline. 

    Args:
        limit:        If set, only process first N movies (for testing).
        force_reembed: If True, re-embed even already-stored movies.
    """
    print("\n" + "="*60)
    print("  LETTERBOXD INTELLIGENCE — PIPELINE")
    print("="*60)

    # ── Pre-flight checks ──────────────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    ok, msg = check_ollama_ready()
    if not ok:
        print(f"\n[ERROR] {msg}")
        sys.exit(1)
    print(f"[✓] {msg}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading Letterboxd data...")
    df = load_letterboxd_data()
    if limit:
        df = df.head(limit)
        print(f"[!]   Running on first {limit} movies (test mode)")

    total = len(df)
    print(f"[✓]   {total} movies to process\n")

    # Current counts
    c1 = collection_count(CHROMA_MY_LENS_COLLECTION)
    c2 = collection_count(CHROMA_MOVIE_DNA_COLLECTION)
    c3 = collection_count(CHROMA_COMMUNITY_COLLECTION)
    print(f"[2/4] Vector store status:")
    print(f"      my_lens:          {c1} docs")
    print(f"      movie_dna:        {c2} docs")
    print(f"      community_reviews:{c3} docs\n")

    # ── Main loop ──────────────────────────────────────────────────────────
    print("[3/4] Enriching & embedding (this takes time)...")
    print("      Progress saves after every movie — safe to interrupt.\n")

    tmdb_hits = 0
    skipped   = 0
    errors    = 0

    with tqdm(total=total, unit="movie", ncols=80) as pbar:
        for _, row in df.iterrows():
            title    = str(row["title"])
            year     = row.get("year")
            rating   = row.get("final_rating")
            review   = str(row.get("my_review", ""))
            is_rw    = bool(row.get("is_rewatch", False))
            tags     = str(row.get("tags", ""))
            watch_dt = row.get("watch_date")

            doc_id = make_doc_id(title, year)
            pbar.set_description(f"{title[:35]}")

            # Check if all 3 spaces already embedded
            all_done = (
                doc_exists(CHROMA_MY_LENS_COLLECTION, doc_id) and
                doc_exists(CHROMA_MOVIE_DNA_COLLECTION, doc_id) and
                doc_exists(CHROMA_COMMUNITY_COLLECTION, doc_id)
            )
            if all_done and not force_reembed:
                skipped += 1
                pbar.update(1)
                continue

            # ── TMDB enrichment ────────────────────────────────────────────
            try:
                info = get_movie_info(title, year)
                if info["found"]:
                    tmdb_hits += 1
            except Exception as e:
                info = {"found": False, "genres": [], "director": "", "cast": [],
                        "overview": "", "keywords": [], "community_reviews": []}

            # ── Build texts ────────────────────────────────────────────────
            dna_text       = build_movie_dna_text(info)
            community_text = build_community_text(info)
            my_lens_text   = build_my_lens_text(title, rating, review, is_rw, tags, watch_dt)

            # ── Embed & store ──────────────────────────────────────────────
            base_meta = {
                "title":  title,
                "year":   str(year) if str(year) not in ("nan", "None", "<NA>") else "",
                "rating": float(rating) if str(rating) not in ("nan", "None", "<NA>") else None,
            }

            try:
                # Space 1: My Lens (most personal, always embed)
                if not doc_exists(CHROMA_MY_LENS_COLLECTION, doc_id) or force_reembed:
                    emb = embed_text(my_lens_text)
                    if emb:
                        meta = {**base_meta, "review_preview": review[:300]}
                        upsert_embedding(CHROMA_MY_LENS_COLLECTION, doc_id, emb, meta, my_lens_text)

                # Space 2: Movie DNA
                if not doc_exists(CHROMA_MOVIE_DNA_COLLECTION, doc_id) or force_reembed:
                    emb = embed_text(dna_text)
                    if emb:
                        meta = {**base_meta,
                                "genres":   ", ".join(info.get("genres", [])),
                                "director": info.get("director", "")}
                        upsert_embedding(CHROMA_MOVIE_DNA_COLLECTION, doc_id, emb, meta, dna_text)

                # Space 3: Community Reviews
                if not doc_exists(CHROMA_COMMUNITY_COLLECTION, doc_id) or force_reembed:
                    emb = embed_text(community_text)
                    if emb:
                        upsert_embedding(CHROMA_COMMUNITY_COLLECTION, doc_id, emb, base_meta, community_text)

            except Exception as e:
                errors += 1
                tqdm.write(f"  [!] Error embedding '{title}': {e}")

            pbar.update(1)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n[4/4] Done!\n")
    print(f"  TMDB matches: {tmdb_hits}/{total - skipped} processed")
    print(f"  Skipped (already embedded): {skipped}")
    print(f"  Errors: {errors}")
    print(f"\n  Final store sizes:")
    print(f"    my_lens:           {collection_count(CHROMA_MY_LENS_COLLECTION)}")
    print(f"    movie_dna:         {collection_count(CHROMA_MOVIE_DNA_COLLECTION)}")
    print(f"    community_reviews: {collection_count(CHROMA_COMMUNITY_COLLECTION)}")
    print("\n  ✅ Launch the app:  streamlit run app.py\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build Letterboxd embeddings")
    parser.add_argument("--limit",  type=int, default=None,  help="Only process first N movies")
    parser.add_argument("--force",  action="store_true",     help="Re-embed even stored movies")
    args = parser.parse_args()
    run_pipeline(limit=args.limit, force_reembed=args.force)

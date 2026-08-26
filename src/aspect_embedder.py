"""
Multi-aspect embedding builder.
Instead of one vector per film, stores 4 aspect vectors:
  structural, emotional, thematic, stylistic

This captures WHY you liked a film, not just that you liked it.
"""
import sys; sys.path.insert(0, ".")
from src.embedder import embed_text
from src.vector_store import upsert_embedding, doc_exists, get_collection
from src.ingest import load_letterboxd_data
from src.tmdb_client import get_movie_info
from tqdm import tqdm
import ollama

ASPECTS = ["structural", "emotional", "thematic", "stylistic"]
ASPECT_COLLECTIONS = {a: f"aspect_{a}" for a in ASPECTS}

# Prompt template: extract a specific aspect from a review
ASPECT_PROMPTS = {
    "structural": "Describe only the NARRATIVE STRUCTURE of this film: how is the story told? Is it linear, non-linear, puzzle-like, episodic? What is the plot mechanism?",
    "emotional":  "Describe only the EMOTIONAL TONE of this film: What feelings does it evoke? Is it warm, cold, devastating, uplifting, tense, comforting?",
    "thematic":   "Describe only the THEMES of this film: What ideas does it explore? Identity, mortality, love, power, freedom, society?",
    "stylistic":  "Describe only the STYLE of this film: cinematography, music, pacing, visual language, director's signature.",
}


def build_aspect_text(title: str, overview: str, my_review: str, aspect: str) -> str:
    """
    Build an aspect-specific text WITHOUT using an LLM (fast path).
    We craft the text from TMDB overview + your review, biased toward the aspect.
    
    For a true implementation, you'd pass this to llama3 to extract the aspect.
    This is the lightweight version.
    """
    base = f"Film: {title}\n"
    if overview:
        base += f"Plot: {overview}\n"
    if my_review:
        base += f"Review: {my_review}\n"
    base += f"\nFocus aspect: {ASPECT_PROMPTS[aspect]}"
    return base


def build_aspect_text_with_llm(title: str, overview: str, my_review: str, aspect: str) -> str:
    """
    LLM-powered aspect extraction via Ollama.
    Uses llama3 to distill the specific aspect from the review.
    Falls back to template if LLM unavailable.
    """
    try:
        client = ollama.Client()
        prompt = f"""You are analysing a film review to extract one specific aspect.

Film: {title}
Overview: {overview[:300]}
Review: {my_review[:500]}

Task: Write 2-3 sentences describing ONLY the {aspect.upper()} aspect of this film based on the above.
{ASPECT_PROMPTS[aspect]}

Be specific and concrete. Do not mention other aspects."""

        response = client.generate(
            model="llama3",   # use whatever you have in ollama
            prompt=prompt,
            options={"temperature": 0.1, "num_predict": 150},
        )
        return response["response"].strip()
    except Exception:
        # Fall back to template-based approach
        return build_aspect_text(title, overview, my_review, aspect)


def run_aspect_pipeline(limit=None, use_llm=False):
    """Build multi-aspect embeddings for all your watched films."""
    df = load_letterboxd_data()
    if limit:
        df = df.head(limit)

    print(f"\nBuilding aspect embeddings for {len(df)} films")
    print(f"LLM extraction: {'ON (llama3)' if use_llm else 'OFF (template)'}")
    print(f"Collections: {list(ASPECT_COLLECTIONS.values())}\n")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        title    = str(row["title"])
        year     = row.get("year")
        rating   = row.get("final_rating")
        review   = str(row.get("my_review", ""))
        doc_id   = f"{title.lower().replace(' ','_')[:40]}_{year}"

        # Get TMDB overview
        try:
            info = get_movie_info(title, year)
            overview = info.get("overview", "")
        except Exception:
            overview = ""

        base_meta = {
            "title":  title,
            "year":   str(year) if year else "",
            "rating": float(rating) if rating and str(rating) not in ("nan","None") else None,
        }

        for aspect in ASPECTS:
            collection = ASPECT_COLLECTIONS[aspect]
            aspect_id  = f"{doc_id}_{aspect}"

            if doc_exists(collection, aspect_id):
                continue

            # Build aspect-specific text
            if use_llm and review and len(review) > 50:
                text = build_aspect_text_with_llm(title, overview, review, aspect)
            else:
                text = build_aspect_text(title, overview, review, aspect)

            emb = embed_text(text)
            if emb:
                upsert_embedding(
                    collection, aspect_id, emb,
                    {**base_meta, "aspect": aspect},
                    text,
                )

    print("\nDone. Aspect collections built.")


def predict_with_aspects(title: str, year=None) -> dict:
    """
    Predict rating using aspect-level matching.
    Finds which aspect matters most for similar films you loved,
    then weights the prediction accordingly.
    """
    from src.vector_store import query_similar
    import numpy as np

    info = get_movie_info(title, year)
    overview = info.get("overview", "")

    aspect_preds = {}
    for aspect in ASPECTS:
        text = build_aspect_text(title, overview, "", aspect)
        emb  = embed_text(text)
        if not emb:
            continue

        qr = query_similar(ASPECT_COLLECTIONS[aspect], emb, n_results=10)
        ratings, sims = [], []
        for i in range(len(qr["ids"][0])):
            meta = qr["metadatas"][0][i]
            dist = qr["distances"][0][i]
            r    = meta.get("rating")
            if r:
                try:
                    ratings.append(float(r))
                    sims.append(1 - dist)
                except (ValueError, TypeError):
                    pass

        if ratings:
            sims_arr    = np.clip(np.array(sims), 0, None)
            ratings_arr = np.array(ratings)
            aspect_preds[aspect] = np.dot(sims_arr, ratings_arr) / sims_arr.sum()

    if not aspect_preds:
        return {"error": "No predictions available"}

    # Equal weight across aspects (could be learned)
    final = np.mean(list(aspect_preds.values()))
    return {
        "predicted_rating": round(float(final) * 2) / 2,
        "aspect_breakdown":  {k: round(v, 2) for k, v in aspect_preds.items()},
        "interpretation": _interpret_aspects(aspect_preds),
    }


def _interpret_aspects(preds: dict) -> str:
    """Tell the user WHICH aspect drives the prediction."""
    if not preds:
        return ""
    best  = max(preds, key=preds.get)
    worst = min(preds, key=preds.get)
    msgs = {
        "structural": "films with similar narrative structure",
        "emotional":  "films with similar emotional tone",
        "thematic":   "films exploring similar themes",
        "stylistic":  "films with similar visual/audio style",
    }
    return (f"Your taste is most consistent for {msgs[best]} "
            f"(predicts {preds[best]:.1f}★) and least consistent for "
            f"{msgs[worst]} (predicts {preds[worst]:.1f}★).")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--llm",   action="store_true", help="Use llama3 for extraction")
    p.add_argument("--predict", type=str, default=None, help="Predict for a film title")
    args = p.parse_args()

    if args.predict:
        result = predict_with_aspects(args.predict)
        print(f"\nPrediction for '{args.predict}':")
        print(f"  Overall: {result.get('predicted_rating')}★")
        print(f"  By aspect: {result.get('aspect_breakdown')}")
        print(f"\n  {result.get('interpretation')}")
    else:
        run_aspect_pipeline(limit=args.limit, use_llm=args.llm)

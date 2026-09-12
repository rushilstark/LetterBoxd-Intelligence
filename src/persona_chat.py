"""
persona_chat.py — Rushil's AI twin, powered by MLX Llama-3.1

HOW THIS WORKS (AI Engineering 101):
=====================================
This is a RAG (Retrieval-Augmented Generation) system. The pipeline is:

  User Question
       │
       ▼
  [Step 1: RETRIEVAL]
  Hybrid search: lexical (keyword) + vector (semantic similarity)
  → Finds your most relevant Letterboxd reviews from ChromaDB
  → Also checks: did you review this EXACT film? Any films by same director?
       │
       ▼
  [Step 2: CONTEXT BUILDING]
  Assembles a structured prompt containing:
    - Rushil's taste profile (built from ALL 3507 films — facts, not guesses)
    - Your writing samples (15 longest reviews for style grounding)
    - The retrieved relevant reviews
    - Conversation history
       │
       ▼
  [Step 3: GENERATION]
  MLX Llama-3.1-8B reads everything above and generates a reply in your voice.
  Temperature=0.65 (higher = more volatile, emotional, risky — matches your writing style)

Why is this better than before?
- Before: 6 random reviews + vague "be Rushil" instructions
- Now: Rich taste profile (3507 films worth of data) + smart retrieval + better prompting
"""

import sys
import re
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

# ── Bootstrap: add gpt-from-scratch to Python path ─────────────────────────
try:
    _V4_ROOT = Path("/Users/rushilreddy/untitled folder 4/gpt-from-scratch")
    _V4_CORE = _V4_ROOT / "v4"

    for _p in [str(_V4_ROOT), str(_V4_CORE)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
except Exception:
    pass

# ── Letterboxd imports ──────────────────────────────────────────────────────
from src.vector_store import get_collection, query_similar
from src.embedder import embed_batch
from src.ingest import load_letterboxd_data
from src.persona_profile import build_profile, format_profile_card, get_writing_samples, get_direct_opinion

CHROMA_REVIEW_TEXT_COLLECTION = "user_raw_reviews"

# ── MLX Engine singleton ─────────────────────────────────────────────────────
# Singletons avoid reloading 4GB of model weights on every call.
_inference_engine = None
_mlx_available = None


def _get_engine():
    global _inference_engine, _mlx_available
    if _inference_engine is not None:
        return _inference_engine, None
    if _mlx_available is False:
        return None, "MLX engine unavailable."
    try:
        from core.config import load_config
        from core.inference import InferenceEngine
        config = load_config(str(_V4_CORE / "config.yaml"))
        config.model.max_tokens["chat"] = 600
        # TEMPERATURE: 0.65 — higher than before (0.2).
        # Your writing is volatile and emotional. Low temp = stiff robot voice.
        # 0.65 = takes risks, contradicts itself, spirals — like you actually write.
        config.model.temperature = 0.65
        print("[persona_chat] Loading MLX model...")
        _inference_engine = InferenceEngine(config)
        _mlx_available = True
        print(f"[persona_chat] Ready: {_inference_engine.model_name}")
        return _inference_engine, None
    except ImportError as e:
        _mlx_available = False
        return None, f"mlx_lm not available: {e}"
    except Exception as e:
        _mlx_available = False
        return None, f"Failed to load MLX engine: {e}"


def _ollama_fallback(system_prompt: str, user_prompt: str) -> str:
    """Fallback to Ollama llama3 if MLX fails."""
    try:
        import ollama
        r = ollama.chat(
            model="llama3:latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            options={"temperature": 0.65}
        )
        return r["message"]["content"]
    except Exception as e:
        return f"[Both MLX and Ollama failed: {e}]"


def _generate(system_prompt: str, user_prompt: str, max_tokens: int = 600) -> str:
    """Run the LLM. Tries MLX first, falls back to Ollama."""
    engine, err = _get_engine()
    if engine is not None:
        try:
            response, _ = engine.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
            return response
        except Exception as e:
            print(f"[persona_chat] MLX failed: {e}. Falling back to Ollama.")
    return _ollama_fallback(system_prompt, user_prompt)


# ── Review indexing ──────────────────────────────────────────────────────────

def index_all_reviews(df: pd.DataFrame = None):
    """
    Embeds all of Rushil's reviews into ChromaDB for vector search.
    This is a one-time setup step. Each review gets a 1024-dim vector
    that captures its SEMANTIC meaning, not just keywords.
    """
    if df is None:
        df = load_letterboxd_data()
    revs = df[df["my_review"].str.strip() != ""].copy()
    if not len(revs):
        return
    col = get_collection(CHROMA_REVIEW_TEXT_COLLECTION)
    if col.count() >= len(revs):
        print("Reviews already indexed.")
        return
    print(f"Indexing {len(revs)} reviews...")
    texts   = revs["my_review"].tolist()
    titles  = revs["title"].tolist()
    ratings = revs["final_rating"].tolist()
    rewatches = revs["is_rewatch"].tolist()
    years = revs["year"].tolist()
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        bt = texts[i:i+batch_size]
        embs = embed_batch([t[:1800] for t in bt])
        for j, emb in enumerate(embs):
            idx = i + j
            yr = str(years[idx]) if str(years[idx]) not in ("nan", "None", "<NA>") else ""
            col.upsert(
                ids=[f"rev_{titles[idx]}_{idx}"],
                embeddings=[emb],
                metadatas=[{
                    "title":    str(titles[idx]),
                    "rating":   float(ratings[idx]) if pd.notna(ratings[idx]) else 0.0,
                    "rewatch":  bool(rewatches[idx]),
                    "year":     yr,
                    "rev_len":  len(texts[idx]),
                }],
                documents=[texts[idx]],
            )
        print(f"Indexed {i+len(bt)} / {len(revs)}")


# ── RAG helpers ──────────────────────────────────────────────────────────────

_df_cached = None
_profile_cached = None


def _get_cached_df():
    global _df_cached
    if _df_cached is None:
        try:
            _df_cached = load_letterboxd_data()
        except Exception:
            _df_cached = pd.DataFrame(columns=["title", "year", "final_rating", "my_review", "tags", "is_rewatch", "watch_date", "uri"])
    return _df_cached


def _get_cached_profile():
    global _profile_cached
    if _profile_cached is None:
        try:
            _profile_cached = build_profile(_get_cached_df())
        except Exception:
            _profile_cached = {
                "total_movies": 0, "total_rated": 0, "total_reviews": 0,
                "rewatches": 0, "avg_rating": 0, "five_star_count": 0,
                "one_star_count": 0, "peak_year": 2024, "dist_str": "",
                "north_stars": [], "top_genres": [], "low_genres": [],
                "top_directors": [], "top_decade": None,
                "best_reviews": [], "_df": pd.DataFrame(), "_reviewed": pd.DataFrame(),
            }
    return _profile_cached


def _extract_keywords(query: str) -> List[str]:
    """Strip stopwords to get content tokens for lexical search."""
    clean = query.lower()
    for c in '?!.,:;"\'()[]{}':
        clean = clean.replace(c, ' ')
    stop_words = {
        'how', 'do', 'i', 'feel', 'about', 'what', 'is', 'a', 'the', 'of', 'to', 'in',
        'on', 'with', 'for', 'and', 'or', 'but', 'you', 'think', 'my', 'opinion', 'me',
        'like', 'movie', 'film', 'movies', 'films', 'director', 'directors',
        'actor', 'actors', 'watched', 'did', 'does', 'was', 'were', 'have', 'has', 'had',
        'who', 'whom', 'why', 'where', 'when', 'which', 'whose', 'your', 'its', 'their',
    }
    return [w for w in clean.split() if w not in stop_words and len(w) > 2]


def retrieve_relevant_reviews(query: str, n: int = 8) -> tuple[str, List[Dict]]:
    """
    RETRIEVAL STEP — The most important function in the whole pipeline.
    
    Strategy (in priority order):
    1. EXACT film title match — if you asked about Tamasha, get your Tamasha review first
    2. Lexical keyword search — find reviews that mention the query words
    3. Vector semantic search — find reviews about similar TOPICS even with different words
    4. Merge, deduplicate, sort by relevance score
    
    Returns (formatted_context_string, raw_matches_list)
    """
    kws = _extract_keywords(query)
    df = _get_cached_df()

    # ── 1. Try to find exact/partial title match ─────────────────────────────
    title_matches = []
    for kw in kws:
        if len(kw) >= 4:  # Only meaningful words
            hit = df[df["title"].str.lower().str.contains(kw, regex=False, na=False)]
            for _, row in hit.iterrows():
                review = str(row.get("my_review", "")).strip()
                if len(review) > 10:
                    title_matches.append({
                        "title": row["title"],
                        "rating": row.get("final_rating"),
                        "review": review,
                        "rewatch": bool(row.get("is_rewatch", False)),
                        "score": 10,  # Highest priority — exact title match
                        "source": "title_match"
                    })

    # ── 2. Lexical search in review bodies ──────────────────────────────────
    lexical_matches = []
    if kws:
        for _, row in df.iterrows():
            title = str(row["title"])
            review = str(row["my_review"])
            rating = row.get("final_rating")
            if not review.strip() or len(review) < 10:
                continue

            score = 0
            title_lower = title.lower()
            review_lower = review.lower()

            for k in kws:
                if k in title_lower:
                    score += 4
                if k in review_lower:
                    score += 1

            # Boost high-emotion reviews (5★ and ≤1★ are the most interesting)
            if rating == 5.0:
                score = int(score * 1.3)
            elif rating is not None and rating <= 1.0:
                score = int(score * 1.2)

            # Boost rewatches (they mean more)
            if row.get("is_rewatch"):
                score = int(score * 1.2)

            if score > 0:
                lexical_matches.append({
                    "title": title,
                    "rating": rating,
                    "review": review,
                    "rewatch": bool(row.get("is_rewatch", False)),
                    "score": score,
                    "source": "lexical"
                })
        lexical_matches.sort(key=lambda x: x["score"], reverse=True)

    # ── 3. Vector semantic search ─────────────────────────────────────────────
    # This finds reviews about SIMILAR THEMES even if they don't share keywords.
    # e.g. asking "films about loneliness" → finds reviews that discuss isolation
    # even if the word "loneliness" never appears.
    vector_matches = []
    try:
        q_emb = embed_batch([query])[0]  # Embed the query
        res = query_similar(CHROMA_REVIEW_TEXT_COLLECTION, q_emb, n_results=n + 4)
        if res and res["ids"] and res["ids"][0]:
            for meta, doc, dist in zip(res["metadatas"][0], res["documents"][0], res["distances"][0]):
                similarity = 1 - dist  # ChromaDB returns cosine distance, convert to similarity
                vector_matches.append({
                    "title":   meta.get("title", ""),
                    "rating":  meta.get("rating", 0.0),
                    "review":  doc,
                    "rewatch": meta.get("rewatch", False),
                    "score":   int(similarity * 5),  # Normalized to ~0-5
                    "source":  "vector"
                })
    except Exception as e:
        print(f"[persona_chat] Vector query failed: {e}")

    # ── 4. Merge & Deduplicate ───────────────────────────────────────────────
    seen_titles = set()
    combined = []

    # Title matches first (highest priority)
    for m in title_matches:
        if m["title"].lower() not in seen_titles:
            seen_titles.add(m["title"].lower())
            combined.append(m)

    # Then top lexical matches
    for m in lexical_matches[:5]:
        if m["title"].lower() not in seen_titles:
            seen_titles.add(m["title"].lower())
            combined.append(m)

    # Then vector matches to fill remaining slots
    for m in vector_matches:
        if m["title"].lower() not in seen_titles:
            seen_titles.add(m["title"].lower())
            combined.append(m)
            if len(combined) >= n:
                break

    final_matches = combined[:n]

    # ── Format for prompt ────────────────────────────────────────────────────
    if not final_matches:
        return "", []

    parts = []
    for m in final_matches:
        rewatch_tag = " [rewatched — really loved this]" if m.get("rewatch") else ""
        # Truncate long reviews but keep enough to show voice
        review_text = str(m["review"])[:500]
        parts.append(
            f"[{m['title']} — {m['rating']}★{rewatch_tag}]:\n{review_text}"
        )

    return "\n\n---\n\n".join(parts), final_matches


# ── System prompt builder ────────────────────────────────────────────────────

def _build_persona_system(context: str, history_str: str, profile: Dict) -> str:
    """
    Builds the full system prompt. This is what the LLM reads before generating.
    
    Structure:
    1. WHO YOU ARE (role instruction + hard facts from data)
    2. STYLE SAMPLES (your actual writing — the LLM learns your voice from these)
    3. STYLE RULES (do's and don'ts derived from your actual patterns)
    4. RETRIEVED REVIEWS (the most relevant ones for this specific question)
    5. CONVERSATION HISTORY (what's been said so far)
    """

    # ── Taste profile card (hard data facts) ────────────────────────────────
    profile_card = format_profile_card(profile)

    # ── Writing style samples (15 longest reviews = richest voice samples) ───
    writing_samples = get_writing_samples(profile, n=15)

    # ── Tight, specific style rules ──────────────────────────────────────────
    style_rules = """
TWO MODES — read the question and pick:

MODE 1: CONVERSATIONAL (someone asks you about taste, opinions, preferences)
This is you texting a friend at midnight, not writing a Letterboxd review.
- Talk directly. No opening "Movies can change lives???" headline — that's for reviews.
- Get to the point fast: "for me a perfect film is either one where I'm doing psychological shit and can't stop thinking, or the opposite — pure chaos that shuts my brain off."
- THEN give 3-5 specific films from your actual history as proof. Name them, rate them, say one thing about each. "Memento did that. Shutter Island. Fight Club. These are the ones."
- One or two sentences per thought. Then next thought. No paragraph essay structure.
- Hindi comes out when emotional: "yaar", "nahi yaar", "sach mein"
- Personal life references feel NATURAL not forced — your actual life: restless at office, feeling stuck, 2am watching films
- DO NOT stop after one sentence. Keep going. Add the examples. Maybe contradict yourself at the end.

MODE 2: SPECIFIC FILM REVIEW (someone asks about a film you've seen)
This is closer to your Letterboxd writing — the punchy opener IS appropriate here.
- Use the review voice: headline, spiral, rhetorical questions, abrupt end
- Quote your actual reviews when you have them

UNIVERSAL RULES (both modes):
- Never "I want to be lost in the world like I am in [film]" — that's film critic speak, not how you talk
- Never chain multiple "like in X... like in Y... and of course Z" comparisons — pick one or two and go deeper
- No neat conclusions. You run out of thought and stop, or you contradict yourself
- No AI phrasing: "On the other hand", "It's important to note", "In conclusion"
- Never copy the writing samples verbatim
- Minimum 3-4 sentences for any conversational answer

THE GOAL: sound like a guy who watches 300+ films a year and has genuine deep opinions, talking casually — NOT like a critic performing analysis."""

    history_section = f"\nRECENT CONVERSATION (continue naturally from here):\n{history_str}" if history_str else ""

    context_section = context if context else "(No directly relevant reviews found — use the profile data above to extrapolate)"

    return f"""You ARE Rushil Reddy. Not an AI playing a character. You ARE him, speaking from real memory and real opinions recorded in 3507 film ratings and 870 written reviews.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{profile_card}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RUSHIL'S WRITING — These show his REVIEW voice. Use only when the question is about a specific film:

{writing_samples}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{style_rules}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR RETRIEVED REVIEWS — your actual opinions on relevant films (use as memory, not as templates):

{context_section}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{history_section}

Respond as Rushil. If this is a conversational question about taste or preferences, use MODE 1 (casual, direct, then give specific film examples). If it's about a specific film, use MODE 2 (review voice). Do not mix them. Write at least 4-6 sentences."""




# ── Persona Chat ──────────────────────────────────────────────────────────────

def chat_with_persona(user_msg: str, chat_history: List[Dict]) -> str:
    """
    Main entry point. Full RAG pipeline:
    1. Load/cache the taste profile (built from all 3507 films)
    2. Retrieve relevant reviews (hybrid lexical + vector)
    3. Build the prompt
    4. Generate with MLX Llama-3.1
    """
    profile = _get_cached_profile()

    # Get relevant context
    context, raw_matches = retrieve_relevant_reviews(user_msg, n=8)

    # Build conversation history string
    history_str = ""
    if chat_history:
        for msg in chat_history[-4:]:  # Last 4 turns
            role = "Rushil" if msg["role"] == "assistant" else "User"
            history_str += f"{role}: {msg['content'][:300]}\n"

    system_prompt = _build_persona_system(context, history_str, profile)
    user_prompt = f"{user_msg}\n\nRushil:"

    return _generate(system_prompt, user_prompt, max_tokens=600)


def get_retrieved_context(user_msg: str) -> List[Dict]:
    """
    Exposes what reviews were retrieved for a query.
    Used by the Streamlit UI to show a debug panel (what did the model read?).
    """
    _, raw_matches = retrieve_relevant_reviews(user_msg, n=8)
    return raw_matches


# ── Review Copilot ────────────────────────────────────────────────────────────

def critique_review_draft(draft: str, df: pd.DataFrame = None) -> str:
    """
    Critiques a review draft against Rushil's best writing.
    Uses the profile to pull the top 5 longest reviews as gold standard.
    """
    profile = _get_cached_profile()
    gold_reviews = profile["best_reviews"][:5]

    gold = ""
    for r in gold_reviews:
        rating_str = f"{r['final_rating']}★" if r.get("final_rating") else ""
        gold += f"\n[{r['title']} — {rating_str}]:\n{str(r['my_review'])[:600]}\n"

    system_prompt = (
        "You are an elite cinematic writing coach who has read ALL of Rushil's Letterboxd reviews.\n\n"
        "Here are his five longest, most considered reviews as the gold standard:\n"
        f"---\n{gold}\n---\n\n"
        "Critique his DRAFT below. Be specific — quote the draft, reference specific past reviews.\n"
        "Structure:\n"
        "**What's Working:** (quote the draft, be specific)\n"
        "**What's Missing:** (what does he do in his best writing that's absent? e.g. personal life tie-in, rhetorical questions, Hindi slips, punchy opening headline)\n"
        "**Crutch Check:** (flag 3 weak/generic words or phrases. suggest replacements in HIS vocabulary)\n"
        "**One Challenge:** (one question to push him deeper — reference a specific past review as example)\n\n"
        "Be warm but brutally honest. No generic praise. Treat him like a friend."
    )
    return _generate(system_prompt, f"My draft:\n\n{draft}", max_tokens=700)


# ── Engine status ─────────────────────────────────────────────────────────────

def get_engine_status() -> dict:
    if _inference_engine is not None:
        return {"backend": "MLX (Local — Apple Silicon)",
                "model": _inference_engine.model_name, "loaded": True}
    elif _mlx_available is False:
        return {"backend": "Ollama (fallback)", "model": "llama3:latest", "loaded": True}
    else:
        return {"backend": "Not yet loaded", "model": "--", "loaded": False}

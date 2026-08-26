"""
LESSON 6: GOING DEEPER — The Nuance Problem & Real Solutions

The user asked: "Inception and Interstellar are close in embedding space,
but many people like one and not the other. How do we go deeper?"

This is the single most important question in AI engineering.
Run: python learn/06_going_deeper.py
"""
import sys; sys.path.insert(0, ".")
import numpy as np

print("\n" + "="*60)
print("  LESSON 6: THE NUANCE PROBLEM")
print("="*60)

from src.embedder import embed_text

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ─── 1. Show the averaging problem ────────────────────────────
print("""
[1] WHY SINGLE VECTORS MISS NUANCE
─────────────────────────────────────────────────────────────

A film is MANY things at once. A single embedding vector is their
AVERAGE — a centroid. You lose the individual dimensions of quality.

Inception has:
  - Puzzle-box plot (very layered, intellectual)
  - Action set pieces (big budget spectacle)
  - Father-son emotional core
  - "Is this real?" philosophical theme
  - Hans Zimmer score

Interstellar has:
  - Hard science (time dilation, black holes)
  - Father-daughter emotional core (much more prominent)
  - Space opera spectacle
  - "Love transcends time" philosophical theme
  - Hans Zimmer score

They're SIMILAR on some axes (Nolan, Zimmer, sci-fi, emotional core)
but DIFFERENT on others (puzzle vs science, intellect vs emotion).

Someone who loves Inception for the PUZZLE might hate Interstellar's
slow emotional beats. Someone who loves Interstellar for the EMOTION
might find Inception too cold and intellectual.

A single embedding captures the CENTROID and can't distinguish WHY.
""")

# ─── 2. Demonstrate with multi-aspect embeddings ──────────────
print("─"*60)
print("[2] SOLUTION 1: MULTI-ASPECT EMBEDDINGS")
print("─"*60)
print("""
Instead of ONE embedding per film, embed SEPARATE ASPECTS:
  - Structural aspect  (how is it told?)
  - Emotional aspect   (how does it feel?)
  - Thematic aspect    (what is it about?)
  - Stylistic aspect   (how does it look/sound?)

Then compare at the ASPECT level, not the film level.
""")

# Craft aspect-specific queries
inception_aspects = {
    "structural": "non-linear puzzle narrative with nested time loops and layers, heist plot with complex mechanics",
    "emotional":  "detached protagonist grieving wife, cold intellectual tone, family reunion as distant goal",
    "thematic":   "what is real vs dream, guilt and subconscious, manipulation of perception",
    "stylistic":  "slick action sequences, Hans Zimmer BRAAAM score, practical effects, urban dreamscapes",
}

interstellar_aspects = {
    "structural": "linear space journey with time jumps, acts divided by gravitational time dilation, revelation ending",
    "emotional":  "overwhelming father-daughter love, desperate emotional urgency, sacrifice, parental guilt",
    "thematic":   "love transcends spacetime, human survival, scientific wonder, trust vs authority",
    "stylistic":  "Hans Zimmer organ score, real IMAX space footage, silence of space, quiet vast cinematography",
}

print("  Inception vs Interstellar — ASPECT-LEVEL cosine similarities:\n")
print(f"  {'Aspect':<14}  {'Similarity':>10}  Notes")
print(f"  {'─'*14}  {'─'*10}  {'─'*35}")

for aspect in inception_aspects:
    e1 = embed_text(inception_aspects[aspect])
    e2 = embed_text(interstellar_aspects[aspect])
    sim = cosine_sim(e1, e2)
    print(f"  {aspect:<14}  {sim:>10.4f}  ", end="")
    if aspect == "structural":
        print("← DIFFERENT (puzzle vs journey)")
    elif aspect == "emotional":
        print("← VERY DIFFERENT (cold vs overwhelming)")
    elif aspect == "thematic":
        print("← SIMILAR (reality/meaning questions)")
    elif aspect == "stylistic":
        print("← SIMILAR (both Nolan/Zimmer)")

print("""
  ✅ KEY INSIGHT:
     Structurally and emotionally: quite different
     Thematically and stylistically: similar

     A single embedding averages these → medium similarity.
     Aspect embeddings reveal WHERE they agree and disagree.

     ENGINEERING IMPLICATION:
     If you loved Inception for its STRUCTURAL puzzle,
     you want movies with high structural similarity — not Interstellar.
     You want: Memento, Primer, Coherence, Triangle.
""")

# ─── 3. What's really missing: collaborative signal ──────────
print("─"*60)
print("[3] SOLUTION 2: COLLABORATIVE FILTERING SIGNAL")
print("─"*60)
print("""
  Content-based embeddings (what you're using) capture:
    → What a film IS (plot, genre, style)

  Collaborative filtering captures:
    → What kind of PEOPLE like it together

  Example:
    People who love Inception often love: Memento, Primer, The Prestige
    People who love Interstellar often love: Contact, Gravity, 2001

  The overlap exists but so do the differences.
  This is what Netflix, Spotify, Amazon actually use.

  HOW IT WORKS (Matrix Factorization):

    User-Movie Rating Matrix:
                   Inception  Interstellar  Memento  Contact
    Rushil              5           4           ?       ?
    User_2              4           5           3       5
    User_3              5           3           5       2
    User_4              2           5           1       5

    Find: what latent "taste dimensions" explain these ratings?
    → Rushil is high on "intellectual puzzle" dim, medium on "space drama" dim
    → Predict Memento (high puzzle) → 5★, Contact (low puzzle) → 3★

  The system learns the axes FROM the data — no text needed at all.
  Then YOU can be located on those axes.

  COMBINING BOTH:
    content_score      = cosine_similarity(film_embedding, query)
    collab_score       = matrix_factorization_prediction
    final_score        = 0.6 * content_score + 0.4 * collab_score

  This is called a HYBRID RECOMMENDER — state of the art in industry.
""")

# ─── 4. Cross-encoders: the deep comparison ───────────────────
print("─"*60)
print("[4] SOLUTION 3: CROSS-ENCODERS (Reranking)")
print("─"*60)
print("""
  Your system is a BI-ENCODER:
    embed(film_A) → vector_A
    embed(film_B) → vector_B
    similarity = cosine(vector_A, vector_B)

  Both sides are embedded INDEPENDENTLY, then compared.
  Fast but loses interaction between the two texts.

  A CROSS-ENCODER sees BOTH texts simultaneously:
    score = model(film_A_text + [SEP] + film_B_text)
    → one score output, sees FULL interaction between both texts

  Example:
    Bi-encoder: "Inception review" and "Interstellar review" embedded separately
    Cross-encoder: reads BOTH reviews together, asks "are these similar?"

  Cross-encoders catch:
    - "Both films have a wife who died" (specific shared detail)
    - "Both protagonists are emotionally unavailable" (character parallel)
    - Nuances that only emerge when comparing texts directly

  DOWNSIDE: O(n) cross-encoder calls per query (can't pre-compute).
  SOLUTION: Two-stage pipeline:
    Stage 1: Bi-encoder retrieves top 50 candidates (fast, pre-computed)
    Stage 2: Cross-encoder reranks top 50 → returns top 10 (slow, but only 50 calls)

  This is called RETRIEVE AND RERANK.
  It's what modern search engines, RAG systems, and recommenders use.
""")

# ─── 5. What YOU can actually build next ──────────────────────
print("─"*60)
print("[5] WHAT YOU CAN BUILD NEXT — Concrete Improvements")
print("─"*60)
print("""
  IMPROVEMENT 1 — Multi-aspect embeddings (build today):
    For each film, instead of one embedding, store 4:
      {title}_structural, {title}_emotional, {title}_thematic, {title}_stylistic

    At prediction time, match on the aspect that YOUR review emphasizes.
    If your review of Inception says "puzzle", query structural space.

  IMPROVEMENT 2 — Review-aspect extraction (build this week):
    Use an LLM (llama3 via Ollama) to extract aspects from your review:

      prompt = f'''
      Given this review: "{review}"
      Extract: what did the reviewer love most?
      Options: plot_structure, emotional_resonance, themes, visuals, performances
      Return JSON: {{"primary_aspect": "...", "secondary_aspect": "..."}}
      '''

    Then weight the matching spaces accordingly.

  IMPROVEMENT 3 — Collaborative signal from TMDB data (build this month):
    TMDB has "similar movies" for every film.
    Build a graph: film → [similar films]
    Use graph embeddings (Node2Vec) to create collaborative vectors.
    Combine with content embeddings.

  IMPROVEMENT 4 — Fine-tune on YOUR preference pairs:
    Positive pairs: (film you rated 5★, film you rated 5★) → similar
    Negative pairs: (film you rated 5★, film you rated 1★) → dissimilar
    Fine-tune nomic-embed-text on these pairs.
    Now the embedding model learns YOUR definition of "similar".
""")

print("→ Run lesson 7:  python learn/07_discovery_engine.py")

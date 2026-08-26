"""
LESSON 3: THE PREDICTION ENGINE — k-NN Regression in Embedding Space
Run: python learn/03_prediction_engine.py
"""
import sys; sys.path.insert(0, ".")
import numpy as np

print("\n" + "="*60)
print("  LESSON 3: PREDICTION ENGINE")
print("="*60)

from src.vector_store import collection_count
if collection_count("my_lens") == 0:
    print("⚠  Run: python pipeline.py --limit 50  first")
    exit(1)

# ─── 1. What is k-NN regression? ──────────────────────────────
print("""
[1] k-NN REGRESSION — the algorithm powering predictions

  Classic k-NN regression on tabular data:
    - You have 100 houses with known prices
    - New house arrives: find 5 most similar houses by features
    - Predict price = average of those 5 houses' prices

  YOUR system: same idea but in embedding space
    - You have 3500 movies with YOUR ratings
    - New movie arrives: find 20 most similar movies by embedding
    - Predict rating = WEIGHTED average (closer = more weight)

  The embedding is the "feature space". Instead of:
    [bedrooms=3, sqft=1200, location=suburban]
  You have:
    [0.023, -0.184, 0.091, ... (768 numbers)]
  where the numbers encode the film's MEANING and YOUR reaction to it.
""")

# ─── 2. Trace one prediction manually ─────────────────────────
print("─"*60)
print("[2] MANUAL PREDICTION TRACE — step by step")
print("─"*60)

from src.embedder import embed_text
from src.vector_store import query_similar
from src.tmdb_client import get_movie_info, build_movie_dna_text, build_my_lens_text

target = "Parasite"
print(f"\n  Predicting your rating for: '{target}'\n")

# Step 1: Build query text
query_text = f"Movie: {target}"
print(f"  Step 1 — Query text:\n    '{query_text}'\n")

# Step 2: Embed it
q_emb = embed_text(query_text)
print(f"  Step 2 — Embedded to vector of length {len(q_emb)}\n")

# Step 3: Find nearest neighbours
result = query_similar("my_lens", q_emb, n_results=10)
print("  Step 3 — 10 nearest neighbours in your watch history:")
print(f"  {'Similarity':>10}  {'Rating':>7}  Title")
print(f"  {'─'*10}  {'─'*7}  {'─'*30}")

sims    = []
ratings = []
for i in range(len(result["ids"][0])):
    meta = result["metadatas"][0][i]
    dist = result["distances"][0][i]
    sim  = 1 - dist
    rat  = meta.get("rating")
    if rat is not None:
        try:
            rat = float(rat)
            sims.append(sim)
            ratings.append(rat)
            print(f"  {sim:>10.4f}  {rat:>7.1f}  {meta['title']}")
        except (ValueError, TypeError):
            pass

# Step 4: Weighted average
print()
if sims:
    sims_arr    = np.array(sims)
    ratings_arr = np.array(ratings)
    weighted    = np.dot(sims_arr, ratings_arr) / sims_arr.sum()
    rounded     = round(weighted * 2) / 2

    print(f"  Step 4 — Weighted average:")
    print(f"    Σ(similarity × rating) / Σ(similarity)")
    print(f"    Raw prediction: {weighted:.4f}")
    print(f"    Rounded to 0.5: {rounded} ★\n")

# ─── 3. Why weighted average beats simple average ─────────────
print("─"*60)
print("[3] WHY WEIGHTED AVERAGE (not simple average)?")
print("─"*60)
if sims:
    simple = np.mean(ratings_arr)
    print(f"""
  Simple average of top-10 ratings: {simple:.4f}
  Weighted average:                  {weighted:.4f}

  The difference: movies that are MORE similar get MORE weight.
  A 0.95-similarity film should count 10x more than a 0.50-similarity film.

  This is called SIMILARITY-WEIGHTED k-NN.
  It's more accurate than uniform k-NN for this kind of prediction.
""")

# ─── 4. The 3-space combination ───────────────────────────────
print("─"*60)
print("[4] 3-SPACE COMBINATION — why it's better than 1 space")
print("─"*60)
print("""
  Imagine you want to predict if you'd like a new restaurant.

  Space 1 — Your diary (what YOU wrote about similar places):
    "I loved this tiny ramen place because it felt personal and authentic"
    → finds restaurants you described with similar language
    Weight: 50% (most personal)

  Space 2 — Restaurant DNA (objective info):
    "Small Japanese noodle restaurant, family-owned, quiet atmosphere"
    → finds restaurants with similar objective features
    Weight: 30%

  Space 3 — Reviews from strangers:
    "Cozy hidden gem, not flashy, but the broth is incredible"
    → finds restaurants with similar public reception
    Weight: 20%

  Each space captures something the others miss.
  Together they're more robust than any single space.

  In ML this is called ENSEMBLE — combining multiple weak signals
  into a stronger prediction. Same idea as Random Forests, boosting, etc.
""")

# ─── 5. Where to improve ──────────────────────────────────────
print("─"*60)
print("[5] HOW TO MAKE THIS BETTER (real AI engineering)")
print("─"*60)
print("""
  Current system: fixed weights (0.5, 0.3, 0.2)
  
  IMPROVEMENT 1 — Learn weights from your data:
    Hold out 200 rated movies you've already watched.
    Try all combinations of weights.
    Pick weights that minimize prediction error on held-out set.
    This is hyperparameter tuning / cross-validation.

  IMPROVEMENT 2 — Add temporal decay:
    Movies you watched recently reflect current taste better.
    Add a time-decay factor: recent watches count more.
    weight_final = similarity × recency_weight

  IMPROVEMENT 3 — Genre-specific models:
    You might rate horror films differently than romance.
    Train separate predictors per genre cluster.

  IMPROVEMENT 4 — Fine-tune the embedding model:
    Train nomic-embed-text further on YOUR reviews.
    Your language patterns ("world class", "must watch", "logicless fights")
    would get embedded more accurately.
    This is called domain adaptation / fine-tuning.
""")

print("→ Run lesson 4:  python learn/04_taste_dna.py")

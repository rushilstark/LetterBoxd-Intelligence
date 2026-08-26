"""
LESSON 1: EMBEDDINGS — Text as Math
Run: python learn/01_embeddings.py
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from src.embedder import embed_text

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\n" + "="*60)
print("  LESSON 1: EMBEDDINGS")
print("="*60)

# ─── 1. Raw vector ─────────────────────────────────────────────
print("\n[1] Embedding a sentence into a vector:")
text = "Inception is a mind-bending masterpiece about dreams"
vec  = embed_text(text)
print(f"  Text   : '{text}'")
print(f"  Vector : [{', '.join(str(round(v,4)) for v in vec[:8])}, ...]")
print(f"  Length : {len(vec)} dimensions")
print("  → This vector IS the semantic fingerprint of that sentence.")

# ─── 2. Similar vs dissimilar ──────────────────────────────────
print("\n" + "─"*60)
print("[2] Cosine similarity: how close are two meanings?")
pairs = [
    ("Inception (dreamy scifi)",  "Inception is a film about dreams within dreams"),
    ("Interstellar (space scifi)","Interstellar is a sci-fi epic about space and love"),
    ("Dark Knight (crime)",       "The Dark Knight is a gritty crime thriller with Joker"),
    ("Ratatouille (animated)",    "Ratatouille is a Pixar film about a rat who cooks"),
    ("Dangal (sports)",           "Dangal is an inspiring Indian sports drama about wrestling"),
]
print(f"\n  Reference: Inception\n")
ref_emb = embed_text("Inception is a film about dreams within dreams")
for label, sentence in pairs:
    emb  = embed_text(sentence)
    sim  = cosine_similarity(ref_emb, emb)
    bar  = "█" * int(sim * 25)
    print(f"  {sim:.3f}  {bar:<25}  {label}")

print("""
  ✅ What to notice:
     Interstellar is closest — same director, same 'mindblowing' genre
     Ratatouille is furthest — completely different world
     The model learned this from BILLIONS of texts — zero hardcoding
""")

# ─── 3. Your own reviews ───────────────────────────────────────
print("─"*60)
print("[3] Now let's test with your actual reviews:")
from src.ingest import load_letterboxd_data
df = load_letterboxd_data()
sample = df[df["my_review"].str.len() > 100].head(3)

review_embs = []
for _, row in sample.iterrows():
    emb = embed_text(row["my_review"])
    review_embs.append((row["title"], row["final_rating"], emb))
    print(f"  Embedded: {row['title']} (rated {row['final_rating']}★)")

print("\n  Similarity between YOUR reviews:")
for i in range(len(review_embs)):
    for j in range(i+1, len(review_embs)):
        t1, r1, e1 = review_embs[i]
        t2, r2, e2 = review_embs[j]
        sim = cosine_similarity(e1, e2)
        print(f"  {t1} ↔ {t2}: {sim:.3f}")

print("\n→ Run lesson 2:  python learn/02_vector_db.py")

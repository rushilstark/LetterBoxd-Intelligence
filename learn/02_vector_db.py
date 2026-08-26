"""
LESSON 2: VECTOR DATABASES — Storing & Searching Meaning
Run: python learn/02_vector_db.py
(Run AFTER pipeline.py --limit 50 so there's data)
"""
import sys; sys.path.insert(0, ".")
import numpy as np
import time
from src.embedder import embed_text
from src.vector_store import get_collection, collection_count, query_similar
from src.config import CHROMA_MY_LENS_COLLECTION, CHROMA_MOVIE_DNA_COLLECTION

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\n" + "="*60)
print("  LESSON 2: VECTOR DATABASES")
print("="*60)

count = collection_count(CHROMA_MY_LENS_COLLECTION)
if count == 0:
    print("\n⚠  No data yet! Run this first:")
    print("     python pipeline.py --limit 50")
    exit(1)

print(f"\n  Vector store has {count} movies embedded.\n")

# ─── 1. Manual brute-force search ─────────────────────────────
print("[1] BRUTE FORCE SEARCH (what we'd do without a vector DB)")
print("─"*60)

query   = "a film that changed how I think about life"
q_emb   = embed_text(query)

# Fetch all stored embeddings manually
from src.vector_store import get_all_embeddings
all_data = get_all_embeddings(CHROMA_MY_LENS_COLLECTION)
embs     = np.array(all_data["embeddings"])
metas    = all_data["metadatas"]

print(f"\n  Query: '{query}'")
print(f"  Searching through {len(embs)} movies manually...\n")

t0 = time.time()
scores = []
for i, emb in enumerate(embs):
    sim = cosine_similarity(q_emb, emb)
    scores.append((sim, metas[i]))
scores.sort(reverse=True)
brute_ms = (time.time() - t0) * 1000

print("  Top 5 results (brute force):")
for sim, meta in scores[:5]:
    print(f"    {sim:.3f}  {meta['title']} (rated {meta.get('rating','?')}★)")
print(f"\n  Time taken: {brute_ms:.1f}ms for {len(embs)} movies")

# ─── 2. ChromaDB HNSW search ──────────────────────────────────
print("\n" + "─"*60)
print("[2] CHROMADB HNSW SEARCH (what we actually use)")
print("─"*60)

t0 = time.time()
result = query_similar(CHROMA_MY_LENS_COLLECTION, q_emb, n_results=5)
chroma_ms = (time.time() - t0) * 1000

print(f"\n  Same query, ChromaDB HNSW:")
for i in range(len(result["ids"][0])):
    meta = result["metadatas"][0][i]
    dist = result["distances"][0][i]
    sim  = 1 - dist  # ChromaDB returns cosine distance, not similarity
    print(f"    {sim:.3f}  {meta['title']} (rated {meta.get('rating','?')}★)")
print(f"\n  Time taken: {chroma_ms:.1f}ms")
print(f"\n  → Same results. But HNSW scales to MILLIONS of vectors.")
print(f"    At 1M movies, brute force = ~1000x slower than HNSW.\n")

# ─── 3. The distance vs similarity thing ──────────────────────
print("─"*60)
print("[3] DISTANCE vs SIMILARITY — the sign flip")
print("─"*60)
print("""
  ChromaDB returns 'distance', not 'similarity'.
  For cosine space:

    distance   = 1 - cosine_similarity
    similarity = 1 - distance

  So distance=0.0 means IDENTICAL, distance=1.0 means UNRELATED.
  We flip it in predictor.py:

    similarity = 1 - dist
    weight     = similarity   (closer = more weight in prediction)
""")

# ─── 4. Upsert semantics ──────────────────────────────────────
print("─"*60)
print("[4] UPSERT — why we use it instead of INSERT")
print("─"*60)
print("""
  In SQL you'd do INSERT. In vector DBs you do UPSERT.

  UPSERT = UPDATE if exists, INSERT if not.

  This is critical for the pipeline because:
  - If you re-run pipeline.py, you don't want duplicates
  - If you update a review, you want to replace the old embedding
  - The doc_id (title + year) is the primary key

  In ChromaDB:
    col.upsert(ids=["inception_2010"], embeddings=[...], ...)
    # → replaces if inception_2010 exists, adds if not
""")

print("→ Run lesson 3:  python learn/03_prediction_engine.py")

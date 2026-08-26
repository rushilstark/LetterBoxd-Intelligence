"""
LESSON 4: TASTE DNA — Dimensionality Reduction & Clustering
Run: python learn/04_taste_dna.py
"""
import sys; sys.path.insert(0, ".")
import numpy as np

print("\n" + "="*60)
print("  LESSON 4: TASTE DNA — Dimensionality Reduction")
print("="*60)

from src.vector_store import get_all_embeddings, collection_count
if collection_count("my_lens") < 20:
    print("⚠  Run: python pipeline.py --limit 100  first"); exit(1)

data    = get_all_embeddings("my_lens")
embs    = np.array(data["embeddings"])
metas   = data["metadatas"]
print(f"\n  Loaded {len(embs)} movie embeddings, each {embs.shape[1]}-dimensional\n")

# ─── 1. Why reduce dimensions? ────────────────────────────────
print("[1] THE CURSE OF DIMENSIONALITY")
print("─"*60)
print(f"""
  You have {embs.shape[1]} dimensions. You can't plot that.
  Even if you could, you'd only see 2 or 3 at a time.

  The goal: find a 2D projection that PRESERVES structure.
  - Movies that are close in 768D → should be close in 2D
  - Movies that are far in 768D  → should be far in 2D

  Two approaches:
    PCA  — linear, fast, deterministic, loses non-linear structure
    UMAP — non-linear, slower, preserves local topology
""")

# ─── 2. PCA from scratch ──────────────────────────────────────
print("[2] PCA — HOW IT ACTUALLY WORKS")
print("─"*60)
print("""
  Step 1: Center the data (subtract mean)
  Step 2: Compute covariance matrix (how do dimensions vary together?)
  Step 3: Find eigenvectors of covariance matrix
           → these are the directions of maximum variance
  Step 4: Project data onto top-2 eigenvectors → 2D coords

  The math:
    cov_matrix = (X.T @ X) / n
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    top2 = eigenvectors[:, -2:]      # 2 directions of most spread
    coords_2d = X @ top2             # project all points
""")

# Manual PCA
X = embs - embs.mean(axis=0)                          # center
cov = (X.T @ X) / len(X)                              # covariance matrix
eigvals, eigvecs = np.linalg.eigh(cov)                # eigen decomposition
top2 = eigvecs[:, -2:]                                # top 2 eigenvectors
coords_pca = X @ top2                                  # project

variance_explained = eigvals[-2:] / eigvals.sum() * 100
print(f"  Manual PCA result: {coords_pca.shape[0]} points in 2D")
print(f"  Variance explained by PC1: {variance_explained[1]:.1f}%")
print(f"  Variance explained by PC2: {variance_explained[0]:.1f}%")
print(f"  Total captured:            {sum(variance_explained):.1f}%")
print(f"  → The other {100-sum(variance_explained):.1f}% of meaning is lost.")
print(f"  → That's why 2D plots are approximate, not perfect.\n")

# ─── 3. K-Means clustering ────────────────────────────────────
print("─"*60)
print("[3] K-MEANS CLUSTERING — finding your taste groups")
print("─"*60)
print("""
  Algorithm:
    1. Place K centres randomly in 768D space
    2. Assign each movie to its nearest centre (by cosine distance)
    3. Move each centre to the MEAN of its assigned movies
    4. Repeat steps 2-3 until centres stop moving (convergence)

  This is an UNSUPERVISED algorithm — no labels needed.
  It discovers natural groups in your data.
""")

from sklearn.cluster import KMeans
import pandas as pd

n_clusters = min(6, len(embs) // 3)
kmeans  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels  = kmeans.fit_predict(embs)

df = pd.DataFrame([{
    "title":   m.get("title",""),
    "rating":  m.get("rating"),
    "cluster": int(labels[i]),
} for i, m in enumerate(metas)])

df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

print(f"\n  Found {n_clusters} clusters in your {len(embs)}-movie history:\n")
for cid in range(n_clusters):
    sub = df[df["cluster"]==cid]
    avg = sub["rating"].mean()
    top = sub.nlargest(3, "rating")["title"].tolist()
    print(f"  Cluster {cid+1}: {len(sub)} films | avg {avg:.1f}★")
    print(f"    Top films: {', '.join(top)}")
    print()

# ─── 4. Inertia — how to pick K ───────────────────────────────
print("─"*60)
print("[4] HOW DO YOU PICK THE RIGHT NUMBER OF CLUSTERS? (Elbow Method)")
print("─"*60)
print("""
  Run K-Means with K=2,3,...,12. For each, compute INERTIA:
  → sum of squared distances from each point to its cluster centre

  Plot inertia vs K. Look for the "elbow" — where adding
  more clusters stops helping much.

  This is a classic unsupervised learning problem.
  There's no single right answer — it's about interpretability.
""")

inertias = []
k_range = range(2, min(10, len(embs)//2))
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=5)
    km.fit(embs)
    inertias.append(km.inertia_)

print("  K → Inertia (lower = tighter clusters):")
for k, inertia in zip(k_range, inertias):
    bar = "█" * int(30 * inertia / inertias[0])
    print(f"    K={k}: {bar}")

print("""
  Pick the K where the bar stops shrinking fast = the "elbow".
  In the Streamlit app, the cluster slider lets you explore this.
""")

print("→ Run lesson 5:  python learn/05_what_next.py")

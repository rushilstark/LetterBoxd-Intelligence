"""
Taste profiler: clusters your watch history in embedding space
and generates human-readable cluster labels.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Dict

from src.vector_store import get_all_embeddings
from src.config import CHROMA_MY_LENS_COLLECTION, CHROMA_MOVIE_DNA_COLLECTION


# Cluster labels based on common patterns in cinema taste
from collections import Counter
import sqlite3
import json
from src.config import TMDB_CACHE_DB, CHROMA_MY_LENS_COLLECTION, CHROMA_MOVIE_DNA_COLLECTION

def _fetch_cached_info_batch(movies_keys: List[tuple]) -> List[dict]:
    if not movies_keys:
        return []
    try:
        conn = sqlite3.connect(TMDB_CACHE_DB)
        cursor = conn.cursor()
        
        results = []
        chunk_size = 50
        for i in range(0, len(movies_keys), chunk_size):
            chunk = movies_keys[i:i+chunk_size]
            placeholders = ", ".join(["(?, ?)"] * len(chunk))
            flat_params = []
            for t, y in chunk:
                flat_params.append(t)
                flat_params.append(int(y) if y and str(y) not in ("nan", "None", "<NA>") else 0)
            
            cursor.execute(f"SELECT title, year, data FROM movie_cache WHERE (title, year) IN ({placeholders})", flat_params)
            for row in cursor.fetchall():
                try:
                    data = json.loads(row[2])
                    results.append(data)
                except Exception:
                    pass
        conn.close()
        return results
    except Exception as e:
        print(f"[taste_profile] Error loading cached batch: {e}")
        return []

def _generate_cluster_label(movies_info: List[dict], default_label: str) -> str:
    if not movies_info:
        return default_label
        
    genres = []
    directors = []
    decades = []
    
    for info in movies_info:
        if not info.get("found"):
            continue
        genres.extend(info.get("genres", []))
        if info.get("director"):
            directors.append(info["director"])
        if info.get("year"):
            try:
                yr = int(info["year"])
                if yr > 1900:
                    decades.append(yr // 10 * 10)
            except ValueError:
                pass
                
    if not genres:
        return default_label
        
    # Get top 2 genres
    genre_counts = Counter(genres)
    top_genres = [g for g, _ in genre_counts.most_common(2)]
    genre_str = " / ".join(top_genres)
    
    # Get top director if they appear multiple times
    dir_counts = Counter(directors)
    dir_str = ""
    if dir_counts:
        top_dir, count = dir_counts.most_common(1)[0]
        if count >= 2:
            # Use last name for clean formatting
            name_parts = top_dir.split()
            last_name = name_parts[-1] if name_parts else top_dir
            # Check if there is a second director
            common = dir_counts.most_common(2)
            if len(common) > 1 and common[1][1] >= 2:
                second_name = common[1][0].split()[-1] if common[1][0].split() else common[1][0]
                dir_str = f" ({last_name} & {second_name})"
            else:
                dir_str = f" (by {top_dir})"
                
    # Get dominant decade if it makes up > 35% of the cluster
    dec_str = ""
    if decades:
        dec_counts = Counter(decades)
        top_dec, count = dec_counts.most_common(1)[0]
        if count / len(decades) >= 0.35:
            dec_str = f"{str(top_dec)}s "
            
    label = f"{dec_str}{genre_str}{dir_str}"
    return label or default_label


def get_taste_clusters(n_clusters: int = 8) -> Dict:
    """
    Cluster the user's watched movies in the 'my_lens' embedding space.

    Returns:
        {
            labels: list[int],          # cluster id per movie
            centers: np.ndarray,        # cluster centroids
            movies: list[dict],         # [{title, year, rating, cluster}]
            n_clusters: int,
            cluster_stats: list[dict],  # per-cluster summary
        }
    """
    data = get_all_embeddings(CHROMA_MY_LENS_COLLECTION)
    embs_raw = data.get("embeddings")
    n_embs = len(embs_raw) if embs_raw is not None else 0
    if n_embs < n_clusters:
        return {"error": "Not enough data. Run the pipeline first."}

    embeddings = np.array(data["embeddings"])
    metas      = data["metadatas"]

    # ── KMeans clustering ──────────────────────────────────────────────────
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # ── Build movie list with cluster assignment ────────────────────────────
    movies = []
    for i, meta in enumerate(metas):
        movies.append({
            "title":   meta.get("title", ""),
            "year":    meta.get("year"),
            "rating":  meta.get("rating"),
            "cluster": int(labels[i]),
        })

    # ── Per-cluster stats ──────────────────────────────────────────────────
    df = pd.DataFrame(movies)
    cluster_stats = []
    for cid in range(n_clusters):
        subset = df[df["cluster"] == cid]
        ratings = pd.to_numeric(subset["rating"], errors="coerce").dropna()
        top_titles = subset.nlargest(5, "rating")["title"].tolist() if len(subset) > 0 else []
        
        # Load cached TMDB data to auto-label
        keys = [(row["title"], row["year"]) for _, row in subset.iterrows()]
        infos = _fetch_cached_info_batch(keys)
        label = _generate_cluster_label(infos, f"Cluster {cid + 1}")
        
        cluster_stats.append({
            "cluster_id":  cid,
            "size":        len(subset),
            "avg_rating":  round(float(ratings.mean()), 2) if len(ratings) > 0 else None,
            "top_titles":  top_titles,
            "label":       label,
        })

    return {
        "labels":        labels.tolist(),
        "centers":       kmeans.cluster_centers_,
        "movies":        movies,
        "n_clusters":    n_clusters,
        "cluster_stats": sorted(cluster_stats, key=lambda x: x["avg_rating"] or 0, reverse=True),
    }



def get_taste_evolution(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average rating per year to show taste evolution over time.
    """
    df = movies_df.copy()
    df["year_watched"] = pd.to_datetime(df["watch_date"], errors="coerce").dt.year
    df["final_rating"] = pd.to_numeric(df["final_rating"], errors="coerce")
    evolution = (
        df.dropna(subset=["year_watched", "final_rating"])
        .groupby("year_watched")
        .agg(
            avg_rating=("final_rating", "mean"),
            count=("final_rating", "count"),
            titles=("title", lambda x: list(x[:3])),
        )
        .reset_index()
    )
    return evolution


def get_embedding_2d(collection_name: str = CHROMA_MY_LENS_COLLECTION) -> pd.DataFrame:
    """
    Reduce embeddings to 2D using UMAP for scatter plot visualization.
    Falls back to PCA if UMAP not available.
    """
    data = get_all_embeddings(collection_name)
    embs_raw = data.get("embeddings")
    n_embs = len(embs_raw) if embs_raw is not None else 0
    if n_embs < 5:
        return pd.DataFrame()

    embeddings = np.array(data["embeddings"])
    metas      = data["metadatas"]

    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        coords  = reducer.fit_transform(embeddings)
    except Exception:
        from sklearn.decomposition import PCA
        pca    = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(embeddings)

    rows = []
    for i, meta in enumerate(metas):
        rows.append({
            "x":      float(coords[i, 0]),
            "y":      float(coords[i, 1]),
            "title":  meta.get("title", ""),
            "year":   meta.get("year"),
            "rating": meta.get("rating"),
            "review": meta.get("review_preview", ""),
        })
    return pd.DataFrame(rows)


def semantic_search(query: str, n_results: int = 10) -> List[Dict]:
    """
    Search your personal watch history using natural language.
    E.g. "movies that made me feel lonely" or "films with amazing endings"
    """
    from src.embedder import embed_text
    from src.vector_store import query_similar

    emb = embed_text(query)
    if not emb:
        return []

    # Search primarily in my_lens (your subjective experience)
    qr = query_similar(CHROMA_MY_LENS_COLLECTION, emb, n_results=n_results)
    results = []
    for i in range(len(qr["ids"][0])):
        meta = qr["metadatas"][0][i]
        dist = qr["distances"][0][i]
        results.append({
            "title":      meta.get("title", ""),
            "year":       meta.get("year"),
            "rating":     meta.get("rating"),
            "similarity": round(1 - dist, 3),
            "review":     meta.get("review_preview", ""),
        })
    return results

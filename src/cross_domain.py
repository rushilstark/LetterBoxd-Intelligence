"""
cross_domain.py

Unified cultural taste engine. Films, books, and music all embedded
into the SAME 1024-dim vector space using mxbai-embed-large.

Key insight: thematic similarity is domain-agnostic.
Murakami's 'Norwegian Wood', Bon Iver's 'Skinny Love', and 
Wong Kar-wai's 'In the Mood for Love' all share: longing, memory,
melancholia, beauty in loss. The embedding model captures this.

When you query the cross-domain space with your film taste centroid,
you find books and music with similar emotional/thematic DNA.
"""

import numpy as np
import pandas as pd
from typing import Optional

# Using local app config/modules
from src.embedder import embed_batch, embed_text
from src.vector_store import get_collection, upsert_embedding, query_similar, get_all_embeddings
from src.config import EMBED_BATCH_SIZE

CHROMA_CULTURAL_COLLECTION = 'cultural_fingerprint'

def index_music(music_df: pd.DataFrame, force: bool = False):
    collection = get_collection(CHROMA_CULTURAL_COLLECTION)
    texts = music_df['text_for_embedding'].tolist()
    names = music_df['name'].tolist()
    playcounts = [0] * len(names)
    
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch_texts = texts[i:i+EMBED_BATCH_SIZE]
        batch_names = names[i:i+EMBED_BATCH_SIZE]
        batch_playcounts = playcounts[i:i+EMBED_BATCH_SIZE]
        
        embeddings = embed_batch(batch_texts)
        for j, emb in enumerate(embeddings):
            doc_id = f"music_{batch_names[j]}"
            metadata = {
                "domain": "music",
                "name": batch_names[j],
                "playcount": batch_playcounts[j]
            }
            upsert_embedding(CHROMA_CULTURAL_COLLECTION, doc_id, emb, metadata, batch_texts[j])
        print(f"Indexed music batch {i//EMBED_BATCH_SIZE + 1}")

def index_books(books_df: pd.DataFrame, force: bool = False):
    collection = get_collection(CHROMA_CULTURAL_COLLECTION)
    texts = books_df['text_for_embedding'].tolist()
    titles = books_df['title'].tolist()
    authors = books_df['author'].tolist()
    ratings = books_df['rating'].tolist() if 'rating' in books_df.columns else [0.0]*len(titles)
    years = books_df['year'].tolist() if 'year' in books_df.columns else [""]*len(titles)
    
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch_texts = texts[i:i+EMBED_BATCH_SIZE]
        embeddings = embed_batch(batch_texts)
        for j, emb in enumerate(embeddings):
            doc_id = f"book_{titles[i+j]}_{authors[i+j]}"
            metadata = {
                "domain": "book",
                "title": titles[i+j],
                "author": authors[i+j],
                "rating": float(ratings[i+j]),
                "year": str(years[i+j])
            }
            upsert_embedding(CHROMA_CULTURAL_COLLECTION, doc_id, emb, metadata, batch_texts[j])

def index_films(films_df: pd.DataFrame, tmdb_data: dict = None, force: bool = False):
    tmdb_data = tmdb_data or {}
    
    titles = films_df['title'].tolist()
    ratings = films_df['rating'].tolist() if 'rating' in films_df.columns else [0.0]*len(titles)
    years = films_df['year'].tolist() if 'year' in films_df.columns else [""]*len(titles)
    
    texts = []
    for i, title in enumerate(titles):
        info = tmdb_data.get(title, {})
        if info:
            director = info.get('director', 'Unknown')
            genres = ", ".join(info.get('genres', []))
            keywords = ", ".join(info.get('keywords', []))
            overview = info.get('overview', '')
            text = f"Film: {title} | Director: {director} | Genre: {genres} | Themes: {keywords} | Plot: {overview} | My Rating: {ratings[i]}/5"
        else:
            text = f"Film: {title} | My Rating: {ratings[i]}/5"
        texts.append(text)
        
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch_texts = texts[i:i+EMBED_BATCH_SIZE]
        embeddings = embed_batch(batch_texts)
        for j, emb in enumerate(embeddings):
            doc_id = f"film_{titles[i+j]}_{years[i+j]}"
            metadata = {
                "domain": "film",
                "title": titles[i+j],
                "rating": float(ratings[i+j]),
                "year": str(years[i+j])
            }
            upsert_embedding(CHROMA_CULTURAL_COLLECTION, doc_id, emb, metadata, batch_texts[j])

def get_domain_embeddings(domain: str) -> tuple[np.ndarray, list[dict]]:
    try:
        col = get_collection(CHROMA_CULTURAL_COLLECTION)
        res = col.get(where={"domain": domain}, include=["embeddings", "metadatas"])
        embeddings = res.get("embeddings", [])
        metadatas = res.get("metadatas", [])
        if not embeddings:
            return np.array([]), []
        return np.array(embeddings), metadatas
    except Exception:
        return np.array([]), []

def get_taste_centroid(domain: str) -> Optional[np.ndarray]:
    embeddings, _ = get_domain_embeddings(domain)
    if embeddings.size == 0:
        return None
    return np.mean(embeddings, axis=0)

def cross_domain_recommend(source_domain: str, target_domain: str, n: int = 10) -> list[dict]:
    centroid = get_taste_centroid(source_domain)
    if centroid is None:
        return []
        
    res = query_similar(
        CHROMA_CULTURAL_COLLECTION, 
        centroid.tolist(), 
        n_results=n, 
        where={"domain": target_domain}
    )
    
    hits = []
    if not res.get("ids") or not res["ids"][0]:
        return hits
        
    for i in range(len(res["ids"][0])):
        hits.append({
            "title_or_name": res["metadatas"][0][i].get("title", res["metadatas"][0][i].get("name", "")),
            "domain": res["metadatas"][0][i].get("domain", ""),
            "similarity": 1 - res["distances"][0][i],
            "metadata": res["metadatas"][0][i]
        })
    return hits

def cross_domain_predict_rating(title: str, domain: str = 'film', year: str = '') -> dict:
    text = f"{domain.capitalize()}: {title}"
    emb = embed_text(text)
    if not emb:
        return {}
        
    res = query_similar(CHROMA_CULTURAL_COLLECTION, emb, n_results=50)
    
    top_films = []
    top_books = []
    top_music = []
    
    film_ratings = []
    film_weights = []
    
    book_ratings = []
    book_weights = []
    
    if res.get("ids") and res["ids"][0]:
        for i in range(len(res["ids"][0])):
            meta = res["metadatas"][0][i]
            dist = res["distances"][0][i]
            sim = max(0.0, 1.0 - dist)
            d = meta.get("domain")
            
            if d == "film":
                top_films.append(meta)
                if "rating" in meta:
                    film_ratings.append(float(meta["rating"]))
                    film_weights.append(sim)
            elif d == "book":
                top_books.append(meta)
                if "rating" in meta:
                    book_ratings.append(float(meta["rating"]))
                    book_weights.append(sim)
            elif d == "music":
                top_music.append(meta)
                
    film_pred = np.average(film_ratings, weights=film_weights) if film_ratings else 0.0
    book_pred = np.average(book_ratings, weights=book_weights) if book_ratings else 0.0
    
    music_bonus = min(1.0, len(top_music) / 10.0) * 5.0
    
    weights = {"film": 0.5, "book": 0.3, "music": 0.2}
    valid_preds = {}
    if film_ratings: valid_preds["film"] = film_pred
    if book_ratings: valid_preds["book"] = book_pred
    if top_music: valid_preds["music"] = music_bonus
    
    if not valid_preds:
        final_pred = 0.0
    else:
        total_w = sum(weights[k] for k in valid_preds)
        final_pred = sum(weights[k] * v for k, v in valid_preds.items()) / total_w
        
    return {
        "predicted_rating": round(final_pred * 2) / 2,
        "confidence": 0.0,
        "top_films": top_films[:5],
        "top_books": top_books[:5],
        "top_music": top_music[:5]
    }

def search_cross_domain(query: str, n: int = 15) -> list[dict]:
    emb = embed_text(query)
    if not emb:
        return []
    res = query_similar(CHROMA_CULTURAL_COLLECTION, emb, n_results=n)
    
    hits = []
    if res.get("ids") and res["ids"][0]:
        for i in range(len(res["ids"][0])):
            hits.append({
                "id": res["ids"][0][i],
                "domain": res["metadatas"][0][i].get("domain", ""),
                "similarity": 1 - res["distances"][0][i],
                "metadata": res["metadatas"][0][i]
            })
    return hits

def get_cultural_coherence_score() -> Optional[float]:
    film_centroid = get_taste_centroid("film")
    music_centroid = get_taste_centroid("music")
    
    if film_centroid is None or music_centroid is None:
        return None
        
    sim = np.dot(film_centroid, music_centroid) / (np.linalg.norm(film_centroid) * np.linalg.norm(music_centroid))
    return float(sim)

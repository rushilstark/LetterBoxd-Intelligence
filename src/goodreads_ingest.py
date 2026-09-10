"""
Goodreads CSV parser + Google Books enrichment.
"""
import time
import json
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

def load_goodreads_data(csv_path: str) -> pd.DataFrame:
    raw_df = pd.read_csv(csv_path, encoding="utf-8-sig")
    raw_df.columns = [c.strip().lower().replace(" ", "_") for c in raw_df.columns]
    
    df = raw_df[raw_df.get("my_rating", 0) > 0].copy()
    
    out = pd.DataFrame()
    out["title"] = df.get("title", "")
    out["author"] = df.get("author", "")
    out["my_rating"] = df.get("my_rating", 0)
    out["avg_rating"] = df.get("average_rating", 0)
    out["year"] = df.get("original_publication_year", df.get("year_published", ""))
    out["shelves"] = df.get("bookshelves", "")
    out["date_read"] = df.get("date_read", "")
    out["my_review"] = df.get("my_review", "")
    
    avg = out["my_rating"].mean() if not out.empty else 0
    print(f"[goodreads] Loaded {len(raw_df)} books | {len(out)} rated | avg rating: {avg:.1f}")
    
    return out


def enrich_with_google_books(title: str, author: str, api_key: str = '') -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = DATA_DIR / "books_cache.json"
    
    cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
        except Exception:
            pass

    cache_key = f"{title}_{author}"
    if cache_key in cache:
        return cache[cache_key]

    url = "https://www.googleapis.com/books/v1/volumes"
    query = f"intitle:{title}+inauthor:{author}"
    params = {"q": query}
    if api_key:
        params["key"] = api_key

    result = {}
    try:
        resp = requests.get(url, params=params, timeout=10)
        time.sleep(0.3)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("items", [])
            if items:
                vol = items[0].get("volumeInfo", {})
                result = {
                    "description": vol.get("description", ""),
                    "categories": vol.get("categories", []),
                    "page_count": vol.get("pageCount", 0)
                }
    except Exception:
        pass

    cache[cache_key] = result
    with open(cache_file, "w") as f:
        json.dump(cache, f)

    return result


def build_book_text(book: dict, enriched: dict = None) -> str:
    enriched = enriched or {}
    genre = ", ".join(enriched.get("categories", [])) if enriched.get("categories") else "Unknown"
    desc = enriched.get("description", "")
    if len(desc) > 100:
        desc = desc[:100] + "..."
        
    parts = []
    parts.append(f"Book: {book.get('title')}")
    parts.append(f"Author: {book.get('author')}")
    parts.append(f"Genre: {genre}")
    parts.append(f"My Rating: {book.get('my_rating')}/5")
    if desc:
        parts.append(f"Description: {desc}")
    if book.get('shelves') and pd.notna(book.get('shelves')):
        parts.append(f"Shelves: {book.get('shelves')}")
        
    return " | ".join(parts)


def load_and_enrich_goodreads(csv_path: str, api_key: str = '') -> pd.DataFrame:
    df = load_goodreads_data(csv_path)
    
    enriched_descriptions = []
    categories_list = []
    text_for_embedding = []
    
    for _, row in df.iterrows():
        book_dict = row.to_dict()
        enriched = enrich_with_google_books(row["title"], row["author"], api_key)
        
        enriched_descriptions.append(enriched.get("description", ""))
        categories_list.append(enriched.get("categories", []))
        text_for_embedding.append(build_book_text(book_dict, enriched))
        
    df["enriched_description"] = enriched_descriptions
    df["categories"] = categories_list
    df["text_for_embedding"] = text_for_embedding
    df["domain"] = "book"
    
    return df

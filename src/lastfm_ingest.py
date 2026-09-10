"""
lastfm_ingest.py
Fetches user's music taste from Last.fm API.
Free API, no OAuth needed. Get key at https://www.last.fm/api

Returns a DataFrame of top artists with their tags for cross-domain embedding.
"""
import time
import json
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

def get_top_artists(username: str, api_key: str, limit: int = 100, period: str = 'overall') -> list[dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = DATA_DIR / "lastfm_cache.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            pass

    url = "http://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "user.gettopartists",
        "user": username,
        "api_key": api_key,
        "limit": limit,
        "period": period,
        "format": "json"
    }
    
    resp = requests.get(url, params=params)
    time.sleep(0.2)
    data = resp.json()
    
    artists_raw = data.get("topartists", {}).get("artist", [])
    results = []
    
    for i, a in enumerate(artists_raw):
        name = a.get("name")
        playcount = int(a.get("playcount", 0))
        
        tag_params = {
            "method": "artist.gettoptags",
            "artist": name,
            "api_key": api_key,
            "format": "json"
        }
        tag_resp = requests.get(url, params=tag_params)
        time.sleep(0.2)
        tag_data = tag_resp.json()
        
        tags_raw = tag_data.get("toptags", {}).get("tag", [])
        tags = [t.get("name") for t in tags_raw[:5]]
        
        results.append({
            "name": name,
            "playcount": playcount,
            "tags": tags
        })
        
        if (i + 1) % 10 == 0:
            print(f"[lastfm] Fetched {i + 1}/{len(artists_raw)} artists...")

    with open(cache_file, "w") as f:
        json.dump(results, f)
        
    return results


def build_artist_text(artist: dict) -> str:
    tags_str = ", ".join(artist.get("tags", []))
    return f"Artist: {artist.get('name')} | Tags: {tags_str} | Playcount: {artist.get('playcount')}"


def load_lastfm_data(username: str, api_key: str) -> pd.DataFrame:
    artists = get_top_artists(username, api_key)
    df = pd.DataFrame(artists)
    df["text_for_embedding"] = df.apply(build_artist_text, axis=1)
    df["domain"] = "music"
    
    all_tags = [tag for tags in df["tags"] for tag in tags]
    from collections import Counter
    tag_counts = Counter(all_tags)
    top_tags_str = ", ".join([f"{k} ({v})" for k, v in tag_counts.most_common(2)])
    
    print(f"[lastfm] Loaded {len(df)} artists | top tags: {top_tags_str}...")
    
    return df

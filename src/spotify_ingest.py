"""
spotify_ingest.py
Fetches user's music taste from Spotify API using spotipy.
Requires: SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET
Get free credentials at: https://developer.spotify.com/dashboard

Audio features are the key insight here:
- valence: 0=sad/dark, 1=happy/euphoric  
- energy: 0=calm/acoustic, 1=intense/loud
- acousticness: 0=electronic, 1=acoustic instruments
- danceability: how rhythmically consistent
- instrumentalness: 0=vocals, 1=no vocals (instrumental)

These correlate with film preferences:
- Low valence + high energy → psychological thrillers, horror
- High valence + high energy → action, comedy
- Low valence + low energy → arthouse, slow cinema
- High acousticness → character-driven dramas
"""

import os
import json
import pandas as pd
from pathlib import Path

# Lazy import — spotipy is optional. Install with: pip install spotipy
try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    _SPOTIPY_AVAILABLE = True
except ImportError:
    _SPOTIPY_AVAILABLE = False

def _require_spotipy():
    if not _SPOTIPY_AVAILABLE:
        raise ImportError(
            "spotipy not installed. Run: pip install spotipy\n"
            "Then get free credentials at: https://developer.spotify.com/dashboard"
        )

def get_spotify_client(client_id: str, client_secret: str):
    _require_spotipy()

    cache_path = Path("src/data/.spotify_cache")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    auth_manager = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri="http://localhost:8888/callback",
        scope="user-top-read user-library-read",
        cache_path=str(cache_path)
    )
    return spotipy.Spotify(auth_manager=auth_manager)

def get_top_artists(sp, limit=50, time_range='medium_term') -> list:
    results = sp.current_user_top_artists(limit=limit, time_range=time_range)
    artists = []
    for item in results.get('items', []):
        artists.append({
            'name': item.get('name'),
            'genres': item.get('genres', []),
            'popularity': item.get('popularity'),
            'followers_count': item.get('followers', {}).get('total', 0)
        })
    return artists

def get_top_tracks_with_features(sp, limit=50, time_range='medium_term') -> list:

    results = sp.current_user_top_tracks(limit=limit, time_range=time_range)
    tracks = []
    track_ids = []
    for item in results.get('items', []):
        track_ids.append(item.get('id'))
        artist_name = item.get('artists', [{}])[0].get('name') if item.get('artists') else 'Unknown'
        tracks.append({
            'name': item.get('name'),
            'artist': artist_name,
            'id': item.get('id')
        })
    
    if not track_ids:
        return []
    
    features = sp.audio_features(tracks=track_ids)
    
    tracks_with_features = []
    for track, feature in zip(tracks, features):
        if feature:
            track.update({
                'valence': feature.get('valence'),
                'energy': feature.get('energy'),
                'acousticness': feature.get('acousticness'),
                'danceability': feature.get('danceability'),
                'instrumentalness': feature.get('instrumentalness'),
                'tempo': feature.get('tempo')
            })
            tracks_with_features.append(track)
    return tracks_with_features

def compute_taste_summary(artists: list, tracks: list) -> dict:
    if not tracks:
        return {}
    
    avg_valence = sum(t.get('valence', 0) for t in tracks) / len(tracks)
    avg_energy = sum(t.get('energy', 0) for t in tracks) / len(tracks)
    avg_acousticness = sum(t.get('acousticness', 0) for t in tracks) / len(tracks)
    
    all_genres = []
    for a in artists:
        all_genres.extend(a.get('genres', []))
        
    genre_counts = {}
    for g in all_genres:
        genre_counts[g] = genre_counts.get(g, 0) + 1
    top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_genres = [g for g, c in top_genres]
    
    if avg_valence > 0.5 and avg_energy > 0.5:
        dominant_mood = 'Euphoric'
    elif avg_valence <= 0.5 and avg_energy > 0.5:
        dominant_mood = 'Intense/Dark'
    elif avg_valence > 0.5 and avg_energy <= 0.5:
        dominant_mood = 'Chill'
    else:
        dominant_mood = 'Melancholic'
        
    return {
        'avg_valence': avg_valence,
        'avg_energy': avg_energy,
        'avg_acousticness': avg_acousticness,
        'top_genres': top_genres,
        'dominant_mood': dominant_mood
    }

def build_artist_text(artist: dict, taste_summary: dict = None) -> str:
    genres = ", ".join(artist.get('genres', []))
    mood_str = f" → {taste_summary['dominant_mood']}" if taste_summary and 'dominant_mood' in taste_summary else ""
    valence = taste_summary.get('avg_valence', 0.5) if taste_summary else 0.5
    energy = taste_summary.get('avg_energy', 0.5) if taste_summary else 0.5
    
    valence_desc = "high valence" if valence > 0.5 else "low valence"
    energy_desc = "high energy" if energy > 0.5 else "low energy"
    
    style = f"{valence_desc} ({valence:.2f}), {energy_desc} ({energy:.2f}){mood_str}"
    return f"Artist: {artist.get('name')} | Genres: {genres} | Style: {style}"

def load_spotify_data(client_id: str, client_secret: str) -> tuple[pd.DataFrame, dict]:
    cache_file = Path("src/data/spotify_cache.json")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
                df = pd.DataFrame(data['artists'])
                return df, data['taste_summary']
        except Exception:
            pass

    sp = get_spotify_client(client_id, client_secret)
    artists = get_top_artists(sp, limit=50)
    tracks = get_top_tracks_with_features(sp, limit=50)
    
    taste_summary = compute_taste_summary(artists, tracks)
    
    for artist in artists:
        artist['text_for_embedding'] = build_artist_text(artist, taste_summary)
        artist['domain'] = 'music'
        
    df = pd.DataFrame(artists)
    
    with open(cache_file, "w") as f:
        data_to_save = {
            'artists': artists,
            'taste_summary': taste_summary
        }
        json.dump(data_to_save, f)
        
    return df, taste_summary

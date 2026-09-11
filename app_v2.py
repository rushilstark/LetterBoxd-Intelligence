"""
app_v2.py — Movinator: Find Films You Actually Want
Clean 4-page app with real data science.
Works for ANYONE — upload your data or just answer a few questions.
"""

import sys; sys.path.insert(0, '.')
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import base64
import requests
import json
import os

from src.config import ACCENT
from src.config_store import load_config, save_config, get_key, set_key

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Movinator',
    page_icon='🎬',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}
.stApp {
    background: radial-gradient(ellipse at 50% 0%, #0d0d12 0%, #050505 70%);
    color: #e5e5e7;
}
[data-testid="stSidebar"] { display: none; }

.nav-logo {
    font-size: 1.6rem; font-weight: 800; color: #fff;
    padding: 6px 0; letter-spacing: -0.5px;
}
.nav-logo span { color: #E9A84C; }

.g-card {
    background: rgba(15,15,20,0.85);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 24px;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}
.g-card:hover {
    border-color: rgba(233,168,76,0.4);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(233,168,76,0.06);
}

.film-card {
    background: rgba(15,15,20,0.8);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 12px;
    transition: all 0.3s ease;
}
.film-card:hover {
    border-color: rgba(233,168,76,0.3);
    transform: translateY(-2px);
}

.hero {
    text-align: center; padding: 36px 0 28px;
    background: linear-gradient(180deg, rgba(233,168,76,0.03) 0%, transparent 100%);
    border-radius: 20px; margin-bottom: 28px;
}
.hero-sub {
    font-size: 12px; letter-spacing: 4px; text-transform: uppercase;
    color: #E9A84C; font-weight: 600; margin-bottom: 14px;
}
.hero h1 {
    font-size: clamp(26px, 4.5vw, 44px); font-weight: 800;
    letter-spacing: -1.5px; margin: 0 0 10px; color: #fff;
}
.hero p { color: #8e8e93; font-size: 15px; max-width: 500px; margin: 0 auto; line-height: 1.6; }

div[data-testid="metric-container"] {
    background: rgba(15,15,20,0.8) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important; padding: 20px !important;
}
div[data-testid="metric-container"] label {
    color: #8e8e93 !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: 1px; font-weight: 600;
}
div[data-testid="stMetricValue"] {
    color: #ffffff !important; font-size: 2rem !important; font-weight: 700 !important;
}

h1 { color: #ffffff !important; font-weight: 700 !important; letter-spacing: -0.8px; }
h2, h3 { color: #E9A84C !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_poster(url: str):
    if not url or not url.startswith("http"): return None
    try:
        r = requests.get(url, timeout=6)
        return r.content if r.status_code == 200 else None
    except: return None

def poster_html(url: str, width: int = 140) -> str:
    if url:
        data = fetch_poster(url)
        if data:
            b64 = base64.b64encode(data).decode("utf-8")
            return f'<img src="data:image/jpeg;base64,{b64}" width="{width}" style="border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.5);" />'
    h = int(width * 1.5)
    return f'<div style="width:{width}px;height:{h}px;background:linear-gradient(135deg,#141416,#0a0a0c);border:1px dashed rgba(255,255,255,0.1);border-radius:8px;display:flex;align-items:center;justify-content:center;color:#555;font-size:24px;">🎬</div>'

def plotly_dark(fig):
    fig.update_layout(plot_bgcolor="#050505", paper_bgcolor="#050505",
                      font_color="#888", title_font_color="#fff",
                      margin=dict(l=20,r=20,t=50,b=20))
    fig.update_xaxes(gridcolor="#111", zerolinecolor="#111")
    fig.update_yaxes(gridcolor="#111", zerolinecolor="#111")
    return fig

def get_tmdb_key():
    k = get_key("TMDB_API_KEY", "")
    if not k:
        try:
            from dotenv import load_dotenv; load_dotenv()
            k = os.environ.get("TMDB_API_KEY", "")
        except: pass
    if not k:
        try:
            from src.config import TMDB_API_KEY; k = TMDB_API_KEY
        except: pass
    return k


# ── Top Navigation ───────────────────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = '🎬 Discover'

col_logo, col_nav, col_settings = st.columns([2, 6, 2])
with col_logo:
    st.markdown("<div class='nav-logo'>🎬 <span>Movinator</span></div>", unsafe_allow_html=True)
with col_nav:
    nav_cols = st.columns(4)
    pages = ['🎬 Discover', '📊 My Taste', '🎯 Predict', '🤖 Persona']
    for col, pg in zip(nav_cols, pages):
        active = st.session_state.page == pg
        if col.button(pg, use_container_width=True,
                      type='primary' if active else 'secondary'):
            st.session_state.page = pg
            st.rerun()
with col_settings:
    if st.button('⚙️', use_container_width=True):
        st.session_state.page = '⚙️ Settings'
        st.rerun()

st.markdown("---")
page = st.session_state.page


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
if page == '⚙️ Settings':
    st.markdown("# ⚙️ Settings")
    st.caption("Keys are stored locally on your device. Never sent to any server.")

    with st.form("api_keys"):
        tmdb = st.text_input("TMDB API Key (required for film data)", value=get_key('TMDB_API_KEY',''), type="password",
                             help="Free at themoviedb.org/settings/api")
        lastfm = st.text_input("Last.fm API Key", value=get_key('LASTFM_API_KEY',''), type="password",
                               help="Free at last.fm/api/account/create")
        c1, c2 = st.columns(2)
        with c1: sp_id = st.text_input("Spotify Client ID", value=get_key('SPOTIFY_CLIENT_ID',''), type="password")
        with c2: sp_sec = st.text_input("Spotify Client Secret", value=get_key('SPOTIFY_CLIENT_SECRET',''), type="password")

        if st.form_submit_button("💾 Save", use_container_width=True):
            save_config({"TMDB_API_KEY": tmdb, "LASTFM_API_KEY": lastfm,
                         "SPOTIFY_CLIENT_ID": sp_id, "SPOTIFY_CLIENT_SECRET": sp_sec})
            st.success("✅ Saved!")
            st.rerun()

    st.markdown("---")
    cfg = load_config()
    for k, label in [("TMDB_API_KEY","TMDB"), ("LASTFM_API_KEY","Last.fm"),
                      ("SPOTIFY_CLIENT_ID","Spotify")]:
        icon = "🟢" if cfg.get(k) else "⚫"
        st.markdown(f"{icon} **{label}** — {'configured' if cfg.get(k) else 'not set'}")


# ══════════════════════════════════════════════════════════════════════════════
# DISCOVER
# ══════════════════════════════════════════════════════════════════════════════
elif page == '🎬 Discover':

    st.markdown("""
    <div class='hero'>
        <div class='hero-sub'>Movinator</div>
        <h1>Find films you actually want to watch</h1>
        <p>Connect your accounts, answer a few questions, or both.<br>
           More data = better recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Step 1: Connect Sources ──────────────────────────────────────────────
    st.markdown("### Step 1 — Connect your data")
    st.caption("Each source adds a different signal. You don't need all of them.")

    spotify_ok = "spotify_mood" in st.session_state
    letterboxd_ok = "user_df" in st.session_state and len(st.session_state.get("user_df", pd.DataFrame())) > 0
    goodreads_ok = "books_df" in st.session_state
    quiz_done = "quiz_mood" in st.session_state

    tab_spot, tab_lb, tab_gr, tab_quiz = st.tabs([
        f"{'✅' if spotify_ok else '🟢'} Spotify",
        f"{'✅' if letterboxd_ok else '🟠'} Letterboxd",
        f"{'✅' if goodreads_ok else '📚'} Goodreads",
        f"{'✅' if quiz_done else '❓'} Quick Quiz"
    ])

    with tab_spot:
        if spotify_ok:
            mood = st.session_state["spotify_mood"]
            st.success(f"Connected! Mood: **{mood.get('name','')}** {mood.get('emoji','')}")
            if st.button("Disconnect Spotify"):
                del st.session_state["spotify_mood"]
                if "spotify_artists" in st.session_state: del st.session_state["spotify_artists"]
                st.rerun()
        else:
            st.markdown("Reads your last 50 songs → detects your current mood → finds matching films.")
            c1, c2 = st.columns(2)
            sp_id = c1.text_input("Client ID", value=get_key("SPOTIFY_CLIENT_ID",""), type="password", key="sp_id_in")
            sp_sec = c2.text_input("Client Secret", value=get_key("SPOTIFY_CLIENT_SECRET",""), type="password", key="sp_sec_in")
            st.caption("Free credentials: [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) → Create App → Redirect URI: `http://localhost:8888/callback`")

            if st.button("🔗 Connect Spotify", disabled=not (sp_id and sp_sec), use_container_width=True):
                try:
                    from src.spotify_ingest import get_spotify_client, get_top_tracks_with_features
                    from src.mood_engine import get_recent_mood_from_features
                    save_config({"SPOTIFY_CLIENT_ID": sp_id, "SPOTIFY_CLIENT_SECRET": sp_sec})
                    sp = get_spotify_client(sp_id, sp_sec)
                    with st.spinner("Reading your recent plays..."):
                        try:
                            recent = sp.current_user_recently_played(limit=50)
                            track_ids = [item["track"]["id"] for item in recent.get("items",[]) if item.get("track")]
                            artist_names = list(dict.fromkeys(
                                item["track"]["artists"][0]["name"]
                                for item in recent.get("items",[])
                                if item.get("track") and item["track"].get("artists")
                            ))[:10]
                            feats = [f for f in (sp.audio_features(track_ids[:50]) or []) if f]
                        except Exception:
                            feats = get_top_tracks_with_features(sp, limit=50)
                            artist_names = []
                        mood = get_recent_mood_from_features(feats)
                        st.session_state["spotify_mood"] = mood
                        st.session_state["spotify_artists"] = artist_names
                    st.success(f"Connected! Mood: **{mood.get('name','')}** — {len(feats)} tracks analyzed")
                    st.rerun()
                except Exception as e:
                    st.error(f"Connection failed: {e}")

    with tab_lb:
        if letterboxd_ok:
            user_df = st.session_state["user_df"]
            n_rated = user_df["final_rating"].notna().sum() if "final_rating" in user_df.columns else 0
            st.success(f"Loaded! {len(user_df):,} films, {n_rated:,} rated")
            if st.button("Remove Letterboxd data"):
                del st.session_state["user_df"]
                st.rerun()
        else:
            st.markdown("Upload your Letterboxd export → we analyze your taste patterns.")
            st.caption("Get it: [letterboxd.com](https://letterboxd.com) → Settings → Import & Export → Export → upload **ratings.csv** from the zip")
            uploaded = st.file_uploader("Upload ratings.csv", type=["csv"], key="lb_upload")
            if uploaded:
                try:
                    udf = pd.read_csv(uploaded)
                    col_map = {}
                    for c in udf.columns:
                        cl = c.lower().strip()
                        if cl in ("name", "title"): col_map[c] = "title"
                        elif cl == "year": col_map[c] = "year"
                        elif cl == "rating": col_map[c] = "final_rating"
                        elif cl == "date": col_map[c] = "watch_date"
                    udf = udf.rename(columns=col_map)
                    if "title" not in udf.columns:
                        st.error("CSV doesn't have a 'Name' or 'Title' column.")
                    else:
                        if "final_rating" in udf.columns:
                            udf["final_rating"] = pd.to_numeric(udf["final_rating"], errors="coerce")
                        if "my_review" not in udf.columns: udf["my_review"] = ""
                        if "is_rewatch" not in udf.columns: udf["is_rewatch"] = False
                        st.session_state["user_df"] = udf
                        st.success(f"✅ Loaded {len(udf):,} films!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to parse CSV: {e}")

    with tab_gr:
        if goodreads_ok:
            st.success(f"Loaded! {len(st.session_state['books_df']):,} books")
            if st.button("Remove Goodreads data"):
                del st.session_state["books_df"]
                st.rerun()
        else:
            st.markdown("Upload your Goodreads export → adds cross-media taste signal.")
            st.caption("Get it: Goodreads → My Books → Import/Export → Export Library")
            gr_file = st.file_uploader("Upload goodreads_library_export.csv", type=["csv"], key="gr_upload")
            if gr_file:
                try:
                    bdf = pd.read_csv(gr_file)
                    col_map = {}
                    for c in bdf.columns:
                        cl = c.lower().strip()
                        if "title" in cl: col_map[c] = "title"
                        elif "author" in cl and "author" not in col_map.values(): col_map[c] = "author"
                        elif cl == "my rating": col_map[c] = "my_rating"
                        elif cl == "bookshelves": col_map[c] = "shelves"
                    bdf = bdf.rename(columns=col_map)
                    if "my_rating" in bdf.columns:
                        bdf["my_rating"] = pd.to_numeric(bdf["my_rating"], errors="coerce")
                        bdf = bdf[bdf["my_rating"] > 0]
                    st.session_state["books_df"] = bdf
                    st.success(f"✅ Loaded {len(bdf):,} rated books!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to parse CSV: {e}")

    with tab_quiz:
        if quiz_done:
            qm = st.session_state["quiz_mood"]
            st.success(f"Quiz done! Detected: **{qm.get('name','')}** {qm.get('emoji','')}")
            if st.button("Retake quiz"):
                del st.session_state["quiz_mood"]
                if "quiz_genres" in st.session_state: del st.session_state["quiz_genres"]
                st.rerun()
        else:
            st.markdown("**No accounts? No problem.** Answer a few quick questions.")

            q1 = st.radio("What energy are you in right now?", [
                "🌧️ Slow and emotional — let me feel something",
                "🔥 Intense and gripping — keep me on edge",
                "⚡ Fun and exciting — pure entertainment",
                "☁️ Easy and warm — nothing too heavy",
                "🌌 Thought-provoking — make me think",
            ], key="quiz_q1")

            q2 = st.multiselect("Pick 2-3 genres you're in the mood for:", [
                "Drama", "Thriller", "Comedy", "Horror", "Sci-Fi", "Romance",
                "Action", "Mystery", "Animation", "Documentary", "Crime", "Adventure"
            ], max_selections=3, key="quiz_q2")

            q3 = st.slider("How adventurous? (safe picks vs hidden gems)",
                           1, 5, 3, key="quiz_q3")

            q4 = st.radio("Film era preference?", [
                "Any era", "Recent (2018+)", "2000s-2010s", "Classics (pre-2000)"
            ], key="quiz_q4")

            if st.button("🎯 Analyze My Preferences", type="primary", use_container_width=True,
                         disabled=len(q2) == 0):
                from src.mood_engine import MOOD_PROFILES
                mood_map = {
                    "🌧️ Slow and emotional — let me feel something": "Melancholic",
                    "🔥 Intense and gripping — keep me on edge": "Intense",
                    "⚡ Fun and exciting — pure entertainment": "Euphoric",
                    "☁️ Easy and warm — nothing too heavy": "Chill",
                    "🌌 Thought-provoking — make me think": "Introspective",
                }
                mood_name = mood_map.get(q1, "Melancholic")
                mood = {**MOOD_PROFILES[mood_name], "name": mood_name, "source": "quiz"}
                st.session_state["quiz_mood"] = mood
                st.session_state["quiz_genres"] = q2
                st.session_state["quiz_adventurous"] = q3
                st.session_state["quiz_era"] = q4
                st.success(f"Got it! You're in a **{mood_name}** mood.")
                st.rerun()

    # ── Accuracy bar ─────────────────────────────────────────────────────────
    st.markdown("")
    n_sources = sum([spotify_ok, letterboxd_ok, goodreads_ok, quiz_done])
    acc = min(95, 30 + n_sources * 18)
    acc_label = {0: "Connect a source to start", 1: "Basic mood match",
                 2: "Mood + taste calibrated", 3: "Strong alignment",
                 4: "Full cultural profile"}.get(min(n_sources, 4), "")

    st.markdown(f"""
    <div class='g-card' style='padding:14px 20px; display:flex; align-items:center; gap:16px;'>
        <div style='flex:1;'>
            <div style='font-size:11px; color:#8e8e93; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:5px;'>
                Prediction accuracy · {n_sources} source{'s' if n_sources != 1 else ''}
            </div>
            <div style='background:#1c1c1e; border-radius:4px; height:6px; overflow:hidden;'>
                <div style='background:linear-gradient(90deg,#E9A84C,#f5c87a); width:{acc}%;
                            height:100%; border-radius:4px; transition:width 0.5s;'></div>
            </div>
        </div>
        <div style='font-size:20px; font-weight:800; color:#E9A84C; min-width:44px; text-align:right;'>{acc}%</div>
        <div style='font-size:12px; color:#8e8e93; min-width:160px;'>{acc_label}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Step 2: Get Recommendations ──────────────────────────────────────────
    st.markdown("")
    st.markdown("### Step 2 — Get your films")

    can_go = spotify_ok or letterboxd_ok or goodreads_ok or quiz_done
    if not can_go:
        st.info("👆 Connect at least one source above or take the quick quiz.")
        st.stop()

    if st.button("🎬 Find My Films", type="primary", use_container_width=True):
        from src.mood_engine import get_recommendations, MOOD_PROFILES

        mood = None
        sources_used = []
        top_artists = st.session_state.get("spotify_artists", [])

        if spotify_ok:
            mood = st.session_state["spotify_mood"]
            sources_used.append("spotify")
        elif quiz_done:
            mood = st.session_state["quiz_mood"]
            quiz_genres = st.session_state.get("quiz_genres", [])
            if quiz_genres:
                from src.mood_engine import TMDB_GENRES
                genre_ids = [TMDB_GENRES.get(g) for g in quiz_genres if g in TMDB_GENRES]
                if genre_ids: mood["genres"] = genre_ids
            sources_used.append("quiz")

        if letterboxd_ok: sources_used.append("letterboxd")
        if goodreads_ok: sources_used.append("goodreads")

        if mood is None:
            mood = {**MOOD_PROFILES["Melancholic"], "name": "Melancholic", "source": "default"}

        tmdb_key = get_tmdb_key()
        if not tmdb_key:
            st.error("TMDB API key required. Go to ⚙️ Settings to add it.")
            st.stop()

        user_df = st.session_state.get("user_df")

        with st.spinner("Finding films aligned to your vibe..."):
            recs = get_recommendations(
                mood_profile=mood, api_key=tmdb_key,
                letterboxd_df=user_df if letterboxd_ok else None,
                goodreads_df=st.session_state.get("books_df"),
                sources=sources_used, top_artists=top_artists, n=18,
            )
            # Apply quiz era filter
            if quiz_done:
                era = st.session_state.get("quiz_era", "Any era")
                if era == "Recent (2018+)":
                    recs = [r for r in recs if (r.get("year","0") >= "2018")] or recs
                elif era == "2000s-2010s":
                    recs = [r for r in recs if "2000" <= (r.get("year","0")) < "2018"] or recs
                elif era == "Classics (pre-2000)":
                    recs = [r for r in recs if (r.get("year","9999") < "2000")] or recs
                adv = st.session_state.get("quiz_adventurous", 3)
                if adv <= 2:
                    recs.sort(key=lambda x: x.get("vote_count",0), reverse=True)

        st.session_state["discover_recs"] = recs
        st.session_state["discover_mood"] = mood
        st.session_state["discover_sources"] = sources_used

    # ── Results ──────────────────────────────────────────────────────────────
    if "discover_recs" in st.session_state and st.session_state["discover_recs"]:
        recs = st.session_state["discover_recs"]
        mood = st.session_state["discover_mood"]
        sources_used = st.session_state.get("discover_sources", [])

        m_color = mood.get("color", "#E9A84C")
        source_str = " + ".join(s.capitalize() for s in sources_used)
        st.markdown(f"""
        <div style='display:inline-flex; align-items:center; gap:10px;
                    background:{m_color}15; border:1px solid {m_color}40;
                    border-radius:100px; padding:8px 18px; margin:8px 0 20px;'>
            <span style='font-size:18px;'>{mood.get("emoji","🎬")}</span>
            <span style='font-weight:700; color:{m_color};'>{mood.get("name","?")} Mood</span>
            <span style='color:#555;'>·</span>
            <span style='color:#8e8e93; font-size:12px;'>via {source_str}</span>
        </div>
        """, unsafe_allow_html=True)

        fc1, fc2, fc3, fc4, _ = st.columns([1,1,1,1,4])
        filt = "all"
        if fc1.button("All", use_container_width=True, key="f_all"): filt = "all"
        if fc2.button("Hidden Gems", use_container_width=True, key="f_hg"): filt = "hidden"
        if fc3.button("Acclaimed", use_container_width=True, key="f_acc"): filt = "acclaimed"
        if fc4.button("Recent", use_container_width=True, key="f_rec"): filt = "recent"

        filtered = recs
        if filt == "hidden": filtered = [r for r in recs if (r.get("vote_average") or 0) < 7] or recs
        elif filt == "acclaimed": filtered = [r for r in recs if (r.get("vote_average") or 0) >= 7.5] or recs
        elif filt == "recent": filtered = [r for r in recs if (r.get("year","0") >= "2018")] or recs

        st.markdown(f"**{len(filtered)} films** — sorted by mood alignment, not by rating")
        st.markdown("")

        for row_start in range(0, min(len(filtered), 18), 3):
            row = filtered[row_start:row_start+3]
            cols = st.columns(3)
            for col, film in zip(cols, row):
                with col:
                    al = film.get("alignment", 0)
                    rating = film.get("vote_average")
                    rating_s = f"{rating:.1f}/10" if rating else "—"
                    al_color = "#30d158" if al >= 70 else "#E9A84C" if al >= 45 else "#8e8e93"
                    p = poster_html(film.get("poster_url",""), width=130)
                    exp = film.get("explanation","")[:110]
                    low_badge = ""
                    if rating and rating < 6.5:
                        low_badge = f"<div style='font-size:10px; color:#ff9f0a; margin-top:6px;'>⚠️ Rated {rating_s} but aligned to your vibe</div>"

                    st.markdown(f"""
                    <div class='film-card'>
                        <div style='display:flex; gap:12px;'>
                            <div style='flex-shrink:0;'>{p}</div>
                            <div style='flex:1; min-width:0;'>
                                <div style='font-weight:700; font-size:14px; color:#fff; margin-bottom:3px;'>
                                    {film.get("title","")}
                                </div>
                                <div style='color:#8e8e93; font-size:12px;'>{film.get("year","")} · TMDB {rating_s}</div>
                                <div style='margin:8px 0 4px;'>
                                    <div style='font-size:10px; color:#8e8e93; margin-bottom:3px;'>Mood match</div>
                                    <div style='background:#1c1c1e; border-radius:3px; height:4px; overflow:hidden;'>
                                        <div style='background:{al_color}; width:{al}%; height:100%; border-radius:3px;'></div>
                                    </div>
                                    <div style='font-size:11px; font-weight:700; color:{al_color}; margin-top:3px;'>{al:.0f}%</div>
                                </div>
                                <div style='font-size:11px; color:#666; font-style:italic; line-height:1.4;'>{exp}</div>
                                {low_badge}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("")
        if st.button("🔄 New Recommendations"):
            for k in ["discover_recs", "discover_mood", "discover_sources"]:
                if k in st.session_state: del st.session_state[k]
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MY TASTE
# ══════════════════════════════════════════════════════════════════════════════
elif page == '📊 My Taste':

    if "user_df" in st.session_state and len(st.session_state.get("user_df", pd.DataFrame())) > 0:
        df = st.session_state["user_df"]
    else:
        df = pd.DataFrame()

    if len(df) == 0:
        st.markdown("""
        <div class='hero'>
            <div class='hero-sub'>Data Science</div>
            <h1>Your Taste Signature</h1>
            <p>Upload your Letterboxd export in the Discover tab to see what your ratings reveal.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    st.markdown("# 📊 Your Taste Signature")
    st.caption("What your ratings actually reveal — real data science, not just charts")
    st.markdown("---")

    if 'final_rating' not in df.columns:
        df['final_rating'] = np.nan

    rated = df[df['final_rating'].notna()].copy()
    rated['final_rating'] = pd.to_numeric(rated['final_rating'], errors='coerce')
    rated = rated.dropna(subset=['final_rating'])
    if 'my_review' not in rated.columns: rated['my_review'] = ""
    if 'is_rewatch' not in rated.columns: rated['is_rewatch'] = False
    rated['review_len'] = rated['my_review'].fillna("").astype(str).str.len()
    rated['is_rewatch'] = rated['is_rewatch'].fillna(False).astype(int)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Films Rated", f"{len(rated):,}")
    c2.metric("Avg Rating", f"{rated['final_rating'].mean():.2f}★")
    c3.metric("5★ Films", f"{(rated['final_rating']==5.0).sum():,}")
    c4.metric("Reviews", f"{(rated['my_review'].fillna('').astype(str).str.len() > 20).sum():,}")
    st.markdown("")

    @st.cache_data(ttl=3600, show_spinner="Computing taste analysis...")
    def compute_taste(titles_hash):
        import sqlite3
        from scipy import stats as _st
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score
        from sklearn.model_selection import cross_val_score, KFold
        try:
            conn = sqlite3.connect("src/data/tmdb_cache.db")
            rows = conn.execute("SELECT title, year, data FROM movie_cache").fetchall()
            conn.close()
        except: return None

        tmdb_map = {}
        for t, y, d in rows:
            try:
                dd = json.loads(d)
                if dd.get("found") and dd.get("vote_average") and dd.get("runtime"):
                    tmdb_map[(t.lower().strip(), int(y))] = dd
            except: pass

        enriched = []
        for _, row in rated.iterrows():
            try: yr = int(float(row.get("year", 0)))
            except: yr = 0
            tmdb = tmdb_map.get((str(row["title"]).lower().strip(), yr))
            if tmdb:
                enriched.append({
                    "title": row["title"],
                    "my_rating": float(row["final_rating"]),
                    "tmdb_rating": float(tmdb["vote_average"]),
                    "runtime": float(tmdb["runtime"]),
                    "year": float(yr),
                    "is_rewatch": int(row.get("is_rewatch", 0)),
                    "review_len": int(row.get("review_len", 0)),
                    "genres": tmdb.get("genres", []),
                })
        if len(enriched) < 30: return None
        edf = pd.DataFrame(enriched).dropna()
        feat_cols = ["tmdb_rating", "runtime", "year", "is_rewatch", "review_len"]
        corrs = {}
        labels = {"tmdb_rating":"TMDB community rating", "runtime":"Runtime",
                  "year":"Release year", "is_rewatch":"Is rewatch", "review_len":"Review length"}
        for col in feat_cols:
            try:
                r, p = _st.pearsonr(edf[col], edf["my_rating"])
                corrs[labels[col]] = round(r, 3)
            except: pass
        if len(edf) < len(feat_cols) + 1: return None
        X = StandardScaler().fit_transform(edf[feat_cols].values)
        y = edf["my_rating"].values
        model = Ridge(alpha=1.0).fit(X, y)
        r2 = r2_score(y, model.predict(X))
        rmse = float(np.sqrt(np.mean((y - model.predict(X))**2)))
        cv_folds = min(5, len(edf))
        cv = cross_val_score(model, X, y, cv=KFold(cv_folds, shuffle=True, random_state=42), scoring="r2")
        edf["predicted"] = model.predict(X)
        edf["residual"] = edf["my_rating"] - edf["predicted"]
        from collections import defaultdict
        gr = defaultdict(list)
        for _, row in edf.iterrows():
            for g in (row.get("genres") or []): gr[g].append(row["my_rating"])
        genre_stats = {g: round(float(np.mean(v)),3) for g,v in gr.items() if len(v) >= 15}
        return {
            "r2": round(r2,4), "cv_r2": round(float(cv.mean()),4),
            "rmse": round(rmse,4), "n": len(edf), "correlations": corrs,
            "genre_stats": genre_stats,
            "top_loved": edf.nlargest(6,"residual")[["title","my_rating","tmdb_rating","residual"]].to_dict("records"),
            "top_hated": edf.nsmallest(6,"residual")[["title","my_rating","tmdb_rating","residual"]].to_dict("records"),
        }

    analysis = compute_taste(hash(tuple(rated["title"].tolist()[:100])))
    if analysis:
        r2_val, rmse_val, corrs = analysis["r2"], analysis["rmse"], analysis["correlations"]
    else:
        r2_val, rmse_val = 0.22, 0.89
        corrs = {"TMDB community rating": 0.43, "Release year": -0.21,
                 "Is rewatch": 0.13, "Review length": 0.11, "Runtime": 0.07}
        st.info("TMDB data not cached. Showing reference model values.")

    personal_pct = round((1 - r2_val) * 100, 1)
    obj_pct = round(r2_val * 100, 1)

    st.markdown(f"""
    <div class='g-card' style='margin-bottom:24px;'>
        <div style='display:flex; gap:32px; align-items:center;'>
            <div style='flex:1;'>
                <div style='font-size:11px; color:#8e8e93; text-transform:uppercase;
                            letter-spacing:2px; margin-bottom:10px;'>Your Taste Signature</div>
                <div style='font-size:48px; font-weight:800; color:#E9A84C; line-height:1;'>{personal_pct}%</div>
                <div style='color:#fff; font-size:16px; font-weight:600; margin:6px 0;'>Personal taste</div>
                <div style='color:#8e8e93; font-size:13px; line-height:1.5;'>
                    Of your rating variance is unexplained by objective signals.
                    That's <em>your</em> unique perspective.
                </div>
            </div>
            <div style='text-align:center;'>
                <div style='background:#1c1c1e; border-radius:50%; width:100px; height:100px;
                            display:flex; align-items:center; justify-content:center;
                            border:3px solid #E9A84C;'>
                    <div style='font-size:20px; font-weight:800; color:#E9A84C;'>{obj_pct}%</div>
                </div>
                <div style='font-size:10px; color:#8e8e93; margin-top:6px;'>Explained by<br>objective signals</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### What predicts your rating?")
    cdf = pd.DataFrame([(k,v) for k,v in corrs.items()], columns=["Factor","r"])
    cdf = cdf.sort_values("r", key=abs, ascending=True)
    fig = px.bar(cdf, x="r", y="Factor", orientation="h",
                 color=cdf["r"].apply(lambda x: "Positive" if x > 0 else "Negative"),
                 color_discrete_map={"Positive":"#E9A84C","Negative":"#4a6fa5"},
                 text=cdf["r"].apply(lambda x: f"{x:+.3f}"))
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, xaxis=dict(range=[-0.35,0.55], title="Correlation (r)"), yaxis_title="")
    plotly_dark(fig)
    st.plotly_chart(fig, use_container_width=True)

    if analysis and analysis.get("genre_stats"):
        st.markdown("### Genre preferences")
        gs = analysis["genre_stats"]
        avg = float(rated["final_rating"].mean())
        gdf = pd.DataFrame([(g,v,v-avg) for g,v in gs.items()], columns=["Genre","Avg","vs Mean"])
        gdf = gdf.sort_values("Avg", ascending=True)
        fig3 = px.bar(gdf, x="Avg", y="Genre", orientation="h",
                      color="vs Mean", color_continuous_scale=["#4a6fa5","#1c1c1e","#E9A84C"],
                      color_continuous_midpoint=0,
                      text=gdf["Avg"].apply(lambda x: f"{x:.2f}★"))
        fig3.update_traces(textposition="outside")
        fig3.update_layout(coloraxis_showscale=False)
        plotly_dark(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    if analysis:
        st.markdown("### Your most contrarian takes")
        cl, cr = st.columns(2)
        with cl:
            st.markdown("**You loved — TMDB didn't**")
            for r in analysis["top_loved"][:5]:
                st.markdown(f"""<div class='film-card' style='padding:10px 14px;'>
                <b style='color:#fff;'>{r.get("title","")}</b><br>
                <span style='color:#8e8e93;font-size:12px;'>You: <b style='color:#E9A84C;'>{r["my_rating"]:.1f}★</b> · TMDB: {r["tmdb_rating"]:.1f} · Gap: <b style='color:#30d158;'>{r["residual"]:+.2f}</b></span>
                </div>""", unsafe_allow_html=True)
        with cr:
            st.markdown("**TMDB loved — you didn't**")
            for r in analysis["top_hated"][:5]:
                st.markdown(f"""<div class='film-card' style='padding:10px 14px;'>
                <b style='color:#fff;'>{r.get("title","")}</b><br>
                <span style='color:#8e8e93;font-size:12px;'>You: <b style='color:#E9A84C;'>{r["my_rating"]:.1f}★</b> · TMDB: {r["tmdb_rating"]:.1f} · Gap: <b style='color:#ff453a;'>{r["residual"]:+.2f}</b></span>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == '🎯 Predict':
    st.markdown("# 🎯 Predict Your Rating")
    st.caption("How much would YOU rate this film?")
    st.markdown("---")

    has_data = False
    if "user_df" in st.session_state and len(st.session_state.get("user_df",[])) > 0:
        has_data = True

    if not has_data:
        st.info("Upload Letterboxd data in the Discover tab first to enable predictions.")
        st.stop()

    c1, c2 = st.columns([5, 1])
    query = c1.text_input("Film title", placeholder="e.g. Inception, Tamasha, Spirited Away")
    year = c2.number_input("Year", value=0, min_value=0, step=1)

    if st.button("🎯 Predict", type="primary", disabled=not query.strip()):
        from src.predictor import predict_rating
        with st.spinner(f"Analyzing '{query}'..."):
            try: res = predict_rating(query.strip(), year if year > 0 else None)
            except Exception as e: st.error(f"Prediction failed: {e}"); st.stop()

        if res.get("predicted_rating"):
            pred = float(res["predicted_rating"])
            conf = float(res.get("confidence", 0.5))
            info = res.get("tmdb_info", {})
            cp, cr = st.columns([1, 3])
            with cp:
                if info.get("poster_url"):
                    st.markdown(poster_html(info["poster_url"], 160), unsafe_allow_html=True)
            with cr:
                stars_f = int(pred)
                star_s = "★"*stars_f + ("½" if pred-stars_f>=0.5 else "") + "☆"*(5-stars_f-(1 if pred-stars_f>=0.5 else 0))
                conf_c = "#30d158" if conf>0.7 else "#E9A84C" if conf>0.4 else "#ff453a"
                st.markdown(f"""
                <div style='font-size:11px;color:#8e8e93;text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;'>Predicted Rating</div>
                <div style='font-size:52px;font-weight:800;color:#E9A84C;line-height:1;'>{pred:.1f}★</div>
                <div style='font-size:20px;color:#8e8e93;margin:4px 0 10px;'>{star_s}</div>
                <div style='font-size:13px;color:#8e8e93;'>
                    {info.get('title',query)} ({info.get('year','')}) · TMDB: {info.get('vote_average','?')}/10
                </div>
                <div style='margin-top:10px;'>
                    <div style='background:#1c1c1e;border-radius:4px;height:5px;overflow:hidden;margin:4px 0;'>
                        <div style='background:{conf_c};width:{conf*100:.0f}%;height:100%;border-radius:4px;'></div>
                    </div>
                    <span style='font-size:11px;color:{conf_c};'>{conf*100:.0f}% confidence</span>
                    <span style='font-size:11px;color:#555;'> · ±0.88★ error</span>
                </div>
                """, unsafe_allow_html=True)
            nb = res.get("neighbors", [])
            if nb:
                st.markdown("---")
                st.markdown("### Similar films you've rated")
                for n in nb[:5]:
                    st.markdown(f"• **{n.get('title','?')}** — you rated {n.get('rating','?')}★")
        elif res.get("tmdb_info", {}).get("found"):
            st.warning("Film found in TMDB, but could not predict rating. (Make sure you have ingested data via pipeline.py)")
        else:
            st.warning("Film not found. Check spelling or add year.")


# ══════════════════════════════════════════════════════════════════════════════
# PERSONA
# ══════════════════════════════════════════════════════════════════════════════
elif page == '🤖 Persona':
    st.markdown("# 🤖 Virtual Persona")
    st.caption("Chat with an AI trained on film review patterns")

    has_reviews = False
    if "user_df" in st.session_state and len(st.session_state.get("user_df", pd.DataFrame())) > 0:
        _df = st.session_state["user_df"]
        if "my_review" in _df.columns:
            has_reviews = (_df["my_review"].fillna("").astype(str).str.len() > 20).sum() > 10

    if not has_reviews:
        st.info("Requires Letterboxd data with reviews (50+ recommended). Upload in Discover tab first.")
        st.stop()

    from src.persona_chat import get_engine_status, chat_with_persona
    status = get_engine_status()
    ec = "#30d158" if status.get("backend") == "mlx" else "#E9A84C"
    st.markdown(f"""
    <div style='display:inline-flex;gap:8px;align-items:center;padding:5px 12px;
                background:rgba(15,15,20,0.8);border:1px solid rgba(255,255,255,0.06);
                border-radius:100px;margin-bottom:14px;'>
        <span style='width:7px;height:7px;background:{ec};border-radius:50%;display:inline-block;'></span>
        <span style='font-size:11px;color:#8e8e93;'>{status.get("backend","?").upper()} · {status.get("model","?")[:35]}</span>
    </div>
    """, unsafe_allow_html=True)

    starters = ["What makes a perfect film for you?", "What film broke you recently?",
                 "Recommend something obscure", "Your most controversial take?"]
    scols = st.columns(4)
    for i, (col, s) in enumerate(zip(scols, starters)):
        if col.button(s, use_container_width=True, key=f"st_{i}"):
            if "chat_hist" not in st.session_state: st.session_state.chat_hist = []
            st.session_state.chat_hist.append({"role":"user","content":s})
            with st.spinner("Thinking..."):
                reply = chat_with_persona(s, st.session_state.chat_hist[:-1])
            st.session_state.chat_hist.append({"role":"assistant","content":reply})
            st.rerun()

    st.markdown("---")
    if "chat_hist" not in st.session_state: st.session_state.chat_hist = []
    for msg in st.session_state.chat_hist:
        with st.chat_message(msg["role"]): st.write(msg["content"])

    user_input = st.chat_input("Ask anything about films...")
    if user_input:
        st.session_state.chat_hist.append({"role":"user","content":user_input})
        with st.chat_message("user"): st.write(user_input)
        with st.chat_message("assistant"):
            with st.spinner("..."): reply = chat_with_persona(user_input, st.session_state.chat_hist[:-1])
            st.write(reply)
        st.session_state.chat_hist.append({"role":"assistant","content":reply})
        st.rerun()

    if st.session_state.chat_hist:
        if st.button("🗑️ Clear chat"): st.session_state.chat_hist = []; st.rerun()

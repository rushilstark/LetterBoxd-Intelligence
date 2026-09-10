"""
Letterboxd Intelligence — Streamlit Dashboard
"""
import sys
sys.path.insert(0, ".")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from src.config import APP_TITLE, ACCENT
from src.ingest import load_letterboxd_data
from src.vector_store import collection_count, get_all_embeddings
from src.config import CHROMA_MY_LENS_COLLECTION, CHROMA_MOVIE_DNA_COLLECTION

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Letterboxd Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Ultra-dark premium minimalist background with subtle radial gradient */
.stApp {
    background: radial-gradient(circle at 50% 50%, #0c0c0e 0%, #050505 100%);
    color: #e5e5e7;
}

[data-testid="stSidebar"] {
    background: #09090b !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Glassmorphic Metric cards */
div[data-testid="metric-container"] {
    background: rgba(20, 20, 25, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 24px 20px !important;
    backdrop-filter: blur(8px);
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
}
div[data-testid="metric-container"]:hover {
    border-color: rgba(233, 168, 76, 0.4) !important;
    background: rgba(233, 168, 76, 0.03) !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(233, 168, 76, 0.05);
}
div[data-testid="metric-container"] label {
    color: #8e8e93 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}
div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    margin-top: 4px;
}

/* Section headers */
h1 {
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: -0.8px;
}
h2, h3 {
    color: #E9A84C !important;
    font-weight: 600 !important;
    letter-spacing: -0.3px;
}
h4 {
    color: #f5f5f7 !important;
    font-weight: 500 !important;
}

/* Premium Movie card */
.movie-card {
    background: rgba(20, 20, 22, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    backdrop-filter: blur(10px);
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}
.movie-card:hover {
    border-color: rgba(233, 168, 76, 0.35);
    transform: translateY(-4px) scale(1.01);
    box-shadow: 0 12px 40px -10px rgba(233, 168, 76, 0.12), 0 4px 20px rgba(0, 0, 0, 0.3);
}
.movie-title {
    font-weight: 600;
    font-size: 16px;
    color: #ffffff;
    letter-spacing: 0.1px;
}
.movie-meta {
    color: #8e8e93;
    font-size: 13px;
    margin-top: 8px;
    line-height: 1.5;
}
.movie-rating {
    color: #E9A84C;
    font-weight: 700;
    font-size: 20px;
}

/* Star rating colors and style */
.stars {
    color: #E9A84C;
    font-size: 15px;
    letter-spacing: 2px;
    text-shadow: 0 0 10px rgba(233, 168, 76, 0.3);
}

/* Inputs & buttons */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: rgba(10, 10, 12, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    font-size: 15px !important;
    transition: all 0.3s !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
    border-color: #E9A84C !important;
    box-shadow: 0 0 15px rgba(233, 168, 76, 0.15) !important;
}

.stButton button {
    background: transparent !important;
    color: #E9A84C !important;
    font-weight: 600 !important;
    border: 1px solid rgba(233, 168, 76, 0.5) !important;
    border-radius: 8px !important;
    padding: 10px 28px !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    font-size: 12px !important;
}
.stButton button:hover {
    background: #E9A84C !important;
    color: #050505 !important;
    box-shadow: 0 0 20px rgba(233, 168, 76, 0.4) !important;
    border-color: #E9A84C !important;
}

/* Custom separator line */
hr {
    border-color: rgba(255, 255, 255, 0.05) !important;
    margin: 2rem 0 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #8e8e93 !important;
    font-weight: 600;
    font-family: 'Outfit';
    font-size: 14px;
    letter-spacing: 0.5px;
    padding: 12px 20px;
    transition: all 0.3s;
}
.stTabs [aria-selected="true"] {
    color: #E9A84C !important;
    border-bottom-color: #E9A84C !important;
}

/* Sidebar selectbox */
div[data-testid="stSelectbox"] > div {
    background: rgba(10, 10, 12, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px;
}

.grid-title {
    font-size: 13px;
    color: #e5e5e7;
    margin-top: 8px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-weight: 600;
}

/* Scrollbar Customization */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #050505;
}
::-webkit-scrollbar-thumb {
    background: #1c1c1e;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #E9A84C;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
import requests as _requests
import base64

def stars(rating):
    if rating is None or str(rating) in ("nan", "None"):
        return "—"
    r = float(rating)
    full  = int(r)
    half  = 1 if (r - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_poster(url: str):
    """Fetch poster image bytes from TMDB. Cached for 24h."""
    if not url or not url.startswith("http"):
        return None
    try:
        r = _requests.get(url, timeout=6)
        return r.content if r.status_code == 200 else None
    except Exception:
        return None

def get_poster_html(url: str, width: int = 200, alt_text: str = "") -> str:
    """Get HTML img tag with base64 encoded poster, or a stylish placeholder."""
    height = int(width * 1.5)
    if url:
        data = fetch_poster(url)
        if data:
            b64 = base64.b64encode(data).decode("utf-8")
            return f'<img src="data:image/jpeg;base64,{b64}" width="{width}" style="border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); object-fit: cover; aspect-ratio: 2/3;" />'
    
    # Minimalist placeholder card
    return f'''
    <div style="
        width: {width}px;
        height: {height}px;
        background: linear-gradient(135deg, #141414 0%, #070707 100%);
        border: 1px dashed rgba(255, 255, 255, 0.15);
        border-radius: 6px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: #777;
        font-family: 'Outfit', sans-serif;
        text-align: center;
        padding: 12px;
        box-sizing: border-box;
    ">
        <span style="font-size: 24px; margin-bottom: 8px;">🎬</span>
        <span style="font-size: 11px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; line-height: 1.3;">
            {alt_text or "No Poster"}
        </span>
    </div>
    '''

def show_poster(url: str, width: int = 200, caption: str = ""):
    """Display a TMDB poster reliably using bytes (bypasses CSP)."""
    # Keep show_poster as fallback, but get_poster_html is preferred for html layouts
    if not url:
        return
    data = fetch_poster(url)
    if data:
        st.image(data, width=width, caption=caption)

@st.cache_data(ttl=3600)
def load_data():
    return load_letterboxd_data()

@st.cache_data(ttl=3600)
def get_store_counts():
    return {
        "my_lens":  collection_count(CHROMA_MY_LENS_COLLECTION),
        "movie_dna": collection_count(CHROMA_MOVIE_DNA_COLLECTION),
    }

def plotly_dark_layout(fig):
    fig.update_layout(
        plot_bgcolor="#050505",
        paper_bgcolor="#050505",
        font_color="#888",
        title_font_color="#fff",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_xaxes(gridcolor="#111", zerolinecolor="#111")
    fig.update_yaxes(gridcolor="#111", zerolinecolor="#111")
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## 🎬 Letterboxd\n### Intelligence Engine")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Dashboard",
        "🔮 Predict Rating",
        "✍️ Writing Analytics",
        "🤖 Virtual Persona",
        "🧠 Review Copilot",
        "🔍 Semantic Search",
        "🧬 Taste DNA",
        "📈 Drift Analysis",
        "⚙️ Setup",
        "🎵 Music Taste",
        "📚 Book Taste",
        "🌐 Cultural DNA",
        "🔮 Cross-Predict",
    ], label_visibility="collapsed")

    st.markdown("---")
    counts = get_store_counts()
    st.caption(f"🗄️ Embedded: **{counts['my_lens']}** movies")
    if counts["my_lens"] == 0:
        st.warning("Run the pipeline first!\n\n`python pipeline.py`")


# ── Load data ──────────────────────────────────────────────────────────────────
df = load_data()
rated_df = df[df["final_rating"].notna()].copy()
rated_df["final_rating"] = pd.to_numeric(rated_df["final_rating"], errors="coerce")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown(f"# 📊 Your Cinematic Universe")
    st.caption("A complete picture of your movie-watching history")
    st.markdown("---")

    # ── Top metrics ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Movies Watched", f"{len(df):,}")
    c2.metric("Rated", f"{len(rated_df):,}")
    c3.metric("Avg Rating", f"{rated_df['final_rating'].mean():.2f} ★")
    c4.metric("5★ Movies", f"{(rated_df['final_rating'] == 5).sum()}")
    c5.metric("With Reviews", f"{df['my_review'].str.len().gt(20).sum()}")

    st.markdown("")

    # ── Rating distribution ───────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Rating Distribution")
        rating_counts = rated_df["final_rating"].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            labels={"x": "Rating", "y": "Count"},
            color=rating_counts.values,
            color_continuous_scale=[[0, "#2a2a35"], [1, ACCENT]],
        )
        fig.update_traces(marker_line_width=0)
        fig.update_layout(coloraxis_showscale=False, showlegend=False)
        plotly_dark_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("### Movies Watched Over Time")
        df_time = df.copy()
        df_time["watch_year"] = pd.to_datetime(df_time["watch_date"], errors="coerce").dt.year
        yearly = df_time.dropna(subset=["watch_year"]).groupby("watch_year").size().reset_index(name="count")
        fig2 = px.area(yearly, x="watch_year", y="count",
                       labels={"watch_year": "Year", "count": "Films Watched"},
                       color_discrete_sequence=[ACCENT])
        fig2.update_traces(fill="tozeroy", fillcolor="rgba(233,168,76,0.15)", line_color=ACCENT)
        plotly_dark_layout(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Top directors & genres (from TMDB cache if available) ────────────────
    st.markdown("<br><h3>🏆 Your 5★ Films</h3>", unsafe_allow_html=True)
    top5 = rated_df[rated_df["final_rating"] == 5][["title", "year"]].head(12)

    # Show as a poster grid (properly row-chunked)
    from src.tmdb_client import get_movie_info as _gmi2
    
    for row_start in range(0, len(top5), 6):
        cols = st.columns(6)
        chunk = top5.iloc[row_start:row_start+6]
        for i, (_, row) in enumerate(chunk.iterrows()):
            yr  = int(row["year"]) if pd.notna(row["year"]) else None
            try:
                inf = _gmi2(row["title"], yr)
                purl = inf.get("poster_url", "")
            except Exception:
                purl = ""
            with cols[i]:
                poster_html = get_poster_html(purl, width=130, alt_text=row["title"])
                st.markdown(poster_html, unsafe_allow_html=True)
                st.markdown(f"<div class='grid-title'>{row['title']}</div>", unsafe_allow_html=True)
                st.markdown("<div class='stars' style='font-size:12px;'>★★★★★</div>", unsafe_allow_html=True)

    st.markdown("<br><hr><br><h3>💀 Your Lowest Rated</h3>", unsafe_allow_html=True)
    bot = rated_df[rated_df["final_rating"] <= 1.0][["title", "year", "final_rating"]].head(6)
    
    for row_start in range(0, len(bot), 6):
        cols = st.columns(6)
        chunk = bot.iloc[row_start:row_start+6]
        for i, (_, row) in enumerate(chunk.iterrows()):
            yr  = int(row["year"]) if pd.notna(row["year"]) else None
            try:
                inf = _gmi2(row["title"], yr)
                purl = inf.get("poster_url", "")
            except Exception:
                purl = ""
            with cols[i]:
                poster_html = get_poster_html(purl, width=130, alt_text=row["title"])
                st.markdown(poster_html, unsafe_allow_html=True)
                st.markdown(f"<div class='grid-title'>{row['title']}</div>", unsafe_allow_html=True)
                s = stars(row["final_rating"])
                st.markdown(f"<div class='stars' style='font-size:12px; color:#aa3333;'>{s}</div>", unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT RATING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Rating":
    st.markdown("# 🔮 Predict Your Rating")
    st.caption("Enter any movie — I'll predict what you'd give it based on your taste DNA")
    st.markdown("---")

    if counts["my_lens"] == 0:
        st.error("⚠️ Vector store is empty. Run `python pipeline.py` first.")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            movie_title = st.text_input("Movie Title", placeholder="e.g. Parasite")
        with col2:
            movie_year = st.number_input("Year (optional, 0 = any)", min_value=0, max_value=2026, value=0, step=1)

        if st.button("🔮 Predict"):
            if not movie_title.strip():
                st.warning("Enter a movie title.")
            else:
                with st.spinner("Analysing your taste..."):
                    from src.predictor import predict_rating
                    result = predict_rating(movie_title.strip(), movie_year if movie_year > 0 else None)

                pred = result["predicted_rating"]
                conf = result["confidence"]
                info = result["tmdb_info"]
                similar = result["top_similar"]
                space_preds = result["space_predictions"]

                if pred is None:
                    st.error("Couldn't predict — not enough similar movies in your history yet.")
                else:
                    # ── Hero row: poster + prediction ──────────────────────
                    poster_url = info.get("poster_url", "").strip()

                    hero_l, hero_r = st.columns([1, 3])
                    with hero_l:
                        p_html = get_poster_html(poster_url, width=210, alt_text=movie_title)
                        st.markdown(p_html, unsafe_allow_html=True)
                    with hero_r:
                        st.markdown(f"""
                        <div class='movie-card' style='border: 1px solid {ACCENT}; padding:24px; height:100%; display: flex; flex-direction: column; justify-content: center; box-shadow: 0 0 25px rgba(233,168,76,0.1); margin:0;'>
                            <div style='font-size:11px;color:#888;margin-bottom:6px;text-transform:uppercase;letter-spacing:1px;'>PREDICTED RATING</div>
                            <div class='movie-rating' style='font-size:3.5rem;line-height:1;margin-bottom:10px;'>{pred} <span style="font-size:1.5rem;color:#666;">/ 5</span></div>
                            <div class='stars' style='font-size:24px;margin-bottom:12px;'>{stars(pred)}</div>
                            <div style='color:#8e8e93;font-size:13px;'>Confidence: <b>{conf:.0%}</b></div>
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Space breakdown ────────────────────────────────────
                    st.markdown("#### Prediction by embedding space")
                    sc1, sc2, sc3 = st.columns(3)
                    for col, label, key in [
                        (sc1, "🧬 My Lens", "my_lens"),
                        (sc2, "🎬 Movie DNA", "movie_dna"),
                        (sc3, "👥 Community", "community"),
                    ]:
                        v = space_preds.get(key)
                        col.metric(label, f"{v:.2f} ★" if v else "—")

                    # ── TMDB info ──────────────────────────────────────────
                    if info["found"]:
                        st.markdown("#### Movie Info (TMDB)")
                        mi1, mi2 = st.columns(2)
                        mi1.markdown(f"**Director:** {info.get('director', '—')}")
                        mi1.markdown(f"**Genres:** {', '.join(info.get('genres', [])) or '—'}")
                        mi2.markdown(f"**Cast:** {', '.join(info.get('cast', [])[:4]) or '—'}")
                        mi2.markdown(f"**TMDB Rating:** {info.get('vote_average', '—')}/10")
                        if info.get("overview"):
                            with st.expander("Plot"):
                                st.write(info["overview"])

                    # ── Similar movies with posters ────────────────────────
                    if similar:
                        st.markdown("<br><h4>Most Similar From Your History</h4>", unsafe_allow_html=True)
                        for row_start in range(0, len(similar[:6]), 3):
                            sim_cols = st.columns(3)
                            chunk = similar[row_start:row_start+3]
                            for idx, h in enumerate(chunk):
                                yr    = f" ({h['year']})" if h.get('year') else ""
                                sim_pct = f"{h['similarity']:.0%}"
                                try:
                                    from src.tmdb_client import get_movie_info as _gmi
                                    h_info   = _gmi(h["title"], h.get("year"))
                                    h_poster = h_info.get("poster_url", "").strip()
                                # Fallback if tmdb fails
                                except Exception:
                                    h_poster = ""
                                with sim_cols[idx]:
                                    p_html = get_poster_html(h_poster, width=70, alt_text=h["title"])
                                    st.markdown(f'''
                                    <div class="movie-card" style="display:flex; gap:16px; align-items:center; min-height: 120px; height: 100%; margin:0;">
                                        <div style="flex-shrink: 0;">
                                            {p_html}
                                        </div>
                                        <div style="flex-grow: 1; min-width: 0;">
                                            <div class="movie-title" style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{h['title']}{yr}">{h['title']}{yr}</div>
                                            <div class="stars" style="font-size:14px; margin:6px 0;">{stars(h['rating'])}</div>
                                            <div class="movie-meta" style="margin-top:0;">Similarity: <b>{sim_pct}</b></div>
                                        </div>
                                    </div>
                                    ''', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: WRITING ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "✍️ Writing Analytics":
    st.markdown("# ✍️ The Depth of Your Data")
    st.caption("Analyzing your 870+ written reviews to understand your cinematic voice")
    st.markdown("---")

    from src.review_analyzer import get_review_analytics
    with st.spinner("Analyzing your writing DNA..."):
        analytics = get_review_analytics(df)

    if not analytics:
        st.warning("No reviews found in your data.")
    else:
        # Top metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Reviews", f"{analytics['total_reviews']:,}")
        c2.metric("Total Words Written", f"{analytics['total_words']:,}")
        c3.metric("Avg Words per Review", f"{analytics['avg_word_count']:.0f}")

        st.markdown("<br>", unsafe_allow_html=True)
        col_l, col_r = st.columns([2, 1])

        with col_l:
            st.markdown("### 📈 Review Length Over Time")
            trend = analytics["trend_by_year"]
            trend_df = pd.DataFrame(list(trend.items()), columns=["Year", "Avg Words"]).sort_values("Year")
            
            # Using plotly for the chart
            fig_trend = px.line(
                trend_df, x="Year", y="Avg Words", 
                markers=True, line_shape="spline"
            )
            fig_trend.update_traces(line_color=ACCENT, marker=dict(size=8))
            fig_trend = plotly_dark_layout(fig_trend)
            st.plotly_chart(fig_trend, use_container_width=True)

            # Passion metrics
            st.markdown("### ❤️ Passion vs. Hate")
            pm = analytics["passion_metrics"]
            st.markdown(f"When you **hate** a movie (1-2★), you write an average of **{pm['avg_words_low_rating']:.0f} words**.")
            st.markdown(f"When you **love** a movie (4-5★), you write an average of **{pm['avg_words_high_rating']:.0f} words**.")

        with col_r:
            st.markdown("### 🧬 Vocabulary DNA")
            st.caption("Your most used stylistic words")
            for word, count in analytics["vocab_dna"].items():
                st.markdown(
                    f"<div style='display:flex; justify-content:space-between; border-bottom:1px solid #222; padding:8px 0;'>"
                    f"<span style='color:#ccc; font-weight:500;'>{word}</span>"
                    f"<span style='color:{ACCENT};'>{count}</span></div>",
                    unsafe_allow_html=True
                )

        if analytics.get("longest_review"):
            st.markdown("<br>### 🏆 Your Magnum Opus (Longest Review)", unsafe_allow_html=True)
            lr = analytics["longest_review"]
            st.markdown(f"**{lr['title']}** — {lr['word_count']} words")
            st.markdown(f"<div style='color:{ACCENT}; font-size:18px; margin-bottom:12px;'>{stars(lr['final_rating'])}</div>", unsafe_allow_html=True)
            with st.expander("Read full review"):
                st.markdown(lr["my_review"].replace('\n', '<br>'), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: VIRTUAL PERSONA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Virtual Persona":
    st.markdown("# 🤖 Chat With Yourself")
    st.caption("An AI twin built from 3,507 film ratings and 870 written reviews. Powered by your personal MLX Llama-3.1 model running locally on Apple Silicon.")
    st.markdown("---")

    from src.persona_chat import (
        chat_with_persona, get_retrieved_context,
        CHROMA_REVIEW_TEXT_COLLECTION, _get_engine, get_engine_status
    )
    from src.persona_profile import build_profile, format_profile_card
    from src.vector_store import collection_count

    # ── Engine status badge ───────────────────────────────────────────────────
    status = get_engine_status()
    badge_bg = "#0a1a0a" if "MLX" in status["backend"] else "#111"
    badge_border = "#2a6e2a" if "MLX" in status["backend"] else "#333"
    badge_color = "#5dba5d" if "MLX" in status["backend"] else "#888"
    badge_icon = "⚡️" if "MLX" in status["backend"] else "💻"
    st.markdown(
        f"<div style='display:inline-block; background:{badge_bg}; border:1px solid {badge_border}; "
        f"border-radius:6px; padding:6px 14px; font-size:13px; color:{badge_color}; margin-bottom:16px;'>"
        f"{badge_icon} {status['backend']} &nbsp;·&nbsp; <code style='font-size:12px; color:#888;'>{status['model'].split('/')[-1]}</code></div>",
        unsafe_allow_html=True
    )

    if collection_count(CHROMA_REVIEW_TEXT_COLLECTION) == 0:
        st.warning("🟡 Your reviews need to be indexed first. Go to ⚙️ Setup and run the pipeline.")
    else:
        # ── Warm up MLX on first visit ────────────────────────────────────────
        if "mlx_warmup_done" not in st.session_state:
            with st.spinner("⚡️ Loading Llama-3.1-8B into Apple Silicon GPU... (~15s first time)"):
                _get_engine()
            st.session_state.mlx_warmup_done = True

        # ── Persona Profile Card (sidebar) ────────────────────────────────────
        with st.sidebar:
            st.markdown("### 🧠 Your Taste DNA")
            try:
                profile = build_profile(df)
                ns = profile.get("north_stars", [])[:5]
                tg = profile.get("top_genres", [])[:4]
                td = profile.get("top_directors", [])[:4]
                avg = profile.get("avg_rating", 0)
                five_ct = profile.get("five_star_count", 0)

                st.markdown(f"**Avg rating:** {avg}★ &nbsp;·&nbsp; **5★ films:** {five_ct}", unsafe_allow_html=True)
                if ns:
                    st.markdown(f"**North stars:**\n" + "\n".join(f"• {t}" for t in ns))
                if tg:
                    st.markdown("**Loved genres:** " + ", ".join(tg))
                if td:
                    st.markdown("**Top directors:** " + ", ".join(td))
                st.markdown("---")
                if st.checkbox("Show what the model reads", key="show_context"):
                    st.session_state["show_retrieved"] = True
                else:
                    st.session_state["show_retrieved"] = False
            except Exception as e:
                st.caption(f"Profile load error: {e}")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "last_retrieved" not in st.session_state:
            st.session_state.last_retrieved = []

        # ── Starter prompts ───────────────────────────────────────────────────
        if not st.session_state.chat_history:
            st.markdown("<div style='color:#666; font-size:13px; margin-bottom:12px;'>Try asking:</div>", unsafe_allow_html=True)
            starters = [
                "What makes a perfect film for me?",
                "How do I feel about Christopher Nolan?",
                "What do I think of Tamasha?",
                "Which films have genuinely changed how I think?",
                "What does my watch history say about who I am?",
                "Which directors keep disappointing me?",
            ]
            r1, r2 = st.columns(2)
            for i, s in enumerate(starters):
                col = r1 if i % 2 == 0 else r2
                if col.button(s, key=f"starter_{i}"):
                    st.session_state.chat_history.append({"role": "user", "content": s})
                    retrieved = get_retrieved_context(s)
                    st.session_state.last_retrieved = retrieved
                    with st.spinner("Thinking like you..."):
                        resp = chat_with_persona(s, [])
                    st.session_state.chat_history.append({"role": "assistant", "content": resp})
                    st.rerun()

        # ── Chat history ──────────────────────────────────────────────────────
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # ── Debug: what did the model read? ───────────────────────────────────
        if st.session_state.get("show_retrieved") and st.session_state.last_retrieved:
            with st.expander(f"📚 Reviews the model read ({len(st.session_state.last_retrieved)} retrieved)", expanded=False):
                for m in st.session_state.last_retrieved:
                    st.markdown(
                        f"**{m['title']}** — {m.get('rating')}★ "
                        f"{'🔁' if m.get('rewatch') else ''} "
                        f"*({m.get('source', '')})*"
                    )
                    st.caption(str(m.get("review", ""))[:200] + "...")
                    st.markdown("---")

        # ── Chat input ────────────────────────────────────────────────────────
        if prompt := st.chat_input("Ask your cinematic clone anything..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Retrieve context (also save for debug panel)
            retrieved = get_retrieved_context(prompt)
            st.session_state.last_retrieved = retrieved

            with st.chat_message("assistant"):
                with st.spinner("🧠 Running through Llama-3.1 on your Mac GPU..."):
                    response = chat_with_persona(prompt, st.session_state.chat_history[:-1])
                    st.markdown(response)

            st.session_state.chat_history.append({"role": "assistant", "content": response})

        # ── Clear button ──────────────────────────────────────────────────────
        if st.session_state.chat_history:
            if st.button("Clear chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.session_state.last_retrieved = []
                st.rerun()



# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REVIEW COPILOT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Review Copilot":
    st.markdown("# 🧠 Review Copilot")
    st.caption("Draft a review and get it critiqued against your best writing — by your own custom Llama-3.1 model.")
    st.markdown("---")

    from src.persona_chat import critique_review_draft, _get_engine, get_engine_status

    # Warm up engine
    if "mlx_warmup_done" not in st.session_state:
        with st.spinner("⚡️ Loading your personal Llama-3.1 model... (~15s first time)"):
            _get_engine()
        st.session_state.mlx_warmup_done = True

    status = get_engine_status()
    if "MLX" in status["backend"]:
        st.markdown(
            f"<div style='display:inline-block; background:#0a1a0a; border:1px solid #2a6e2a; "
            f"border-radius:6px; padding:6px 14px; font-size:13px; color:#5dba5d; margin-bottom:20px;'>"
            f"⚡️ {status['backend']} &nbsp;·&nbsp; <code style='font-size:12px; color:#888;'>{status['model'].split('/')[-1]}</code></div>",
            unsafe_allow_html=True
        )

    draft = st.text_area(
        "Write your review draft here...",
        height=220,
        placeholder="e.g. Dune 2 was visually stunning but I felt disconnected from the characters. The scale was epic but..."
    )

    if st.button("🔍 Critique My Draft", key="critique_btn"):
        if not draft.strip():
            st.warning("Please write a draft first.")
        else:
            with st.spinner("🧠 Analyzing against your best past reviews..."):
                critique = critique_review_draft(draft, df)

            st.markdown("### 📝 Coach's Feedback")
            # Render markdown in the critique (it uses **bold** formatting)
            st.markdown(
                f"<div style='background:#0d0d0d; padding:28px; border-radius:8px; border:1px solid #333; line-height:1.8;'>{critique.replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SEMANTIC SEARCH
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Semantic Search":
    st.markdown("# 🔍 Semantic Search")
    st.caption("Search your entire watch history with natural language")
    st.markdown("---")

    st.markdown("**Try these:**")
    examples = [
        "movies that left me feeling empty and devastated",
        "films with amazing plot twists I never saw coming",
        "slow burn films that reward patience",
        "Indian cinema that deserves more recognition",
        "movies I watched alone late at night",
        "films where the villain was better than the hero",
    ]
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        if cols[i % 3].button(ex, key=f"ex_{i}"):
            st.session_state["search_query"] = ex

    query = st.text_input("Or type your own query",
                          value=st.session_state.get("search_query", ""),
                          placeholder="e.g. films about loneliness and identity")

    n_results = st.slider("Results", 5, 20, 10)

    if st.button("🔍 Search") and query.strip():
        if counts["my_lens"] == 0:
            st.error("Run `python pipeline.py` first.")
        else:
            with st.spinner("Searching..."):
                from src.taste_profile import semantic_search
                results = semantic_search(query.strip(), n_results=n_results)

            if not results:
                st.info("No results found.")
            else:
                st.markdown(f"### Top {len(results)} results for: *{query}*")
                for r in results:
                    yr = f" ({r['year']})" if r.get("year") else ""
                    s = stars(r.get("rating"))
                    preview = r.get("review", "")[:200]
                    
                    # Fetch poster
                    try:
                        from src.tmdb_client import get_movie_info as _gmi
                        yr_int = int(r["year"]) if r.get("year") and str(r["year"]).isdigit() else None
                        info = _gmi(r["title"], yr_int)
                        purl = info.get("poster_url", "")
                    except Exception:
                        purl = ""
                        
                    p_html = get_poster_html(purl, width=60, alt_text=r["title"])
                    st.markdown(f'''
                    <div class="movie-card" style="display:flex; gap:16px; align-items:center; min-height: 100px;">
                        <div style="flex-shrink: 0;">
                            {p_html}
                        </div>
                        <div style="flex-grow: 1; min-width: 0;">
                            <div style="display:flex; justify-content:space-between; align-items: baseline; flex-wrap: wrap;">
                                <span class="movie-title">{r['title']}{yr}</span>
                                <span style="color:#E9A84C; font-weight:600; font-size:13px;">{r['similarity']:.0%} match</span>
                            </div>
                            <div class="stars" style="margin:4px 0;">{s}</div>
                            <div class="movie-meta" style="margin-top:4px;">{preview}{'...' if len(str(preview))==200 else ''}</div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TASTE DNA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧬 Taste DNA":
    st.markdown("# 🧬 Your Taste DNA")
    st.caption("Your entire watch history visualised in 2D embedding space")
    st.markdown("---")

    if counts["my_lens"] < 20:
        st.error("Need at least 20 movies embedded. Run `python pipeline.py`.")
    else:
        with st.spinner("Computing 2D layout (first run may take ~30s)..."):
            from src.taste_profile import get_embedding_2d, get_taste_clusters
            emb2d = get_embedding_2d(CHROMA_MY_LENS_COLLECTION)

        if emb2d.empty:
            st.error("Could not compute 2D projection.")
        else:
            emb2d["rating_num"] = pd.to_numeric(emb2d["rating"], errors="coerce")
            emb2d["rating_str"] = emb2d["rating_num"].apply(lambda x: f"{x:.1f}★" if pd.notna(x) else "unrated")

            fig = px.scatter(
                emb2d, x="x", y="y",
                color="rating_num",
                hover_data={"title": True, "rating_str": True, "x": False, "y": False, "rating_num": False},
                color_continuous_scale=[[0, "#2a0a0a"], [0.5, "#7a4a10"], [1, "#E9A84C"]],
                labels={"rating_num": "Your Rating"},
                title="Your Taste Map — Every Movie You've Watched",
            )
            fig.update_traces(marker=dict(size=6, opacity=0.8))
            plotly_dark_layout(fig)
            fig.update_layout(height=600, coloraxis_colorbar=dict(title="Rating", tickfont_color="#e8e8e8"))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("*Movies closer together feel similar to you. Color = your rating (amber = loved it).*")

            # Cluster breakdown
            st.markdown("### Taste Clusters")
            n_k = st.slider("Number of clusters", 4, 12, 7)
            with st.spinner("Clustering..."):
                clusters = get_taste_clusters(n_clusters=n_k)

            if "error" not in clusters:
                stats = clusters["cluster_stats"]
                for row_start in range(0, len(stats), 3):
                    cols = st.columns(3)
                    chunk = stats[row_start:row_start+3]
                    for i, cs in enumerate(chunk):
                        avg = cs["avg_rating"]
                        s = stars(avg)
                        titles = ", ".join(cs["top_titles"][:3]) or "—"
                        with cols[i]:
                            st.markdown(
                                f"<div class='movie-card' style='min-height: 140px; display: flex; flex-direction: column; justify-content: space-between; height: 100%; margin: 0;'>"
                                f"<div>"
                                f"<span class='movie-title' style='display:block; margin-bottom:4px;'>{cs['label']}</span>"
                                f"<span class='stars'>{s}</span>"
                                f"<span class='movie-meta' style='display:block; margin-top:8px;'>avg: <b>{avg or '—'}</b> ★ &nbsp;·&nbsp; <b>{cs['size']}</b> films</span>"
                                f"</div>"
                                f"<div class='movie-meta' style='font-size:11px; color:#666; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:8px;' title='{titles}'>{titles}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DRIFT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Drift Analysis":
    st.markdown("# 📈 How Your Taste Has Evolved")
    st.caption("Tracking the shift in your cinematic sensibility over time")
    st.markdown("---")

    from src.taste_profile import get_taste_evolution
    evo = get_taste_evolution(df)

    if evo.empty:
        st.info("Not enough dated watch history to show drift.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=evo["year_watched"], y=evo["avg_rating"],
            mode="lines+markers",
            name="Avg Rating",
            line=dict(color=ACCENT, width=2.5),
            marker=dict(size=8, color=ACCENT),
            hovertemplate="<b>%{x}</b><br>Avg: %{y:.2f}★<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=evo["year_watched"], y=evo["count"],
            name="Films Watched",
            marker_color="rgba(233,168,76,0.2)",
            yaxis="y2",
        ))
        fig.update_layout(
            yaxis=dict(title="Avg Rating", range=[0, 5.5]),
            yaxis2=dict(title="Films Watched", overlaying="y", side="right", showgrid=False),
            legend=dict(x=0.01, y=0.99),
            title="Your Ratings & Watching Volume Over Time",
        )
        plotly_dark_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Observations
        if len(evo) >= 3:
            early_avg = evo.nsmallest(2, "year_watched")["avg_rating"].mean()
            recent_avg = evo.nlargest(2, "year_watched")["avg_rating"].mean()
            diff = recent_avg - early_avg

            if diff > 0.3:
                obs = f"📈 Your ratings have gone **up** by ~{diff:.2f}★ — you're pickier or happier lately."
            elif diff < -0.3:
                obs = f"📉 Your ratings have gone **down** by ~{abs(diff):.2f}★ — your standards are rising."
            else:
                obs = "📊 Your rating patterns have stayed fairly **consistent** over time."
            st.info(obs)

        # Rating by decade of film
        st.markdown("### How You Rate Films by Decade")
        df_dec = rated_df.copy()
        df_dec["decade"] = (pd.to_numeric(df_dec["year"], errors="coerce") // 10 * 10).astype("Int64")
        decade_avg = df_dec.groupby("decade").agg(
            avg=("final_rating", "mean"), count=("final_rating", "count")
        ).reset_index().dropna()
        decade_avg = decade_avg[decade_avg["count"] >= 3]

        fig2 = px.bar(decade_avg, x="decade", y="avg",
                      color="avg", color_continuous_scale=[[0, "#2a2a35"], [1, ACCENT]],
                      labels={"decade": "Film Decade", "avg": "Avg Your Rating"},
                      text=decade_avg["avg"].round(2))
        fig2.update_traces(textposition="outside", textfont_color="#e8e8e8", marker_line_width=0)
        fig2.update_layout(coloraxis_showscale=False)
        plotly_dark_layout(fig2)
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SETUP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Setup":
    st.markdown("# ⚙️ Setup & Status")
    st.markdown("---")

    st.markdown("### Step 1 — Start Ollama")
    st.code("ollama serve", language="bash")

    st.markdown("### Step 2 — Pull the embedding model")
    st.code("ollama pull nomic-embed-text", language="bash")

    st.markdown("### Step 3 — Install dependencies")
    st.code("pip install -r requirements.txt", language="bash")

    st.markdown("### Step 4 — Run the pipeline")
    st.code("""
# Full run (3500 movies ≈ 45-90 min)
python pipeline.py

# Quick test (first 50 movies)
python pipeline.py --limit 50
    """, language="bash")

    st.markdown("### Step 5 — Launch this app")
    st.code("streamlit run app.py", language="bash")

    st.markdown("---")
    st.markdown("### Current Status")

    from src.embedder import check_ollama_ready
    from src.config import CHROMA_COMMUNITY_COLLECTION

    ok, msg = check_ollama_ready()
    if ok:
        st.success(f"✅ {msg}")
    else:
        st.error(f"❌ {msg}")

    c1, c2, c3 = st.columns(3)
    c1.metric("my_lens", collection_count(CHROMA_MY_LENS_COLLECTION))
    c2.metric("movie_dna", collection_count(CHROMA_MOVIE_DNA_COLLECTION))
    c3.metric("community", collection_count(CHROMA_COMMUNITY_COLLECTION))

    st.markdown("---")
    st.markdown("### Data Files Found")
    from src.config import DIARY_CSV, RATINGS_CSV, REVIEWS_CSV
    for label, path in [("diary.csv", DIARY_CSV), ("ratings.csv", RATINGS_CSV), ("reviews.csv", REVIEWS_CSV)]:
        if Path(path).exists():
            size = Path(path).stat().st_size // 1024
            st.success(f"✅ {label} ({size} KB)")
        else:
            st.error(f"❌ {label} not found at {path}")

    st.markdown("---")
    st.markdown("### Want better embeddings?")
    st.info("""
**Upgrade to mxbai-embed-large** (1024-dim, better quality):
```bash
ollama pull mxbai-embed-large
```
Then edit `src/config.py`:
```python
OLLAMA_EMBED_MODEL = "mxbai-embed-large"
```
Then re-run the pipeline with `--force` to re-embed.
    """)

elif page == '🎵 Music Taste':
    st.markdown('# 🎵 Your Music Taste')
    st.caption('Connect Last.fm to map your music taste into the same intelligence space as your films.')
    st.markdown('---')
    
    # Two-column input
    c1, c2 = st.columns(2)
    lastfm_user = c1.text_input('Last.fm Username', placeholder='your_username')
    lastfm_key  = c2.text_input('Last.fm API Key', placeholder='Get free key at last.fm/api', type='password')
    
    # Prefill from env
    if not lastfm_key:
        from src.config import LASTFM_API_KEY
        lastfm_key = LASTFM_API_KEY
    
    if st.button('🎵 Load Music Data', disabled=not (lastfm_user and lastfm_key)):
        with st.spinner(f'Fetching your Last.fm data for {lastfm_user}...'):
            try:
                from src.lastfm_ingest import load_lastfm_data
                music_df = load_lastfm_data(lastfm_user, lastfm_key)
                st.session_state['music_df'] = music_df
                st.success(f'Loaded {len(music_df)} artists!')
            except Exception as e:
                st.error(f'Failed: {e}')
    
    if 'music_df' in st.session_state:
        music_df = st.session_state['music_df']
        
        # Top artists
        st.markdown('### 🎤 Your Top Artists')
        top10 = music_df.head(10)
        cols = st.columns(5)
        for i, (_, row) in enumerate(top10.iterrows()):
            with cols[i % 5]:
                tags_preview = ', '.join(row.get('tags', [])[:2]) if row.get('tags') else ''
                st.markdown(
                    f"<div class='movie-card' style='text-align:center; min-height:80px;'>"
                    f"<div class='movie-title' style='font-size:13px;'>{row['name']}</div>"
                    f"<div class='movie-meta'>{tags_preview}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        
        # Tag cloud as bar chart
        st.markdown('### 🏷️ Your Music DNA (Most Common Tags)')
        from collections import Counter
        all_tags = []
        for tags in music_df['tags']:
            if isinstance(tags, list):
                all_tags.extend(tags[:3])
        tag_counts = Counter(all_tags).most_common(15)
        if tag_counts:
            import plotly.express as px
            tag_df = pd.DataFrame(tag_counts, columns=['tag', 'count'])
            fig = px.bar(tag_df, x='count', y='tag', orientation='h',
                        color='count', color_continuous_scale=[[0,'#1a1a2e'],[1,ACCENT]])
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False)
            plotly_dark_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        # Index button
        st.markdown('---')
        if st.button('🧠 Index Music for Cross-Domain Intelligence'):
            with st.spinner('Embedding your music taste...'):
                from src.cross_domain import index_music
                index_music(music_df)
            st.success('Music indexed! Go to 🌐 Cultural DNA to see cross-domain insights.')

elif page == '📚 Book Taste':
    st.markdown('# 📚 Your Book Taste')
    st.caption('Upload your Goodreads export to add books to your cultural intelligence.')
    st.markdown('---')
    
    st.info('**Get your Goodreads export:** goodreads.com → My Books → Import/Export → Export Library')
    
    uploaded = st.file_uploader('Upload goodreads_library_export.csv', type='csv')
    
    if uploaded:
        import tempfile, os as _os
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        
        try:
            from src.goodreads_ingest import load_goodreads_data
            books_df = load_goodreads_data(tmp_path)
            st.session_state['books_df'] = books_df
            _os.unlink(tmp_path)
        except Exception as e:
            st.error(f'Parse error: {e}')
            books_df = None
        
        if 'books_df' in st.session_state:
            books_df = st.session_state['books_df']
            rated = books_df[books_df['my_rating'] > 0]
            
            # Stats
            m1, m2, m3 = st.columns(3)
            m1.metric('Books loaded', len(books_df))
            m2.metric('Rated', len(rated))
            m3.metric('Avg rating', f"{rated['my_rating'].mean():.1f}★" if len(rated) > 0 else '—')
            
            # Top rated books
            st.markdown('### ⭐ Your Top Books')
            top_books = rated.nlargest(10, 'my_rating')
            for _, row in top_books.iterrows():
                rating_str = '★' * int(row['my_rating'])
                st.markdown(
                    f"<div class='movie-card'>"
                    f"<span class='movie-title'>{row['title']}</span>"
                    f"<span class='movie-meta'> by {row.get('author','?')}</span>"
                    f"<div class='stars'>{rating_str}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            # Rating distribution
            st.markdown('### ⭐ Rating Distribution')
            dist = rated['my_rating'].value_counts().sort_index()
            fig = px.bar(x=dist.index, y=dist.values, labels={'x':'Rating','y':'Books'},
                        color=dist.values, color_continuous_scale=[[0,'#1a1a2e'],[1,ACCENT]])
            fig.update_layout(coloraxis_showscale=False)
            plotly_dark_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('---')
            if st.button('🧠 Index Books for Cross-Domain Intelligence'):
                with st.spinner('Embedding your book taste...'):
                    from src.cross_domain import index_books
                    from src.goodreads_ingest import build_book_text
                    # Add text_for_embedding column
                    books_df['text_for_embedding'] = books_df.apply(
                        lambda r: build_book_text(r.to_dict()), axis=1
                    )
                    index_books(books_df)
                st.success('Books indexed! Go to 🌐 Cultural DNA.')

elif page == '🌐 Cultural DNA':
    st.markdown('# 🌐 Your Cultural DNA')
    st.caption('Films + Books + Music — all in one unified taste map.')
    st.markdown('---')
    
    from src.cross_domain import (
        cross_domain_recommend, get_cultural_coherence_score,
        search_cross_domain, CHROMA_CULTURAL_COLLECTION
    )
    from src.vector_store import collection_count
    
    cultural_count = collection_count(CHROMA_CULTURAL_COLLECTION)
    film_count = counts.get('my_lens', 0)
    
    if cultural_count < 10:
        st.warning('Index your music or books first using the 🎵 and 📚 pages.')
    else:
        st.metric('Items in Cultural Space', cultural_count)
        
        # Cross-domain recommendations
        st.markdown('### 🎵 → 🎬 Films Matching Your Music Taste')
        with st.spinner('Computing cross-domain matches...'):
            music_to_film = cross_domain_recommend('music', 'film', n=6)
        
        if music_to_film:
            cols = st.columns(3)
            for i, item in enumerate(music_to_film[:6]):
                with cols[i % 3]:
                    sim_pct = f"{item['similarity']:.0%}"
                    st.markdown(
                        f"<div class='movie-card'>"
                        f"<span class='movie-title'>{item.get('title', item.get('name','?'))}</span>"
                        f"<div class='movie-meta'>Match: <b style='color:{ACCENT}'>{sim_pct}</b></div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
        else:
            st.info('Index your music first.')
        
        st.markdown('### 📚 → 🎬 Films Matching Your Book Taste')
        with st.spinner('Computing cross-domain matches...'):
            book_to_film = cross_domain_recommend('book', 'film', n=6)
        
        if book_to_film:
            cols = st.columns(3)
            for i, item in enumerate(book_to_film[:6]):
                with cols[i % 3]:
                    sim_pct = f"{item['similarity']:.0%}"
                    st.markdown(
                        f"<div class='movie-card'>"
                        f"<span class='movie-title'>{item.get('title', item.get('name','?'))}</span>"
                        f"<div class='movie-meta'>Match: <b style='color:{ACCENT}'>{sim_pct}</b></div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
        else:
            st.info('Index your books first.')
        
        # Coherence score
        score = get_cultural_coherence_score()
        if score is not None:
            st.markdown('---')
            st.markdown('### 🧬 Cultural Coherence')
            st.markdown(f'How aligned is your film + music taste?')
            st.progress(float(score))
            pct = f'{score:.0%}'
            if score > 0.7:
                st.success(f'{pct} — Very coherent. Your music and film taste share deep thematic DNA.')
            elif score > 0.4:
                st.info(f'{pct} — Moderate alignment. You explore different moods across mediums.')
            else:
                st.warning(f'{pct} — Low alignment. You use music and film very differently.')
        
        # Cross-domain search
        st.markdown('---')
        st.markdown('### 🔍 Search Across All Domains')
        xq = st.text_input('Search films + books + music together', placeholder='melancholic and introspective...')
        if xq and st.button('Search Everything'):
            results = search_cross_domain(xq, n=12)
            if results:
                for r in results:
                    domain_icon = {'film':'🎬','book':'📚','music':'🎵'}.get(r.get('domain',''),'•')
                    st.markdown(
                        f"{domain_icon} **{r.get('title', r.get('name','?'))}** — {r.get('domain','')} — {r['similarity']:.0%} match"
                    )

elif page == '🔮 Cross-Predict':
    st.markdown('# 🔮 Cross-Domain Prediction')
    st.caption("Predict how much you'd love something — based on your taste across ALL domains.")
    st.markdown('---')
    
    from src.cross_domain import cross_domain_predict_rating
    
    title_input = st.text_input('Film, book, or album title', placeholder='e.g. Dune: Part Two')
    domain_input = st.selectbox('What type?', ['film', 'book', 'music'])
    year_input = st.text_input('Year (optional)', placeholder='2024')
    
    if st.button('🔮 Predict', disabled=not title_input.strip()):
        with st.spinner('Analyzing across all your cultural taste...'):
            result = cross_domain_predict_rating(title_input.strip(), domain_input, year_input)
        
        if result.get('predicted_rating'):
            pred = result['predicted_rating']
            conf = result.get('confidence', 0)
            
            st.markdown(
                f"<div class='movie-card' style='text-align:center; padding:32px;'>"
                f"<div style='font-size:48px; color:{ACCENT};'>{pred:.1f}★</div>"
                f"<div class='movie-meta'>Predicted rating · {conf:.0%} confidence</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            
            # Why
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown('**Similar films you loved:**')
                for f in result.get('top_films', [])[:3]:
                    st.markdown(f"• {f.get('title','?')} ({f.get('rating','?')}★)")
            with c2:
                st.markdown('**Similar books you loved:**')
                for b in result.get('top_books', [])[:3]:
                    st.markdown(f"• {b.get('title','?')}")
            with c3:
                st.markdown('**Similar music:**')
                for m in result.get('top_music', [])[:3]:
                    st.markdown(f"• {m.get('name','?')}")
        else:
            st.info('Not enough cross-domain data yet. Index your music and books first.')

# 🎬 Letterboxd Intelligence

A **fully local, private AI system** built on your Letterboxd watch history. No cloud. No OpenAI. Everything runs on your Mac.

Turn 3,500+ film ratings and 870+ written reviews into a working AI that knows your taste better than any algorithm — because it's built from your own words.

---

## What It Does

| Feature | Description |
|---|---|
| **📊 Dashboard** | Overview of your entire watch history — ratings distribution, top films, key metrics |
| **🔮 Predict Rating** | Enter any unwatched film → predicts what you'd give it based on K-NN across 3 embedding spaces |
| **✍️ Writing Analytics** | Deep analysis of your reviews: vocabulary DNA, passion metrics, length trends |
| **🤖 Virtual Persona** | Chat with an AI clone trained on all your reviews, running locally via Llama-3.1 |
| **🧠 Review Copilot** | Draft a review → get critiqued against your own best writing |
| **🔍 Semantic Search** | Natural language search over your entire watch history |
| **🧬 Taste DNA** | Every film you've seen visualised in 2D embedding space + KMeans taste clusters |
| **📈 Drift Analysis** | How your taste has evolved since you started tracking |

---

## Architecture

```
Your Letterboxd CSVs (ratings, reviews, diary)
        │
        ▼
   ingest.py → merge into single DataFrame
        │
        ▼
  tmdb_client.py → enrich with genres, director, cast, plot (SQLite cached)
        │
        ▼
  embedder.py → Ollama mxbai-embed-large → 1024-dim vectors
        │
        ▼
  ChromaDB (3 collections)
  ┌─────────────────────────────┐
  │  my_lens      ← your personal experience (rating + review)   │
  │  movie_dna    ← objective movie attributes                    │
  │  community    ← TMDB community reviews                        │
  └─────────────────────────────┘
        │
        ├──► predictor.py     → K-NN rating prediction
        ├──► taste_profile.py → clustering, UMAP, semantic search
        └──► persona_chat.py  → RAG + MLX Llama-3.1 generation
```

**ML concepts used:** Embeddings, Vector databases (ChromaDB), RAG, K-NN, KMeans clustering, UMAP/PCA dimensionality reduction, LLM inference (MLX).

---

## Tech Stack

- **Embeddings:** [Ollama](https://ollama.com) + `mxbai-embed-large` (1024-dim, runs locally)
- **Vector DB:** [ChromaDB](https://www.trychroma.com/) (persistent, local)
- **LLM:** [MLX](https://github.com/ml-explore/mlx) + `Llama-3.1-8B-Instruct` (Apple Silicon GPU)
- **UI:** [Streamlit](https://streamlit.io)
- **Metadata:** [TMDB API](https://www.themoviedb.org/documentation/api) (cached to SQLite)
- **Analytics:** scikit-learn, Plotly, pandas

---

## Setup

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3) for MLX
- [Ollama](https://ollama.com) installed
- Python 3.11+
- Your Letterboxd data export (Settings → Import & Export → Export Your Data)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/letterboxd-intelligence
cd letterboxd-intelligence
pip install -r requirements.txt
```

### 2. Place Your Letterboxd Data

Extract your Letterboxd export and update the path in `src/config.py`:

```python
LETTERBOXD_DIR = Path("/path/to/your/letterboxd/export")
```

The folder should contain: `diary.csv`, `ratings.csv`, `reviews.csv`, `watchlist.csv`

### 3. Pull the Embedding Model

```bash
ollama pull mxbai-embed-large
```

### 4. Run the Pipeline (one-time, ~45 min for 3500 films)

```bash
python pipeline.py

# Quick test with first 50 films:
python pipeline.py --limit 50
```

This builds all ChromaDB vector stores. Incremental — safe to interrupt and resume.

### 5. Launch

```bash
streamlit run app.py
```

Open `http://localhost:8501`

---

## Virtual Persona — How It Works (RAG)

The persona chat is a **RAG (Retrieval-Augmented Generation)** pipeline:

```
Your question
    ↓
Hybrid retrieval: lexical keyword search + vector semantic search
→ finds your 8 most relevant reviews
    ↓
Context assembly:
- Taste profile card (computed from all your ratings: avg ★, north stars, top genres)
- 15 of your longest reviews (style grounding)
- The 8 retrieved reviews
    ↓
MLX Llama-3.1-8B generates a response in your voice
Temperature: 0.65 (emotional, volatile — like you actually write)
```

The LLM runs entirely on your Apple Silicon GPU via MLX. ~2-5s per response.

---

## Configuration (`src/config.py`)

```python
LETTERBOXD_DIR     = Path("/path/to/letterboxd/export")
OLLAMA_EMBED_MODEL = "mxbai-embed-large"   # 1024-dim
TMDB_API_KEY       = "your_tmdb_key"       # free at themoviedb.org
WEIGHTS = {
    "my_lens":   0.50,   # your personal experience — most trusted
    "movie_dna": 0.30,   # objective movie attributes
    "community": 0.20,   # TMDB community signal
}
```

Get a free TMDB API key at [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api).

---

## Project Structure

```
letterboxd-intelligence/
│
├── pipeline.py         ← Run once to build vector stores
├── app.py              ← Streamlit UI (9 pages)
├── requirements.txt
│
└── src/
    ├── config.py           ← Paths, model names, weights
    ├── ingest.py           ← Merge Letterboxd CSVs
    ├── tmdb_client.py      ← TMDB API + SQLite cache
    ├── embedder.py         ← Ollama embedding wrapper
    ├── vector_store.py     ← ChromaDB wrapper
    ├── predictor.py        ← K-NN rating prediction
    ├── taste_profile.py    ← Clustering, UMAP, semantic search
    ├── review_analyzer.py  ← Text analytics on your reviews
    ├── persona_profile.py  ← Data-driven taste profile builder
    └── persona_chat.py     ← RAG pipeline + MLX generation
```

---

## Privacy

**Your personal data never leaves your machine.**
- Letterboxd CSVs stay local (gitignored)
- ChromaDB embeddings stay local (gitignored)  
- SQLite caches stay local (gitignored)
- The only external call is TMDB API for movie metadata (film titles only, cached after first fetch)
- LLM inference runs entirely on your GPU via MLX

---

## Credits

Built with: Ollama, ChromaDB, MLX, Streamlit, scikit-learn, Plotly, TMDB API.

Data source: Your own [Letterboxd](https://letterboxd.com) watch history.

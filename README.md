# 🎬 Movinator (formerly Letterboxd Intelligence)

Movinator is a data science-driven movie recommendation and taste analysis engine. It moves beyond generic "highly rated" film suggestions by analysing cross-domain signals—your music taste, your reading habits, and your past movie ratings—to predict what *you* specifically want to watch right now.

---

## What It Does

| Feature | Description |
|---|---|
| **🎬 Discover (Cross-Domain Moods)** | Connects to your **Spotify** to analyze the *valence* (happiness/darkness) and *energy* of your recent listening history. It translates this into a mood profile (e.g., Intense, Melancholic, Euphoric) and finds films that match your exact emotional state. No Spotify? Use the Quick Quiz. |
| **📊 My Taste (The Data Science)** | Upload your **Letterboxd** history to see what actually drives your ratings. Using **Ridge Regression** and **K-Fold Cross Validation**, it calculates your "Taste Signature"—how much of your rating variance is explained by objective signals (runtime, TMDB community rating, release year) vs. your own subjective bias. Identifies your true genre preferences and contrarian takes. |
| **🎯 Predict** | A predictive pipeline that looks at any unwatched film and calculates exactly what you would rate it (out of 5 stars), complete with a confidence interval. |
| **🤖 Virtual Persona** | A conversational AI agent that adopts your personality by training on your Letterboxd reviews. Ask your clone for recommendations or debate a film. |

---

## Architecture & Data Science

Movinator isn't a simple API wrapper. It uses real statistical models to understand users:

*   **Spotify Audio Features:** Maps Spotify's `valence`, `energy`, and `acousticness` metrics to film genres. (e.g., Low valence + High energy = Psychological Thrillers).
*   **Taste Signature (Ridge Regression):** Fits a linear model on your Letterboxd ratings vs TMDB metadata. The resulting $R^2$ score represents the "objective" portion of your taste. The residual error (1 - $R^2$) represents your unique personal perspective.
*   **Pearson Correlation:** Analyzes the linear relationship ($r$) between your ratings and variables like film runtime and community average.

---

## Tech Stack

- **Data Science:** `pandas`, `scikit-learn` (Ridge, KFold, StandardScaler), `scipy.stats`
- **Visualization:** `plotly` (interactive dark-mode charts)
- **UI:** `streamlit` (multi-tab interactive dashboard)
- **APIs:** `spotipy` (Spotify OAuth), `requests` (TMDB REST API)
- **Local Config:** Persists API keys locally to `~/.culturaliq/config.json`

---

## Setup & Installation

### 1. Clone & Install
```bash
git clone https://github.com/rushilstark/LetterBoxd-Intelligence.git
cd LetterBoxd-Intelligence
pip install -r requirements.txt
```

### 2. Launch the App
```bash
streamlit run app_v2.py
```
*Open `http://localhost:8501` in your browser.*

### 3. Configure API Keys (In-App)
You do **not** need to touch code to set up API keys. 
1. Open the app and navigate to the **⚙️ Settings** tab.
2. Enter your **TMDB API Key** (Get one free at [themoviedb.org](https://www.themoviedb.org/)).
3. *(Optional)* Enter your **Spotify Client ID / Secret** for music analysis.
4. Keys are saved securely to your local machine (`~/.culturaliq/config.json`).

### 4. Upload Your Data
In the **🎬 Discover** tab, upload your Letterboxd export (`ratings.csv` or `reviews.csv`) to instantly unlock the **My Taste** and **Predict** dashboards.

---

## Privacy
Your personal data stays on your machine.
- Uploaded CSVs are stored securely in Streamlit's session state and never sent to a third-party server.
- API keys are written strictly to your local home directory.

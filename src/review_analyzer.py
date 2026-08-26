import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def get_review_analytics(df: pd.DataFrame):
    """
    Takes the merged Letterboxd DataFrame (from load_letterboxd_data),
    filters for rows with reviews, and calculates analytics.
    """
    # Filter to only rows with non-empty reviews
    revs = df[df["my_review"].str.strip() != ""].copy()
    if len(revs) == 0:
        return None
    
    # Basic stats
    revs["word_count"] = revs["my_review"].str.split().str.len()
    
    # Evolution over time
    # Parse watch_date or fall back to Year
    if "watch_date" in revs.columns:
        revs["date_parsed"] = pd.to_datetime(revs["watch_date"], errors="coerce")
    else:
        revs["date_parsed"] = pd.NaT

    # Group by Year for length trend
    revs["year_watched"] = revs["date_parsed"].dt.year
    revs["year_watched"] = revs["year_watched"].fillna(revs["year"])
    
    # Cast to int to avoid json issues
    trend_raw = revs.groupby("year_watched")["word_count"].mean().round(1).to_dict()
    trend = {int(k): float(v) for k, v in trend_raw.items()}

    # Top words (Vocabulary DNA)
    # We want to find the user's stylistic crutches.
    # Exclude standard stop words, plus movie-specific common words
    custom_stops = list(ENGLISH_STOP_WORDS) + [
        "movie", "film", "just", "like", "really", "good", "great", "bad",
        "watch", "watching", "watched", "time", "one", "even", "much", "well",
        "make", "made", "story", "character", "characters", "people", "way",
        "think", "know", "see", "seen", "say", "little", "never", "always"
    ]
    
    vec = CountVectorizer(stop_words=custom_stops, max_features=50, ngram_range=(1, 2))
    try:
        counts = vec.fit_transform(revs["my_review"])
        words = vec.get_feature_names_out()
        sums = counts.sum(axis=0).A1
        vocab_dna = sorted(zip(words, sums), key=lambda x: x[1], reverse=True)[:20]
        vocab_dict = {w: int(c) for w, c in vocab_dna}
    except ValueError:
        vocab_dict = {}

    # Sentiment approx (crude: rating vs word count)
    # Do they write more when they hate or love a movie?
    # 1-2 stars vs 4-5 stars
    low_rated = revs[revs["final_rating"] <= 2.5]
    high_rated = revs[revs["final_rating"] >= 4.0]
    
    avg_len_low = low_rated["word_count"].mean() if len(low_rated) > 0 else 0
    avg_len_high = high_rated["word_count"].mean() if len(high_rated) > 0 else 0

    longest_rev_dict = None
    if len(revs) > 0:
        longest = revs.loc[revs["word_count"].idxmax()]
        longest_rev_dict = {
            "title": str(longest["title"]),
            "my_review": str(longest["my_review"]),
            "word_count": int(longest["word_count"]),
            "final_rating": float(longest["final_rating"]) if pd.notna(longest["final_rating"]) else None
        }

    return {
        "total_reviews": int(len(revs)),
        "total_words": int(revs["word_count"].sum()),
        "avg_word_count": float(revs["word_count"].mean()),
        "longest_review": longest_rev_dict,
        "trend_by_year": trend,
        "vocab_dna": vocab_dict,
        "passion_metrics": {
            "avg_words_low_rating": float(avg_len_low),
            "avg_words_high_rating": float(avg_len_high)
        }
    }

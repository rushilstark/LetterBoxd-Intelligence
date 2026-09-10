"""
IMDb ratings CSV parser. Users export from imdb.com/list/ratings.
"""
import pandas as pd

def load_imdb_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    
    total = len(df)
    
    # Safely get title_type
    tt_col = "title_type" if "title_type" in df.columns else None
    if tt_col:
        df = df[df[tt_col].str.lower() == "movie"].copy()
    
    out = pd.DataFrame()
    out["title"] = df.get("title", "")
    out["year"] = pd.to_numeric(df.get("year", ""), errors="coerce").astype("Int64")
    
    out["final_rating"] = pd.to_numeric(df.get("your_rating", 0), errors="coerce") / 2.0
    out["my_review"] = ""
    out["tags"] = df.get("genres", "")
    out["is_rewatch"] = False
    out["watch_date"] = pd.to_datetime(df.get("date_rated", ""), errors="coerce")
    out["uri"] = df.get("url", "")
    
    avg = out["final_rating"].mean() if not out.empty else 0
    print(f"[imdb] Loaded {total} movies | {len(out)} rated | avg: {avg:.1f}")
    
    return out


def merge_with_letterboxd(imdb_df: pd.DataFrame, letterboxd_df: pd.DataFrame) -> pd.DataFrame:
    l_df = letterboxd_df.copy()
    i_df = imdb_df.copy()
    
    l_df["_merge_key"] = l_df["title"].astype(str).str.lower() + "_" + l_df["year"].astype(str)
    i_df["_merge_key"] = i_df["title"].astype(str).str.lower() + "_" + i_df["year"].astype(str)
    
    i_df_only = i_df[~i_df["_merge_key"].isin(l_df["_merge_key"])]
    
    merged = pd.concat([l_df, i_df_only], ignore_index=True)
    merged = merged.drop(columns=["_merge_key"]).drop_duplicates(subset=["title", "year"])
    
    print(f"[merge] {len(l_df)} Letterboxd + {len(i_df_only)} IMDb = {len(merged)} unique films after merge")
    
    return merged

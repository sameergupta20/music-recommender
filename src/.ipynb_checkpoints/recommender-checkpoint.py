import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def recommend_songs(
    track_name: str, 
    df: pd.DataFrame, 
    scaled_df: pd.DataFrame, 
    top_n: int = 5, 
    same_genre: bool = False
) -> pd.DataFrame:
    """Finds top_n similar tracks within the same K-Means cluster (and optionally same genre)."""
    matches = df[df["track_name"].str.lower() == track_name.lower()]
    if matches.empty:
        raise ValueError(f"Track '{track_name}' not found in dataset.")
    
    track_idx = matches.index[0]
    track_cluster = df.loc[track_idx, "cluster"]
    
    # Filter candidate pool by cluster (and optionally by track_genre)
    if same_genre and "track_genre" in df.columns:
        target_genre = df.loc[track_idx, "track_genre"]
        condition = (df["cluster"] == track_cluster) & (df["track_genre"] == target_genre)
    else:
        condition = (df["cluster"] == track_cluster)
        
    cluster_indices = df[condition].index
    cluster_vectors = scaled_df.loc[cluster_indices]
    
    # Compute cosine similarity
    target_vector = scaled_df.loc[[track_idx]]
    similarities = cosine_similarity(target_vector, cluster_vectors)[0]
    
    # Rank results
    cols = ["track_name", "artists", "cluster"]
    if "track_genre" in df.columns:
        cols.append("track_genre")
        
    results = df.loc[cluster_indices, cols].copy()
    results["similarity_score"] = similarities
    results = results[results.index != track_idx]
    
    return results.sort_values(by="similarity_score", ascending=False).head(top_n)
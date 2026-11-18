from src.preprocessing import load_and_clean_data, scale_features
from src.clustering import train_kmeans
from src.recommender import recommend_songs

def main():
    print("Loading and scaling Spotify dataset...")
    df = load_and_clean_data("data/spotify_tracks.csv")
    scaled_df, _ = scale_features(df)

    print("Clustering tracks into 8 musical vibes...")
    clusters, _ = train_kmeans(scaled_df, n_clusters=8)
    df["cluster"] = clusters

    while True:
        song = input("\nEnter a song name (or 'q' to quit): ").strip()
        if song.lower() == "q":
            break
        try:
            recs = recommend_songs(song, df, scaled_df, top_n=5)
            print(f"\nTop 5 recommendations for '{song}':")
            print(recs.to_string(index=False))
        except ValueError as e:
            print(f"Notice: {e}")

if __name__ == "__main__":
    main()
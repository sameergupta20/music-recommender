# 🎵 Unsupervised Music Recommendation Engine

A modular, content-based music recommendation system built with Python and Scikit-Learn that groups **81,343 unique Spotify tracks** into distinct musical "vibe" clusters without using genre labels.

## 🧠 Core Concepts & Pipeline
1. **Standardization (`StandardScaler`)**: Normalized 9 audio features (`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`) to zero mean and unit variance so high-magnitude features like BPM do not skew distance calculations.
2. **Dimensionality Reduction (`PCA`)**: Projected 9-dimensional scaled audio vectors onto 2 principal components to visualize cluster separation.
3. **Clustering (`K-Means`)**: Segmented tracks into **8 distinct musical clusters**, validated via the **Elbow Method (Inertia)**.
4. **Similarity Ranking (`Cosine Similarity`)**: Filtered candidate songs to the target track's cluster and ranked recommendations by cosine similarity (`0.97–0.99` match scores).

## 📁 Project Structure
```text
music-recommender/
├── data/               # Raw Spotify tracks dataset (CSV)
├── notebooks/          # Exploratory data analysis & Elbow Method plots
├── src/                # Modular preprocessing, clustering & recommender logic
├── main.py             # CLI application entrypoint
└── requirements.txt    # Dependencies
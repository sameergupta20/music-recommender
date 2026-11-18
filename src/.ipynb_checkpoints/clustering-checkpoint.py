import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Tuple

def reduce_dimensions(scaled_df: pd.DataFrame, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
    """Compresses scaled audio features into fewer dimensions using PCA."""
    pca = PCA(n_components=n_components, random_state=42)
    reduced_features = pca.fit_transform(scaled_df)
    return reduced_features, pca

def train_kmeans(scaled_df: pd.DataFrame, n_clusters: int = 8) -> Tuple[np.ndarray, KMeans]:
    """Groups songs into distinct vibe clusters using K-Means."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_df)
    return clusters, kmeans
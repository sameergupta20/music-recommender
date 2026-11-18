import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

AUDIO_FEATURES: List[str] = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Loads dataset, drops missing values, and removes duplicate tracks."""
    df = pd.read_csv(filepath)
    df = df.dropna(subset=AUDIO_FEATURES + ["track_name", "artists"])
    df = df.drop_duplicates(subset=["track_name", "artists"]).reset_index(drop=True)
    return df

def scale_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scales numerical audio features to zero mean and unit variance."""
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df[AUDIO_FEATURES])
    scaled_df = pd.DataFrame(scaled_array, columns=AUDIO_FEATURES)
    return scaled_df, scaler
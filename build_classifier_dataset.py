import pandas as pd
import numpy as np
import os

WINDOW_FILE = "data/ACCIDENT/accident_window_analysis.csv"
PAIR_FILE = "data/ACCIDENT/pair_interaction_analysis.csv"
OUTPUT_FILE = "data/ACCIDENT/classifier_dataset.csv"

# Load existing analysis results
window = pd.read_csv(WINDOW_FILE)
pairs = pd.read_csv(PAIR_FILE)

# Aggregate pair-interaction information to ONE row per video
if len(pairs) > 0:
    pair_features = pairs.groupby("video").agg(
        pair_count=("video", "size"),
        pair_min_distance=("min_distance", "min"),
        pair_mean_distance=("min_distance", "mean"),
        pair_min_distance_frame=("min_distance_frame", "min"),
        pair_distance_at_accident=("distance_at_accident", "min"),
        pair_mean_observations=("observations", "mean")
    ).reset_index()
else:
    pair_features = pd.DataFrame(columns=[
        "video",
        "pair_count",
        "pair_min_distance",
        "pair_mean_distance",
        "pair_min_distance_frame",
        "pair_distance_at_accident",
        "pair_mean_observations"
    ])

# Merge with the 25 accident-window records
dataset = window.merge(
    pair_features,
    on="video",
    how="left"
)

# Videos without pair interactions get zero pair count.
dataset["pair_count"] = dataset["pair_count"].fillna(0)

# Numeric missing values are kept as NaN for now.
# We will handle them during classifier training/preprocessing.
numeric_columns = dataset.select_dtypes(include=[np.number]).columns

# Make sure the dataset is sorted consistently
dataset = dataset.sort_values("video").reset_index(drop=True)

# Save
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
dataset.to_csv(OUTPUT_FILE, index=False)

print("=" * 50)
print("CLASSIFIER DATASET CREATED")
print("=" * 50)
print("Rows:", len(dataset))
print("Columns:", len(dataset.columns))
print("Videos:", dataset["video"].nunique())

print("\nAccident classes:")
print(dataset["accident_type"].value_counts().to_string())

print("\nVideos without pair interactions:")
print((dataset["pair_count"] == 0).sum())

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nColumns:")
print(list(dataset.columns))

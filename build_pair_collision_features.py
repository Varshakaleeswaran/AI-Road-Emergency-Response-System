import pandas as pd
import numpy as np
import os

PAIR_FILE = "data/ACCIDENT/pair_interaction_analysis.csv"
OUT_FILE = "data/ACCIDENT/pair_collision_features.csv"

pairs = pd.read_csv(PAIR_FILE)

# Remove rows without a usable pair
pairs = pairs.dropna(
    subset=["vehicle_1", "vehicle_2"]
).copy()

# Basic pair features
pairs["min_distance"] = pd.to_numeric(
    pairs["min_distance"],
    errors="coerce"
)

pairs["distance_at_accident"] = pd.to_numeric(
    pairs["distance_at_accident"],
    errors="coerce"
)

pairs["pair_min_distance_frame"] = pd.to_numeric(
    pairs["min_distance_frame"],
    errors="coerce"
)

pairs["accident_frame"] = pd.to_numeric(
    pairs["accident_frame"],
    errors="coerce"
)

# Temporal distance from closest approach to accident
pairs["collision_timing"] = (
    pairs["pair_min_distance_frame"]
    - pairs["accident_frame"]
)

# Interaction duration proxy
pairs["interaction_observations"] = pd.to_numeric(
    pairs["observations"],
    errors="coerce"
)

# Group by video + vehicle pair
features = pairs.groupby(
    [
        "video",
        "accident_type",
        "vehicle_1",
        "vehicle_2"
    ]
).agg(
    min_distance=("min_distance", "min"),
    distance_at_accident=("distance_at_accident", "mean"),
    min_distance_frame=("pair_min_distance_frame", "mean"),
    accident_frame=("accident_frame", "first"),
    collision_timing=("collision_timing", "mean"),
    interaction_observations=("interaction_observations", "mean")
).reset_index()

# Absolute timing
features["abs_collision_timing"] = np.abs(
    features["collision_timing"]
)

# Sort by accident and interaction quality
features = features.sort_values(
    [
        "video",
        "min_distance"
    ]
)

features.to_csv(
    OUT_FILE,
    index=False
)

print("=" * 60)
print("PAIR COLLISION DATASET")
print("=" * 60)

print("Rows:", len(features))
print("Videos:", features["video"].nunique())

print("\nClasses:")
print(
    features["accident_type"]
    .value_counts()
    .to_string()
)

print("\nColumns:")
print(features.columns.tolist())

print("\nSaved:")
print(OUT_FILE)

print("=" * 60)

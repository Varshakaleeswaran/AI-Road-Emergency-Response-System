import pandas as pd
import numpy as np

INPUT = "data/ACCIDENT/accident_event_window_dataset.csv"
TEMPORAL = "data/ACCIDENT/temporal_collision_features.csv"
OUTPUT = "data/ACCIDENT/accident_event_fusion_dataset.csv"

df = pd.read_csv(INPUT)
temp = pd.read_csv(TEMPORAL)

# Keep only numeric temporal collision features.
temporal_features = [
    "pre_relative_speed",
    "collision_relative_speed",
    "post_relative_speed",
    "max_relative_speed",
    "speed_change_1",
    "speed_change_2",
    "relative_speed_change",
    "observations",
    "pre_distance",
    "collision_distance",
    "post_distance",
    "distance_reduction",
]

# Aggregate accident-centered pair features by video.
agg = (
    temp.groupby("video")[temporal_features]
    .agg(["mean", "max", "min"])
)

agg.columns = [
    f"temporal_{feature}_{stat}"
    for feature, stat in agg.columns
]

agg = agg.reset_index()

# Merge temporal information by video.
df = df.merge(agg, on="video", how="left")

# Add indicators showing whether temporal evidence exists.
temporal_cols = [
    c for c in df.columns
    if c.startswith("temporal_")
]

df["has_temporal_collision"] = (
    df[temporal_cols].notna().any(axis=1).astype(int)
)

df = df.replace([np.inf, -np.inf], np.nan)

df.to_csv(OUTPUT, index=False)

print("=" * 60)
print("ACCIDENT EVENT FEATURE FUSION DATASET")
print("=" * 60)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nEvent labels:")
print(df["event_label"].value_counts().to_string())

print("\nTemporal evidence:")
print(df["has_temporal_collision"].value_counts().to_string())

print("\nTemporal evidence by event label:")
print(
    pd.crosstab(
        df["event_label"],
        df["has_temporal_collision"]
    ).to_string()
)

print("\nSaved:")
print(OUTPUT)

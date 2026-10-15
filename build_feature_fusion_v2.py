import pandas as pd
import numpy as np
import os

BASE = "data/ACCIDENT"

BASE_FILE = os.path.join(
    BASE,
    "classifier_dataset.csv"
)

TEMPORAL_FILE = os.path.join(
    BASE,
    "temporal_collision_features.csv"
)

OUT_FILE = os.path.join(
    BASE,
    "feature_fusion_v2_dataset.csv"
)

# ------------------------------------------------------------
# Load base feature dataset
# ------------------------------------------------------------

base = pd.read_csv(BASE_FILE)

print("=" * 60)
print("FEATURE FUSION V2")
print("=" * 60)

print("Base videos:", len(base))

# ------------------------------------------------------------
# Load temporal collision features
# ------------------------------------------------------------

temporal = pd.read_csv(TEMPORAL_FILE)

print("Temporal rows:", len(temporal))

# ------------------------------------------------------------
# Aggregate temporal pair-level features to video level
# ------------------------------------------------------------

numeric_temporal = [
    "pre_relative_speed",
    "collision_relative_speed",
    "post_relative_speed",
    "max_relative_speed",
    "pre_speed_1",
    "pre_speed_2",
    "post_speed_1",
    "post_speed_2",
    "speed_change_1",
    "speed_change_2",
    "relative_speed_change",
    "observations",
    "pre_distance",
    "collision_distance",
    "post_distance",
    "distance_reduction",
    "pre_iou",
    "collision_iou",
    "post_iou"
]

# Keep only columns that actually exist
numeric_temporal = [
    c for c in numeric_temporal
    if c in temporal.columns
]

# Convert numeric columns safely
for col in numeric_temporal:
    temporal[col] = pd.to_numeric(
        temporal[col],
        errors="coerce"
    )

# Aggregate pair-level evidence
temporal_agg = (
    temporal
    .groupby("video")[numeric_temporal]
    .agg(["mean", "max", "min"])
)

# Flatten column names
temporal_agg.columns = [
    f"temporal_{col}_{stat}"
    for col, stat in temporal_agg.columns
]

temporal_agg = temporal_agg.reset_index()

# ------------------------------------------------------------
# Temporal evidence indicators
# ------------------------------------------------------------

temporal_info = (
    temporal
    .groupby("video")
    .agg(
        temporal_pair_count=(
            "vehicle_1",
            lambda x: x.notna().sum()
        ),
        temporal_observations=(
            "observations",
            "sum"
        )
    )
    .reset_index()
)

# Relative-motion availability
temporal_info["has_temporal_relative_motion"] = (
    temporal_info["temporal_observations"] > 0
).astype(int)

# Pair availability
temporal_info["has_temporal_pair"] = (
    temporal_info["temporal_pair_count"] > 0
).astype(int)

# ------------------------------------------------------------
# Merge temporal features
# ------------------------------------------------------------

fusion = base.merge(
    temporal_agg,
    on="video",
    how="left"
)

fusion = fusion.merge(
    temporal_info,
    on="video",
    how="left"
)

# ------------------------------------------------------------
# Missing-data indicators for existing feature families
# ------------------------------------------------------------

# Relative-motion evidence
if "relative_motion_observations" in fusion.columns:

    fusion["has_relative_motion"] = (
        pd.to_numeric(
            fusion["relative_motion_observations"],
            errors="coerce"
        )
        .fillna(0)
        > 0
    ).astype(int)

# Distance evidence
if "distance_observations" in fusion.columns:

    fusion["has_distance"] = (
        pd.to_numeric(
            fusion["distance_observations"],
            errors="coerce"
        )
        .fillna(0)
        > 0
    ).astype(int)

# Pair interaction evidence
if "pair_count" in fusion.columns:

    fusion["has_pair_interaction"] = (
        pd.to_numeric(
            fusion["pair_count"],
            errors="coerce"
        )
        .fillna(0)
        > 0
    ).astype(int)

# ------------------------------------------------------------
# Fill missing indicator columns
# ------------------------------------------------------------

indicator_columns = [
    "has_temporal_relative_motion",
    "has_temporal_pair",
    "has_relative_motion",
    "has_distance",
    "has_pair_interaction"
]

for col in indicator_columns:

    if col in fusion.columns:
        fusion[col] = fusion[col].fillna(0).astype(int)

# ------------------------------------------------------------
# Sort by video
# ------------------------------------------------------------

fusion = fusion.sort_values(
    "video"
).reset_index(drop=True)

# ------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------

print("\nFinal videos:", len(fusion))

print(
    "Unique videos:",
    fusion["video"].nunique()
)

print(
    "Duplicate videos:",
    fusion["video"].duplicated().sum()
)

print("\nClasses:")

print(
    fusion["accident_type"]
    .value_counts()
    .to_string()
)

print("\nTemporal coverage:")

if "has_temporal_relative_motion" in fusion.columns:
    print(
        fusion["has_temporal_relative_motion"]
        .value_counts()
        .rename(
            {
                0: "No relative-motion evidence",
                1: "Relative-motion available"
            }
        )
        .to_string()
    )

print("\nFeature count:")

print(
    "Total columns:",
    len(fusion.columns)
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

fusion.to_csv(
    OUT_FILE,
    index=False
)

print("\nSaved:")
print(OUT_FILE)

print("=" * 60)
print("FEATURE FUSION V2 COMPLETE")
print("=" * 60)
import pandas as pd

BASE = "data/ACCIDENT"

base = pd.read_csv(
    f"{BASE}/accident_event_window_dataset.csv"
)

temporal = pd.read_csv(
    f"{BASE}/window_temporal_collision_features_labeled.csv"
)

keys = [
    "video",
    "window_start",
    "window_end"
]

numeric = [
    "pre_relative_speed",
    "window_relative_speed",
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
    "window_distance",
    "post_distance",
    "distance_reduction"
]

available = [
    c for c in numeric
    if c in temporal.columns
]

agg = temporal.groupby(keys)[available].agg(
    ["mean", "max", "min"]
)

agg.columns = [
    f"temporal_{a}_{b}"
    for a, b in agg.columns
]

agg = agg.reset_index()

# Pair count in each window
pair_count = (
    temporal.groupby(keys)
    .size()
    .reset_index(name="temporal_pair_count")
)

# Keep exactly one base row per window
base_cols = [
    "video",
    "label",
    "event_label",
    "window_start",
    "window_end",
    "movement_mean",
    "movement_max",
    "distance_min",
    "distance_mean",
    "relative_speed_mean",
    "relative_speed_max",
    "unique_vehicle_ids"
]

base_cols = [
    c for c in base_cols
    if c in base.columns
]

base_unique = base[base_cols].drop_duplicates(keys)

fusion = base_unique.merge(
    agg,
    on=keys,
    how="left"
)

fusion = fusion.merge(
    pair_count,
    on=keys,
    how="left"
)

OUTPUT = (
    f"{BASE}/accident_event_window_fusion_dataset.csv"
)

fusion.to_csv(
    OUTPUT,
    index=False
)

print("=" * 60)
print("WINDOW FEATURE FUSION DATASET")
print("=" * 60)

print("Rows:", len(fusion))
print("Columns:", len(fusion.columns))

print("\nEvent labels:")

print(
    fusion["event_label"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nVideos:")

print(
    fusion["video"].nunique()
)

print("\nMissing values:")

print(
    fusion.isna()
    .sum()
    .sort_values(ascending=False)
    .head(15)
    .to_string()
)

print("\nSaved:")
print(OUTPUT)

print("=" * 60)

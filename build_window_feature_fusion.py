
import pandas as pd

BASE = "data/ACCIDENT"

BASE_INPUT = f"{BASE}/accident_event_window_dataset.csv"
TEMPORAL_INPUT = f"{BASE}/window_temporal_collision_features_labeled.csv"
OUTPUT = f"{BASE}/accident_event_window_fusion_dataset.csv"

# --------------------------------------------------
# 1. Load datasets
# --------------------------------------------------

base = pd.read_csv(BASE_INPUT)
temporal = pd.read_csv(TEMPORAL_INPUT)

keys = [
    "video",
    "window_start",
    "window_end"
]

required_base = keys + ["label", "event_label"]
required_temporal = keys + ["vehicle_1", "vehicle_2"]

missing_base = [c for c in required_base if c not in base.columns]
missing_temporal = [c for c in required_temporal if c not in temporal.columns]

if missing_base:
    raise ValueError(f"Missing columns in base dataset: {missing_base}")

if missing_temporal:
    raise ValueError(f"Missing columns in temporal dataset: {missing_temporal}")

# --------------------------------------------------
# 2. Define temporal numeric features
# --------------------------------------------------

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

if not available:
    raise ValueError("No expected temporal numeric features were found.")

# Ensure numeric feature columns are numeric
for col in available:
    temporal[col] = pd.to_numeric(temporal[col], errors="coerce")

# --------------------------------------------------
# 3. Aggregate temporal features per window
# --------------------------------------------------

agg = temporal.groupby(keys, dropna=False)[available].agg(
    ["mean", "max", "min"]
)

agg.columns = [
    f"temporal_{feature}_{stat}"
    for feature, stat in agg.columns
]

agg = agg.reset_index()

# --------------------------------------------------
# 4. Count distinct vehicle pairs per window
# --------------------------------------------------

pair_rows = temporal.dropna(
    subset=["vehicle_1", "vehicle_2"]
).copy()

# Treat (vehicle 1, vehicle 2) and (vehicle 2, vehicle 1)
# as the same pair.
pair_rows["_pair_low"] = pair_rows[
    ["vehicle_1", "vehicle_2"]
].min(axis=1)

pair_rows["_pair_high"] = pair_rows[
    ["vehicle_1", "vehicle_2"]
].max(axis=1)

pair_count = (
    pair_rows[
        keys + ["_pair_low", "_pair_high"]
    ]
    .drop_duplicates(
        subset=keys + ["_pair_low", "_pair_high"]
    )
    .groupby(keys, dropna=False)
    .size()
    .reset_index(name="temporal_pair_count")
)

# --------------------------------------------------
# 5. Keep one base row per window
# --------------------------------------------------

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

base_unique = base[base_cols].drop_duplicates(subset=keys).copy()

# Check that each window has only one event label
label_conflicts = (
    base.groupby(keys, dropna=False)["event_label"]
    .nunique(dropna=False)
)

if (label_conflicts > 1).any():
    raise ValueError(
        "Some windows have conflicting event labels in the base dataset."
    )

# --------------------------------------------------
# 6. Merge the window-level features
# --------------------------------------------------

fusion = base_unique.merge(
    agg,
    on=keys,
    how="left",
    validate="one_to_one"
)

fusion = fusion.merge(
    pair_count,
    on=keys,
    how="left",
    validate="one_to_one"
)

fusion["temporal_pair_count"] = (
    fusion["temporal_pair_count"]
    .fillna(0)
    .astype("int64")
)

# --------------------------------------------------
# 7. Validate the final dataset
# --------------------------------------------------

if fusion.duplicated(subset=keys).any():
    raise ValueError("Duplicate windows detected in fusion dataset.")

if fusion["event_label"].isna().any():
    raise ValueError("Missing event labels detected in fusion dataset.")

# --------------------------------------------------
# 8. Save
# --------------------------------------------------

fusion.to_csv(OUTPUT, index=False)

print("=" * 60)
print("WINDOW FEATURE FUSION DATASET")
print("=" * 60)

print("Rows:", len(fusion))
print("Columns:", len(fusion.columns))
print("Videos:", fusion["video"].nunique())
print("Duplicate windows:", fusion.duplicated(subset=keys).sum())
print("Missing event labels:", fusion["event_label"].isna().sum())

print("\nEvent labels:")
print(fusion["event_label"].value_counts(dropna=False).to_string())

print("\nDistinct vehicle-pair count by event label:")
print(
    fusion.groupby("event_label")["temporal_pair_count"]
    .agg(["count", "mean", "min", "max"])
    .to_string()
)

print("\nTop missing-value counts:")
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

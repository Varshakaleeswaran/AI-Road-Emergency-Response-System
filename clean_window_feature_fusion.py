import pandas as pd

BASE = "data/ACCIDENT"

INPUT = f"{BASE}/accident_event_window_fusion_dataset.csv"
OUTPUT = f"{BASE}/accident_event_window_fusion_dataset_clean.csv"

d = pd.read_csv(INPUT)

# Remove duplicate pair-count columns
if "temporal_pair_count_x" in d.columns:
    d["temporal_pair_count"] = d["temporal_pair_count_x"]

d = d.drop(
    columns=[
        "temporal_pair_count_x",
        "temporal_pair_count_y"
    ],
    errors="ignore"
)

d.to_csv(OUTPUT, index=False)

print("=" * 60)
print("CLEAN WINDOW FEATURE FUSION DATASET")
print("=" * 60)

print("Rows:", len(d))
print("Columns:", len(d.columns))

print("\nEvent labels:")
print(
    d["event_label"]
    .value_counts()
    .to_string()
)

print("\nPair count:")
print(
    d.groupby("event_label")["temporal_pair_count"]
    .agg(["count", "mean", "min", "max"])
    .round(2)
    .to_string()
)

print("\nSaved:")
print(OUTPUT)

print("=" * 60)

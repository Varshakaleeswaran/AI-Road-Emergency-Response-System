import pandas as pd

INPUT = r"data\ACCIDENT\accident_event_window_fusion_dataset_clean.csv"
OUTPUT = r"data\ACCIDENT\gru_sequence_dataset.csv"

df = pd.read_csv(INPUT)

FEATURES = [
    "movement_mean",
    "movement_max",
    "distance_min",
    "distance_mean",
    "relative_speed_mean",
    "relative_speed_max",
    "unique_vehicle_ids",
    "temporal_max_relative_speed_mean",
    "temporal_pre_relative_speed_mean",
    "temporal_max_relative_speed_max",
    "temporal_pre_speed_1_max",
    "temporal_window_relative_speed_max",
    "temporal_distance_reduction_mean",
]

required = ["video", "window_start", "window_end", "event_label"] + FEATURES
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df[required].copy()

df["event_label"] = df["event_label"].map({
    "non_event": 0,
    "accident_event": 1
})

df = df.dropna(subset=["event_label"])
df["event_label"] = df["event_label"].astype(int)

for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values(["video", "window_start"]).reset_index(drop=True)

df.to_csv(OUTPUT, index=False)

print("=" * 60)
print("GRU SEQUENCE DATASET")
print("=" * 60)
print("Rows:", len(df))
print("Videos:", df["video"].nunique())
print("Features:", len(FEATURES))
print()
print("Labels:")
print(df["event_label"].value_counts().sort_index())
print()
print("Saved:")
print(OUTPUT)

import pandas as pd
import numpy as np

INPUT = r"data\ACCIDENT\gru_sequence_dataset.csv"
OUTPUT = r"data\ACCIDENT\gru_sequences.npz"

SEQUENCE_LENGTH = 3

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

df = pd.read_csv(INPUT)
df = df.sort_values(["video", "window_start"]).reset_index(drop=True)

X = []
y = []
groups = []

for video, group in df.groupby("video"):
    group = group.sort_values("window_start").reset_index(drop=True)

    for i in range(len(group) - SEQUENCE_LENGTH + 1):
        window = group.iloc[i:i + SEQUENCE_LENGTH]

        features = window[FEATURES].values.astype(np.float32)

        if np.isnan(features).all():
            continue

        label = int(window["event_label"].max())

        X.append(features)
        y.append(label)
        groups.append(video)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)
groups = np.array(groups)

np.savez(
    OUTPUT,
    X=X,
    y=y,
    groups=groups
)

print("=" * 60)
print("GRU SEQUENCES")
print("=" * 60)
print("Sequence length:", SEQUENCE_LENGTH)
print("Sequences:", len(X))
print("Shape:", X.shape)
print("Labels:")
print(pd.Series(y).value_counts().sort_index())
print("Videos:", len(np.unique(groups)))
print("Saved:", OUTPUT)

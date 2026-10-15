import pandas as pd

BASE = "data/ACCIDENT"

windows = pd.read_csv(
    f"{BASE}/accident_event_window_dataset.csv"
)

temporal = pd.read_csv(
    f"{BASE}/window_temporal_collision_features.csv"
)

# Build lookup from the original event-window dataset
lookup = windows[
    ["video", "window_start", "window_end", "label", "event_label"]
].drop_duplicates(
    ["video", "window_start", "window_end"]
)

# Remove the empty event_label column
temporal = temporal.drop(
    columns=["event_label"],
    errors="ignore"
)

# Attach window-level event label
temporal = temporal.merge(
    lookup[
        ["video", "window_start", "window_end", "event_label"]
    ],
    on=["video", "window_start", "window_end"],
    how="left"
)

OUTPUT = f"{BASE}/window_temporal_collision_features_labeled.csv"

temporal.to_csv(
    OUTPUT,
    index=False
)

print("=" * 60)
print("LABELED WINDOW TEMPORAL DATASET")
print("=" * 60)

print("Rows:", len(temporal))

print(
    "Event labels:"
)

print(
    temporal["event_label"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nWindows:")

print(
    temporal[
        ["video", "window_start", "window_end", "event_label"]
    ].drop_duplicates()["event_label"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nSaved:")
print(OUTPUT)

print("=" * 60)

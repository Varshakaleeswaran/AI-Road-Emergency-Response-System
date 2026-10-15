import pandas as pd
from pathlib import Path

METADATA = Path("data/ACCIDENT/metadata-real.csv")
STARTER = Path("data/ACCIDENT/starter_25.csv")
OUTPUT = Path("data/ACCIDENT/starter_25_ground_truth.csv")

metadata = pd.read_csv(METADATA)
starter = pd.read_csv(STARTER)

print("Metadata rows:", len(metadata))
print("Starter videos:", len(starter))

# Keep only the columns we need from the official metadata
columns = [
    "path",
    "type",
    "rollover",
    "accident_time",
    "accident_frame",
    "center_x",
    "center_y",
    "x1",
    "y1",
    "x2",
    "y2",
    "region",
    "scene_layout",
    "weather",
    "day_time",
    "quality",
    "no_frames",
    "duration"
]

metadata_small = metadata[columns].copy()

# Match the 25 starter videos using their relative path
result = starter.merge(
    metadata_small,
    on="path",
    how="left",
    suffixes=("_starter", "_metadata")
)

# Check whether every starter video matched
missing = result["accident_frame_metadata"].isna().sum()

print("Matched ground-truth records:", len(result))
print("Missing ground-truth records:", missing)

result.to_csv(OUTPUT, index=False)

print(f"\nSaved: {OUTPUT}")
print("\nGround-truth summary:")
print(
    result[
        [
    "path",
    "type_metadata",
    "accident_time_metadata",
    "accident_frame_metadata",
    "center_x_metadata",
    "center_y_metadata"
]
    ].to_string(index=False)
)
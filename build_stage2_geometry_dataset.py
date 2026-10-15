import pandas as pd
import numpy as np
import glob
import os

DATA_FILE = "data/ACCIDENT/classifier_dataset.csv"
OUT_FILE = "data/ACCIDENT/stage2_geometry_dataset.csv"

gt = pd.read_csv(DATA_FILE)[
    ["video", "accident_type", "accident_frame"]
]

frames = []

for f in glob.glob("data/ACCIDENT/tracking/*_relative_motion.csv"):
    try:
        d = pd.read_csv(f)

        if d.empty:
            continue

        d["video"] = os.path.basename(f).replace(
            "_relative_motion.csv", ""
        )

        frames.append(d)

    except pd.errors.EmptyDataError:
        continue

df = pd.concat(frames, ignore_index=True)

df = df.merge(
    gt,
    on="video",
    how="inner"
)

# Keep only multi-vehicle accidents
df = df[df["accident_type"] != "single"].copy()

# Accident window
df = df[
    (df["frame"] >= df["accident_frame"] - 10) &
    (df["frame"] <= df["accident_frame"] + 10)
].copy()

# Vehicle speeds
df["speed_1"] = np.sqrt(
    df["vx_1"] ** 2 + df["vy_1"] ** 2
)

df["speed_2"] = np.sqrt(
    df["vx_2"] ** 2 + df["vy_2"] ** 2
)

# Direction angles
df["angle_1"] = np.degrees(
    np.arctan2(df["vy_1"], df["vx_1"])
)

df["angle_2"] = np.degrees(
    np.arctan2(df["vy_2"], df["vx_2"])
)

df["rel_angle"] = np.degrees(
    np.arctan2(
        df["relative_vy"],
        df["relative_vx"]
    )
)

# Absolute direction difference
angle_diff = np.abs(
    df["angle_1"] - df["angle_2"]
)

# Normalize to 0-180 degrees
df["direction_difference"] = np.minimum(
    angle_diff,
    360 - angle_diff
)

# Speed difference
df["speed_difference"] = np.abs(
    df["speed_1"] - df["speed_2"]
)

# Speed ratio
df["speed_ratio"] = (
    df["speed_1"] /
    (df["speed_2"] + 1e-6)
)

# Aggregate each video
features = df.groupby(
    ["video", "accident_type"]
).agg(
    mean_speed_1=("speed_1", "mean"),
    mean_speed_2=("speed_2", "mean"),
    max_speed_1=("speed_1", "max"),
    max_speed_2=("speed_2", "max"),
    mean_relative_speed=("relative_speed", "mean"),
    max_relative_speed=("relative_speed", "max"),
    mean_angle_1=("angle_1", "mean"),
    mean_angle_2=("angle_2", "mean"),
    mean_direction_difference=("direction_difference", "mean"),
    max_direction_difference=("direction_difference", "max"),
    mean_relative_angle=("rel_angle", "mean"),
    mean_speed_difference=("speed_difference", "mean"),
    mean_speed_ratio=("speed_ratio", "mean")
).reset_index()

features.to_csv(
    OUT_FILE,
    index=False
)

print("=" * 60)
print("STAGE-2 GEOMETRY DATASET CREATED")
print("=" * 60)
print("Videos:", len(features))
print("\nClasses:")
print(features["accident_type"].value_counts().to_string())

print("\nFeatures:")
print(features.columns.tolist())

print("\nSaved:")
print(OUT_FILE)
print("=" * 60)

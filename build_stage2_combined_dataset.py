import pandas as pd
import numpy as np
import glob
import os

base = pd.read_csv(
    "data/ACCIDENT/classifier_dataset.csv"
)

base = base[base["accident_type"] != "single"].copy()

frames = []

for f in glob.glob(
    "data/ACCIDENT/tracking/*_relative_motion.csv"
):
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

# Merge only by video
df = df.merge(
    base[["video", "accident_type", "accident_frame"]],
    on="video",
    how="inner"
)

df = df[
    (df["frame"] >= df["accident_frame"] - 10) &
    (df["frame"] <= df["accident_frame"] + 10)
].copy()

df["speed_1"] = np.sqrt(
    df["vx_1"] ** 2 + df["vy_1"] ** 2
)

df["speed_2"] = np.sqrt(
    df["vx_2"] ** 2 + df["vy_2"] ** 2
)

df["angle_1"] = np.degrees(
    np.arctan2(df["vy_1"], df["vx_1"])
)

df["angle_2"] = np.degrees(
    np.arctan2(df["vy_2"], df["vx_2"])
)

angle_diff = np.abs(
    df["angle_1"] - df["angle_2"]
)

df["direction_difference"] = np.minimum(
    angle_diff,
    360 - angle_diff
)

geometry = df.groupby(
    ["video", "accident_type"]
).agg(
    mean_speed_1=("speed_1", "mean"),
    mean_speed_2=("speed_2", "mean"),
    mean_relative_speed_geom=("relative_speed", "mean"),
    max_relative_speed_geom=("relative_speed", "max"),
    mean_direction_difference=("direction_difference", "mean"),
    max_direction_difference=("direction_difference", "max")
).reset_index()

combined = base.merge(
    geometry,
    on=["video", "accident_type"],
    how="left"
)

combined.to_csv(
    "data/ACCIDENT/stage2_combined_dataset.csv",
    index=False
)

print("=" * 60)
print("COMBINED STAGE-2 DATASET")
print("=" * 60)

print("Videos:", len(combined))

print("\nClasses:")
print(
    combined["accident_type"]
    .value_counts()
    .to_string()
)

print("\nGeometry coverage:")
print(
    combined["mean_speed_1"]
    .notna()
    .sum(),
    "/",
    len(combined)
)

print("\nSaved:")
print("data/ACCIDENT/stage2_combined_dataset.csv")

print("=" * 60)

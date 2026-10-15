import pandas as pd
import numpy as np
import glob
import os

gt = pd.read_csv("data/ACCIDENT/classifier_dataset.csv")

gt = gt[[
    "video",
    "accident_type",
    "accident_frame"
]].copy()

frames = []
skipped = 0

files = glob.glob("data/ACCIDENT/tracking/*_relative_motion.csv")

for f in files:
    try:
        d = pd.read_csv(f)

        if d.empty:
            skipped += 1
            continue

        d["video"] = os.path.basename(f).replace(
            "_relative_motion.csv", ""
        )

        frames.append(d)

    except pd.errors.EmptyDataError:
        skipped += 1

df = pd.concat(frames, ignore_index=True)

df = df.merge(
    gt,
    on="video",
    how="inner"
)

df["rel_angle"] = np.degrees(
    np.arctan2(df["relative_vy"], df["relative_vx"])
)

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

w = df[
    (df["frame"] >= df["accident_frame"] - 10) &
    (df["frame"] <= df["accident_frame"] + 10)
].copy()

print("Relative-motion files:", len(files))
print("Skipped empty files:", skipped)
print("Rows in accident windows:", len(w))

print("\nMean directional features by class:")

print(
    w.groupby("accident_type")[
        [
            "rel_angle",
            "speed_1",
            "speed_2",
            "angle_1",
            "angle_2",
            "relative_speed"
        ]
    ].mean().round(3).to_string()
)

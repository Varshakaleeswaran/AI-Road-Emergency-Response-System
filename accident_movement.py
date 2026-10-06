import pandas as pd
from pathlib import Path
import numpy as np

INPUT = "data/ACCIDENT/tracking/unS0-TLF1ao_00_tracking.csv"

OUTPUT = "data/ACCIDENT/tracking/unS0-TLF1ao_00_movement.csv"

df = pd.read_csv(INPUT)

# Sort by vehicle and frame
df = df.sort_values(["vehicle_id", "frame"]).copy()

# Previous position of the same vehicle
df["prev_x"] = df.groupby("vehicle_id")["center_x"].shift(1)
df["prev_y"] = df.groupby("vehicle_id")["center_y"].shift(1)
df["prev_frame"] = df.groupby("vehicle_id")["frame"].shift(1)

# Frame gap
df["frame_gap"] = df["frame"] - df["prev_frame"]

# Movement in X and Y
df["dx"] = df["center_x"] - df["prev_x"]
df["dy"] = df["center_y"] - df["prev_y"]

# Distance moved between observations
df["movement"] = np.sqrt(df["dx"]**2 + df["dy"]**2)

# If there is a large frame gap, movement is not directly comparable
df.loc[df["frame_gap"] != 1, ["dx", "dy", "movement"]] = np.nan

# Save
df.to_csv(OUTPUT, index=False)

print("\nMovement analysis completed!")
print("Rows:", len(df))
print("Output:", OUTPUT)

print("\nMovement around accident frame:")
print(
    df[(df["frame"] >= 55) & (df["frame"] <= 70)]
    [["frame", "vehicle_id", "center_x", "center_y", "dx", "dy", "movement", "frame_gap"]]
    .to_string(index=False)
)

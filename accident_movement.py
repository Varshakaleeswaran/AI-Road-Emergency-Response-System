
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument(
    "--max-displacement",
    type=float,
    default=40.0,
    help="Flag movement above this many pixels per frame; use 0 to disable.",
)
args = parser.parse_args()

input_path = Path(args.input)
output_path = input_path.with_name(
    input_path.stem.replace("_tracking", "_movement") + ".csv"
)

df = pd.read_csv(input_path)
required = {"frame", "vehicle_id", "center_x", "center_y"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df = df.sort_values(["vehicle_id", "frame"]).copy()
group = df.groupby("vehicle_id", sort=False)

df["prev_x"] = group["center_x"].shift(1)
df["prev_y"] = group["center_y"].shift(1)
df["prev_frame"] = group["frame"].shift(1)
df["frame_gap"] = df["frame"] - df["prev_frame"]

df["dx_raw"] = df["center_x"] - df["prev_x"]
df["dy_raw"] = df["center_y"] - df["prev_y"]
df["movement_raw"] = np.hypot(df["dx_raw"], df["dy_raw"])

consecutive = df["frame_gap"].eq(1)
df["suspicious_movement"] = (
    consecutive
    & (df["movement_raw"] > args.max_displacement)
    if args.max_displacement > 0
    else False
)

valid = consecutive & ~df["suspicious_movement"]
df["dx"] = df["dx_raw"].where(valid)
df["dy"] = df["dy_raw"].where(valid)
df["movement"] = df["movement_raw"].where(valid)

df.to_csv(output_path, index=False)

print("Movement calculation complete.")
print("Input rows:", len(df))
print("Consecutive-frame rows:", int(consecutive.sum()))
print("Flagged large movements:", int(df["suspicious_movement"].sum()))
print("Trusted movement rows:", int(df["movement"].notna().sum()))
print("Saved:", output_path)
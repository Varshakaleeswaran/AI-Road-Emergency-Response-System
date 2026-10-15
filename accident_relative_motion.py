
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
    help="Exclude vehicle movements above this pixels-per-frame threshold; use 0 to disable.",
)
args = parser.parse_args()

if args.max_displacement < 0:
    parser.error("--max-displacement must be >= 0")

input_path = Path(args.input)
output_path = input_path.with_name(
    input_path.stem.replace("_tracking", "_relative_motion") + ".csv"
)

df = pd.read_csv(input_path)
required = {"frame", "vehicle_id", "center_x", "center_y"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df = df[list(required)].copy()
df = df.sort_values(["vehicle_id", "frame"])

group = df.groupby("vehicle_id", sort=False)
df["prev_x"] = group["center_x"].shift(1)
df["prev_y"] = group["center_y"].shift(1)
df["prev_frame"] = group["frame"].shift(1)
df["frame_gap"] = df["frame"] - df["prev_frame"]

df["vx"] = df["center_x"] - df["prev_x"]
df["vy"] = df["center_y"] - df["prev_y"]
df["movement"] = np.hypot(df["vx"], df["vy"])

valid = df["frame_gap"].eq(1)
if args.max_displacement > 0:
    valid &= df["movement"].le(args.max_displacement)

df.loc[~valid, ["vx", "vy"]] = np.nan

rows = []
for frame, frame_data in df.groupby("frame", sort=True):
    vehicles = frame_data.to_dict("records")

    for i in range(len(vehicles)):
        a = vehicles[i]
        if pd.isna(a["vx"]) or pd.isna(a["vy"]):
            continue

        for j in range(i + 1, len(vehicles)):
            b = vehicles[j]
            if pd.isna(b["vx"]) or pd.isna(b["vy"]):
                continue

            rvx = a["vx"] - b["vx"]
            rvy = a["vy"] - b["vy"]

            rows.append({
                "frame": int(frame),
                "vehicle_1": int(a["vehicle_id"]),
                "vehicle_2": int(b["vehicle_id"]),
                "relative_vx": rvx,
                "relative_vy": rvy,
                "relative_speed": float(np.hypot(rvx, rvy)),
            })

columns = [
    "frame", "vehicle_1", "vehicle_2",
    "relative_vx", "relative_vy", "relative_speed",
]
result = pd.DataFrame(rows, columns=columns)

output_path.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(output_path, index=False)

print("Relative-motion analysis completed.")
print("Input tracking rows:", len(df))
print("Consecutive-frame observations:", int(df["frame_gap"].eq(1).sum()))
print("Excluded large movements:", int(
    (df["frame_gap"].eq(1) & (df["movement"] > args.max_displacement)).sum()
) if args.max_displacement > 0 else 0)
print("Relative-motion rows:", len(result))
print("Output:", output_path)
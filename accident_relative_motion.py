import pandas as pd
import numpy as np

INPUT = "data/ACCIDENT/tracking/unS0-TLF1ao_00_tracking.csv"
OUTPUT = "data/ACCIDENT/tracking/unS0-TLF1ao_00_relative_motion.csv"

df = pd.read_csv(INPUT)

# Keep only required columns
df = df[
    ["frame", "vehicle_id", "center_x", "center_y"]
].copy()

# Calculate each vehicle's movement
df = df.sort_values(["vehicle_id", "frame"])

df["prev_x"] = df.groupby("vehicle_id")["center_x"].shift(1)
df["prev_y"] = df.groupby("vehicle_id")["center_y"].shift(1)
df["prev_frame"] = df.groupby("vehicle_id")["frame"].shift(1)

df["frame_gap"] = df["frame"] - df["prev_frame"]

df["vx"] = df["center_x"] - df["prev_x"]
df["vy"] = df["center_y"] - df["prev_y"]

# Only use continuous frames
df.loc[df["frame_gap"] != 1, ["vx", "vy"]] = np.nan

rows = []

# Compare vehicles appearing in the same frame
for frame, frame_data in df.groupby("frame"):

    vehicles = frame_data.to_dict("records")

    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):

            a = vehicles[i]
            b = vehicles[j]

            # Need movement information for both vehicles
            if pd.isna(a["vx"]) or pd.isna(a["vy"]):
                continue

            if pd.isna(b["vx"]) or pd.isna(b["vy"]):
                continue

            relative_vx = a["vx"] - b["vx"]
            relative_vy = a["vy"] - b["vy"]

            relative_speed = np.sqrt(
                relative_vx**2 + relative_vy**2
            )

            rows.append({
                "frame": int(frame),
                "vehicle_1": int(a["vehicle_id"]),
                "vehicle_2": int(b["vehicle_id"]),
                "relative_vx": relative_vx,
                "relative_vy": relative_vy,
                "relative_speed": relative_speed
            })

result = pd.DataFrame(rows)

result.to_csv(OUTPUT, index=False)

print("\nRelative motion analysis completed!")
print("Rows:", len(result))
print("Output:", OUTPUT)

if len(result) > 0:
    print("\nHighest relative-motion interactions:")
    print(
        result.sort_values(
            "relative_speed",
            ascending=False
        ).head(20).to_string(index=False)
    )
else:
    print("\nNo continuous vehicle pairs were available.")

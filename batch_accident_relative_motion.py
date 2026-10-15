import pandas as pd
import numpy as np
from pathlib import Path

INPUT_DIR = Path("data/ACCIDENT/tracking")

movement_files = sorted(INPUT_DIR.glob("*_movement.csv"))

print("Movement files found:", len(movement_files))

for input_file in movement_files:

    output_file = INPUT_DIR / (
        f"{input_file.stem.replace('_movement', '')}_relative_motion.csv"
    )

    print(f"\nProcessing: {input_file.name}")

    df = pd.read_csv(input_file)

    if df.empty:
        print("No data. Skipping.")
        continue

    df = df.sort_values(["vehicle_id", "frame"]).copy()

    # Calculate frame-to-frame velocity
    df["prev_x"] = df.groupby("vehicle_id")["center_x"].shift(1)
    df["prev_y"] = df.groupby("vehicle_id")["center_y"].shift(1)
    df["prev_frame"] = df.groupby("vehicle_id")["frame"].shift(1)

    df["frame_gap"] = df["frame"] - df["prev_frame"]

    df["vx"] = df["center_x"] - df["prev_x"]
    df["vy"] = df["center_y"] - df["prev_y"]

    # Only use consecutive frames
    df.loc[
        df["frame_gap"] != 1,
        ["vx", "vy"]
    ] = np.nan

    rows = []

    # Compare vehicles appearing in the same frame
    for frame, frame_data in df.groupby("frame"):

        vehicles = frame_data.to_dict("records")

        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):

                v1 = vehicles[i]
                v2 = vehicles[j]

                # Skip if either vehicle has no valid motion
                if (
                    pd.isna(v1["vx"]) or
                    pd.isna(v1["vy"]) or
                    pd.isna(v2["vx"]) or
                    pd.isna(v2["vy"])
                ):
                    continue

                relative_vx = v1["vx"] - v2["vx"]
                relative_vy = v1["vy"] - v2["vy"]

                relative_speed = np.sqrt(
                    relative_vx ** 2 +
                    relative_vy ** 2
                )

                rows.append({
                    "frame": int(frame),
                    "vehicle_1": int(v1["vehicle_id"]),
                    "vehicle_2": int(v2["vehicle_id"]),
                    "type_1": v1["vehicle_type"],
                    "type_2": v2["vehicle_type"],
                    "vx_1": v1["vx"],
                    "vy_1": v1["vy"],
                    "vx_2": v2["vx"],
                    "vy_2": v2["vy"],
                    "relative_vx": relative_vx,
                    "relative_vy": relative_vy,
                    "relative_speed": relative_speed
                })

    result = pd.DataFrame(rows)

    result.to_csv(output_file, index=False)

    print(
        f"Saved: {output_file.name} | "
        f"Pair observations: {len(result)}"
    )

print("\nAll relative-motion analysis completed.")
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_DIR = Path("data/ACCIDENT/tracking")

tracking_files = sorted(INPUT_DIR.glob("*_tracking.csv"))

print("Tracking files found:", len(tracking_files))

for input_file in tracking_files:

    output_file = INPUT_DIR / f"{input_file.stem.replace('_tracking', '')}_movement.csv"

    print(f"\nProcessing: {input_file.name}")

    df = pd.read_csv(input_file)

    if df.empty:
        print("No tracking data. Skipping.")
        continue

    df = df.sort_values(["vehicle_id", "frame"]).copy()

    # Previous position of each tracked vehicle
    df["prev_x"] = df.groupby("vehicle_id")["center_x"].shift(1)
    df["prev_y"] = df.groupby("vehicle_id")["center_y"].shift(1)
    df["prev_frame"] = df.groupby("vehicle_id")["frame"].shift(1)

    # Difference between current and previous frame
    df["frame_gap"] = df["frame"] - df["prev_frame"]

    df["dx"] = df["center_x"] - df["prev_x"]
    df["dy"] = df["center_y"] - df["prev_y"]

    # Movement distance in pixels
    df["movement"] = np.sqrt(
        df["dx"] ** 2 + df["dy"] ** 2
    )

    # Only calculate movement for consecutive frames
    df.loc[
        df["frame_gap"] != 1,
        ["dx", "dy", "movement"]
    ] = np.nan

    df.to_csv(output_file, index=False)

    print(
        f"Saved: {output_file.name} | "
        f"Rows: {len(df)} | "
        f"Vehicles: {df['vehicle_id'].nunique()}"
    )

print("\nAll movement analysis completed.")
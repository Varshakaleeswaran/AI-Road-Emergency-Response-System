import pandas as pd
import numpy as np
from pathlib import Path

INPUT_DIR = Path("data/ACCIDENT/tracking")

movement_files = sorted(INPUT_DIR.glob("*_movement.csv"))

print("Movement files found:", len(movement_files))

for input_file in movement_files:

    output_file = INPUT_DIR / f"{input_file.stem.replace('_movement', '')}_distance.csv"

    print(f"\nProcessing: {input_file.name}")

    df = pd.read_csv(input_file)

    if df.empty:
        print("No data. Skipping.")
        continue

    rows = []

    # Compare vehicles that appear in the same frame
    for frame, frame_data in df.groupby("frame"):

        vehicles = frame_data.to_dict("records")

        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):

                v1 = vehicles[i]
                v2 = vehicles[j]

                dx = v2["center_x"] - v1["center_x"]
                dy = v2["center_y"] - v1["center_y"]

                distance = np.sqrt(dx**2 + dy**2)

                rows.append({
                    "frame": int(frame),
                    "vehicle_1": int(v1["vehicle_id"]),
                    "vehicle_2": int(v2["vehicle_id"]),
                    "type_1": v1["vehicle_type"],
                    "type_2": v2["vehicle_type"],
                    "distance": distance
                })

    result = pd.DataFrame(rows)

    result.to_csv(output_file, index=False)

    print(
        f"Saved: {output_file.name} | "
        f"Pair observations: {len(result)}"
    )

print("\nAll vehicle-distance analysis completed.")
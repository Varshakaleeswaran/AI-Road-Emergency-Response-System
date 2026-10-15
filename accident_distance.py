import pandas as pd
import numpy as np
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
args = parser.parse_args()

INPUT = Path(args.input)
OUTPUT = INPUT.with_name(
    INPUT.stem.replace("_tracking", "_vehicle_distance") + ".csv"
)

df = pd.read_csv(INPUT)

rows = []

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
                "distance": distance
            })

result = pd.DataFrame(rows)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)

print("\nVehicle distance analysis completed!")
print("Rows:", len(result))
print("Output:", OUTPUT)

print("\nClosest vehicle pairs:")
if len(result) > 0:
    print(
        result.sort_values("distance")
        .head(20)
        .to_string(index=False)
    )
else:
    print("No vehicle pairs found.")

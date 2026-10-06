import pandas as pd
import numpy as np

INPUT = "data/ACCIDENT/tracking/unS0-TLF1ao_00_tracking.csv"
OUTPUT = "data/ACCIDENT/tracking/unS0-TLF1ao_00_vehicle_distance.csv"

df = pd.read_csv(INPUT)

rows = []

# Process each frame
for frame, frame_data in df.groupby("frame"):

    vehicles = frame_data.to_dict("records")

    # Compare every pair of vehicles in the same frame
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

result.to_csv(OUTPUT, index=False)

print("\nVehicle distance analysis completed!")
print("Rows:", len(result))
print("Output:", OUTPUT)

print("\nClosest vehicle pairs:")
print(
    result.sort_values("distance")
    .head(20)
    .to_string(index=False)
)

import pandas as pd
import numpy as np

# ---------------------------------------------------
# Load data
# ---------------------------------------------------

df = pd.read_csv("vehicle_movement_smoothed.csv")

print("Smoothed movement data loaded.")
print("Total rows:", len(df))

# ---------------------------------------------------
# Keep only rows with valid vehicle positions
# ---------------------------------------------------

df = df.dropna(
    subset=["center_x", "center_y"]
).copy()

# ---------------------------------------------------
# Group vehicles by frame
# ---------------------------------------------------

proximity_records = []

for frame, frame_data in df.groupby("frame"):

    vehicles = frame_data.to_dict("records")

    # Compare every vehicle with every other vehicle
    for i in range(len(vehicles)):

        vehicle_a = vehicles[i]

        for j in range(i + 1, len(vehicles)):

            vehicle_b = vehicles[j]

            # Same vehicle should never be compared
            if vehicle_a["vehicle_id"] == vehicle_b["vehicle_id"]:
                continue

            # ---------------------------------------------------
            # Calculate center-point distance
            # ---------------------------------------------------

            dx = (
                vehicle_a["center_x"]
                - vehicle_b["center_x"]
            )

            dy = (
                vehicle_a["center_y"]
                - vehicle_b["center_y"]
            )

            distance = np.sqrt(
                dx ** 2 + dy ** 2
            )

            proximity_records.append(
                [
                    frame,
                    vehicle_a["vehicle_id"],
                    vehicle_b["vehicle_id"],
                    vehicle_a["stable_vehicle_type"],
                    vehicle_b["stable_vehicle_type"],
                    distance
                ]
            )

# ---------------------------------------------------
# Create DataFrame
# ---------------------------------------------------

proximity_df = pd.DataFrame(
    proximity_records,
    columns=[
        "frame",
        "vehicle_a",
        "vehicle_b",
        "vehicle_a_type",
        "vehicle_b_type",
        "center_distance"
    ]
)

# ---------------------------------------------------
# Save all pair distances
# ---------------------------------------------------

proximity_df.to_csv(
    "vehicle_proximity.csv",
    index=False
)

print()
print("Vehicle proximity analysis completed!")
print("Saved: vehicle_proximity.csv")

print()
print("Total vehicle-pair records:")
print(len(proximity_df))

# ---------------------------------------------------
# Show closest vehicle pairs
# ---------------------------------------------------

print()
print("========== 30 CLOSEST VEHICLE PAIRS ==========")

closest = (
    proximity_df
    .sort_values("center_distance")
    .head(30)
)

print(
    closest.to_string(index=False)
)

# ---------------------------------------------------
# Create proximity candidates
# ---------------------------------------------------

# NOTE:
# This is NOT an accident threshold.
# It is only used to find close vehicles
# for further investigation.

PROXIMITY_THRESHOLD = 50

close_pairs = proximity_df[
    proximity_df["center_distance"]
    <= PROXIMITY_THRESHOLD
].copy()

close_pairs.to_csv(
    "close_vehicle_pairs.csv",
    index=False
)

print()
print("========== PROXIMITY SUMMARY ==========")

print(
    "Pairs within",
    PROXIMITY_THRESHOLD,
    "pixels:",
    len(close_pairs)
)

print(
    "Saved: close_vehicle_pairs.csv"
)
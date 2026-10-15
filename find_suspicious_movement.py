import pandas as pd

# Load trajectory data
df = pd.read_csv("vehicle_trajectory.csv")

print("Trajectory data loaded.")
print("Total rows:", len(df))

# ---------------------------------------------------
# 1. Highest movement
# ---------------------------------------------------

print()
print("========== TOP 20 HIGHEST MOVEMENT ==========")

top_movement = (
    df.dropna(subset=["movement_per_sec"])
      .sort_values("movement_per_sec", ascending=False)
      .head(20)
)

print(
    top_movement[
        [
            "frame",
            "vehicle_id",
            "stable_vehicle_type",
            "movement_per_sec",
            "movement_change",
            "deceleration"
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------
# 2. Strongest deceleration
# ---------------------------------------------------

print()
print("========== TOP 20 STRONGEST DECELERATION ==========")

top_deceleration = (
    df.dropna(subset=["deceleration"])
      .sort_values("deceleration", ascending=False)
      .head(20)
)

print(
    top_deceleration[
        [
            "frame",
            "vehicle_id",
            "stable_vehicle_type",
            "movement_per_sec",
            "movement_change",
            "deceleration"
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------
# 3. Largest direction changes
# ---------------------------------------------------

print()
print("========== TOP 20 DIRECTION CHANGES ==========")

top_direction = (
    df.dropna(subset=["direction_change_angle"])
      .sort_values("direction_change_angle", ascending=False)
      .head(20)
)

print(
    top_direction[
        [
            "frame",
            "vehicle_id",
            "stable_vehicle_type",
            "movement_per_sec",
            "direction_change_angle"
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------
# 4. Save suspicious candidates
# ---------------------------------------------------

# These are NOT accident detections.
# They are only movement candidates for further analysis.

suspicious = df[
    (df["movement_per_sec"] > 300) |
    (df["deceleration"] > 150)
].copy()

suspicious.to_csv(
    "suspicious_movement.csv",
    index=False
)

print()
print("========== SUMMARY ==========")
print("Suspicious movement rows:", len(suspicious))
print("Saved: suspicious_movement.csv")

print()
print("Vehicles involved:")

print(
    suspicious[
        [
            "vehicle_id",
            "stable_vehicle_type"
        ]
    ]
    .drop_duplicates()
    .sort_values("vehicle_id")
    .to_string(index=False)
)
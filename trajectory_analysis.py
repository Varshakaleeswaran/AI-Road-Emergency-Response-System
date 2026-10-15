import pandas as pd
import numpy as np

# Load clean movement data
input_file = "vehicle_movement_clean.csv"
df = pd.read_csv(input_file)

print("Clean movement data loaded.")
print("Total rows:", len(df))

# Sort by vehicle and frame
df = df.sort_values(["vehicle_id", "frame"]).reset_index(drop=True)

# ---------------------------------------------------
# 1. Calculate movement change
# ---------------------------------------------------

df["movement_change"] = (
    df.groupby("vehicle_id")["movement_per_sec"].diff()
)

# ---------------------------------------------------
# 2. Calculate movement ratio
# ---------------------------------------------------

previous_movement = (
    df.groupby("vehicle_id")["movement_per_sec"].shift(1)
)

df["movement_ratio"] = (
    df["movement_per_sec"] / previous_movement.replace(0, np.nan)
)

# ---------------------------------------------------
# 3. Calculate direction
# ---------------------------------------------------

df["direction_x"] = df["dx"]
df["direction_y"] = df["dy"]

# ---------------------------------------------------
# 4. Calculate direction change
# ---------------------------------------------------

previous_dx = df.groupby("vehicle_id")["dx"].shift(1)
previous_dy = df.groupby("vehicle_id")["dy"].shift(1)

dot_product = (
    df["dx"] * previous_dx +
    df["dy"] * previous_dy
)

magnitude_current = np.sqrt(
    df["dx"] ** 2 + df["dy"] ** 2
)

magnitude_previous = np.sqrt(
    previous_dx ** 2 + previous_dy ** 2
)

cos_angle = (
    dot_product /
    (magnitude_current * magnitude_previous)
)

cos_angle = cos_angle.clip(-1, 1)

df["direction_change_angle"] = np.degrees(
    np.arccos(cos_angle)
)

# ---------------------------------------------------
# 5. Calculate sudden deceleration
# ---------------------------------------------------

df["deceleration"] = -df["movement_change"]

# Positive value means movement decreased
# Larger value = stronger sudden slowdown

# ---------------------------------------------------
# 6. Save trajectory data
# ---------------------------------------------------

output_file = "vehicle_trajectory.csv"

df.to_csv(output_file, index=False)

print()
print("Trajectory analysis completed!")
print("Saved:", output_file)

print()
print("Columns:")
print(df.columns.tolist())

print()
print("First 20 rows:")
print(df.head(20).to_string(index=False))

print()
print("Maximum movement:")
print(df["movement_per_sec"].max())

print()
print("Maximum deceleration:")
print(df["deceleration"].max())

print()
print("Maximum direction change:")
print(df["direction_change_angle"].max())
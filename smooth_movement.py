import pandas as pd

# Load trajectory data
df = pd.read_csv("vehicle_trajectory.csv")

print("Trajectory data loaded.")
print("Total rows:", len(df))

# Sort by vehicle and frame
df = df.sort_values(
    ["vehicle_id", "frame"]
).reset_index(drop=True)

# ---------------------------------------------------
# Smooth movement
# ---------------------------------------------------

# Use a 5-frame rolling median.
# Median is useful because it reduces sudden noisy spikes.

df["smoothed_movement"] = (
    df.groupby("vehicle_id")["movement_per_sec"]
      .transform(
          lambda x: x.rolling(
              window=5,
              center=True,
              min_periods=3
          ).median()
      )
)

# ---------------------------------------------------
# Smooth deceleration
# ---------------------------------------------------

df["smoothed_deceleration"] = (
    -df.groupby("vehicle_id")["smoothed_movement"].diff()
)

# ---------------------------------------------------
# Save result
# ---------------------------------------------------

output_file = "vehicle_movement_smoothed.csv"

df.to_csv(output_file, index=False)

print()
print("Smoothed movement analysis completed!")
print("Saved:", output_file)

print()
print("Columns:")
print(df.columns.tolist())

print()
print("First 20 rows:")
print(
    df[
        [
            "frame",
            "vehicle_id",
            "stable_vehicle_type",
            "movement_per_sec",
            "smoothed_movement",
            "smoothed_deceleration"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print()
print("Maximum raw movement:")
print(df["movement_per_sec"].max())

print()
print("Maximum smoothed movement:")
print(df["smoothed_movement"].max())

print()
print("Maximum smoothed deceleration:")
print(df["smoothed_deceleration"].max())
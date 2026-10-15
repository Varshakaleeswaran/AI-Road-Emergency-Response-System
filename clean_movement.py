import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. Load the stabilized tracking data
# --------------------------------------------------

input_file = "vehicle_movement_stable.csv"

df = pd.read_csv(input_file)

print("Input data loaded.")
print("Total rows:", len(df))


# --------------------------------------------------
# 2. Sort the data
# --------------------------------------------------

df = df.sort_values(
    ["vehicle_id", "frame"]
).reset_index(drop=True)


# --------------------------------------------------
# 3. Calculate frame gap
# --------------------------------------------------

df["frame_gap"] = (
    df.groupby("vehicle_id")["frame"].diff()
)


# --------------------------------------------------
# 4. Calculate position change
# --------------------------------------------------

df["dx"] = (
    df.groupby("vehicle_id")["center_x"].diff()
)

df["dy"] = (
    df.groupby("vehicle_id")["center_y"].diff()
)


# --------------------------------------------------
# 5. Calculate displacement
# --------------------------------------------------

df["displacement"] = np.sqrt(
    df["dx"] ** 2 +
    df["dy"] ** 2
)


# --------------------------------------------------
# 6. Calculate movement per second
# --------------------------------------------------

# Video FPS = 50
FPS = 50

df["movement_per_sec"] = (
    df["displacement"] * FPS
)


# --------------------------------------------------
# 7. Remove invalid movement calculations
# --------------------------------------------------

# Movement is valid only when the previous
# observation was exactly one frame earlier.

df.loc[
    df["frame_gap"] != 1,
    [
        "dx",
        "dy",
        "displacement",
        "movement_per_sec"
    ]
] = np.nan


# --------------------------------------------------
# 8. Save cleaned movement data
# --------------------------------------------------

output_file = "vehicle_movement_clean.csv"

df.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# 9. Display useful information
# --------------------------------------------------

print()
print("Clean movement analysis completed!")
print("Saved:", output_file)

print()
print("Columns:")
print(df.columns.tolist())

print()
print("First 20 rows:")
print(
    df.head(20).to_string(index=False)
)

print()
print("Valid movement rows:",
      df["movement_per_sec"].notna().sum())

print("Invalid/gap rows:",
      df["movement_per_sec"].isna().sum())
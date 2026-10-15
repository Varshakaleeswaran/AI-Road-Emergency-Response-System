import pandas as pd
from pathlib import Path

BASE = Path("data/ACCIDENT")
TRACKING_DIR = BASE / "tracking"

GROUND_TRUTH = BASE / "starter_25_ground_truth.csv"
OUTPUT = BASE / "accident_window_analysis.csv"

WINDOW_BEFORE = 10
WINDOW_AFTER = 10

ground_truth = pd.read_csv(GROUND_TRUTH)

rows = []

for _, gt in ground_truth.iterrows():

    video_path = gt["path"]
    video_name = Path(video_path).stem

    accident_frame = int(gt["accident_frame_metadata"])
    accident_type = gt["type_metadata"]

    distance_file = TRACKING_DIR / f"{video_name}_distance.csv"
    relative_file = TRACKING_DIR / f"{video_name}_relative_motion.csv"

    print(f"\nProcessing: {video_name}")
    print(f"Accident frame: {accident_frame}")

    # Read distance data safely
    distance_df = pd.DataFrame()

    if distance_file.exists():

        try:
            if distance_file.stat().st_size > 0:
                distance_df = pd.read_csv(distance_file)
        except pd.errors.EmptyDataError:
            distance_df = pd.DataFrame()

    # Read relative motion data safely
    relative_df = pd.DataFrame()

    if relative_file.exists():

        try:
            if relative_file.stat().st_size > 0:
                relative_df = pd.read_csv(relative_file)
        except pd.errors.EmptyDataError:
            relative_df = pd.DataFrame()

    # Accident window
    start_frame = accident_frame - WINDOW_BEFORE
    end_frame = accident_frame + WINDOW_AFTER

    # Distance window
    if len(distance_df) > 0:

        distance_window = distance_df[
            (distance_df["frame"] >= start_frame) &
            (distance_df["frame"] <= end_frame)
        ].copy()

    else:

        distance_window = pd.DataFrame()

    if len(distance_window) > 0:

        min_distance = distance_window["distance"].min()
        mean_distance = distance_window["distance"].mean()

    else:

        min_distance = None
        mean_distance = None

    # Relative motion window
    if len(relative_df) > 0:

        relative_window = relative_df[
            (relative_df["frame"] >= start_frame) &
            (relative_df["frame"] <= end_frame)
        ].copy()

    else:

        relative_window = pd.DataFrame()

    if len(relative_window) > 0:

        max_relative_speed = relative_window["relative_speed"].max()
        mean_relative_speed = relative_window["relative_speed"].mean()

    else:

        max_relative_speed = None
        mean_relative_speed = None

    # Save result
    rows.append({
        "video": video_name,
        "accident_type": accident_type,
        "accident_frame": accident_frame,
        "window_start": start_frame,
        "window_end": end_frame,
        "distance_observations": len(distance_window),
        "min_distance": min_distance,
        "mean_distance": mean_distance,
        "relative_motion_observations": len(relative_window),
        "max_relative_speed": max_relative_speed,
        "mean_relative_speed": mean_relative_speed
    })


# Create final table
result = pd.DataFrame(rows)

result.to_csv(OUTPUT, index=False)

print("\n======================================")
print("ACCIDENT WINDOW ANALYSIS COMPLETED")
print("======================================")

print("Videos analyzed:", len(result))
print("Saved to:", OUTPUT)

print("\nSummary:")
print(result.to_string(index=False))
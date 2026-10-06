import pandas as pd
from pathlib import Path

BASE = Path("data/ACCIDENT")
TRACKING_DIR = BASE / "tracking"

GROUND_TRUTH = BASE / "starter_25_ground_truth.csv"
OUTPUT = BASE / "pair_interaction_analysis.csv"

WINDOW_BEFORE = 10
WINDOW_AFTER = 10

ground_truth = pd.read_csv(GROUND_TRUTH)

results = []

for _, gt in ground_truth.iterrows():

    video_name = Path(gt["path"]).stem
    accident_frame = int(gt["accident_frame_metadata"])
    accident_type = gt["type_metadata"]

    distance_file = TRACKING_DIR / f"{video_name}_distance.csv"

    print(f"\nProcessing: {video_name}")
    print(f"Accident frame: {accident_frame}")

    # -------------------------
    # Read distance file
    # -------------------------

    if not distance_file.exists():
        print("Distance file not found.")
        continue

    try:
        if distance_file.stat().st_size == 0:
            print("Distance file is empty.")
            continue

        distance_df = pd.read_csv(distance_file)

    except pd.errors.EmptyDataError:
        print("Distance file is empty.")
        continue

    if len(distance_df) == 0:
        print("No distance observations.")
        continue

    # -------------------------
    # Accident window
    # -------------------------

    start_frame = accident_frame - WINDOW_BEFORE
    end_frame = accident_frame + WINDOW_AFTER

    window = distance_df[
        (distance_df["frame"] >= start_frame) &
        (distance_df["frame"] <= end_frame)
    ].copy()

    if len(window) == 0:
        print("No vehicle pairs in accident window.")
        continue

    # -------------------------
    # Group by vehicle pair
    # -------------------------

    pair_groups = window.groupby(
        ["vehicle_1", "vehicle_2", "type_1", "type_2"]
    )

    for pair, group in pair_groups:

        vehicle_1, vehicle_2, type_1, type_2 = pair

        # Minimum distance of this pair
        min_distance = group["distance"].min()

        # Frame where minimum distance occurred
        min_distance_frame = group.loc[
            group["distance"].idxmin(), "frame"
        ]

        # Distance at/near accident frame
        accident_distances = group[
            group["frame"] == accident_frame
        ]["distance"]

        if len(accident_distances) > 0:
            distance_at_accident = accident_distances.iloc[0]
        else:
            distance_at_accident = None

        # Number of observations
        observations = len(group)

        results.append({
            "video": video_name,
            "accident_type": accident_type,
            "accident_frame": accident_frame,
            "vehicle_1": vehicle_1,
            "vehicle_2": vehicle_2,
            "type_1": type_1,
            "type_2": type_2,
            "observations": observations,
            "min_distance": min_distance,
            "min_distance_frame": min_distance_frame,
            "distance_at_accident": distance_at_accident
        })


# -------------------------
# Save results
# -------------------------

result = pd.DataFrame(results)

result.to_csv(OUTPUT, index=False)

print("\n======================================")
print("PAIR INTERACTION ANALYSIS COMPLETED")
print("======================================")

print("Vehicle-pair records:", len(result))
print("Saved to:", OUTPUT)

print("\nTop closest vehicle pairs:")

if len(result) > 0:

    print(
        result
        .sort_values("min_distance")
        .head(30)
        .to_string(index=False)
    )

else:

    print("No vehicle-pair records found.")
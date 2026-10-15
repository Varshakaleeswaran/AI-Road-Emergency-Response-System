import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("data/ACCIDENT/tracking")
OUTPUT = Path("data/ACCIDENT/generic_video_features.csv")

rows = []

tracking_files = sorted(BASE.glob("*_tracking.csv"))

for tracking_file in tracking_files:

    stem = tracking_file.stem.replace("_tracking", "")
    movement_file = BASE / f"{stem}_movement.csv"
    relative_file = BASE / f"{stem}_relative_motion.csv"

    distance_file = BASE / f"{stem}_vehicle_distance.csv"

    if not distance_file.exists():
        distance_file = BASE / f"{stem}_distance.csv"

    if not movement_file.exists():
        print(f"SKIP movement: {stem}")
        continue

    if not relative_file.exists():
        print(f"SKIP relative: {stem}")
        continue

    if not distance_file.exists():
        print(f"SKIP distance: {stem}")
        continue

    try:
        tracking = pd.read_csv(tracking_file)
        movement = pd.read_csv(movement_file)
        distance = pd.read_csv(distance_file)
        relative = pd.read_csv(relative_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"SKIP empty/invalid file: {stem} -> {e}")
        continue

    if distance.empty:
        print(f"SKIP empty distance: {stem}")
        continue

    def clean(df):
        return df.replace([np.inf, -np.inf], np.nan)

    tracking = clean(tracking)
    movement = clean(movement)
    distance = clean(distance)
    relative = clean(relative)

    label = "normal" if stem == "road_video" else "accident"

    rows.append({
        "video": stem,
        "label": label,

        "tracking_rows": len(tracking),
        "unique_vehicle_ids": tracking["vehicle_id"].nunique(),
        "frames": tracking["frame"].nunique(),

        "movement_rows": len(movement),
        "movement_mean": movement["movement"].mean(),
        "movement_max": movement["movement"].max(),

        "distance_rows": len(distance),
        "distance_min": distance["distance"].min(),
        "distance_mean": distance["distance"].mean(),

        "relative_motion_rows": len(relative),
        "relative_speed_mean": relative["relative_speed"].mean(),
        "relative_speed_max": relative["relative_speed"].max()
    })

out = pd.DataFrame(rows)

if out.empty:
    print("No valid videos found.")
    raise SystemExit(1)

out = out.sort_values("video").reset_index(drop=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT, index=False)

print("=" * 60)
print("GENERIC VIDEO FEATURE DATASET")
print("=" * 60)
print("Rows:", len(out))
print("Accident videos:", (out["label"] == "accident").sum())
print("Normal videos:", (out["label"] == "normal").sum())

print("\nLabels:")
print(out["label"].value_counts().to_string())

print("\nSaved:")
print(OUTPUT)

print("\nVideos:")
print(out[["video", "label"]].to_string(index=False))

from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("data/ACCIDENT/tracking")
OUTPUT = Path("data/ACCIDENT/generic_window_features.csv")

WINDOW = 50

rows = []

tracking_files = sorted(BASE.glob("*_tracking.csv"))

for tracking_file in tracking_files:

    stem = tracking_file.stem.replace("_tracking", "")

    movement_file = BASE / f"{stem}_movement.csv"
    relative_file = BASE / f"{stem}_relative_motion.csv"
    distance_file = BASE / f"{stem}_vehicle_distance.csv"

    if not distance_file.exists():
        distance_file = BASE / f"{stem}_distance.csv"

    if not movement_file.exists() or not relative_file.exists() or not distance_file.exists():
        continue

    try:
        tracking = pd.read_csv(tracking_file)
        movement = pd.read_csv(movement_file)
        relative = pd.read_csv(relative_file)
        distance = pd.read_csv(distance_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        continue

    if tracking.empty or movement.empty or relative.empty or distance.empty:
        continue

    def clean(df):
        return df.replace([np.inf, -np.inf], np.nan)

    tracking = clean(tracking)
    movement = clean(movement)
    relative = clean(relative)
    distance = clean(distance)

    label = "normal" if stem in ["road_video", "normal_01", "normal_02"] else "accident"

    min_frame = int(tracking["frame"].min())
    max_frame = int(tracking["frame"].max())

    for start in range(min_frame, max_frame + 1, WINDOW):

        end = min(start + WINDOW - 1, max_frame)

        t = tracking[
            (tracking["frame"] >= start) &
            (tracking["frame"] <= end)
        ]

        m = movement[
            (movement["frame"] >= start) &
            (movement["frame"] <= end)
        ]

        r = relative[
            (relative["frame"] >= start) &
            (relative["frame"] <= end)
        ]

        d = distance[
            (distance["frame"] >= start) &
            (distance["frame"] <= end)
        ]

        if t.empty:
            continue

        rows.append({
            "video": stem,
            "label": label,
            "window_start": start,
            "window_end": end,

            "tracking_rows": len(t),
            "unique_vehicle_ids": t["vehicle_id"].nunique(),

            "movement_rows": len(m),
            "movement_mean": m["movement"].mean(),
            "movement_max": m["movement"].max(),

            "distance_rows": len(d),
            "distance_min": d["distance"].min(),
            "distance_mean": d["distance"].mean(),

            "relative_motion_rows": len(r),
            "relative_speed_mean": r["relative_speed"].mean(),
            "relative_speed_max": r["relative_speed"].max()
        })

out = pd.DataFrame(rows)

if out.empty:
    print("No valid windows found.")
    raise SystemExit(1)

out = out.sort_values(
    ["video", "window_start"]
).reset_index(drop=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT, index=False)

print("=" * 60)
print("GENERIC WINDOW FEATURE DATASET")
print("=" * 60)

print("Rows:", len(out))
print("Accident windows:", (out["label"] == "accident").sum())
print("Normal windows:", (out["label"] == "normal").sum())

print("\nVideos:")
print(out.groupby(["label"])["video"].nunique().to_string())

print("\nWindows per video:")
print(out.groupby(["video", "label"]).size().to_string())

print("\nSaved:")
print(OUTPUT)

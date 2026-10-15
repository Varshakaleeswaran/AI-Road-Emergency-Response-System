import pandas as pd
import numpy as np
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tracking", required=True)
parser.add_argument("--movement", required=True)
parser.add_argument("--distance", required=True)
parser.add_argument("--relative", required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

tracking = pd.read_csv(args.tracking)
movement = pd.read_csv(args.movement)
distance = pd.read_csv(args.distance)
relative = pd.read_csv(args.relative)

def clean(df):
    return df.replace([np.inf, -np.inf], np.nan)

tracking = clean(tracking)
movement = clean(movement)
distance = clean(distance)
relative = clean(relative)

features = {
    "label": args.label,

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
    "relative_speed_max": relative["relative_speed"].max(),
}

out = pd.DataFrame([features])
out.to_csv(args.output, index=False)

print("=" * 60)
print("GENERIC VIDEO FEATURE SUMMARY")
print("=" * 60)
print(out.to_string(index=False))
print("\nSaved:")
print(args.output)

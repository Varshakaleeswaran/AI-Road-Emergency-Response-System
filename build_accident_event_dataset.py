import pandas as pd
from pathlib import Path

INPUT = Path("data/ACCIDENT/generic_window_features.csv")
GT_FILE = Path("data/ACCIDENT/starter_25_ground_truth.csv")
OUTPUT = Path("data/ACCIDENT/accident_event_window_dataset.csv")

df = pd.read_csv(INPUT)
gt = pd.read_csv(GT_FILE)

gt["video"] = (
    gt["path"]
    .str.replace("real_videos/", "", regex=False)
    .str.replace(".mp4", "", regex=False)
)

gt = gt[["video", "accident_frame_starter"]].rename(
    columns={"accident_frame_starter": "accident_frame"}
)

df = df.merge(gt, on="video", how="left")

df["event_label"] = "non_event"

normal_mask = df["label"] == "normal"
df.loc[normal_mask, "event_label"] = "normal"

accident_mask = (
    (df["label"] == "accident") &
    (df["window_start"] <= df["accident_frame"]) &
    (df["window_end"] >= df["accident_frame"])
)

df.loc[accident_mask, "event_label"] = "accident_event"

df.to_csv(OUTPUT, index=False)

print("=" * 60)
print("ACCIDENT EVENT WINDOW DATASET")
print("=" * 60)
print("Rows:", len(df))

print("\nLabels:")
print(df["event_label"].value_counts().to_string())

print("\nVideos per label:")
print(
    df.groupby("event_label")["video"]
    .nunique()
    .to_string()
)

print("\nEvent windows:")
print(
    df[df["event_label"] == "accident_event"]
    [["video", "accident_frame", "window_start", "window_end"]]
    .to_string(index=False)
)

print("\nSaved:")
print(OUTPUT)

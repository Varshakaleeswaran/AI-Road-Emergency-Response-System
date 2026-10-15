from ultralytics import YOLO
import pandas as pd
from pathlib import Path

VIDEO = "data/ACCIDENT/real_videos/unS0-TLF1ao_00.mp4"

OUT_DIR = Path("data/ACCIDENT/tracking")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "unS0-TLF1ao_00_tracking.csv"

# Load YOLO26
model = YOLO("yolo26n.pt")

# Track vehicles using ByteTrack
results = model.track(
    source=VIDEO,
    tracker="bytetrack.yaml",
    stream=True,
    persist=True,
    verbose=False
)

rows = []

# Vehicle classes we want
vehicle_classes = {"car", "motorcycle", "bus", "truck"}

for frame_idx, result in enumerate(results):

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        continue

    if boxes.id is None:
        continue

    xyxy = boxes.xyxy.cpu().numpy()
    track_ids = boxes.id.int().cpu().numpy()
    class_ids = boxes.cls.int().cpu().numpy()

    for box, track_id, class_id in zip(xyxy, track_ids, class_ids):

        class_name = result.names[int(class_id)]

        if class_name not in vehicle_classes:
            continue

        x1, y1, x2, y2 = box.tolist()

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        rows.append({
            "frame": frame_idx,
            "vehicle_id": int(track_id),
            "vehicle_type": class_name,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "center_x": center_x,
            "center_y": center_y
        })

# Save tracking data
df = pd.DataFrame(rows)

df.to_csv(OUTPUT, index=False)

print("\nTracking extraction completed!")
print("Rows:", len(df))
print("Unique vehicles:", df["vehicle_id"].nunique())
print("Saved to:", OUTPUT)

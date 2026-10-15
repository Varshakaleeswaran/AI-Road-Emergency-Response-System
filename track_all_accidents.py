from ultralytics import YOLO
import pandas as pd
from pathlib import Path

VIDEO_DIR = Path("data/ACCIDENT/real_videos")
OUT_DIR = Path("data/ACCIDENT/tracking")
OUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO("yolo26n.pt")

vehicle_classes = {"car", "motorcycle", "bus", "truck"}

videos = sorted(VIDEO_DIR.glob("*.mp4"))

print("Videos found:", len(videos))

for video in videos:

    output = OUT_DIR / f"{video.stem}_tracking.csv"

    if output.exists():
        print(f"\nSkipping existing: {video.name}")
        continue

    print(f"\nProcessing: {video.name}")

    results = model.track(
        source=str(video),
        tracker="bytetrack.yaml",
        stream=True,
        persist=True,
        verbose=False
    )

    rows = []

    for frame_idx, result in enumerate(results):

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        if boxes.id is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.int().cpu().numpy()
        class_ids = boxes.cls.int().cpu().numpy()

        for box, track_id, class_id in zip(
            xyxy, track_ids, class_ids
        ):

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

    pd.DataFrame(rows).to_csv(output, index=False)

    print(
        f"Saved: {output.name} | "
        f"Rows: {len(rows)} | "
        f"Vehicles: {len(set(r['vehicle_id'] for r in rows))}"
    )

print("\nAll available videos processed.")

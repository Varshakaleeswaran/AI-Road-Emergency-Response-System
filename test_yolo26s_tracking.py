from pathlib import Path
import csv
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
VIDEO = ROOT / "data/ACCIDENT/real_videos/7b4ffyFOo_I_00.mp4"
MODEL = ROOT / "yolo26s.pt"
TRACKER = ROOT / "data/ACCIDENT/tracking/bytetrack_recovery_test.yaml"
OUT = ROOT / "data/ACCIDENT/tracking"
CSV_OUT = OUT / "7b4ffyFOo_I_00_YOLO26S_RECOVERY_TRACKS.csv"
VIDEO_OUT = OUT / "7b4ffyFOo_I_00_YOLO26S_RECOVERY_REVIEW.mp4"

for path in (VIDEO, MODEL, TRACKER):
    if not path.is_file():
        raise FileNotFoundError(f"Required input missing: {path}")

if CSV_OUT.exists() or VIDEO_OUT.exists():
    raise FileExistsError("An output already exists; refusing to overwrite it.")

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO}")
fps = cap.get(cv2.CAP_PROP_FPS) or 12.25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

OUT.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(
    str(VIDEO_OUT), cv2.VideoWriter_fourcc(*"mp4v"),
    fps, (width, height)
)
if not writer.isOpened():
    raise RuntimeError(f"Cannot create review video: {VIDEO_OUT}")

rows = []
frames = 0
try:
    model = YOLO(str(MODEL))
    results = model.track(
        source=str(VIDEO),
        tracker=str(TRACKER),
        stream=True,
        persist=True,
        verbose=False,
        imgsz=640,
        conf=0.10,
        device="cpu",
        classes=[2, 3, 5, 7],
    )

    for frames, result in enumerate(results, start=1):
        image = result.orig_img.copy()
        boxes = result.boxes

        if boxes is not None and len(boxes):
            coords = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.int().cpu().numpy() if boxes.id is not None else [-1] * len(boxes)

            for box, score, cls_id, tid in zip(coords, scores, classes, ids):
                x1, y1, x2, y2 = map(float, box)
                tid = int(tid)
                rows.append({
                    "frame": frames, "track_id": tid,
                    "class_id": int(cls_id), "confidence": float(score),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })
                color = (0, 220, 0) if tid >= 0 else (0, 165, 255)
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"ID {tid} {score:.2f}" if tid >= 0 else f"No ID {score:.2f}"
                cv2.putText(
                    image, label, (int(x1), max(16, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
                )

        cv2.putText(
            image, f"YOLO26s recovery | Frame {frames}",
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 255, 255), 2, cv2.LINE_AA
        )
        writer.write(image)

        if frames % 20 == 0:
            print(f"Processed {frames}/{expected} frames; rows={len(rows)}", flush=True)
finally:
    writer.release()

with CSV_OUT.open("x", newline="", encoding="utf-8") as f:
    fields = ["frame", "track_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]
    output = csv.DictWriter(f, fieldnames=fields)
    output.writeheader()
    output.writerows(rows)

ids = {r["track_id"] for r in rows if r["track_id"] >= 0}
print("\n--- Experiment complete ---")
print(f"Frames processed: {frames}/{expected}")
print(f"Detection/track rows: {len(rows)}")
print(f"Unique assigned IDs: {len(ids)}")
print(f"Rows without IDs: {sum(r['track_id'] < 0 for r in rows)}")
print(f"CSV: {CSV_OUT}")
print(f"Review video: {VIDEO_OUT}")

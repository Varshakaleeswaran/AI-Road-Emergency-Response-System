from ultralytics import YOLO
import pandas as pd
from pathlib import Path
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Input video path")
parser.add_argument("--model", default="yolo26n.pt", help="YOLO model weights")
parser.add_argument("--checkpoint", type=int, default=500)
parser.add_argument("--output", default=None, help="Tracked vehicle CSV")
parser.add_argument("--detections-output", default=None, help="Separate detector-only CSV")
args = parser.parse_args()

video_path = Path(args.input)
if not video_path.exists():
    parser.error(f"Video not found: {video_path}")
if args.checkpoint < 1:
    parser.error("--checkpoint must be at least 1")

output_dir = Path("data/ACCIDENT/tracking")
output_dir.mkdir(parents=True, exist_ok=True)

output = Path(args.output) if args.output else output_dir / f"{video_path.stem}_tracking.csv"
detections_output = (
    Path(args.detections_output) if args.detections_output
    else output.with_name(output.stem + "_detections.csv")
)
output.parent.mkdir(parents=True, exist_ok=True)
detections_output.parent.mkdir(parents=True, exist_ok=True)

if output.resolve() == detections_output.resolve():
    parser.error("Tracking and detections outputs must be different files.")

checkpoint = output.with_name(output.stem + "_checkpoint.csv")
working = output.with_name(output.stem + "_working.csv")
detection_checkpoint = detections_output.with_name(detections_output.stem + "_checkpoint.csv")
detection_working = detections_output.with_name(detections_output.stem + "_working.csv")

tracking_columns = [
    "frame", "vehicle_id", "vehicle_type",
    "x1", "y1", "x2", "y2", "center_x", "center_y"
]
detection_columns = [
    "frame", "vehicle_type", "confidence",
    "x1", "y1", "x2", "y2", "center_x", "center_y"
]
vehicle_classes = {"car", "motorcycle", "bus", "truck"}
INFERENCE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.10
batch_size = 1000
tracking_rows = []
detection_rows = []

cap = cv2.VideoCapture(str(video_path))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# Refuse to overwrite existing final outputs.
for path in (output, detections_output):
    if path.exists():
        raise FileExistsError(f"Output already exists; choose a new path: {path}")

working.write_text(",".join(tracking_columns) + "\n", encoding="utf-8")
checkpoint.write_text(",".join(tracking_columns) + "\n", encoding="utf-8")
detection_working.write_text(",".join(detection_columns) + "\n", encoding="utf-8")
detection_checkpoint.write_text(",".join(detection_columns) + "\n", encoding="utf-8")

def flush_tracking():
    if tracking_rows:
        pd.DataFrame(tracking_rows, columns=tracking_columns).to_csv(
            working, mode="a", header=False, index=False
        )
        tracking_rows.clear()

def flush_detections():
    if detection_rows:
        pd.DataFrame(detection_rows, columns=detection_columns).to_csv(
            detection_working, mode="a", header=False, index=False
        )
        detection_rows.clear()

def save_checkpoint():
    flush_tracking()
    flush_detections()
    pd.read_csv(working).to_csv(checkpoint, index=False)
    pd.read_csv(detection_working).to_csv(detection_checkpoint, index=False)

model_path = Path(args.model)
if not model_path.exists():
    parser.error(f"Model weights not found: {model_path}")
model = YOLO(str(model_path))

try:
    # Independent detector pass: no tracking callback can remove these boxes.
    for frame, result in enumerate(model.predict(
        source=str(video_path),
        stream=True,
        verbose=False,
        imgsz=INFERENCE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        device="cpu",
        classes=[2, 3, 5, 7]
    ), start=1):
        boxes = result.boxes
        if boxes is not None and len(boxes):
            classes = boxes.cls.cpu().numpy().astype(int)
            coords = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()

            for cls, box, confidence in zip(classes, coords, confidences):
                class_name = result.names[int(cls)]
                if class_name not in vehicle_classes:
                    continue
                x1, y1, x2, y2 = map(float, box)
                if x2 <= x1 or y2 <= y1:
                    continue
                detection_rows.append({
                    "frame": frame,
                    "vehicle_type": class_name,
                    "confidence": float(confidence),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "center_x": (x1 + x2) / 2,
                    "center_y": (y1 + y2) / 2
                })

        if len(detection_rows) >= batch_size:
            flush_detections()
        if frame % args.checkpoint == 0:
            flush_detections()
            pd.read_csv(detection_working).to_csv(detection_checkpoint, index=False)
            print(f"Detection pass: {frame}/{total_frames} frames", flush=True)

    flush_detections()
    print("Detector pass completed.", flush=True)

    # Separate tracking pass; preserve the existing tracker configuration.
    for frame, result in enumerate(model.track(
        source=str(video_path),
        tracker="bytetrack.yaml",
        stream=True,
        persist=True,
        verbose=False,
        imgsz=INFERENCE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        batch=1,
        device="cpu",
        classes=[2, 3, 5, 7]
    ), start=1):
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            classes = boxes.cls.cpu().numpy().astype(int)
            coords = boxes.xyxy.cpu().numpy()

            for vehicle_id, cls, box in zip(ids, classes, coords):
                class_name = result.names[int(cls)]
                if class_name not in vehicle_classes:
                    continue
                x1, y1, x2, y2 = map(float, box)
                if x2 <= x1 or y2 <= y1:
                    continue
                tracking_rows.append({
                    "frame": frame,
                    "vehicle_id": int(vehicle_id),
                    "vehicle_type": class_name,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "center_x": (x1 + x2) / 2,
                    "center_y": (y1 + y2) / 2
                })

        if len(tracking_rows) >= batch_size:
            flush_tracking()
        if frame % args.checkpoint == 0:
            flush_tracking()
            pd.read_csv(working).to_csv(checkpoint, index=False)
            print(f"Tracking pass: {frame}/{total_frames} frames", flush=True)

    flush_tracking()
    pd.read_csv(working).to_csv(output, index=False)
    pd.read_csv(detection_working).to_csv(detections_output, index=False)

    checkpoint.unlink(missing_ok=True)
    working.unlink(missing_ok=True)
    detection_checkpoint.unlink(missing_ok=True)
    detection_working.unlink(missing_ok=True)

    tracked = pd.read_csv(output)
    detected = pd.read_csv(detections_output)
    print("\nProcessing completed.")
    print("Frames reported by video:", total_frames)
    print("Tracked rows:", len(tracked))
    print("Tracked IDs:", tracked["vehicle_id"].nunique())
    print("Detector-only rows:", len(detected))
    print("Detector frames:", detected["frame"].nunique())
    print("Tracking CSV:", output)
    print("Detection CSV:", detections_output)

except KeyboardInterrupt:
    save_checkpoint()
    print("\nInterrupted; final CSVs were not published.")
    print("Tracking checkpoint:", checkpoint)
    print("Detection checkpoint:", detection_checkpoint)
    raise

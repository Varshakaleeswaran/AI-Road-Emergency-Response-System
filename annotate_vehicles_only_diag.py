from pathlib import Path
import cv2
from ultralytics import YOLO

video = Path(r"data\ACCIDENT\real_videos\7b4ffyFOo_I_00.mp4")
out = Path(r"data\ACCIDENT\tracking\7b4ffyFOo_I_00_vehicles_only_DIAG.mp4")

model = YOLO("yolo26n.pt")
print("Model class IDs:")
for class_id in (2, 3, 5, 6, 7):
    print(f"  {class_id}: {model.names.get(class_id, 'UNKNOWN')}")

vehicle_ids = [2, 3, 5, 7]
cap = cv2.VideoCapture(str(video))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open input video: {video}")

fps = cap.get(cv2.CAP_PROP_FPS) or 12.25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

writer = cv2.VideoWriter(
    str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)
if not writer.isOpened():
    raise RuntimeError(f"Cannot create output video: {out}")

frames = 0
try:
    for result in model.track(
        source=str(video),
        tracker="bytetrack.yaml",
        stream=True,
        persist=True,
        verbose=False,
        imgsz=416,
        batch=1,
        device="cpu",
        classes=vehicle_ids,
    ):
        writer.write(result.plot())
        frames += 1
        if frames % 20 == 0:
            print(f"Processed {frames} frames")
finally:
    writer.release()

check = cv2.VideoCapture(str(out))
saved_frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
check.release()

if frames != 82 or saved_frames != frames:
    raise RuntimeError(
        f"Frame verification failed: processed={frames}, saved={saved_frames}"
    )

print(f"SUCCESS: {frames} frames")
print(f"Output: {out}")

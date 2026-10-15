from pathlib import Path
import cv2
from ultralytics import YOLO

video = Path(r"data\ACCIDENT\real_videos\7b4ffyFOo_I_00.mp4")
out = Path(r"data\ACCIDENT\tracking\7b4ffyFOo_I_00_annotated_DIAG.mp4")

if not video.exists():
    raise FileNotFoundError(video)
if out.exists():
    raise FileExistsError(out)

cap = cv2.VideoCapture(str(video))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video}")

fps = cap.get(cv2.CAP_PROP_FPS) or 12.25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

writer = cv2.VideoWriter(
    str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)
if not writer.isOpened():
    raise RuntimeError(f"Cannot create video writer: {out}")

model = YOLO("yolo26n.pt")
frames = 0
try:
    results = model.track(
        source=str(video),
        tracker="bytetrack.yaml",
        stream=True,
        persist=True,
        verbose=False,
        imgsz=416,
        batch=1,
        device="cpu",
    )
    for result in results:
        frame = result.plot()
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
        frames += 1
        if frames % 20 == 0:
            print(f"Annotated {frames} frames")
finally:
    writer.release()

if frames == 0:
    raise RuntimeError("No frames were annotated.")

check = cv2.VideoCapture(str(out))
written = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
check.release()
if written != frames:
    raise RuntimeError(f"Output frame-count mismatch: processed={frames}, saved={written}")

print(f"SUCCESS: {frames} frames annotated")
print(f"Output: {out}")

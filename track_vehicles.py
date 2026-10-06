from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("yolo26n.pt")

# Track vehicles in the road video
results = model.track(
    source="data/videos/road_video.mp4",
    tracker="bytetrack.yaml",
    stream=True,
    save=True
)

# Process every frame
for result in results:
    pass

print("Vehicle tracking completed!")
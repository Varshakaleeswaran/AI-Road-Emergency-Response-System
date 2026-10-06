from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("yolo26n.pt")

# Run YOLO on our road video
results = model.predict(
    source="data/videos/road_video.mp4",
    save=True
)

print("Video processing completed!")
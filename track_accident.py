from ultralytics import YOLO

# Load the pretrained YOLO26 model
model = YOLO("yolo26n.pt")

# Track vehicles in one ACCIDENT dataset video
results = model.track(
    source="data/ACCIDENT/real_videos/unS0-TLF1ao_00.mp4",
    tracker="bytetrack.yaml",
    stream=True,
    save=True
)

# Process every frame
for result in results:
    pass

print("ACCIDENT video tracking completed!")

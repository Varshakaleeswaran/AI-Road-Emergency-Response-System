from ultralytics import YOLO

# Load a small pretrained YOLO model
model = YOLO("yolo26n.pt")

# Run YOLO on a test image
results = model("https://ultralytics.com/images/bus.jpg")

# Display the detected objects
results[0].show()
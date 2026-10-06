from ultralytics import YOLO
import csv

# Load YOLO
model = YOLO("yolo26n.pt")

# Vehicle classes
vehicle_classes = {
    "car",
    "motorcycle",
    "bus",
    "truck"
}

# Start tracking
results = model.track(
    source="data/videos/road_video.mp4",
    tracker="bytetrack.yaml",
    stream=True,
    persist=True
)

# ---------------------------------------------------
# Create CSV
# ---------------------------------------------------

output_file = "vehicle_tracking_boxes.csv"

with open(output_file, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "frame",
        "vehicle_id",
        "vehicle_type",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y"
    ])

    # ---------------------------------------------------
    # Process frames
    # ---------------------------------------------------

    for frame_number, result in enumerate(results):

        if result.boxes.id is None:
            continue

        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        classes = (
            result.boxes.cls
            .int()
            .cpu()
            .tolist()
        )

        boxes = (
            result.boxes.xyxy
            .cpu()
            .tolist()
        )

        for track_id, class_id, box in zip(
            track_ids,
            classes,
            boxes
        ):

            vehicle_type = model.names[class_id]

            if vehicle_type not in vehicle_classes:
                continue

            x1, y1, x2, y2 = box

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            writer.writerow([
                frame_number,
                track_id,
                vehicle_type,
                round(x1, 2),
                round(y1, 2),
                round(x2, 2),
                round(y2, 2),
                round(center_x, 2),
                round(center_y, 2)
            ])

print()
print("Vehicle tracking with bounding boxes completed!")
print("Saved:", output_file)
from ultralytics import YOLO
import csv

# --------------------------------------------------
# 1. Load YOLO model
# --------------------------------------------------

model = YOLO("yolo26n.pt")

# Vehicle classes that we want to analyze
vehicle_classes = {
    "car",
    "motorcycle",
    "bus",
    "truck"
}

# --------------------------------------------------
# 2. Start tracking
# --------------------------------------------------

results = model.track(
    source="data/videos/road_video.mp4",
    tracker="bytetrack.yaml",
    stream=True,
    persist=True
)

# --------------------------------------------------
# 3. Create CSV file
# --------------------------------------------------

with open("vehicle_movement.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "frame",
        "vehicle_id",
        "vehicle_type",
        "center_x",
        "center_y"
    ])

    # --------------------------------------------------
    # 4. Process every frame
    # --------------------------------------------------

    for frame_number, result in enumerate(results):

        # If tracking IDs don't exist
        if result.boxes.id is None:
            continue

        # Get tracking IDs
        track_ids = result.boxes.id.int().cpu().tolist()

        # Get class IDs
        classes = result.boxes.cls.int().cpu().tolist()

        # Get bounding boxes
        boxes = result.boxes.xyxy.cpu().tolist()

        # --------------------------------------------------
        # 5. Process each tracked object
        # --------------------------------------------------

        for track_id, class_id, box in zip(
            track_ids,
            classes,
            boxes
        ):

            # Convert class ID to class name
            vehicle_type = model.names[class_id]

            # Ignore people and other non-vehicle objects
            if vehicle_type not in vehicle_classes:
                continue

            # Bounding box coordinates
            x1, y1, x2, y2 = box

            # Calculate center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Save data
            writer.writerow([
                frame_number,
                track_id,
                vehicle_type,
                round(center_x, 2),
                round(center_y, 2)
            ])

# --------------------------------------------------
# 6. Completion message
# --------------------------------------------------

print("Movement analysis completed!")
print("Vehicle movement data saved to vehicle_movement.csv")
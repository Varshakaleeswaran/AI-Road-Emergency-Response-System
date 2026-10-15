import cv2
import pandas as pd
from pathlib import Path

VIDEO = Path("data/ACCIDENT/real_videos/unS0-TLF1ao_00.mp4")
TRACKING = Path("data/ACCIDENT/tracking/unS0-TLF1ao_00_tracking.csv")
MOVEMENT = Path("data/ACCIDENT/tracking/unS0-TLF1ao_00_movement.csv")
DISTANCE = Path("data/ACCIDENT/tracking/unS0-TLF1ao_00_distance.csv")
RELATIVE = Path("data/ACCIDENT/tracking/unS0-TLF1ao_00_relative_motion.csv")
IOU = Path("data/ACCIDENT/tracking/unS0-TLF1ao_00_iou.csv")

OUTPUT_DIR = Path("runs/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "unS0-TLF1ao_00_features.mp4"

tracking = pd.read_csv(TRACKING)
movement = pd.read_csv(MOVEMENT)
distance = pd.read_csv(DISTANCE)
relative = pd.read_csv(RELATIVE)
iou = pd.read_csv(IOU)

print("Tracking columns:", list(tracking.columns))
print("Movement columns:", list(movement.columns))
print("Distance columns:", list(distance.columns))
print("Relative motion columns:", list(relative.columns))
print("IoU columns:", list(iou.columns))

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    str(OUTPUT),
    fourcc,
    fps,
    (width, height)
)

tracking["frame"] = tracking["frame"].astype(int)

if "frame" in movement.columns:
    movement["frame"] = movement["frame"].astype(int)

if "frame" in distance.columns:
    distance["frame"] = distance["frame"].astype(int)

if "frame" in relative.columns:
    relative["frame"] = relative["frame"].astype(int)

if "frame" in iou.columns:
    iou["frame"] = iou["frame"].astype(int)


def get_frame_data(df, frame):
    if "frame" not in df.columns:
        return pd.DataFrame()
    return df[df["frame"] == frame]


frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    current_tracking = get_frame_data(tracking, frame_number)

    current_movement = get_frame_data(movement, frame_number)
    current_distance = get_frame_data(distance, frame_number)
    current_relative = get_frame_data(relative, frame_number)
    current_iou = get_frame_data(iou, frame_number)

    # -------------------------------------------------
    # DRAW VEHICLE BOUNDING BOXES
    # -------------------------------------------------

    for _, row in current_tracking.iterrows():

        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])

        vehicle_id = int(row["vehicle_id"])

        vehicle_type = str(
            row.get("vehicle_type", row.get("class_name", "vehicle"))
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"{vehicle_type} | ID {vehicle_id}"

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        # Center point
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cv2.circle(
            frame,
            (cx, cy),
            4,
            (0, 255, 255),
            -1
        )

    # -------------------------------------------------
    # DISPLAY FRAME NUMBER
    # -------------------------------------------------

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # -------------------------------------------------
    # DISPLAY FEATURE COUNTS
    # -------------------------------------------------

    info_y = 70

    cv2.putText(
        frame,
        f"Vehicles tracked: {len(current_tracking)}",
        (20, info_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    info_y += 25

    cv2.putText(
        frame,
        f"Distance pairs: {len(current_distance)}",
        (20, info_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    info_y += 25

    cv2.putText(
        frame,
        f"Relative-motion pairs: {len(current_relative)}",
        (20, info_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    info_y += 25

    cv2.putText(
        frame,
        f"IoU pairs: {len(current_iou)}",
        (20, info_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # -------------------------------------------------
    # SHOW DISTANCE + RELATIVE MOTION + IoU
    # FOR PAIRS WHEN AVAILABLE
    # -------------------------------------------------

    panel_y = height - 120

    cv2.putText(
        frame,
        "Interaction features",
        (20, panel_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    panel_y += 25

    # Show first few distance interactions
    for _, row in current_distance.head(3).iterrows():

        v1 = int(row["vehicle_1"])
        v2 = int(row["vehicle_2"])

        distance_value = row.get("distance", row.get("min_distance", 0))

        text = f"V{v1}-V{v2}  Distance: {float(distance_value):.2f}"

        cv2.putText(
            frame,
            text,
            (20, panel_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

        panel_y += 22

    # Relative motion
    for _, row in current_relative.head(2).iterrows():

        v1 = int(row["vehicle_1"])
        v2 = int(row["vehicle_2"])

        value = row.get(
            "relative_speed",
            row.get("relative_motion", 0)
        )

        text = f"V{v1}-V{v2}  Relative motion: {float(value):.2f}"

        cv2.putText(
            frame,
            text,
            (20, panel_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

        panel_y += 22

    # IoU
    for _, row in current_iou.head(2).iterrows():

        v1 = int(row["vehicle_1"])
        v2 = int(row["vehicle_2"])

        value = float(row["iou"])

        text = f"V{v1}-V{v2}  IoU: {value:.3f}"

        cv2.putText(
            frame,
            text,
            (20, panel_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

        panel_y += 22

    writer.write(frame)

    if frame_number % 50 == 0:
        print(f"Processed frame {frame_number}")

cap.release()
writer.release()

print()
print("Visualization completed!")
print(f"Output: {OUTPUT}")

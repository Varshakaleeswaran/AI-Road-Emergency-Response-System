from ultralytics import YOLO
import cv2
import os

VIDEO = "data/ACCIDENT/real_videos/YiQToVQInNw_00.mp4"
OUTPUT = "runs/visualizations/YiQToVQInNw_00_full_features.mp4"

# ==========================================
# CCTV CAMERA LOCATION
# ==========================================
CAMERA_LAT = 11.0168
CAMERA_LON = 0.9558

# ==========================================
# PIXEL-TO-GPS DEMONSTRATION SETTINGS
# ==========================================
# Reference point in the video image
REFERENCE_X = 640
REFERENCE_Y = 360

# Approximate coordinate change per pixel
LAT_PER_PIXEL = 0.000001
LON_PER_PIXEL = 0.000001

# ==========================================

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

model = YOLO("yolo26n.pt")

cap = cv2.VideoCapture(VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    OUTPUT,
    fourcc,
    fps,
    (width, height)
)

frame_number = 0

results = model.track(
    source=VIDEO,
    tracker="bytetrack.yaml",
    stream=True,
    persist=True,
    verbose=False
)

for result in results:

    frame = result.orig_img.copy()
    frame_number += 1

    boxes = result.boxes

    if boxes is not None:

        for box in boxes:

            # ==========================================
            # BOUNDING BOX
            # ==========================================
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # ==========================================
            # CONFIDENCE
            # ==========================================
            confidence = float(box.conf[0])

            # ==========================================
            # YOLO CLASS
            # ==========================================
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # ==========================================
            # BYTETRACK ID
            # ==========================================
            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = -1

            # ==========================================
            # VEHICLE CENTER
            # ==========================================
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # ==========================================
            # ESTIMATED VEHICLE GPS
            # ==========================================
            latitude = (
                CAMERA_LAT
                - ((center_y - REFERENCE_Y) * LAT_PER_PIXEL)
            )

            longitude = (
                CAMERA_LON
                + ((center_x - REFERENCE_X) * LON_PER_PIXEL)
            )

            # ==========================================
            # DRAW BOUNDING BOX
            # ==========================================
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ==========================================
            # DRAW CENTER POINT
            # ==========================================
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # ==========================================
            # VEHICLE INFORMATION
            # ==========================================
            text_lines = [
                f"Class: {class_name}",
                f"Confidence: {confidence:.2f}",
                f"ByteTrack ID: {track_id}",
                f"Center: ({center_x}, {center_y})",
                f"Estimated Latitude: {latitude:.6f}",
                f"Estimated Longitude: {longitude:.6f}"
            ]

            text_x = x1
            text_y = max(25, y1 - 110)

            # ==========================================
            # INFORMATION BACKGROUND
            # ==========================================
            box_height = len(text_lines) * 20 + 10

            cv2.rectangle(
                frame,
                (text_x, text_y - 20),
                (text_x + 350, text_y + box_height),
                (0, 0, 0),
                -1
            )

            # ==========================================
            # WRITE INFORMATION
            # ==========================================
            for i, text in enumerate(text_lines):

                cv2.putText(
                    frame,
                    text,
                    (text_x + 5, text_y + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

    # ==========================================
    # CAMERA LOCATION
    # ==========================================
    cv2.rectangle(
        frame,
        (10, 45),
        (330, 105),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "CCTV CAMERA LOCATION",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Lat: {CAMERA_LAT:.6f}",
        (20, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Lon: {CAMERA_LON:.6f}",
        (170, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # ==========================================
    # FRAME NUMBER
    # ==========================================
    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    writer.write(frame)

    if frame_number % 50 == 0:
        print(f"Processed frame {frame_number}")

writer.release()
cap.release()

print()
print("Visualization completed!")
print(f"Output: {OUTPUT}")

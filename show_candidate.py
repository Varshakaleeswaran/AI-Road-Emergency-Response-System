import cv2
import pandas as pd

# -----------------------------
# Load existing tracking data
# -----------------------------
df = pd.read_csv("vehicle_tracking_boxes.csv")

# Only show vehicles 26 and 260
df = df[
    (df["frame"] >= 120) &
    (df["frame"] <= 169) &
    (df["vehicle_id"].isin([26, 260]))
]

# -----------------------------
# Open original video
# -----------------------------
video_path = "data/videos/road_video.mp4"

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# -----------------------------
# Create output video
# -----------------------------
output_path = "candidate_26_260_annotated.mp4"

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# Start at frame 120
cap.set(cv2.CAP_PROP_POS_FRAMES, 120)

for frame_number in range(120, 170):

    success, frame = cap.read()

    if not success:
        break

    # Get tracking boxes for this frame
    frame_data = df[df["frame"] == frame_number]

    for _, row in frame_data.iterrows():

        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])

        vehicle_id = int(row["vehicle_id"])

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        # Label
        label = f"Vehicle ID {vehicle_id}"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Show frame number
    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    out.write(frame)

cap.release()
out.release()

print()
print("Annotated candidate video created!")
print("Saved:", output_path)
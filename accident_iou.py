import pandas as pd

INPUT = "data/ACCIDENT/tracking/unS0-TLF1ao_00_tracking.csv"
OUTPUT = "data/ACCIDENT/tracking/unS0-TLF1ao_00_iou.csv"

df = pd.read_csv(INPUT)

rows = []

def calculate_iou(a, b):

    # Intersection coordinates
    x_left = max(a["x1"], b["x1"])
    y_top = max(a["y1"], b["y1"])
    x_right = min(a["x2"], b["x2"])
    y_bottom = min(a["y2"], b["y2"])

    intersection_width = max(0, x_right - x_left)
    intersection_height = max(0, y_bottom - y_top)

    intersection_area = intersection_width * intersection_height

    # Areas of both boxes
    area_a = max(0, a["x2"] - a["x1"]) * max(0, a["y2"] - a["y1"])
    area_b = max(0, b["x2"] - b["x1"]) * max(0, b["y2"] - b["y1"])

    union_area = area_a + area_b - intersection_area

    if union_area == 0:
        return 0

    return intersection_area / union_area


# Compare vehicles appearing in the same frame
for frame, frame_data in df.groupby("frame"):

    vehicles = frame_data.to_dict("records")

    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):

            v1 = vehicles[i]
            v2 = vehicles[j]

            iou = calculate_iou(v1, v2)

            rows.append({
                "frame": int(frame),
                "vehicle_1": int(v1["vehicle_id"]),
                "vehicle_2": int(v2["vehicle_id"]),
                "iou": iou
            })


result = pd.DataFrame(rows)

result.to_csv(OUTPUT, index=False)

print("\nIoU analysis completed!")
print("Rows:", len(result))
print("Output:", OUTPUT)

print("\nHighest IoU interactions:")
print(
    result.sort_values("iou", ascending=False)
    .head(20)
    .to_string(index=False)
)

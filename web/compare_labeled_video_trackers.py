"""Compare conservative ByteTrack variants at annotated accident frames.

This exploratory positive-frame benchmark does not estimate ID switches or
false-track rates because the current labels do not annotate identities over
time or every visible vehicle.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METADATA = ROOT / "data" / "ACCIDENT" / "starter_25_ground_truth.csv"
DEFAULT_VIDEO_DIR = ROOT / "data" / "ACCIDENT" / "real_videos"
DEFAULT_MODEL = ROOT / "yolo26n.pt"
INFERENCE_SIZE = 640
DETECTOR_CONFIDENCE = 0.10
EVENT_IOU_THRESHOLDS = (0.3, 0.5)
TRACKER_VARIANTS = {
    "baseline": {},
    "low_score_recovery": {"track_low_thresh": 0.05},
    "low_conf_track_start": {"new_track_thresh": 0.15},
    "longer_occlusion_buffer": {"track_buffer": 45},
}


def _box_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _load_tracker_settings() -> dict:
    import ultralytics

    tracker_path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml"
    with tracker_path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if not isinstance(settings, dict) or settings.get("tracker_type") != "bytetrack":
        raise ValueError(f"Unexpected ByteTrack configuration at {tracker_path}")
    return settings


def _video_frames(path: Path, event_frame: int, context_frames: int) -> tuple[list, int, tuple[int, int]]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Unable to open labeled video: {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = max(1, event_frame - context_frames)
        end_frame = min(frame_count, event_frame + context_frames)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
        frames = []
        for _ in range(start_frame, end_frame + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not read the expected event context from {path}")
            frames.append(frame)
        height, width = frames[0].shape[:2]
        return frames, start_frame, (width, height)
    finally:
        capture.release()


def _run_video(
    model: YOLO,
    video_path: Path,
    event_frame: int,
    normalized_ground_truth: tuple[float, float, float, float],
    settings: dict,
    variant: str,
    context_frames: int,
) -> dict:
    frames, start_frame, image_size = _video_frames(video_path, event_frame, context_frames)
    width, height = image_size
    ground_truth = (
        normalized_ground_truth[0] * width,
        normalized_ground_truth[1] * height,
        normalized_ground_truth[2] * width,
        normalized_ground_truth[3] * height,
    )
    if ground_truth[2] <= ground_truth[0] or ground_truth[3] <= ground_truth[1]:
        raise ValueError(f"Invalid annotated bounding box for {video_path.name}")

    with tempfile.TemporaryDirectory(prefix="resqlink-bytetrack-") as temporary:
        tracker_settings = settings | TRACKER_VARIANTS[variant]
        tracker_file = Path(temporary) / "bytetrack.yaml"
        tracker_file.write_text(yaml.safe_dump(tracker_settings, sort_keys=False), encoding="utf-8")
        event_result = None
        frame_results = {}
        for offset, result in enumerate(model.track(
            source=frames,
            tracker=str(tracker_file),
            stream=True,
            persist=False,
            verbose=False,
            imgsz=INFERENCE_SIZE,
            conf=DETECTOR_CONFIDENCE,
            device="cpu",
            classes=[2, 3, 5, 7],
        )):
            frame_number = start_frame + offset
            boxes = result.boxes
            coordinates = boxes.xyxy.cpu().numpy() if boxes is not None and len(boxes) else []
            identities = (
                boxes.id.cpu().numpy().astype(int).tolist()
                if boxes is not None and boxes.id is not None
                else [None] * len(coordinates)
            )
            confidences = (
                boxes.conf.cpu().numpy().tolist()
                if boxes is not None and len(boxes)
                else []
            )
            frame_detections = []
            for coordinate, identity, confidence in zip(coordinates, identities, confidences):
                box = tuple(float(value) for value in coordinate)
                frame_detections.append({
                    "box": box,
                    "track_id": identity,
                    "confidence": float(confidence),
                    "iou": _box_iou(ground_truth, box) if frame_number == event_frame else None,
                })
            frame_results[frame_number] = frame_detections
            if frame_number == event_frame:
                event_result = frame_detections

    if event_result is None:
        raise ValueError(f"Annotated event frame {event_frame} was not processed for {video_path.name}")

    best_detection = max(event_result, key=lambda row: row["iou"], default=None)
    best_tracked_detection = max(
        (row for row in event_result if row["track_id"] is not None),
        key=lambda row: row["iou"],
        default=None,
    )
    event_track_id = None
    if best_tracked_detection and best_tracked_detection["iou"] >= min(EVENT_IOU_THRESHOLDS):
        event_track_id = best_tracked_detection["track_id"]

    before_observations = after_observations = 0
    if event_track_id is not None:
        before_observations = sum(
            any(row["track_id"] == event_track_id for row in frame_results.get(frame, []))
            for frame in range(max(start_frame, event_frame - 10), event_frame)
        )
        after_observations = sum(
            any(row["track_id"] == event_track_id for row in frame_results.get(frame, []))
            for frame in range(event_frame + 1, min(start_frame + len(frames), event_frame + 11))
        )

    return {
        "variant": variant,
        "video": video_path.stem,
        "event_frame": event_frame,
        "context_start_frame": start_frame,
        "context_end_frame": start_frame + len(frames) - 1,
        "target_box_width_px": ground_truth[2] - ground_truth[0],
        "target_box_height_px": ground_truth[3] - ground_truth[1],
        "target_box_area_px": (ground_truth[2] - ground_truth[0]) * (ground_truth[3] - ground_truth[1]),
        "best_event_detection_iou": best_detection["iou"] if best_detection else 0.0,
        "best_event_detection_confidence": best_detection["confidence"] if best_detection else None,
        "best_event_tracked_iou": best_tracked_detection["iou"] if best_tracked_detection else 0.0,
        "event_target_assigned_track_id": event_track_id is not None,
        "target_track_observed_in_previous_10_frames": before_observations,
        "target_track_observed_in_next_10_frames": after_observations,
        "unique_track_ids_in_context": len({
            row["track_id"]
            for detections in frame_results.values()
            for row in detections
            if row["track_id"] is not None
        }),
        "tracked_observations_in_context": sum(
            row["track_id"] is not None
            for detections in frame_results.values()
            for row in detections
        ),
        "frames_in_context": len(frames),
        "image_width": width,
        "image_height": height,
    }


def _summarize(records: list[dict]) -> dict:
    summaries = {}
    for variant in TRACKER_VARIANTS:
        rows = [row for row in records if row["variant"] == variant]
        if not rows:
            continue
        summaries[variant] = {
            "labeled_videos": len(rows),
            "event_frame_detection_iou_at_least_0_3": sum(
                row["best_event_detection_iou"] >= 0.3 for row in rows
            ),
            "event_frame_detection_iou_at_least_0_5": sum(
                row["best_event_detection_iou"] >= 0.5 for row in rows
            ),
            "event_frame_target_assigned_track_id": sum(
                row["event_target_assigned_track_id"] for row in rows
            ),
            "mean_best_event_detection_iou": sum(
                row["best_event_detection_iou"] for row in rows
            ) / len(rows),
            "mean_best_event_tracked_iou": sum(
                row["best_event_tracked_iou"] for row in rows
            ) / len(rows),
            "mean_event_detection_confidence": (
                sum(row["best_event_detection_confidence"] for row in rows
                    if row["best_event_detection_confidence"] is not None)
                / sum(row["best_event_detection_confidence"] is not None for row in rows)
                if any(row["best_event_detection_confidence"] is not None for row in rows)
                else None
            ),
            "mean_target_track_observations_previous_10_frames": sum(
                row["target_track_observed_in_previous_10_frames"] for row in rows
            ) / len(rows),
            "mean_target_track_observations_next_10_frames": sum(
                row["target_track_observed_in_next_10_frames"] for row in rows
            ) / len(rows),
            "mean_unique_track_ids_in_context": sum(
                row["unique_track_ids_in_context"] for row in rows
            ) / len(rows),
            "note": (
                "Positive-frame target association only. Context ID observations are an unverified continuity "
                "proxy; no all-object GT exists to measure false tracks, ID switches, or fragmentation."
            ),
        }

    for variant, summary in summaries.items():
        for group_name, predicate in (
            ("smallest_target_box_quartile", lambda row: row["target_box_area_px"] <= _quantile(records, 0.25)),
            ("largest_target_box_quartile", lambda row: row["target_box_area_px"] >= _quantile(records, 0.75)),
            ("poor_or_very_poor_labeled_quality", lambda row: row["quality"] in {"poor", "very_poor"}),
        ):
            subset = [row for row in records if row["variant"] == variant and predicate(row)]
            summary[group_name] = {
                "labeled_videos": len(subset),
                "event_frame_target_assigned_track_id": sum(row["event_target_assigned_track_id"] for row in subset),
            }
    return summaries


def _quantile(records: list[dict], fraction: float) -> float:
    values = sorted({row["target_box_area_px"] for row in records})
    if not values:
        return 0.0
    index = round((len(values) - 1) * fraction)
    return values[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--context-frames", type=int, default=50)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "ACCIDENT" / "tracker_config_event_frame_comparison.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "data" / "ACCIDENT" / "tracker_config_event_frame_comparison.json")
    args = parser.parse_args()
    if args.context_frames < 10:
        parser.error("--context-frames must be at least 10 to measure the local continuity proxy.")
    if not args.metadata.is_file():
        parser.error(f"Metadata file not found: {args.metadata}")
    if not args.model.is_file():
        parser.error(f"Model weights not found: {args.model}")

    metadata = pd.read_csv(args.metadata)
    required = {
        "path", "accident_frame_starter", "x1_starter", "y1_starter", "x2_starter", "y2_starter",
        "quality_starter",
    }
    missing = required.difference(metadata.columns)
    if missing:
        parser.error(f"Metadata is missing columns: {', '.join(sorted(missing))}")
    if metadata["path"].duplicated().any():
        parser.error("Metadata contains duplicate video paths.")

    settings = _load_tracker_settings()
    records = []
    for variant in TRACKER_VARIANTS:
        model = YOLO(str(args.model))
        for row in metadata.to_dict("records"):
            video_name = Path(row["path"]).stem
            video_path = args.video_dir / f"{video_name}.mp4"
            if not video_path.is_file():
                raise FileNotFoundError(f"Labeled video not found: {video_path}")
            event_frame = int(row["accident_frame_starter"])
            normalized_box = tuple(float(row[column]) for column in (
                "x1_starter", "y1_starter", "x2_starter", "y2_starter",
            ))
            box_width = (normalized_box[2] - normalized_box[0]) * int(row["width"])
            box_height = (normalized_box[3] - normalized_box[1]) * int(row["height"])
            print(f"Running {variant} on {video_name} around frame {event_frame}", flush=True)
            record = _run_video(
                model,
                video_path,
                event_frame,
                normalized_box,
                settings,
                variant,
                args.context_frames,
            )
            record["quality"] = str(row["quality_starter"]).strip().lower().replace("_", " ")
            record["metadata_target_box_area_px"] = box_width * box_height
            records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "inference_imgsz": INFERENCE_SIZE,
        "detector_confidence": DETECTOR_CONFIDENCE,
        "context_frames_each_side": args.context_frames,
        "tracker_parameter_variants": TRACKER_VARIANTS,
        "results": _summarize(records),
        "limitation": (
            "The labels contain one target bounding box per accident event frame, not exhaustive boxes or "
            "persistent GT identities. This comparison cannot measure precision, ID switches, or fragmentation "
            "and must not be used alone to select production tracker settings."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

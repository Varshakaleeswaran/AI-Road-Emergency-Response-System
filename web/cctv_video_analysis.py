"""Run the existing YOLO/ByteTrack pipeline and summarize video evidence."""

from __future__ import annotations

import csv
import importlib
import math
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
TRACKING_SCRIPT = BASE_DIR / "track_vehicles.py"


def _read_tracking_summary(path: Path, fps: float) -> dict:
    points_by_track = defaultdict(list)
    observed_frames = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            track_id = int(row["vehicle_id"])
            center = (float(row["center_x"]), float(row["center_y"]))
            observed_frames.add(frame)
            points_by_track[track_id].append((frame, center))

    path_lengths = []
    maximum_speed = None
    for points in points_by_track.values():
        points.sort(key=lambda item: item[0])
        path_length = 0.0
        for (previous_frame, previous), (frame, current) in zip(points, points[1:]):
            distance = math.dist(previous, current)
            path_length += distance
            frame_delta = frame - previous_frame
            if fps > 0 and frame_delta > 0:
                speed = distance * fps / frame_delta
                maximum_speed = speed if maximum_speed is None else max(maximum_speed, speed)
        path_lengths.append(path_length)

    return {
        "tracked_observations": sum(len(points) for points in points_by_track.values()),
        "tracked_frames": len(observed_frames),
        "unique_tracks": len(points_by_track),
        "mean_track_path_length_pixels": (
            round(sum(path_lengths) / len(path_lengths), 2) if path_lengths else None
        ),
        "maximum_track_speed_pixels_per_second": (
            round(maximum_speed, 2) if maximum_speed is not None else None
        ),
    }


def _read_detection_summary(path: Path) -> dict:
    classes = Counter()
    confidences = []
    frames = set()
    count = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            count += 1
            classes[row["vehicle_type"]] += 1
            frames.add(int(row["frame"]))
            confidence = float(row["confidence"])
            if math.isfinite(confidence):
                confidences.append(confidence)

    return {
        "vehicle_detection_observations": count,
        "detection_frames": len(frames),
        "vehicle_types": dict(sorted(classes.items())),
        "mean_vehicle_detector_confidence": (
            round(sum(confidences) / len(confidences), 4) if confidences else None
        ),
    }


def summarize_tracking_outputs(tracking_csv: Path, detections_csv: Path, fps: float) -> dict:
    """Summarize detector and track CSVs without assigning accident labels."""
    return {
        "pipeline": "YOLO vehicle detection + ByteTrack",
        "decision_scope": "vehicle evidence only; accident classification not performed",
        "fps": round(fps, 3) if math.isfinite(fps) and fps > 0 else None,
        "tracking": _read_tracking_summary(tracking_csv, fps),
        "detections": _read_detection_summary(detections_csv),
        "incident_created": False,
        "dispatch_triggered": False,
    }


def analyze_video_evidence(video_path: Path, model_path: Path, timeout_seconds: int = 1800) -> dict:
    """Run the repository's YOLO/ByteTrack script and return review evidence."""
    video_path = Path(video_path).resolve()
    model_path = Path(model_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Uploaded video not found: {video_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model weights not found: {model_path}")
    if not TRACKING_SCRIPT.is_file():
        raise FileNotFoundError(f"Vehicle tracking script not found: {TRACKING_SCRIPT}")

    try:
        cv2 = importlib.import_module("cv2")
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is unavailable. Install the optional dependencies with "
            "'pip install -r requirements-ai.txt' from the web directory."
        ) from error

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError("The uploaded file could not be opened as a video.")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()

    with tempfile.TemporaryDirectory(prefix="resqlink-cv-") as temporary:
        output_dir = Path(temporary)
        tracking_csv = output_dir / "tracking.csv"
        detections_csv = output_dir / "detections.csv"
        command = [
            sys.executable,
            str(TRACKING_SCRIPT),
            "--input",
            str(video_path),
            "--model",
            str(model_path),
            "--output",
            str(tracking_csv),
            "--detections-output",
            str(detections_csv),
        ]
        try:
            process = subprocess.run(
                command,
                cwd=BASE_DIR,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(
                f"Vehicle analysis exceeded the {timeout_seconds}-second limit."
            ) from error
        if process.returncode:
            detail = (process.stderr or process.stdout or "No worker output was reported.").strip()
            raise RuntimeError(f"YOLO/ByteTrack worker failed: {detail[-2000:]}")

        summary = summarize_tracking_outputs(tracking_csv, detections_csv, fps)
        summary["video"] = {
            "frame_count": frame_count if frame_count > 0 else None,
            "duration_seconds": (
                round(frame_count / fps, 2) if frame_count > 0 and fps > 0 else None
            ),
        }
        summary["model"] = model_path.name
        return summary

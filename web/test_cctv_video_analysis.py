import csv
import tempfile
import unittest
from pathlib import Path

from cctv_video_analysis import analyze_video_evidence, summarize_tracking_outputs


class TestCCTVVideoAnalysis(unittest.TestCase):
    def test_tracking_summary_reports_motion_without_accident_classification(self):
        with tempfile.TemporaryDirectory() as temporary:
            tracking_csv = Path(temporary) / "tracking.csv"
            detections_csv = Path(temporary) / "detections.csv"
            with tracking_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "frame", "vehicle_id", "vehicle_type", "x1", "y1", "x2",
                    "y2", "center_x", "center_y",
                ])
                writer.writerows([
                    (1, 4, "car", 0, 0, 2, 2, 1, 1),
                    (2, 4, "car", 3, 4, 5, 6, 4, 5),
                    (4, 4, "car", 9, 12, 11, 14, 10, 13),
                    (1, 8, "bus", 10, 0, 12, 2, 11, 1),
                ])
            with detections_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "frame", "vehicle_type", "confidence", "x1", "y1", "x2",
                    "y2", "center_x", "center_y",
                ])
                writer.writerows([
                    (1, "car", 0.8, 0, 0, 2, 2, 1, 1),
                    (1, "bus", 0.6, 10, 0, 12, 2, 11, 1),
                ])

            summary = summarize_tracking_outputs(tracking_csv, detections_csv, 10.0)

        self.assertEqual(summary["tracking"]["unique_tracks"], 2)
        self.assertEqual(summary["tracking"]["tracked_frames"], 3)
        self.assertEqual(summary["tracking"]["mean_track_path_length_pixels"], 7.5)
        self.assertEqual(
            summary["tracking"]["maximum_track_speed_pixels_per_second"], 50.0
        )
        self.assertEqual(summary["detections"]["vehicle_detection_observations"], 2)
        self.assertEqual(summary["detections"]["detection_frames"], 1)
        self.assertEqual(summary["detections"]["vehicle_types"], {"bus": 1, "car": 1})
        self.assertEqual(
            summary["decision_scope"],
            "vehicle evidence only; accident classification not performed",
        )
        self.assertFalse(summary["incident_created"])
        self.assertFalse(summary["dispatch_triggered"])

    def test_missing_video_is_reported_before_model_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing_video = Path(temporary) / "missing.mp4"
            model = Path(temporary) / "model.pt"
            model.touch()

            with self.assertRaisesRegex(FileNotFoundError, "Uploaded video not found"):
                analyze_video_evidence(missing_video, model)

    def test_missing_model_is_reported_before_video_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "video.mp4"
            video.touch()
            missing_model = Path(temporary) / "missing.pt"

            with self.assertRaisesRegex(FileNotFoundError, "YOLO model weights not found"):
                analyze_video_evidence(video, missing_model)

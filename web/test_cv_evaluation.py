import unittest

from cv_evaluation import evaluate_boxes, evaluate_events, evaluate_tracks


def box(frame, track_id="", vehicle_class="car", x1=0, y1=0, x2=10, y2=10):
    return {
        "video": "clip",
        "frame": frame,
        "class": vehicle_class,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "track_id": track_id,
        "confidence": 0.8,
    }


class DetectionEvaluationTests(unittest.TestCase):
    def test_class_errors_count_as_both_false_positive_and_miss(self):
        result = evaluate_boxes(
            [box(1, vehicle_class="car"), box(2, x1=20, x2=30)],
            [box(1, vehicle_class="truck"), box(2, x1=40, x2=50)],
        )

        self.assertEqual(result["geometric_matches"], 1)
        self.assertEqual(result["class_errors_among_matches"], 1)
        self.assertEqual(result["true_positive_class_matches"], 0)
        self.assertEqual(result["false_positive_boxes"], 2)
        self.assertEqual(result["missed_ground_truth_boxes"], 2)
        self.assertEqual(result["overlapping_prediction_pairs_at_threshold"], 0)


class TrackingEvaluationTests(unittest.TestCase):
    def test_counts_adjacent_id_switch_and_fragment_after_miss(self):
        ground_truth = [box(frame, track_id="vehicle-1") for frame in range(1, 5)]
        predictions = [
            box(1, track_id="track-a"),
            box(2, track_id="track-b"),
            box(4, track_id="track-c"),
        ]

        result = evaluate_tracks(ground_truth, predictions)

        self.assertEqual(result["id_switches_between_adjacent_matched_frames"], 1)
        self.assertEqual(result["track_fragments_after_missed_observations"], 1)
        self.assertEqual(result["matched_track_observations"], 3)
        self.assertEqual(result["ground_truth_tracks"], 1)


class EventEvaluationTests(unittest.TestCase):
    def test_false_alarm_rate_uses_verified_negative_exposure(self):
        ground_truth = [{"video": "clip", "start_frame": 10, "end_frame": 11}]
        predictions = [
            {"video": "clip", "start_frame": 10, "end_frame": 10},
            {"video": "clip", "start_frame": 10, "end_frame": 11, "decision_frames": [5, 12, 13]},
            {"video": "clip", "start_frame": 20, "end_frame": 21},
        ]
        ground_truth[0]["event_frame"] = 10
        manifest = [{
            "video": "clip",
            "frame_count": "3600",
            "fps": "1",
            "event_free_verified": "true",
            "verified_by": "reviewer",
            "verified_at": "2026-10-13T00:00:00Z",
        }]

        result = evaluate_events(ground_truth, predictions, manifest)

        self.assertEqual(result["true_positive_events"], 1)
        self.assertEqual(result["false_positive_event_intervals"], 2)
        self.assertEqual(result["missed_events"], 0)
        self.assertAlmostEqual(result["false_alarms_per_verified_event_free_hour"], 1 / (3598 / 3600))
        self.assertEqual(result["detection_delay_seconds"]["matched_events_with_timing"], 1)
        self.assertEqual(result["detection_delay_seconds"]["mean"], 2.0)
        self.assertEqual(result["detection_delay_seconds"]["matched_episodes_started_before_impact"], 1)

    def test_rate_is_unavailable_without_verified_negative_hours(self):
        result = evaluate_events([], [], [{
            "video": "clip",
            "frame_count": "3600",
            "fps": "1",
            "event_free_verified": "false",
        }])

        self.assertIsNone(result["false_alarms_per_verified_event_free_hour"])
        self.assertIsNone(result["verified_event_free_hours"])

    def test_false_alarm_rate_uses_only_reviewed_intervals(self):
        ground_truth = [{"video": "clip", "start_frame": 150, "end_frame": 159}]
        predictions = [
            {"video": "clip", "start_frame": 110, "end_frame": 111},
            {"video": "clip", "start_frame": 700, "end_frame": 720},
            {"video": "clip", "start_frame": 150, "end_frame": 151},
        ]
        manifest = [{
            "video": "clip",
            "frame_count": "1000",
            "fps": "10",
            "event_free_verified": "true",
            "event_free_intervals": "100-199;301-500",
            "verified_by": "reviewer",
            "verified_at": "2026-10-13T00:00:00Z",
        }]

        result = evaluate_events(ground_truth, predictions, manifest)

        self.assertAlmostEqual(result["verified_event_free_hours"], 290 / 10 / 3600)
        self.assertAlmostEqual(result["false_alarms_per_verified_event_free_hour"], 1 / (290 / 10 / 3600))

    def test_verified_exposure_requires_reviewer_and_time(self):
        with self.assertRaisesRegex(ValueError, "verified_by and verified_at"):
            evaluate_events([], [], [{
                "video": "clip",
                "frame_count": "100",
                "fps": "10",
                "event_free_verified": "true",
            }])


if __name__ == "__main__":
    unittest.main()

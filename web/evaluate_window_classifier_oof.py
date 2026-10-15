"""Audit saved grouped-CV classifier outputs at window and event-episode levels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2

from cv_evaluation import evaluate_events


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ACCIDENT"
DEFAULT_PREDICTIONS = DATA / "window_fusion_event_predictions.csv"
DEFAULT_FOLDS = DATA / "window_fusion_event_fold_results.csv"
DEFAULT_DATASET = DATA / "accident_event_window_fusion_dataset_clean.csv"
DEFAULT_METADATA = DATA / "starter_25_ground_truth.csv"
DEFAULT_NORMAL_MANIFEST = DATA / "normal_traffic_evaluation_manifest.csv"
DEFAULT_VIDEO_DIR = DATA / "real_videos"
DEFAULT_NORMAL_VIDEO_DIR = ROOT / "data" / "videos"
DEFAULT_OUTPUT = DATA / "window_classifier_oof_event_audit.json"
DEFAULT_EPISODES = DATA / "window_classifier_oof_alarm_episodes.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def _video_info(path: Path) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frame_count < 1 or not fps > 0:
        raise ValueError(f"Could not read valid frame metadata from {path}")
    return frame_count, fps


def _positive_window_episodes(rows: list[dict[str, str]], model: str) -> list[dict]:
    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["model"] != model or int(row["y_pred"]) != 1:
            continue
        start = int(row["window_start"])
        end = int(row["window_end"])
        if start < 0 or end < start:
            raise ValueError(f"Invalid window frame bounds for {row['video']}: {start}-{end}")
        by_video[row["video"]].append({
            "video": row["video"],
            "start_frame": start,
            "end_frame": end,
            "decision_frame": end,
            "fold": int(row["fold"]),
            "window_count": 1,
            "true_event_labels": {row["true_event_label"]},
        })

    episodes = []
    for windows in by_video.values():
        windows.sort(key=lambda item: item["start_frame"])
        merged = []
        for window in windows:
            if merged and window["start_frame"] <= merged[-1]["end_frame"] + 1:
                merged[-1]["end_frame"] = max(merged[-1]["end_frame"], window["end_frame"])
                merged[-1]["window_count"] += 1
                merged[-1]["decision_frames"].append(window["decision_frame"])
                merged[-1]["true_event_labels"].update(window["true_event_labels"])
            else:
                episode = window.copy()
                episode["decision_frames"] = [window["decision_frame"]]
                merged.append(episode)
        episodes.extend(merged)
    return episodes


def _video_metrics(rows: list[dict[str, str]], model: str) -> dict:
    selected = [row for row in rows if row["model"] == model]
    tn = sum(int(row["y_true"]) == 0 and int(row["y_pred"]) == 0 for row in selected)
    fp = sum(int(row["y_true"]) == 0 and int(row["y_pred"]) == 1 for row in selected)
    fn = sum(int(row["y_true"]) == 1 and int(row["y_pred"]) == 0 for row in selected)
    tp = sum(int(row["y_true"]) == 1 and int(row["y_pred"]) == 1 for row in selected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "windows": len(selected),
        "videos": len({row["video"] for row in selected}),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "pooled_window_precision": precision,
        "pooled_window_recall": recall,
        "pooled_window_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--fold-results", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--normal-manifest", type=Path, default=DEFAULT_NORMAL_MANIFEST)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--normal-video-dir", type=Path, default=DEFAULT_NORMAL_VIDEO_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes-output", type=Path, default=DEFAULT_EPISODES)
    args = parser.parse_args()

    predictions = _read_csv(args.predictions)
    folds = _read_csv(args.fold_results)
    dataset = _read_csv(args.dataset)
    metadata = _read_csv(args.metadata)
    normal_manifest = _read_csv(args.normal_manifest)
    models = sorted({row["model"] for row in predictions})

    ground_truth_events = []
    all_manifest = []
    quality_by_video = {}
    for row in metadata:
        video = Path(row["path"]).stem
        quality_by_video[video] = str(row["quality_starter"]).strip().lower().replace("_", " ")
        video_path = args.video_dir / f"{video}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"Accident video not found: {video_path}")
        event_frame = int(row["accident_frame_starter"])
        ground_truth_events.append({
            "video": video,
            "start_frame": event_frame,
            "end_frame": event_frame,
            "event_frame": event_frame,
        })
        capture = cv2.VideoCapture(str(video_path))
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        if frame_count < 1 or fps <= 0:
            raise ValueError(f"Invalid frame metadata for {video_path}")
        all_manifest.append({
            "video": video,
            "frame_count": frame_count,
            "fps": fps,
            "event_free_verified": "false",
        })

    feature_windows_by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in dataset:
        feature_windows_by_video[row["video"]].append(row)
    impact_coverage = {
        "ground_truth_impact_frames": len(ground_truth_events),
        "impact_frames_covered_by_any_feature_window": 0,
        "impact_frames_labeled_accident_event": 0,
        "impacts_without_feature_window": [],
        "impacts_covered_but_not_labeled_accident_event": [],
    }
    for event in ground_truth_events:
        covering_windows = [
            row for row in feature_windows_by_video.get(event["video"], [])
            if int(row["window_start"]) <= event["event_frame"] <= int(row["window_end"])
        ]
        if covering_windows:
            impact_coverage["impact_frames_covered_by_any_feature_window"] += 1
            if any(row["event_label"] == "accident_event" for row in covering_windows):
                impact_coverage["impact_frames_labeled_accident_event"] += 1
            else:
                impact_coverage["impacts_covered_but_not_labeled_accident_event"].append({
                    "video": event["video"],
                    "event_frame": event["event_frame"],
                    "covering_labels": sorted({row["event_label"] for row in covering_windows}),
                })
        else:
            impact_coverage["impacts_without_feature_window"].append({
                "video": event["video"],
                "event_frame": event["event_frame"],
                "feature_rows_for_video": len(feature_windows_by_video.get(event["video"], [])),
            })

    normal_videos = {row["video"] for row in normal_manifest}
    for row in normal_manifest:
        video = row["video"]
        video_path = args.normal_video_dir / f"{video}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"Normal-traffic video not found: {video_path}")
        frame_count = int(row["frame_count"])
        fps = float(row["fps"])
        actual_frames, actual_fps = _video_info(video_path)
        if frame_count != actual_frames or abs(fps - actual_fps) > 0.01:
            raise ValueError(
                f"Manifest metadata does not match {video_path}: "
                f"{frame_count} frames at {fps} fps vs {actual_frames} frames at {actual_fps} fps"
            )
        all_manifest.append({
            "video": video,
            "frame_count": frame_count,
            "fps": fps,
            "event_free_verified": row["event_free_verified"],
        })

    missing_event_videos = {event["video"] for event in ground_truth_events}.difference(
        {row["video"] for row in all_manifest}
    )
    if missing_event_videos:
        raise ValueError(f"Missing manifest entries for accident clips: {', '.join(sorted(missing_event_videos))}")

    model_results = {}
    episode_output_rows = []
    for model in models:
        episodes = _positive_window_episodes(predictions, model)
        event_result = evaluate_events(ground_truth_events, episodes, all_manifest)
        normal_episodes = [episode for episode in episodes if episode["video"] in normal_videos]
        normal_metrics = evaluate_events([], normal_episodes, normal_manifest)
        quality_results = {}
        for group_name, qualities in (
            ("clear_metadata_quality", {"excellent", "fine", "good"}),
            ("difficult_metadata_quality", {"poor", "very poor"}),
        ):
            selected_videos = {
                video for video, quality in quality_by_video.items()
                if quality in qualities
            }
            selected_gt = [event for event in ground_truth_events if event["video"] in selected_videos]
            selected_episodes = [episode for episode in episodes if episode["video"] in selected_videos]
            selected_manifest = [row for row in all_manifest if row["video"] in selected_videos]
            quality_results[group_name] = {
                "videos": len(selected_videos),
                "metrics": evaluate_events(selected_gt, selected_episodes, selected_manifest),
            }
        normal_recorded_hours = sum(
            int(row["frame_count"]) / float(row["fps"]) / 3600
            for row in normal_manifest
        )
        candidate_by_video = defaultdict(int)
        for episode in normal_episodes:
            candidate_by_video[episode["video"]] += 1

        model_results[model] = {
            "window_level": _video_metrics(predictions, model),
            "fold_results": [row for row in folds if row["model"] == model],
            "point_event_level_all_scored_videos": event_result,
            "point_event_metrics_by_metadata_quality": quality_results,
            "currently_normal_labeled_clips": {
                "candidate_alarm_episodes": len(normal_episodes),
                "episodes_per_recorded_hour_not_false_alarms_per_hour": (
                    len(normal_episodes) / normal_recorded_hours if normal_recorded_hours else None
                ),
                "episodes_by_video": dict(candidate_by_video),
                "verified_event_free_hours": normal_metrics["verified_event_free_hours"],
                "false_alarms_per_verified_event_free_hour": normal_metrics[
                    "false_alarms_per_verified_event_free_hour"
                ],
            },
        }
        for episode in episodes:
            episode_output_rows.append({
                "model": model,
                "video": episode["video"],
                "start_frame": episode["start_frame"],
                "end_frame": episode["end_frame"],
                "first_decision_frame": episode["decision_frame"],
                "decision_frames": "|".join(map(str, episode["decision_frames"])),
                "positive_window_count": episode["window_count"],
                "fold": episode["fold"],
                "contains_ground_truth_impact_frame": any(
                    event["video"] == episode["video"]
                    and episode["start_frame"] <= event["event_frame"] <= episode["end_frame"]
                    for event in ground_truth_events
                ),
                "source_window_labels": "|".join(sorted(episode["true_event_labels"])),
            })

    dataset_counts = defaultdict(int)
    for row in dataset:
        dataset_counts[row["event_label"]] += 1
    result = {
        "source_dataset": str(args.dataset),
        "source_prediction_file": str(args.predictions),
        "video_grouped_cv_preserved": True,
        "dataset": {
            "window_count": len(dataset),
            "video_count": len({row["video"] for row in dataset}),
            "label_counts": dict(dataset_counts),
            "positive_accident_event_window_fraction": (
                dataset_counts["accident_event"] / len(dataset) if dataset else 0.0
            ),
            "impact_frame_feature_coverage": impact_coverage,
        },
        "label_and_feature_audit": [
            "Each positive label is the one fixed window containing the annotated impact frame; other windows from the same accident clip are labeled non_event without independently annotated event-start/end intervals.",
            "Each 50-frame window may span several seconds depending on video fps, so window-end decision times and reported event delay need event-time annotations at the same frame indexing.",
            "Current features are pixel-space movement/distance summaries across videos with different resolutions and contain within-window pre/middle/post aggregates; scale and timing normalization should be evaluated before retraining.",
            "Positive event windows are sparse relative to normal/non-event windows; fold results should be inspected individually, not only by mean F1.",
        ],
        "model_results": model_results,
        "normal_video_status": (
            "Known file durations are listed, but these clips are not independently verified event-free. "
            "Reported normal-clip alarm counts/rates are screening candidates only; false alarms per hour remain unavailable."
        ),
        "event_scope": (
            "Event intervals are currently represented by the annotated impact frame only. "
            "Point-event recall and delay are not substitutes for full event-interval or operational alert validation."
        ),
        "alert_consolidation": (
            "Only adjacent positive windows are merged for this offline audit. This is an evaluation convention, "
            "not a validated deployed alert policy."
        ),
        "operator_integration": (
            "Predictions remain disconnected from incident creation and dispatch pending held-out event-level "
            "validation with verified normal footage; human verification remains required."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.episodes_output.parent.mkdir(parents=True, exist_ok=True)
    with args.episodes_output.open("w", newline="", encoding="utf-8") as handle:
        fields = list(episode_output_rows[0]) if episode_output_rows else [
            "model", "video", "start_frame", "end_frame", "first_decision_frame",
            "positive_window_count", "fold", "contains_ground_truth_impact_frame", "source_window_labels",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(episode_output_rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

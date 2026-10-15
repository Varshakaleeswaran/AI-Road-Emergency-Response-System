"""Evaluate vehicle detections, tracks, and accident-event intervals against labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


BOX_COLUMNS = ("video", "frame", "class", "x1", "y1", "x2", "y2")
EVENT_COLUMNS = ("video", "start_frame", "end_frame")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def _require_columns(rows: list[dict[str, str]], columns: tuple[str, ...], path: Path) -> None:
    if not rows:
        with Path(path).open(newline="", encoding="utf-8-sig") as handle:
            headers = next(csv.reader(handle), [])
    else:
        headers = list(rows[0])
    missing = set(columns).difference(headers)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")


def _box_rows(path: Path) -> list[dict]:
    rows = _read_csv(path)
    _require_columns(rows, BOX_COLUMNS, path)
    parsed = []
    for line, row in enumerate(rows, start=2):
        try:
            frame = int(row["frame"])
            x1, y1, x2, y2 = (float(row[column]) for column in ("x1", "y1", "x2", "y2"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid box or frame in {path}, row {line}") from error
        if not row["video"] or frame < 1 or x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid video, frame, or box geometry in {path}, row {line}")
        parsed.append({
            **row,
            "video": row["video"],
            "frame": frame,
            "class": row["class"].strip().lower(),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "track_id": row.get("track_id", "").strip(),
        })
        confidence = row.get("confidence", "").strip()
        parsed[-1]["confidence"] = float(confidence) if confidence else None
        if parsed[-1]["confidence"] is not None and not math.isfinite(parsed[-1]["confidence"]):
            raise ValueError(f"Non-finite confidence in {path}, row {line}")
    return parsed


def box_iou(first: dict, second: dict) -> float:
    intersection_width = max(0.0, min(first["x2"], second["x2"]) - max(first["x1"], second["x1"]))
    intersection_height = max(0.0, min(first["y2"], second["y2"]) - max(first["y1"], second["y1"]))
    intersection = intersection_width * intersection_height
    first_area = (first["x2"] - first["x1"]) * (first["y2"] - first["y1"])
    second_area = (second["x2"] - second["x1"]) * (second["y2"] - second["y1"])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _classification_metrics(true_positive: int, false_positive: int, false_negative: int) -> dict:
    precision = _rate(true_positive, true_positive + false_positive)
    recall = _rate(true_positive, true_positive + false_negative)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _rate(2 * precision * recall, precision + recall),
    }


def evaluate_boxes(
    ground_truth: list[dict],
    predictions: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    """Match same-frame boxes by IoU, independent of class, then report errors."""
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be greater than 0 and at most 1.")

    gt_by_frame: dict[tuple[str, int], list[dict]] = defaultdict(list)
    pred_by_frame: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in ground_truth:
        gt_by_frame[(row["video"], row["frame"])].append(row)
    for row in predictions:
        pred_by_frame[(row["video"], row["frame"])].append(row)

    matched_pairs = []
    false_positive_predictions = []
    false_negative_ground_truth = []
    duplicate_prediction_pairs = 0

    for key in sorted(set(gt_by_frame) | set(pred_by_frame)):
        gt_frame = gt_by_frame[key]
        pred_frame = pred_by_frame[key]
        candidates = sorted(
            (
                (box_iou(gt_box, pred_box), gt_index, pred_index)
                for gt_index, gt_box in enumerate(gt_frame)
                for pred_index, pred_box in enumerate(pred_frame)
            ),
            reverse=True,
        )
        used_gt: set[int] = set()
        used_predictions: set[int] = set()
        for overlap, gt_index, pred_index in candidates:
            if overlap < iou_threshold:
                break
            if gt_index in used_gt or pred_index in used_predictions:
                continue
            used_gt.add(gt_index)
            used_predictions.add(pred_index)
            matched_pairs.append((gt_frame[gt_index], pred_frame[pred_index], overlap))

        false_negative_ground_truth.extend(
            row for index, row in enumerate(gt_frame) if index not in used_gt
        )
        false_positive_predictions.extend(
            row for index, row in enumerate(pred_frame) if index not in used_predictions
        )
        duplicate_prediction_pairs += sum(
            box_iou(pred_frame[first], pred_frame[second]) >= iou_threshold
            for first in range(len(pred_frame))
            for second in range(first + 1, len(pred_frame))
        )

    class_errors = sum(gt["class"] != pred["class"] for gt, pred, _ in matched_pairs)
    geometric_matches = len(matched_pairs)
    tp = geometric_matches - class_errors
    fp = len(false_positive_predictions) + class_errors
    fn = len(false_negative_ground_truth) + class_errors
    result = {
        "iou_threshold": iou_threshold,
        "ground_truth_boxes": len(ground_truth),
        "predicted_boxes": len(predictions),
        "geometric_matches": geometric_matches,
        "true_positive_class_matches": tp,
        "false_positive_boxes": fp,
        "missed_ground_truth_boxes": fn,
        "class_errors_among_matches": class_errors,
        "overlapping_prediction_pairs_at_threshold": duplicate_prediction_pairs,
        **_classification_metrics(tp, fp, fn),
        "matched_confidence_mean": (
            sum(pred["confidence"] for _, pred, _ in matched_pairs if pred["confidence"] is not None)
            / sum(pred["confidence"] is not None for _, pred, _ in matched_pairs)
            if any(pred["confidence"] is not None for _, pred, _ in matched_pairs)
            else None
        ),
    }
    result["missed_boxes"] = [
        {"video": row["video"], "frame": row["frame"], "class": row["class"]}
        for row in false_negative_ground_truth
    ]
    result["false_positive_details"] = [
        {
            "video": row["video"],
            "frame": row["frame"],
            "class": row["class"],
            "confidence": row["confidence"],
        }
        for row in false_positive_predictions
    ]
    return result


def evaluate_tracks(ground_truth: list[dict], predictions: list[dict], iou_threshold: float = 0.5) -> dict:
    """Measure annotated-track coverage, adjacent-frame ID switches, and fragments."""
    gt_tracks = {(row["video"], row["track_id"]) for row in ground_truth if row["track_id"]}
    if not ground_truth or not gt_tracks:
        raise ValueError("Tracking evaluation requires ground-truth boxes with track_id values.")
    if any(not row["track_id"] for row in ground_truth):
        raise ValueError("Every ground-truth box must have a track_id for tracking evaluation.")

    gt_by_frame: dict[tuple[str, int], list[dict]] = defaultdict(list)
    pred_by_frame: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in ground_truth:
        if row["track_id"]:
            gt_by_frame[(row["video"], row["frame"])].append(row)
    for row in predictions:
        if row["track_id"]:
            pred_by_frame[(row["video"], row["frame"])].append(row)

    matched_gt_to_pred = {}
    for key, gt_frame in gt_by_frame.items():
        pred_frame = pred_by_frame.get(key, [])
        candidates = sorted(
            (
                (box_iou(gt_box, pred_box), gt_index, pred_index)
                for gt_index, gt_box in enumerate(gt_frame)
                for pred_index, pred_box in enumerate(pred_frame)
            ),
            reverse=True,
        )
        used_gt: set[int] = set()
        used_pred: set[int] = set()
        for overlap, gt_index, pred_index in candidates:
            if overlap < iou_threshold:
                break
            if gt_index in used_gt or pred_index in used_pred:
                continue
            gt_box = gt_frame[gt_index]
            if gt_box["track_id"]:
                matched_gt_to_pred[(key[0], key[1], gt_box["track_id"])] = pred_frame[pred_index]["track_id"]
            used_gt.add(gt_index)
            used_pred.add(pred_index)

    switches = 0
    fragments = 0
    track_observations = 0
    matched_observations = 0
    rows_by_track: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in ground_truth:
        rows_by_track[(row["video"], row["track_id"])].append(row)
    for video, gt_id in sorted(gt_tracks):
        rows = rows_by_track[(video, gt_id)]
        rows.sort(key=lambda row: row["frame"])
        track_observations += len(rows)
        previous_frame = None
        previous_prediction_id = None
        in_matched_segment = False
        has_matched_observation = False
        for row in rows:
            prediction_id = matched_gt_to_pred.get((video, row["frame"], gt_id))
            if prediction_id is None:
                in_matched_segment = False
                continue
            matched_observations += 1
            if in_matched_segment and previous_frame is not None and row["frame"] == previous_frame + 1:
                if prediction_id != previous_prediction_id:
                    switches += 1
            elif not in_matched_segment and has_matched_observation:
                fragments += 1
            previous_frame = row["frame"]
            previous_prediction_id = prediction_id
            in_matched_segment = True
            has_matched_observation = True

    return {
        "iou_threshold": iou_threshold,
        "ground_truth_tracks": len(gt_tracks),
        "ground_truth_track_observations": track_observations,
        "matched_track_observations": matched_observations,
        "track_observation_coverage": _rate(matched_observations, track_observations),
        "id_switches_between_adjacent_matched_frames": switches,
        "track_fragments_after_missed_observations": fragments,
    }


def _event_rows(path: Path) -> list[dict]:
    rows = _read_csv(path)
    _require_columns(rows, EVENT_COLUMNS, path)
    parsed = []
    for line, row in enumerate(rows, start=2):
        try:
            start = int(row["start_frame"])
            end = int(row["end_frame"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid event interval in {path}, row {line}") from error
        if not row["video"] or start < 1 or end < start:
            raise ValueError(f"Invalid video or event interval in {path}, row {line}")
        parsed_row = {"video": row["video"], "start_frame": start, "end_frame": end}
        for optional_frame_column in ("event_frame", "decision_frame"):
            value = row.get(optional_frame_column, "").strip()
            if value:
                try:
                    frame = int(value)
                except ValueError as error:
                    raise ValueError(f"Invalid {optional_frame_column} in {path}, row {line}") from error
                if frame < 1:
                    raise ValueError(f"Invalid {optional_frame_column} in {path}, row {line}")
                parsed_row[optional_frame_column] = frame
        if "event_frame" in parsed_row and not start <= parsed_row["event_frame"] <= end:
            raise ValueError(f"event_frame must fall inside the interval in {path}, row {line}")
        parsed.append(parsed_row)
    return parsed


def _union_frame_count(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    return sum(end - start + 1 for start, end in _merge_frame_intervals(intervals))


def _merge_frame_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _parse_verified_intervals(row: dict[str, str], frame_count: int, line: int) -> list[tuple[int, int]]:
    if row.get("event_free_verified", "").strip().lower() != "true":
        if row.get("event_free_intervals", "").strip():
            raise ValueError(f"event_free_intervals require event_free_verified=true in manifest row {line}")
        return []

    if not row.get("verified_by", "").strip() or not row.get("verified_at", "").strip():
        raise ValueError(f"Verified event-free footage requires verified_by and verified_at in manifest row {line}")

    raw_intervals = row.get("event_free_intervals", "").strip()
    if not raw_intervals:
        return [(1, frame_count)]

    intervals = []
    for raw_interval in raw_intervals.split(";"):
        try:
            start_text, end_text = raw_interval.strip().split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid event_free_intervals value in manifest row {line}") from error
        if start < 1 or end < start or end > frame_count:
            raise ValueError(f"Out-of-range event_free_intervals value in manifest row {line}")
        intervals.append((start, end))
    return intervals


def _overlap_frame_count(first: tuple[int, int], second: tuple[int, int]) -> int:
    return max(0, min(first[1], second[1]) - max(first[0], second[0]) + 1)


def evaluate_events(ground_truth: list[dict], predictions: list[dict], manifest: list[dict] | None = None) -> dict:
    """Match predicted intervals by temporal overlap and compute verified negative exposure."""
    def has_post_event_decision(event: dict, prediction: dict) -> bool:
        if "event_frame" not in event:
            return True
        decisions = prediction.get("decision_frames")
        if decisions is None:
            decision = prediction.get("decision_frame")
            return decision is None or decision >= event["event_frame"]
        return any(frame >= event["event_frame"] for frame in decisions)

    gt_by_video: dict[str, list[dict]] = defaultdict(list)
    pred_by_video: dict[str, list[dict]] = defaultdict(list)
    for row in ground_truth:
        gt_by_video[row["video"]].append(row)
    for row in predictions:
        pred_by_video[row["video"]].append(row)

    matched_gt = set()
    matched_predictions = set()
    matched_pairs = []
    candidates = sorted(
        (
            (
                min(gt["end_frame"], pred["end_frame"]) - max(gt["start_frame"], pred["start_frame"]) + 1,
                video,
                gt_index,
                pred_index,
            )
            for video, gt_rows in gt_by_video.items()
            for gt_index, gt in enumerate(gt_rows)
            for pred_index, pred in enumerate(pred_by_video.get(video, []))
            if min(gt["end_frame"], pred["end_frame"]) >= max(gt["start_frame"], pred["start_frame"])
            and has_post_event_decision(gt, pred)
        ),
        reverse=True,
    )
    for _, video, gt_index, pred_index in candidates:
        gt_key = (video, gt_index)
        pred_key = (video, pred_index)
        if gt_key not in matched_gt and pred_key not in matched_predictions:
            matched_gt.add(gt_key)
            matched_predictions.add(pred_key)
            matched_pairs.append((video, gt_index, pred_index))

    tp = len(matched_predictions)
    fp = len(predictions) - tp
    fn = len(ground_truth) - len(matched_gt)
    false_alarm_rate = None
    verified_negative_hours = None
    fps_by_video = {}
    if manifest is not None:
        verified_hours = 0.0
        manifest_videos = set()
        verified_intervals_by_video = {}
        for line, row in enumerate(manifest, start=2):
            try:
                frame_count = int(row["frame_count"])
                fps = float(row["fps"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid video manifest row {line}") from error
            if not row.get("video") or frame_count < 1 or fps <= 0 or not math.isfinite(fps):
                raise ValueError(f"Invalid video manifest row {line}")
            if row["video"] in manifest_videos:
                raise ValueError(f"Duplicate video in manifest row {line}: {row['video']}")
            manifest_videos.add(row["video"])
            fps_by_video[row["video"]] = fps
            verified_intervals = _parse_verified_intervals(row, frame_count, line)
            verified_intervals_by_video[row["video"]] = verified_intervals
            event_intervals = _merge_frame_intervals([
                (event["start_frame"], event["end_frame"])
                for event in gt_by_video.get(row["video"], [])
            ])
            verified_intervals = _merge_frame_intervals(verified_intervals)
            verified_frames = _union_frame_count(verified_intervals)
            event_frames = sum(
                _overlap_frame_count(verified_interval, event_interval)
                for verified_interval in verified_intervals
                for event_interval in event_intervals
            )
            verified_hours += max(0, verified_frames - event_frames) / fps / 3600
        unlisted = (set(gt_by_video) | set(pred_by_video)).difference(manifest_videos)
        if unlisted:
            raise ValueError(f"Events reference videos absent from manifest: {', '.join(sorted(unlisted))}")
        verified_negative_hours = verified_hours if verified_hours > 0 else None
        if verified_hours > 0:
            false_alarms = 0
            for pred_index, prediction in enumerate(predictions):
                if (prediction["video"], pred_index) in matched_predictions:
                    continue
                verified_intervals = verified_intervals_by_video.get(prediction["video"], [])
                if not any(
                    _overlap_frame_count(
                        (prediction["start_frame"], prediction["end_frame"]),
                        interval,
                    )
                    for interval in verified_intervals
                ):
                    continue
                overlaps_ground_truth = any(
                    min(event["end_frame"], prediction["end_frame"])
                    >= max(event["start_frame"], prediction["start_frame"])
                    for event in gt_by_video.get(prediction["video"], [])
                )
                if not overlaps_ground_truth:
                    false_alarms += 1
            false_alarm_rate = false_alarms / verified_hours

    delays_seconds = []
    episodes_started_before_impact = 0
    for video, gt_index, pred_index in matched_pairs:
        event = ground_truth[gt_index]
        prediction = predictions[pred_index]
        decisions = prediction.get("decision_frames")
        if decisions is None:
            decisions = [prediction["decision_frame"]] if "decision_frame" in prediction else []
        else:
            decisions = list(decisions)
        if (
            video in fps_by_video
            and "event_frame" in event
            and any(frame >= event["event_frame"] for frame in decisions)
        ):
            if decisions and min(decisions) < event["event_frame"]:
                episodes_started_before_impact += 1
            delays_seconds.append(
                (min(frame for frame in decisions if frame >= event["event_frame"]) - event["event_frame"])
                / fps_by_video[video]
            )
    sorted_delays = sorted(delays_seconds)
    median_delay = None
    if sorted_delays:
        middle = len(sorted_delays) // 2
        median_delay = (
            sorted_delays[middle]
            if len(sorted_delays) % 2
            else (sorted_delays[middle - 1] + sorted_delays[middle]) / 2
        )

    return {
        "ground_truth_events": len(ground_truth),
        "predicted_event_intervals": len(predictions),
        "true_positive_events": tp,
        "false_positive_event_intervals": fp,
        "missed_events": fn,
        **_classification_metrics(tp, fp, fn),
        "verified_event_free_hours": verified_negative_hours,
        "false_alarms_per_verified_event_free_hour": false_alarm_rate,
        "detection_delay_seconds": {
            "matched_events_with_timing": len(delays_seconds),
            "mean": sum(delays_seconds) / len(delays_seconds) if delays_seconds else None,
            "median": median_delay,
            "minimum": min(delays_seconds) if delays_seconds else None,
            "maximum": max(delays_seconds) if delays_seconds else None,
            "matched_episodes_started_before_impact": episodes_started_before_impact,
        },
    }


def _read_manifest(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    _require_columns(rows, ("video", "frame_count", "fps", "event_free_verified"), path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-boxes", type=Path)
    parser.add_argument("--predicted-boxes", type=Path)
    parser.add_argument("--ground-truth-events", type=Path)
    parser.add_argument("--predicted-events", type=Path)
    parser.add_argument("--video-manifest", type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True, help="Write evaluation JSON to this path.")
    args = parser.parse_args()

    if bool(args.ground_truth_boxes) != bool(args.predicted_boxes):
        parser.error("--ground-truth-boxes and --predicted-boxes must be supplied together.")
    if bool(args.ground_truth_events) != bool(args.predicted_events):
        parser.error("--ground-truth-events and --predicted-events must be supplied together.")
    if args.video_manifest and not args.ground_truth_events:
        parser.error("--video-manifest requires event CSV inputs.")
    if not args.ground_truth_boxes and not args.ground_truth_events:
        parser.error("Supply box inputs, event inputs, or both.")

    result = {
        "warning": (
            "Metrics describe only the supplied annotations and predictions. "
            "They are not production certification or automatic-dispatch authorization."
        )
    }
    if args.ground_truth_boxes:
        ground_truth = _box_rows(args.ground_truth_boxes)
        predictions = _box_rows(args.predicted_boxes)
        result["detection"] = evaluate_boxes(ground_truth, predictions, args.iou_threshold)
        if any(row["track_id"] for row in ground_truth):
            result["tracking"] = evaluate_tracks(ground_truth, predictions, args.iou_threshold)
    if args.ground_truth_events:
        manifest = _read_manifest(args.video_manifest) if args.video_manifest else None
        result["accident_events"] = evaluate_events(
            _event_rows(args.ground_truth_events),
            _event_rows(args.predicted_events),
            manifest,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("data/ACCIDENT")
TRACKING = BASE / "tracking"

WINDOW_FILE = BASE / "generic_window_features.csv"
OUTPUT = BASE / "window_temporal_collision_features.csv"

windows = pd.read_csv(WINDOW_FILE)

records = []


def safe_read_csv(path):
    """Read a CSV safely; return an empty DataFrame if unavailable."""
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        if df.empty:
            return df

        return df.replace([np.inf, -np.inf], np.nan)

    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def numeric_column(df, column):
    """Return a numeric version of a column, or an empty numeric Series."""
    if df.empty or column not in df.columns:
        return pd.Series(dtype="float64")

    return pd.to_numeric(df[column], errors="coerce")


def mean_col(df, column):
    values = numeric_column(df, column)
    return values.mean() if not values.empty else np.nan


def max_col(df, column):
    values = numeric_column(df, column)
    return values.max() if not values.empty else np.nan


def vehicle_speed(movement_df, vehicle_id, start_frame, end_frame):
    """
    Mean per-frame movement for one vehicle within an inclusive
    frame interval. Units are pixels/frame, not physical speed.
    """
    if movement_df.empty:
        return np.nan

    required = {"frame", "vehicle_id", "movement"}
    if not required.issubset(movement_df.columns):
        return np.nan

    selected = movement_df[
        (movement_df["vehicle_id"] == vehicle_id)
        & (movement_df["frame"] >= start_frame)
        & (movement_df["frame"] <= end_frame)
    ]

    values = pd.to_numeric(selected["movement"], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()

    return float(values.mean()) if not values.empty else np.nan


def distance_for_pair(dist_df, pair, start_frame, end_frame):
    """Mean center-to-center distance for a vehicle pair in a frame range."""
    if dist_df.empty:
        return np.nan

    required = {"frame", "vehicle_1", "vehicle_2", "distance"}
    if not required.issubset(dist_df.columns):
        return np.nan

    selected = dist_df[
        (dist_df["frame"] >= start_frame)
        & (dist_df["frame"] <= end_frame)
        & (dist_df["vehicle_1"] == pair[0])
        & (dist_df["vehicle_2"] == pair[1])
    ]

    values = pd.to_numeric(selected["distance"], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()

    return float(values.mean()) if not values.empty else np.nan


for video, video_windows in windows.groupby("video"):

    print("Processing:", video)

    rel_file = TRACKING / f"{video}_relative_motion.csv"
    movement_file = TRACKING / f"{video}_movement.csv"

    distance_file = TRACKING / f"{video}_distance.csv"
    if not distance_file.exists():
        distance_file = TRACKING / f"{video}_vehicle_distance.csv"

    # Load each feature source once per video.
    rel_all = safe_read_csv(rel_file)
    movement_all = safe_read_csv(movement_file)
    dist_all = safe_read_csv(distance_file)

    if not rel_all.empty:
        if "frame" in rel_all.columns:
            rel_all["frame"] = pd.to_numeric(
                rel_all["frame"], errors="coerce"
            )

        for column in ("vehicle_1", "vehicle_2"):
            if column in rel_all.columns:
                rel_all[column] = pd.to_numeric(
                    rel_all[column], errors="coerce"
                )

        if "relative_speed" in rel_all.columns:
            rel_all["relative_speed"] = pd.to_numeric(
                rel_all["relative_speed"], errors="coerce"
            )

    if not movement_all.empty:
        for column in ("frame", "vehicle_id", "movement"):
            if column in movement_all.columns:
                movement_all[column] = pd.to_numeric(
                    movement_all[column], errors="coerce"
                )

    if not dist_all.empty:
        for column in ("frame", "vehicle_1", "vehicle_2", "distance"):
            if column in dist_all.columns:
                dist_all[column] = pd.to_numeric(
                    dist_all[column], errors="coerce"
                )

    # Preserve all windows, including windows with no relative-motion rows.
    for _, w in video_windows.iterrows():

        start = int(w["window_start"])
        end = int(w["window_end"])
        window_length = end - start + 1

        if window_length <= 0:
            print(
                f"Skipping invalid window for {video}: "
                f"{start}-{end}"
            )
            continue

        # Divide each window into three consecutive sections.
        third = max(1, window_length // 3)

        pre_start = start
        pre_end = min(end, start + third - 1)

        middle_start = pre_end + 1
        middle_end = min(end, start + 2 * third - 1)

        post_start = middle_end + 1
        post_end = end

        rel = pd.DataFrame()

        if (
            not rel_all.empty
            and {"frame", "vehicle_1", "vehicle_2"}.issubset(
                rel_all.columns
            )
        ):
            rel = rel_all[
                (rel_all["frame"] >= start)
                & (rel_all["frame"] <= end)
            ].copy()

        # A window with no relative-motion rows still gets a record.
        if rel.empty:
            records.append({
                "video": video,
                "label": w.get("label", "unknown"),
                "event_label": w.get("event_label", None),
                "window_start": start,
                "window_end": end,
                "vehicle_1": np.nan,
                "vehicle_2": np.nan,
                "pre_relative_speed": np.nan,
                "window_relative_speed": np.nan,
                "post_relative_speed": np.nan,
                "max_relative_speed": np.nan,
                "pre_speed_1": np.nan,
                "pre_speed_2": np.nan,
                "post_speed_1": np.nan,
                "post_speed_2": np.nan,
                "speed_change_1": np.nan,
                "speed_change_2": np.nan,
                "relative_speed_change": np.nan,
                "observations": 0,
                "pre_distance": np.nan,
                "window_distance": np.nan,
                "post_distance": np.nan,
                "distance_reduction": np.nan,
            })
            continue

        # Aggregate each observed vehicle pair independently.
        for pair, g in rel.groupby(["vehicle_1", "vehicle_2"]):

            g = g.sort_values("frame")

            pre = g[
                (g["frame"] >= pre_start)
                & (g["frame"] <= pre_end)
            ]

            middle = g[
                (g["frame"] >= middle_start)
                & (g["frame"] <= middle_end)
            ]

            post = g[
                (g["frame"] >= post_start)
                & (g["frame"] <= post_end)
            ]

            pre_rel = mean_col(pre, "relative_speed")
            middle_rel = mean_col(middle, "relative_speed")
            post_rel = mean_col(post, "relative_speed")

            # Corrected: individual vehicle speeds come from the movement CSV,
            # not nonexistent vx_1/vy_1/vx_2/vy_2 columns in relative motion.
            pre_v1 = vehicle_speed(
                movement_all, pair[0], pre_start, pre_end
            )
            pre_v2 = vehicle_speed(
                movement_all, pair[1], pre_start, pre_end
            )
            post_v1 = vehicle_speed(
                movement_all, pair[0], post_start, post_end
            )
            post_v2 = vehicle_speed(
                movement_all, pair[1], post_start, post_end
            )

            pre_distance = distance_for_pair(
                dist_all, pair, pre_start, pre_end
            )
            middle_distance = distance_for_pair(
                dist_all, pair, middle_start, middle_end
            )
            post_distance = distance_for_pair(
                dist_all, pair, post_start, post_end
            )

            distance_reduction = (
                pre_distance - middle_distance
                if pd.notna(pre_distance)
                and pd.notna(middle_distance)
                else np.nan
            )

            records.append({
                "video": video,
                "label": w.get("label", "unknown"),
                "event_label": w.get("event_label", None),
                "window_start": start,
                "window_end": end,

                "vehicle_1": pair[0],
                "vehicle_2": pair[1],

                "pre_relative_speed": pre_rel,
                "window_relative_speed": middle_rel,
                "post_relative_speed": post_rel,
                "max_relative_speed": max_col(g, "relative_speed"),

                "pre_speed_1": pre_v1,
                "pre_speed_2": pre_v2,
                "post_speed_1": post_v1,
                "post_speed_2": post_v2,

                "speed_change_1": (
                    post_v1 - pre_v1
                    if pd.notna(pre_v1) and pd.notna(post_v1)
                    else np.nan
                ),
                "speed_change_2": (
                    post_v2 - pre_v2
                    if pd.notna(pre_v2) and pd.notna(post_v2)
                    else np.nan
                ),
                "relative_speed_change": (
                    post_rel - pre_rel
                    if pd.notna(pre_rel) and pd.notna(post_rel)
                    else np.nan
                ),

                "observations": len(g),

                "pre_distance": pre_distance,
                "window_distance": middle_distance,
                "post_distance": post_distance,
                "distance_reduction": distance_reduction,
            })


# Build the output dataset.
out = pd.DataFrame(records)

if not out.empty:
    out = out.replace([np.inf, -np.inf], np.nan)

# Do not overwrite the output with an empty dataset if no rows were produced.
if out.empty:
    raise RuntimeError(
        "No temporal feature rows were generated. "
        "The existing output file was not overwritten."
    )

out.to_csv(OUTPUT, index=False)

print("=" * 60)
print("WINDOW TEMPORAL COLLISION DATASET")
print("=" * 60)
print("Rows:", len(out))
print("Videos:", out["video"].nunique())

print("\nLabels:")
if "label" in out.columns:
    print(out["label"].value_counts(dropna=False).to_string())

print("\nUnique windows:", out[
    ["video", "window_start", "window_end"]
].drop_duplicates().shape[0])

print("\nIndividual speed availability:")
for col in ("pre_speed_1", "pre_speed_2", "post_speed_1", "post_speed_2"):
    available = int(out[col].notna().sum())
    print(f"{col}: {available}/{len(out)} non-missing")

print("\nSaved:", OUTPUT)
print("=" * 60)
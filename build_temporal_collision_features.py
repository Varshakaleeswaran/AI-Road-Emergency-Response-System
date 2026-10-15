import pandas as pd
import numpy as np
import os

BASE = "data/ACCIDENT"
GT_FILE = os.path.join(BASE, "classifier_dataset.csv")
OUT_FILE = os.path.join(BASE, "temporal_collision_features.csv")

gt = pd.read_csv(GT_FILE)

records = []

for _, row in gt.iterrows():

    video = row["video"]
    accident_type = row["accident_type"]
    accident_frame = int(row["accident_frame"])

    rel_file = os.path.join(
        BASE, "tracking", f"{video}_relative_motion.csv"
    )

    distance_file = os.path.join(
        BASE, "tracking", f"{video}_distance.csv"
    )

    iou_file = os.path.join(
        BASE, "tracking", f"{video}_iou.csv"
    )

    # ---------------------------------------------------------
    # Read relative motion safely
    # ---------------------------------------------------------

    rel = pd.DataFrame()

    if os.path.exists(rel_file):
        try:
            rel = pd.read_csv(rel_file)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            rel = pd.DataFrame()

    if not rel.empty:

        rel = rel.replace(
            [np.inf, -np.inf],
            np.nan
        )

        rel = rel[
            (rel["frame"] >= accident_frame - 10) &
            (rel["frame"] <= accident_frame + 10)
        ].copy()

    # ---------------------------------------------------------
    # If no relative motion exists, preserve the video
    # ---------------------------------------------------------

    if rel.empty:

        records.append({
            "video": video,
            "accident_type": accident_type,
            "accident_frame": accident_frame,
            "vehicle_1": np.nan,
            "vehicle_2": np.nan,
            "pre_relative_speed": np.nan,
            "collision_relative_speed": np.nan,
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
            "collision_distance": np.nan,
            "post_distance": np.nan,
            "distance_reduction": np.nan,
            "pre_iou": np.nan,
            "collision_iou": np.nan,
            "post_iou": np.nan
        })

        continue

    # ---------------------------------------------------------
    # Process each vehicle pair
    # ---------------------------------------------------------

    for pair, g in rel.groupby(["vehicle_1", "vehicle_2"]):

        g = g.sort_values("frame")

        pre = g[
            (g["frame"] >= accident_frame - 10) &
            (g["frame"] < accident_frame - 2)
        ]

        collision = g[
            (g["frame"] >= accident_frame - 2) &
            (g["frame"] <= accident_frame + 2)
        ]

        post = g[
            (g["frame"] > accident_frame + 2) &
            (g["frame"] <= accident_frame + 10)
        ]

        def mean_col(df, col):
            if col not in df.columns or df.empty:
                return np.nan

            return pd.to_numeric(
                df[col],
                errors="coerce"
            ).mean()

        def max_col(df, col):
            if col not in df.columns or df.empty:
                return np.nan

            return pd.to_numeric(
                df[col],
                errors="coerce"
            ).max()

        def speed(vx, vy):
            if np.isnan(vx) or np.isnan(vy):
                return np.nan

            return np.sqrt(vx ** 2 + vy ** 2)

        # -----------------------------------------------------
        # Relative motion
        # -----------------------------------------------------

        pre_rel = mean_col(
            pre,
            "relative_speed"
        )

        collision_rel = mean_col(
            collision,
            "relative_speed"
        )

        post_rel = mean_col(
            post,
            "relative_speed"
        )

        pre_v1 = speed(
            mean_col(pre, "vx_1"),
            mean_col(pre, "vy_1")
        )

        pre_v2 = speed(
            mean_col(pre, "vx_2"),
            mean_col(pre, "vy_2")
        )

        post_v1 = speed(
            mean_col(post, "vx_1"),
            mean_col(post, "vy_1")
        )

        post_v2 = speed(
            mean_col(post, "vx_2"),
            mean_col(post, "vy_2")
        )

        # -----------------------------------------------------
        # Base record
        # -----------------------------------------------------

        rec = {
            "video": video,
            "accident_type": accident_type,
            "accident_frame": accident_frame,

            "vehicle_1": pair[0],
            "vehicle_2": pair[1],

            "pre_relative_speed": pre_rel,
            "collision_relative_speed": collision_rel,
            "post_relative_speed": post_rel,

            "max_relative_speed": max_col(
                g,
                "relative_speed"
            ),

            "pre_speed_1": pre_v1,
            "pre_speed_2": pre_v2,

            "post_speed_1": post_v1,
            "post_speed_2": post_v2,

            "speed_change_1": (
                post_v1 - pre_v1
                if not np.isnan(pre_v1)
                and not np.isnan(post_v1)
                else np.nan
            ),

            "speed_change_2": (
                post_v2 - pre_v2
                if not np.isnan(pre_v2)
                and not np.isnan(post_v2)
                else np.nan
            ),

            "relative_speed_change": (
                post_rel - pre_rel
                if not np.isnan(pre_rel)
                and not np.isnan(post_rel)
                else np.nan
            ),

            "observations": len(g),

            "pre_distance": np.nan,
            "collision_distance": np.nan,
            "post_distance": np.nan,
            "distance_reduction": np.nan,

            "pre_iou": np.nan,
            "collision_iou": np.nan,
            "post_iou": np.nan
        }

        # -----------------------------------------------------
        # Distance
        # -----------------------------------------------------

        if os.path.exists(distance_file):

            try:
                d = pd.read_csv(distance_file)

                if not d.empty:

                    d = d.replace(
                        [np.inf, -np.inf],
                        np.nan
                    )

                    d = d[
                        (d["frame"] >= accident_frame - 10) &
                        (d["frame"] <= accident_frame + 10)
                    ]

                    d = d[
                        (d["vehicle_1"] == pair[0]) &
                        (d["vehicle_2"] == pair[1])
                    ]

                    if not d.empty:

                        d_pre = d[
                            d["frame"] < accident_frame - 2
                        ]

                        d_col = d[
                            (d["frame"] >= accident_frame - 2) &
                            (d["frame"] <= accident_frame + 2)
                        ]

                        d_post = d[
                            d["frame"] > accident_frame + 2
                        ]

                        pre_distance = pd.to_numeric(
                            d_pre["distance"],
                            errors="coerce"
                        ).mean()

                        collision_distance = pd.to_numeric(
                            d_col["distance"],
                            errors="coerce"
                        ).mean()

                        post_distance = pd.to_numeric(
                            d_post["distance"],
                            errors="coerce"
                        ).mean()

                        rec["pre_distance"] = pre_distance
                        rec["collision_distance"] = (
                            collision_distance
                        )
                        rec["post_distance"] = post_distance

                        if (
                            not np.isnan(pre_distance)
                            and
                            not np.isnan(collision_distance)
                        ):
                            rec["distance_reduction"] = (
                                pre_distance
                                - collision_distance
                            )

            except (
                pd.errors.EmptyDataError,
                pd.errors.ParserError
            ):
                pass

        # -----------------------------------------------------
        # IoU
        # -----------------------------------------------------

        if os.path.exists(iou_file):

            try:
                iou = pd.read_csv(iou_file)

                if not iou.empty:

                    iou = iou.replace(
                        [np.inf, -np.inf],
                        np.nan
                    )

                    iou = iou[
                        (iou["frame"] >= accident_frame - 10) &
                        (iou["frame"] <= accident_frame + 10)
                    ]

                    iou = iou[
                        (iou["vehicle_1"] == pair[0]) &
                        (iou["vehicle_2"] == pair[1])
                    ]

                    if not iou.empty:

                        i_pre = iou[
                            iou["frame"] < accident_frame - 2
                        ]

                        i_col = iou[
                            (iou["frame"] >= accident_frame - 2) &
                            (iou["frame"] <= accident_frame + 2)
                        ]

                        i_post = iou[
                            iou["frame"] > accident_frame + 2
                        ]

                        rec["pre_iou"] = pd.to_numeric(
                            i_pre["iou"],
                            errors="coerce"
                        ).mean()

                        rec["collision_iou"] = pd.to_numeric(
                            i_col["iou"],
                            errors="coerce"
                        ).max()

                        rec["post_iou"] = pd.to_numeric(
                            i_post["iou"],
                            errors="coerce"
                        ).mean()

            except (
                pd.errors.EmptyDataError,
                pd.errors.ParserError
            ):
                pass

        records.append(rec)


# -------------------------------------------------------------
# Create dataset
# -------------------------------------------------------------

out = pd.DataFrame(records)

out = out.replace(
    [np.inf, -np.inf],
    np.nan
)

out.to_csv(
    OUT_FILE,
    index=False
)

# -------------------------------------------------------------
# Report
# -------------------------------------------------------------

print("=" * 60)
print("TEMPORAL COLLISION DATASET")
print("=" * 60)

print("Rows:", len(out))
print("Videos:", out["video"].nunique())

print("\nClasses:")
print(
    out["accident_type"]
    .value_counts()
    .to_string()
)

print("\nVideos with no relative-motion evidence:")

missing_rel = out[
    out["observations"] == 0
]["video"].unique()

if len(missing_rel) == 0:
    print(" None")
else:
    for video in missing_rel:
        print(" ", video)

print("\nColumns:")
print(out.columns.tolist())

print("\nSaved:")
print(OUT_FILE)

print("=" * 60)
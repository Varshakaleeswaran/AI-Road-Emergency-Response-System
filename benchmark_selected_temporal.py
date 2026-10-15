import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, f1_score

BASE = "data/ACCIDENT"
INPUT = f"{BASE}/accident_event_window_fusion_dataset_clean.csv"

d = pd.read_csv(INPUT)

d = d[
    d["event_label"].isin(
        ["accident_event", "non_event"]
    )
].copy()

y = (
    d["event_label"] == "accident_event"
).astype(int)

groups = d["video"]

base = [
    "movement_mean",
    "movement_max",
    "distance_min",
    "distance_mean",
    "relative_speed_mean",
    "relative_speed_max",
    "unique_vehicle_ids"
]

temporal_relative = [
    "temporal_pre_relative_speed_mean",
    "temporal_window_relative_speed_mean",
    "temporal_post_relative_speed_mean",
    "temporal_max_relative_speed_mean",
    "temporal_pre_relative_speed_max",
    "temporal_window_relative_speed_max",
    "temporal_post_relative_speed_max"
]

temporal_distance = [
    "temporal_distance_reduction_mean",
    "temporal_distance_reduction_max",
    "temporal_distance_reduction_min"
]

top_temporal = [
    "temporal_max_relative_speed_mean",
    "temporal_pre_relative_speed_mean",
    "temporal_max_relative_speed_max",
    "temporal_pre_speed_1_max",
    "temporal_window_relative_speed_max",
    "temporal_distance_reduction_mean"
]

feature_sets = {
    "Base": base,
    "Base_Temporal_Relative": base + temporal_relative,
    "Base_Temporal_Distance": base + temporal_distance,
    "Base_Top_Temporal": base + top_temporal
}

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for name, features in feature_sets.items():

    X = d[features].select_dtypes(
        include=[np.number]
    )

    bal_scores = []
    f1_scores = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups),
        start=1
    ):

        model = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True
                )
            ),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=300,
                    max_depth=4,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ])

        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx]
        )

        pred = model.predict(
            X.iloc[test_idx]
        )

        bal_scores.append(
            balanced_accuracy_score(
                y.iloc[test_idx],
                pred
            )
        )

        f1_scores.append(
            f1_score(
                y.iloc[test_idx],
                pred,
                zero_division=0
            )
        )

    results.append({
        "feature_set": name,
        "balanced_accuracy_mean": np.mean(bal_scores),
        "balanced_accuracy_std": np.std(bal_scores),
        "f1_mean": np.mean(f1_scores),
        "f1_std": np.std(f1_scores)
    })

result_df = pd.DataFrame(results)

print("=" * 60)
print("SELECTED TEMPORAL FEATURE BENCHMARK")
print("=" * 60)

print(
    result_df.round(4).to_string(index=False)
)

OUTPUT = f"{BASE}/selected_temporal_benchmark.csv"

result_df.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)

print("=" * 60)

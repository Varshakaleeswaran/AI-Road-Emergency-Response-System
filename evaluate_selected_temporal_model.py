import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

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

features = [
    "movement_mean",
    "movement_max",
    "distance_min",
    "distance_mean",
    "relative_speed_mean",
    "relative_speed_max",
    "unique_vehicle_ids",
    "temporal_max_relative_speed_mean",
    "temporal_pre_relative_speed_mean",
    "temporal_max_relative_speed_max",
    "temporal_pre_speed_1_max",
    "temporal_window_relative_speed_max",
    "temporal_distance_reduction_mean"
]

X = d[features]

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

all_true = []
all_pred = []

print("=" * 60)
print("SELECTED TEMPORAL MODEL")
print("=" * 60)

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
                n_estimators=500,
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

    bal = balanced_accuracy_score(
        y.iloc[test_idx],
        pred
    )

    f1 = f1_score(
        y.iloc[test_idx],
        pred,
        zero_division=0
    )

    all_true.extend(
        y.iloc[test_idx].tolist()
    )

    all_pred.extend(
        pred.tolist()
    )

    print(
        f"Fold {fold}: "
        f"balanced_acc={bal:.3f} "
        f"F1={f1:.3f}"
    )

print("\n" + "=" * 60)
print("OUT-OF-FOLD CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    all_true,
    all_pred
)

print(
    pd.DataFrame(
        cm,
        index=["non_event", "accident_event"],
        columns=["non_event", "accident_event"]
    )
)

print("\nClassification report:")
print(
    classification_report(
        all_true,
        all_pred,
        target_names=[
            "non_event",
            "accident_event"
        ],
        digits=4,
        zero_division=0
    )
)

print("=" * 60)

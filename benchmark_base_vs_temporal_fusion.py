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

base_features = [
    "movement_mean",
    "movement_max",
    "distance_min",
    "distance_mean",
    "relative_speed_mean",
    "relative_speed_max",
    "unique_vehicle_ids"
]

temporal_features = [
    c for c in d.columns
    if c.startswith("temporal_")
]

feature_sets = {
    "Base": base_features,
    "Base_Temporal": base_features + temporal_features
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

        bal = balanced_accuracy_score(
            y.iloc[test_idx],
            pred
        )

        f1 = f1_score(
            y.iloc[test_idx],
            pred,
            zero_division=0
        )

        bal_scores.append(bal)
        f1_scores.append(f1)

        print(
            f"{name} | Fold {fold}: "
            f"balanced_acc={bal:.3f} "
            f"F1={f1:.3f}"
        )

    results.append({
        "feature_set": name,
        "balanced_accuracy_mean": np.mean(bal_scores),
        "balanced_accuracy_std": np.std(bal_scores),
        "f1_mean": np.mean(f1_scores),
        "f1_std": np.std(f1_scores)
    })

print("\n" + "=" * 60)
print("BASE VS TEMPORAL FUSION")
print("=" * 60)

result_df = pd.DataFrame(results)

print(
    result_df.round(4).to_string(index=False)
)

OUTPUT = f"{BASE}/base_vs_temporal_fusion_benchmark.csv"

result_df.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)

print("=" * 60)

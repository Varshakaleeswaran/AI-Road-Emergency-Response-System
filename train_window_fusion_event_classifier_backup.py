import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

BASE = "data/ACCIDENT"

INPUT = f"{BASE}/accident_event_window_fusion_dataset_clean.csv"
OUTPUT = f"{BASE}/window_fusion_event_benchmark.csv"

d = pd.read_csv(INPUT)

# Event detection only:
# accident_event vs non_event
d = d[d["event_label"].isin(["accident_event", "non_event", "normal"])].copy()

y = (d["event_label"] == "accident_event").astype(int)

groups = d["video"]

drop_cols = [
    "video",
    "label",
    "event_label",
    "window_start",
    "window_end"
]

X = d.drop(columns=drop_cols, errors="ignore")

# Keep numeric features only
X = X.select_dtypes(include=[np.number])

print("=" * 60)
print("LEAKAGE-SAFE WINDOW FUSION BENCHMARK")
print("=" * 60)

print("Samples:", len(d))
print("Features:", X.shape[1])
print("Videos:", groups.nunique())

print("\nClass counts:")
print(d["event_label"].value_counts().to_string())

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = []

for name, model in models.items():

    fold_balanced = []
    fold_f1 = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        pipe = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True
                )
            ),
            (
                "model",
                model
            )
        ])

        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)

        bal = balanced_accuracy_score(
            y_test,
            pred
        )

        f1 = f1_score(
            y_test,
            pred,
            zero_division=0
        )

        fold_balanced.append(bal)
        fold_f1.append(f1)

        print(
            f"{name} | Fold {fold}: "
            f"balanced_acc={bal:.3f} "
            f"F1={f1:.3f}"
        )

    results.append({
        "model": name,
        "balanced_accuracy_mean": np.mean(fold_balanced),
        "balanced_accuracy_std": np.std(fold_balanced),
        "f1_mean": np.mean(fold_f1),
        "f1_std": np.std(fold_f1)
    })

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

result_df = pd.DataFrame(results)

print(
    result_df.round(4).to_string(index=False)
)

result_df.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)

print("=" * 60)

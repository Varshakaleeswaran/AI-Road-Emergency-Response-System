
import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.inspection import permutation_importance

BASE = "data/ACCIDENT"

INPUT = f"{BASE}/accident_event_window_fusion_dataset_clean.csv"
OUTPUT = f"{BASE}/window_fusion_grouped_feature_importance.csv"

d = pd.read_csv(INPUT)

d = d[
    d["event_label"].isin(
        ["accident_event", "non_event", "normal"]
    )
].copy().reset_index(drop=True)

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
X = X.select_dtypes(include=[np.number])

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

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

    model.fit(X_train, y_train)

    # Permutation importance on held-out videos.
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="balanced_accuracy",
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )

    for i, feature in enumerate(X.columns):
        results.append({
            "fold": fold,
            "feature": feature,
            "importance_mean": perm.importances_mean[i],
            "importance_std": perm.importances_std[i]
        })

importance = pd.DataFrame(results)

summary = (
    importance.groupby("feature")
    .agg(
        importance_mean=("importance_mean", "mean"),
        importance_std_across_folds=("importance_mean", "std")
    )
    .reset_index()
    .sort_values("importance_mean", ascending=False)
)

print("=" * 65)
print("GROUPED-CV PERMUTATION FEATURE IMPORTANCE")
print("=" * 65)
print("Rows:", len(d))
print("Features:", X.shape[1])
print("Videos:", groups.nunique())
print("\nTop 20 features:")
print(summary.head(20).round(6).to_string(index=False))

summary.to_csv(OUTPUT, index=False)

print("\nSaved:")
print(OUTPUT)
print("\nPositive importance means shuffling that feature reduced")
print("balanced accuracy on held-out videos. This is diagnostic,")
print("not proof of causation.")
print("=" * 65)

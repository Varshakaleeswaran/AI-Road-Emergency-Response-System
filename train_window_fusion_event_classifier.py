
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.model_selection import StratifiedGroupKFold

BASE = "data/ACCIDENT"

INPUT = f"{BASE}/accident_event_window_fusion_dataset_clean.csv"
OUTPUT = f"{BASE}/window_fusion_event_benchmark.csv"
FOLD_OUTPUT = f"{BASE}/window_fusion_event_fold_results.csv"
PREDICTION_OUTPUT = f"{BASE}/window_fusion_event_predictions.csv"

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

d = pd.read_csv(INPUT)

# Accident-event detection:
# Positive (1): accident_event
# Negative (0): non_event and normal
d = d[
    d["event_label"].isin(
        ["accident_event", "non_event", "normal"]
    )
].copy()

d = d.reset_index(drop=True)

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

if X.shape[1] == 0:
    raise ValueError("No numeric features found in the dataset.")

if y.nunique() < 2:
    raise ValueError("The dataset must contain positive and negative samples.")

# --------------------------------------------------
# 2. Dataset overview
# --------------------------------------------------

print("=" * 70)
print("ACCIDENT EVENT DETECTION - GROUPED CV BENCHMARK")
print("=" * 70)

print("Samples:", len(d))
print("Numeric features:", X.shape[1])
print("Videos:", groups.nunique())

print("\nEvent-label counts:")
print(d["event_label"].value_counts().to_string())

print("\nPositive accident-event windows:", int(y.sum()))
print("Negative windows:", int((y == 0).sum()))

# --------------------------------------------------
# 3. Grouped cross-validation
# --------------------------------------------------

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

summary_results = []
fold_results = []
prediction_results = []
# --------------------------------------------------
# 4. Train and evaluate each model
# --------------------------------------------------

for name, model in models.items():

    fold_balanced = []
    fold_precision = []
    fold_recall = []
    fold_f1 = []

    total_tn = 0
    total_fp = 0
    total_fn = 0
    total_tp = 0

    print("\n" + "-" * 70)
    print("MODEL:", name)
    print("-" * 70)

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        train_videos = set(groups.iloc[train_idx])
        test_videos = set(groups.iloc[test_idx])

        # Ensure no video appears in both sets.
        overlap = train_videos.intersection(test_videos)

        if overlap:
            raise RuntimeError(
                f"Video leakage detected in fold {fold}: {overlap}"
            )

        # Imputation is fitted only on the training fold.
        pipe = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True
                )
            ),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        # Save each window's out-of-fold prediction.
        test_rows = d.iloc[test_idx]

        for j, row_idx in enumerate(test_idx):
            actual_label = d.iloc[row_idx]["event_label"]
            predicted_value = int(pred[j])
            actual_value = int(y_test.iloc[j])

            prediction_results.append({
                "model": name,
                "fold": fold,
                "video": d.iloc[row_idx]["video"],
                "window_start": d.iloc[row_idx]["window_start"],
                "window_end": d.iloc[row_idx]["window_end"],
                "true_event_label": actual_label,
                "y_true": actual_value,
                "y_pred": predicted_value,
                "prediction": (
                    "accident_event"
                    if predicted_value == 1
                    else "non_event_or_normal"
                ),
                "is_false_positive": int(
                    actual_value == 0 and predicted_value == 1
                ),
                "is_false_negative": int(
                    actual_value == 1 and predicted_value == 0
                )
            })

        balanced_acc = balanced_accuracy_score(y_test, pred)

        precision = precision_score(
            y_test,
            pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            pred,
            zero_division=0
        )

        # Label order: 0 = non-accident, 1 = accident event
        tn, fp, fn, tp = confusion_matrix(
            y_test,
            pred,
            labels=[0, 1]
        ).ravel()

        fold_balanced.append(balanced_acc)
        fold_precision.append(precision)
        fold_recall.append(recall)
        fold_f1.append(f1)

        total_tn += int(tn)
        total_fp += int(fp)
        total_fn += int(fn)
        total_tp += int(tp)

        test_rows = d.iloc[test_idx]

        print(
            f"Fold {fold} | "
            f"test_windows={len(test_idx)} | "
            f"test_videos={test_rows['video'].nunique()} | "
            f"accident_windows={int(y_test.sum())}"
        )

        print(
            f"  Balanced accuracy={balanced_acc:.3f} | "
            f"Precision={precision:.3f} | "
            f"Recall={recall:.3f} | "
            f"F1={f1:.3f}"
        )

        print(
            f"  TN={tn} | FP={fp} | "
            f"FN={fn} | TP={tp}"
        )

        fold_results.append({
            "model": name,
            "fold": fold,
            "test_windows": len(test_idx),
            "test_videos": test_rows["video"].nunique(),
            "actual_accident_windows": int(y_test.sum()),
            "balanced_accuracy": balanced_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        })

    # --------------------------------------------------
    # 5. Aggregate results
    # --------------------------------------------------

    summary_results.append({
        "model": name,
        "balanced_accuracy_mean": np.mean(fold_balanced),
        "balanced_accuracy_std": np.std(fold_balanced),
        "precision_mean": np.mean(fold_precision),
        "precision_std": np.std(fold_precision),
        "recall_mean": np.mean(fold_recall),
        "recall_std": np.std(fold_recall),
        "f1_mean": np.mean(fold_f1),
        "f1_std": np.std(fold_f1),
        "total_TN": total_tn,
        "total_FP": total_fp,
        "total_FN": total_fn,
        "total_TP": total_tp
    })

    print("\nCombined out-of-fold counts:")
    print("True negatives (TN):", total_tn)
    print("False positives (FP):", total_fp)
    print("False negatives (FN):", total_fn)
    print("True positives (TP):", total_tp)

# --------------------------------------------------
# 6. Save benchmark summaries
# --------------------------------------------------

summary_df = pd.DataFrame(summary_results)
fold_df = pd.DataFrame(fold_results)

print("\n" + "=" * 70)
print("SUMMARY: MEAN AND STANDARD DEVIATION ACROSS FOLDS")
print("=" * 70)

print(summary_df.round(4).to_string(index=False))

summary_df.to_csv(OUTPUT, index=False)
fold_df.to_csv(FOLD_OUTPUT, index=False)

prediction_df = pd.DataFrame(prediction_results)
prediction_df.to_csv(PREDICTION_OUTPUT, index=False)

print("\nSaved individual out-of-fold predictions:")
print(PREDICTION_OUTPUT)
print("Prediction rows:", len(prediction_df))

print("\nSaved summary:")
print(OUTPUT)

print("\nSaved per-fold metrics:")
print(FOLD_OUTPUT)

print("\nNote:")
print(
    "These are experimental results from grouped cross-validation. "
    "They do not establish production readiness."
)

print("=" * 70)
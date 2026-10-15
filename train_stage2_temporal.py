import pandas as pd
import numpy as np
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
    make_scorer,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

DATA_FILE = "data/ACCIDENT/temporal_collision_features.csv"
MODEL_FILE = "data/ACCIDENT/stage2_temporal_classifier.joblib"

df = pd.read_csv(DATA_FILE)

df = df.replace([np.inf, -np.inf], np.nan)

# ------------------------------------------------------------
# ONLY MULTI-VEHICLE ACCIDENTS
# ------------------------------------------------------------

df = df[df["accident_type"] != "single"].copy()

print("=" * 70)
print("STEP 14 V2 - TEMPORAL ACCIDENT-TYPE CLASSIFIER")
print("=" * 70)

print("\nRaw temporal rows:", len(df))
print("Videos:", df["video"].nunique())

# ------------------------------------------------------------
# TEMPORAL FEATURES
# ------------------------------------------------------------

numeric_features = [
    "pre_relative_speed",
    "collision_relative_speed",
    "post_relative_speed",
    "max_relative_speed",
    "pre_speed_1",
    "pre_speed_2",
    "post_speed_1",
    "post_speed_2",
    "speed_change_1",
    "speed_change_2",
    "relative_speed_change",
    "observations",
    "pre_distance",
    "collision_distance",
    "post_distance",
    "distance_reduction"
]

# Intentionally exclude IoU because the current dataset has
# essentially no usable IoU observations.

agg_functions = ["mean", "max", "min"]

grouped = df.groupby(
    ["video", "accident_type"]
)[numeric_features].agg(agg_functions)

grouped.columns = [
    f"{feature}_{agg}"
    for feature, agg in grouped.columns
]

grouped = grouped.reset_index()

# ------------------------------------------------------------
# TEMPORAL COUNTS
# ------------------------------------------------------------

counts = df.groupby(
    ["video", "accident_type"]
).agg(
    temporal_pair_count=("vehicle_1", "count"),
    temporal_vehicle_pair_count=(
        "vehicle_1",
        lambda x: x.nunique()
    ),
    temporal_total_observations=(
        "observations",
        "sum"
    )
).reset_index()

data = grouped.merge(
    counts,
    on=["video", "accident_type"],
    how="left"
)

# ------------------------------------------------------------
# DERIVED COLLISION FEATURES
# ------------------------------------------------------------

eps = 1e-6

data["relative_speed_increase"] = (
    data["collision_relative_speed_mean"]
    -
    data["pre_relative_speed_mean"]
)

data["relative_speed_post_change"] = (
    data["post_relative_speed_mean"]
    -
    data["collision_relative_speed_mean"]
)

data["speed_change_total"] = (
    data["speed_change_1_mean"].abs()
    +
    data["speed_change_2_mean"].abs()
)

data["speed_change_difference"] = (
    data["speed_change_1_mean"]
    -
    data["speed_change_2_mean"]
).abs()

data["distance_drop_ratio"] = (
    data["distance_reduction_mean"]
    /
    (
        data["pre_distance_mean"].abs()
        +
        eps
    )
)

data["collision_to_pre_distance_ratio"] = (
    data["collision_distance_mean"]
    /
    (
        data["pre_distance_mean"].abs()
        +
        eps
    )
)

data["post_to_collision_distance_ratio"] = (
    data["post_distance_mean"]
    /
    (
        data["collision_distance_mean"].abs()
        +
        eps
    )
)

data["relative_speed_ratio"] = (
    data["collision_relative_speed_mean"]
    /
    (
        data["pre_relative_speed_mean"].abs()
        +
        eps
    )
)

data["post_relative_speed_ratio"] = (
    data["post_relative_speed_mean"]
    /
    (
        data["collision_relative_speed_mean"].abs()
        +
        eps
    )
)

data["collision_speed_distance_score"] = (
    data["collision_relative_speed_max"]
    /
    (
        data["collision_distance_min"].abs()
        +
        eps
    )
)

# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

y = data["accident_type"]

# ------------------------------------------------------------
# REMOVE IDENTIFIERS
# ------------------------------------------------------------

X = data.drop(
    columns=["video", "accident_type"]
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

FEATURES = list(X.columns)

print("\nVideo-level dataset:", len(data))
print("Features:", len(FEATURES))

print("\nClasses:")
print(y.value_counts().to_string())

# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------

models = {

    "RandomForest": Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=4,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),

    "ExtraTrees": Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        ),
        (
            "classifier",
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),

    "LogisticRegression": Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=42
            )
        )
    ]),

    "SVM": Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                class_weight="balanced",
                probability=True,
                random_state=42
            )
        )
    ])
}

# ------------------------------------------------------------
# CROSS VALIDATION
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": make_scorer(
        balanced_accuracy_score
    ),
    "macro_f1": make_scorer(
        f1_score,
        average="macro"
    )
}

results = []

print("\n" + "=" * 70)
print("MODEL BENCHMARK")
print("=" * 70)

for name, model in models.items():

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    accuracy = scores["test_accuracy"].mean()
    balanced = scores["test_balanced_accuracy"].mean()
    macro_f1 = scores["test_macro_f1"].mean()

    results.append({
        "model": name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "macro_f1": macro_f1
    })

    print(
        f"{name:20s} "
        f"Accuracy={accuracy:.4f} "
        f"BalancedAcc={balanced:.4f} "
        f"MacroF1={macro_f1:.4f}"
    )

results_df = pd.DataFrame(results)

# ------------------------------------------------------------
# SELECT BEST MODEL
# ------------------------------------------------------------

best_row = results_df.sort_values(
    ["balanced_accuracy", "macro_f1"],
    ascending=False
).iloc[0]

best_name = best_row["model"]
best_model = models[best_name]

# ------------------------------------------------------------
# OUT-OF-FOLD PREDICTIONS
# ------------------------------------------------------------

predictions = cross_val_predict(
    best_model,
    X,
    y,
    cv=cv,
    n_jobs=-1
)

print("\n" + "=" * 70)
print("OUT-OF-FOLD CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y,
        predictions,
        digits=4
    )
)

print("Confusion matrix:")

labels = sorted(y.unique())

cm = confusion_matrix(
    y,
    predictions,
    labels=labels
)

print(
    pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )
)

# ------------------------------------------------------------
# TRAIN FINAL MODEL
# ------------------------------------------------------------

best_model.fit(X, y)

# ------------------------------------------------------------
# SAVE DATASET
# ------------------------------------------------------------

DATASET_FILE = (
    "data/ACCIDENT/"
    "stage2_temporal_dataset.csv"
)

data.to_csv(
    DATASET_FILE,
    index=False
)

# ------------------------------------------------------------
# SAVE BENCHMARK
# ------------------------------------------------------------

BENCHMARK_FILE = (
    "data/ACCIDENT/"
    "stage2_temporal_benchmark.csv"
)

results_df.to_csv(
    BENCHMARK_FILE,
    index=False
)

# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------

os.makedirs(
    os.path.dirname(MODEL_FILE),
    exist_ok=True
)

joblib.dump(
    {
        "model": best_model,
        "features": FEATURES,
        "classes": labels,
        "model_name": best_name,
        "benchmark": results_df
    },
    MODEL_FILE
)

# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BEST STEP-14 MODEL")
print("=" * 70)

print("Model:", best_name)

print(
    "Balanced accuracy:",
    round(
        best_row["balanced_accuracy"],
        4
    )
)

print(
    "Macro F1:",
    round(
        best_row["macro_f1"],
        4
    )
)

print("\nDataset saved:")
print(DATASET_FILE)

print("\nBenchmark saved:")
print(BENCHMARK_FILE)

print("\nModel saved:")
print(MODEL_FILE)

print("=" * 70)

print("\nIMPORTANT:")
print(
    "This model classifies MULTI-VEHICLE accident types:"
)

for label in labels:
    print("-", label)

print(
    "\nSingle-vehicle accidents remain handled by Step 15."
)

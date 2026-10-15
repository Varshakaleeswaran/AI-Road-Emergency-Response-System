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
MODEL_FILE = "data/ACCIDENT/stage2_v3_classifier.joblib"

df = pd.read_csv(DATA_FILE)

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

# ------------------------------------------------------------
# STEP 14:
# MULTI-VEHICLE ACCIDENT TYPES ONLY
# ------------------------------------------------------------

df = df[
    df["accident_type"] != "single"
].copy()

print("=" * 70)
print("STEP 14 V3 - COLLISION-PAIR ACCIDENT-TYPE CLASSIFIER")
print("=" * 70)

print("\nTemporal rows:", len(df))
print("Videos:", df["video"].nunique())

# ------------------------------------------------------------
# SELECT THE MOST RELEVANT PAIR PER VIDEO
#
# Priority:
# 1. smallest collision distance
# 2. highest collision relative speed
# ------------------------------------------------------------

df["collision_distance_rank"] = (
    df.groupby("video")["collision_distance"]
    .rank(
        method="first",
        ascending=True,
        na_option="bottom"
    )
)

df["relative_speed_rank"] = (
    df.groupby("video")["collision_relative_speed"]
    .rank(
        method="first",
        ascending=False,
        na_option="bottom"
    )
)

# Primary selection: smallest collision distance.
selected = (
    df.sort_values(
        [
            "video",
            "collision_distance",
            "collision_relative_speed"
        ],
        ascending=[
            True,
            True,
            False
        ],
        na_position="last"
    )
    .groupby("video", as_index=False)
    .first()
)

print("\nSelected collision pairs:", len(selected))
print("Expected videos:", df["video"].nunique())

# ------------------------------------------------------------
# PHYSICALLY MEANINGFUL FEATURES
# ------------------------------------------------------------

features = [
    "pre_relative_speed",
    "collision_relative_speed",
    "post_relative_speed",

    "pre_speed_1",
    "pre_speed_2",
    "post_speed_1",
    "post_speed_2",

    "speed_change_1",
    "speed_change_2",
    "relative_speed_change",

    "pre_distance",
    "collision_distance",
    "post_distance",
    "distance_reduction",

    "max_relative_speed",
    "observations"
]

X = selected[features].copy()

# ------------------------------------------------------------
# DERIVED FEATURES
# ------------------------------------------------------------

eps = 1e-6

X["relative_speed_increase"] = (
    X["collision_relative_speed"]
    -
    X["pre_relative_speed"]
)

X["relative_speed_post_change"] = (
    X["post_relative_speed"]
    -
    X["collision_relative_speed"]
)

X["speed_change_total"] = (
    X["speed_change_1"].abs()
    +
    X["speed_change_2"].abs()
)

X["speed_change_difference"] = (
    X["speed_change_1"]
    -
    X["speed_change_2"]
).abs()

X["speed_change_symmetry"] = (
    1.0
    -
    (
        X["speed_change_difference"]
        /
        (
            X["speed_change_total"].abs()
            +
            eps
        )
    )
)

X["distance_drop_ratio"] = (
    X["distance_reduction"]
    /
    (
        X["pre_distance"].abs()
        +
        eps
    )
)

X["collision_distance_ratio"] = (
    X["collision_distance"]
    /
    (
        X["pre_distance"].abs()
        +
        eps
    )
)

X["post_distance_ratio"] = (
    X["post_distance"]
    /
    (
        X["collision_distance"].abs()
        +
        eps
    )
)

X["collision_relative_speed_ratio"] = (
    X["collision_relative_speed"]
    /
    (
        X["pre_relative_speed"].abs()
        +
        eps
    )
)

X["post_relative_speed_ratio"] = (
    X["post_relative_speed"]
    /
    (
        X["collision_relative_speed"].abs()
        +
        eps
    )
)

# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

y = selected["accident_type"]

print("\nClasses:")
print(
    y.value_counts().to_string()
)

print("\nFeatures:", len(X.columns))

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
                max_depth=3,
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
                max_depth=3,
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
        f"{name:20s}"
        f" Accuracy={accuracy:.4f}"
        f" BalancedAcc={balanced:.4f}"
        f" MacroF1={macro_f1:.4f}"
    )

results_df = pd.DataFrame(results)

# ------------------------------------------------------------
# BEST MODEL
# ------------------------------------------------------------

best_row = results_df.sort_values(
    [
        "balanced_accuracy",
        "macro_f1"
    ],
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

labels = sorted(
    y.unique()
)

print("\n" + "=" * 70)
print("OUT-OF-FOLD CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y,
        predictions,
        labels=labels,
        digits=4,
        zero_division=0
    )
)

print("Confusion matrix:")

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

best_model.fit(
    X,
    y
)

# ------------------------------------------------------------
# SAVE DATASET
# ------------------------------------------------------------

dataset = selected[
    [
        "video",
        "accident_type",
        "accident_frame",
        "vehicle_1",
        "vehicle_2"
    ]
].copy()

dataset = pd.concat(
    [
        dataset.reset_index(drop=True),
        X.reset_index(drop=True)
    ],
    axis=1
)

DATASET_FILE = (
    "data/ACCIDENT/"
    "stage2_v3_dataset.csv"
)

dataset.to_csv(
    DATASET_FILE,
    index=False
)

# ------------------------------------------------------------
# SAVE BENCHMARK
# ------------------------------------------------------------

BENCHMARK_FILE = (
    "data/ACCIDENT/"
    "stage2_v3_benchmark.csv"
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
        "features": list(X.columns),
        "classes": labels,
        "model_name": best_name
    },
    MODEL_FILE
)

# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 14 V3 FINAL")
print("=" * 70)

print("Best model:", best_name)

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

print("\nDataset:")
print(DATASET_FILE)

print("\nBenchmark:")
print(BENCHMARK_FILE)

print("\nModel:")
print(MODEL_FILE)

print("=" * 70)

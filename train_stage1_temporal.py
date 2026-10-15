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
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, balanced_accuracy_score, f1_score

DATA_FILE = "data/ACCIDENT/temporal_collision_features.csv"
MODEL_FILE = "data/ACCIDENT/stage1_temporal_classifier.joblib"

df = pd.read_csv(DATA_FILE)

df = df.replace([np.inf, -np.inf], np.nan)

# ------------------------------------------------------------
# AGGREGATE TEMPORAL PAIR FEATURES TO VIDEO LEVEL
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
    "distance_reduction",
    "pre_iou",
    "collision_iou",
    "post_iou"
]

agg_functions = ["mean", "max", "min"]

grouped = df.groupby(
    ["video", "accident_type"]
)[numeric_features].agg(agg_functions)

grouped.columns = [
    f"temporal_{feature}_{agg}"
    for feature, agg in grouped.columns
]

grouped = grouped.reset_index()

# ------------------------------------------------------------
# ADD VIDEO-LEVEL TEMPORAL COUNTS
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
# DERIVED TEMPORAL FEATURES
# ------------------------------------------------------------

eps = 1e-6

data["temporal_relative_speed_increase"] = (
    data["temporal_collision_relative_speed_mean"]
    - data["temporal_pre_relative_speed_mean"]
)

data["temporal_relative_speed_post_change"] = (
    data["temporal_post_relative_speed_mean"]
    - data["temporal_collision_relative_speed_mean"]
)

data["temporal_speed_change_total"] = (
    data["temporal_speed_change_1_mean"].abs()
    + data["temporal_speed_change_2_mean"].abs()
)

data["temporal_distance_drop_ratio"] = (
    data["temporal_distance_reduction_mean"]
    /
    (
        data["temporal_pre_distance_mean"].abs()
        + eps
    )
)

data["temporal_collision_closeness"] = (
    1.0
    /
    (
        data["temporal_collision_distance_min"].abs()
        + eps
    )
)

data["temporal_iou_change"] = (
    data["temporal_collision_iou_mean"]
    -
    data["temporal_pre_iou_mean"]
)

# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

y = data["accident_type"].apply(
    lambda x:
        "single"
        if x == "single"
        else "multi_vehicle"
)

# ------------------------------------------------------------
# REMOVE IDENTIFIERS / TARGET
# ------------------------------------------------------------

drop_columns = [
    "video",
    "accident_type"
]

X = data.drop(
    columns=drop_columns
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

FEATURES = list(X.columns)

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

print("=" * 70)
print("STEP 15 - TEMPORAL SINGLE VS MULTI-VEHICLE")
print("=" * 70)

print("\nVideo-level dataset:")
print("Rows:", len(data))
print("Features:", len(FEATURES))

print("\nClasses:")
print(y.value_counts().to_string())

print("\nTemporal records:")
print("Raw temporal rows:", len(df))
print("Videos:", df["video"].nunique())

print("\nModel benchmark:")
print("-" * 70)

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

best_model.fit(X, y)

# ------------------------------------------------------------
# SAVE DATASET
# ------------------------------------------------------------

DATASET_FILE = (
    "data/ACCIDENT/"
    "stage1_temporal_dataset.csv"
)

data.to_csv(
    DATASET_FILE,
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
        "model_name": best_name,
        "benchmark": results_df
    },
    MODEL_FILE
)

BENCHMARK_FILE = (
    "data/ACCIDENT/"
    "stage1_temporal_benchmark.csv"
)

results_df.to_csv(
    BENCHMARK_FILE,
    index=False
)

# ------------------------------------------------------------
# FINAL REPORT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BEST TEMPORAL MODEL")
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

print("\nTemporal dataset saved:")
print(DATASET_FILE)

print("\nModel saved:")
print(MODEL_FILE)

print("\nBenchmark saved:")
print(BENCHMARK_FILE)

print("=" * 70)

print("\nIMPORTANT:")
print(
    "This is still SINGLE vs MULTI-VEHICLE "
    "among known accident videos."
)

print(
    "It is NOT the final ACCIDENT vs NORMAL detector."
)

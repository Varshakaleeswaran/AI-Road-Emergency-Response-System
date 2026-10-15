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
from sklearn.inspection import permutation_importance

DATA_FILE = "data/ACCIDENT/classifier_dataset.csv"
MODEL_FILE = "data/ACCIDENT/stage1_enhanced_classifier.joblib"

df = pd.read_csv(DATA_FILE)

# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

y = df["accident_type"].apply(
    lambda x: "single" if x == "single" else "multi_vehicle"
)

# ------------------------------------------------------------
# BASE FEATURES
# ------------------------------------------------------------

features = [
    "pair_min_distance",
    "min_distance",
    "mean_distance",
    "pair_count",
    "max_relative_speed",
    "mean_relative_speed",
    "distance_observations",
    "relative_motion_observations",
    "pair_distance_at_accident",
    "pair_mean_distance",
    "pair_mean_observations",
]

X = df[features].copy()

X = X.replace([np.inf, -np.inf], np.nan)

# ------------------------------------------------------------
# DERIVED FEATURES
# ------------------------------------------------------------

eps = 1e-6

X["distance_reduction"] = (
    X["mean_distance"] - X["min_distance"]
)

X["distance_reduction_ratio"] = (
    X["distance_reduction"] /
    (X["mean_distance"].abs() + eps)
)

X["accident_distance_ratio"] = (
    X["pair_distance_at_accident"] /
    (X["pair_mean_distance"].abs() + eps)
)

X["relative_speed_per_pair"] = (
    X["max_relative_speed"] /
    (X["pair_count"] + eps)
)

X["relative_motion_density"] = (
    X["relative_motion_observations"] /
    (X["distance_observations"] + eps)
)

X["pair_observation_density"] = (
    X["pair_mean_observations"] /
    (X["distance_observations"] + eps)
)

X["interaction_strength"] = (
    X["pair_count"] *
    X["max_relative_speed"]
)

X["collision_closeness"] = (
    1.0 /
    (X["pair_min_distance"] + eps)
)

X["motion_distance_score"] = (
    X["max_relative_speed"] /
    (X["min_distance"] + eps)
)

FEATURES = list(X.columns)

# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------

models = {

    "RandomForest": Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True
        )),
        ("classifier", RandomForestClassifier(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "ExtraTrees": Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True
        )),
        ("classifier", ExtraTreesClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "LogisticRegression": Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True
        )),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=42
        ))
    ]),

    "SVM": Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True
        )),
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            class_weight="balanced",
            probability=True,
            random_state=42
        ))
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
print("ENHANCED STAGE-1 CLASSIFIER")
print("=" * 70)

print("\nDataset:")
print("Videos:", len(df))
print("Features:", len(FEATURES))

print("\nClasses:")
print(y.value_counts().to_string())

print("\nDerived features:")
print(
    X.columns.difference(features).tolist()
)

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
# PERMUTATION IMPORTANCE
# ------------------------------------------------------------

try:

    importance = permutation_importance(
        best_model,
        X,
        y,
        scoring="balanced_accuracy",
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": importance.importances_mean
    }).sort_values(
        "importance",
        ascending=False
    )

except Exception as e:

    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": np.nan
    })

    print("\nPermutation importance unavailable:", e)

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
        "benchmark": results_df,
        "feature_importance": importance_df
    },
    MODEL_FILE
)

benchmark_file = (
    "data/ACCIDENT/stage1_enhanced_benchmark.csv"
)

results_df.to_csv(
    benchmark_file,
    index=False
)

importance_file = (
    "data/ACCIDENT/stage1_enhanced_feature_importance.csv"
)

importance_df.to_csv(
    importance_file,
    index=False
)

# ------------------------------------------------------------
# FINAL REPORT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Model:", best_name)
print(
    "Balanced accuracy:",
    round(best_row["balanced_accuracy"], 4)
)
print(
    "Macro F1:",
    round(best_row["macro_f1"], 4)
)

print("\nTop features:")
print(
    importance_df.head(15).to_string(index=False)
)

print("\nModel saved:")
print(MODEL_FILE)

print("\nBenchmark saved:")
print(benchmark_file)

print("\nFeature importance saved:")
print(importance_file)

print("=" * 70)

print("\nIMPORTANT:")
print(
    "This classifier predicts SINGLE vs MULTI-VEHICLE "
    "among known accident videos."
)
print(
    "It is NOT yet the final ACCIDENT vs NORMAL detector."
)

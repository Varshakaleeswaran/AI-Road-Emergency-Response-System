import pandas as pd
import numpy as np
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

DATA_FILE = "data/ACCIDENT/classifier_dataset.csv"
MODEL_FILE = "data/ACCIDENT/stage2_accident_classifier.joblib"

FEATURES = [
    "pair_min_distance",
    "min_distance",
    "pair_count",
    "max_relative_speed",
    "distance_observations",
    "relative_motion_observations",
    "pair_distance_at_accident",
    "pair_mean_distance",
    "pair_mean_observations",
    "mean_distance",
    "mean_relative_speed"
]

df = pd.read_csv(DATA_FILE)

# Stage 2 uses only multi-vehicle accidents
df = df[df["accident_type"] != "single"].copy()

X = df[FEATURES].replace([np.inf, -np.inf], np.nan)
y = df["accident_type"]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=3,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    ))
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("=" * 60)
print("STAGE-2 ACCIDENT CLASSIFIER")
print("=" * 60)
print("Training videos:", len(df))
print("\nClasses:")
print(y.value_counts().to_string())

print("\n5-FOLD CROSS-VALIDATION")
print("Fold accuracies:", np.round(scores, 3))
print("Mean accuracy:", round(scores.mean(), 4))
print("Std:", round(scores.std(), 4))

# Train final model on all 20 videos
model.fit(X, y)

os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "classes": sorted(y.unique().tolist())
    },
    MODEL_FILE
)

print("\nFinal model trained on all 20 multi-vehicle videos.")
print("Model saved to:")
print(MODEL_FILE)

print("=" * 60)

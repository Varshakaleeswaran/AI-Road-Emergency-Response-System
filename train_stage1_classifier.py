import pandas as pd
import numpy as np
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

DATA_FILE = "data/ACCIDENT/classifier_dataset.csv"
MODEL_FILE = "data/ACCIDENT/stage1_accident_classifier.joblib"

FEATURES = [
    "pair_min_distance",
    "min_distance",
    "pair_count",
    "max_relative_speed",
    "distance_observations",
    "relative_motion_observations",
    "pair_distance_at_accident"
]

df = pd.read_csv(DATA_FILE)

X = df[FEATURES].replace([np.inf, -np.inf], np.nan)

y = df["accident_type"].apply(
    lambda x: "single" if x == "single" else "multi_vehicle"
)

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

model.fit(X, y)

os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": FEATURES
    },
    MODEL_FILE
)

print("=" * 60)
print("STAGE-1 ACCIDENT CLASSIFIER TRAINED")
print("=" * 60)

print("Training videos:", len(df))
print("Features:", len(FEATURES))
print("Classes:")
print(y.value_counts().to_string())

print("\nModel saved to:")
print(MODEL_FILE)

print("\nStage 1:")
print("single")
print("multi_vehicle")

print("=" * 60)

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

DATA = "data/ACCIDENT/accident_event_window_dataset.csv"

FEATURES = [
    "movement_mean",
    "movement_max",
    "distance_min",
    "distance_mean",
    "relative_speed_mean",
    "relative_speed_max",
    "unique_vehicle_ids"
]

df = pd.read_csv(DATA)

df = df[df["event_label"].isin(["accident_event", "non_event"])].copy()

X = df[FEATURES].replace([np.inf, -np.inf], np.nan)
y = df["event_label"]
groups = df["video"]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42
    ))
])

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    groups=groups,
    cv=cv,
    scoring="balanced_accuracy"
)

print("=" * 60)
print("ACCIDENT EVENT VS NON-EVENT CLASSIFIER")
print("=" * 60)

print("Samples:", len(df))
print("Videos:", df["video"].nunique())

print("\nClasses:")
print(y.value_counts().to_string())

print("\nGROUP-SAFE 5-FOLD CROSS-VALIDATION")
print("Fold balanced accuracy:", np.round(scores, 3))
print("Mean balanced accuracy:", round(scores.mean(), 4))
print("Std:", round(scores.std(), 4))

model.fit(X, y)

pred = model.predict(X)

print("\nTRAINING-SET REPORT")
print(classification_report(y, pred))

print("Confusion matrix:")
print(confusion_matrix(y, pred))

print("\nFEATURE IMPORTANCE")

rf = model.named_steps["classifier"]

importance = pd.Series(
    rf.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

print(importance.to_string())

print("=" * 60)

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

DATA_FILE = "data/ACCIDENT/classifier_dataset.csv"

df = pd.read_csv(DATA_FILE)

TARGET = "accident_type"

EXCLUDE = [
    "video",
    "accident_type",
    "accident_frame",
    "window_start",
    "window_end",
    "pair_min_distance_frame"
]

X = df.drop(columns=EXCLUDE)
y = df[TARGET]

X = X.replace([np.inf, -np.inf], np.nan)

print("=" * 60)
print("ACCIDENT CLASSIFIER CROSS-VALIDATION")
print("=" * 60)

print("Dataset rows:", len(df))
print("Features:", list(X.columns))

print("\nClasses:")
print(y.value_counts().to_string())

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

predictions = cross_val_predict(
    pipeline,
    X,
    y,
    cv=cv
)

accuracy = accuracy_score(y, predictions)
macro_f1 = f1_score(y, predictions, average="macro")

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y,
    predictions,
    zero_division=0
))

print("Confusion Matrix:")
labels = sorted(y.unique())

cm = confusion_matrix(
    y,
    predictions,
    labels=labels
)

print("Labels:", labels)
print(cm)

result = pd.DataFrame({
    "video": df["video"],
    "actual": y,
    "predicted": predictions
})

print("\nPredictions:")
print(result.to_string(index=False))

OUTPUT_FILE = "data/ACCIDENT/classifier_cv_predictions.csv"
result.to_csv(OUTPUT_FILE, index=False)

print("\nSaved predictions to:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("CROSS-VALIDATION COMPLETED")
print("=" * 60)

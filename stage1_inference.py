import pandas as pd
import numpy as np
import joblib

MODEL_FILE = "data/ACCIDENT/stage1_accident_classifier.joblib"
DATA_FILE = "data/ACCIDENT/classifier_dataset.csv"

bundle = joblib.load(MODEL_FILE)
model = bundle["model"]
features = bundle["features"]

df = pd.read_csv(DATA_FILE)

X = df[features].replace([np.inf, -np.inf], np.nan)

predictions = model.predict(X)

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X)
    confidence = probabilities.max(axis=1)
else:
    confidence = np.full(len(predictions), np.nan)

result = pd.DataFrame({
    "video": df["video"],
    "actual_accident_type": df["accident_type"],
    "stage1_prediction": predictions,
    "stage1_confidence": confidence
})

OUTPUT_FILE = "data/ACCIDENT/stage1_predictions.csv"
result.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("STAGE-1 INFERENCE COMPLETED")
print("=" * 60)

print(result.to_string(index=False))

print("\nSaved to:")
print(OUTPUT_FILE)

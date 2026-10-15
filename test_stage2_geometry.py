import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

df = pd.read_csv(
    "data/ACCIDENT/stage2_geometry_dataset.csv"
)

features = [
    "mean_speed_1",
    "mean_speed_2",
    "max_speed_1",
    "max_speed_2",
    "mean_relative_speed",
    "max_relative_speed",
    "mean_angle_1",
    "mean_angle_2",
    "mean_direction_difference",
    "max_direction_difference",
    "mean_relative_angle",
    "mean_speed_difference",
    "mean_speed_ratio"
]

X = df[features]
y = df["accident_type"]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=500,
        max_depth=3,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42
    ))
])

cv = StratifiedKFold(
    n_splits=3,
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
print("STAGE-2 GEOMETRY CLASSIFIER")
print("=" * 60)

print("Videos:", len(df))
print("\nClass counts:")
print(y.value_counts().to_string())

print("\nFold accuracies:")
print(scores)

print("\nMean accuracy:", round(scores.mean(), 4))
print("Std:", round(scores.std(), 4))

print("=" * 60)

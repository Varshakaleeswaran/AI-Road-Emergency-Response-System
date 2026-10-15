import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

df = pd.read_csv(
    "data/ACCIDENT/stage2_combined_dataset.csv"
)

features = [
    # Original Stage-2 features
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
    "mean_relative_speed",

    # Geometry features
    "mean_speed_1",
    "mean_speed_2",
    "mean_relative_speed_geom",
    "max_relative_speed_geom",
    "mean_direction_difference",
    "max_direction_difference"
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
print("STAGE-2 COMBINED CLASSIFIER")
print("=" * 60)

print("Videos:", len(df))

print("\nClasses:")
print(y.value_counts().to_string())

print("\nFeatures:", len(features))

print("\nFold accuracies:")
print(scores)

print("\nMean accuracy:", round(scores.mean(), 4))
print("Std:", round(scores.std(), 4))

print("\nPrevious best accuracy: 0.5000")

if scores.mean() > 0.50:
    print("\nRESULT: IMPROVED")
elif scores.mean() == 0.50:
    print("\nRESULT: SAME")
else:
    print("\nRESULT: WORSE")

print("=" * 60)

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

FILE = "data/ACCIDENT/pair_collision_features.csv"

df = pd.read_csv(FILE)

# Stage 2 only
df = df[df["accident_type"] != "single"].copy()

# Remove rows with unusable collision distance
df = df.replace([np.inf, -np.inf], np.nan)

# Select the most collision-relevant pair from each video.
# Priority: smallest distance, then strongest interaction duration.
df = df.sort_values(
    ["video", "min_distance", "abs_collision_timing"],
    ascending=[True, True, True]
)

df = df.groupby("video", as_index=False).first()

features = [
    "min_distance",
    "distance_at_accident",
    "collision_timing",
    "abs_collision_timing",
    "interaction_observations"
]

X = df[features]
y = df["accident_type"]
groups = df["video"]

print("=" * 60)
print("PAIR-LEVEL STAGE-2 BENCHMARK")
print("=" * 60)

print("Videos:", len(df))

print("\nClasses:")
print(y.value_counts().to_string())

print("\nFeatures:")
print(features)

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=10000,
            class_weight="balanced"
        ))
    ])
}

for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring="accuracy"
    )

    print("\n" + name)
    print("Folds:", np.round(scores, 4))
    print("Mean:", round(scores.mean(), 4))
    print("Std :", round(scores.std(), 4))

print("\n" + "=" * 60)

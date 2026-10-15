import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv(
    "data/ACCIDENT/classifier_dataset.csv"
)

df = df[df["accident_type"] != "single"].copy()

features = [
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

X = df[features]
y = df["accident_type"]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=3,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=500,
        max_depth=4,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        C=1.0,
        random_state=42
    ),

    "SVM Linear": SVC(
        kernel="linear",
        class_weight="balanced",
        C=1.0
    ),

    "SVM RBF": SVC(
        kernel="rbf",
        class_weight="balanced",
        C=1.0
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=3
    )
}

print("=" * 70)
print("STAGE-2 CLASSIFIER BENCHMARK")
print("=" * 70)

print("Videos:", len(df))
print("Classes:")
print(y.value_counts().to_string())

print("\n")

results = []

for name, classifier in models.items():

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", classifier)
    ])

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    results.append({
        "model": name,
        "mean_accuracy": scores.mean(),
        "std": scores.std(),
        "folds": scores
    })

    print("-" * 70)
    print(name)
    print("Folds:", np.round(scores, 4))
    print("Mean:", round(scores.mean(), 4))
    print("Std :", round(scores.std(), 4))

print("\n")
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

for r in sorted(
    results,
    key=lambda x: x["mean_accuracy"],
    reverse=True
):
    print(
        f"{r['model']:22s} "
        f"{r['mean_accuracy']:.4f} "
        f"+/- {r['std']:.4f}"
    )

print("=" * 70)

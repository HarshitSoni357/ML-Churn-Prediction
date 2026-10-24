
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SPLIT_DIR = Path("data/processed/splits")
ARTIFACT_DIR = Path("artifacts/random_forest")
TARGET = "inactive_next_30d"

DROP_COLUMNS = [
    "customer_id",
    "signup_date",
    "prediction_timestamp",
    TARGET,
]


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"scikit-learn version: {sklearn.__version__}")
    print("Loading train and validation splits...")

    train = pd.read_csv(SPLIT_DIR / "train.csv")
    validation = pd.read_csv(SPLIT_DIR / "validation.csv")

    X_train = train.drop(columns=DROP_COLUMNS)
    y_train = train[TARGET].astype(int)

    X_val = validation.drop(columns=DROP_COLUMNS)
    y_val = validation[TARGET].astype(int)

    categorical = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numeric = X_train.select_dtypes(include=["number"]).columns.tolist()

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Numeric features: {len(numeric)}")
    print(f"Categorical features: {len(categorical)}")

    # Fit imputers and one-hot encoding on training data only.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant", fill_value="__MISSING__"
        )),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])

    # Conservative starting point. We can tune these after establishing
    # a reliable baseline and measuring resource usage.
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight=None,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("\nFitting Random Forest...")
    started = time.time()
    model.fit(X_train, y_train)
    training_seconds = time.time() - started

    print(f"Training time: {training_seconds / 60:.2f} minutes")

    print("\nEvaluating on validation data...")
    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": "RandomForestClassifier",
        "sklearn_version": sklearn.__version__,
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "train_positive_rate": float(y_train.mean()),
        "validation_positive_rate": float(y_val.mean()),
        "training_seconds": round(training_seconds, 2),
        "roc_auc": float(roc_auc_score(y_val, probabilities)),
        "pr_auc": float(average_precision_score(y_val, probabilities)),
        "brier_score": float(brier_score_loss(y_val, probabilities)),
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y_val, predictions)),
        "precision": float(
            precision_score(y_val, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_val, predictions, zero_division=0)),
        "f1": float(f1_score(y_val, predictions, zero_division=0)),
    }

    joblib.dump(model, ARTIFACT_DIR / "model.joblib")

    with (ARTIFACT_DIR / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    print("\n=== RANDOM FOREST VALIDATION RESULTS ===")
    print(json.dumps(metrics, indent=2))

    # Permutation importance is more expensive; use native importance
    # for this initial diagnostic only.
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print("\n=== TOP 20 FEATURES (IMPURITY IMPORTANCE) ===")
    print(importance_df.head(20).to_string(index=False))

    importance_df.to_csv(
        ARTIFACT_DIR / "feature_importance.csv", index=False
    )

    print(f"\nArtifacts saved to: {ARTIFACT_DIR}")
    print("Test split has not been loaded or evaluated.")


if __name__ == "__main__":
    main()

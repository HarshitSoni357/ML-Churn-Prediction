
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd  # type: ignore[reportMissingModuleSource]
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SPLIT_DIR = Path("data/processed/splits")
ARTIFACT_DIR = Path("artifacts/baseline")
TARGET = "inactive_next_30d"

# IDs and raw timestamps are excluded. We already have engineered tenure_days.
DROP_COLUMNS = [
    "customer_id",
    "signup_date",
    "prediction_timestamp",
    TARGET,
]


def evaluate(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


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

    # Identify columns from the training schema only.
    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Numeric features: {len(numeric_columns)}")
    print(f"Categorical features: {len(categorical_columns)}")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant",
                                      fill_value="__MISSING__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    # No class weighting or threshold tuning in the first baseline.
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=100,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )

    # Majority-class benchmark: predict every row as class 0.
    majority_predictions = np.zeros(len(y_val), dtype=int)
    majority_metrics = {
        "accuracy": float(accuracy_score(y_val, majority_predictions)),
        "precision": float(
            precision_score(y_val, majority_predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_val, majority_predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_val, majority_predictions, zero_division=0)),
        "note": "ROC-AUC and PR-AUC omitted because this baseline has no scores.",
    }

    print("\nFitting Logistic Regression...")
    model.fit(X_train, y_train)

    print("Evaluating on validation data...")
    probabilities = model.predict_proba(X_val)[:, 1]
    logistic_metrics = evaluate(y_val, probabilities)

    results = {
        "sklearn_version": sklearn.__version__,
        "target": TARGET,
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(y_val)),
        "train_positive_rate": float(y_train.mean()),
        "validation_positive_rate": float(y_val.mean()),
        "majority_class_baseline": majority_metrics,
        "logistic_regression": logistic_metrics,
        "features": {
            "numeric": numeric_columns,
            "categorical": categorical_columns,
            "excluded": DROP_COLUMNS,
        },
    }

    with (ARTIFACT_DIR / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2)

    joblib.dump(model, ARTIFACT_DIR / "model.joblib")

    print("\n=== VALIDATION RESULTS ===")
    print(json.dumps(results, indent=2))
    print(f"\nModel saved to: {ARTIFACT_DIR / 'model.joblib'}")
    print(f"Metrics saved to: {ARTIFACT_DIR / 'validation_metrics.json'}")
    print("\nThe test split has not been loaded or evaluated.")


if __name__ == "__main__":
    main()

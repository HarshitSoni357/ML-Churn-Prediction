
import json
from pathlib import Path

import joblib
import pandas as pd
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
ARTIFACT_DIR = Path("artifacts/ablation")
TARGET = "inactive_next_30d"

DROP_COLUMNS = [
    "customer_id",
    "signup_date",
    "prediction_timestamp",
    TARGET,
]


def build_model(X: pd.DataFrame) -> Pipeline:
    categorical = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numeric = X.select_dtypes(include=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(
                        strategy="constant", fill_value="__MISSING__"
                    )),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ),
        ]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=100, random_state=42)),
    ])


def metrics(y, probabilities, threshold=0.125):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "threshold": threshold,
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "flagged_rate": float(predictions.mean()),
    }


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(SPLIT_DIR / "train.csv")
    val = pd.read_csv(SPLIT_DIR / "validation.csv")

    y_train = train[TARGET].astype(int)
    y_val = val[TARGET].astype(int)

    experiments = {
        "full_model": [],
        "without_customer_segment": ["customer_segment"],
    }

    results = {}

    for experiment_name, excluded in experiments.items():
        print(f"\nTraining experiment: {experiment_name}")

        X_train = train.drop(columns=DROP_COLUMNS + excluded)
        X_val = val.drop(columns=DROP_COLUMNS + excluded)

        model = build_model(X_train)
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_val)[:, 1]
        results[experiment_name] = metrics(y_val, probabilities)

        joblib.dump(model, ARTIFACT_DIR / f"{experiment_name}.joblib")

        print(json.dumps(results[experiment_name], indent=2))

    with (ARTIFACT_DIR / "comparison.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2)

    print("\n=== ABLATION COMPARISON ===")
    print(pd.DataFrame(results).T.round(4).to_string())
    print(f"\nSaved results to {ARTIFACT_DIR / 'comparison.json'}")
    print("Validation only; test data remains untouched.")


if __name__ == "__main__":
    main()

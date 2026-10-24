
from __future__ import annotations

import json
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
OUTPUT_DIR = ROOT / "artifacts" / "xgboost"

TARGET = "inactive_next_30d"
EXCLUDED = {
    "customer_id",
    "signup_date",
    "prediction_timestamp",
    TARGET,
}

CATEGORICAL_CANDIDATES = [
    "region",
    "customer_segment",
    "acquisition_channel",
    "subscription_plan",
    "billing_cycle",
]

PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "tree_method": "hist",
    "max_bin": 256,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training and validation data...")
    train = pd.read_csv(SPLIT_DIR / "train.csv")
    validation = pd.read_csv(SPLIT_DIR / "validation.csv")

    y_train = train[TARGET].astype("int32")
    y_val = validation[TARGET].astype("int32")

    # Numeric features only. Raw dates, identifiers and target are excluded.
    numeric_cols = [
        col
        for col in train.select_dtypes(include=["number"]).columns
        if col not in EXCLUDED
    ]

    categorical_cols = [
        col
        for col in CATEGORICAL_CANDIDATES
        if col in train.columns and col not in EXCLUDED
    ]

    feature_cols = numeric_cols + categorical_cols

    if not feature_cols:
        raise ValueError("No eligible model features found.")

    # Ensure train and validation use the same columns and ordering.
    X_train = train[feature_cols]
    X_val = validation[feature_cols]

    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="missing",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    classifier = XGBClassifier(**PARAMS)

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )

    # Keep MLflow's tracking store inside the project.
    tracking_db = ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
    mlflow.set_experiment("customer-inactivity-prediction")

    with mlflow.start_run(run_name="xgboost-baseline-v1") as run:
        mlflow.log_params(PARAMS)
        mlflow.log_param("numeric_feature_count", len(numeric_cols))
        mlflow.log_param(
            "categorical_feature_count", len(categorical_cols)
        )
        mlflow.log_param("preprocessing", "median_imputation_and_onehot")
        mlflow.log_param("target", TARGET)
        mlflow.log_param("train_rows", len(train))
        mlflow.log_param("validation_rows", len(validation))
        mlflow.log_param("train_positive_rate", float(y_train.mean()))
        mlflow.log_param("validation_positive_rate", float(y_val.mean()))
        mlflow.log_param("xgboost_version", xgboost.__version__)
        mlflow.log_param("pandas_version", pd.__version__)

        print("\nTraining XGBoost...")
        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        training_seconds = time.perf_counter() - started

        print("Scoring validation data...")
        probabilities = pipeline.predict_proba(X_val)[:, 1]

        metrics = {
            "roc_auc": float(roc_auc_score(y_val, probabilities)),
            "pr_auc": float(
                average_precision_score(y_val, probabilities)
            ),
            "brier_score": float(
                brier_score_loss(y_val, probabilities)
            ),
            "training_seconds": float(training_seconds),
        }

        mlflow.log_metrics(metrics)

        # Save a portable copy of the evaluation results.
        report = {
            "run_id": run.info.run_id,
            "experiment": "customer-inactivity-prediction",
            "model": "XGBoost",
            "metrics": metrics,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "numeric_features": numeric_cols,
            "categorical_features": categorical_cols,
        }

        with open(
            OUTPUT_DIR / "validation_metrics.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(report, file, indent=2)

        # Save feature importance with the transformed feature names.
        fitted_preprocessor = pipeline.named_steps["preprocessing"]
        transformed_names = fitted_preprocessor.get_feature_names_out()
        importances = pipeline.named_steps[
            "classifier"
        ].feature_importances_

        importance_df = pd.DataFrame(
            {
                "feature": transformed_names,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)

        importance_df.to_csv(
            OUTPUT_DIR / "feature_importance.csv",
            index=False,
        )

        mlflow.log_artifact(
            str(OUTPUT_DIR / "validation_metrics.json")
        )
        mlflow.log_artifact(
            str(OUTPUT_DIR / "feature_importance.csv")
        )

        # Use a realistic example with floating-point numeric columns.
                # Use a realistic example with floating-point numeric columns.
        input_example = X_train.head(3).copy()

        input_example = X_train.head(3).copy()
        numeric_columns = input_example.select_dtypes(include=["number"]).columns
        input_example[numeric_columns] = input_example[numeric_columns].astype("float64")

        # Explicitly trust only the XGBoost types used by our own model.
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="xgboost_pipeline",
            input_example=input_example,
            skops_trusted_types=[
                "numpy.dtype",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )

        print("\nExperiment completed.")
        print(f"Run ID: {run.info.run_id}")
        print(f"Model artifact: {OUTPUT_DIR}")
        print("\nValidation metrics:")

        for name, value in metrics.items():
            print(f"{name}: {value:.6f}")

        print("\nTop 15 features:")
        print(importance_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()

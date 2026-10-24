
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
OUTPUT_DIR = ROOT / "artifacts" / "segment_vs_behavior"
TARGET = "inactive_next_30d"

train_df = pd.read_csv(SPLIT_DIR / "train.csv")
validation_df = pd.read_csv(SPLIT_DIR / "validation.csv")

# Never allow identifiers, dates, or the target into the feature set.
EXCLUDED = {
    "customer_id",
    "prediction_timestamp",
    "signup_date",
    TARGET,
}

# Match the categorical columns used in the baseline model.
BASELINE_CATEGORICAL = [
    "region",
    "customer_segment",
    "acquisition_channel",
    "subscription_plan",
    "billing_cycle",
]

# These are customer behavior and activity signals.
BEHAVIOR_PREFIXES = (
    "orders_",
    "payments_",
    "support_",
    "marketing_",
    "feedback_",
)

BEHAVIOR_EXACT = {"days_since_last_order"}


def get_feature_groups(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Return numeric and categorical features, excluding raw dates."""
    numeric_cols = [
        col
        for col in df.select_dtypes(include=["number"]).columns
        if col not in EXCLUDED
    ]

    categorical_cols = [
        col
        for col in BASELINE_CATEGORICAL
        if col in df.columns and col not in EXCLUDED
    ]

    return numeric_cols, categorical_cols


def build_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    """Build the same preprocessing and Logistic Regression approach."""
    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(
            ("numeric", numeric_pipeline, numeric_cols)
        )

    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="missing"),
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_cols)
        )

    preprocessing = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_model(
    name: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> dict:
    """Fit on train data and evaluate on the untouched validation period."""
    feature_cols = numeric_cols + categorical_cols

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET].astype(int)
    X_val = validation_df[feature_cols]
    y_val = validation_df[TARGET].astype(int)

    model = build_pipeline(numeric_cols, categorical_cols)

    started = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - started

    probabilities = model.predict_proba(X_val)[:, 1]

    results = {
        "experiment": name,
        "feature_count": len(feature_cols),
        "numeric_feature_count": len(numeric_cols),
        "categorical_feature_count": len(categorical_cols),
        "roc_auc": float(roc_auc_score(y_val, probabilities)),
        "pr_auc": float(average_precision_score(y_val, probabilities)),
        "brier_score": float(brier_score_loss(y_val, probabilities)),
        "training_seconds": round(training_seconds, 2),
        "validation_rows": len(y_val),
    }

    print(f"\n{name}")
    print("-" * len(name))
    for key, value in results.items():
        if key != "experiment":
            print(f"{key}: {value}")

    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    numeric_cols, categorical_cols = get_feature_groups(train_df)

    # 1. Segment-only: tests how much signal segment alone provides.
    segment_cols = (
        ["customer_segment"]
        if "customer_segment" in train_df.columns
        else []
    )

    # 2. Behavior-only: activity and interaction signals.
    behavior_cols = [
        col
        for col in train_df.columns
        if (
            col.startswith(BEHAVIOR_PREFIXES)
            or col in BEHAVIOR_EXACT
        )
        and col not in EXCLUDED
    ]

    behavior_numeric = [
        col for col in behavior_cols
        if col in train_df.select_dtypes(include=["number"]).columns
    ]
    behavior_categorical = [
        col for col in behavior_cols
        if col in train_df.select_dtypes(
            include=["object", "category", "string"]
        ).columns
        and "date" not in col.lower()
        and "timestamp" not in col.lower()
    ]

    experiments = [
        (
            "segment_only",
            [],
            segment_cols,
        ),
        (
            "behavior_only",
            behavior_numeric,
            behavior_categorical,
        ),
        (
            "full_model",
            numeric_cols,
            categorical_cols,
        ),
    ]

    results = [
        evaluate_model(name, nums, cats)
        for name, nums, cats in experiments
    ]

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        OUTPUT_DIR / "comparison.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "comparison.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    print("\nSaved results to:", OUTPUT_DIR)
    print(
        results_df[
            ["experiment", "roc_auc", "pr_auc", "brier_score"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

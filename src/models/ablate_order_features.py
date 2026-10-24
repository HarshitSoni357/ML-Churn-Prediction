
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SPLIT_DIR = Path("data/processed/splits")
FULL_MODEL_PATH = Path("artifacts/random_forest/model.joblib")
OUTPUT_DIR = Path("artifacts/order_ablation")
TARGET = "inactive_next_30d"

DROP_COLUMNS = [
    "customer_id",
    "signup_date",
    "prediction_timestamp",
    TARGET,
]

# The first experiment removes only count/sum/mean.
AGGREGATE_90D = [
    "orders_count_90d",
    "orders_sum_90d",
    "orders_mean_90d",
]

# The second removes every 90-day order-window feature,
# but deliberately keeps days_since_last_order.
ALL_ORDER_WINDOW_90D = [
    "orders_count_90d",
    "orders_sum_90d",
    "orders_mean_90d",
    "orders_category_electronics_90d",
    "orders_category_home_90d",
    "orders_category_services_90d",
    "orders_category_appliances_90d",
    "orders_category_other_90d",
]


def evaluate(y, probabilities):
    # Rank exactly 30% of validation rows, avoiding threshold ties.
    n_flagged = int(round(0.30 * len(y)))
    ranking = np.argsort(-probabilities, kind="stable")
    predictions = np.zeros(len(y), dtype=int)
    predictions[ranking[:n_flagged]] = 1

    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "capacity_pct": float(predictions.mean() * 100),
        "precision_at_30pct": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall_at_30pct": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1_at_30pct": float(f1_score(y, predictions, zero_division=0)),
    }


def build_model(X):
    categorical = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numeric = X.select_dtypes(include=["number"]).columns.tolist()

    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
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
    ])

    return Pipeline([
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(SPLIT_DIR / "train.csv")
    validation = pd.read_csv(SPLIT_DIR / "validation.csv")

    y_train = train[TARGET].astype(int)
    y_val = validation[TARGET].astype(int).to_numpy()

    experiments = {
        "without_90d_aggregates": AGGREGATE_90D,
        "without_all_90d_order_features": ALL_ORDER_WINDOW_90D,
    }

    results = {}

    # Score the existing full model on the same validation set.
    print("Evaluating existing full Random Forest...")
    full_model = joblib.load(FULL_MODEL_PATH)
    X_full_val = validation.drop(columns=DROP_COLUMNS)
    full_probabilities = full_model.predict_proba(X_full_val)[:, 1]
    results["full_model"] = evaluate(y_val, full_probabilities)

    print(json.dumps(results["full_model"], indent=2))

    # Train the two ablation models using the same configuration.
    for name, excluded_features in experiments.items():
        print(f"\n=== Training {name} ===")

        X_train = train.drop(columns=DROP_COLUMNS + excluded_features)
        X_val = validation.drop(columns=DROP_COLUMNS + excluded_features)

        model = build_model(X_train)

        started = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - started

        probabilities = model.predict_proba(X_val)[:, 1]
        results[name] = evaluate(y_val, probabilities)
        results[name]["training_seconds"] = round(elapsed, 2)
        results[name]["removed_features"] = excluded_features

        joblib.dump(model, OUTPUT_DIR / f"{name}.joblib")

        print(json.dumps(results[name], indent=2))

    with (OUTPUT_DIR / "comparison.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2)

    comparison = pd.DataFrame({
        name: {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        for name, metrics in results.items()
    }).T

    print("\n=== ORDER FEATURE ABLATION COMPARISON ===")
    print(comparison.round(4).to_string())
    print(f"\nResults saved to {OUTPUT_DIR / 'comparison.json'}")
    print("Validation only; test data remains untouched.")


if __name__ == "__main__":
    main()

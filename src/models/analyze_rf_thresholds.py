
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

SPLIT_DIR = Path("data/processed/splits")
MODEL_PATH = Path("artifacts/random_forest/model.joblib")
OUTPUT_PATH = Path("artifacts/random_forest/threshold_analysis.csv")
TARGET = "inactive_next_30d"

DROP_COLUMNS = [
    "customer_id",
    "signup_date",
    "prediction_timestamp",
    TARGET,
]


def main():
    model = joblib.load(MODEL_PATH)
    validation = pd.read_csv(SPLIT_DIR / "validation.csv")

    X = validation.drop(columns=DROP_COLUMNS)
    y = validation[TARGET].astype(int).to_numpy()
    probabilities = model.predict_proba(X)[:, 1]

    print("=== RANDOM FOREST PROBABILITY DISTRIBUTION ===")
    print(
        pd.Series(probabilities)
        .describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
        .round(4)
        .to_string()
    )
    print(f"\nValidation positive rate: {y.mean():.4%}")
    print(f"PR-AUC: {average_precision_score(y, probabilities):.4f}")

    # Evaluate operational capacity: contact the highest-scoring customers.
    results = []
    for capacity in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        threshold = float(np.quantile(probabilities, 1 - capacity))
        predictions = (probabilities >= threshold).astype(int)

        results.append({
            "target_flagged_rate": capacity,
            "actual_flagged_rate": float(predictions.mean()),
            "threshold": threshold,
            "precision": float(
                precision_score(y, predictions, zero_division=0)
            ),
            "recall": float(recall_score(y, predictions, zero_division=0)),
            "f1": float(f1_score(y, predictions, zero_division=0)),
            "true_positives": int(((predictions == 1) & (y == 1)).sum()),
            "false_positives": int(((predictions == 1) & (y == 0)).sum()),
            "false_negatives": int(((predictions == 0) & (y == 1)).sum()),
        })

    table = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)

    print("\n=== RANDOM FOREST: OUTREACH CAPACITY ANALYSIS ===")
    print(table.round(4).to_string(index=False))
    print(f"\nSaved to {OUTPUT_PATH}")
    print("Validation only; test data remains untouched.")


if __name__ == "__main__":
    main()

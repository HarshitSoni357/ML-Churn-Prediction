
import json
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
ARTIFACT_DIR = Path("artifacts/baseline")
TARGET = "inactive_next_30d"

DROP_COLUMNS = [
    "customer_id",
    "signup_date",
    "prediction_timestamp",
    TARGET,
]


def main():
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    validation = pd.read_csv(SPLIT_DIR / "validation.csv")

    X_val = validation.drop(columns=DROP_COLUMNS)
    y_val = validation[TARGET].astype(int).to_numpy()

    probabilities = model.predict_proba(X_val)[:, 1]

    print("=== PROBABILITY DISTRIBUTION ===")
    print(
        pd.Series(probabilities)
        .describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
        .round(4)
        .to_string()
    )
    print(f"\nActual positive rate: {y_val.mean():.4%}")
    print(f"PR-AUC: {average_precision_score(y_val, probabilities):.4f}")

    # Explore thresholds without touching the test set.
    thresholds = np.round(np.arange(0.05, 0.501, 0.025), 3)
    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        tp = int(((predictions == 1) & (y_val == 1)).sum())
        fp = int(((predictions == 1) & (y_val == 0)).sum())
        fn = int(((predictions == 0) & (y_val == 1)).sum())

        results.append(
            {
                "threshold": float(threshold),
                "flagged_customers": int(predictions.sum()),
                "flagged_pct": float(predictions.mean()),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": float(
                    precision_score(y_val, predictions, zero_division=0)
                ),
                "recall": float(
                    recall_score(y_val, predictions, zero_division=0)
                ),
                "f1": float(f1_score(y_val, predictions, zero_division=0)),
            }
        )

    table = pd.DataFrame(results)
    table.to_csv(ARTIFACT_DIR / "threshold_analysis.csv", index=False)

    print("\n=== THRESHOLD COMPARISON ===")
    print(
        table[
            [
                "threshold",
                "flagged_pct",
                "precision",
                "recall",
                "f1",
                "false_positives",
                "false_negatives",
            ]
        ].round(4).to_string(index=False)
    )

    best_f1 = table.loc[table["f1"].idxmax()]
    print("\n=== BEST F1 AMONG TESTED THRESHOLDS ===")
    print(best_f1.to_string())

    eligible = table[table["recall"] >= 0.50]
    if not eligible.empty:
        # Among tested thresholds with recall >= 50%, find the one
        # with the highest precision.
        choice = eligible.loc[eligible["precision"].idxmax()]
        print("\n=== HIGHEST PRECISION WITH RECALL >= 50% ===")
        print(choice.to_string())
    else:
        print("\nNo tested threshold achieved recall >= 50%.")

    print(
        "\nSaved threshold analysis to "
        f"{ARTIFACT_DIR / 'threshold_analysis.csv'}"
    )
    print("Validation only; test data remains untouched.")


if __name__ == "__main__":
    main()

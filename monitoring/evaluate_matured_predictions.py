
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = (
    ROOT / "artifacts" / "monitoring" / "matured_prediction_ledger.csv"
)


def main():
    df = pd.read_csv(LEDGER_PATH)

    # Only rows with known actual outcomes can be evaluated.
    df = df.dropna(
        subset=["actual_label", "inactivity_probability"]
    ).copy()

    if df.empty:
        raise ValueError("No matured predictions available for evaluation.")

    y_true = df["actual_label"].astype(int)
    y_score = df["inactivity_probability"].astype(float)

    if not y_true.isin([0, 1]).all():
        raise ValueError("Actual labels must be binary.")

    if not y_score.between(0, 1).all():
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    if y_true.nunique() != 2:
        raise ValueError("ROC-AUC requires both outcome classes.")

    print("Model:", df["model_name"].iloc[0])
    print("Version:", df["model_version"].iloc[0])
    print("Evaluated predictions:", len(df))
    print("Inactive rate:", round(y_true.mean(), 4))
    print("ROC-AUC:", round(roc_auc_score(y_true, y_score), 5))
    print("PR-AUC (average precision):", round(
        average_precision_score(y_true, y_score), 5
    ))
    print("Brier score:", round(brier_score_loss(y_true, y_score), 5))


if __name__ == "__main__":
    main()

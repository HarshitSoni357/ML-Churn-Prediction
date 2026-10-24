
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = ROOT / "data" / "processed" / "splits" / "test.csv"
MODEL_PATH = ROOT / "artifacts" / "random_forest" / "model.joblib"
OUTPUT_DIR = ROOT / "artifacts" / "final_evaluation"
DB_PATH = ROOT / "mlflow.db"

TARGET = "inactive_next_30d"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Selected model not found: {MODEL_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_PATH}")

    # Load the already-fitted model. Do not retrain on test data.
    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "Model does not expose feature_names_in_. "
            "Inspect the saved pipeline before proceeding."
        )

    feature_cols = list(model.feature_names_in_)
    required_cols = set(feature_cols) | {TARGET}

    # Read only required columns to reduce memory usage.
    test = pd.read_csv(
        TEST_PATH,
        usecols=lambda col: col in required_cols,
    )

    missing = required_cols - set(test.columns)
    if missing:
        raise ValueError(f"Missing required test columns: {sorted(missing)}")

    X_test = test[feature_cols]
    y_test = test[TARGET].astype(int)

    if y_test.nunique() != 2:
        raise ValueError("Test target must contain both classes.")

    print(f"Selected model: Random Forest")
    print(f"Test rows: {len(test):,}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Test positive rate: {y_test.mean():.4%}")
    print("\nScoring the held-out test set...")

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
    }

    report = {
        "evaluation_type": "final_held_out_test",
        "selected_model": "random_forest",
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "test_rows": len(test),
        "feature_count": len(feature_cols),
        "test_positive_rate": float(y_test.mean()),
        "metrics": metrics,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Test metrics are for final evaluation, not for iterative "
            "model tuning. Threshold selection remains a separate "
            "business decision."
        ),
    }

    report_path = OUTPUT_DIR / "random_forest_test_metrics.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    # Record final evaluation in the same local MLflow tracking database.
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.as_posix()}")
    mlflow.set_experiment("customer-inactivity-prediction")

    with mlflow.start_run(
        run_name="random-forest-final-test-evaluation"
    ):
        mlflow.log_param("selected_model", "random_forest")
        mlflow.log_param("evaluation_split", "test")
        mlflow.log_param("test_rows", len(test))
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(report_path))

    print("\nFinal held-out test metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    print(f"\nSaved report: {report_path}")
    print("MLflow test-evaluation run logged successfully.")


if __name__ == "__main__":
    main()

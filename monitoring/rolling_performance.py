
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "weekly_customer_features.csv"
MODEL_PATH = ROOT / "artifacts" / "deployment" / "random_forest_v1.joblib"
METADATA_PATH = ROOT / "artifacts" / "deployment" / "model_metadata.json"
OUTPUT_DIR = ROOT / "artifacts" / "monitoring"

# Use the previously held-out test period.
TEST_START = pd.Timestamp("2026-04-06", tz="UTC")
WINDOW_DAYS = 60
STEP_DAYS = 7
CHUNK_SIZE = 50_000


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = OUTPUT_DIR / "test_period_predictions.csv"
    report_path = OUTPUT_DIR / "rolling_performance.csv"

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    features = metadata.get("features", metadata.get("feature_names"))
    if not features:
        raise ValueError("Model metadata has no feature list.")

    model = joblib.load(MODEL_PATH)
    positive_index = list(model.classes_).index(1)

    columns = [
        "customer_id",
        "prediction_timestamp",
        "inactive_next_30d",
        *features,
    ]

    # Score the held-out test period in chunks.
    # Recreate this output on each run so results are deterministic.
    if predictions_path.exists():
        predictions_path.unlink()

    rows_scored = 0
    as_of = pd.Timestamp.now(tz="UTC")

    for chunk in pd.read_csv(
        DATA_PATH,
        usecols=columns,
        chunksize=CHUNK_SIZE,
    ):
        timestamps = pd.to_datetime(
            chunk["prediction_timestamp"], utc=True
        )

        mask = timestamps >= TEST_START
        if not mask.any():
            continue

        selected = chunk.loc[mask].copy()
        selected_timestamps = timestamps.loc[mask]

        # Do not evaluate labels before their 30-day outcome window closes.
        mature_mask = (
            selected_timestamps + pd.Timedelta(days=30) <= as_of
        )
        selected = selected.loc[mature_mask].copy()
        selected_timestamps = selected_timestamps.loc[mature_mask]

        if selected.empty:
            continue

        probabilities = model.predict_proba(selected[features])[
            :, positive_index
        ]

        output = pd.DataFrame({
            "customer_id": selected["customer_id"].astype(str).to_numpy(),
            "prediction_timestamp": selected_timestamps.to_numpy(),
            "inactivity_probability": probabilities,
            "actual_label": selected["inactive_next_30d"].astype(int).to_numpy(),
        })

        output.to_csv(
            predictions_path,
            mode="a",
            header=not predictions_path.exists(),
            index=False,
        )
        rows_scored += len(output)
        print(f"Scored {rows_scored:,} matured test predictions")

    if rows_scored == 0:
        raise ValueError("No matured test-period predictions were scored.")

    # This compact file is small enough to load for window calculations.
    predictions = pd.read_csv(predictions_path)
    predictions["prediction_timestamp"] = pd.to_datetime(
        predictions["prediction_timestamp"], utc=True
    )

    first_date = predictions["prediction_timestamp"].min()
    last_date = predictions["prediction_timestamp"].max()

    # Evaluate at weekly intervals once a complete 60-day window exists.
    window_end = first_date + pd.Timedelta(days=WINDOW_DAYS - 1)
    rows = []

    while window_end <= last_date:
        window_start = window_end - pd.Timedelta(days=WINDOW_DAYS - 1)

        window = predictions[
            predictions["prediction_timestamp"].between(
                window_start, window_end, inclusive="both"
            )
        ]

        y_true = window["actual_label"]
        y_score = window["inactivity_probability"]

        row = {
            "window_start": window_start.date().isoformat(),
            "window_end": window_end.date().isoformat(),
            "sample_count": len(window),
            "inactive_rate": (
                float(y_true.mean()) if len(window) else np.nan
            ),
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "brier_score": np.nan,
            "status": "ok",
        }

        if len(window) == 0:
            row["status"] = "no_data"
        elif y_true.nunique() < 2:
            row["status"] = "one_class_only"
        else:
            row["roc_auc"] = roc_auc_score(y_true, y_score)
            row["pr_auc"] = average_precision_score(y_true, y_score)
            row["brier_score"] = brier_score_loss(y_true, y_score)

        rows.append(row)
        window_end += pd.Timedelta(days=STEP_DAYS)

    report = pd.DataFrame(rows)
    report.to_csv(report_path, index=False)

    print("\nRolling performance report")
    print(report.to_string(index=False))
    print(f"\nPredictions: {predictions_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

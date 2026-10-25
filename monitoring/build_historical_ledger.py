
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "weekly_customer_features.csv"
MODEL_PATH = ROOT / "artifacts" / "deployment" / "random_forest_v1.joblib"
METADATA_PATH = ROOT / "artifacts" / "deployment" / "model_metadata.json"
OUTPUT_PATH = ROOT / "artifacts" / "monitoring" / "historical_prediction_ledger.csv"

SAMPLE_SIZE = 5_000
RANDOM_STATE = 42


def main():
    import json

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    features = metadata.get("features", metadata.get("feature_names"))

    if not features:
        raise ValueError("No feature list found in model metadata.")

    # Read only the columns needed for scoring and matching.
    columns = [
        "customer_id",
        "prediction_timestamp",
        "inactive_next_30d",
        *features,
    ]

    data = pd.read_csv(DATA_PATH, usecols=columns)

    # Reproducible sample for our first implementation.
    sample = data.sample(
        n=min(SAMPLE_SIZE, len(data)),
        random_state=RANDOM_STATE,
    ).copy()

    model = joblib.load(MODEL_PATH)

    # Keep model input order identical to training.
    probabilities = model.predict_proba(sample[features])[
        :, list(model.classes_).index(1)
    ]

    sample["prediction_id"] = (
        sample["customer_id"].astype(str)
        + "|"
        + sample["prediction_timestamp"].astype(str)
        + "|"
        + str(metadata["model_name"])
        + "|"
        + str(metadata["model_version"])
    )

    sample["model_name"] = metadata["model_name"]
    sample["model_version"] = str(metadata["model_version"])
    sample["inactivity_probability"] = probabilities

    # The label is intentionally kept separate from the prediction records.
    predictions = sample[
        [
            "prediction_id",
            "customer_id",
            "prediction_timestamp",
            "model_name",
            "model_version",
            "inactivity_probability",
        ]
    ].copy()

    outcomes = sample[
        [
            "customer_id",
            "prediction_timestamp",
            "inactive_next_30d",
        ]
    ].copy()

    predictions["prediction_timestamp"] = pd.to_datetime(
        predictions["prediction_timestamp"], utc=True
    )
    predictions["outcome_window_end"] = (
        predictions["prediction_timestamp"] + pd.Timedelta(days=30)
    )
    predictions["actual_label"] = pd.NA
    predictions["label_available_at"] = pd.NA

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_PATH, index=False)

    # Save historical labels separately to simulate outcome maturation.
    outcomes_path = OUTPUT_PATH.with_name("historical_outcomes.csv")
    outcomes.to_csv(outcomes_path, index=False)

    print(f"Prediction records: {len(predictions):,}")
    print(f"Unique prediction IDs: {predictions['prediction_id'].nunique():,}")
    print(f"Predictions file: {OUTPUT_PATH}")
    print(f"Historical outcomes file: {outcomes_path}")
    print("\nSample predictions:")
    print(predictions.head().to_string(index=False))


if __name__ == "__main__":
    main()

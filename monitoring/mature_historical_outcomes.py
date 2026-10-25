
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

LEDGER_PATH = (
    ROOT / "artifacts" / "monitoring" / "historical_prediction_ledger.csv"
)
OUTCOMES_PATH = (
    ROOT / "artifacts" / "monitoring" / "historical_outcomes.csv"
)
OUTPUT_PATH = (
    ROOT / "artifacts" / "monitoring" / "matured_prediction_ledger.csv"
)


def main():
    ledger = pd.read_csv(LEDGER_PATH)
    outcomes = pd.read_csv(OUTCOMES_PATH)

    ledger["customer_id"] = ledger["customer_id"].astype(str)
    outcomes["customer_id"] = outcomes["customer_id"].astype(str)

    ledger["prediction_timestamp"] = pd.to_datetime(
        ledger["prediction_timestamp"], utc=True
    )
    outcomes["prediction_timestamp"] = pd.to_datetime(
        outcomes["prediction_timestamp"], utc=True
    )
    ledger["outcome_window_end"] = pd.to_datetime(
        ledger["outcome_window_end"], utc=True
    )

    if outcomes.duplicated(
        ["customer_id", "prediction_timestamp"]
    ).any():
        raise ValueError("Duplicate keys found in historical outcomes.")

    now = pd.Timestamp(datetime.now(timezone.utc))

    # Only evaluate snapshots whose complete 30-day window has elapsed.
    eligible = ledger["outcome_window_end"] <= now
    pending = ledger.loc[~eligible].copy()
    eligible_predictions = ledger.loc[eligible].copy()

    # Join the actual labels to the eligible predictions.
    joined = eligible_predictions.merge(
        outcomes,
        on=["customer_id", "prediction_timestamp"],
        how="left",
        validate="one_to_one",
    )

    if joined["inactive_next_30d"].isna().any():
        raise ValueError(
            "Some eligible predictions have no matching historical label."
        )

    joined["actual_label"] = joined["inactive_next_30d"].astype(int)
    joined["label_available_at"] = now.isoformat()
    joined = joined.drop(columns=["inactive_next_30d"])

    # Keep future-window predictions unlabeled.
    pending["actual_label"] = pd.NA
    pending["label_available_at"] = pd.NA

    result = pd.concat([joined, pending], ignore_index=True)
    result = result.sort_values(
        ["prediction_timestamp", "customer_id"]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("Total predictions:", len(result))
    print("Matured predictions:", len(joined))
    print("Still pending:", len(pending))
    print("Matured label counts:")
    print(joined["actual_label"].value_counts().sort_index())
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()

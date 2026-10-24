
from pathlib import Path

import pandas as pd # type: ignore


INPUT_PATH = Path("data/processed/weekly_customer_features.csv")
OUTPUT_DIR = Path("data/processed/splits")

TRAIN_END = pd.Timestamp("2025-12-01")
VALIDATION_START = pd.Timestamp("2026-01-05")
VALIDATION_END = pd.Timestamp("2026-03-02")
TEST_START = pd.Timestamp("2026-04-06")
LABEL_HORIZON_DAYS = 30


def main() -> None:
    print("Loading validated feature dataset...")
    df = pd.read_csv(INPUT_PATH)
    df["prediction_timestamp"] = pd.to_datetime(
        df["prediction_timestamp"], errors="raise"
    )

    # Enforce the embargo needed for the 30-day forward-looking label.
    require_gap = (
        VALIDATION_START - TRAIN_END
    ).days > LABEL_HORIZON_DAYS and (
        TEST_START - VALIDATION_END
    ).days > LABEL_HORIZON_DAYS

    if not require_gap:
        raise ValueError("Temporal split gaps are too short for the label horizon.")

    train = df[df["prediction_timestamp"] <= TRAIN_END].copy()

    validation = df[
        df["prediction_timestamp"].between(
            VALIDATION_START, VALIDATION_END, inclusive="both"
        )
    ].copy()

    test = df[df["prediction_timestamp"] >= TEST_START].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("At least one split is empty. Check the date boundaries.")

    # Check that no label window crosses into the next split.
    if (
        train["prediction_timestamp"].max()
        + pd.Timedelta(days=LABEL_HORIZON_DAYS)
        >= validation["prediction_timestamp"].min()
    ):
        raise ValueError("Training labels overlap the validation period.")

    if (
        validation["prediction_timestamp"].max()
        + pd.Timedelta(days=LABEL_HORIZON_DAYS)
        >= test["prediction_timestamp"].min()
    ):
        raise ValueError("Validation labels overlap the test period.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    summary = []

    for name, part in splits.items():
        path = OUTPUT_DIR / f"{name}.csv"
        part.to_csv(path, index=False)

        summary.append(
            {
                "split": name,
                "rows": len(part),
                "unique_customers": part["customer_id"].nunique(),
                "first_snapshot": part["prediction_timestamp"].min(),
                "last_snapshot": part["prediction_timestamp"].max(),
                "positive_rate": part["inactive_next_30d"].mean(),
            }
        )

        print(
            f"{name.upper():10} | rows={len(part):,} | "
            f"positive_rate={part['inactive_next_30d'].mean():.4%} | "
            f"{part['prediction_timestamp'].min().date()} to "
            f"{part['prediction_timestamp'].max().date()}"
        )

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)

    excluded = len(df) - sum(len(part) for part in splits.values())

    print(f"\nRows excluded by embargo: {excluded:,}")
    print(f"Split files written to: {OUTPUT_DIR}")
    print("PASS: temporal boundaries and label-window embargoes")


if __name__ == "__main__":
    main()

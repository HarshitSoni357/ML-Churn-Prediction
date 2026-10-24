
from pathlib import Path

import numpy as np
import pandas as pd # type: ignore


FEATURE_PATH = Path("data/processed/weekly_customer_features.csv")
ORDERS_PATH = Path("data/raw/orders.csv")
REPORT_PATH = Path("data/processed/feature_validation_missingness.csv")

LABEL = "inactive_next_30d"
KEYS = ["customer_id", "prediction_timestamp"]


def require(condition: bool, message: str) -> None:
    """Stop validation when a required invariant fails."""
    if not condition:
        raise ValueError(f"VALIDATION FAILED: {message}")


def validate_features() -> None:
    print("Loading feature dataset...")
    df = pd.read_csv(FEATURE_PATH)

    required_columns = KEYS + [
        LABEL,
        "orders_count_7d",
        "orders_count_30d",
        "orders_count_90d",
        "days_since_last_order",
    ]
    missing_columns = sorted(set(required_columns) - set(df.columns))
    require(not missing_columns, f"Missing required columns: {missing_columns}")

    # 1. Structural integrity
    print("\n[1/6] Checking structural integrity...")
    require(not df[KEYS].isna().any().any(), "Null customer/timestamp key")
    require(not df.duplicated(KEYS).any(), "Duplicate customer-week keys")
    require(df[LABEL].notna().all(), "Missing target labels")
    require(set(df[LABEL].unique()).issubset({0, 1}), "Target is not binary")

    df["prediction_timestamp"] = pd.to_datetime(
        df["prediction_timestamp"], errors="coerce"
    )
    require(
        df["prediction_timestamp"].notna().all(),
        "Invalid prediction timestamps",
    )

    # Prediction cadence is weekly on Mondays.
    require(
        df["prediction_timestamp"].dt.dayofweek.eq(0).all(),
        "Found prediction timestamps that are not Mondays",
    )

    require(
        df["signup_date"].notna().all(),
        "Missing signup dates",
    )
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    require(df["signup_date"].notna().all(), "Invalid signup dates")
    require(
        (df["signup_date"] <= df["prediction_timestamp"] - pd.Timedelta(days=30)).all(),
        "Found snapshots before the 30-day customer eligibility threshold",
    )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Unique customers: {df['customer_id'].nunique():,}")
    print(
        "Snapshot dates:",
        df["prediction_timestamp"].min(),
        "to",
        df["prediction_timestamp"].max(),
    )
    print(f"Positive label rate: {df[LABEL].mean():.4%}")

    # 2. Numeric validity
    print("\n[2/6] Checking numeric values...")

    # Allow legitimate NaN values, but reject positive/negative infinity.
    numeric = df.select_dtypes(include="number")
    infinite_counts = {
        col: int(np.isinf(df[col].dropna().to_numpy()).sum())
        for col in numeric.columns
    }
    infinite_counts = {
        col: count
        for col, count in infinite_counts.items()
        if count > 0
    }
    require(
        not infinite_counts,
        f"Infinite numeric values found: {infinite_counts}",
    )


    # 3. Nested order windows
    print("\n[3/6] Checking order-window consistency...")
    require(
        (df["orders_count_7d"] <= df["orders_count_30d"]).all(),
        "7-day order count exceeds 30-day count",
    )
    require(
        (df["orders_count_30d"] <= df["orders_count_90d"]).all(),
        "30-day order count exceeds 90-day count",
    )
    require(
        df["days_since_last_order"].notna().all(),
        "Missing days_since_last_order for eligible customers",
    )

    # 4. Missingness report — missing values are reported, not automatically rejected.
    print("\n[4/6] Creating missingness report...")
    missingness = (
        df.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .rename("missing_pct")
        .rename_axis("feature")
        .reset_index()
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    missingness.to_csv(REPORT_PATH, index=False)
    print(missingness.head(15).to_string(index=False))
    print(f"Full missingness report: {REPORT_PATH}")

    # 5. Recompute the target independently from raw order events.
    print("\n[5/6] Recomputing labels from raw orders...")
    orders = pd.read_csv(ORDERS_PATH)

    expected_order_columns = {
        "customer_id",
        "order_timestamp",
        "order_value",
        "product_category",
    }
    missing_order_columns = expected_order_columns - set(orders.columns)
    require(
        not missing_order_columns,
        f"Unexpected orders schema; missing columns: {sorted(missing_order_columns)}",
    )

    orders["order_timestamp"] = pd.to_datetime(
        orders["order_timestamp"], errors="coerce"
    )
    require(
        orders["order_timestamp"].notna().all(),
        "Raw orders contain invalid timestamps",
    )

    # Match the feature builder's business-event deduplication rule.
    dedup_keys = [
        "customer_id",
        "order_timestamp",
        "order_value",
        "product_category",
    ]
    original_order_rows = len(orders)
    orders = orders.drop_duplicates(subset=dedup_keys)
    print(
        f"Deduplicated {original_order_rows - len(orders):,} "
        "duplicate business events for label validation"
    )

    orders_by_customer = {
        customer_id: np.sort(
            group["order_timestamp"].to_numpy(dtype="datetime64[ns]")
        )
        for customer_id, group in orders.groupby("customer_id", sort=False)
    }

    mismatches = 0
    checked = 0
    horizon = np.timedelta64(30, "D")

    # Process one customer at a time to avoid a huge many-to-many join.
    for customer_id, group in df.groupby("customer_id", sort=False):
        prediction_times = group["prediction_timestamp"].to_numpy(
            dtype="datetime64[ns]"
        )
        order_times = orders_by_customer.get(customer_id)

        if order_times is None or len(order_times) == 0:
            expected_labels = np.ones(len(group), dtype=np.int8)
        else:
            # Open lower boundary: order_timestamp > T
            lower = np.searchsorted(
                order_times, prediction_times, side="right"
            )
            # Closed upper boundary: order_timestamp <= T + 30 days
            upper = np.searchsorted(
                order_times, prediction_times + horizon, side="right"
            )
            future_order_counts = upper - lower
            expected_labels = (future_order_counts == 0).astype(np.int8)

        actual_labels = group[LABEL].to_numpy(dtype=np.int8)
        mismatches += int(np.count_nonzero(actual_labels != expected_labels))
        checked += len(group)

    require(
        mismatches == 0,
        f"{mismatches:,} target labels disagree with raw order history",
    )
    print(f"Verified labels: {checked:,}; mismatches: {mismatches:,}")

    # 6. Final summary
    print("\n[6/6] Validation complete.")
    print("PASS: structural integrity")
    print("PASS: numeric validity")
    print("PASS: order-window consistency")
    print("PASS: missingness report generated")
    print("PASS: labels independently verified against raw orders")
    print(
        "NOTE: feature-window boundaries are also covered by "
        "tests/test_temporal_boundaries.py."
    )


if __name__ == "__main__":
    validate_features()

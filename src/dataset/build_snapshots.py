
from pathlib import Path
import argparse
import pandas as pd # type: ignore


def load_table(data_dir: Path, name: str) -> pd.DataFrame:
    csv_file = data_dir / f"{name}.csv"
    parquet_file = data_dir / f"{name}.parquet"

    if csv_file.exists():
        return pd.read_csv(csv_file)

    if parquet_file.exists():
        return pd.read_parquet(parquet_file)

    raise FileNotFoundError(
        f"Cannot find {name}.csv or {name}.parquet in {data_dir}"
    )


def build_snapshots(data_dir: Path, output_dir: Path) -> None:
    customers = load_table(data_dir, "customers")
    orders = load_table(data_dir, "orders")

    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], utc=True
    ).dt.tz_localize(None)

    orders["order_timestamp"] = pd.to_datetime(
        orders["order_timestamp"], utc=True
    ).dt.tz_localize(None)

    # Raw data deliberately contains duplicate order records.
    raw_order_count = len(orders)
    orders = orders.drop_duplicates(subset=["order_id"]).copy()

    print(f"Raw orders: {raw_order_count:,}")
    print(f"Unique orders: {len(orders):,}")
    print(f"Duplicate records removed: {raw_order_count - len(orders):,}")

    # Never generate a label when the future observation window is incomplete.
    source_end = orders["order_timestamp"].max().normalize()
    final_snapshot = source_end - pd.Timedelta(days=30)

    first_snapshot = (
        customers["signup_date"].min().normalize()
        + pd.Timedelta(days=90)
    )
    first_snapshot += pd.Timedelta(
        days=(7 - first_snapshot.weekday()) % 7
    )

    snapshots = pd.date_range(
        first_snapshot, final_snapshot, freq="W-MON"
    )

    result = []

    for t in snapshots:
        print(f"Building snapshot: {t.date()}")

        # Eligibility is decided entirely from historical data.
        eligible = customers.loc[
            customers["signup_date"] <= t - pd.Timedelta(days=30)
        ].copy()

        historical_orders = orders.loc[
            orders["order_timestamp"].between(
                t - pd.Timedelta(days=90),
                t,
                inclusive="both",
            )
        ]

        active_ids = historical_orders["customer_id"].unique()
        eligible = eligible.loc[
            eligible["customer_id"].isin(active_ids)
        ].copy()

        if eligible.empty:
            continue

        frame = eligible[
            [
                "customer_id",
                "signup_date",
                "age",
                "region",
                "customer_segment",
                "acquisition_channel",
            ]
        ].copy()

        frame["prediction_timestamp"] = t

        # Historical order features: strictly no data after T.
        for days in (7, 30, 90):
            window = orders.loc[
                orders["order_timestamp"].between(
                    t - pd.Timedelta(days=days),
                    t,
                    inclusive="both",
                )
            ]

            grouped = window.groupby("customer_id")["order_value"].agg(
                order_count="count",
                order_value_sum="sum",
                order_value_mean="mean",
            )

            for metric in grouped.columns:
                frame[f"{metric}_{days}d"] = (
                    frame["customer_id"]
                    .map(grouped[metric])
                    .fillna(0)
                )

        last_order = (
            orders.loc[orders["order_timestamp"] <= t]
            .groupby("customer_id")["order_timestamp"]
            .max()
        )

        frame["days_since_last_order"] = (
            t - frame["customer_id"].map(last_order)
        ).dt.total_seconds() / 86400

        # Future data is used ONLY to construct the label.
        future_orders = orders.loc[
            (orders["order_timestamp"] > t)
            & (
                orders["order_timestamp"]
                <= t + pd.Timedelta(days=30)
            )
        ]

        future_counts = future_orders.groupby("customer_id").size()

        frame["inactive_next_30d"] = (
            frame["customer_id"]
            .map(future_counts)
            .fillna(0)
            .eq(0)
            .astype("int8")
        )

        result.append(frame)

    if not result:
        raise ValueError(
            "No snapshots generated. Check the date range and source data."
        )

    dataset = pd.concat(result, ignore_index=True)
    dataset = dataset.sort_values(
        ["prediction_timestamp", "customer_id"]
    ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset.to_csv(
        output_dir / "weekly_customer_snapshots.csv",
        index=False,
    )

    summary = {
        "rows": len(dataset),
        "unique_customers": dataset["customer_id"].nunique(),
        "snapshot_count": dataset["prediction_timestamp"].nunique(),
        "positive_label_rate": dataset["inactive_next_30d"].mean(),
        "first_snapshot": dataset["prediction_timestamp"].min(),
        "last_snapshot": dataset["prediction_timestamp"].max(),
    }

    pd.Series(summary).to_csv(
        output_dir / "snapshot_build_summary.csv"
    )

    print("\nDataset build complete.")
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    build_snapshots(
        Path(args.data_dir),
        Path(args.output_dir),
    )

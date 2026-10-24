from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd # type: ignore
import numpy as np


def load_table(data_dir: Path, name: str) -> pd.DataFrame:
    for suffix in (".csv", ".parquet"):
        path = data_dir / f"{name}{suffix}"
        if path.exists():
            return pd.read_csv(path) if suffix == ".csv" else pd.read_parquet(path)
    raise FileNotFoundError(f"Missing {name}.csv/.parquet in {data_dir}")


def parse_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_convert(None)
    return df


def window(df: pd.DataFrame, ts_col: str, t: pd.Timestamp, days: int) -> pd.DataFrame:
    # Left-open, right-closed: (T - days, T]
    return df.loc[df[ts_col].gt(t - pd.Timedelta(days=days)) & df[ts_col].le(t)]


def map_feature(frame: pd.DataFrame, feature: str, series: pd.Series,
                fill: float | int | None = None) -> None:
    frame[feature] = frame["customer_id"].map(series)
    if fill is not None:
        frame[feature] = frame[feature].fillna(fill)


def build(data_dir: Path, output_dir: Path, max_snapshots: int | None = None) -> None:
    customers = load_table(data_dir, "customers")
    orders = load_table(data_dir, "orders")
    payments = load_table(data_dir, "payments")
    support = load_table(data_dir, "support_tickets")
    subscriptions = load_table(data_dir, "subscriptions")
    marketing = load_table(data_dir, "marketing_events")
    feedback = load_table(data_dir, "customer_feedback")

    customers = parse_dates(customers, ["signup_date"])
    orders = parse_dates(orders, ["order_timestamp"])
    payments = parse_dates(payments, ["payment_timestamp"])
    support = parse_dates(support, ["created_at", "resolved_at"])
    subscriptions = parse_dates(subscriptions, ["start_date", "end_date"])
    marketing = parse_dates(marketing, ["timestamp"])
    feedback = parse_dates(feedback, ["timestamp"])

    # The source generator duplicates business events but assigns fresh IDs.
    raw_order_rows = len(orders)
    orders = orders.drop_duplicates(
        ["customer_id", "order_timestamp", "order_value", "product_category"]
    ).copy()
    duplicate_events_removed = raw_order_rows - len(orders)

    observed_end = max(
        orders["order_timestamp"].max(),
        payments["payment_timestamp"].max(),
        support["created_at"].max(),
        marketing["timestamp"].max(),
        feedback["timestamp"].max(),
    ).normalize()
    final_snapshot = observed_end - pd.Timedelta(days=30)
    earliest = customers["signup_date"].min().normalize() + pd.Timedelta(days=90)
    earliest += pd.Timedelta(days=(7 - earliest.weekday()) % 7)
    dates = list(pd.date_range(earliest, final_snapshot, freq="W-MON"))
    if max_snapshots is not None:
        dates = dates[:max_snapshots]

    customer_index = customers.set_index("customer_id")
    output_rows = []

    for index, t in enumerate(dates, 1):
        print(f"[{index}/{len(dates)}] {t.date()}", flush=True)

        eligible_ids = customers.loc[
            customers["signup_date"].le(t - pd.Timedelta(days=30)), "customer_id"
        ]
        hist90 = window(orders, "order_timestamp", t, 90)
        active_ids = hist90.loc[hist90["customer_id"].isin(eligible_ids), "customer_id"].unique()
        if len(active_ids) == 0:
            continue

        frame = customer_index.loc[active_ids].reset_index()
        frame["prediction_timestamp"] = t
        frame["tenure_days"] = (t - frame["signup_date"]).dt.total_seconds() / 86400

        # Orders
        for days in (7, 30, 90):
            w = window(orders, "order_timestamp", t, days)
            g = w.groupby("customer_id")["order_value"].agg(["count", "sum", "mean"])
            for metric in ("count", "sum", "mean"):
                map_feature(frame, f"orders_{metric}_{days}d", g[metric], 0)
            cats = w.pivot_table(
                index="customer_id", columns="product_category",
                values="order_value", aggfunc="count", fill_value=0
            )
            for category in ["electronics", "home", "services", "appliances", "other"]:
                map_feature(
                    frame, f"orders_category_{category}_{days}d",
                    cats[category] if category in cats else pd.Series(dtype=float), 0
                )

        last_order = orders.loc[orders["order_timestamp"].le(t)].groupby("customer_id")["order_timestamp"].max()
        frame["days_since_last_order"] = (t - frame["customer_id"].map(last_order)).dt.total_seconds() / 86400

        # Payments
        for days in (30, 90):
            w = window(payments, "payment_timestamp", t, days)
            total = w.groupby("customer_id").size()
            failed = w.loc[w["payment_status"].eq("failed")].groupby("customer_id").size()
            map_feature(frame, f"payments_count_{days}d", total, 0)
            map_feature(frame, f"payment_failures_{days}d", failed, 0)
            frame[f"payment_failure_rate_{days}d"] = np.where(
                frame[f"payments_count_{days}d"] > 0,
                frame[f"payment_failures_{days}d"] / frame[f"payments_count_{days}d"],
                0.0
            )
            methods = w.pivot_table(
                index="customer_id", columns="payment_method",
                values="payment_id", aggfunc="count", fill_value=0
            )
            for method in ["card", "upi", "net_banking", "wallet"]:
                map_feature(
                    frame, f"payments_method_{method}_{days}d",
                    methods[method] if method in methods else pd.Series(dtype=float), 0
                )

        # Support: ticket creation is historical; resolution only if resolved by T.
        for days in (30, 90):
            w = window(support, "created_at", t, days)
            map_feature(frame, f"support_tickets_{days}d", w.groupby("customer_id").size(), 0)
            for priority in ["low", "medium", "high"]:
                g = w.loc[w["priority"].eq(priority)].groupby("customer_id").size()
                map_feature(frame, f"support_priority_{priority}_{days}d", g, 0)

            resolved = support.loc[
                support["created_at"].gt(t - pd.Timedelta(days=days))
                & support["created_at"].le(t)
                & support["resolved_at"].notna()
                & support["resolved_at"].le(t)
            ].copy()
            resolved["resolution_hours"] = (
                resolved["resolved_at"] - resolved["created_at"]
            ).dt.total_seconds() / 3600
            map_feature(
                frame, f"mean_resolution_hours_{days}d",
                resolved.groupby("customer_id")["resolution_hours"].mean()
            )

        # Subscription state as-of T; never use future end_date as a label proxy.
        active_subs = subscriptions.loc[
            subscriptions["start_date"].le(t)
            & (subscriptions["end_date"].isna() | subscriptions["end_date"].ge(t))
        ].sort_values("start_date").drop_duplicates("customer_id", keep="last")
        subs = active_subs.set_index("customer_id")
        for source, target in [("plan", "subscription_plan"), ("billing_cycle", "billing_cycle")]:
            frame[target] = frame["customer_id"].map(subs[source])
        frame["subscription_monthly_price"] = frame["customer_id"].map(subs["monthly_price"])
        sub_start = frame["customer_id"].map(subs["start_date"])
        frame["subscription_tenure_days"] = (t - sub_start).dt.total_seconds() / 86400

        # Marketing
        for days in (30, 90):
            w = window(marketing, "timestamp", t, days)
            map_feature(frame, f"marketing_events_{days}d", w.groupby("customer_id").size(), 0)
            for event_type in ["email_open", "email_click", "offer_view", "offer_accept"]:
                g = w.loc[w["event_type"].eq(event_type)].groupby("customer_id").size()
                map_feature(frame, f"marketing_{event_type}_{days}d", g, 0)
            opens = frame[f"marketing_email_open_{days}d"]
            clicks = frame[f"marketing_email_click_{days}d"]
            frame[f"marketing_click_to_open_ratio_{days}d"] = np.where(
                opens > 0, clicks / opens, 0.0
            )

        # Feedback. No feedback => missing score/rate, while counts remain zero.
        for days in (30, 90):
            w = window(feedback, "timestamp", t, days)
            map_feature(frame, f"feedback_mean_rating_{days}d",
                        w.groupby("customer_id")["rating"].mean())
            map_feature(frame, f"feedback_count_{days}d", w.groupby("customer_id").size(), 0)
            neg = w.loc[w["sentiment"].eq("negative")].groupby("customer_id").size()
            map_feature(frame, f"feedback_negative_count_{days}d", neg, 0)
            count_col = frame[f"feedback_count_{days}d"]
            frame[f"feedback_negative_rate_{days}d"] = np.where(
                count_col > 0,
                frame[f"feedback_negative_count_{days}d"] / count_col,
                np.nan
            )

        # Target generation is deliberately separated from feature generation.
        future_orders = orders.loc[
            orders["order_timestamp"].gt(t)
            & orders["order_timestamp"].le(t + pd.Timedelta(days=30))
        ]
        future_count = future_orders.groupby("customer_id").size()
        frame["inactive_next_30d"] = (
            frame["customer_id"].map(future_count).fillna(0).eq(0).astype("int8")
        )
        output_rows.append(frame)

    if not output_rows:
        raise ValueError("No rows produced. Check source dates and snapshot boundaries.")

    dataset = pd.concat(output_rows, ignore_index=True)
    dataset = dataset.sort_values(["prediction_timestamp", "customer_id"]).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "weekly_customer_features.csv"
    dataset.to_csv(output_path, index=False)

    summary = {
        "rows": len(dataset),
        "columns": len(dataset.columns),
        "unique_customers": int(dataset["customer_id"].nunique()),
        "snapshot_count": int(dataset["prediction_timestamp"].nunique()),
        "positive_rate": float(dataset["inactive_next_30d"].mean()),
        "missing_cells": int(dataset.isna().sum().sum()),
        "business_duplicate_order_events_removed": int(duplicate_events_removed),
        "first_snapshot": str(dataset["prediction_timestamp"].min()),
        "last_snapshot": str(dataset["prediction_timestamp"].max()),
        "output_file": str(output_path),
    }
    pd.Series(summary, name="value").to_csv(output_dir / "feature_build_summary.csv", header=True)
    print("\n=== FEATURE BUILD SUMMARY ===")
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--max-snapshots", type=int, default=None)
    args = parser.parse_args()
    build(Path(args.data_dir), Path(args.output_dir), args.max_snapshots)

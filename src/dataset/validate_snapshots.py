
from pathlib import Path
import pandas as pd # type: ignore

DATA = Path("data/processed/weekly_customer_snapshots.csv")
df = pd.read_csv(DATA)

print("\n=== 1. STRUCTURE ===")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Duplicate customer-snapshot keys:",
      df.duplicated(["customer_id", "prediction_timestamp"]).sum())

print("\n=== 2. MISSING VALUES ===")
missing = df.isna().sum()
print(missing[missing > 0].sort_values(ascending=False))
if missing.sum() == 0:
    print("PASS: No missing values in snapshot dataset.")

print("\n=== 3. LABEL DISTRIBUTION ===")
print(df["inactive_next_30d"].value_counts())
print("Positive rate:",
      round(df["inactive_next_30d"].mean(), 4))

print("\n=== 4. FEATURE CONSISTENCY ===")
checks = {
    "7d orders <= 30d orders":
        (df["order_count_7d"] <= df["order_count_30d"]).all(),
    "30d orders <= 90d orders":
        (df["order_count_30d"] <= df["order_count_90d"]).all(),
    "No negative order counts":
        (df[["order_count_7d",
             "order_count_30d",
             "order_count_90d"]] >= 0).all().all(),
    "No negative days since last order":
        (df["days_since_last_order"] >= 0).all(),
    "All labels are binary":
        set(df["inactive_next_30d"].unique()).issubset({0, 1}),
}

for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {name}")

print("\n=== 5. LABEL RATE BY CUSTOMER SEGMENT ===")
print(
    df.groupby("customer_segment", observed=True)
      ["inactive_next_30d"]
      .agg(["count", "mean"])
      .assign(positive_rate=lambda x: x["mean"].round(4))
      .drop(columns="mean")
)

print("\n=== 6. LABEL RATE OVER TIME ===")
df["prediction_timestamp"] = pd.to_datetime(
    df["prediction_timestamp"]
)
monthly = (
    df.assign(month=df["prediction_timestamp"].dt.to_period("M"))
      .groupby("month", observed=True)["inactive_next_30d"]
      .agg(["count", "mean"])
)
monthly["positive_rate"] = monthly["mean"].round(4)
print(monthly[["count", "positive_rate"]].to_string())

print("\n=== 7. SNAPSHOT CADENCE ===")
dates = (
    df[["prediction_timestamp"]]
      .drop_duplicates()
      .sort_values("prediction_timestamp")
      ["prediction_timestamp"]
)
print("Snapshot count:", len(dates))
print("First:", dates.min())
print("Last:", dates.max())
print("All intervals are 7 days:",
      dates.diff().dropna().eq(pd.Timedelta(days=7)).all())

print("\n=== VALIDATION COMPLETE ===")

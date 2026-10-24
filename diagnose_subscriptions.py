
import pandas as pd # type: ignore

raw = pd.read_csv("data/raw/subscriptions.csv")
features = pd.read_csv(
    "data/processed/weekly_customer_features.csv",
    usecols=[
        "customer_id",
        "prediction_timestamp",
        "subscription_plan",
        "subscription_monthly_price",
    ],
)

for col in ["start_date", "end_date"]:
    raw[col] = pd.to_datetime(raw[col], errors="coerce")

features["prediction_timestamp"] = pd.to_datetime(
    features["prediction_timestamp"], errors="coerce"
)

# Check whether the feature-builder's active-subscription rule
# agrees with the raw subscription dates.
check = features.merge(
    raw[["customer_id", "start_date", "end_date", "plan"]],
    on="customer_id",
    how="left",
    validate="many_to_one",
)

check["expected_active"] = (
    (check["start_date"] <= check["prediction_timestamp"])
    & (check["end_date"] >= check["prediction_timestamp"])
)

check["feature_has_plan"] = check["subscription_plan"].notna()

print("RAW SUBSCRIPTION DATE RANGE")
print("Start:", raw["start_date"].min())
print("End:", raw["end_date"].min())
print("Median end date:", raw["end_date"].median())

print("\nACTIVE SUBSCRIPTION COVERAGE")
print(
    check.groupby("expected_active")["feature_has_plan"]
    .agg(["count", "mean"])
    .to_string()
)

print("\nEXPECTED ACTIVE BUT FEATURE MISSING")
print(
    check.loc[
        check["expected_active"] & ~check["feature_has_plan"],
        ["customer_id", "prediction_timestamp", "start_date", "end_date"],
    ].head(10).to_string(index=False)
)

print("\nFEATURE PRESENT BUT SUBSCRIPTION NOT ACTIVE")
print(
    check.loc[
        check["feature_has_plan"] & ~check["expected_active"],
        ["customer_id", "prediction_timestamp", "start_date", "end_date"],
    ].head(10).to_string(index=False)
)

print("\nCOVERAGE BY YEAR")
check["year"] = check["prediction_timestamp"].dt.year
print(
    check.groupby("year").agg(
        rows=("customer_id", "size"),
        expected_active_rate=("expected_active", "mean"),
        feature_plan_rate=("feature_has_plan", "mean"),
    ).round(3).to_string()
)

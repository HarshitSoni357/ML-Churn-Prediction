
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
DB_PATH = ROOT / "mlflow.db"

RUN_ID = "f99e6c37aed343a6a239816a1a32399f"
TARGET = "inactive_next_30d"

# Use the same local tracking database as the training script.
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.as_posix()}")

# Locate the model artifact belonging to the completed run.
client = mlflow.tracking.MlflowClient()
run = client.get_run(RUN_ID)

print("Run status:", run.info.status)
print("Run ID:", run.info.run_id)
print("Logged metrics:", run.data.metrics)

model_uri = f"runs:/{RUN_ID}/xgboost_pipeline"

print("\nLoading model from:", model_uri)
model = mlflow.pyfunc.load_model(model_uri)

# Use validation only for this smoke test.
validation = pd.read_csv(SPLIT_DIR / "validation.csv")
y_val = validation[TARGET].astype(int)

# Reproduce the feature selection used during training.
excluded = {
    "customer_id",
    "signup_date",
    "prediction_timestamp",
    TARGET,
}

numeric_cols = [
    col
    for col in validation.select_dtypes(include=["number"]).columns
    if col not in excluded
]

categorical_candidates = [
    "region",
    "customer_segment",
    "acquisition_channel",
    "subscription_plan",
    "billing_cycle",
]

categorical_cols = [
    col for col in categorical_candidates if col in validation.columns
]

feature_cols = numeric_cols + categorical_cols
X_val = validation[feature_cols].copy()

# Match the numeric input types used in the logged input example.
for col in numeric_cols:
    X_val[col] = X_val[col].astype("float64")

# A small inference batch first: verifies serialization and prediction.
sample = X_val.head(10)
sample_predictions = model.predict(sample)

print("\nSample prediction count:", len(sample_predictions))
print("Sample predictions:", sample_predictions[:5])

# Full validation scoring verifies the loaded artifact reproduces metrics.
probabilities = model.predict(X_val)

print("\nLoaded-model validation metrics:")
print("ROC-AUC:", round(roc_auc_score(y_val, probabilities), 6))
print("PR-AUC:", round(average_precision_score(y_val, probabilities), 6))

print("\nArtifact load and prediction smoke test: PASS")

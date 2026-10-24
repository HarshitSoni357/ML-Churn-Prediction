import hashlib
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

ROOT = Path(__file__).resolve().parents[2]

mlflow.set_tracking_uri(
    f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
)

MODEL_NAME = "CustomerInactivityRandomForest"
MODEL_VERSION = "1"
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

OUTPUT_DIR = ROOT / "artifacts" / "deployment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "random_forest_v1.joblib"
METADATA_PATH = OUTPUT_DIR / "model_metadata.json"

print(f"Loading registered model: {MODEL_URI}")
model = mlflow.sklearn.load_model(MODEL_URI)

print(f"Exporting model to: {MODEL_PATH}")
joblib.dump(model, MODEL_PATH, compress=3)

sha256 = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()

metadata = {
    "model_name": MODEL_NAME,
    "model_version": MODEL_VERSION,
    "model_uri": MODEL_URI,
    "model_file": MODEL_PATH.name,
    "target": "inactive_next_30d",
    "feature_count": len(model.feature_names_in_),
    "features": list(model.feature_names_in_),
    "classes": [int(value) for value in model.classes_],
    "sha256": sha256,
}

METADATA_PATH.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print("\nExport complete.")
print(f"Model file: {MODEL_PATH}")
print(f"Metadata file: {METADATA_PATH}")
print(f"Feature count: {metadata['feature_count']}")
print(f"SHA-256: {sha256}")

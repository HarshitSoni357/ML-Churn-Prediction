
import hashlib
import json
import logging
import math
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT_DIR = ROOT / "artifacts" / "deployment"

MODEL_PATH = DEPLOYMENT_DIR / "random_forest_v1.joblib"
METADATA_PATH = DEPLOYMENT_DIR / "model_metadata.json"

MODEL_NAME = "CustomerInactivityRandomForest"
MODEL_VERSION = "1"
MODEL_ID = f"{MODEL_NAME}-v{MODEL_VERSION}"
TARGET_NAME = "inactive_next_30d"

CATEGORICAL_FEATURES = {
    "region",
    "customer_segment",
    "acquisition_channel",
    "subscription_plan",
    "billing_cycle",
}

METRICS_PATHS = {"/health", "/metrics"}


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

logger = logging.getLogger("customer_inactivity_api")


def log_event(event: dict[str, Any]) -> None:
    """Emit one JSON-formatted log event."""
    logger.info(json.dumps(event, allow_nan=False))


# -------------------------------------------------------------------
# Model loading and artifact verification
# -------------------------------------------------------------------

def load_deployment_model():
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")

    if not METADATA_PATH.is_file():
        raise RuntimeError(f"Model metadata not found: {METADATA_PATH}")

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    if metadata.get("model_name") != MODEL_NAME:
        raise RuntimeError("Unexpected model name in metadata.")

    if str(metadata.get("model_version")) != MODEL_VERSION:
        raise RuntimeError("Unexpected model version in metadata.")

    if metadata.get("target") != TARGET_NAME:
        raise RuntimeError("Unexpected prediction target in metadata.")

    expected_hash = metadata.get("sha256")
    actual_hash = hashlib.sha256(
        MODEL_PATH.read_bytes()
    ).hexdigest()

    if not expected_hash or actual_hash != expected_hash:
        raise RuntimeError(
            "Model checksum verification failed."
        )

    # Only load serialized artifacts from this trusted project.
    loaded_model = joblib.load(MODEL_PATH)

    features = metadata.get("features")

    if not isinstance(features, list) or not features:
        raise RuntimeError("Invalid feature schema in metadata.")

    if list(loaded_model.feature_names_in_) != features:
        raise RuntimeError(
            "Model feature names do not match deployment metadata."
        )

    if len(features) != metadata.get("feature_count"):
        raise RuntimeError(
            "Feature count does not match deployment metadata."
        )

    if 1 not in list(loaded_model.classes_):
        raise RuntimeError(
            "Model does not contain positive class 1."
        )

    log_event({
        "event": "model_loaded",
        "model_name": MODEL_NAME,
        "model_version": MODEL_ID,
        "feature_count": len(features),
        "checksum_verified": True,
    })

    return loaded_model, features, metadata


model, FEATURES, MODEL_METADATA = load_deployment_model()


# -------------------------------------------------------------------
# In-memory metrics
# -------------------------------------------------------------------

_metrics_lock = threading.Lock()

_request_counts = Counter()
_request_duration_seconds_sum = Counter()

_prediction_count = 0
_prediction_probability_sum = 0.0

# These counters are cumulative and reset when the process restarts.
_probability_bounds = ("0.10", "0.25", "0.50", "0.75", "1.00")
_probability_buckets = Counter()


# -------------------------------------------------------------------
# Application
# -------------------------------------------------------------------

app = FastAPI(
    title="Customer Inactivity Prediction API",
    description=(
        "Predicts the probability that an eligible customer "
        "will become inactive in the next 30 days."
    ),
    version="1.0.0",
)


# -------------------------------------------------------------------
# Request and response schemas
# -------------------------------------------------------------------

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any] = Field(
        ...,
        description=(
            "Feature names and values for one customer snapshot. "
            "Send null for missing values."
        ),
    )


class PredictionResponse(BaseModel):
    model_version: str
    inactivity_probability: float
    target: str


# -------------------------------------------------------------------
# Request logging and HTTP metrics
# -------------------------------------------------------------------

@app.middleware("http")
async def observe_requests(
    request: Request,
    call_next,
):
    start_time = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response

    except Exception:
        log_event({
            "event": "request_failed",
            "method": request.method,
            "path": request.url.path,
            "status_code": 500,
        })
        logger.exception("Unhandled HTTP request exception")
        raise

    finally:
        duration = time.perf_counter() - start_time
        path = request.url.path
        method = request.method

        with _metrics_lock:
            _request_counts[(method, path, status_code)] += 1
            _request_duration_seconds_sum[(method, path)] += duration

        # Avoid logging request bodies, customer features, or predictions.
        log_event({
            "event": "http_request",
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": round(duration * 1000, 2),
        })


# -------------------------------------------------------------------
# Feature validation
# -------------------------------------------------------------------

def validate_features(
    input_features: dict[str, Any],
) -> pd.DataFrame:
    """Validate one feature record and preserve training column order."""

    expected = set(FEATURES)
    provided = set(input_features)

    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)

    if missing or unexpected:
        details = []

        if missing:
            details.append(f"Missing features: {missing}")

        if unexpected:
            details.append(f"Unexpected features: {unexpected}")

        raise HTTPException(
            status_code=422,
            detail="; ".join(details),
        )

    row = {}

    for feature in FEATURES:
        value = input_features[feature]

        if feature in CATEGORICAL_FEATURES:
            if value is not None and not isinstance(value, str):
                raise HTTPException(
                    status_code=422,
                    detail=f"'{feature}' must be a string or null.",
                )

            row[feature] = value
            continue

        if value is None:
            row[feature] = float("nan")
            continue

        if isinstance(value, bool) or not isinstance(
            value, (int, float)
        ):
            raise HTTPException(
                status_code=422,
                detail=f"'{feature}' must be numeric or null.",
            )

        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            raise HTTPException(
                status_code=422,
                detail=f"'{feature}' must be finite. Use null if missing.",
            )

        row[feature] = numeric_value

    return pd.DataFrame([row], columns=FEATURES)


# -------------------------------------------------------------------
# Health endpoint
# -------------------------------------------------------------------

@app.get("/health")
def health():
    """Report application and loaded-model health."""

    return {
        "status": "healthy",
        "model_name": MODEL_NAME,
        "model_version": MODEL_ID,
        "target": TARGET_NAME,
        "feature_count": len(FEATURES),
        "model_checksum_verified": True,
    }


# -------------------------------------------------------------------
# Prediction endpoint
# -------------------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict the probability of inactivity in the next 30 days."""

    global _prediction_count, _prediction_probability_sum

    try:
        input_df = validate_features(request.features)

        probabilities = model.predict_proba(input_df)
        positive_class_index = list(model.classes_).index(1)

        probability = float(
            probabilities[0, positive_class_index]
        )

        with _metrics_lock:
            _prediction_count += 1
            _prediction_probability_sum += probability

            for bound in _probability_bounds:
                if probability <= float(bound):
                    _probability_buckets[bound] += 1

        log_event({
            "event": "prediction_completed",
            "model_version": MODEL_ID,
            "prediction_count": 1,
        })

        return PredictionResponse(
            model_version=MODEL_ID,
            inactivity_probability=probability,
            target=TARGET_NAME,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the API server logs.",
        ) from exc


# -------------------------------------------------------------------
# Prometheus-compatible metrics endpoint
# -------------------------------------------------------------------

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Expose basic process-local metrics in Prometheus text format."""

    with _metrics_lock:
        request_counts = dict(_request_counts)
        duration_sums = dict(_request_duration_seconds_sum)
        prediction_count = _prediction_count
        probability_sum = _prediction_probability_sum
        bucket_counts = dict(_probability_buckets)

    lines = [
        "# HELP api_requests_total Total HTTP requests.",
        "# TYPE api_requests_total counter",
    ]

    for (method, path, status), count in sorted(
        request_counts.items()
    ):
        lines.append(
            f'api_requests_total{{method="{method}",'
            f'path="{path}",status="{status}"}} {count}'
        )

    lines.extend([
        "",
        "# HELP api_request_duration_seconds_sum "
        "Cumulative HTTP request duration in seconds.",
        "# TYPE api_request_duration_seconds_sum counter",
    ])

    for (method, path), duration_sum in sorted(
        duration_sums.items()
    ):
        lines.append(
            f'api_request_duration_seconds_sum{{'
            f'method="{method}",path="{path}"}} {duration_sum:.6f}'
        )

    lines.extend([
        "",
        "# HELP model_predictions_total Total successful predictions.",
        "# TYPE model_predictions_total counter",
        f"model_predictions_total {prediction_count}",
        "",
        "# HELP prediction_probability_sum "
        "Sum of successful prediction probabilities.",
        "# TYPE prediction_probability_sum counter",
        f"prediction_probability_sum {probability_sum:.8f}",
        "",
        "# HELP prediction_probability_bucket "
        "Cumulative counts of prediction probabilities.",
        "# TYPE prediction_probability_bucket counter",
    ])

    cumulative = 0

    for bound in _probability_bounds:
        cumulative += bucket_counts.get(bound, 0)
        lines.append(
            f'prediction_probability_bucket{{le="{bound}"}} '
            f"{cumulative}"
        )

    lines.append(
        f'prediction_probability_bucket{{le="+Inf"}} '
        f"{prediction_count}"
    )

    return "\n".join(lines) + "\n"

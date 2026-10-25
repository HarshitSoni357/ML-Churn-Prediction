
import hashlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter as PrometheusCounter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "artifacts" / "deployment" / "random_forest_v1.joblib"
METADATA_PATH = ROOT / "artifacts" / "deployment" / "model_metadata.json"

MODEL_NAME = "CustomerInactivityRandomForest"
MODEL_VERSION = "1"
MODEL_ID = f"{MODEL_NAME}-v{MODEL_VERSION}"
TARGET = "inactive_next_30d"

CATEGORICAL_FEATURES = {
    "region",
    "customer_segment",
    "acquisition_channel",
    "subscription_plan",
    "billing_cycle",
}

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger("customer_inactivity_api")


def log_event(event: str, **fields: Any) -> None:
    """Write structured logs without customer feature values."""
    logger.info(
        json.dumps(
            {"event": event, **fields},
            default=str,
        )
    )


# ---------------------------------------------------------------------
# Model loading and integrity verification
# ---------------------------------------------------------------------


def load_deployment_model():
    """Load the deployment model after validating its metadata and hash."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Deployment model not found: {MODEL_PATH}")

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Model metadata not found: {METADATA_PATH}"
        )

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if metadata.get("model_name") != MODEL_NAME:
        raise ValueError("Unexpected model name in deployment metadata.")

    if str(metadata.get("model_version")) != MODEL_VERSION:
        raise ValueError("Unexpected model version in deployment metadata.")

    metadata_target = metadata.get("target", metadata.get("target_column"))
    if metadata_target != TARGET:
        raise ValueError("Unexpected prediction target in deployment metadata.")

    # Support common checksum key names.
    expected_hash = next(
        (
            metadata[key]
            for key in (
                "sha256",
                "model_sha256",
                "model_checksum_sha256",
                "checksum_sha256",
            )
            if metadata.get(key)
        ),
        None,
    )

    if not isinstance(expected_hash, str):
        raise ValueError(
            "Deployment metadata is missing a SHA-256 checksum."
        )

    actual_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()

    if actual_hash.lower() != expected_hash.lower():
        raise ValueError(
            "Deployment model checksum mismatch. "
            "Refusing to load the model."
        )

    features = metadata.get(
        "features",
        metadata.get("feature_names"),
    )

    if not isinstance(features, list) or not features:
        raise ValueError(
            "Deployment metadata must contain a non-empty feature list."
        )

    if len(features) != len(set(features)):
        raise ValueError("Duplicate feature names found in model metadata.")

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Deployment model does not support predict_proba.")

    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "Model is missing feature_names_in_; "
            "cannot verify its input schema."
        )

    model_features = list(model.feature_names_in_)

    if model_features != features:
        raise ValueError(
            "Model feature names or order do not match deployment metadata."
        )

    if 1 not in list(model.classes_):
        raise ValueError("Model does not contain the positive class 1.")

    metadata_feature_count = metadata.get("feature_count")
    if (
        metadata_feature_count is not None
        and int(metadata_feature_count) != len(features)
    ):
        raise ValueError("Feature count does not match deployment metadata.")

    log_event(
        "model_loaded",
        model_name=MODEL_NAME,
        model_version=MODEL_ID,
        feature_count=len(features),
        checksum_verified=True,
    )

    return model, features, metadata


model, FEATURES, MODEL_METADATA = load_deployment_model()

POSITIVE_CLASS_INDEX = list(model.classes_).index(1)

# ---------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------

# The Python Counter API appends "_total" to the exported counter name.
API_REQUESTS = PrometheusCounter(
    "api_requests",
    "Total HTTP requests.",
    ["method", "path", "status"],
)

API_REQUEST_DURATION = Histogram(
    "api_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "path"],
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
    ),
)

MODEL_PREDICTIONS = PrometheusCounter(
    "model_predictions",
    "Total successful predictions.",
)

PREDICTION_PROBABILITY = Histogram(
    "prediction_probability",
    "Distribution of successful prediction probabilities.",
    buckets=(0.10, 0.25, 0.50, 0.75, 1.00),
)

# ---------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------

app = FastAPI(
    title="Customer Inactivity Prediction API",
    description=(
        "Predicts the probability that a customer will be inactive "
        "over the next 30 days."
    ),
    version=MODEL_ID,
)


# ---------------------------------------------------------------------
# Request and response schemas
# ---------------------------------------------------------------------


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any]


class PredictionResponse(BaseModel):
    model_version: str
    inactivity_probability: float
    target: str


# ---------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------


def validate_features(raw_features: dict[str, Any]) -> pd.DataFrame:
    """Validate the exact model input schema and build an ordered DataFrame."""

    expected = set(FEATURES)
    received = set(raw_features)

    missing = sorted(expected - received)
    unexpected = sorted(received - expected)

    if missing or unexpected:
        details = {}

        if missing:
            details["missing_features"] = missing

        if unexpected:
            details["unexpected_features"] = unexpected

        raise HTTPException(
            status_code=422,
            detail=details,
        )

    cleaned: dict[str, Any] = {}

    for feature in FEATURES:
        value = raw_features[feature]

        if feature in CATEGORICAL_FEATURES:
            if value is not None and not isinstance(value, str):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Feature '{feature}' must be a string or null."
                    ),
                )

            cleaned[feature] = value

        else:
            if value is None:
                # Preserve missing numeric values for the model's
                # configured preprocessing pipeline.
                cleaned[feature] = float("nan")
                continue

            # bool is a subclass of int in Python, so reject it explicitly.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Feature '{feature}' must be numeric or null."
                    ),
                )

            numeric_value = float(value)

            if not math.isfinite(numeric_value):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Feature '{feature}' must be a finite number."
                    ),
                )

            cleaned[feature] = numeric_value

    # Preserve the exact training-time feature order.
    return pd.DataFrame(
        [[cleaned[feature] for feature in FEATURES]],
        columns=FEATURES,
    )


# ---------------------------------------------------------------------
# HTTP observability middleware
# ---------------------------------------------------------------------


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    start_time = time.perf_counter()

    method = request.method
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response

    finally:
        duration = time.perf_counter() - start_time

        # Use route templates where available to avoid creating a
        # different metric series for every arbitrary URL.
        route = request.scope.get("route")
        route_path = getattr(route, "path", None)

        if route_path in {"/health", "/predict", "/metrics"}:
            path = route_path
        else:
            path = "unmatched"

        API_REQUESTS.labels(
            method=method,
            path=path,
            status=str(status_code),
        ).inc()

        API_REQUEST_DURATION.labels(
            method=method,
            path=path,
        ).observe(duration)

        # Never log request bodies or customer feature values.
        log_event(
            "http_request",
            method=method,
            path=path,
            status_code=status_code,
            duration_seconds=round(duration, 6),
        )


# ---------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_name": MODEL_NAME,
        "model_version": MODEL_ID,
        "target": TARGET,
        "feature_count": len(FEATURES),
        "model_checksum_verified": True,
    }


# ---------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        input_df = validate_features(request.features)

        probabilities = model.predict_proba(input_df)
        probability = float(
            probabilities[0][POSITIVE_CLASS_INDEX]
        )

        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError(
                "Model returned an invalid prediction probability."
            )

        # Record model metrics only after a successful prediction.
        MODEL_PREDICTIONS.inc()
        PREDICTION_PROBABILITY.observe(probability)

        log_event(
            "prediction_succeeded",
            model_version=MODEL_ID,
            target=TARGET,
            probability=round(probability, 6),
        )

        return PredictionResponse(
            model_version=MODEL_ID,
            inactivity_probability=probability,
            target=TARGET,
        )

    except HTTPException:
        # Preserve intentional client-facing validation errors.
        raise

    except Exception:
        logger.exception(
            "Prediction failed for model %s",
            MODEL_ID,
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check service logs.",
        ) from None


# ---------------------------------------------------------------------
# Prometheus scrape endpoint
# ---------------------------------------------------------------------


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )

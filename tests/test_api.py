import math
from pathlib import Path
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import app, FEATURES, MODEL_ID, ROOT


client = TestClient(app)


@pytest.fixture(scope="module")
def valid_features():
    fixture_path = ROOT / "tests" / "fixtures" / "sample_features.json"
    features = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert set(features) == set(FEATURES)
    return features

def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_version"] == MODEL_ID
    assert body["feature_count"] == len(FEATURES)


def test_valid_prediction(valid_features):
    response = client.post(
        "/predict",
        json={"features": valid_features},
    )

    assert response.status_code == 200

    body = response.json()
    assert body["model_version"] == MODEL_ID
    assert body["target"] == "inactive_next_30d"
    assert 0.0 <= body["inactivity_probability"] <= 1.0


def test_missing_feature_is_rejected(valid_features):
    features = valid_features.copy()
    features.pop(FEATURES[0])

    response = client.post(
        "/predict",
        json={"features": features},
    )

    assert response.status_code == 422


def test_unexpected_feature_is_rejected(valid_features):
    features = valid_features.copy()
    features["unexpected_feature"] = 123

    response = client.post(
        "/predict",
        json={"features": features},
    )

    assert response.status_code == 422


def test_invalid_numeric_feature_is_rejected(valid_features):
    numeric_feature = next(
        name for name in FEATURES
        if name not in {
            "region",
            "customer_segment",
            "acquisition_channel",
            "subscription_plan",
            "billing_cycle",
        }
    )

    features = valid_features.copy()
    features[numeric_feature] = "not-a-number"

    response = client.post(
        "/predict",
        json={"features": features},
    )

    assert response.status_code == 422


def test_missing_numeric_value_is_supported(valid_features):
    categorical_features = {
        "region",
        "customer_segment",
        "acquisition_channel",
        "subscription_plan",
        "billing_cycle",
    }

    numeric_feature = next(
        name for name in FEATURES
        if name not in categorical_features
    )

    features = valid_features.copy()
    features[numeric_feature] = None

    response = client.post(
        "/predict",
        json={"features": features},
    )

    assert response.status_code == 200
    probability = response.json()["inactivity_probability"]
    assert math.isfinite(probability)
    assert 0.0 <= probability <= 1.0

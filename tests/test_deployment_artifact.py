
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = ROOT / "artifacts" / "deployment"
METADATA_PATH = DEPLOYMENT_DIR / "model_metadata.json"
EXPECTED_FEATURE_COUNT = 78


@pytest.fixture(scope="module")
def metadata():
    assert METADATA_PATH.is_file(), (
        f"Deployment metadata not found: {METADATA_PATH}"
    )
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def model_and_metadata(metadata):
    model_path = DEPLOYMENT_DIR / metadata["model_file"]
    assert model_path.is_file(), (
        f"Deployment model not found: {model_path}"
    )
    return joblib.load(model_path), metadata


def test_deployment_metadata_contract(metadata):
    assert metadata["model_name"] == "CustomerInactivityRandomForest"
    assert metadata["model_version"] == "1"
    assert metadata["target"] == "inactive_next_30d"
    assert metadata["feature_count"] == EXPECTED_FEATURE_COUNT

    features = metadata["features"]
    assert len(features) == EXPECTED_FEATURE_COUNT
    assert len(set(features)) == EXPECTED_FEATURE_COUNT
    assert all(isinstance(feature, str) and feature for feature in features)

    assert metadata["classes"] == [0, 1]
    assert metadata["model_file"] == "random_forest_v1.joblib"
    assert metadata["model_uri"] == "models:/CustomerInactivityRandomForest/1"


def test_deployment_model_checksum(metadata):
    model_path = DEPLOYMENT_DIR / metadata["model_file"]
    assert model_path.is_file()

    actual_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert actual_sha256 == metadata["sha256"], (
        "Deployment model checksum does not match model_metadata.json. "
        "Regenerate the metadata when intentionally replacing the model."
    )


def test_deployment_model_feature_contract(model_and_metadata):
    model, metadata = model_and_metadata

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

    # This project deploys a scikit-learn Pipeline with a named preprocessor.
    assert hasattr(model, "named_steps")
    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps

    expected_features = metadata["features"]
    actual_features = list(model.feature_names_in_)

    assert actual_features == expected_features, (
        "The deployed model's feature order differs from its metadata."
    )


def test_deployment_model_prediction_smoke(model_and_metadata):
    model, metadata = model_and_metadata
    features = metadata["features"]

    # A deterministic, correctly shaped synthetic input. Numeric fields
    # use 0; categorical fields use a string the encoder can handle.
    categorical_features = {
        "region",
        "customer_segment",
        "acquisition_channel",
        "subscription_plan",
        "billing_cycle",
    }
    sample = {
        name: ("__unknown_test_category__" if name in categorical_features else 0)
        for name in features
    }

    import pandas as pd

    frame = pd.DataFrame([sample], columns=features)
    probabilities = model.predict_proba(frame)

    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert probabilities.sum(axis=1)[0] == pytest.approx(1.0)

    predicted_class = model.predict(frame)[0]
    assert predicted_class in metadata["classes"]


"""Offline feature and prediction drift monitoring for churn prediction."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "weekly_customer_features.csv"
MODEL_PATH = ROOT / "artifacts" / "deployment" / "random_forest_v1.joblib"
METADATA_PATH = ROOT / "artifacts" / "deployment" / "model_metadata.json"
OUTPUT_DIR = ROOT / "artifacts" / "monitoring"

DATE_COLUMN = "prediction_timestamp"
TARGET_COLUMN = "inactive_next_30d"

# Historical training period vs a later, unseen-in-time comparison period.
REFERENCE_END = "2025-12-01"
CURRENT_START = "2026-04-06"

MAX_ROWS_PER_PERIOD = 50_000
CHUNK_SIZE = 100_000
RANDOM_STATE = 42
PSI_WARNING = 0.20
PSI_CRITICAL = 0.25

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Calculate PSI from two distributions with aligned bins."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Smoothing avoids log(0) for empty bins.
    epsilon = 1e-6
    expected = np.clip(expected, epsilon, None)
    actual = np.clip(actual, epsilon, None)
    expected = expected / expected.sum()
    actual = actual / actual.sum()

    return float(np.sum((actual - expected) * np.log(actual / expected)))


def distribution(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Return normalized histogram proportions, including missing values."""
    values = pd.to_numeric(values, errors="coerce")
    missing_share = float(values.isna().mean())

    observed = values.dropna().to_numpy(dtype=float)
    counts, _ = np.histogram(observed, bins=edges)

    counts = counts.astype(float)
    counts = np.append(counts, values.isna().sum())
    total = counts.sum()

    if total == 0:
        return np.ones(len(counts)) / len(counts)

    return counts / total


def categorical_distribution(values: pd.Series, categories: list[str]) -> np.ndarray:
    """Return proportions for fixed categories plus missing and unseen values."""
    normalized = values.astype("string").fillna("__MISSING__")
    known = set(categories)

    normalized = normalized.map(
        lambda value: value if value in known else "__UNSEEN__"
    )

    counts = normalized.value_counts()
    result = np.array(
        [counts.get(category, 0) for category in categories]
        + [
            counts.get("__MISSING__", 0),
            counts.get("__UNSEEN__", 0),
        ],
        dtype=float,
    )

    if result.sum() == 0:
        return np.ones(len(result)) / len(result)

    return result / result.sum()


def collect_period_samples() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stream the CSV and retain bounded samples from both time periods."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    reference_parts: list[pd.DataFrame] = []
    current_parts: list[pd.DataFrame] = []
    reference_count = 0
    current_count = 0

    rng = np.random.default_rng(RANDOM_STATE)

    logger.info("Streaming dataset in chunks: %s", DATA_PATH)

    for chunk in pd.read_csv(
        DATA_PATH,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunk[DATE_COLUMN] = pd.to_datetime(
            chunk[DATE_COLUMN], errors="coerce"
        )

        reference = chunk[
            chunk[DATE_COLUMN] <= pd.Timestamp(REFERENCE_END)
        ]
        current = chunk[
            chunk[DATE_COLUMN] >= pd.Timestamp(CURRENT_START)
        ]

        reference_count += len(reference)
        current_count += len(current)

        # Reservoir-style bounded sampling across chunks.
        for batch, parts in (
            (reference, reference_parts),
            (current, current_parts),
        ):
            if batch.empty:
                continue

            parts.append(batch)

            combined = pd.concat(parts, ignore_index=True)
            if len(combined) > MAX_ROWS_PER_PERIOD:
                # Keep a random sample so early chunks do not dominate.
                positions = rng.choice(
                    len(combined),
                    size=MAX_ROWS_PER_PERIOD,
                    replace=False,
                )
                combined = combined.iloc[positions]

            parts[:] = [combined]

    if not reference_parts or not current_parts:
        raise ValueError(
            "Could not sample both periods. Check date column and date boundaries."
        )

    reference_df = pd.concat(reference_parts, ignore_index=True)
    current_df = pd.concat(current_parts, ignore_index=True)

    logger.info(
        "Rows scanned by period: reference=%s, current=%s",
        reference_count,
        current_count,
    )
    logger.info(
        "Sample sizes: reference=%s, current=%s",
        len(reference_df),
        len(current_df),
    )

    return reference_df, current_df


def get_feature_names(model, metadata: dict) -> list[str]:
    """Use the model's learned feature order, with metadata fallback."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    for key in ("feature_names", "features", "feature_columns"):
        value = metadata.get(key)
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            return value

    raise ValueError(
        "Could not determine model feature order from model or metadata. "
        "Inspect model_metadata.json and configure feature names explicitly."
    )


def score_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_names: list[str],
) -> list[dict]:
    """Compare feature distributions using fixed reference-derived bins."""
    results = []

    for feature in feature_names:
        if feature not in reference or feature not in current:
            logger.warning("Skipping missing feature column: %s", feature)
            continue

        ref = reference[feature]
        cur = current[feature]

        if (
            pd.api.types.is_numeric_dtype(ref)
            and not pd.api.types.is_bool_dtype(ref)
        ):
            ref_num = pd.to_numeric(ref, errors="coerce")
            cur_num = pd.to_numeric(cur, errors="coerce")

            observed = ref_num.dropna().to_numpy(dtype=float)
            if len(observed) == 0:
                continue

            # Quantile bins are fixed using reference data only.
            edges = np.unique(
                np.quantile(observed, np.linspace(0, 1, 11))
            )

            if len(edges) < 2:
                center = float(edges[0])
                edges = np.array(
                    [-np.inf, center - 1e-9, center + 1e-9, np.inf]
                )
            else:
                edges[0] = -np.inf
                edges[-1] = np.inf

            ref_dist = distribution(ref_num, edges)
            cur_dist = distribution(cur_num, edges)
            score = psi(ref_dist, cur_dist)

            results.append(
                {
                    "feature": feature,
                    "type": "numeric",
                    "psi": score,
                    "reference_missing_rate": float(ref_num.isna().mean()),
                    "current_missing_rate": float(cur_num.isna().mean()),
                }
            )
        else:
            ref_cat = ref.astype("string").fillna("__MISSING__")
            categories = sorted(ref_cat.unique().tolist())
            ref_dist = categorical_distribution(ref, categories)
            cur_dist = categorical_distribution(cur, categories)
            score = psi(ref_dist, cur_dist)

            results.append(
                {
                    "feature": feature,
                    "type": "categorical",
                    "psi": score,
                    "reference_missing_rate": float(ref.isna().mean()),
                    "current_missing_rate": float(cur.isna().mean()),
                    "current_unseen_rate": float(
                        cur.astype("string")
                        .fillna("__MISSING__")
                        .map(lambda value: value not in set(categories))
                        .mean()
                    ),
                }
            )

    return sorted(results, key=lambda row: row["psi"], reverse=True)


def prediction_summary(
    model,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_names: list[str],
) -> dict:
    """Compare predicted churn-probability distributions."""
    summaries = {}

    for name, frame in (("reference", reference), ("current", current)):
        X = frame[feature_names].copy()
        probabilities = model.predict_proba(X)[:, 1]

        summaries[name] = {
            "row_count": int(len(probabilities)),
            "mean_probability": float(np.mean(probabilities)),
            "median_probability": float(np.median(probabilities)),
            "p90_probability": float(np.quantile(probabilities, 0.90)),
            "high_risk_rate_at_0_5": float(np.mean(probabilities >= 0.5)),
            "probabilities": probabilities,
        }

    reference_prob = summaries["reference"]["probabilities"]
    current_prob = summaries["current"]["probabilities"]

    edges = np.linspace(0, 1, 11)
    ref_dist, _ = np.histogram(
        reference_prob, bins=edges, density=False
    )
    cur_dist, _ = np.histogram(
        current_prob, bins=edges, density=False
    )

    summaries["comparison"] = {
        "prediction_psi": psi(ref_dist, cur_dist),
        "mean_probability_delta": (
            summaries["current"]["mean_probability"]
            - summaries["reference"]["mean_probability"]
        ),
        "high_risk_rate_delta": (
            summaries["current"]["high_risk_rate_at_0_5"]
            - summaries["reference"]["high_risk_rate_at_0_5"]
        ),
    }

    # Keep JSON report compact; never serialize every prediction.
    del summaries["reference"]["probabilities"]
    del summaries["current"]["probabilities"]

    return summaries


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Deployment model or metadata is missing from artifacts/deployment."
        )

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    feature_names = get_feature_names(model, metadata)

    reference, current = collect_period_samples()

    missing_features = [
        name
        for name in feature_names
        if name not in reference.columns or name not in current.columns
    ]
    if missing_features:
        raise ValueError(f"Missing model features: {missing_features}")

    # Ensure model input order exactly matches training.
    reference_X = reference[feature_names]
    current_X = current[feature_names]

    # Validate inference before computing any drift results.
    model.predict_proba(reference_X.head(5))
    model.predict_proba(current_X.head(5))

    feature_results = score_feature_drift(
        reference, current, feature_names
    )
    predictions = prediction_summary(
        model, reference, current, feature_names
    )

    for result in feature_results:
        score = result["psi"]
        result["status"] = (
            "critical" if score >= PSI_CRITICAL
            else "warning" if score >= PSI_WARNING
            else "normal"
        )

    prediction_psi = predictions["comparison"]["prediction_psi"]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "reference_period": {
            "prediction_timestamp_lte": REFERENCE_END,
            "sample_size": len(reference),
        },
        "current_period": {
            "prediction_timestamp_gte": CURRENT_START,
            "sample_size": len(current),
        },
        "thresholds": {
            "psi_warning": PSI_WARNING,
            "psi_critical": PSI_CRITICAL,
        },
        "feature_drift": feature_results,
        "prediction_drift": predictions,
        "summary": {
            "features_scored": len(feature_results),
            "features_warning_or_critical": sum(
                result["status"] != "normal"
                for result in feature_results
            ),
            "features_critical": sum(
                result["status"] == "critical"
                for result in feature_results
            ),
            "prediction_drift_status": (
                "critical" if prediction_psi >= PSI_CRITICAL
                else "warning" if prediction_psi >= PSI_WARNING
                else "normal"
            ),
        },
        "interpretation": (
            "Drift is a diagnostic signal, not proof of model performance "
            "degradation. Actual labels are required to evaluate performance."
        ),
    }

    output_path = OUTPUT_DIR / "drift_report.json"
    output_path.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    top_features = feature_results[:10]

    print(f"\nDrift report written to: {output_path}")
    print("\nTop 10 feature drift scores:")
    for result in top_features:
        print(
            f"{result['feature']:<42} "
            f"PSI={result['psi']:.4f}  "
            f"status={result['status']}"
        )

    print("\nPrediction drift:")
    print(json.dumps(predictions["comparison"], indent=2))
    print("\nSummary:")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()


"""
Compare a registered champion model against a candidate model.

- Uses validation data only; never uses the held-out test set.
- Diagnoses feature-schema mismatches.
- Aligns feature order independently for each model.
- Refuses comparison when required features are missing.
- Applies configurable promotion gates.
- Saves a JSON comparison report.

Run from the project root:
    python -m src.models.compare_champion_candidate
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]

VALIDATION_PATH = ROOT / "data" / "processed" / "splits" / "validation.csv"
OUTPUT_PATH = (
    ROOT / "docs" / "model-evaluation" / "champion-candidate-comparison.json"
)
MLFLOW_DB = ROOT / "mlflow.db"

CHAMPION_URI = "models:/CustomerInactivityRandomForest/1"
CANDIDATE_URI = "models:/CustomerInactivityXGBoost/3"

TARGET_COLUMN = "inactive_next_30d"

# Promotion gates: candidate must meet all safety gates and at least
# one meaningful-improvement gate.
MAX_ROC_AUC_REGRESSION = 0.005
MAX_PR_AUC_REGRESSION_FRACTION = 0.02
MAX_BRIER_REGRESSION_FRACTION = 0.05
MIN_ROC_AUC_IMPROVEMENT = 0.002
MIN_PR_AUC_IMPROVEMENT_FRACTION = 0.02

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def inspect_feature_names(model: Any, model_name: str) -> list[str] | None:
    """Find feature names on a model or a pipeline step."""
    names = getattr(model, "feature_names_in_", None)

    if names is not None:
        result = list(names)
        logger.info(
            "%s: found %d feature names on model",
            model_name,
            len(result),
        )
        return result

    # Pipelines and wrappers may expose names on an internal estimator.
    named_steps = getattr(model, "named_steps", None)
    if named_steps:
        for step_name, step in reversed(list(named_steps.items())):
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                result = list(names)
                logger.info(
                    "%s: found %d feature names in step '%s'",
                    model_name,
                    len(result),
                    step_name,
                )
                return result

    logger.warning(
        "%s: model does not expose feature_names_in_; "
        "will try the validation data's feature columns",
        model_name,
    )
    return None


def load_model(uri: str, model_name: str) -> Any:
    logger.info("Loading %s: %s", model_name, uri)
    return mlflow.sklearn.load_model(uri)


def prepare_features(
    validation_df: pd.DataFrame,
    model: Any,
    model_name: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepare validation features for one model.

    If the model declares its feature schema, use that exact order.
    If it does not, use the validation columns as a fallback.
    """
    excluded = {
        TARGET_COLUMN,
        "customer_id",
        "signup_date",
        "prediction_timestamp",
    }

    available_columns = [
        column for column in validation_df.columns
        if column not in excluded
    ]

    model_features = inspect_feature_names(model, model_name)

    if model_features is None:
        # A model without a declared schema cannot be safely reordered
        # from model metadata. Use the dataset's existing feature order.
        model_features = available_columns
        logger.warning(
            "%s: using validation dataset feature order (%d features). "
            "Confirm this matches the model's training schema.",
            model_name,
            len(model_features),
        )

    missing = sorted(set(model_features) - set(validation_df.columns))
    if missing:
        raise ValueError(
            f"{model_name} requires features absent from validation data: "
            f"{missing}. Check model training and registration."
        )

    unexpected = sorted(set(available_columns) - set(model_features))
    if unexpected:
        logger.info(
            "%s: ignoring %d validation columns not used by the model: %s",
            model_name,
            len(unexpected),
            unexpected[:20],
        )

    X = validation_df.loc[:, model_features].copy()

# Normalize boolean columns only; preserve categorical strings.
    for column in X.columns:
        if pd.api.types.is_bool_dtype(X[column]):
            X[column] = X[column].astype("int8")

    if X.isna().any().any():
        logger.warning(
            "%s validation features contain missing values. "
            "The model must support these values as trained.",
            model_name,
        )

    return X, model_features


def predict_probability(model: Any, X: pd.DataFrame, model_name: str) -> np.ndarray:
    """Return positive-class probabilities with basic validation."""
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"{model_name} does not expose predict_proba(). "
            "Add model-specific probability handling before comparison."
        )

    probabilities = np.asarray(model.predict_proba(X))

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            f"{model_name} returned unexpected predict_proba shape: "
            f"{probabilities.shape}"
        )

    classes = getattr(model, "classes_", None)
    if classes is None:
        # Standard binary classifiers generally order classes as [0, 1].
        positive_index = 1
    else:
        matches = np.where(np.asarray(classes) == 1)[0]
        if len(matches) != 1:
            raise ValueError(
                f"{model_name} does not have exactly one positive class 1. "
                f"Classes: {classes}"
            )
        positive_index = int(matches[0])

    result = probabilities[:, positive_index]

    if not np.isfinite(result).all():
        raise ValueError(f"{model_name} produced non-finite probabilities.")

    if ((result < 0) | (result > 1)).any():
        raise ValueError(f"{model_name} produced probabilities outside [0, 1].")

    return result


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def compare_metrics(
    champion: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, Any]:
    """Apply safety gates and meaningful-improvement requirements."""
    roc_gate = (
        candidate["roc_auc"]
        >= champion["roc_auc"] - MAX_ROC_AUC_REGRESSION
    )
    pr_gate = (
        candidate["pr_auc"]
        >= champion["pr_auc"] * (1 - MAX_PR_AUC_REGRESSION_FRACTION)
    )
    brier_gate = (
        candidate["brier_score"]
        <= champion["brier_score"] * (1 + MAX_BRIER_REGRESSION_FRACTION)
    )

    roc_improvement = (
        candidate["roc_auc"] - champion["roc_auc"]
        >= MIN_ROC_AUC_IMPROVEMENT
    )
    pr_improvement = (
        candidate["pr_auc"]
        >= champion["pr_auc"] * (1 + MIN_PR_AUC_IMPROVEMENT_FRACTION)
    )

    safety_gates_passed = roc_gate and pr_gate and brier_gate
    meaningful_improvement = roc_improvement or pr_improvement
    eligible = safety_gates_passed and meaningful_improvement

    return {
        "safety_gates": {
            "roc_auc_non_regression": roc_gate,
            "pr_auc_non_regression": pr_gate,
            "brier_score_non_regression": brier_gate,
        },
        "meaningful_improvement": {
            "roc_auc_improvement": roc_improvement,
            "pr_auc_improvement": pr_improvement,
        },
        "all_safety_gates_passed": safety_gates_passed,
        "meaningful_improvement_found": meaningful_improvement,
        "eligible_for_release_review": eligible,
        "decision": (
            "PROMOTE_TO_RELEASE_REVIEW"
            if eligible
            else "DO_NOT_PROMOTE"
        ),
        "note": (
            "Eligibility is not an automatic deployment. Review model "
            "schema, calibration, segment performance, operational behavior, "
            "and release/rollback readiness before deployment."
        ),
    }


def main() -> None:
    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {VALIDATION_PATH}"
        )

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")

    logger.info("Loading validation dataset: %s", VALIDATION_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    if TARGET_COLUMN not in validation_df.columns:
        raise ValueError(f"Target column missing: {TARGET_COLUMN}")

    y = validation_df[TARGET_COLUMN].to_numpy()

    if pd.isna(y).any():
        raise ValueError("Validation target contains missing labels.")

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            "Expected binary validation labels containing both 0 and 1."
        )

    champion_model = load_model(CHAMPION_URI, "champion")
    candidate_model = load_model(CANDIDATE_URI, "candidate")

    X_champion, champion_features = prepare_features(
        validation_df,
        champion_model,
        "champion",
    )
    X_candidate, candidate_features = prepare_features(
        validation_df,
        candidate_model,
        "candidate",
    )

    logger.info(
        "Champion feature count: %d; candidate feature count: %d",
        len(champion_features),
        len(candidate_features),
    )

    logger.info(
        "Same feature names and order: %s",
        champion_features == candidate_features,
    )
    logger.info(
        "Features only in champion: %s",
        sorted(set(champion_features) - set(candidate_features)),
    )
    logger.info(
        "Features only in candidate: %s",
        sorted(set(candidate_features) - set(champion_features)),
    )

    logger.info("Scoring champion on validation data...")
    champion_probabilities = predict_probability(
        champion_model, X_champion, "champion"
    )

    logger.info("Scoring candidate on validation data...")
    candidate_probabilities = predict_probability(
        candidate_model, X_candidate, "candidate"
    )

    champion_metrics = calculate_metrics(y, champion_probabilities)
    candidate_metrics = calculate_metrics(y, candidate_probabilities)

    comparison = compare_metrics(champion_metrics, candidate_metrics)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_dataset": str(VALIDATION_PATH.relative_to(ROOT)),
        "evaluation_split": "validation",
        "test_set_used": False,
        "target": TARGET_COLUMN,
        "sample_count": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "champion_uri": CHAMPION_URI,
        "candidate_uri": CANDIDATE_URI,
        "feature_schema": {
            "champion_feature_count": len(champion_features),
            "candidate_feature_count": len(candidate_features),
            "same_feature_names": (
                set(champion_features) == set(candidate_features)
            ),
            "same_feature_order": champion_features == candidate_features,
            "champion_only_features": sorted(
                set(champion_features) - set(candidate_features)
            ),
            "candidate_only_features": sorted(
                set(candidate_features) - set(champion_features)
            ),
        },
        "metrics": {
            "champion": champion_metrics,
            "candidate": candidate_metrics,
            "candidate_minus_champion": {
                key: candidate_metrics[key] - champion_metrics[key]
                for key in champion_metrics
            },
        },
        "promotion_gates": {
            "max_roc_auc_regression": MAX_ROC_AUC_REGRESSION,
            "max_pr_auc_regression_fraction": (
                MAX_PR_AUC_REGRESSION_FRACTION
            ),
            "max_brier_regression_fraction": (
                MAX_BRIER_REGRESSION_FRACTION
            ),
            "min_roc_auc_improvement": MIN_ROC_AUC_IMPROVEMENT,
            "min_pr_auc_improvement_fraction": (
                MIN_PR_AUC_IMPROVEMENT_FRACTION
            ),
        },
        "decision": comparison,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== Champion vs Candidate ===")
    print(f"Validation rows: {len(y):,}")
    print(f"Champion: {CHAMPION_URI}")
    print(f"Candidate: {CANDIDATE_URI}")

    print("\nMetric                 Champion      Candidate")
    for metric in ("roc_auc", "pr_auc", "brier_score"):
        print(
            f"{metric:<22} "
            f"{champion_metrics[metric]:.6f}      "
            f"{candidate_metrics[metric]:.6f}"
        )

    print("\nSafety gates:")
    for gate, passed in comparison["safety_gates"].items():
        print(f"  {gate}: {'PASS' if passed else 'FAIL'}")

    print("\nMeaningful improvement:")
    for gate, passed in comparison["meaningful_improvement"].items():
        print(f"  {gate}: {'PASS' if passed else 'FAIL'}")

    print(f"\nDecision: {comparison['decision']}")
    print(f"Report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

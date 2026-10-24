
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


SPLIT_DIR = Path("data/processed/splits")
MODEL_PATH = Path("artifacts/baseline/model.joblib")
TARGET = "inactive_next_30d"

DROP_COLUMNS = [
    "customer_id",
    "signup_date",
    "prediction_timestamp",
    TARGET,
]


def main():
    model = joblib.load(MODEL_PATH)
    validation = pd.read_csv(SPLIT_DIR / "validation.csv")

    X = validation.drop(columns=DROP_COLUMNS)
    y = validation[TARGET].astype(int)
    scores = model.predict_proba(X)[:, 1]

    diagnostic = validation[
        ["customer_segment", TARGET]
    ].copy()
    diagnostic["predicted_probability"] = scores
    diagnostic["predicted_positive_0125"] = (scores >= 0.125).astype(int)

    print("=== RESULTS BY CUSTOMER SEGMENT ===")
    segment_results = diagnostic.groupby("customer_segment").agg(
        rows=(TARGET, "size"),
        actual_positive_rate=(TARGET, "mean"),
        mean_predicted_probability=("predicted_probability", "mean"),
        median_predicted_probability=("predicted_probability", "median"),
        flagged_rate=("predicted_positive_0125", "mean"),
    )
    print(segment_results.round(4).to_string())

    print("\n=== PROBABILITY QUANTILES ===")
    print(
        diagnostic["predicted_probability"]
        .quantile([0, 0.1, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99, 1])
        .round(4)
        .to_string()
    )

    print("\n=== RANKING METRICS BY SEGMENT ===")
    for segment, group in diagnostic.groupby("customer_segment"):
        if group[TARGET].nunique() < 2:
            continue
        print(
            f"{segment}: "
            f"ROC-AUC={roc_auc_score(group[TARGET], group['predicted_probability']):.4f}, "
            f"PR-AUC={average_precision_score(group[TARGET], group['predicted_probability']):.4f}"
        )

    # Inspect learned coefficients to understand which features drive scores.
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": classifier.coef_[0],
        }
    )
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()

    print("\n=== TOP 20 LEARNED COEFFICIENTS BY ABSOLUTE VALUE ===")
    print(
        coefficients.sort_values("abs_coefficient", ascending=False)
        .head(20)[["feature", "coefficient"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

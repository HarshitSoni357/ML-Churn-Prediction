from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[2]

mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
client = MlflowClient()

EXPERIMENT_NAME = "customer-inactivity-prediction"

MODELS = [
    {
        "name": "CustomerInactivityLogisticRegression",
        "path": ROOT / "artifacts" / "baseline" / "model.joblib",
        "metrics": {
            "roc_auc": 0.686180,
            "pr_auc": 0.238063,
            "brier_score": 0.116942,
        },
        "description": "Logistic regression baseline for predicting inactivity in the next 30 days.",
        "selection_status": "baseline",
    },
    {
        "name": "CustomerInactivityRandomForest",
        "path": ROOT / "artifacts" / "random_forest" / "model.joblib",
        "metrics": {
            "roc_auc": 0.687386,
            "pr_auc": 0.238627,
            "brier_score": 0.116970,
        },
        "description": "Random Forest candidate for predicting inactivity in the next 30 days.",
        "selection_status": "provisional_champion",
    },
]

EXISTING_XGBOOST = {
    "name": "CustomerInactivityXGBoost",
    "run_id": "f99e6c37aed343a6a239816a1a32399f",
    "artifact_path": "xgboost_pipeline",
    "metrics": {
        "roc_auc": 0.686406,
        "pr_auc": 0.235758,
        "brier_score": 0.117036,
    },
    "description": "Regularized XGBoost candidate for predicting inactivity in the next 30 days.",
    "selection_status": "candidate",
}


def register_from_run(name, model_uri, metrics, description, selection_status):
    print(f"\nRegistering: {name}")

    # Avoid creating another version if this exact model URI
    # has already been registered under this name.
    try:
        versions = client.search_model_versions(f"name='{name}'")
        for version in versions:
            if version.source == model_uri:
                print(f"Already registered as version {version.version}; skipping.")
                return version.version
    except Exception:
        pass

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=name,
    )

    version = str(registered.version)

    client.set_model_version_tag(
        name, version, "lifecycle_status", "candidate"
    )
    client.set_model_version_tag(
        name, version, "selection_status", selection_status
    )

    client.update_model_version(
        name=name,
        version=version,
        description=description,
    )

    print(f"Registered version: {version}")
    print(f"Model URI: models:/{name}/{version}")
    print(f"Validation metrics: {metrics}")

    return version


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Register the existing XGBoost model run.
    xgb_uri = (
        f"runs:/{EXISTING_XGBOOST['run_id']}/"
        f"{EXISTING_XGBOOST['artifact_path']}"
    )

    register_from_run(
        name=EXISTING_XGBOOST["name"],
        model_uri=xgb_uri,
        metrics=EXISTING_XGBOOST["metrics"],
        description=EXISTING_XGBOOST["description"],
        selection_status=EXISTING_XGBOOST["selection_status"],
    )

    # Log Logistic Regression and Random Forest as MLflow models,
    # then register their model versions.
    for config in MODELS:
        if not config["path"].exists():
            raise FileNotFoundError(
                f"Model file not found: {config['path']}"
            )

        model = joblib.load(config["path"])

        with mlflow.start_run(
            run_name=f"register-{config['name']}"
        ) as run:
            mlflow.set_tags({
                "model_family": config["name"],
                "selection_status": config["selection_status"],
                "target": "inactive_next_30d",
            })

            mlflow.log_metrics(config["metrics"])

            mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
             skops_trusted_types=[
            "numpy.dtype",
            "sklearn.tree._tree.Tree",
            ],
)
            model_uri = f"runs:/{run.info.run_id}/model"

        register_from_run(
            name=config["name"],
            model_uri=model_uri,
            metrics=config["metrics"],
            description=config["description"],
            selection_status=config["selection_status"],
        )

    print("\nRegistration complete.")
    print("\nRegistered models:")
    for model_name in [
        "CustomerInactivityLogisticRegression",
        "CustomerInactivityRandomForest",
        "CustomerInactivityXGBoost",
    ]:
        try:
            versions = client.search_model_versions(
                f"name='{model_name}'"
            )
            for version in versions:
                print(
                    f"{model_name} | version={version.version} "
                    f"| status={version.status} "
                    f"| run_id={version.run_id}"
                )
        except Exception as exc:
            print(f"Could not list {model_name}: {exc}")


if __name__ == "__main__":
    main()

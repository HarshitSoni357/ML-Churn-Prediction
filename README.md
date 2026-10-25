# Production ML Churn Prediction

An end-to-end, production-oriented machine learning portfolio project
demonstrating a time-aware customer inactivity prediction workflow:
synthetic data generation, feature engineering, temporal evaluation,
experiment tracking, model registration, API serving, artifact
validation, and monitoring.

> **Scope note:** This project uses synthetic customer data. It
> demonstrates ML engineering and MLOps patterns; it does not claim real
> business lift or validated production performance.

## At a glance

-   **Problem:** Predict whether a customer will be inactive over the
    next 30 days.
-   **Target:** `inactive_next_30d` (`1` = inactive, `0` = active).
-   **Serving:** FastAPI with `/health`, `/predict`, and `/metrics`.
-   **Model workflow:** Logistic Regression, Random Forest, and XGBoost;
    MLflow tracking/registration; validation-only champion/candidate
    gates.
-   **Operations:** Docker Compose, Prometheus, Grafana, Alertmanager,
    and offline drift analysis.
-   **Evaluation:** Temporal splits with a 30-day label-window embargo.

## Architecture

``` mermaid
flowchart TD
    A[Raw synthetic customer data] --> B[Build weekly snapshots]
    B --> C[Feature engineering and label creation]
    C --> D[Schema and data-quality validation]
    D --> E[Temporal train / validation / test split]
    E --> F[Train baseline and candidate models]
    F --> G[MLflow tracking and model registry]
    G --> H[Validation-only champion / candidate gates]
    H --> I[Deployment bundle and metadata]
    I --> J[FastAPI inference service]
    J --> K[Health, prediction, and metrics endpoints]
    K --> L[Prometheus and Grafana]
    C --> M[Offline feature and prediction drift checks]
```

The architecture diagram describes the repository workflow. It does not
imply automated production deployment or live drift-triggered
retraining.

## Quick start

### Option A --- Run the committed deployment bundle

A clean clone can run the API using the tracked deployment bundle in
`artifacts/deployment/`; it does not need to regenerate the dataset or
train a model first.

Use Python 3.11:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the API:

``` powershell
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Check health:

``` powershell
Invoke-RestMethod http://localhost:8000/health
```

Open the interactive API documentation at `http://127.0.0.1:8000/docs`.

Run the tests:

``` powershell
python -m pytest -q
```

The API and deployment-artifact tests require the committed files
`artifacts/deployment/random_forest_v1.joblib` and
`artifacts/deployment/model_metadata.json`.

### Option B --- Start the local monitoring stack

Create a local environment file from the example:

``` powershell
Copy-Item .env.example .env
```

Edit `.env` and replace the Grafana password placeholder with a unique
local password. Do not commit `.env`; it is excluded by `.gitignore`.

Validate and start the stack:

``` powershell
docker compose config --quiet
docker compose up --build -d
docker compose ps
```

Local endpoints:

  Service        URL
  -------------- ------------------------------
  API            `http://localhost:8000`
  API docs       `http://localhost:8000/docs`
  Prometheus     `http://localhost:9090`
  Grafana        `http://localhost:3000`
  Alertmanager   `http://localhost:9093`

Grafana uses the username and password configured by
`GRAFANA_ADMIN_USER` and `GRAFANA_ADMIN_PASSWORD` in `.env`. If Grafana
has already initialized its persistent volume, changing `.env` may not
change the existing Grafana account password.

Stop the stack with `docker compose down`. Avoid
`docker compose down -v` unless you intentionally want to remove
persisted Prometheus and Grafana data.

### Option C --- Rebuild data and model experiments

This is the longer workflow. Run commands from the repository root, in
order:

``` powershell
python src/dataset/build_snapshots.py
python src/features/build_features.py
python src/features/validate_features.py
python src/features/create_temporal_split.py
python src/models/train_baseline.py
python src/models/analyze_thresholds.py
python src/models/compare_champion_candidate.py
```

The dataset and training workflow writes generated data and experiment
outputs that are intentionally not all tracked in Git. Inspect each
script and its output paths before running the full pipeline. Training
and registration may depend on local MLflow tracking state and generated
artifacts; this workflow is separate from starting the API with the
committed deployment bundle.

Run offline drift analysis when the expected input/reference data is
available:

``` powershell
python monitoring/drift_monitor.py
```

The drift script is an offline diagnostic; it is not a continuously
running live drift service.

## Why this project exists

The goal is to predict whether a customer will be inactive over the next
30 days using a temporal, production-style ML workflow.

The project covers:

-   Customer snapshot dataset construction
-   Feature engineering from historical activity
-   Temporal train / validation / test splits
-   Baseline and candidate model training
-   MLflow experiment tracking and model registration
-   Validation-only model comparison gates
-   Deployment artifact packaging and integrity checks
-   FastAPI inference service
-   Prometheus metrics and offline drift checks
-   Docker-based local deployment
-   Manual release and rollback procedures

## Problem definition and dataset

The binary target is `inactive_next_30d`:

-   `1`: the customer is inactive over the next 30 days
-   `0`: the customer is active over the next 30 days

The processed feature dataset contains:

-   **3,239,195** customer-snapshot rows
-   **48,704** unique customers
-   **126** weekly snapshots
-   **82** columns in the processed dataset
-   Observation window: **2024-04-01 through 2026-08-24**
-   Positive-label rate: approximately **14.4%**

Snapshots are keyed by `(customer_id, prediction_timestamp)`. Validation
checks found no duplicate keys in the processed dataset. This is a
temporal evaluation problem, not a customer-disjoint evaluation:
customers may appear in multiple snapshots over time.

The project builds customer snapshots from synthetic records
representing:

-   Customers
-   Orders
-   Payments
-   Support tickets
-   Subscriptions
-   Marketing events
-   Customer feedback

Features are engineered using information available up to each
prediction timestamp. The target uses future order activity in the next
30 days. The feature and target construction is intentionally separated
to reduce the risk of future-data leakage.

The project uses time-based splits with an embargo between periods so
the 30-day label window does not leak into adjacent splits.

### Split sizes

  Split               Rows
  ------------ -----------
  Train          1,654,654
  Validation       341,513
  Test             939,500

The held-out test period starts on **2026-04-06** and ends on
**2026-08-24**. It is reserved for final evaluation and descriptive
time-window analysis, not repeated model tuning.

## Repository structure

``` text
.
├── .github/
│   └── workflows/
├── artifacts/
│   └── deployment/
│       ├── model_metadata.json
│       └── random_forest_v1.joblib
├── data/                    # Generated locally; not tracked
├── docs/
│   ├── model-evaluation/
│   └── model-release-and-rollback.md
├── monitoring/
├── src/
│   ├── dataset/
│   ├── features/
│   ├── models/
│   └── api/
├── tests/
│   └── fixtures/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

The tree is representative rather than exhaustive. Local MLflow
databases, virtual environments, generated datasets, logs, and `.env`
files should not be committed.

## Modeling pipeline

1.  Build weekly customer snapshots.
2.  Engineer features and future inactivity labels.
3.  Validate schema and data quality.
4.  Create temporal train, validation, and test splits with an embargo.
5.  Train and evaluate baseline and candidate models.
6.  Track experiments and register models in MLflow.
7.  Compare candidates with the current champion using validation-only
    gates.
8.  Package the selected model and metadata for API serving.
9.  Validate the deployment bundle and serve predictions through
    FastAPI.
10. Inspect service metrics and run offline drift analysis.

Models evaluated include Logistic Regression, Random Forest, and
XGBoost.

## Model results

### Validation comparison

  Model                    ROC-AUC   PR-AUC (average precision)   Brier score
  --------------------- ---------- ---------------------------- -------------
  Logistic Regression     0.686180                     0.238063      0.116942
  Random Forest           0.687386                     0.238627      0.116970
  XGBoost                 0.686406                     0.235758      0.117036

The Random Forest is retained as the current champion because XGBoost
did not demonstrate the predefined meaningful improvement required for
promotion. The differences are small, so the decision is intentionally
conservative.

### Held-out test results

  Metric                         Random Forest
  ---------------------------- ---------------
  ROC-AUC                             0.690006
  PR-AUC (average precision)          0.237343
  Brier score                         0.115705

Interpretation:

-   **ROC-AUC** measures ranking ability.
-   **PR-AUC** is useful when evaluating an imbalanced target.
-   **Brier score** measures probability prediction error; lower is
    better.

These results depend on the synthetic data and its label-generation
logic. They should not be interpreted as evidence of real-world customer
impact.

### Historical monitoring demonstration

A 5,000-row historical ledger produced:

-   ROC-AUC: **0.75162**
-   PR-AUC: **0.35004**
-   Brier score: **0.11318**

This is a monitoring demonstration, not a directly comparable
model-selection result. It has a different temporal composition and is
not the final held-out test metric.

## Model release and promotion policy

The champion/candidate comparison script evaluates candidates on the
validation split using these gates:

-   ROC-AUC cannot drop by more than **0.005**
-   PR-AUC cannot drop by more than **2%**
-   Brier score cannot worsen by more than **5%**
-   At least one meaningful improvement is required: ROC-AUC **+0.002**
    or PR-AUC **+2%**

The current candidate passed the safety gates but did not pass the
meaningful-improvement gate, so the decision is `DO_NOT_PROMOTE`.

This is a release-review signal, not an automatic production deployment
decision. Before deployment, verify schema compatibility, model
identity, API behavior, segment performance, and rollback readiness.

See
[`docs/model-release-and-rollback.md`](docs/model-release-and-rollback.md)
for the manual release, verification, and rollback procedure.

## Deployment artifact integrity

The committed deployment bundle contains:

-   `artifacts/deployment/random_forest_v1.joblib`
-   `artifacts/deployment/model_metadata.json`

The metadata records the model identity, version, target, ordered
feature list, feature count, classes, model URI, and SHA-256 checksum.
The deployment-artifact test suite checks metadata consistency, checksum
integrity, feature ordering, and a prediction smoke test. CI runs these
tests independently of a local MLflow database.

If a model artifact is intentionally replaced, regenerate and review its
metadata as part of the same release. Do not update the checksum
manually without validating that the model and metadata belong together.

## API service

The service lives in `src/api/app.py` and exposes:

-   `GET /health` --- service and model metadata
-   `POST /predict` --- inactivity probability for one customer feature
    record
-   `GET /metrics` --- Prometheus metrics

Use `http://127.0.0.1:8000/docs` for interactive API documentation after
starting the service.

Example request shape:

``` json
{
  "features": {
    "age": 42,
    "region": "north",
    "customer_segment": "premium",
    "acquisition_channel": "web",
    "subscription_plan": "standard",
    "billing_cycle": "monthly",
    "tenure_days": 250
  }
}
```

**Important:** The example above is illustrative only. The API validates
the complete feature contract, so a real request must include every
feature required by `model_metadata.json`, not just the subset shown
here.

A successful response includes the model version, the
`inactive_next_30d` target, and the predicted inactivity probability.
Exact response fields are defined by the API implementation.

## Monitoring and drift checks

The project includes:

-   API metrics exposed for Prometheus scraping
-   Prometheus alert rules and Alertmanager configuration
-   Grafana for local dashboarding
-   Offline feature and prediction drift analysis in
    `monitoring/drift_monitor.py`

The drift monitor compares a reference period with a later comparison
period using Population Stability Index (PSI) for features and
prediction scores. It writes a JSON report to
`artifacts/monitoring/drift_report.json` when its required inputs are
available.

Drift is diagnostic, not proof of model failure. It is a signal to
investigate data quality, concept drift, pipeline changes, or changes in
customer behavior. Offline drift analysis should not be described as
live monitoring or automatic retraining.

## Tests and CI

Run the full test suite:

``` powershell
python -m pytest -q
```

Useful targeted checks:

``` powershell
python -m pytest tests/test_api.py -v
python -m pytest tests/test_deployment_artifact.py -v
python -m pytest tests/test_temporal_boundaries.py -v
docker compose config --quiet
docker build -t customer-inactivity-api:local .
```

GitHub Actions runs the configured tests and builds the API Docker
image. Review `.github/workflows/ci.yml` for the exact checks enforced
by the current workflow.

## Main files and responsibilities

-   `src/dataset/build_snapshots.py` --- builds weekly customer
    snapshots from raw inputs
-   `src/features/build_features.py` --- creates engineered features and
    labels
-   `src/features/validate_features.py` --- validates schema, data
    quality, and target consistency
-   `src/features/create_temporal_split.py` --- creates train /
    validation / test splits with temporal embargoes
-   `src/models/train_baseline.py` --- trains the baseline Logistic
    Regression model
-   `src/models/analyze_thresholds.py` --- explores probability
    thresholds on validation data
-   `src/models/diagnose_baseline.py` --- inspects segment performance
    and learned coefficients
-   `src/models/compare_champion_candidate.py` --- compares champion and
    candidate on validation data
-   `src/api/app.py` --- FastAPI inference service
-   `monitoring/drift_monitor.py` --- offline feature and prediction
    drift analysis
-   `Dockerfile` --- containerizes the API
-   `docker-compose.yml` --- local service orchestration
-   `.dockerignore` --- excludes unnecessary local artifacts from the
    Docker build context

## Limitations and next steps

-   The dataset is synthetic, so it demonstrates a realistic workflow
    rather than real-world customer behavior or direct business impact.
-   Temporal splits are used, but customers can appear across multiple
    snapshots.
-   Rolling evaluation windows overlap and should not be treated as
    independent estimates.
-   PSI thresholds and promotion gates are engineering defaults that
    should be calibrated to business impact, acceptable risk, and
    expected changes in customer behavior.
-   The project demonstrates an end-to-end ML engineering workflow, not
    a live production deployment in a business environment.
-   The release and rollback procedure is manual; automated model
    promotion, deployment, and rollback are not claimed.

Potential future improvements include end-to-end tests for monitoring
workflows and calibration of thresholds using business-specific risk and
cost assumptions.

## Repository hygiene

Before committing or publishing:

1.  Run `git diff --check`.
2.  Review `git status --short`.
3.  Review both staged and unstaged changes with `git diff --cached` and
    `git diff`.
4.  Search for credentials, tokens, local paths, and private data.
5.  Ensure generated datasets, local MLflow databases, logs, and `.env`
    files are excluded unless intentionally versioned.
6.  Run `python -m pytest -q`.
7.  Run `docker compose config --quiet`.

## Summary

This repository demonstrates a practical ML engineering workflow: build
and validate data, evaluate models over time, register and compare
candidates, package a deployment bundle, serve predictions, inspect
metrics, and document release readiness.

The emphasis is on reproducibility, evaluation discipline, artifact
integrity, and operational thinking---not on claiming real-world
performance from synthetic data.

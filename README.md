# Production ML Churn Prediction

A portfolio project that demonstrates an end-to-end, production-oriented churn-prediction workflow: time-aware data generation, feature engineering, model training, validation, model registry, API serving, monitoring, and release review.

> Scope note: this project uses synthetic customer data. The goal is to demonstrate the workflow and engineering patterns, not to claim real business lift or production performance.

## Why this project exists

The goal is to predict whether a customer will be inactive over the next 30 days using a temporal, production-style ML workflow.

The project covers:

- snapshot dataset construction
- feature engineering from historical customer activity
- temporal train / validation / test splits
- baseline and candidate model training
- MLflow experiment tracking and model registration
- validation-only model comparison gates
- FastAPI inference service
- model deployment artifact packaging
- monitoring and drift checks
- Docker-based local deployment

## Quick start

Use Python 3.11 and a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then run the pipeline in this order:

```powershell
python src/dataset/build_snapshots.py
python src/features/build_features.py
python src/features/validate_features.py
python src/features/create_temporal_split.py
python src/models/train_baseline.py
python src/models/analyze_thresholds.py
python src/models/compare_champion_candidate.py
```

Run the test suite:

```powershell
python -m pytest -q
```

Start the local API:

```powershell
python -m uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

Run the drift monitor:

```powershell
python monitoring/drift_monitor.py
```

Start the local Docker stack if configured:

```powershell
docker compose config -q
docker compose up --build
```

## Problem definition

The binary target is `inactive_next_30d`:

- `1`: the customer is inactive over the next 30 days
- `0`: the customer is active over the next 30 days

The processed feature dataset contains:

- 3,239,195 customer-snapshot rows
- 48,704 unique customers
- 126 weekly snapshots
- 82 columns
- observation window from 2024-04-01 through 2026-08-24
- positive-label rate of about 14.4%

Snapshots are keyed by `(customer_id, prediction_timestamp)`. The validation checks confirm there are no duplicate keys in the processed dataset. This is a temporal evaluation problem, not a customer-disjoint evaluation, because users can appear in multiple snapshots across time.

## Dataset and evaluation design

The project creates a customer snapshot dataset using historical records from:

- customers
- orders
- payments
- support tickets
- subscriptions
- marketing events
- customer feedback

Features are engineered using only information available up to the prediction timestamp. The target is generated using future order activity in the next 30 days. This is intentionally separated to avoid leakage from future data into feature construction.

The project uses a time-based split with an embargo between periods so the 30-day label window does not leak into adjacent splits.

### Split sizes

| Split | Rows |
|---|---:|
| Train | 1,654,654 |
| Validation | 341,513 |
| Test | 939,500 |

The held-out test period starts on 2026-04-06 and ends on 2026-08-24. The test set is reserved for final evaluation and descriptive time-window analysis, not repeated tuning.

## Repository structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── dataset/
│   ├── features/
│   ├── models/
│   └── api/
├── monitoring/
├── docs/
├── artifacts/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── .dockerignore
├── .gitignore
└── mlflow.db
```

## Modeling pipeline

```text
Processed customer snapshots
          |
          v
Time-aware train / validation / test split
          |
          v
Feature preprocessing + model training
  - Logistic Regression
  - Random Forest
  - XGBoost
          |
          v
MLflow tracking + model registration
          |
          +--> Champion/candidate validation-only gates
          |
          v
Selected deployment model bundle
          |
          v
FastAPI service
          |
          +--> /health
          +--> /predict
          +--> /metrics
                    |
                    v
         Prometheus + drift monitoring
```

## Model results

### Validation comparison

| Model | ROC-AUC | PR-AUC (average precision) | Brier score |
|---|---:|---:|---:|
| Logistic Regression | 0.686180 | 0.238063 | 0.116942 |
| Random Forest | 0.687386 | 0.238627 | 0.116970 |
| XGBoost | 0.686406 | 0.235758 | 0.117036 |

The Random Forest is retained as the current champion because XGBoost did not demonstrate the pre-defined meaningful improvement required for promotion. The differences are small, so the decision is intentionally conservative.

### Held-out test results

| Metric | Random Forest |
|---|---:|
| ROC-AUC | 0.690006 |
| PR-AUC (average precision) | 0.237343 |
| Brier score | 0.115705 |

Interpretation:

- ROC-AUC measures ranking ability
- PR-AUC is especially relevant for imbalanced targets
- Brier score measures calibration error and lower is better

These metrics depend on the synthetic data and the label generation logic.

### Historical monitoring demonstration

A 5,000-row historical ledger produced:

- ROC-AUC: 0.75162
- PR-AUC: 0.35004
- Brier score: 0.11318

This is a monitoring demonstration, not a directly comparable model-selection result. It has a different temporal composition and is not the final held-out test metric.

## Model release and promotion policy

The champion/candidate comparison script compares the current champ and a candidate on the validation split using the following gates:

- ROC-AUC cannot drop by more than 0.005
- PR-AUC cannot drop by more than 2%
- Brier score cannot worsen by more than 5%
- At least one meaningful improvement must be observed: ROC-AUC +0.002 or PR-AUC +2%

The current candidate passed the safety gates but did not pass the meaningful-improvement gate, so the decision is `DO_NOT_PROMOTE`.

This is a release-review signal, not an automatic production deployment decision. Before deployment, verify schema compatibility, model identity, API behavior, segment performance, and rollback readiness.

## API service

The service lives in `src/api/app.py` and exposes:

- `GET /health` for service health
- `POST /predict` to get a probability for a single customer
- `GET /metrics` for Prometheus scraping

Example request body:

```json
{
  "features": {
    "age": 42,
    "region": "north",
    "customer_segment": "premium",
    "acquisition_channel": "web",
    "subscription_plan": "standard",
    "billing_cycle": "monthly",
    "tenure_days": 250,
    "orders_count_7d": 1,
    "orders_count_30d": 3,
    "orders_count_90d": 7,
    "days_since_last_order": 12,
    "marketing_events_30d": 2,
    "support_tickets_30d": 0
  }
}
```

Example response:

```json
{
  "model_version": "CustomerInactivityRandomForest-v1",
  "inactivity_probability": 0.2145,
  "target": "inactive_next_30d"
}
```

## Monitoring and drift checks

The project includes offline drift monitoring in `monitoring/drift_monitor.py`.

It compares a reference period with a later comparison period and checks:

- feature drift using PSI (Population Stability Index)
- prediction drift using model score distribution changes
- status labels such as `normal`, `warning`, and `critical`

It writes a JSON report to `artifacts/monitoring/drift_report.json`.

Drift is diagnostic, not proof of failure. It is a signal to investigate data quality, concept drift, pipeline changes, or changes in customer behavior.

## Local setup and repo hygiene

Before committing or publishing:

1. Run `git diff --check`
2. Review `git status --short`
3. Check `git diff --cached` / `git diff`
4. Search for credentials, tokens, local paths, and private data
5. Ensure generated artifacts, local MLflow databases, logs, and large local data are excluded unless intentionally versioned
6. Run `python -m pytest -q`
7. Run `docker compose config -q`

## Main files and responsibilities

- `src/dataset/build_snapshots.py` — builds a weekly customer snapshot dataset from raw files
- `src/features/build_features.py` — creates engineered features and labels
- `src/features/validate_features.py` — validates schema, data quality, and target consistency
- `src/features/create_temporal_split.py` — creates train / validation / test splits with temporal embargoes
- `src/models/train_baseline.py` — trains the baseline logistic regression model
- `src/models/analyze_thresholds.py` — explores probability thresholds on validation data
- `src/models/diagnose_baseline.py` — inspects segment performance and learned coefficients
- `src/models/compare_champion_candidate.py` — compares champion vs candidate on validation data
- `src/api/app.py` — FastAPI inference service
- `monitoring/drift_monitor.py` — offline monitoring for feature and prediction drift
- `Dockerfile` — containerizes the API
- `docker-compose.yml` — local service orchestration
- `.dockerignore` — keeps build context lean and excludes local-only artifacts

## Limitations and next steps

- The dataset is synthetic, so it demonstrates a realistic workflow rather than real-world customer behavior or direct business impact.
- The project uses temporal splits, but customers can appear across multiple snapshots.
- Rolling evaluation windows overlap and should not be treated as independent estimates.
- PSI thresholds and promotion gates are engineering defaults and should be calibrated to business impact, acceptable risk, and expected changes in customer behavior.
- This project demonstrates the end-to-end ML engineering and MLOps workflow, rather than a live production deployment in a business environment.

Potential next steps:

- Add end-to-end tests for monitoring and deployment workflows.
- Make thresholds and drift rules configurable.
- Validate parity between registered models and the deployment bundle in CI.
- Add release and rollback review gates before deployment.
- Strengthen alerting and dashboarding around API and model health.

## Summary

This repository is a hands-on example of a production ML workflow for churn prediction:

- build data
- validate it
- split over time
- train and compare models
- register the chosen model
- serve it via a FastAPI API
- monitor drift and release readiness

It is a strong portfolio project for demonstrating MLOps and ML engineering concepts, without claiming a live business deployment or real-world operational validation.


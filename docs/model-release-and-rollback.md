
# Model Release and Rollback Runbook

## 1. Purpose

This runbook describes how to validate, release, verify, and roll back the customer inactivity prediction API.

The deployed model bundle consists of:

- `artifacts/deployment/random_forest_v1.joblib`
- `artifacts/deployment/model_metadata.json`

The API serves predictions for `inactive_next_30d`. The model bundle and its metadata must remain consistent.

## 2. Pre-release checklist

Before releasing a model or changing the deployment bundle:

- [ ] Confirm the intended model version and evaluation results.
- [ ] Review the model's validation and held-out test metrics.
- [ ] Confirm the feature schema and target are unchanged or intentionally versioned.
- [ ] Ensure the model file and metadata belong to the same release.
- [ ] Run the automated tests.
- [ ] Confirm the Docker image builds successfully.
- [ ] Record the Git commit and model version associated with the release.

Do not promote a model solely because one metric improves. Review the relevant metrics, data limitations, and operational impact.

## 3. Validate the release candidate

Run these commands from the repository root in PowerShell:

```powershell
python -m pytest tests/test_api.py tests/test_deployment_artifact.py tests/test_temporal_boundaries.py -v
docker compose config --quiet
docker build -t customer-inactivity-api:release-candidate .
```

Resolve test, configuration, or build failures before proceeding.

The deployment-artifact tests verify the bundle's metadata, checksum, feature contract, and prediction behavior.

## 4. Start and verify the local stack

Ensure the local `.env` file contains a unique Grafana administrator password.

Start the services:

```powershell
docker compose up -d --build
docker compose ps
```

Check the API health endpoint:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Confirm that:

- The API reports a healthy status.
- The reported model version and feature count are expected.
- The API prediction tests pass.
- Prometheus and Grafana are reachable at their configured local ports.
- Prometheus can scrape the API metrics endpoint.
- Expected alert rules are loaded.

Review service logs if a check fails:

```powershell
docker compose logs --tail 100 api
docker compose logs --tail 100 prometheus
docker compose logs --tail 100 grafana
docker compose logs --tail 100 alertmanager
```

Do not release if the API is unhealthy or the deployed model does not match its metadata.

## 5. Record the release

Before changing the deployed model, record:

- Git commit SHA
- Model name and version
- Model artifact SHA-256 checksum
- Evaluation report location
- Test results
- Release date and release notes

Keep a known-good Git commit and its corresponding deployment bundle available for rollback.

The Docker image should be tagged with an immutable release identifier or Git commit SHA rather than relying only on a mutable tag.

## 6. Rollback procedure

Rollback may be necessary if the API becomes unhealthy, predictions fail validation, the model bundle is inconsistent, or a release causes an unacceptable regression.

### A. Stop and identify the problem

```powershell
docker compose ps
docker compose logs --tail 200 api
```

Check the health endpoint and recent changes. If the issue is limited to a model release, identify the last known-good commit and model bundle.

### B. Restore the known-good deployment bundle

First, ensure your current work is committed or backed up.

Replace `<known-good-commit>` with the actual Git commit SHA that contains the known-good bundle:

```powershell
git restore --source=<known-good-commit> -- artifacts/deployment/random_forest_v1.joblib artifacts/deployment/model_metadata.json
```

This changes only the two deployment-bundle files in your working tree. It does not move your current branch to the older commit.

Run the deployment integrity and API tests:

```powershell
python -m pytest tests/test_deployment_artifact.py tests/test_api.py -v
```

If the restored bundle passes, commit the rollback as a new change and record the reason.

### C. Rebuild and restart

```powershell
docker compose up -d --build api
docker compose ps
Invoke-RestMethod http://localhost:8000/health
```

Run the API prediction tests and confirm that the expected model version is serving.

### D. Confirm recovery

- [ ] API health check succeeds.
- [ ] Prediction requests succeed.
- [ ] The deployed model checksum matches its metadata.
- [ ] The expected model version is reported.
- [ ] Prometheus metrics are available.
- [ ] No new critical service errors appear in the logs.

Document the incident, root cause, rollback commit, and follow-up actions.

## 7. Important operational notes

- This runbook documents a manual release and rollback process. It does not imply automated model promotion or rollback.
- Restoring a deployment bundle does not restore the MLflow tracking database.
- Avoid running `docker compose down -v` during routine troubleshooting; removing volumes can delete persisted monitoring data.
- Historical monitoring reports should not be interpreted as proof of live production performance. Record whether metrics came from offline evaluation or live serving.
- A successful health check confirms service health, not model quality.

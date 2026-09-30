# 20 MLOps

**Goal.** Take a model out of the notebook and keep it healthy: track experiments, package and register versions, serve it behind a validated API, version data alongside code, containerise it, watch it for drift, and protect releases with automated tests and quality gates. The running example is the churn model from module 12.

**Prerequisites.** [12 Machine Learning](../12_Machine_Learning/README.md) (pipelines, cross-validation), [19 Data Pipelines](../19_Data_Pipelines/README.md), Python and a little HTTP. Tools that are not available offline (Docker daemon, DVC, GitHub Actions) are shown as correct files and commands and paired with runnable analogues (a Dockerfile linter and build-cache simulator, a mini `dvc repro`, a workflow checker). Everything MLflow writes goes to temp folders.

## Notebooks (about 10 hours in total)

| # | Notebook | What you do | Time |
|---|---|---|---|
| 1 | [01_mlops_lifecycle_and_maturity](01_mlops_lifecycle_and_maturity.ipynb) | The lifecycle, a silent training-serving skew you measure, a run manifest, maturity levels | 60 min |
| 2 | [02_experiment_tracking_with_mlflow](02_experiment_tracking_with_mlflow.ipynb) | Params, metrics, artifacts, nested runs; compare a model family and query runs | 75 min |
| 3 | [03_model_packaging_and_registry](03_model_packaging_and_registry.ipynb) | joblib limits, MLflow model format, signatures, environment capture, registry aliases, promotion gate, rollback | 90 min |
| 4 | [04_serving_models_with_fastapi](04_serving_models_with_fastapi.ipynb) | Pydantic validation, health/ready, batch endpoint, latency percentiles, API versioning, safe 500s | 80 min |
| 5 | [05_data_and_model_versioning](05_data_and_model_versioning.ipynb) | Content-addressed storage, `dvc.yaml`/`params.yaml`, a working mini `repro`, restore old versions | 75 min |
| 6 | [06_docker_for_ml_services](06_docker_for_ml_services.ipynb) | Dockerfile best practice, a linter, a build-cache simulator, `.dockerignore`, compose validation | 60 min |
| 7 | [07_monitoring_and_drift_detection](07_monitoring_and_drift_detection.ipynb) | PSI, KS, chi-square, false-alarm rates, simulated drift, delayed labels, response policy | 90 min |
| 8 | [08_ci_cd_and_testing_ml_systems](08_ci_cd_and_testing_ml_systems.ipynb) | Data/feature/behaviour tests with real pytest, sabotage to prove tests fail, metric gate, Actions workflow | 90 min |

Run them in order from JupyterLab. Notebooks 2, 3, 5 and 8 start subprocesses or MLflow for real, so the first cells can take up to a minute.

## You are done when you can...

- [ ] name the lifecycle stages and explain training-serving skew with an example you reproduced
- [ ] record a run so it can be reproduced: params, seed, data hash, metrics, artifacts, environment
- [ ] compare models fairly by selecting on cross-validation and reporting the test set once, with a dummy baseline in the table
- [ ] explain why a bare pickle is risky (code execution, environment, no interface) and package a model with a signature and pinned requirements
- [ ] promote and roll back a model with registry aliases, and encode the promotion rule in code
- [ ] build a FastAPI service with strict input validation, health and readiness, a batch endpoint, versioned paths, and report p50/p95 latency
- [ ] describe how DVC works (hashes, pointers, `dvc.yaml`, lock file) and restore a past data or model version
- [ ] write a Dockerfile that is cache-friendly, non-root, pinned and secret-free, and review someone else's
- [ ] compute PSI, KS and chi-square drift checks, state their false-alarm rate, and explain what input monitoring cannot see
- [ ] write invariance, directional and minimum-functionality tests, and a metric gate whose tolerance comes from measured noise
- [ ] read a GitHub Actions workflow and list its risks (unpinned actions, secrets, missing timeouts)

## Try next (real resources)

- MLflow documentation (mlflow.org/docs/latest): Tracking, Models, Model Registry
- Google Cloud architecture guide, "MLOps: Continuous delivery and automation pipelines in machine learning"
- *Designing Machine Learning Systems*, Chip Huyen (O'Reilly)
- Sculley et al., "Hidden Technical Debt in Machine Learning Systems" (NeurIPS 2015)
- DVC documentation (dvc.org/doc), FastAPI documentation (fastapi.tiangolo.com), Evidently documentation (docs.evidentlyai.com)

# Capstone: Customer Churn Prediction Service

Everything you learned, in one repository you can show to an interviewer: a tested service that takes customer records,
returns a churn probability and a retention decision, refuses bad input, tells you which model answered, and warns you when the
world stops looking like the training data.

## The brief

> A subscription business loses about a third of its customers. The retention team can contact a limited number of customers with
> an offer. **Build the system that tells them whom to contact, prove it beats the obvious alternatives, ship it as a service, and
> describe honestly where it can fail.**

Data: `datasets/churn.csv`, 5,000 synthetic customers (see `docs/DATASETS.md`; generate it with `python datasets/make_datasets.py`).
Because it is synthetic, the numbers show you the *method*. They say nothing about real telecom churn.

What is in the repository:

| Path | What it is | Verified here? |
|---|---|---|
| `src/churn/` | `data`, `features`, `train`, `predict`, `api`, `monitor`, `registry`, `config` | yes, by `tests/` |
| `configs/config.yaml` | one source of truth: paths, split, **cost matrix**, models, release gate, drift thresholds | yes |
| `tests/` | data validation, no-leakage, quality gate (floor derived from baselines), API via `TestClient`, drift sanity, infra files | yes (pytest) |
| `docker/` | `Dockerfile` (slim, non-root, layer-cached), `docker-compose.yml`, `requirements-serve.txt` | **files parsed only; never built (no Docker here)** |
| `airflow/dags/churn_training_dag.py` | weekly validate, train, gate, register | **parsed only; never run (Airflow not installed)** |
| `dvc/dvc.yaml`, `dvc/params.yaml` | reproducible training stage | **unverified sketch (DVC not installed)** |
| `.github/workflows/ci.yml` | lint and test on push | **never run on GitHub; `ruff` step unverified** |
| `sources/25_End_to_End_Project/*.py` | the seven notebooks' source (edit these, then rebuild) | executed by `tools/check_notebooks.py` |

## The seven notebooks

| # | Notebook | You leave with |
|---|---|---|
| 01 | Problem framing and success criteria | a cost matrix with its derivation, baselines, a definition of done |
| 02 | Data understanding and cleaning | a schema, cleaning rules, and the `None`-is-not-missing trap understood |
| 03 | Features and a baseline model | a leakage-proof pipeline and the number every later model must beat |
| 04 | Modelling, tuning and threshold selection | a cost-based threshold and its sensitivity to the assumptions |
| 05 | Evaluation, interpretation and the model card | intervals, calibration, subgroup checks, importance, a written card |
| 06 | Packaging, serving and testing | the real package run in a temp folder, API calls, a baseline-derived gate floor |
| 07 | Monitoring, deployment and next steps | drift simulations, a retraining policy, shadow/canary, what each infra file does |

Open them in JupyterLab from the `25_End_to_End_Project/` folder (they import the package through `sys.path`; nothing is installed).

## Milestone checklist

Tick these in order. Each one has a command that proves it.

- [ ] **M0 Environment.** `python --version` shows 3.11 and `python -c "import pandas, sklearn, fastapi"` works.
- [ ] **M1 Frame it.** Notebook 01 done; you can explain the cost matrix (`tn fp fn tp`) from memory and why 0.5 is the wrong threshold.
- [ ] **M2 Clean it.** Notebook 02 done; `load_dataset` returns a validated frame and you can name each cleaning rule.
- [ ] **M3 Baseline.** Notebook 03 done; you know the baseline AUC and the cost of the best dumb policy.
- [ ] **M4 Model and threshold.** Notebook 04 done; you can predict which way the threshold moves when the offer gets more expensive.
- [ ] **M5 Evaluate.** Notebook 05 done; you have a model card you would be comfortable having a stranger read.
- [ ] **M6 Ship.** `python -m pytest tests -q` passes; you trained with the CLI and `curl`-ed `/health` and `/predict`.
- [ ] **M7 Operate.** Notebook 07 done; you wrote a runbook entry (exercise 4) and can say which infra files you have actually run.
- [ ] **M8 Make it yours.** Change one thing on purpose (a new feature in `features.py`, another cost matrix, a new validation rule), add a test for it, and explain the effect.
- [ ] **M9 Publish.** Put the repository on GitHub with this README, the model card and a short "what I would do next" section.

## Self-assessment rubric

Score each row 0 to 3 (0 = not attempted, 1 = copied it, 2 = can explain and modify it, 3 = can defend it in an interview and knows where it breaks).
Below 24 of 36: redo the weak rows before moving on.

| Area | 3 looks like |
|---|---|
| Problem framing | costs are traceable to assumptions; you can argue the assumptions are wrong and show what changes |
| Data quality | you can list what cleaning does, what it refuses to do, and why "None" is not NaN |
| Leakage | you can demonstrate leakage on noise and show the Pipeline preventing it |
| Validation discipline | train fits, validation chooses, test is opened once; you can say what happens if you break that |
| Model comparison | you test whether a gap is noise before picking a winner, and prefer the simpler model on a tie |
| Threshold and cost | you explain why the threshold moves with the cost matrix, and what calibration has to do with it |
| Evaluation honesty | every headline number has an interval; limitations in the model card are specific, not generic |
| Serving | bad input gets 422, not a guess; responses carry a model version; the model loads once |
| Testing | you can name what each test file protects you from, and which checks are only static |
| Operations | you can distinguish input drift from concept drift and describe a retraining policy with reasons |
| Release safety | you can describe shadow, canary and rollback, and deterministic bucketing |
| Reproducibility and honesty | one config, seeds, pinned serving versions, and a clear statement of what was never executed |

## Run it: Windows PowerShell

From the project root (`Data-Science-Roadmap`). Replace `.venv\Scripts\python` with `python` if you activated the environment.

```powershell
# one-time: generate data (skip if datasets\churn.csv exists)
.venv\Scripts\python datasets\make_datasets.py

# tests (under a minute; nothing is written into the repository)
.venv\Scripts\python -m pytest tests -q -p no:cacheprovider

# train: writes artifacts\model\{model.joblib, metrics.json, model_card.md, reference.csv}
$env:PYTHONPATH = "src"
.venv\Scripts\python -m churn.train --config configs\config.yaml --fail-on-gate

# serve, then call it from a second window
$env:CHURN_MODEL_PATH = "artifacts\model\model.joblib"
.venv\Scripts\python -m uvicorn churn.api:app --port 8000
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" `
  -d '{\"tenure_months\":3,\"contract\":\"Month-to-month\",\"internet_service\":\"Fiber\",\"monthly_charges\":89.9,\"total_charges\":270.0,\"support_calls\":5,\"paperless_billing\":\"Yes\",\"autopay\":\"No\",\"senior_citizen\":0}'
# interactive docs: http://127.0.0.1:8000/docs

# drift report (here: the training reference against itself, so everything is "ok")
.venv\Scripts\python -m churn.monitor --reference artifacts\model\reference.csv --current artifacts\model\reference.csv

# rebuild and execute the notebooks
.venv\Scripts\python tools\build_notebooks.py 25_End_to_End_Project
.venv\Scripts\python tools\check_notebooks.py 25_End_to_End_Project
```

## Run it: Git Bash

```bash
.venv/Scripts/python datasets/make_datasets.py      # one-time, if datasets/churn.csv is missing

.venv/Scripts/python -m pytest tests -q -p no:cacheprovider

export PYTHONPATH=src
.venv/Scripts/python -m churn.train --config configs/config.yaml --fail-on-gate

export CHURN_MODEL_PATH=artifacts/model/model.joblib
.venv/Scripts/python -m uvicorn churn.api:app --port 8000
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"tenure_months":3,"contract":"Month-to-month","internet_service":"Fiber","monthly_charges":89.9,"total_charges":270.0,"support_calls":5,"paperless_billing":"Yes","autopay":"No","senior_citizen":0}'

.venv/Scripts/python -m churn.monitor --reference artifacts/model/reference.csv --current artifacts/model/reference.csv
.venv/Scripts/python tools/build_notebooks.py 25_End_to_End_Project
.venv/Scripts/python tools/check_notebooks.py 25_End_to_End_Project
```

The train, serve and monitor commands above were run in Git Bash while writing this README (training reported gate PASS, `/health`
and `/predict` answered, the self-drift report said `ok`). The PowerShell quoting of the JSON body for `curl.exe` was **not** run;
if it fights you, save the JSON to a file and use `-d "@payload.json"`.

The `artifacts/` folder is git-ignored. With `--fail-on-gate` the command exits with code 2 when the release gate fails, which is what a pipeline keys on.

### The files that were not executed here

```bash
# Docker (needs Docker Desktop / engine; never built by the author)
.venv/Scripts/python -m churn.train                        # the compose file mounts artifacts/model
docker compose -f docker/docker-compose.yml up --build api
curl http://localhost:8000/health

# DVC (needs `pip install dvc`; unverified sketch)
dvc init && dvc repro dvc/dvc.yaml

# Airflow: install it in a separate environment following the Airflow docs (with its constraints file),
# make the `churn` package importable there, then copy airflow/dags/churn_training_dag.py into your dags folder.

# GitHub Actions: push the repository; the workflow at .github/workflows/ci.yml runs on push and pull requests.
```

Expect to fix small things the first time (an unavailable pin, an Airflow version difference). Fixing them is part of the exercise; send the fix back as a pull request to your own repo.

## Four more portfolio projects

Each one proves a different skill set. Every dataset below is public and exists; check each one's licence and size before you download, and keep raw data out of git.
Use the same habits as this capstone: a baseline first, a cost or business metric, a test suite, a README with limitations.

### 1. Demand forecasting with honest backtesting
- **Dataset:** the M5 Forecasting (Accuracy) data from the Walmart-based Kaggle competition (daily unit sales by item and store, with prices and calendar events). A lighter alternative is the UCI "Individual household electric power consumption" dataset.
- **Build:** a forecasting pipeline with a seasonal-naive baseline, gradient boosting with lag and calendar features, and a rolling-origin backtest (never a random split).
- **Skills proven:** time-series validation, leakage through the calendar, choosing a scale-free error metric (MASE or RMSSE), reporting error by segment.
- **Stretch:** predict quantiles and report **interval coverage** (do "90%" intervals contain the truth 90% of the time?), then serve the forecast behind an API with a drift monitor like `churn.monitor`.

### 2. Question answering over documents (RAG) with a real evaluation
- **Dataset:** SQuAD 2.0 (Stanford Question Answering Dataset): Wikipedia paragraphs with questions, including questions that have no answer in the paragraph.
- **Build:** chunk and index the paragraphs, retrieve with BM25 and/or embeddings, and answer with a language model (you need a model you can call; a small local one is fine). Evaluate **retrieval** separately from **generation**.
- **Skills proven:** retrieval metrics (recall at k, MRR), answer metrics (exact match, F1), abstaining when the answer is not there, prompt and chunk-size ablations.
- **Stretch:** hybrid retrieval plus a re-ranker, and a citation check that flags answers whose quoted span does not appear in the retrieved text.

### 3. SQL warehouse and BI dashboard
- **Dataset:** NYC Taxi and Limousine Commission (TLC) Trip Record Data: monthly files of yellow and green taxi trips (published as Parquet), plus the taxi zone lookup table.
- **Build:** load a few months into DuckDB or another warehouse, model a star schema (trips fact, zone, time and payment dimensions), write the KPI queries in SQL, and publish a dashboard in Power BI or Tableau.
- **Skills proven:** dimensional modelling, window functions and CTEs, data-quality checks (negative fares, impossible durations, timestamps outside the file's month), a dashboard with a clear question.
- **Stretch:** incremental monthly loads with automated data-quality tests, and a documented metric layer so two dashboards cannot define "revenue" differently.

### 4. A database agent behind an MCP server
- **Dataset:** the Chinook sample database (a digital music store: artists, albums, tracks, customers, invoices), available as SQLite among other formats.
- **Build:** an MCP server exposing read-only tools (list tables, describe a table, run a limited `SELECT`) and a schema resource, plus an agent that answers business questions by calling them. Write 20 to 30 questions with hand-verified answers as an evaluation set.
- **Skills proven:** tool design, guardrails (read-only connection, row limits, timeouts, rejecting non-`SELECT` statements), agent evaluation against gold answers, and logging every tool call.
- **Stretch:** add this capstone's `/predict` endpoint as a second tool to an agent that also reads customer records, and test prompt-injection attempts planted in text fields.

## If you get stuck

- A test fails after you change the config: the gate floor and the threshold are **derived from data** (notebook 06); the failing message tells you which one to re-derive.
- `ModuleNotFoundError: churn`: set `PYTHONPATH=src` (or run inside pytest, which adds it for you).
- A warning about scikit-learn versions when loading a model: retrain with the version you serve with. Model files are tied to the library version that wrote them.

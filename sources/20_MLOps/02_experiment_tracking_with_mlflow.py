# %% [markdown]
# # Experiment Tracking with MLflow
#
# **Why this matters.** By the third week of a modelling project you have forty runs, and someone asks "which settings gave the 0.82 we showed on Tuesday?" If the answer lives in notebook cell history and a spreadsheet you stopped updating, you cannot reproduce your best result.
# An **experiment tracker** records, for every run, the parameters, metrics, artifacts, code and data identity, so the question takes ten seconds to answer and the result can be rebuilt.
#
# **You will learn**
# - record **parameters, metrics (with steps), tags and artifacts** for a run with MLflow
# - run a small model comparison on the churn data and log every candidate
# - **query** past runs (`search_runs`) and pick the winner with a filter and a sort
# - group related runs with **nested runs** (for example one child per cross-validation fold)
# - avoid the classic tracking mistakes: selecting on the test set, unlogged data versions, unseeded runs
#
# **Prerequisites:** [MLOps lifecycle](01_mlops_lifecycle_and_maturity.ipynb) and [Cross-validation](../12_Machine_Learning/07_cross_validation_and_model_evaluation.ipynb).
#
# > Data used: `datasets/churn.csv` (synthetic, see `docs/DATASETS.md`). Everything MLflow writes goes to a temp folder that is deleted at exit.

# %%
import os

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import atexit
import hashlib
import json
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
DATA = Path("../datasets")
TMP = Path(tempfile.mkdtemp(prefix="mlflow_demo_"))
atexit.register(shutil.rmtree, TMP, ignore_errors=True)

# %% [markdown]
# ## 1. A tracking store in a temp folder
# MLflow needs two places: a **backend store** (run metadata: params, metrics, tags) and an **artifact store** (files). Locally, a SQLite file plus a folder is the simplest durable setup.
# (MLflow's older plain-folder `./mlruns` store is now in maintenance mode, and recent versions refuse to use it by default, so we use SQLite, which is also what you would point a team server at.)

# %%
mlflow.set_tracking_uri(f"sqlite:///{(TMP / 'mlflow.db').as_posix()}")
EXP = "churn-baselines"
exp_id = mlflow.create_experiment(EXP, artifact_location=(TMP / "artifacts").as_uri())
mlflow.set_experiment(EXP)
client = MlflowClient()
print("tracking store:", mlflow.get_tracking_uri().split("/")[0], "| experiment id:", exp_id)

# %% [markdown]
# ## 2. The data and a helper that evaluates honestly
# Same cleaning as the previous notebook. The discipline that matters for tracking: **choose models with cross-validation on the training set; touch the test set once.** We will log both numbers but select on the first.

# %%
raw = pd.read_csv(DATA / "churn.csv", keep_default_na=False, na_values=[""])
NUM = ["tenure_months", "monthly_charges", "total_charges", "support_calls"]
CAT = ["contract", "internet_service", "paperless_billing", "autopay", "senior_citizen"]

def clean(df):
    out = df.copy()
    out["contract"] = out["contract"].str.strip().str.lower()
    for c in ("paperless_billing", "autopay", "senior_citizen"):
        out[c] = out[c].astype(str)
    return out[NUM + CAT]

train_raw, test_raw = train_test_split(raw, test_size=0.25, stratify=raw["churned"], random_state=42)
X_train, y_train = clean(train_raw), train_raw["churned"]
X_test, y_test = clean(test_raw), test_raw["churned"]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def preprocessor(scale=True):
    num = Pipeline([("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else []))
    return ColumnTransformer([("num", num, NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])

def data_fingerprint():
    return hashlib.sha256((DATA / "churn.csv").read_bytes()).hexdigest()[:16]

print(X_train.shape, X_test.shape, "data fingerprint:", data_fingerprint())

# %% [markdown]
# ## 3. Your first tracked run
# A **run** is one execution you want to remember. Use it as a context manager so it is always closed (and marked failed if an exception escapes).
# The four things you log:
# * `log_param`: inputs you chose (hyper-parameters, the data version). Logged once, immutable.
# * `log_metric`: outputs you measured. A metric may be logged many times with a `step` (a training curve).
# * `set_tag`: free-form labels for searching (`model_family`, `owner`, `data_sha`).
# * `log_artifact` / `log_figure` / `log_dict`: files (plots, reports, serialized objects).

# %%
with mlflow.start_run(run_name="logreg-C1") as run:
    pipe = Pipeline([("pre", preprocessor()), ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42))])
    mlflow.log_params({"model": "logreg", "C": 1.0, "max_iter": 1000, "seed": 42})
    mlflow.set_tags({"model_family": "linear", "data_sha": data_fingerprint()})
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=CV, scoring="roc_auc")
    mlflow.log_metric("cv_auc", float(cv_scores.mean()))
    mlflow.log_metric("cv_auc_std", float(cv_scores.std()))
    first_run_id = run.info.run_id

print("run id:", first_run_id[:8], "| cv AUC:", round(cv_scores.mean(), 4))

# %% [markdown]
# *What just happened?* MLflow opened a run, stored the params/metrics/tags in the SQLite file, and closed the run. Read it back from the store, not from your variables. That is the point: the record outlives the notebook.

# %%
r = client.get_run(first_run_id)
print("status:", r.info.status)
print("params:", r.data.params)
print("metrics:", {k: round(v, 4) for k, v in r.data.metrics.items()})
print("tags (ours):", {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")})

# %% [markdown]
# Note that MLflow stores params as **strings** (`C` came back as `'1.0'`) and metrics as floats. Convert when you query.

# %% [markdown]
# ## 4. Metrics over steps, and artifacts
# A training curve is one metric logged at many steps. We train a gradient-boosted model stage by stage and log the validation AUC after each of 40 trees, then save a plot and a small JSON file as artifacts.

# %%
from sklearn.ensemble import GradientBoostingClassifier

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=42)
pre = preprocessor().fit(X_tr)
gb = GradientBoostingClassifier(n_estimators=40, learning_rate=0.1, max_depth=2, random_state=42).fit(pre.transform(X_tr), y_tr)

with mlflow.start_run(run_name="gbm-learning-curve") as run:
    mlflow.log_params({"model": "gbm", "n_estimators": 40, "learning_rate": 0.1, "max_depth": 2})
    curve = []
    for step, proba in enumerate(gb.staged_predict_proba(pre.transform(X_val)), start=1):
        auc = roc_auc_score(y_val, proba[:, 1])
        curve.append(auc)
        mlflow.log_metric("val_auc", auc, step=step)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(1, len(curve) + 1), curve)
    ax.set(title="Validation AUC while boosting", xlabel="trees (boosting stage)", ylabel="validation ROC-AUC")
    mlflow.log_figure(fig, "val_auc_curve.png")
    plt.show()
    mlflow.log_dict({"features": NUM + CAT, "n_train_rows": len(X_tr)}, "feature_list.json")
    curve_run_id = run.info.run_id

hist = client.get_metric_history(curve_run_id, "val_auc")
print("logged", len(hist), "points; first/last:", round(hist[0].value, 4), "/", round(hist[-1].value, 4))
print("artifacts:", [a.path for a in client.list_artifacts(curve_run_id)])

# %% [markdown]
# Artifacts are stored by the artifact store (here a folder, in production S3/GCS/Azure Blob) and linked to the run, so the plot you show in a review is the plot of *that* run.

# %% [markdown]
# ## 5. A real comparison: log every candidate
# Now the usual loop: several model families and settings, each one a run. Write the loop once and let the tracker remember the results. Keep it small so it runs quickly.

# %%
candidates = {
    "logreg-C0.1":   ("linear", {"model": "logreg", "C": 0.1},                     lambda: Pipeline([("pre", preprocessor()), ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=42))])),
    "logreg-C10":    ("linear", {"model": "logreg", "C": 10.0},                    lambda: Pipeline([("pre", preprocessor()), ("clf", LogisticRegression(C=10.0, max_iter=1000, random_state=42))])),
    "rf-depth4":     ("rf",     {"model": "rf", "max_depth": 4, "n_estimators": 100},  lambda: Pipeline([("pre", preprocessor(False)), ("clf", RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=1))])),
    "rf-depth12":    ("rf",     {"model": "rf", "max_depth": 12, "n_estimators": 100}, lambda: Pipeline([("pre", preprocessor(False)), ("clf", RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=1))])),
    "hgb-lr0.05":    ("gbm",    {"model": "hgb", "learning_rate": 0.05, "max_iter": 150}, lambda: Pipeline([("pre", preprocessor(False)), ("clf", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=150, random_state=42))])),
    "hgb-lr0.2":     ("gbm",    {"model": "hgb", "learning_rate": 0.2, "max_iter": 150},  lambda: Pipeline([("pre", preprocessor(False)), ("clf", HistGradientBoostingClassifier(learning_rate=0.2, max_iter=150, random_state=42))])),
}

def track_candidate(name, family, params, build):
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({**params, "seed": 42})
        mlflow.set_tags({"model_family": family, "data_sha": data_fingerprint()})
        pipe = build()
        t0 = time.perf_counter()
        scores = cross_val_score(pipe, X_train, y_train, cv=CV, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        mlflow.log_metric("fit_and_cv_seconds", time.perf_counter() - t0)
        mlflow.log_metrics({"cv_auc": float(scores.mean()), "cv_auc_std": float(scores.std()),
                            "test_auc": float(roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]))})
        return run.info.run_id

t0 = time.perf_counter()
ids = {n: track_candidate(n, *spec) for n, spec in candidates.items()}
print(f"tracked {len(ids)} candidates in {time.perf_counter() - t0:.1f}s")

# %% [markdown]
# ## 6. Querying runs
# `mlflow.search_runs` returns a pandas DataFrame with one row per run and columns `params.*`, `metrics.*`, `tags.*`. You can filter with a small SQL-like string and sort on the server side.

# %%
runs = mlflow.search_runs(experiment_names=[EXP], order_by=["metrics.cv_auc DESC"])
view = runs[["tags.mlflow.runName", "tags.model_family", "metrics.cv_auc", "metrics.cv_auc_std", "metrics.test_auc"]].dropna(subset=["metrics.cv_auc"])
view.columns = ["run", "family", "cv_auc", "cv_std", "test_auc"]
view.round(4)

# %% [markdown]
# (The `NaN` in one `test_auc` cell is the very first run from section 3, which never logged it: tracked data is only as complete as what you log, and missing values are how you notice inconsistent logging.)
# Read this table the way a reviewer would. Differences smaller than the fold-to-fold spread (`cv_std`) are not evidence that one model is better. And we **select on `cv_auc`**: `test_auc` is logged for the report, but if you sort by it and pick the top, you have leaked the test set into model selection.
# Filters answer targeted questions:

# %%
rf = mlflow.search_runs(experiment_names=[EXP], filter_string="tags.model_family = 'rf' AND metrics.cv_auc > 0",
                        order_by=["metrics.cv_auc DESC"])
print("best random forest:", rf.loc[0, "tags.mlflow.runName"], "| params.max_depth =", rf.loc[0, "params.max_depth"], "(string!)")
best = mlflow.search_runs(experiment_names=[EXP], filter_string="metrics.cv_auc > 0", order_by=["metrics.cv_auc DESC"], max_results=1).iloc[0]
print("overall best:", best["tags.mlflow.runName"], "| run id", best["run_id"][:8])

# %% [markdown]
# ## 7. Nested runs: one child per fold
# A parent run can own child runs, handy for cross-validation folds or a hyper-parameter search where each trial is itself a run. The UI shows them as a tree, and each child has its own metrics.

# %%
def tracked_cv(name, pipe, X, y, cv):
    """Parent run = summary; one nested child run per fold."""
    with mlflow.start_run(run_name=name) as parent:
        fold_aucs = []
        for k, (tr, va) in enumerate(cv.split(X, y)):
            with mlflow.start_run(run_name=f"{name}-fold{k}", nested=True):
                fitted = pipe.fit(X.iloc[tr], y.iloc[tr])
                auc = roc_auc_score(y.iloc[va], fitted.predict_proba(X.iloc[va])[:, 1])
                mlflow.log_metric("fold_auc", auc)
                fold_aucs.append(auc)
        mlflow.log_metric("cv_auc", float(np.mean(fold_aucs)))
        return parent.info.run_id

parent_id = tracked_cv("logreg-nested", Pipeline([("pre", preprocessor()), ("clf", LogisticRegression(max_iter=1000, random_state=42))]), X_train, y_train, CV)
children = mlflow.search_runs(experiment_names=[EXP], filter_string=f"tags.mlflow.parentRunId = '{parent_id}'")
print("children found:", len(children), "| fold AUCs:", children["metrics.fold_auc"].round(4).tolist())

# %% [markdown]
# ## 8. Autologging, and a shared server (shown, not run)
# `mlflow.autolog()` hooks into scikit-learn (and PyTorch, XGBoost, ...) and records params, metrics and the model for you. It is convenient for exploration. Its trade-off: you get everything, including things you did not want, and you lose control over naming. Serious projects usually log explicitly (as above) for the metrics that matter and use autolog for the rest.
# The first `log_model` call in a fresh environment can be slow while MLflow infers package requirements; we come back to models in the next notebook.

# %% tags=["no-run"]
# Autologging (one line before training):
mlflow.sklearn.autolog(log_models=False)
pipe.fit(X_train, y_train)            # params, training metrics and a few diagnostics are logged automatically

# A shared tracking server for a team (a terminal command, not Python):
#   mlflow server --backend-store-uri postgresql://user:pass@db:5432/mlflow \
#                 --default-artifact-root s3://my-bucket/mlflow-artifacts --host 0.0.0.0 --port 5000
# and every teammate then does:  mlflow.set_tracking_uri("http://tracking-host:5000")
# (Do not expose such a server to the internet as is: put it behind TLS and authentication, and keep the database credentials out of shell history. Check the MLflow docs for the auth options of your version.)

# %% [markdown]
# ## What to log (a checklist)
# * **Params**: every hyper-parameter, seeds, feature-set name, train/test split rule.
# * **Data identity**: a content hash or table version (we used a short SHA as a tag). Without this, "same code, different result" is unsolvable.
# * **Metrics**: the selection metric (CV), the final report metric (test), and cost metrics (fit time, model size).
# * **Code version**: the git commit. MLflow records it automatically if you run from a repository.
# * **Artifacts**: plots, the confusion matrix, feature list, and the serialized model (next notebook).
# * **Environment**: library versions (the model-logging step records them).

# %% [markdown]
# ## Common mistakes
#
# 1. **Picking the winner by test-set metric.** Every time you look at the test score and choose, it becomes a validation set. Select on CV; report the test score once, for the chosen model.
# 2. **Not logging the data version.** Two runs with identical params can differ because the data changed. Log a hash, a table snapshot id, or a dataset version.
# 3. **Forgetting params are strings.** `params.max_depth` comes back as `'12'`; sorting it gives `'12' < '4'`. Cast before you compare.
# 4. **Unseeded runs.** If a rerun gives a different number you cannot tell noise from change. Log and fix the seed.
# 5. **Logging nothing on failure.** Use `with mlflow.start_run()` so failed runs are recorded as FAILED; add a tag with the error type when you catch one.
# 6. **One giant experiment for everything.** Use experiments per problem and tags/run names for variants, or finding anything becomes impossible.
# 7. **Treating tracking as a leaderboard.** A 0.002 AUC gap inside the fold spread is noise. Look at `cv_auc_std`.

# %% [markdown]
# ## Interview corner
#
# **1. What do you record for each ML experiment, and why?**
# <details><summary>Answer</summary>
# Parameters and seeds, metrics (including the validation strategy), data version, code commit, environment, and artifacts like the model and plots. Together they let you reproduce the run, compare it fairly with others, and audit why a model was chosen.
# </details>
#
# **2. How do you compare runs without fooling yourself?**
# <details><summary>Answer</summary>
# Same data splits and metric for all runs, select on cross-validation or a validation set rather than the test set, look at the spread across folds or seeds and not just the mean, and keep a dummy baseline in the same table so you know how much signal there is.
# </details>
#
# **3. What is the difference between a tracking server's backend store and artifact store?**
# <details><summary>Answer</summary>
# The backend store is a database holding run metadata (params, metrics, tags, run status, registry entries). The artifact store holds files (models, plots, datasets), typically object storage. Separating them lets metadata stay queryable and small while large files live in cheap storage.
# </details>
#
# **4. When would you use nested runs?**
# <details><summary>Answer</summary>
# Whenever one logical activity consists of many similar sub-runs: CV folds, the trials of a hyper-parameter search, or per-segment models. The parent holds the summary and the children keep detail, so the top-level run list stays readable.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Track the baseline you must beat
# Every comparison needs a dumb baseline. Inside one MLflow run named `dummy-prior`, fit `DummyClassifier(strategy="prior")` in the standard pipeline style, cross-validate **AUC** on the training set with `CV`, and log: param `model="dummy"`, metric `cv_auc`, and tag `model_family="baseline"`.
# Store the run's id in `dummy_id`.

# %% tags=["starter"]
dummy_id = ...

# %%
r = client.get_run(dummy_id)
assert r.info.status == "FINISHED", "run must finish normally"
assert r.data.tags.get("mlflow.runName") == "dummy-prior", "run name must be 'dummy-prior'"
assert r.data.params.get("model") == "dummy" and r.data.tags.get("model_family") == "baseline"
expected = cross_val_score(DummyClassifier(strategy="prior"), X_train, y_train, cv=CV, scoring="roc_auc").mean()
assert abs(r.data.metrics["cv_auc"] - expected) < 1e-9, f"cv_auc should be {expected}"
print("Exercise 1 passed: a constant predictor has AUC", round(r.data.metrics["cv_auc"], 3))

# %% tags=["solution"]
with mlflow.start_run(run_name="dummy-prior") as run:
    mlflow.log_param("model", "dummy")
    mlflow.set_tag("model_family", "baseline")
    scores = cross_val_score(DummyClassifier(strategy="prior"), X_train, y_train, cv=CV, scoring="roc_auc")
    mlflow.log_metric("cv_auc", float(scores.mean()))
    dummy_id = run.info.run_id

# %% [markdown]
# ### Exercise 2: Query, do not scroll
# Using `mlflow.search_runs` (not by reading your own variables), find: `best_hgb_name`, the **run name** of the gradient-boosting (`model_family = 'gbm'`) run with the highest `cv_auc` among runs in this experiment; and `n_beating_baseline`, the **number of runs with `cv_auc` strictly greater than the dummy baseline's** `cv_auc` (from Exercise 1).
# *Hint: `filter_string` supports `AND`, `>`, and `=`; params/tags need the `params.` / `tags.` prefix. Rows with no `cv_auc` (for example the learning-curve run) must not be counted.*

# %% tags=["starter"]
best_hgb_name = ...
n_beating_baseline = ...

# %%
all_runs = mlflow.search_runs(experiment_names=[EXP])
g = all_runs[all_runs["tags.model_family"] == "gbm"].dropna(subset=["metrics.cv_auc"]).sort_values("metrics.cv_auc", ascending=False)
assert best_hgb_name == g.iloc[0]["tags.mlflow.runName"], f"expected {g.iloc[0]['tags.mlflow.runName']}"
base = client.get_run(dummy_id).data.metrics["cv_auc"]
assert n_beating_baseline == int((all_runs["metrics.cv_auc"] > base).sum()), "count only runs whose cv_auc is strictly above the dummy"
print("Exercise 2 passed:", best_hgb_name, "|", n_beating_baseline, "runs beat the baseline")

# %% tags=["solution"]
gbm = mlflow.search_runs(experiment_names=[EXP], filter_string="tags.model_family = 'gbm' AND metrics.cv_auc > 0", order_by=["metrics.cv_auc DESC"])
best_hgb_name = gbm.loc[0, "tags.mlflow.runName"]
base = client.get_run(dummy_id).data.metrics["cv_auc"]
n_beating_baseline = len(mlflow.search_runs(experiment_names=[EXP], filter_string=f"metrics.cv_auc > {base}"))

# %% [markdown]
# ### Exercise 3: Log a run that can be reproduced
# Write `log_reproducible_run(name, C)` that trains a logistic regression with regularisation `C` (pipeline via `preprocessor()`, `random_state=42`) and inside a run logs: param `C`, tag `data_sha` (use `data_fingerprint()`), and metric `cv_auc`. It returns the run id.
# Then call it twice with the same `C=0.5` and store the ids in `id_a`, `id_b`. The two runs must have the same data tag and (because everything is seeded and the folds are fixed) the **same** `cv_auc`.

# %% tags=["starter"]
def log_reproducible_run(name, C):
    ...

id_a = ...
id_b = ...

# %%
ra, rb = client.get_run(id_a), client.get_run(id_b)
assert id_a != id_b, "two separate runs are expected"
assert ra.data.tags["data_sha"] == rb.data.tags["data_sha"] == data_fingerprint()
assert float(ra.data.params["C"]) == 0.5
assert ra.data.metrics["cv_auc"] == rb.data.metrics["cv_auc"], "a seeded run with fixed folds must reproduce exactly"
print("Exercise 3 passed: identical cv_auc", round(ra.data.metrics["cv_auc"], 5))

# %% tags=["solution"]
def log_reproducible_run(name, C):
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("C", C)
        mlflow.set_tag("data_sha", data_fingerprint())
        pipe = Pipeline([("pre", preprocessor()), ("clf", LogisticRegression(C=C, max_iter=1000, random_state=42))])
        mlflow.log_metric("cv_auc", float(cross_val_score(pipe, X_train, y_train, cv=CV, scoring="roc_auc").mean()))
        return run.info.run_id

id_a = log_reproducible_run("repro-a", 0.5)
id_b = log_reproducible_run("repro-b", 0.5)

# %% [markdown]
# ### Exercise 4: A hyper-parameter search as nested runs
# Write `tracked_search(Cs)` that opens a parent run `search`, and for each `C` in `Cs` a **nested child run** that logs param `C` and metric `cv_auc` (logistic regression as before). The parent logs metric `best_cv_auc` (the max over children) and param `best_C`. Return the parent run id. Call it with `[0.01, 0.1, 1.0, 10.0]` and store the result in `search_id`.

# %% tags=["starter"]
def tracked_search(Cs):
    ...

search_id = ...

# %%
kids = mlflow.search_runs(experiment_names=[EXP], filter_string=f"tags.mlflow.parentRunId = '{search_id}'")
assert len(kids) == 4, f"expected 4 child runs, found {len(kids)}"
parent = client.get_run(search_id)
assert abs(parent.data.metrics["best_cv_auc"] - kids["metrics.cv_auc"].max()) < 1e-12, "best_cv_auc must equal the best child"
best_child = kids.loc[kids["metrics.cv_auc"].idxmax()]
assert float(parent.data.params["best_C"]) == float(best_child["params.C"]), "best_C must be the C of the best child"
print("Exercise 4 passed: best C =", parent.data.params["best_C"])

# %% tags=["solution"]
def tracked_search(Cs):
    with mlflow.start_run(run_name="search") as parent:
        results = {}
        for C in Cs:
            with mlflow.start_run(run_name=f"search-C{C}", nested=True):
                pipe = Pipeline([("pre", preprocessor()), ("clf", LogisticRegression(C=C, max_iter=1000, random_state=42))])
                auc = float(cross_val_score(pipe, X_train, y_train, cv=CV, scoring="roc_auc").mean())
                mlflow.log_param("C", C)
                mlflow.log_metric("cv_auc", auc)
                results[C] = auc
        best_C = max(results, key=results.get)
        mlflow.log_param("best_C", best_C)
        mlflow.log_metric("best_cv_auc", results[best_C])
        return parent.info.run_id

search_id = tracked_search([0.01, 0.1, 1.0, 10.0])

# %% [markdown]
# ### Exercise 5 (open-ended): Would you trust this table?
# Look at the `view` table from section 6. In a markdown cell, write what you would need to know before telling a stakeholder "logistic regression is the best model": is its lead over the next run inside the fold spread, how far is everything above the dummy baseline, what does fit time cost, and what is still missing from what we logged?

# %% [markdown]
# ## Recap
# - A tracked **run** holds params (strings), metrics (floats, optionally per step), tags, and artifacts; it lives in a backend store (SQLite here) plus an artifact store (a folder here).
# - Log everything needed to reproduce: params, seed, **data fingerprint**, metrics, plots. Verify by re-running (same seed and folds gave the same AUC).
# - Query with `search_runs(filter_string=..., order_by=...)` and get a DataFrame; cast string params before comparing.
# - **Select on cross-validation, report the test set once.** Compare differences to the fold spread, and always keep a dummy baseline in the table.
# - Nested runs group CV folds or search trials under a parent.
#
# **Go deeper:** the MLflow documentation (mlflow.org/docs/latest), sections "Tracking" and "Search Runs"; scikit-learn user guide "Cross-validation: evaluating estimator performance".
#
# **Next:** [Model packaging and registry](03_model_packaging_and_registry.ipynb)

# %% [markdown]
# # The MLOps Lifecycle and Maturity Levels
#
# **Why this matters.** A notebook with a good AUC is maybe 10% of a working ML system. The other 90% is getting data in reliably, training the same way every time, shipping the model behind an interface, noticing when it gets worse, and being able to roll it back.
# Teams that skip this produce models that work once, on one laptop, and quietly rot. **MLOps** is the set of habits (and some tooling) that turns a model into a dependable service.
#
# **You will learn**
# - draw the ML lifecycle and say which step each later notebook in this module covers
# - reproduce a training run by recording a **run manifest** (data hash, parameters, library versions, seed)
# - see a **silent failure** for yourself: a serving bug that raises no error but costs accuracy
# - place a team on the three common **maturity levels** and pick the next improvement that pays off
# - decide honestly whether a project needs heavy MLOps at all
#
# **Prerequisites:** [Feature engineering and pipelines](../12_Machine_Learning/08_feature_engineering_and_pipelines.ipynb) and [Pipeline design](../19_Data_Pipelines/01_pipeline_design_and_idempotency.ipynb).
#
# > Data used: `datasets/churn.csv`, a synthetic telecom table (see `docs/DATASETS.md`). Read with `keep_default_na=False` because the text `None` in `internet_service` is a real category.

# %%
import hashlib
import json
import platform
import warnings
from graphlib import TopologicalSorter
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path("../datasets")
CSV = DATA / "churn.csv"
raw = pd.read_csv(CSV, keep_default_na=False, na_values=[""])
print(raw.shape, "| churn rate:", round(raw["churned"].mean(), 3))

# %% [markdown]
# ## 1. The lifecycle
# ML systems are loops, not lines. Every arrow below is a place where work is either automated or done by hand (and therefore sometimes done wrong).
# We can write the dependencies as data and let `graphlib` order them, the same idea as the DAG notebook:

# %%
lifecycle = {
    "collect_data":      set(),
    "validate_data":     {"collect_data"},
    "build_features":    {"validate_data"},
    "train":             {"build_features"},
    "evaluate":          {"train"},
    "register_model":    {"evaluate"},
    "deploy":            {"register_model"},
    "monitor":           {"deploy"},
    "retrain_trigger":   {"monitor"},          # drift or decay sends you back to the start
}
order = list(TopologicalSorter(lifecycle).static_order())
print(" -> ".join(order), "-> (back to collect_data)")

# %% [markdown]
# | Stage | What goes wrong without discipline | Notebook |
# |---|---|---|
# | validate / features | training on data nobody checked | [Data pipelines](../19_Data_Pipelines/) |
# | train / evaluate | "which run was the good one?" | 02 Experiment tracking |
# | register / package | the model file on someone's laptop is the production model | 03 Packaging and registry |
# | deploy | training-serving skew, no validation of inputs | 04 Serving with FastAPI, 06 Docker |
# | data + model versions | cannot reproduce last quarter's model | 05 Versioning |
# | monitor | accuracy decays for weeks unnoticed | 07 Monitoring and drift |
# | automate | releases need a hero | 08 CI/CD for ML |

# %% [markdown]
# ## 2. A baseline model to operate
# We need something to operate. A small, honest pipeline: clean the table, impute and scale numbers, one-hot encode categories, logistic regression. The **cleaning step is part of the system**, and that matters in a moment.

# %%
NUM = ["tenure_months", "monthly_charges", "total_charges", "support_calls"]
CAT = ["contract", "internet_service", "paperless_billing", "autopay", "senior_citizen"]

def clean(df):
    """Training-time cleaning: fix the inconsistent casing in `contract` and cast flags to text categories."""
    out = df.copy()
    out["contract"] = out["contract"].str.strip().str.lower()
    for c in ("paperless_billing", "autopay", "senior_citizen"):
        out[c] = out[c].astype(str)
    return out

def make_model(seed=42):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, random_state=seed))])

train_raw, test_raw = train_test_split(raw, test_size=0.25, stratify=raw["churned"], random_state=42)
model = make_model().fit(clean(train_raw)[NUM + CAT], train_raw["churned"])
auc_ok = roc_auc_score(test_raw["churned"], model.predict_proba(clean(test_raw)[NUM + CAT])[:, 1])
print(f"test ROC-AUC with the cleaning step: {auc_ok:.3f}")

# %% [markdown]
# ## 3. A silent failure
# Now the model is "deployed". A second developer writes the serving code, reads the request straight into the model, and forgets to call `clean`. Nothing crashes: `OneHotEncoder(handle_unknown="ignore")` happily encodes the unseen strings `Month-to-month` or `MONTH-TO-MONTH` as all zeros.
# Watch what happens to quality:

# %%
print("contract spellings in the raw data:", sorted(raw["contract"].unique()))
serving_df = test_raw.copy()
serving_df[["paperless_billing", "autopay", "senior_citizen"]] = serving_df[["paperless_billing", "autopay", "senior_citizen"]].astype(str)   # types are fine, only casing is wrong
auc_bug = roc_auc_score(test_raw["churned"], model.predict_proba(serving_df[NUM + CAT])[:, 1])
print(f"AUC with the skew bug: {auc_bug:.3f}  (vs {auc_ok:.3f} when cleaned)  -> no exception was raised")

# %% [markdown]
# *What just happened?* This is **training-serving skew**: the model saw data in one shape at training time and another at serving time. There was no error, no log line, just worse predictions.
# MLOps exists largely to make this class of failure loud. Three defences you will build in this module: **one pipeline object that contains the cleaning** (notebook 03/04), **input validation at the API boundary** (04), and **monitoring of inputs and predictions** (07).
# How big was the damage, and could you have seen it without labels? Yes: compare a simple input statistic between training and serving.

# %%
def share_unseen(train_col, serve_col):
    known = set(train_col.unique())
    return float((~serve_col.isin(known)).mean())

cleaned_contract = clean(train_raw)["contract"]
print(f"share of serving rows whose `contract` value was never seen in training: {share_unseen(cleaned_contract, serving_df['contract']):.1%}")
print(f"same check when the serving code does clean():                           {share_unseen(cleaned_contract, clean(test_raw)['contract']):.1%}")

# %% [markdown]
# A check that costs three lines and needs no labels would have flagged the bug on day one. That is monitoring in miniature.

# %% [markdown]
# ## 4. Reproducibility: the run manifest
# "Can you rebuild the model you shipped in March?" needs four things pinned: the **data**, the **code**, the **parameters and seed**, and the **environment**. We cannot do git here, but we can record the rest in a small JSON file, the **run manifest**. Hash the data by content, not by filename:

# %%
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()

def run_manifest(data_path, params, seed, metrics=None):
    return {
        "data_sha256": sha256_file(data_path),
        "params": params,
        "seed": seed,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "metrics": metrics or {},
    }

m1 = run_manifest(CSV, {"model": "logreg", "C": 1.0}, seed=42, metrics={"auc": round(auc_ok, 4)})
print(json.dumps({**m1, "data_sha256": m1["data_sha256"][:12] + "..."}, indent=1))

# %% [markdown]
# With the manifest you can answer "what changed?" between two runs. Even without it, the test is simple: **same inputs + same code + same seed must give the same result**. Check the claim instead of assuming it:

# %%
def train_rf(seed):
    cl = clean(train_raw)
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    pipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=60, random_state=seed, n_jobs=1))])
    pipe.fit(cl[NUM + CAT], train_raw["churned"])
    return roc_auc_score(test_raw["churned"], pipe.predict_proba(clean(test_raw)[NUM + CAT])[:, 1])

seeded = [train_rf(7), train_rf(7)]
unseeded = [train_rf(None), train_rf(None)]
print("seeded  :", [round(a, 5) for a in seeded], "-> identical:", seeded[0] == seeded[1])
print("unseeded:", [round(a, 5) for a in unseeded], "-> identical:", unseeded[0] == unseeded[1])

# %% [markdown]
# Unseeded runs differ slightly. That looks harmless until you try to tell whether a change *improved* the model or you just drew a lucky seed. Seed everything you can, and record the seed.

# %% [markdown]
# ## 5. Maturity levels
# A common way to describe where a team stands (popularised in Google Cloud's MLOps guidance) is three levels:
#
# | Level | How it feels | Training | Deployment | Typical pain |
# |---|---|---|---|---|
# | **0: manual** | notebooks, a model file handed over by email/chat | by hand, occasionally | by hand | nobody can reproduce it; no monitoring |
# | **1: ML pipeline automation** | training is a pipeline that runs on a schedule or trigger | automated, with data validation and tracked experiments | model registered, serving separate | releasing the *pipeline* code is still manual |
# | **2: CI/CD for the pipeline** | pipeline and serving code are tested and released automatically | triggered by data or code change | automated, with quality gates and rollback | the maintenance cost of the platform itself |
#
# Higher is not automatically better. Match the level to the stakes: a monthly internal report model at level 1 is fine; a fraud model scoring live payments should not be at level 0.
# Below, a scorer you can run against a checklist for your own team, which makes the conversation concrete.

# %%
CHECKLIST = {
    # practice: level it belongs to
    "training code lives in version control":             0,
    "training is one command, not notebook cell order":   1,
    "experiments and metrics are tracked":                1,
    "input data is validated before training":            1,
    "models are versioned in a registry":                 1,
    "serving has input validation and a health check":    1,
    "monitoring covers inputs, predictions and latency":  1,
    "tests run on every change (data, features, model)":  2,
    "a metric gate blocks a worse model from release":    2,
    "deploys are automated with a rollback path":         2,
}

def maturity(answers):
    """Return (level, missing practices). You hold level L only if EVERY practice of levels 0..L is in place."""
    level = 0
    for L in (1, 2):
        if all(answers.get(p, False) for p, lv in CHECKLIST.items() if lv <= L):
            level = L
        else:
            break
    return level, [p for p in CHECKLIST if not answers.get(p, False)]

team = {p: True for p, lv in CHECKLIST.items() if lv == 0}
team.update({"experiments and metrics are tracked": True, "models are versioned in a registry": True})
lvl, missing = maturity(team)
level1 = [p for p, lv in CHECKLIST.items() if lv == 1]
print(f"level: {lvl}  (level-1 practices in place: {sum(team.get(p, False) for p in level1)} of {len(level1)})")
print("next best steps:", missing[:3])

# %% [markdown]
# The scorer is deliberately strict: having CI/CD tooling does not make you level 2 if your level-1 basics (validated data, monitoring) are missing. Fix the lowest missing practice first.

# %% [markdown]
# ## 6. Do you need all of this?
# Honest answer: often not yet. Cost of ceremony is real. A rough decision aid:

# %%
def suggested_level(retrains_per_year, people_touching_model, decisions_are_costly, served_online):
    score = 0
    score += 1 if retrains_per_year >= 4 else 0
    score += 1 if people_touching_model >= 3 else 0
    score += 1 if decisions_are_costly else 0
    score += 1 if served_online else 0
    return {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}[score]

cases = {
    "one-off analysis for a slide":          (0, 1, False, False),
    "weekly batch scoring, 2 analysts":      (12, 2, False, False),
    "live fraud scoring, 6-person team":     (12, 6, True, True),
}
for name, args in cases.items():
    print(f"{name:40s} -> level {suggested_level(*args)}")

# %% [markdown]
# ## Common mistakes
#
# 1. **Treating deployment as the finish line.** A model is a living thing: data changes, the world changes, and accuracy decays. Plan the monitoring before the launch.
# 2. **Two code paths for features** (one in the notebook, one rewritten in the service). Every difference is a skew bug waiting to happen. Put the transformation inside the model pipeline and ship them together.
# 3. **Not seeding** (or seeding only one of numpy, sklearn, framework RNGs), then chasing "improvements" that are noise.
# 4. **Automating before understanding.** Wrapping a pipeline nobody can explain in Airflow gives you a scheduled mystery.
# 5. **Tool shopping first.** Pick the practice (tracking, registry, CI), then the simplest tool that provides it. MLflow and a CI runner cover a lot.
# 6. **Judging a model only by offline metrics.** Offline AUC says nothing about latency, input validation, or whether the business outcome moved.

# %% [markdown]
# ## Interview corner
#
# **1. What is MLOps, and how does it differ from DevOps?**
# <details><summary>Answer</summary>
# MLOps applies DevOps ideas (automation, CI/CD, monitoring) to systems whose behaviour depends on <em>data</em> as well as code. Extra concerns: versioning data and models, reproducible training, experiment tracking, evaluation gates, training-serving skew, and monitoring for drift and decay that code tests cannot catch.
# </details>
#
# **2. What is training-serving skew and how do you prevent it?**
# <details><summary>Answer</summary>
# A mismatch between how features are computed at training and at serving. Prevent it by sharing one implementation (the preprocessing lives inside the serialized pipeline or a shared feature library), validating request schemas, logging served features, and monitoring input distributions against training data.
# </details>
#
# **3. A model's accuracy dropped two months after launch. Where do you look?**
# <details><summary>Answer</summary>
# First, is it a data problem (schema change, null spike, a broken upstream) or a real change in the world (drift)? Compare recent input distributions with training, check pipeline and feature health, then look at label delay and whether labels themselves changed definition. Retrain only after you understand which it is, otherwise you bake the bug into a new model.
# </details>
#
# **4. How would you make a training run reproducible?**
# <details><summary>Answer</summary>
# Pin data by content hash or version, code by commit, parameters and seeds in config, and the environment by locked dependencies or a container image. Log all of that with the run, and verify by re-running and comparing metrics.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: What changed between two runs?
# Write `manifest_diff(a, b)` that takes two manifests like `run_manifest` returns and gives back a dict `{key: (value_in_a, value_in_b)}` for every **top-level key whose value differs** (including keys present in only one of them; use `None` for the missing side).

# %% tags=["starter"]
def manifest_diff(a, b):
    ...

# %%
m_a = run_manifest(CSV, {"model": "logreg", "C": 1.0}, seed=42, metrics={"auc": 0.8})
m_b = {**m_a, "seed": 7, "params": {"model": "logreg", "C": 0.1}, "extra": "x"}
d = manifest_diff(m_a, m_b)
assert set(d) == {"seed", "params", "extra"}, f"expected exactly seed, params, extra; got {sorted(d)}"
assert d["seed"] == (42, 7) and d["extra"] == (None, "x")
assert manifest_diff(m_a, dict(m_a)) == {}, "identical manifests have an empty diff"
print("Exercise 1 passed:", sorted(d))

# %% tags=["solution"]
def manifest_diff(a, b):
    keys = set(a) | set(b)
    return {k: (a.get(k), b.get(k)) for k in sorted(keys) if a.get(k) != b.get(k)}

# %% [markdown]
# ### Exercise 2: Catch the skew without labels
# Write `skew_report(train_df, serve_df, cat_cols)` returning `{column: share_of_serve_rows_with_unseen_category}` only for columns where that share is **greater than 0**.
# Use it on `clean(train_raw)` versus the buggy `serving_df` and store the result in `report`. It should flag `contract` and nothing else.
# *Hint: you already have `share_unseen` above.*

# %% tags=["starter"]
def skew_report(train_df, serve_df, cat_cols):
    ...

report = ...

# %%
assert set(report) == {"contract"}, f"only `contract` should be flagged, got {sorted(report)}"
expected = float((~serving_df["contract"].isin(set(clean(train_raw)["contract"]))).mean())
assert abs(report["contract"] - expected) < 1e-12
assert skew_report(clean(train_raw), clean(test_raw), CAT) == {}, "after cleaning, nothing should be flagged"
print(f"Exercise 2 passed: {report['contract']:.1%} of serving rows carry an unseen `contract` value")

# %% tags=["solution"]
def skew_report(train_df, serve_df, cat_cols):
    out = {}
    for c in cat_cols:
        s = share_unseen(train_df[c], serve_df[c])
        if s > 0:
            out[c] = s
    return out

report = skew_report(clean(train_raw), serving_df, CAT)

# %% [markdown]
# ### Exercise 3: Put a team on the map
# A team answers "yes" to: version control, one-command training, experiment tracking, data validation, model registry. They say "no" to everything else. Use `maturity` to store their level in `team_level` and the **first missing practice** (in checklist order) in `first_gap`.
# Then answer in a markdown cell: why is the level still 0 although they have most of level 1, and what would you do first?

# %%
answers = {p: False for p in CHECKLIST}
for p in ["training code lives in version control", "training is one command, not notebook cell order", "experiments and metrics are tracked",
          "input data is validated before training", "models are versioned in a registry"]:
    answers[p] = True

# %% tags=["starter"]
team_level = ...
first_gap = ...

# %%
exp_level, exp_missing = maturity(answers)
assert team_level == exp_level == 0, "level 1 needs ALL level-1 practices; two are still missing, so the level is 0"
assert first_gap == exp_missing[0] == "serving has input validation and a health check", f"first gap should be the serving validation practice, got {first_gap!r}"
print("Exercise 3 passed:", team_level, "|", first_gap)

# %% tags=["solution"]
team_level, missing_list = maturity(answers)
first_gap = missing_list[0]

# %% [markdown]
# ### Exercise 4 (open-ended): Your own lifecycle
# Take a model you have built (or the churn model above). For each stage in the `lifecycle` dict, write one line: how is it done today (by hand, by script, automated), and what is the single most likely thing to break?
# Circle the one failure that would hurt the most and be the hardest to notice. That is where your first MLOps effort should go.

# %% [markdown]
# ## Recap
# - The ML lifecycle is a loop: data, features, train, evaluate, register, deploy, monitor, retrain. MLOps is about making each arrow repeatable and each failure loud.
# - **Training-serving skew** is silent: the unseen `contract` spellings cost accuracy without a single error. A label-free input check catches it.
# - A **run manifest** (data hash, params, seed, library versions) plus seeding makes runs comparable and reproducible; verify by re-running.
# - Maturity levels: 0 manual, 1 automated training pipeline, 2 CI/CD for the pipeline. Fix the lowest missing practice first, and match the level to the stakes.
#
# **Go deeper:** Google Cloud, "MLOps: Continuous delivery and automation pipelines in machine learning" (cloud.google.com/architecture); Sculley et al., "Hidden Technical Debt in Machine Learning Systems" (NeurIPS 2015); Chip Huyen, *Designing Machine Learning Systems* (O'Reilly).
#
# **Next:** [Experiment tracking with MLflow](02_experiment_tracking_with_mlflow.ipynb)

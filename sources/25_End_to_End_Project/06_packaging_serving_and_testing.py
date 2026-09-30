# %% [markdown]
# # Capstone 6: Packaging, Serving and Testing
#
# **Why this matters.** A model in a notebook helps nobody. The job is a *service*: something another system can call at 3 a.m., that refuses garbage,
# tells you which model version answered, and keeps working after the person who wrote it has moved on. The difference between a demo and a product is mostly
# **validation, versioning and tests**, not a fancier algorithm. Here you run the real package in a temporary folder and exercise it the way production would.
#
# **You will learn**
# - what lives where in `src/churn` and why training and serving share one data-cleaning path
# - train through the command-line interface and inspect the artifacts it writes
# - call the FastAPI service in-process with `TestClient` (good input, bad input, oversized batches)
# - derive a quality-gate floor from baselines instead of inventing it, and watch the gate fail on purpose
# - detect input drift with the PSI/KS monitor, and run the project's own test suite
#
# **Prerequisites:** [Notebook 05](05_evaluation_interpretation_and_model_card.ipynb) and the [MLOps module](../20_MLOps/).

# %%
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path("..").resolve()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.metrics import roc_auc_score

from churn import monitor, train
from churn.api import create_app
from churn.config import load_config, resolve
from churn.data import FEATURE_COLUMNS, TARGET, DataValidationError, load_dataset, load_raw, prepare_for_inference
from churn.features import build_pipeline, make_estimator
from churn.predict import ChurnModel

CFG = load_config(ROOT / "configs" / "config.yaml")
TMP = tempfile.TemporaryDirectory()
WORK = Path(TMP.name)  # everything this notebook writes goes here and is deleted at the end

# %% [markdown]
# ## 1. The package, one file per responsibility
# | module | job |
# |---|---|
# | `config.py` | load `configs/config.yaml`, merge an optional overlay (DVC's `params.yaml`), resolve paths |
# | `data.py` | `load_raw` -> `clean` -> `validate`; the **schema** is the contract shared by training, serving and monitoring |
# | `features.py` | leakage-safe sklearn `Pipeline`: everything that learns lives inside it |
# | `train.py` | split, select on validation, cost-based threshold, test once, write artifacts, release gate |
# | `predict.py` | load the artifact; same validation as training |
# | `api.py` | FastAPI app: `/health`, `/predict`, `/predict/batch` |
# | `monitor.py` | PSI and KS drift report |
# | `registry.py` | a tiny versioned model folder plus an atomic `latest.json` pointer |
#
# The first line of each module's docstring is its contract. Print them, plus the size, so you know what you are reading:

# %%
rows = []
for f in sorted((SRC / "churn").glob("*.py")):
    if f.name.startswith("__"):
        continue
    mod = importlib.import_module(f"churn.{f.stem}")
    rows.append({"module": f.name, "lines": len(f.read_text(encoding="utf-8").splitlines()),
                 "docstring": (mod.__doc__ or "").strip().splitlines()[0][:90]})
pd.DataFrame(rows).set_index("module")

# %% [markdown]
# ## 2. One data path for training and serving
# The most common production bug is **train/serve skew**: the service cleans data differently from training. Here both call `data.clean` and `data.validate`.
# The classic trap that cleaning must respect is pandas turning the text `None` into missing. Compare:

# %%
default = pd.read_csv(ROOT / "datasets" / "churn.csv")
careful = load_raw(ROOT / "datasets" / "churn.csv")
print("internet_service missing | pandas default:", int(default["internet_service"].isna().sum()),
      "| load_raw:", int(careful["internet_service"].isna().sum()))
print("'None' rows kept as a real category by load_raw:", int((careful["internet_service"] == "None").sum()))

# %% [markdown]
# Pandas' default merges roughly a thousand *no internet* customers with the genuinely unknown ones. `load_raw` keeps them apart.
# The schema in `data.py` then rejects anything malformed with **all** the problems at once:

# %%
bad = pd.DataFrame([{"tenure_months": -3, "contract": "Weekly", "internet_service": "Fiber", "monthly_charges": 50.0,
                     "total_charges": 10.0, "support_calls": 2, "paperless_billing": "Maybe", "autopay": "Yes", "senior_citizen": 0}])
try:
    prepare_for_inference(bad)  # the serving entry point: same clean + validate as training
except DataValidationError as e:
    print(e)
    assert len(e.problems) >= 3  # several problems reported together, not one at a time

# %% [markdown]
# ## 3. Train through the command line
# This is exactly what a scheduler, a Dockerfile or `dvc repro` runs. We call it in a subprocess with `PYTHONPATH=src` (the package is not installed, and we never `pip install` in this course),
# writing into the temporary folder so your repository stays clean.

# %%
MODEL_DIR = WORK / "model"
env = {**os.environ, "PYTHONPATH": str(SRC)}
run = subprocess.run([sys.executable, "-m", "churn.train", "--config", str(ROOT / "configs" / "config.yaml"),
                      "--output-dir", str(MODEL_DIR), "--fail-on-gate"], cwd=ROOT, env=env, capture_output=True, text=True)
print(run.stdout.strip())
assert run.returncode == 0, run.stderr
pd.Series({p.name: f"{p.stat().st_size / 1024:.1f} KiB" for p in sorted(MODEL_DIR.iterdir())}, name="artifact size")

# %% [markdown]
# Four files, four jobs: `model.joblib` (pipeline + threshold + version + library versions), `metrics.json` (everything the release gate and the card use),
# `model_card.md`, and `reference.csv` (the training inputs the drift monitor compares against). The model is stamped with `version+hash`; change the data,
# costs or split and the hash changes.

# %%
metrics = json.loads((MODEL_DIR / train.METRICS_FILE).read_text(encoding="utf-8"))
print("version:", metrics["model_version"], "| threshold:", metrics["threshold"]["value"], "| gate passed:", metrics["gate"]["passed"])
bundle_keys = sorted(__import__("joblib").load(MODEL_DIR / train.MODEL_FILE))
print("bundle keys:", bundle_keys)

# %% [markdown]
# ## 4. Predict in Python
# `ChurnModel` is a thin wrapper. It validates, scores, and applies the stored threshold. Note the lower-case contract: the same cleaning as training makes it work.

# %%
model = ChurnModel.load(MODEL_DIR / train.MODEL_FILE)
risky = {"customer_id": "C-1", "tenure_months": 3, "contract": "month-to-month", "internet_service": "Fiber",
         "monthly_charges": 89.9, "total_charges": 270.0, "support_calls": 5, "paperless_billing": "Yes", "autopay": "No", "senior_citizen": 0}
loyal = {**risky, "customer_id": "C-2", "tenure_months": 60, "contract": "Two year", "support_calls": 0, "autopay": "Yes", "total_charges": 5000.0}
model.predict([risky, loyal])

# %% [markdown]
# ## 5. Serve: call the API in-process
# `TestClient` runs the FastAPI app without opening a network port, so the same code path as `uvicorn churn.api:app` is tested quickly.
# The `with` block triggers the startup hook that loads the model **once**.

# %%
client = TestClient(create_app(str(MODEL_DIR / train.MODEL_FILE), max_batch_size=50))
client.__enter__()  # run the lifespan (model load); closed in the last cell
print(client.get("/health").json())
r = client.post("/predict", json=risky)
print(r.status_code, r.json())

# %% [markdown]
# Every response carries `model_version` and the `threshold`, so a wrong decision can be traced to the exact model that made it. Now the failure modes. A service earns trust by how it says no:

# %%
cases = {
    "negative tenure": {**risky, "tenure_months": -1},
    "typo in a field name": {**{k: v for k, v in risky.items() if k != "support_calls"}, "support_call": 2},
    "unknown contract level": {**risky, "contract": "Weekly"},
    "missing required field": {k: v for k, v in risky.items() if k != "contract"},
}
for name, payload in cases.items():
    resp = client.post("/predict", json=payload)
    print(f"{name:26s} -> {resp.status_code}")
    assert resp.status_code == 422, name

too_big = client.post("/predict/batch", json={"customers": [risky] * 51})
print("batch of 51 with max 50   ->", too_big.status_code, too_big.json()["detail"])
assert too_big.status_code == 413

# %% [markdown]
# `422` means "your request is malformed" (the client should fix it); `413` means "too large". Never return `200` with a made-up number for bad input: downstream systems would not notice.
#
# ### Is the service the same model as the notebook?
# Send 50 test customers through the API in one batch and compare with scoring the same rows directly. They must agree to floating-point precision; if they do not you have skew.

# %%
_, _, test_df = train.split_data(load_dataset(resolve(CFG, CFG["paths"]["data"])), CFG)
sample = test_df.head(50).copy()
payload = [{k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()}
           for row in sample[["customer_id", *FEATURE_COLUMNS]].to_dict("records")]
resp = client.post("/predict/batch", json={"customers": payload}).json()
api_p = np.array([p["churn_probability"] for p in resp["predictions"]])
direct_p = model.predict_proba(sample[FEATURE_COLUMNS])
print("max |API - direct| =", float(np.abs(api_p - direct_p).max()), "| n =", resp["n"])
assert np.allclose(api_p, direct_p)

# %% [markdown]
# ## 6. Where does the quality-gate floor come from?
# `configs/config.yaml` has `gate.min_test_auc`. A floor nobody can justify is superstition. Ours is **derived from baselines on the validation split**:
# 1. a naive one-line rule (month-to-month, plus support calls) gives the level that needs no model at all;
# 2. the logistic baseline's validation AUC, with a bootstrap interval, gives the normal range of a healthy model;
# 3. the floor is the interval's lower end, rounded down. Below it, a retrained model is statistically no better than what we already had.

# %%
train_df, val_df, _ = train.split_data(load_dataset(resolve(CFG, CFG["paths"]["data"])), CFG)
y_va = val_df[TARGET].to_numpy()
naive = val_df["contract"].map({"Month-to-month": 2, "One year": 1, "Two year": 0}) + 0.3 * val_df["support_calls"]
naive_auc = roc_auc_score(y_va, naive)
base = build_pipeline(make_estimator("logistic", CFG["models"]["logistic"], 42)).fit(train_df[FEATURE_COLUMNS], train_df[TARGET])
p_va = base.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
rng = np.random.default_rng(0)
boots = [roc_auc_score(y_va[i], p_va[i]) for i in (rng.integers(0, len(y_va), len(y_va)) for _ in range(500))]
lo, hi = np.percentile(boots, [2.5, 97.5])
derived_floor = np.floor(lo * 100) / 100
print(f"naive rule AUC {naive_auc:.3f} | logistic baseline {roc_auc_score(y_va, p_va):.3f} (95% interval {lo:.3f} to {hi:.3f})")
print(f"derived floor {derived_floor:.2f} | configured floor {CFG['gate']['min_test_auc']}")
assert naive_auc < derived_floor == CFG["gate"]["min_test_auc"], "re-derive the floor in config.yaml if the data or split changed"

# %% [markdown]
# *What just happened?* The configured floor is reproducible from data, and it sits above a model-free rule. It is deliberately *not* a promise of quality: it is a tripwire for
# "the pipeline broke" (a column shifted, labels leaked the wrong way). `tests/test_train.py` recomputes the same derivation, so the config cannot silently drift from it.
# Watch the gate do its job when the bar is unreachable, and when the model is worse than doing nothing:

# %%
ok = train.evaluate_gate(metrics, CFG)
strict = train.evaluate_gate(metrics, {**CFG, "gate": {**CFG["gate"], "min_test_auc": 0.99}})
broken = json.loads(json.dumps(metrics))
broken["test"]["cost_per_customer"] = broken["test"]["cost_flag_none_per_customer"] * 1.2  # model worse than "contact nobody"
worse = train.evaluate_gate(broken, CFG)
print("as shipped:", ok["passed"], "| floor 0.99:", strict["passed"], "| costlier than contacting nobody:", worse["passed"])
assert ok["passed"] and not strict["passed"] and not worse["passed"]

# %% [markdown]
# ## 7. Drift monitoring in one call
# `monitor.drift_report` compares new inputs to `reference.csv` with **PSI** (population stability index, a bin-by-bin distance between two distributions) and **KS** (largest gap between
# cumulative distributions). Rules of thumb: PSI below 0.1 stable, 0.1 to 0.25 moderate, above 0.25 major. First a sanity check: test customers come from the same population, so nothing should fire.

# %%
reference = load_raw(MODEL_DIR / train.REFERENCE_FILE)
same = monitor.drift_report(reference, test_df[FEATURE_COLUMNS], CFG["drift"])
print("same population ->", same.attrs["overall"])
assert same.attrs["overall"] == "ok"

shifted = test_df[FEATURE_COLUMNS].copy()
shifted["monthly_charges"] = shifted["monthly_charges"] * 1.3
shifted["tenure_months"] = (shifted["tenure_months"] * 0.6).round()
rep = monitor.drift_report(reference, shifted, CFG["drift"])
print("after a price rise and newer customers ->", rep.attrs["overall"])
rep[["feature", "psi", "ks_stat", "status"]].round(3).head(4)

# %% [markdown]
# Only the two columns we changed move, and they are ranked first. That specificity is what you want: drift tells you *where to look*, and it cannot tell you whether the model got worse
# (that needs labels, which arrive later; notebook 07 shows both cases).

# %% [markdown]
# ## 8. The test suite is part of the product
# `tests/` holds data validation, leakage, the model quality gate, the API (via `TestClient`), drift sanity, and static checks of the Docker/Airflow/DVC/CI files. Run it from here
# (`-p no:cacheprovider` keeps the repository free of a `.pytest_cache` folder):

# %%
res = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
print(res.stdout.strip().splitlines()[-1])
assert res.returncode == 0, res.stdout[-2000:]

# %% [markdown]
# Two honesty notes. First, the infrastructure tests prove those files are *well-formed and consistent with the code* (parsed YAML, AST of the DAG, pinned versions); they do not prove a Docker daemon,
# Airflow or DVC accepts them, because none of those are installed here. Second, a green suite is evidence, not proof: it cannot tell you the model works on real customers.

# %% [markdown]
# ## Common mistakes
#
# 1. **Cleaning differently at serving time.** One `clean()` and one schema for training, serving and monitoring.
# 2. **Loading the model per request.** Load once at startup; scoring is then milliseconds.
# 3. **Returning a prediction for invalid input** instead of a clear `422`.
# 4. **No version in the response.** You cannot debug a decision you cannot attribute to a model.
# 5. **Unpinned library versions.** A pickled/joblib model must be read by the scikit-learn version that wrote it; `ChurnModel.load` warns on a mismatch.
# 6. **Loading a joblib file you did not create.** It can execute code. Treat model files like executables.

# %% [markdown]
# ## Interview corner
#
# **1. What is train/serve skew and how do you prevent it?**
# <details><summary>Answer</summary>
# Serving computes features differently from training. Prevent it by sharing one code path (here `clean`, `validate` and the sklearn `Pipeline`), and by testing that the service output equals offline scoring for the same rows.
# </details>
#
# **2. Why is the model not baked into the Docker image?**
# <details><summary>Answer</summary>
# Models change more often than code. Mounting the artifact (or pulling a registry version at start-up) means a new model does not require a rebuild, and the image stays identical across environments.
# </details>
#
# **3. How do you choose a quality gate threshold?**
# <details><summary>Answer</summary>
# From baselines and noise: a naive rule sets the level that needs no model, and a bootstrap interval around the current model sets its normal range. Pick a floor that blocks broken pipelines without blocking normal variation, and re-derive it when data changes.
# </details>
#
# **4. What status codes should a prediction endpoint return for bad input?**
# <details><summary>Answer</summary>
# 422 for semantically invalid or malformed payloads, 413 for oversized bodies or batches, 503 when the model is unavailable, and 200 only for real predictions. Never a silent default.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: A new validation rule
# Customers must never have `support_calls` above 100 (already enforced) but also ask: what does the service do with `support_calls = 101`? Call `client.post("/predict", json={**risky, "support_calls": 101})`
# and store the status code in `code_101`, and the status code for `support_calls = 100` in `code_100`. Hint: the limit lives in the pydantic model in `api.py` and in `SCHEMA` in `data.py`.

# %% tags=["starter"]
code_101, code_100 = ..., ...

# %%
assert code_101 == client.post("/predict", json={**risky, "support_calls": 101}).status_code == 422
assert code_100 == client.post("/predict", json={**risky, "support_calls": 100}).status_code == 200
print("Exercise 1 passed: 101 ->", code_101, "and 100 ->", code_100)

# %% tags=["solution"]
code_101 = client.post("/predict", json={**risky, "support_calls": 101}).status_code
code_100 = client.post("/predict", json={**risky, "support_calls": 100}).status_code

# %% [markdown]
# ### Exercise 2: A leakage canary
# Build a *leaky* model on purpose to see what leakage looks like. Fit a `LogisticRegression` whose only input is the label itself (`train_df[[TARGET]]`, a perfect leak), and compute its validation AUC on `val_df[[TARGET]]`. Store it in `leaky_auc`. Store in `honest_auc` the validation AUC of the honest pipeline `base`. Then state in `suspicious` whether `leaky_auc > 0.99`.
# (Lesson: a score that is too good is a bug until proven otherwise; `tests/test_features.py` exists to catch this.)

# %% tags=["starter"]
leaky_auc, honest_auc, suspicious = ..., ..., ...

# %%
from sklearn.linear_model import LogisticRegression

_lk = LogisticRegression().fit(train_df[[TARGET]], train_df[TARGET])
_expected_leaky = roc_auc_score(y_va, _lk.predict_proba(val_df[[TARGET]])[:, 1])
assert abs(leaky_auc - _expected_leaky) < 1e-9, "fit on train[['churned']], score on val[['churned']]"
assert abs(honest_auc - roc_auc_score(y_va, base.predict_proba(val_df[FEATURE_COLUMNS])[:, 1])) < 1e-9
assert suspicious is True and honest_auc < leaky_auc
print(f"Exercise 2 passed: leaky {leaky_auc:.3f} vs honest {honest_auc:.3f}")

# %% tags=["solution"]
from sklearn.linear_model import LogisticRegression

leaky = LogisticRegression().fit(train_df[[TARGET]], train_df[TARGET])
leaky_auc = roc_auc_score(y_va, leaky.predict_proba(val_df[[TARGET]])[:, 1])
honest_auc = roc_auc_score(y_va, base.predict_proba(val_df[FEATURE_COLUMNS])[:, 1])
suspicious = bool(leaky_auc > 0.99)

# %% [markdown]
# ### Exercise 3: Drift you can see
# Starting from `test_df[FEATURE_COLUMNS]`, change **only** `support_calls` by adding 2 to every customer, then run `monitor.drift_report(reference, changed, CFG["drift"])`.
# Store the resulting report in `rep3`, and in `flagged` the **list of features whose status is not "ok"**. Hint: `rep3.loc[rep3["status"] != "ok", "feature"]`.

# %% tags=["starter"]
rep3, flagged = ..., ...

# %%
_chg = test_df[FEATURE_COLUMNS].copy()
_chg["support_calls"] = _chg["support_calls"] + 2
_exp = monitor.drift_report(reference, _chg, CFG["drift"])
assert list(flagged) == list(_exp.loc[_exp["status"] != "ok", "feature"]), "flag every feature with status warn or alert"
assert "support_calls" in list(flagged) and rep3.attrs["overall"] == "alert"
assert not {"contract", "autopay", "paperless_billing"} & set(flagged), "untouched columns must stay quiet"
print("Exercise 3 passed, flagged:", list(flagged))

# %% tags=["solution"]
changed = test_df[FEATURE_COLUMNS].copy()
changed["support_calls"] = changed["support_calls"] + 2
rep3 = monitor.drift_report(reference, changed, CFG["drift"])
flagged = list(rep3.loc[rep3["status"] != "ok", "feature"])

# %% [markdown]
# ### Exercise 4 (open-ended): Break the service on purpose
# Think of one request the tests do not cover (an empty string for `contract`, a 10,000-character `customer_id`, `total_charges` of `1e308`, unicode in `autopay`). Send it through `client`,
# record the status code, and decide whether the behaviour is right. If it is not, write the test that would catch it in `tests/test_api.py` (do not merge it; just describe it).

# %% [markdown]
# ## Recap
# - One cleaning and validation path serves training, the API and the monitor; it prevents train/serve skew.
# - The CLI writes four artifacts: model, metrics, card and a drift reference; the model carries a version and the library versions it needs.
# - The API validates strictly, returns versioned results, fails fast without a model, and says no with `422` or `413`.
# - The quality-gate floor comes from a naive rule and a bootstrap interval of the baseline, not from a hunch, and a test re-derives it.
# - Drift reports say *where to look*; the test suite proves the pieces work together, with the limits stated above.
#
# **Go deeper:** FastAPI docs on lifespan events and testing; scikit-learn "Model persistence" (including its security notes); the pytest documentation on fixtures and `tmp_path`.
#
# **Next:** [Monitoring, deployment and next steps](07_monitoring_deployment_and_next_steps.ipynb)

# %%
client.__exit__(None, None, None)
TMP.cleanup()

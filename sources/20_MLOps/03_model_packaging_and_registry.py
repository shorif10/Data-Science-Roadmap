# %% [markdown]
# # Model Packaging and the Model Registry
#
# **Why this matters.** "The model" that a service needs is not a `.pkl` file. It is the fitted weights **plus** the preprocessing code, **plus** the exact library versions, **plus** an agreement on what columns go in and what comes out.
# Lose any one and the file that worked on your laptop throws an import error in production, or worse, loads fine and silently predicts garbage. A **registry** then answers the question every incident starts with: *which model version is live, where did it come from, and how do we go back?*
#
# **You will learn**
# - save a model with `joblib` and see precisely where that approach breaks (other processes, other versions, untrusted files)
# - package a model in **MLflow's model format** with a **signature**, an input example, pinned requirements and its own code
# - enforce the signature at load time and read the **environment** the model needs
# - register versions and move **aliases** (`champion`, `challenger`) to promote and roll back, with a simple quality gate
#
# **Prerequisites:** [Experiment tracking with MLflow](02_experiment_tracking_with_mlflow.ipynb) and [Pipelines](../12_Machine_Learning/08_feature_engineering_and_pipelines.ipynb).
#
# > Data used: `datasets/churn.csv` (synthetic). All MLflow state lives in a temp folder removed at exit.

# %%
import os

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import atexit
import importlib.metadata as md
import json
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import warnings
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
DATA = Path("../datasets")
TMP = Path(tempfile.mkdtemp(prefix="mlflow_pkg_"))
atexit.register(shutil.rmtree, TMP, ignore_errors=True)

mlflow.set_tracking_uri(f"sqlite:///{(TMP / 'mlflow.db').as_posix()}")
EXP = "churn-packaging"
mlflow.create_experiment(EXP, artifact_location=(TMP / "artifacts").as_uri())
mlflow.set_experiment(EXP)
client = MlflowClient()

# %% [markdown]
# ## 1. A model that carries its own cleaning
# In the first notebook a serving bug skipped the cleaning step and quietly cost accuracy. The fix is structural: the cleaning must live **inside** the pipeline object that we ship.
# A scikit-learn `FunctionTransformer` can wrap a function, but the function has to be **importable** wherever the model is loaded. So we put it in a real module. (We write it to the temp folder, exactly as you would keep it in your project's `src/` package.)

# %%
PKG = TMP / "code"
PKG.mkdir()
(PKG / "churn_prep.py").write_text(textwrap.dedent('''
    import pandas as pd

    FEATURES = ["tenure_months", "monthly_charges", "total_charges", "support_calls",
                "contract", "internet_service", "paperless_billing", "autopay", "senior_citizen"]

    def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Raw request rows -> model-ready features. Lives in the model package, not in the caller."""
        out = df[FEATURES].copy()
        out["contract"] = out["contract"].astype("str").str.strip().str.lower()
        out["internet_service"] = out["internet_service"].fillna("unknown")
        for c in ("paperless_billing", "autopay", "senior_citizen"):
            out[c] = out[c].astype("str")
        return out
'''))
sys.path.insert(0, str(PKG))
import churn_prep                                    # noqa: E402  (importable, like a real project module)

raw = pd.read_csv(DATA / "churn.csv", keep_default_na=False, na_values=[""])
NUM = ["tenure_months", "monthly_charges", "total_charges", "support_calls"]
CAT = ["contract", "internet_service", "paperless_billing", "autopay", "senior_citizen"]
train_raw, test_raw = train_test_split(raw, test_size=0.25, stratify=raw["churned"], random_state=42)
fit_raw, val_raw = train_test_split(train_raw, test_size=0.25, stratify=train_raw["churned"], random_state=42)

def build_pipeline(C=1.0):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("clean", FunctionTransformer(churn_prep.clean_frame)), ("pre", pre),
                     ("clf", LogisticRegression(C=C, max_iter=1000, random_state=42))])

model = build_pipeline().fit(fit_raw, fit_raw["churned"])
val_auc = roc_auc_score(val_raw["churned"], model.predict_proba(val_raw)[:, 1])
print(f"the pipeline takes RAW rows; validation AUC = {val_auc:.3f}")

# %% [markdown]
# ## 2. The joblib way, and where it breaks
# `joblib.dump` is the standard way to save scikit-learn objects. It is fast and fine for a notebook. Look at what you actually get:

# %%
path = TMP / "churn_model.joblib"
joblib.dump(model, path)
print(f"file size: {path.stat().st_size / 1024:.0f} KiB")
reloaded = joblib.load(path)
same = np.allclose(model.predict_proba(val_raw)[:, 1], reloaded.predict_proba(val_raw)[:, 1])
print("reloaded model gives identical predictions:", same)

# %% [markdown]
# It works *in this process*, where `churn_prep` is already imported. The file does **not** contain that function's code; it contains a reference (`churn_prep.clean_frame`). Load it in a fresh Python process that does not have the module and see what happens:

# %%
def load_in_fresh_process(joblib_path, pythonpath=None):
    code = f"import joblib; m = joblib.load(r'{joblib_path}'); print('loaded', type(m).__name__)"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTHONPATH", None)
    if pythonpath:
        env["PYTHONPATH"] = str(pythonpath)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=TMP, timeout=120)
    return proc.returncode, (proc.stdout.strip() or proc.stderr.strip().splitlines()[-1])

print("without the module :", load_in_fresh_process(path))
print("with the module    :", load_in_fresh_process(path, PKG))

# %% [markdown]
# That is the most common deployment surprise: a model pickled from a notebook references `__main__.something` or a module that only exists on your machine. Three more limits of a bare `.joblib` file:
#
# 1. **No environment.** It does not say which scikit-learn version made it. Loading across versions may warn, break, or (rarely) change results.
# 2. **No interface.** Nothing says "expects these nine columns of these types". Callers find out by crashing.
# 3. **Unsafe by design.** `joblib`/`pickle` can run **arbitrary code while loading**. Never load a pickle from a source you do not trust. A harmless demonstration (the payload only prints):

# %%
class Harmless:
    def __reduce__(self):                     # pickle calls this to learn how to rebuild the object
        return (print, ("!!! code ran during joblib.load: this could have been anything !!!",))

joblib.dump(Harmless(), TMP / "payload.joblib")
_ = joblib.load(TMP / "payload.joblib")       # prints: the 'model' file executed code

# %% [markdown]
# A model file is an **executable**. Treat it like one: load only files you produced or can verify, store them in access-controlled locations, and prefer formats with an audit step. MLflow's scikit-learn flavor now uses `skops`, which audits the types inside a file and refuses unknown ones unless you list them as trusted. We will meet that in a second.

# %% [markdown]
# ## 3. The MLflow model format
# An MLflow model is a **directory** with a standard layout: the serialized model, an `MLmodel` description file, a pinned environment, an optional input example, and (optionally) your own code. Any tool that understands the layout can load and serve it.
# Three inputs make it trustworthy:
# * **signature**: the input column names/types and the output shape, inferred from real data;
# * **pip_requirements**: the packages (and versions) needed to load it;
# * **code_paths**: your own modules, copied into the model directory.
#
# First, what happens if we forget to declare our custom function as trusted:

# %%
sample = val_raw.drop(columns=["churned"])
signature = infer_signature(sample[churn_prep.FEATURES], model.predict_proba(sample)[:, 1])
TRUSTED = ["churn_prep.clean_frame", "numpy.dtype"]          # the types we reviewed and accept (see the audit message below)
REQS = [f"scikit-learn=={sklearn.__version__}", f"pandas=={pd.__version__}", f"numpy=={np.__version__}"]

try:
    with mlflow.start_run(run_name="untrusted-attempt"):
        mlflow.sklearn.log_model(model, name="model", signature=signature, pip_requirements=REQS,
                                 code_paths=[str(PKG / "churn_prep.py")])
except Exception as e:                                    # noqa: BLE001
    print(type(e).__name__, "->", str(e).split("Root error:")[-1].strip()[:140])

# %% [markdown]
# `skops` audited the file, listed the types it does not know by default (our `churn_prep.clean_frame`, plus `numpy.dtype`, which the encoder stores), and refused. Good: the loader cannot be tricked into running a stranger's code, and we make a **conscious decision** after reading the list: our own function and numpy's dtype are fine. Declare them:

# %%
with mlflow.start_run(run_name="logreg-C1") as run:
    mlflow.log_params({"model": "logreg", "C": 1.0})
    mlflow.log_metric("val_auc", val_auc)
    info = mlflow.sklearn.log_model(
        model, name="model",
        signature=signature,                              # what goes in, what comes out
        input_example=sample[churn_prep.FEATURES].head(3),
        pip_requirements=REQS,                            # pinned environment
        code_paths=[str(PKG / "churn_prep.py")],          # our module travels with the model
        skops_trusted_types=TRUSTED,   # we audited and trust this type
        pyfunc_predict_fn="predict_proba",                # the generic interface should return probabilities, not 0/1
    )
    run1_id = run.info.run_id
print("model uri:", info.model_uri[:40] + "...")

# %% [markdown]
# Open the package and look at what it contains. The `MLmodel` file is the table of contents:

# %%
local = Path(mlflow.artifacts.download_artifacts(artifact_uri=info.model_uri, dst_path=str(TMP / "downloaded")))
print(sorted(p.name for p in local.iterdir()))
mlmodel = (local / "MLmodel").read_text()
for line in mlmodel.splitlines():
    if re.match(r"\s*(loader_module|python_version|sklearn_version|serialization_format|skops_trusted_types|mlflow_version)\b", line) or "inputs:" in line:
        print(line)

# %% [markdown]
# Notice `code/` (our module), `requirements.txt` (pinned packages) and the signature inside `MLmodel`. One artifact, one directory, everything needed to rebuild the environment and call the model.

# %% [markdown]
# ## 4. The signature is a contract you can enforce
# Load the model through MLflow's **generic `pyfunc` interface**: the same `predict(DataFrame)` call no matter which library trained the model. Because we stored a signature, MLflow validates every request against it.

# %%
pyfunc_model = mlflow.pyfunc.load_model(info.model_uri)
good = test_raw.drop(columns=["churned"]).head(5)
print("probabilities for 5 raw rows:", pyfunc_model.predict(good)[:, 1].round(3))
print("matches the original pipeline:", np.allclose(pyfunc_model.predict(good)[:, 1], model.predict_proba(good)[:, 1]))

# %%
def try_predict(df, label):
    try:
        pyfunc_model.predict(df)
        print(f"{label:28s} accepted")
    except mlflow.exceptions.MlflowException as e:
        msg = str(e).split("Error:")[-1].strip()
        print(f"{label:28s} REJECTED -> {msg[:85]}")

try_predict(good[churn_prep.FEATURES], "correct columns")
try_predict(good[churn_prep.FEATURES].drop(columns=["contract"]), "missing column")
try_predict(good[churn_prep.FEATURES].assign(tenure_months=good["tenure_months"].astype(str)), "wrong type (text tenure)")

# %% [markdown]
# *What just happened?* A caller that forgets a column or sends text for a number gets a clear error **before** the model runs, instead of a silently wrong score. This is the model-level version of the data contract idea from the pipelines module. You can also read the contract programmatically, which is how a serving layer can build its own request validation (Exercise 4):

# %%
sig = mlflow.models.get_model_info(info.model_uri).signature
print([(c.name, str(c.type)) for c in sig.inputs.inputs][:4], "...")

# %% [markdown]
# ## 5. Environment capture, and checking it
# A model trained under scikit-learn 1.9 may not load (or behave the same) under another version. The pinned `requirements.txt` is the record. Before serving, compare it with what is actually installed:

# %%
print((local / "requirements.txt").read_text())

def env_mismatches(requirements, installed):
    """requirements: lines like 'pkg==1.2.3'; installed: {name: version}. Return human-readable problems (empty list = environment matches)."""
    problems = []
    for line in requirements:
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, want = (s.strip() for s in line.split("==", 1))
        have = installed.get(name.lower())
        if have is None:
            problems.append(f"{name}: not installed (need {want})")
        elif have != want:
            problems.append(f"{name}: installed {have}, model needs {want}")
    return problems

installed = {d.metadata["Name"].lower(): d.version for d in md.distributions()}
print("mismatches here:", env_mismatches((local / "requirements.txt").read_text().splitlines(), installed))
print("against a stale environment:", env_mismatches(["scikit-learn==1.3.0", "nonexistent-pkg==1.0"], installed))

# %% [markdown]
# The real test of "self-contained" is a **fresh process with nothing but the model URI**. We start a new Python interpreter in an empty folder (no `churn_prep` on its path), load the model, and score a row:

# %%
code = (
    "import os; os.environ['MLFLOW_DISABLE_AGENT_HINT']='1'\n"
    "import mlflow, pandas as pd\n"
    f"m = mlflow.pyfunc.load_model(r'{info.model_uri}')\n"
    f"df = pd.read_csv(r'{(DATA / 'churn.csv').resolve()}', keep_default_na=False, na_values=[''], nrows=3)\n"
    "print(m.predict(df)[:, 1].round(3).tolist())\n"
)
env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
env["MLFLOW_TRACKING_URI"] = mlflow.get_tracking_uri()
empty = TMP / "empty_cwd"
empty.mkdir()
proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=empty, env=env, timeout=120)
print("fresh process exit code:", proc.returncode, "| output:", proc.stdout.strip() or proc.stderr.strip().splitlines()[-1])

# %% [markdown]
# It loads and predicts in a process that never saw our project, which is exactly what a container or a batch job will do. (MLflow copied `churn_prep.py` from `code_paths` into the model.)

# %% [markdown]
# ## 6. The registry: versions and aliases
# The registry gives a model a **name** and numbered **versions**, each linked to the run that produced it. An **alias** is a movable pointer such as `champion` ("what production serves") or `challenger` ("being evaluated").
# Consumers load `models:/churn-model@champion`; promoting is just *moving the alias*, with no code change and no redeploy of the consumers. (Older MLflow "stages" like `Staging`/`Production` are deprecated in favour of aliases and tags.)

# %%
NAME = "churn-model"
v1 = mlflow.register_model(info.model_uri, NAME)
client.set_registered_model_alias(NAME, "champion", v1.version)
client.set_model_version_tag(NAME, v1.version, "validated_by", "offline-eval")
print("registered version", v1.version, "-> champion")

# train a challenger with stronger regularisation, tracked like any run
model2 = build_pipeline(C=0.05).fit(fit_raw, fit_raw["churned"])
auc2 = roc_auc_score(val_raw["churned"], model2.predict_proba(val_raw)[:, 1])
with mlflow.start_run(run_name="logreg-C0.05") as run:
    mlflow.log_params({"model": "logreg", "C": 0.05})
    mlflow.log_metric("val_auc", auc2)
    info2 = mlflow.sklearn.log_model(model2, name="model", signature=signature, pip_requirements=REQS,
                                     code_paths=[str(PKG / "churn_prep.py")], skops_trusted_types=TRUSTED,
                                     pyfunc_predict_fn="predict_proba")
v2 = mlflow.register_model(info2.model_uri, NAME)
client.set_registered_model_alias(NAME, "challenger", v2.version)
reg = client.get_registered_model(NAME)
print("aliases now:", reg.aliases, "| challenger val_auc:", round(auc2, 4), "vs champion:", round(val_auc, 4))

# %% [markdown]
# A consumer resolves the alias at load time; it neither knows nor cares which version number that is:

# %%
champ = mlflow.pyfunc.load_model(f"models:/{NAME}@champion")
print("champion serves version:", client.get_model_version_by_alias(NAME, "champion").version)

def version_table(name):
    alias_of = {}                                                   # version -> aliases, from the registered model
    for alias, ver in client.get_registered_model(name).aliases.items():
        alias_of.setdefault(int(ver), []).append(alias)
    rows = []
    for mv in client.search_model_versions(f"name='{name}'"):
        metrics = client.get_run(mv.run_id).data.metrics
        rows.append({"version": int(mv.version), "aliases": sorted(alias_of.get(int(mv.version), [])), "val_auc": round(metrics.get("val_auc", float("nan")), 4), "run": mv.run_id[:8]})
    return pd.DataFrame(rows).sort_values("version").reset_index(drop=True)

version_table(NAME)

# %% [markdown]
# ### Promotion with a quality gate, and rollback
# Promotion should be a **rule**, not a click: move `champion` to the candidate only if it is better on the validation metric by a margin (a tiny gain is noise) - and keep the old version so you can go back in one line.

# %%
def maybe_promote(name, candidate_version, min_delta=0.002):
    """Move the 'champion' alias to candidate_version only if its val_auc beats the current champion by more than min_delta."""
    cur = client.get_model_version_by_alias(name, "champion")
    cur_auc = client.get_run(cur.run_id).data.metrics["val_auc"]
    cand_auc = client.get_run(client.get_model_version(name, str(candidate_version)).run_id).data.metrics["val_auc"]
    if cand_auc > cur_auc + min_delta:
        client.set_registered_model_alias(name, "champion", str(candidate_version))
        return True, f"promoted v{candidate_version}: {cand_auc:.4f} > {cur_auc:.4f} + {min_delta}"
    return False, f"kept v{cur.version}: candidate {cand_auc:.4f} is not better than {cur_auc:.4f} by more than {min_delta}"

print(maybe_promote(NAME, v2.version, min_delta=0.002))
print("champion is now version", client.get_model_version_by_alias(NAME, "champion").version)

# %% [markdown]
# Whether the gate fired depends on the two real numbers above (the two models are nearly identical, so a gate with a margin declining to promote is the correct, boring outcome).
# Rolling back is the same operation in reverse: set the alias to the previous version. To see it, pretend someone overrode the gate and promoted version 2, then a problem shows up:

# %%
client.set_registered_model_alias(NAME, "champion", v2.version)       # manual override: v2 goes live
print("champion:", client.get_model_version_by_alias(NAME, "champion").version)
client.set_registered_model_alias(NAME, "champion", v1.version)       # rollback: one call, no redeploy of consumers
print("after rollback, champion =", client.get_model_version_by_alias(NAME, "champion").version)
version_table(NAME)

# %% [markdown]
# That is why aliases beat overwriting a file: the old version is still there, tagged, and loadable.

# %% [markdown]
# ## 7. Serving straight from the registry (shown, not run)
# MLflow can serve a packaged model with one command, and build a container image from it. We do real serving with FastAPI in the next notebook.

# %% tags=["no-run"]
# In a terminal, with MLFLOW_TRACKING_URI pointing at your registry:
#   mlflow models serve -m "models:/churn-model@champion" --port 5001 --env-manager local
#   curl -X POST http://127.0.0.1:5001/invocations -H "Content-Type: application/json" \
#        -d '{"dataframe_split": {"columns": [...], "data": [[...]]}}'
#
#   mlflow models build-docker -m "models:/churn-model@champion" -n churn-model:1   # needs a Docker daemon

# %% [markdown]
# ## Common mistakes
#
# 1. **Shipping the estimator without its preprocessing.** Put cleaning and encoding inside the saved pipeline, or you will have two implementations that drift apart.
# 2. **Pickling functions defined in a notebook.** They are stored as `__main__.f`, which does not exist elsewhere. Keep the code in an importable module and ship it with the model (`code_paths`).
# 3. **Loading a pickle from an untrusted place.** It can execute code on load. Verify provenance; prefer audited formats.
# 4. **No signature.** Without it, a wrong-typed request is a mystery at 2 a.m. instead of a clear 400.
# 5. **Unpinned environments.** `scikit-learn` alone means "whatever is newest". Pin what you trained with, and compare at deploy time.
# 6. **Overwriting `model_latest.pkl`.** You lose rollback and provenance. Use versions and aliases.
# 7. **Promoting on a hunch.** Encode the gate (metric, margin, required checks) so the decision is reproducible and reviewable.
# 8. **Trusting the type of a version number.** In this MLflow version some registry calls return versions as `int` and others as `str`; `"2" == 2` is `False`. Convert with `int()` before comparing.

# %% [markdown]
# ## Interview corner
#
# **1. Why is pickling a model risky, and what are the alternatives?**
# <details><summary>Answer</summary>
# Pickle executes code during load, so a tampered file is remote code execution, and it is also brittle across library versions and depends on importable code. Mitigate by trusting only verified artifacts, pinning environments, and using audited or data-only formats (skops, ONNX, framework-native formats) where possible.
# </details>
#
# **2. What is a model signature and why log an input example?**
# <details><summary>Answer</summary>
# The signature is the declared schema of inputs and outputs (names, types, shapes). It lets the serving layer validate requests and lets consumers see the interface. An input example makes the expected payload concrete, enables smoke tests at deploy time, and documents the model.
# </details>
#
# **3. What does a model registry give you over a folder of model files?**
# <details><summary>Answer</summary>
# Named, versioned models with lineage (which run, data and code produced it), metadata and tags, and aliases/stages that define what is in production. Promotion and rollback become pointer moves that are auditable, and consumers load by alias without code changes.
# </details>
#
# **4. How do you promote a model safely?**
# <details><summary>Answer</summary>
# Automatically evaluate the candidate against the champion on a held-out set and required slices, require a margin, run checks on the signature and environment, optionally shadow or canary it on live traffic, then move the alias. Keep the previous version so rollback is one step.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Spot a wrong environment
# Write `stale_packages(model_uri)` that downloads the model directory (as in section 3, `mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=...)`), reads its `requirements.txt` and returns the list of problems from `env_mismatches` against the `installed` dict. Store the result for `info.model_uri` in `problems_now`. Then check whether **a modified copy of the requirements** with `scikit-learn==0.0.1` produces exactly one problem and store it in `problems_fake`.

# %% tags=["starter"]
def stale_packages(model_uri):
    ...

problems_now = ...
reqs_fake = ["scikit-learn==0.0.1", f"pandas=={pd.__version__}"]
problems_fake = ...

# %%
assert problems_now == [], f"the environment that trained the model should match its requirements, got {problems_now}"
assert len(problems_fake) == 1 and problems_fake[0].startswith("scikit-learn: installed"), problems_fake
print("Exercise 1 passed:", problems_fake)

# %% tags=["solution"]
def stale_packages(model_uri):
    model_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(TMP / "reqcheck")))
    return env_mismatches((model_dir / "requirements.txt").read_text().splitlines(), installed)

problems_now = stale_packages(info.model_uri)
reqs_fake = ["scikit-learn==0.0.1", f"pandas=={pd.__version__}"]
problems_fake = env_mismatches(reqs_fake, installed)

# %% [markdown]
# ### Exercise 2: One-step rollback
# Write `rollback(name, alias="champion")` that moves the alias to the **highest version number lower than the current one**, returns the new version number as an int, and raises `ValueError` if there is no earlier version.

# %%
def make_versions(name, n):
    """Register n versions of the packaged model under `name` and point 'champion' at the last one."""
    for _ in range(n):
        mlflow.register_model(info.model_uri, name)
    client.set_registered_model_alias(name, "champion", str(n))

make_versions("rb-demo", 3)
make_versions("rb-single", 1)

# %% tags=["starter"]
def rollback(name, alias="champion"):
    ...

# %%
assert rollback("rb-demo") == 2 and int(client.get_model_version_by_alias("rb-demo", "champion").version) == 2
assert rollback("rb-demo") == 1 and int(client.get_model_version_by_alias("rb-demo", "champion").version) == 1
for name in ("rb-demo", "rb-single"):
    try:
        rollback(name)
        raise AssertionError(f"rollback on {name} at version 1 should raise ValueError")
    except ValueError:
        pass
print("Exercise 2 passed")

# %% tags=["solution"]
def rollback(name, alias="champion"):
    current = int(client.get_model_version_by_alias(name, alias).version)
    older = [int(mv.version) for mv in client.search_model_versions(f"name='{name}'") if int(mv.version) < current]
    if not older:
        raise ValueError(f"{name}: version {current} is the oldest, nothing to roll back to")
    target = max(older)
    client.set_registered_model_alias(name, alias, str(target))
    return target

# %% [markdown]
# ### Exercise 3: A gate that also checks the signature
# Extend the promotion rule. Write `gate(name, candidate_version, min_delta=0.002)` returning `(ok, reason)` where `ok` is True only if **both** hold: the candidate's `val_auc` beats the champion's by more than `min_delta` **and** the candidate's signature input column names are **identical** (same set) to the champion's.
# It must **not** move any alias; it only decides. (Breaking the input schema silently would break every consumer.)

# %%
def log_version(name, val_auc_value, drop_col=None):
    """Register a packaged model whose run has a chosen val_auc; optionally log a signature that lacks a column."""
    sig = infer_signature(sample[churn_prep.FEATURES].drop(columns=[drop_col]) if drop_col else sample[churn_prep.FEATURES], model.predict_proba(sample)[:, 1])
    with mlflow.start_run(run_name=f"{name}-synthetic"):
        mlflow.log_metric("val_auc", val_auc_value)
        i = mlflow.sklearn.log_model(model, name="model", signature=sig, pip_requirements=REQS, code_paths=[str(PKG / "churn_prep.py")],
                                     skops_trusted_types=TRUSTED)
    return mlflow.register_model(i.model_uri, name).version

log_version("gate-demo", 0.80)
client.set_registered_model_alias("gate-demo", "champion", "1")
v_better = log_version("gate-demo", 0.85)
v_tiny = log_version("gate-demo", 0.801)
v_break = log_version("gate-demo", 0.90, drop_col="support_calls")

# %% tags=["starter"]
def gate(name, candidate_version, min_delta=0.002):
    ...

# %%
assert gate("gate-demo", v_better)[0] is True, "clearly better and same schema should pass"
assert gate("gate-demo", v_tiny)[0] is False, "0.001 improvement is under the margin"
ok, why = gate("gate-demo", v_break)
assert ok is False and "schema" in why.lower() or "column" in why.lower(), f"better metric but changed schema must fail with a reason mentioning schema/columns, got {why!r}"
assert int(client.get_model_version_by_alias("gate-demo", "champion").version) == 1, "the gate must not move the alias"
print("Exercise 3 passed:", why)

# %% tags=["solution"]
def gate(name, candidate_version, min_delta=0.002):
    champ = client.get_model_version_by_alias(name, "champion")
    cand = client.get_model_version(name, str(candidate_version))
    champ_auc = client.get_run(champ.run_id).data.metrics["val_auc"]
    cand_auc = client.get_run(cand.run_id).data.metrics["val_auc"]
    champ_cols = set(mlflow.models.get_model_info(f"models:/{name}/{champ.version}").signature.inputs.input_names())
    cand_cols = set(mlflow.models.get_model_info(f"models:/{name}/{cand.version}").signature.inputs.input_names())
    if cand_cols != champ_cols:
        return False, f"input schema changed (columns differ: {sorted(champ_cols ^ cand_cols)})"
    if cand_auc <= champ_auc + min_delta:
        return False, f"val_auc gain {cand_auc - champ_auc:.4f} does not exceed the margin {min_delta}"
    return True, "better metric and compatible schema"

# %% [markdown]
# ### Exercise 4: Validate a request from the signature
# Write `missing_and_extra(model_uri, df)` returning `(missing, extra)` as two **sorted lists**: columns the model's signature requires that `df` lacks, and columns in `df` the signature does not know. A serving layer can use this to return a clean HTTP 422 before calling `predict`.
# *Hint: `mlflow.models.get_model_info(uri).signature.inputs.input_names()`.*

# %% tags=["starter"]
def missing_and_extra(model_uri, df):
    ...

# %%
full = test_raw.head(3)                                   # has churned + customer_id, which the model does not use
m, x = missing_and_extra(info.model_uri, full)
assert m == [] and x == sorted(set(full.columns) - set(churn_prep.FEATURES)), (m, x)
m2, x2 = missing_and_extra(info.model_uri, full[churn_prep.FEATURES].drop(columns=["tenure_months", "autopay"]))
assert m2 == ["autopay", "tenure_months"] and x2 == [], (m2, x2)
print("Exercise 4 passed:", m, x[:2], "...")

# %% tags=["solution"]
def missing_and_extra(model_uri, df):
    required = set(mlflow.models.get_model_info(model_uri).signature.inputs.input_names())
    have = set(df.columns)
    return sorted(required - have), sorted(have - required)

# %% [markdown]
# ### Exercise 5 (open-ended): Design your promotion policy
# Write, in a markdown cell, the promotion policy for a model you might own: which metric and margin, which slices must not regress (for example new vs long-tenure customers), what must be true of the signature and environment, who approves, and how long the previous version stays available for rollback.

# %% [markdown]
# ## Recap
# - A raw `.joblib` has no environment, no interface, no code, and **can execute code on load**. Fine for notebooks, risky as a deployment unit.
# - The **MLflow model format** packages the model with a **signature**, **pinned requirements**, an input example and your own code; `skops` audits the types and makes you declare what you trust.
# - The **signature** is enforced at `predict` time and readable by code, so validation can be built from it.
# - Test self-containment by loading in a **fresh process** in an empty folder.
# - The **registry** gives versions with lineage; **aliases** (`champion`, `challenger`) make promotion and rollback pointer moves. Put the promotion rule in code (metric margin plus compatibility checks).
#
# **Go deeper:** the MLflow documentation, "MLflow Models" and "Model Registry" (mlflow.org/docs/latest); the skops documentation on secure persistence (skops.readthedocs.io); the scikit-learn user guide "Model persistence".
#
# **Next:** [Serving models with FastAPI](04_serving_models_with_fastapi.ipynb)

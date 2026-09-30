# %% [markdown]
# # Capstone 7: Monitoring, Deployment and Next Steps
#
# **Why this matters.** A model starts decaying the day it ships: prices change, a competitor launches a promotion, the customer mix shifts.
# Nobody sends an error message when this happens, because the service still answers with `200` and a perfectly valid probability. Operating a model means building the
# *eyes* (monitoring), the *reflexes* (retraining and rollback rules) and the *safe way to change it* (shadow and canary releases). Then you explain the Docker, Airflow, DVC and CI files that
# automate all of it.
#
# **You will learn**
# - tell input drift from performance decay, and see a case of each that the other misses
# - write a retraining policy as code, with the reasons it gives
# - run a shadow comparison and a deterministic canary split between a champion and a candidate model
# - read what the Docker, Airflow, DVC and GitHub Actions files do, and exactly how to run them
#
# **Prerequisites:** [Notebook 06](06_packaging_serving_and_testing.ipynb).

# %%
import ast
import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path("..").resolve()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from churn import monitor, train
from churn.config import load_config, resolve
from churn.data import FEATURE_COLUMNS, TARGET, load_dataset, load_raw
from churn.features import build_pipeline, make_estimator

CFG = load_config(ROOT / "configs" / "config.yaml")
TH = CFG["drift"]
df = load_dataset(resolve(CFG, CFG["paths"]["data"]))
train_df, val_df, test_df = train.split_data(df, CFG)

TMP = tempfile.TemporaryDirectory()
metrics = train.run_training(CFG, output_dir=TMP.name)  # the champion: what "production" is running
champion = joblib.load(Path(TMP.name) / train.MODEL_FILE)
reference = load_raw(Path(TMP.name) / train.REFERENCE_FILE)
pool = pd.concat([val_df, test_df], ignore_index=True)  # 2,000 customers the model never trained on: our stand-in for live traffic
print(f"champion {champion['model_version']} ({champion['model_name']}), threshold {champion['threshold']}; pool of {len(pool)} 'live' customers")

# %% [markdown]
# ## 1. Three ways a model goes stale
# | kind | what changes | visible immediately? | how you notice |
# |---|---|---|---|
# | **Data (covariate) drift** | the distribution of the inputs `X` | yes | PSI / KS against the training reference |
# | **Label (prior) drift** | how often the outcome happens | no, labels arrive later | churn rate of recent labelled customers |
# | **Concept drift** | the relationship between `X` and `y` | no | model performance on recent labelled data |
#
# Input drift is an **early warning** and may be harmless. Concept drift **is** the damage, but you only see it once labels arrive (for churn: weeks). A monitoring setup needs both.
# We simulate ten weekly batches of 400 customers. Weeks 0 to 3 are normal, then something changes.

# %% [markdown]
# ## 2. Simulation A: input drift (a price rise)
# From week 4 every customer's `monthly_charges` rises a further 6% per week. Labels are left alone.

# %%
def weekly_batch(week, n=400):
    return pool.sample(n, random_state=100 + week).reset_index(drop=True)

def score(batch):
    return champion["pipeline"].predict_proba(batch[FEATURE_COLUMNS])[:, 1]

def scenario_a(week):
    b = weekly_batch(week)
    if week >= 4:
        b["monthly_charges"] = b["monthly_charges"] * (1 + 0.06 * (week - 3))
    return b

def summarise(batches):
    rows = []
    for w, b in enumerate(batches):
        rep = monitor.drift_report(reference, b[FEATURE_COLUMNS], TH)
        p = score(b)
        rows.append({"week": w, "psi_monthly_charges": float(rep.set_index("feature").loc["monthly_charges", "psi"]),
                     "max_psi": float(rep["psi"].max()), "drift_status": rep.attrs["overall"],
                     "auc": roc_auc_score(b[TARGET], p), "mean_pred": p.mean(), "actual_churn": b[TARGET].mean()})
    return pd.DataFrame(rows).set_index("week")

sim_a = summarise([scenario_a(w) for w in range(10)])
sim_a.round(3)

# %% [markdown]
# Drift fires (the status goes from `ok` to `alert` in the first week of the shift, and PSI keeps growing) as the price rises. Now look at the AUC column: the ranking barely moves. That is not a bug in the monitor. In this
# synthetic data the labels do not depend on price in a way the model would miss, so a shift in `monthly_charges` moves the inputs without hurting the ranking (the `mean_pred`
# column shows scores do creep up). **An alert is a reason to look, not a reason to retrain blindly.** In a real business a price rise often *does* change behaviour; only labels can tell you.

# %% [markdown]
# ## 3. Simulation B: concept drift (the world changes, the inputs do not)
# From week 4 the month-to-month customers are increasingly likely to be retained (say a competitor's promotion ended): each week a larger share of their labels is redrawn from a
# 20% churn rate. The inputs have **exactly the same distribution**, so a drift monitor on `X` sees nothing.

# %%
def scenario_b(week):
    b = weekly_batch(week)
    if week >= 4:
        rng = np.random.default_rng(500 + week)
        redraw = (b["contract"] == "Month-to-month") & (rng.random(len(b)) < min(1.0, 0.15 * (week - 3)))
        b.loc[redraw, TARGET] = (rng.random(int(redraw.sum())) < 0.20).astype(int)
    return b

sim_b = summarise([scenario_b(w) for w in range(10)])
sim_b.round(3)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, sim, title in [(axes[0], sim_a, "A: input drift (price rise)"), (axes[1], sim_b, "B: concept drift (labels change)")]:
    ax.bar(sim.index, sim["max_psi"], color="lightgray", label="max PSI over features (left)")
    ax.axhline(TH["psi_warn"], linestyle=":", color="black")
    ax.axhline(TH["psi_alert"], linestyle="--", color="black")
    ax.set_xlabel("week")
    ax.set_ylabel("PSI (dotted 0.10 warn, dashed 0.25 alert)")
    ax.set_title(title)
    ax2 = ax.twinx()
    ax2.plot(sim.index, sim["auc"], "o-", color="black", label="ROC AUC on that week's labels (right)")
    ax2.set_ylabel("ROC AUC")
    ax2.set_ylim(0.55, 0.9)
axes[0].legend(loc="upper left", fontsize=7)
axes[1].legend(loc="upper left", fontsize=7)
plt.tight_layout()
plt.show()

# %% [markdown]
# *What just happened?* Two failure modes, two different detectors. In **A** the input monitor is loud and performance is steady. In **B** the input monitor is silent while AUC and calibration
# (`mean_pred` versus `actual_churn`) degrade. A team that only watches inputs would miss B; a team that only waits for labels would be slow on everything. Also note how noisy a
# 400-customer AUC is: week-to-week wiggles of a few hundredths are normal, which is why retraining triggers below need a minimum number of labels and a margin.

# %% [markdown]
# ## 4. A retraining policy as code
# "Retrain when it feels stale" is not a policy. Encode the triggers, return the **reasons**, and unit-test the function. Ours combines four signals:
# performance (needs enough labels), calibration, *persistent* input drift, and age.

# %%
def retrain_decision(*, auc, n_labels, floor, mean_pred, actual_rate, drift_alert_weeks, days_since_train,
                     min_labels=300, calib_tolerance=0.08, drift_weeks=2, max_age_days=90):
    """Return (retrain: bool, reasons: list[str]). `auc` may be None while labels are still arriving."""
    reasons = []
    if n_labels >= min_labels:
        if auc is not None and auc < floor:
            reasons.append(f"labelled AUC {auc:.3f} below floor {floor}")
        if abs(mean_pred - actual_rate) > calib_tolerance:
            reasons.append(f"calibration gap {abs(mean_pred - actual_rate):.2f} above {calib_tolerance}")
    if drift_alert_weeks >= drift_weeks:
        reasons.append(f"input drift alert for {drift_alert_weeks} consecutive weeks")
    if days_since_train > max_age_days:
        reasons.append(f"model is {days_since_train} days old (limit {max_age_days})")
    return bool(reasons), reasons

def consecutive_alerts(sim, week):
    n = 0
    for w in range(week, -1, -1):
        if sim.loc[w, "drift_status"] != "alert":
            break
        n += 1
    return n

rows = []
for name, sim in [("A: input drift", sim_a), ("B: concept drift", sim_b)]:
    for week in (3, 6, 9):
        # for this demo a week's 400 labels are pooled with the previous two weeks' to reach min_labels
        window = range(max(0, week - 2), week + 1)
        ys = np.concatenate([(scenario_a if name.startswith("A") else scenario_b)(w)[TARGET].to_numpy() for w in window])
        ps = np.concatenate([score((scenario_a if name.startswith("A") else scenario_b)(w)) for w in window])
        retrain, why = retrain_decision(auc=roc_auc_score(ys, ps), n_labels=len(ys), floor=CFG["gate"]["min_test_auc"],
                                        mean_pred=ps.mean(), actual_rate=ys.mean(),
                                        drift_alert_weeks=consecutive_alerts(sim, week), days_since_train=30)
        rows.append({"scenario": name, "week": week, "retrain": retrain, "reasons": "; ".join(why) or "none"})
pd.DataFrame(rows)

# %% [markdown]
# Read the table as a story rather than as a proof: which signal fired first in each scenario, and was it a signal you could have acted on? Scenario A retrains on persistent drift alone although AUC stayed healthy; that is a *policy choice*, and many teams would rather have persistent drift open an investigation than trigger an automatic retrain. Two cautions that apply to any policy like this:
# - **Retraining on drifted data is not automatically a fix.** If the cause is a pipeline bug (a column now in cents), retraining bakes the bug in. Investigate first, retrain second.
# - **Every retrain must pass the same release gate** as the first model (`evaluate_gate`), and the decision to promote should be logged.
#
# The policy itself is tested like any function:

# %%
assert retrain_decision(auc=0.70, n_labels=500, floor=0.77, mean_pred=0.37, actual_rate=0.37, drift_alert_weeks=0, days_since_train=10)[0]
assert not retrain_decision(auc=0.70, n_labels=50, floor=0.77, mean_pred=0.37, actual_rate=0.37, drift_alert_weeks=0, days_since_train=10)[0], "too few labels: noise"
assert retrain_decision(auc=None, n_labels=0, floor=0.77, mean_pred=0, actual_rate=0, drift_alert_weeks=3, days_since_train=10)[0]
assert not retrain_decision(auc=0.83, n_labels=500, floor=0.77, mean_pred=0.37, actual_rate=0.36, drift_alert_weeks=1, days_since_train=10)[0]
print("policy behaves as designed on four hand-written cases")

# %% [markdown]
# ## 5. Changing the model safely: shadow, then canary
# A model that scored well offline can still surprise you in production (a missing column, different traffic). Two release patterns reduce the blast radius:
#
# - **Shadow**: the candidate scores every request **but its answers are not used**. You compare it with the champion on real traffic, at no risk.
# - **Canary**: a small, *deterministic* share of customers is served by the candidate; you watch guard metrics and either ramp up or roll back.
#
# Our candidate is the gradient-boosting model, trained on the same training rows.

# %%
candidate = build_pipeline(make_estimator("hgb", CFG["models"]["hgb"], 42)).fit(train_df[FEATURE_COLUMNS], train_df[TARGET])
p_val_c = candidate.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
cand_threshold = train.choose_threshold(val_df[TARGET], p_val_c, CFG["costs"])  # its own cost-optimal threshold, chosen on validation

live = test_df.reset_index(drop=True)
p_champ = champion["pipeline"].predict_proba(live[FEATURE_COLUMNS])[:, 1]
p_cand = candidate.predict_proba(live[FEATURE_COLUMNS])[:, 1]
flag_champ, flag_cand = p_champ >= champion["threshold"], p_cand >= cand_threshold
shadow = pd.Series({
    "score correlation": np.corrcoef(p_champ, p_cand)[0, 1],
    "mean |score difference|": np.abs(p_champ - p_cand).mean(),
    "decisions that differ": (flag_champ != flag_cand).mean(),
    "champion flag rate": flag_champ.mean(), "candidate flag rate": flag_cand.mean(),
})
shadow.round(3).to_frame("shadow comparison")

# %% [markdown]
# In shadow mode you would log these for a week and look for surprises: a flag rate that differs a lot, or disagreement concentrated in one segment. Only once **labels** arrive can you ask
# which model is *better* (cost per customer, notebook 04). Agreement on decisions says the candidate is consistent with the champion; it does not say it is an improvement.
#
# For the canary, bucketing must be **deterministic** (the same customer always sees the same model, or you cannot measure anything) and **stateless** (any server computes the same answer):

# %%
def in_canary(customer_id: str, percent: float, salt: str = "churn-v2") -> bool:
    """Stable hash bucket in [0, 10000). Customers below the cut-off are served by the candidate."""
    digest = hashlib.sha256(f"{salt}:{customer_id}".encode()).hexdigest()
    return int(digest[:8], 16) % 10_000 < percent * 100

ids = df["customer_id"].tolist()
for pct in (1, 5, 25, 100):
    share = np.mean([in_canary(i, pct) for i in ids])
    print(f"canary {pct:>3}%: observed share {share:.3%} of {len(ids):,} customers")
assert all(in_canary(i, 25) for i in ids if in_canary(i, 5)), "ramping up must never move a customer back to the champion"

# %% [markdown]
# A ramp plan is just data: stage, traffic, how long, and the condition that stops it. The guard rules below are *examples of the shape*, not numbers measured on this project; pick yours from
# the baseline noise you observed (notebook 05 and the weekly AUC above).

# %%
pd.DataFrame([
    {"stage": "shadow", "traffic served by candidate": "0%", "min duration": "1 week", "advance if": "no errors; flag rate within a few points of champion"},
    {"stage": "canary 1%", "traffic served by candidate": "1%", "min duration": "2 days", "advance if": "latency and error rate equal to champion"},
    {"stage": "canary 10%", "traffic served by candidate": "10%", "min duration": "1 week", "advance if": "no drift alerts; enough labels to compare cost"},
    {"stage": "full", "traffic served by candidate": "100%", "min duration": "-", "advance if": "keep the old version loadable for instant rollback"},
    {"stage": "rollback (any stage)", "traffic served by candidate": "0%", "min duration": "-", "advance if": "error spike, latency spike, or cost per customer worse than champion"},
]).set_index("stage")

# %% [markdown]
# ## 6. What the Docker, Airflow, DVC and CI files do
# These files automate what you ran by hand. We **cannot execute** them here (no Docker daemon, Airflow, DVC, or GitHub runner), so the honest thing to do is *parse* them and check
# they describe the same commands as the code. The tests in `tests/test_infra.py` do the same, permanently.

# %%
dag_src = (ROOT / "airflow" / "dags" / "churn_training_dag.py").read_text(encoding="utf-8")
dag_tasks = [n.name for n in ast.walk(ast.parse(dag_src)) if isinstance(n, ast.FunctionDef) and n.name != "churn_training"]
stage = yaml.safe_load((ROOT / "dvc" / "dvc.yaml").read_text(encoding="utf-8"))["stages"]["train"]
ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
compose = yaml.safe_load((ROOT / "docker" / "docker-compose.yml").read_text(encoding="utf-8"))
docker_lines = [ln for ln in (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]

print("Airflow DAG tasks, in file order:", dag_tasks)
print("DVC train stage command:        ", stage["cmd"])
print("CI steps:                       ", [s.get("name", s.get("uses")) for s in ci["jobs"]["test"]["steps"]])
print("Compose services:               ", list(compose["services"]))
print("Dockerfile base / user:         ", docker_lines[0], "/", [ln for ln in docker_lines if ln.startswith("USER")])

# %% [markdown]
# | file | what it automates | how you run it | status here |
# |---|---|---|---|
# | `docker/Dockerfile` | slim image, dependencies in their own cached layer, non-root user, health check; the model is mounted, not baked in | `docker build -f docker/Dockerfile -t churn-api:1.0.0 .` | **not built** (no Docker) |
# | `docker/docker-compose.yml` | runs the API with the trained model mounted read-only; optional `train` profile | `docker compose -f docker/docker-compose.yml up --build api` | **not run** |
# | `docker/requirements-serve.txt` | pinned serving dependencies (the versions the model was trained with) | used by the Dockerfile | versions read from the local environment |
# | `airflow/dags/churn_training_dag.py` | weekly: validate data, train, evaluate the gate, register | copy into your Airflow `dags/` folder with the package importable | **not executed** (only parsed) |
# | `dvc/dvc.yaml`, `dvc/params.yaml` | reproducible training stage; changing a tracked param or the data re-runs it | `dvc init` once, then `dvc repro dvc/dvc.yaml` | **unverified sketch** (DVC not installed) |
# | `.github/workflows/ci.yml` | lint and test on every push and pull request | automatic on GitHub | **not run on GitHub**; YAML parsed only |
#
# To run each yourself, from the project root (see the `25_End_to_End_Project/README.md` for PowerShell and Git Bash variants):
# 1. **Docker**: train locally (`python -m churn.train`), build, `docker compose ... up`, then `curl http://localhost:8000/health`. If the build fails, read the failing layer: the usual culprits are a pin that no longer exists or a package that needs a compiler.
# 2. **Airflow**: install Airflow in a *separate* environment with its constraint file (it pins hundreds of packages), make `churn` importable there, and trigger the DAG from the UI. Expect to adapt imports to your Airflow version.
# 3. **DVC**: `dvc init`, then `dvc repro`. Commit `dvc.lock` with the code so a teammate's `dvc repro` reproduces your model.
# 4. **CI**: push to GitHub; the workflow installs the package with dev extras, lints with `ruff` and runs `pytest`. The lint step is unverified because `ruff` was not available here.

# %% [markdown]
# ## 7. Next steps, in order of value
# 1. **Real data.** Everything here is synthetic. The first real-data milestone is a *time-based* split and a fresh look at leakage (columns that are only filled in after a customer leaves).
# 2. **Measure the treatment, not just the risk.** Hold out a random 5-10% of flagged customers from the retention offer. Only that comparison tells you the save rate that the cost matrix assumed.
# 3. **Labels pipeline.** Decide when a customer counts as churned, record the score given at decision time, and join labels back automatically; without it, performance monitoring is a wish.
# 4. **Fairness and privacy review** with the people who own those decisions, including whether `senior_citizen` should be an input at all.
# 5. **Promote the tiny pieces to real tools** when the team needs them: the file registry to MLflow's model registry, the drift script to a scheduled job with alerting, the CLI to an orchestrated pipeline.

# %% [markdown]
# ## Common mistakes
#
# 1. **Alert on every PSI blip.** Require persistence (consecutive windows) and a minimum sample, or people learn to ignore the pager.
# 2. **Monitoring inputs only.** Concept drift is invisible to `X`; track labelled performance and calibration as labels arrive.
# 3. **Auto-retraining with no gate.** A retrain that is worse than the current model must not ship. Always run the release gate and keep the previous version for rollback.
# 4. **Random canary assignment per request.** The same customer flips between models; you cannot measure anything. Hash a stable id.
# 5. **Trusting infrastructure files you never ran.** A file that parses is not a file that works. Say which ones are verified.

# %% [markdown]
# ## Interview corner
#
# **1. Your model's input drift alert fired. What do you do?**
# <details><summary>Answer</summary>
# Investigate before retraining: is it a pipeline change or a real shift? Check which features moved, compare recent labelled performance and calibration, and only retrain (through the gate) if performance or calibration is hurt or the shift is persistent.
# </details>
#
# **2. Difference between shadow deployment and a canary?**
# <details><summary>Answer</summary>
# Shadow runs the new model on live traffic but discards its output, so there is zero user impact; you compare outputs. A canary serves real users from the new model for a small share of traffic, so you can also measure outcomes, with an automatic rollback.
# </details>
#
# **3. Why can labels be delayed and how does that change monitoring?**
# <details><summary>Answer</summary>
# Churn, fraud and default are known only weeks or months later. Until then you rely on proxy signals (input drift, score distribution, calibration on matured cohorts) and evaluate performance on cohorts whose outcomes have arrived.
# </details>
#
# **4. How would you make retraining reproducible?**
# <details><summary>Answer</summary>
# Version the data, code and parameters together (DVC or a data lake snapshot id plus a git commit and a config), seed everything, record the library versions with the model, and log the training run in an experiment tracker.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: How big a price rise is too big?
# Multiply `monthly_charges` of `test_df[FEATURE_COLUMNS]` by each value in `multipliers = [1.00, 1.05, 1.10, ..., 1.50]` and call `monitor.drift_report(reference, changed, TH)`.
# Store in `first_alert` the smallest multiplier at which `monthly_charges` has status `"alert"` (or `None` if never). Then look at the PSI and KS columns for that row: which one fired? In a sentence, say whether you think the alert threshold is too sensitive for a column like this.

# %% tags=["starter"]
multipliers = [round(1.0 + 0.05 * k, 2) for k in range(11)]
first_alert = ...

# %%
def _status(m):
    c = test_df[FEATURE_COLUMNS].copy()
    c["monthly_charges"] = c["monthly_charges"] * m
    return monitor.drift_report(reference, c, TH).set_index("feature").loc["monthly_charges", "status"]

_statuses = {m: _status(m) for m in multipliers}
_expected = next((m for m in multipliers if _statuses[m] == "alert"), None)
assert first_alert == _expected, f"expected {_expected}"
assert first_alert is None or first_alert > 1.0, "no change must not alert"
assert first_alert is not None and _statuses[multipliers[multipliers.index(first_alert) - 1]] != "alert"
print("Exercise 1 passed: first alert at x", first_alert, "| statuses:", _statuses)

# %% tags=["solution"]
multipliers = [round(1.0 + 0.05 * k, 2) for k in range(11)]
first_alert = None
for m in multipliers:
    changed = test_df[FEATURE_COLUMNS].copy()
    changed["monthly_charges"] = changed["monthly_charges"] * m
    rep = monitor.drift_report(reference, changed, TH).set_index("feature")
    if rep.loc["monthly_charges", "status"] == "alert":
        first_alert = m
        break

# %% [markdown]
# ### Exercise 2: Your own canary bucketing
# Write `my_bucket(customer_id, percent, salt)` returning `True` when the customer belongs in the canary. Requirements: deterministic, based on a hash (not `random`), about `percent`% of
# customers for any `percent`, and nested (anyone in at 5% is also in at 25%). A different `salt` must select a different set. Use `hashlib`.

# %% tags=["starter"]
def my_bucket(customer_id, percent, salt="x"):
    ...

# %%
_ids = df["customer_id"].tolist()
_share = np.mean([my_bucket(i, 10, "s1") for i in _ids])
assert abs(_share - 0.10) < 0.02, f"10% canary should select about 10% of customers, got {_share:.3f}"
assert [my_bucket(i, 10, "s1") for i in _ids[:200]] == [my_bucket(i, 10, "s1") for i in _ids[:200]], "must be deterministic"
assert all(my_bucket(i, 30, "s1") for i in _ids if my_bucket(i, 5, "s1")), "nested: 5% subset of 30%"
assert not any(my_bucket(i, 0, "s1") for i in _ids) and all(my_bucket(i, 100, "s1") for i in _ids[:500])
_a = {i for i in _ids if my_bucket(i, 10, "s1")}
_b = {i for i in _ids if my_bucket(i, 10, "s2")}
assert _a != _b and len(_a & _b) < 0.5 * len(_a), "a different salt must pick a different set"
print(f"Exercise 2 passed: {_share:.3%} selected")

# %% tags=["solution"]
def my_bucket(customer_id, percent, salt="x"):
    h = int(hashlib.sha256(f"{salt}:{customer_id}".encode()).hexdigest()[:8], 16)
    return h % 10_000 < percent * 100

# %% [markdown]
# ### Exercise 3: Do not retrain on noise
# Write `performance_trigger(auc, n_labels, floor, min_labels=300)` that returns `True` only when there are at least `min_labels` labelled customers **and** `auc` is below `floor`;
# and `False` when `auc` is `None` or there are too few labels. Why is the label-count condition essential? (Compare: the AUC of 50 customers is very noisy. Bootstrap `roc_auc_score` on 50 random
# validation rows to see the spread, and store its standard deviation in `sd_50`; do the same for 500 rows in `sd_500`, using `np.random.default_rng(0)`, 300 resamples each, on the first 50 and first 500 rows of `val_df`.)

# %% tags=["starter"]
def performance_trigger(auc, n_labels, floor, min_labels=300):
    ...

sd_50, sd_500 = ..., ...

# %%
assert performance_trigger(0.70, 500, 0.77) is True
assert performance_trigger(0.70, 100, 0.77) is False
assert performance_trigger(0.80, 500, 0.77) is False
assert performance_trigger(None, 500, 0.77) is False
assert performance_trigger(0.70, 300, 0.77) is True, "min_labels is inclusive"

def _sd(n):
    y = val_df[TARGET].to_numpy()[:n]
    p = champion["pipeline"].predict_proba(val_df[FEATURE_COLUMNS].iloc[:n])[:, 1]
    rng = np.random.default_rng(0)
    return float(np.std([roc_auc_score(y[i], p[i]) for i in (rng.integers(0, n, n) for _ in range(300))]))

assert abs(sd_50 - _sd(50)) < 1e-9 and abs(sd_500 - _sd(500)) < 1e-9
assert sd_50 > 2 * sd_500 > 0, "50 labels should be far noisier than 500"
print(f"Exercise 3 passed: AUC std on 50 rows {sd_50:.3f} vs 500 rows {sd_500:.3f}")

# %% tags=["solution"]
def performance_trigger(auc, n_labels, floor, min_labels=300):
    return bool(auc is not None and n_labels >= min_labels and auc < floor)

def _bootstrap_sd(n):
    y = val_df[TARGET].to_numpy()[:n]
    p = champion["pipeline"].predict_proba(val_df[FEATURE_COLUMNS].iloc[:n])[:, 1]
    rng = np.random.default_rng(0)
    return float(np.std([roc_auc_score(y[i], p[i]) for i in (rng.integers(0, n, n) for _ in range(300))]))

sd_50, sd_500 = _bootstrap_sd(50), _bootstrap_sd(500)

# %% [markdown]
# ### Exercise 4 (open-ended): Write the runbook entry
# Pick one alert (input drift on `monthly_charges`, calibration gap, error-rate spike, or a failed nightly gate) and write a five-line runbook entry: who is paged, the first three things to check,
# what "safe to ignore" looks like, what action is allowed without approval (for example rollback), and how you would tell the alert was a false alarm. Then decide whether the alert
# should page a human at 3 a.m. or wait for the morning.

# %% [markdown]
# ## Recap
# - Input drift is an early, cheap warning; concept drift is the real damage and needs labels. Monitor both, and expect noise in small windows.
# - Encode retraining triggers as a tested function that returns reasons, with a minimum sample size and persistence.
# - Release changes through shadow then a deterministic, nested canary, with rollback conditions decided up front.
# - Docker, Airflow, DVC and CI files automate the same commands you ran by hand; say plainly which of them you have actually executed.
#
# **Go deeper:** the scikit-learn, FastAPI and Docker documentation; "Hidden Technical Debt in Machine Learning Systems" (Sculley et al., 2015); Google's "Rules of Machine Learning" (Zinkevich).
#
# **Next:** the project README in `25_End_to_End_Project/README.md` has the milestone checklist, self-assessment rubric and four more portfolio project ideas.

# %%
TMP.cleanup()

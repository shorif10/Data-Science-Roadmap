# %% [markdown]
# # Monitoring and Drift Detection
#
# **Why this matters.** A deployed model never raises an exception when it gets worse. The world changes (a pricing change, a new customer segment, a bug upstream), the inputs shift, and the model keeps returning confident numbers that are quietly wrong. Worse, in many businesses you learn the true outcome **months later** (did the customer actually churn?), so accuracy cannot be your only alarm.
# Monitoring is how you find out in days, not quarters: watch what the model **sees**, what it **says**, and, when labels finally arrive, how **right** it was.
#
# **You will learn**
# - tell apart **data (covariate) drift**, **prediction drift**, **concept drift** and **pipeline bugs**, and which signal can see which
# - implement **PSI**, the **Kolmogorov-Smirnov** test (numeric features) and the **chi-square** test (categoricals) from scratch or with SciPy
# - read the results honestly: statistical significance vs effect size, and why "no drift" runs still raise false alarms
# - simulate drift on the churn data and build a per-feature **drift report**
# - handle **delayed labels**: a timeline where input and score monitors fire weeks before accuracy can be measured
# - turn monitoring into a simple decision rule (investigate, retrain, roll back)
#
# **Prerequisites:** [Serving models with FastAPI](04_serving_models_with_fastapi.ipynb) (which logs served inputs and predictions) and [Hypothesis testing](../03_Statistics_and_Probability/).
#
# > Data used: `datasets/churn.csv` (synthetic). The "production" traffic and every drift below are **simulated by us** from that table, so we know the ground truth and can check whether the monitors catch it.

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
DATA = Path("../datasets")
raw = pd.read_csv(DATA / "churn.csv", keep_default_na=False, na_values=[""])
NUM = ["tenure_months", "monthly_charges", "total_charges", "support_calls"]
CAT = ["contract", "internet_service", "paperless_billing", "autopay", "senior_citizen"]

def clean(df):
    out = df.copy()
    out["contract"] = out["contract"].str.strip().str.lower()
    out["internet_service"] = out["internet_service"].fillna("unknown")
    for c in ("paperless_billing", "autopay", "senior_citizen"):
        out[c] = out[c].astype(str)
    return out

# Half the data is the "reference" (what the model was trained on); the other half will play the role of future production traffic.
ref_raw, prod_raw = train_test_split(raw, test_size=0.5, stratify=raw["churned"], random_state=42)
ref, prod = clean(ref_raw).reset_index(drop=True), clean(prod_raw).reset_index(drop=True)

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
])
model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, random_state=42))]).fit(ref[NUM + CAT], ref["churned"])
score = lambda df: model.predict_proba(df[NUM + CAT])[:, 1]
print(f"reference rows: {len(ref)} | production pool: {len(prod)} | AUC on healthy pool: {roc_auc_score(prod['churned'], score(prod)):.3f}")

# %% [markdown]
# ## 1. A vocabulary of things that go wrong
# | Name | What changed | Visible without labels? | Example |
# |---|---|---|---|
# | **Data / covariate drift** | distribution of inputs `P(X)` | yes | a marketing push brings many new, short-tenure customers |
# | **Prediction drift** | distribution of the model's outputs | yes | average predicted churn jumps from 37% to 55% |
# | **Concept drift** | the relationship `P(y given X)` | **no** | a new retention team makes high-support-call customers stop churning |
# | **Label / prior shift** | base rate of the target | only with labels | churn rises after a competitor launch |
# | **Pipeline bug** | upstream data is wrong, not the world | yes (and often the real cause) | a casing change makes `Month-to-month` unseen |
#
# The two most important consequences: (1) input and prediction monitoring are **early warnings** but cannot prove the model got worse; (2) **concept drift is invisible to input monitoring**, so you need labels eventually.
# And a practical rule: when an alarm fires, suspect the **pipeline first**, the world second.

# %% [markdown]
# ## 2. Drift statistics, one by one
# ### PSI, the Population Stability Index
# Bin a feature using the **reference** distribution (deciles), then compare how much of the current data falls in each bin:
#
# `PSI = sum over bins of (cur% - ref%) * ln(cur% / ref%)`
#
# It is zero when the distributions match and grows as they diverge. The usual rule of thumb (a convention from credit-risk scoring, not a law of nature): **below 0.1 stable, 0.1 to 0.25 moderate shift, above 0.25 major shift**. Empty bins would give `ln(0)`, so proportions are floored at a tiny epsilon.

# %%
def psi_from_props(ref_p, cur_p, eps=1e-4):
    r, c = np.clip(np.asarray(ref_p, float), eps, None), np.clip(np.asarray(cur_p, float), eps, None)
    return float(np.sum((c - r) * np.log(c / r)))

def psi(ref, cur, bins=10):
    """PSI for a numeric feature with bin edges taken from the reference deciles."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))[1:-1]      # interior cut points only
    r = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1) / len(ref)
    c = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1) / len(cur)
    return psi_from_props(r, c)

rng = np.random.default_rng(42)
a = rng.normal(0, 1, 5000)
for label, b in [("same distribution", rng.normal(0, 1, 5000)), ("mean +0.2 sd", rng.normal(0.2, 1, 5000)), ("mean +0.5 sd", rng.normal(0.5, 1, 5000)), ("mean +1.0 sd", rng.normal(1.0, 1, 5000))]:
    print(f"{label:20s} PSI = {psi(a, b):.3f}")

# %% [markdown]
# ### Kolmogorov-Smirnov test (numeric features)
# The two-sample KS test measures the **largest gap between the two empirical cumulative distribution functions**. It returns a statistic (0 to 1, the gap) and a p-value (is such a gap plausible if both samples came from the same distribution?). Treat the statistic as the effect size.
#
# ### Chi-square test (categorical features)
# Put the category counts of reference and current side by side in a 2 x k table and test whether the **proportions are independent of which sample** a row came from. A low p-value says the category mix changed.
#
# A warning that matters in practice: **with large samples, tiny and harmless changes become "significant"**. Look at the effect size (PSI, KS statistic, change in proportions), not the p-value alone. Demonstration:

# %%
big_ref = rng.normal(0, 1, 200_000)
for shift in (0.0, 0.01, 0.05):
    cur = rng.normal(shift, 1, 200_000)
    ks = stats.ks_2samp(big_ref, cur)
    print(f"true shift {shift:4.2f} sd | KS stat {ks.statistic:.4f}  p-value {ks.pvalue:.2g} | PSI {psi(big_ref, cur):.4f}")

# %% [markdown]
# With 200,000 rows a shift of **0.05 standard deviations** is wildly "significant" (tiny p-value) while PSI says "stable" (far below 0.1). Decide with an effect-size threshold (and a minimum sample size), and use the p-value as a sanity check.
# A reusable report for one reference/current pair:

# %%
def cat_psi(ref_s, cur_s):
    cats = sorted(set(ref_s.unique()) | set(cur_s.unique()))
    r = ref_s.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    c = cur_s.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return psi_from_props(r, c)

def chi2_p(ref_s, cur_s):
    cats = sorted(set(ref_s.unique()) | set(cur_s.unique()))
    table = np.array([ref_s.value_counts().reindex(cats, fill_value=0), cur_s.value_counts().reindex(cats, fill_value=0)])
    table = table[:, table.sum(axis=0) > 0]
    return float(stats.chi2_contingency(table)[1])

def level(p):
    return "major" if p >= 0.25 else "moderate" if p >= 0.1 else "ok"

def drift_report(ref, cur, num=NUM, cat=CAT):
    rows = []
    for c in num:
        ks = stats.ks_2samp(ref[c].dropna(), cur[c].dropna())
        rows.append({"feature": c, "kind": "num", "psi": psi(ref[c], cur[c]), "effect": ks.statistic, "p_value": ks.pvalue})
    for c in cat:
        rows.append({"feature": c, "kind": "cat", "psi": cat_psi(ref[c], cur[c]), "effect": np.nan, "p_value": chi2_p(ref[c], cur[c])})
    out = pd.DataFrame(rows)
    out["level"] = out["psi"].map(level)
    return out

# %% [markdown]
# ## 3. Five simulated futures
# We draw 1,000 "production" rows from the healthy pool and then inject one kind of change at a time, so we know exactly what the monitors *should* report:
#
# 1. **no drift**: a plain random sample (the control: what does a false alarm look like?);
# 2. **covariate drift**: a wave of new customers (short tenure, more month-to-month contracts);
# 3. **pricing change**: every monthly charge rises 15%;
# 4. **concept drift**: same inputs, but high-support-call customers churn much less (a retention program), so the **labels** change;
# 5. **pipeline bug**: the upstream feed stops normalising the casing of `contract`.

# %%
rng = np.random.default_rng(7)

def draw(n=1000, weights=None):
    return prod.sample(n, replace=True, weights=weights, random_state=int(rng.integers(1_000_000_000))).reset_index(drop=True)

def scenario_none():
    return draw()

def scenario_covariate():
    w = np.exp(-prod["tenure_months"] / 24) * (1 + 1.5 * (prod["contract"] == "month-to-month"))
    return draw(weights=w)

def scenario_pricing():
    df = draw()
    df["monthly_charges"] *= 1.15
    return df

def scenario_concept():
    df = draw()
    flip = (df["support_calls"] >= 3) & (df["churned"] == 1) & (rng.random(len(df)) < 0.6)
    df.loc[flip, "churned"] = 0
    return df

def scenario_bug():
    df = draw()
    df["contract"] = df["contract"].map({"month-to-month": "Month-to-month", "one year": "One year", "two year": "Two year"})   # un-normalised casing
    return df

scenarios = {"no drift": scenario_none(), "covariate (new customers)": scenario_covariate(), "pricing +15%": scenario_pricing(),
             "concept drift (labels)": scenario_concept(), "pipeline bug (casing)": scenario_bug()}

# %% [markdown]
# First, one scenario in full detail, so you can read a report:

# %%
rep = drift_report(ref, scenarios["covariate (new customers)"])
rep.round(4)

# %% [markdown]
# Read it top-down: `tenure_months` shows a major shift (PSI above 0.25) and `contract` a moderate one, exactly what we injected; `total_charges` moves a lot too (it grows with tenure), which is a *consequence* of the same change, a reminder that drifting features often travel together. The untouched features sit near zero.
# Now the summary across all five scenarios. For each: how many features are at `major` level, the PSI of the model's **score** (prediction drift), and the AUC (which needs the labels, so it is the slow signal):

# %%
def scenario_summary(name, df):
    r = drift_report(ref, df)
    return {"scenario": name, "features major": int((r["level"] == "major").sum()), "worst feature": r.sort_values("psi").iloc[-1]["feature"],
            "score PSI": round(psi(score(ref), score(df)), 3), "mean score": round(float(score(df).mean()), 3),
            "AUC (needs labels)": round(roc_auc_score(df["churned"], score(df)), 3)}

summary = pd.DataFrame([scenario_summary(n, d) for n, d in scenarios.items()])
print("reference mean score:", round(float(score(ref).mean()), 3), "| reference AUC:", round(roc_auc_score(ref["churned"], score(ref)), 3))
summary

# %% [markdown]
# What this table teaches (check each claim against the numbers above, they come from a run, not from memory):
#
# * **No drift** is the control. Watch its row: a healthy random sample should show no `major` features and a small score PSI. Some single feature can still wobble into "moderate" by chance with n = 1,000; that is the false-alarm rate you must budget for.
# * **Covariate drift** moves many features and the scores, and accuracy usually suffers too, because the model is now scoring a population it saw little of.
# * **Pricing +15%** hits `monthly_charges` hard and `total_charges` not at all, and its impact on the score is modest: a drifting feature is not automatically a harmful one. Drift is a question to investigate, not a verdict.
# * **Concept drift** leaves every input feature and the score distribution essentially untouched, yet the AUC drops: **no input monitor can see it**.
# * **The pipeline bug** lights up `contract` (and the model's behaviour, since the encoder sees unseen strings). If one categorical feature suddenly looks "completely different" overnight, suspect a data problem before a market shift.

# %% [markdown]
# The bug is also the clearest view of the chi-square test, because a categorical feature went from 3 known values to 3 unseen ones:

# %%
bug = scenarios["pipeline bug (casing)"]
print("reference contract mix :", ref["contract"].value_counts(normalize=True).round(3).to_dict())
print("production contract mix:", bug["contract"].value_counts(normalize=True).round(3).to_dict())
print("unseen values in production:", sorted(set(bug["contract"]) - set(ref["contract"])))
print(f"chi-square p-value: {chi2_p(ref['contract'], bug['contract']):.2g} | category PSI: {cat_psi(ref['contract'], bug['contract']):.1f}")

# %% [markdown]
# ## 4. The false-alarm budget
# Every monitor has a false-positive rate. Quantify yours instead of guessing: draw many **healthy** batches (no drift injected) and count how often each rule fires. Do it for two batch sizes, because small batches are noisier:

# %%
rng = np.random.default_rng(11)
N_BATCH = 200
table = {}
for n in (100, 1000):
    fires = {"any feature PSI >= 0.1": 0, "any feature PSI >= 0.25": 0, "any numeric KS p < 0.01": 0, "score PSI >= 0.1": 0}
    for _ in range(N_BATCH):
        b = prod.sample(n, replace=True, random_state=int(rng.integers(1_000_000_000)))
        r = drift_report(ref, b)
        fires["any feature PSI >= 0.1"] += bool((r["psi"] >= 0.1).any())
        fires["any feature PSI >= 0.25"] += bool((r["psi"] >= 0.25).any())
        fires["any numeric KS p < 0.01"] += bool((r.loc[r["kind"] == "num", "p_value"] < 0.01).any())
        fires["score PSI >= 0.1"] += psi(score(ref), score(b)) >= 0.1
    table[f"batch of {n}"] = {k: v / N_BATCH for k, v in fires.items()}
pd.DataFrame(table).map(lambda v: f"{v:.0%}").rename_axis(f"false alarm rate over {N_BATCH} healthy batches")

# %% [markdown]
# Read the table before you set thresholds: the same rule is far noisier on small batches (PSI with 10 bins on 100 rows is mostly sampling noise), and testing many features at once multiplies the chance that *one* of them looks odd (a multiple-comparisons effect). A rule that fires on a large share of healthy batches will page you for nothing, and people then mute the channel.
# Remedies: a higher effect-size threshold, a **minimum sample size** per batch, requiring the alarm for **k consecutive batches** (Exercise 2), and alerting on the few features and the score that actually matter instead of "any of 30 features".

# %% [markdown]
# ## 5. Delayed labels: a timeline
# Churn is only known **months after** the prediction. Here: 12 weekly batches of 500 customers; drift (covariate **and** concept) starts in **week 6**; labels for a week's predictions arrive **4 weeks later**. So in week `w` we can compute input and score monitors for batches up to `w`, but AUC only for batches up to `w - 4`.

# %%
rng = np.random.default_rng(3)
LAG, DRIFT_START, WEEKS = 4, 6, 12
w_drift = np.exp(-prod["tenure_months"] / 24) * (1 + 1.5 * (prod["contract"] == "month-to-month"))

def weekly_batch(week):
    drifted = week >= DRIFT_START
    df = prod.sample(500, replace=True, weights=w_drift if drifted else None, random_state=int(rng.integers(1_000_000_000))).reset_index(drop=True)
    if drifted:
        flip = (df["support_calls"] >= 3) & (df["churned"] == 1) & (rng.random(len(df)) < 0.6)
        df.loc[flip, "churned"] = 0
    return df

rows = []
for week in range(WEEKS):
    b = weekly_batch(week)
    rows.append({"week": week, "psi_tenure": psi(ref["tenure_months"], b["tenure_months"]), "psi_score": psi(score(ref), score(b)),
                 "auc_true": roc_auc_score(b["churned"], score(b))})
tl = pd.DataFrame(rows)
tl["auc_known_at_week"] = tl["week"] + LAG                                   # when the label for this batch arrives
tl["auc_visible_now"] = [tl.loc[tl["week"] <= w - LAG, "auc_true"].tolist()[-1] if w - LAG >= 0 else np.nan for w in tl["week"]]   # the latest AUC you could actually compute in week w
tl.round(3)

# %% [markdown]
# Column `auc_true` is what actually happened to that week's predictions; `auc_visible_now` is what you could **see** in that week, which lags by four weeks. Now define alarms from healthy-period behaviour (a threshold derived from noise, not a number pulled from the air) and compare **when each signal would have fired**:

# %%
healthy = tl[tl["week"] < DRIFT_START]
psi_thr = 0.1
auc_thr = healthy["auc_true"].mean() - 3 * healthy["auc_true"].std()
print(f"healthy AUC mean {healthy['auc_true'].mean():.3f}, sd {healthy['auc_true'].std():.3f} -> AUC alarm below {auc_thr:.3f}")

def first_week(mask):
    idx = np.flatnonzero(mask)
    return int(tl["week"].iloc[idx[0]]) if len(idx) else None

input_alarm = first_week((tl["psi_tenure"] >= psi_thr).to_numpy())
score_alarm = first_week((tl["psi_score"] >= psi_thr).to_numpy())
auc_week_true = first_week((tl["auc_true"] < auc_thr).to_numpy())
auc_alarm = None if auc_week_true is None else auc_week_true + LAG               # the AUC for that week becomes known LAG weeks later
print(f"drift starts in week {DRIFT_START}")
print(f"input monitor (PSI tenure) fires in week: {input_alarm}")
print(f"score monitor (PSI of predictions) fires in week: {score_alarm}")
print(f"AUC monitor fires in week: {auc_alarm}  (the degraded week was {auc_week_true}, the label lag is {LAG})")

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].plot(tl["week"], tl["psi_tenure"], marker="o", label="PSI of tenure")
ax[0].plot(tl["week"], tl["psi_score"], marker="s", label="PSI of score")
ax[0].axhline(psi_thr, color="gray", ls="--", label="alarm threshold")
ax[0].axvline(DRIFT_START, color="red", ls=":", label="drift starts")
ax[0].set(title="Label-free monitors (available immediately)", xlabel="week", ylabel="PSI (unitless)")
ax[0].legend(fontsize=8)
ax[1].plot(tl["week"], tl["auc_true"], marker="o", label="AUC of that week's predictions (true)")
ax[1].plot(tl["week"], tl["auc_visible_now"], marker="x", ls="--", label=f"AUC you can compute now (lag {LAG} wk)")
ax[1].axhline(auc_thr, color="gray", ls="--", label="alarm threshold")
ax[1].axvline(DRIFT_START, color="red", ls=":")
ax[1].set(title="Label-based monitor (delayed)", xlabel="week", ylabel="ROC-AUC")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# *What just happened?* The input and score monitors fired in the very week drift started; the AUC monitor could only fire `LAG` weeks later, after the labels matured. That gap is the whole argument for monitoring without labels: **the early monitors buy you weeks**, and the label-based monitor is the one that tells you whether the drift actually cost accuracy.
# Also notice the AUC sequence is noisy batch to batch (500 rows per week); the threshold came from that noise, which is why a single low week should not trigger a retrain by itself.

# %% [markdown]
# ## 6. From signals to decisions
# Monitoring is only useful if each alarm has an owner and a next step. A simple playbook:
#
# | Signal | First move | Retrain? |
# |---|---|---|
# | one categorical suddenly all-unseen, nulls spike, a feature is constant | **pipeline bug**: fix upstream, do not retrain | no |
# | input and score drift, labels not yet available | investigate cause, flag stakeholders, prepare a retraining set, consider shadow model | not yet |
# | input drift and AUC drop confirmed | retrain on recent data (keep a validation slice from the new period), compare with the champion, promote through the registry gate | yes |
# | AUC drop with **no** input drift | suspect concept drift or label definition change: talk to the business | yes, with new labels |
# | drift but AUC stable | accept, document, raise thresholds for that feature | no |
#
# A small function that encodes the table makes the policy explicit and testable:

# %%
def decide(input_drift, auc_dropped, new_category_values=False, labels_available=True):
    if new_category_values:
        return "fix pipeline (unseen categories), do not retrain yet"
    if not labels_available:
        return "investigate drift; prepare retraining data" if input_drift else "no action"
    if auc_dropped:
        return "retrain and compare with champion" if input_drift else "suspect concept drift; review labels with the business, then retrain"
    return "accept and document drift" if input_drift else "no action"

for case in [dict(input_drift=True, auc_dropped=True), dict(input_drift=False, auc_dropped=True), dict(input_drift=True, auc_dropped=False),
             dict(input_drift=True, auc_dropped=False, labels_available=False), dict(input_drift=True, auc_dropped=False, new_category_values=True)]:
    print(f"{str(case):95s} -> {decide(**case)}")

# %% [markdown]
# ## Common mistakes
#
# 1. **Monitoring only accuracy.** With delayed labels you learn about a problem months late. Add input and prediction monitors.
# 2. **Trusting p-values with big samples.** Everything is "significant" at n = 200,000. Use effect sizes (PSI, KS statistic) and minimum sample sizes.
# 3. **Not measuring the false-alarm rate.** A monitor that fires on 10% of healthy batches is noise; people will mute it. Calibrate on healthy history.
# 4. **Reference window = whatever is handy.** Compare against the training distribution (or a known-good period), not against "last week", or slow drift never triggers.
# 5. **Retraining at every alarm.** If the cause is a pipeline bug, retraining bakes the bug into a new model. Diagnose first.
# 6. **Monitoring 50 features equally.** Watch the features that matter (importance, business meaning) and the score, or multiple comparisons will bury you in false alarms.
# 7. **Forgetting to log inputs and predictions** at serving time. You cannot compute drift you did not record (see the FastAPI notebook's prediction log).
# 8. **Equating drift with harm.** Some drift is harmless; always connect it to performance when labels arrive.

# %% [markdown]
# ## Interview corner
#
# **1. Data drift vs concept drift?**
# <details><summary>Answer</summary>
# Data (covariate) drift is a change in the input distribution P(X); the model may still be right for the inputs it now sees, or may be extrapolating. Concept drift is a change in the relationship P(y given X): the same inputs now imply different outcomes, so the model is wrong even if inputs look unchanged. Input monitoring can detect the first; only labels reveal the second.
# </details>
#
# **2. How do you monitor a model when labels arrive weeks late?**
# <details><summary>Answer</summary>
# Use proxy signals available immediately: input feature drift, prediction-distribution drift, data quality checks, and business KPIs. Compute label-based metrics on matured cohorts once labels arrive and compare with expectations. Where possible obtain partial or earlier labels (a short-horizon proxy, a labelled sample).
# </details>
#
# **3. What is PSI and what do its thresholds mean?**
# <details><summary>Answer</summary>
# Population Stability Index compares the binned distribution of a variable between a reference and a current sample: sum of (cur% - ref%) times ln(cur%/ref%). A common convention reads below 0.1 as stable, 0.1 to 0.25 as a moderate shift, above 0.25 as a major shift. It is a heuristic effect size, not a hypothesis test; thresholds should be validated against your own false-alarm rates.
# </details>
#
# **4. A drift alert fires. What do you do?**
# <details><summary>Answer</summary>
# Check for a data pipeline or schema problem first (new categories, nulls, unit changes). If the data is correct, see whether the drift is large, affects important features, and whether performance (when labels are available) changed. Decide between accepting, adjusting thresholds, retraining on recent data, or rolling back, and record the decision.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: PSI on a categorical feature from raw counts
# Write `psi_counts(ref_counts, cur_counts)` taking two dicts `{category: count}` (categories may differ between them; a missing category has count 0) and returning the PSI over the union of categories, using `psi_from_props` for the formula (epsilon floor included).
# Then compute `bug_psi`: the PSI between the reference `contract` counts and the pipeline-bug batch's `contract` counts (`scenarios["pipeline bug (casing)"]`).

# %% tags=["starter"]
def psi_counts(ref_counts, cur_counts):
    ...

bug_psi = ...

# %%
a = {"x": 50, "y": 50}
b = {"x": 50, "y": 50}
assert abs(psi_counts(a, b)) < 1e-12, "identical distributions have PSI 0"
expected = (0.7 - 0.5) * np.log(0.7 / 0.5) + (0.3 - 0.5) * np.log(0.3 / 0.5)
assert abs(psi_counts({"x": 50, "y": 50}, {"x": 70, "y": 30}) - expected) < 1e-9, "check the formula and the normalisation of counts to proportions"
assert psi_counts({"x": 10}, {"y": 10}) > 5, "completely disjoint categories must give a very large PSI"
assert abs(psi_counts({"x": 1, "y": 3}, {"y": 6, "x": 2}) - psi_counts({"y": 3, "x": 1}, {"x": 2, "y": 6})) < 1e-12, "key order must not matter"
assert abs(bug_psi - cat_psi(ref["contract"], scenarios["pipeline bug (casing)"]["contract"])) < 1e-9
print(f"Exercise 1 passed: the casing bug has PSI {bug_psi:.1f}")

# %% tags=["solution"]
def psi_counts(ref_counts, cur_counts):
    cats = sorted(set(ref_counts) | set(cur_counts))
    r = np.array([ref_counts.get(c, 0) for c in cats], float)
    c = np.array([cur_counts.get(c, 0) for c in cats], float)
    return psi_from_props(r / r.sum(), c / c.sum())

bug_psi = psi_counts(ref["contract"].value_counts().to_dict(), scenarios["pipeline bug (casing)"]["contract"].value_counts().to_dict())

# %% [markdown]
# ### Exercise 2: Require k consecutive alarms
# Single-batch alarms flap. Write `consecutive_alarm(values, threshold, k)` that returns the **index at which the alarm first fires**: the first position `i` such that the last `k` values up to and including `i` are all **greater than or equal** to `threshold`. Return `None` if that never happens.
# Then use it on the timeline's `psi_score` column with `k=2` and store the week in `confirmed_week`; compare with the unconfirmed `score_alarm` (the price of confirmation is a delay of `k - 1` weeks).

# %% tags=["starter"]
def consecutive_alarm(values, threshold, k):
    ...

confirmed_week = ...

# %%
assert consecutive_alarm([0, 0.2, 0, 0.2, 0.3, 0.4], 0.2, 2) == 4, "first pair of consecutive alarms ends at index 4"
assert consecutive_alarm([0.2, 0.2], 0.2, 2) == 1, ">= threshold counts"
assert consecutive_alarm([0.3, 0, 0.3, 0], 0.2, 2) is None
assert consecutive_alarm([0.3, 0.3, 0.3], 0.2, 3) == 2
assert consecutive_alarm([], 0.1, 1) is None
assert confirmed_week == consecutive_alarm(tl["psi_score"].tolist(), psi_thr, 2)
print(f"Exercise 2 passed: unconfirmed score alarm in week {score_alarm}, confirmed (k=2) in week {confirmed_week}")

# %% tags=["solution"]
def consecutive_alarm(values, threshold, k):
    run = 0
    for i, v in enumerate(values):
        run = run + 1 if v >= threshold else 0
        if run >= k:
            return i
    return None

confirmed_week = consecutive_alarm(tl["psi_score"].tolist(), psi_thr, 2)

# %% [markdown]
# ### Exercise 3: Which features drifted, and is it a bug?
# Write `suspects(report, cur, ref)` returning a dict with two keys: `"drifted"`, the sorted list of feature names whose `level` is `major`, and `"unseen"`, a dict `{categorical_feature: sorted list of values in cur that never appear in ref}` for categoricals that have at least one unseen value.
# Use it on the **pipeline bug** batch; the diagnosis should point at `contract` in **both** lists, which is the signature of a data problem rather than a market shift.

# %% tags=["starter"]
def suspects(report, cur, ref):
    ...

bug_diagnosis = ...

# %%
bug_batch = scenarios["pipeline bug (casing)"]
bug_report = drift_report(ref, bug_batch)
assert bug_diagnosis == suspects(bug_report, bug_batch, ref)
assert "contract" in bug_diagnosis["drifted"], bug_diagnosis
assert bug_diagnosis["unseen"] == {"contract": ["Month-to-month", "One year", "Two year"]}, bug_diagnosis["unseen"]
healthy_diag = suspects(drift_report(ref, scenarios["no drift"]), scenarios["no drift"], ref)
assert healthy_diag["unseen"] == {}, "a healthy sample has no unseen categories"
print("Exercise 3 passed:", bug_diagnosis)

# %% tags=["solution"]
def suspects(report, cur, ref):
    drifted = sorted(report.loc[report["level"] == "major", "feature"])
    unseen = {}
    for c in report.loc[report["kind"] == "cat", "feature"]:
        new = sorted(set(cur[c].unique()) - set(ref[c].unique()))
        if new:
            unseen[c] = new
    return {"drifted": drifted, "unseen": unseen}

bug_batch0 = scenarios["pipeline bug (casing)"]
bug_diagnosis = suspects(drift_report(ref, bug_batch0), bug_batch0, ref)

# %% [markdown]
# ### Exercise 4 (open-ended): Design the monitoring for a real model
# Pick a model you know. In a markdown cell write: the five signals you would monitor (inputs, score, data quality, labels, business KPI), the threshold and false-alarm budget for each, how long its labels take to arrive, who gets paged for which alarm, and the first three things on the runbook when `score PSI` goes above your threshold.

# %% [markdown]
# ## Recap
# - Four things go wrong: **covariate drift** (inputs), **prediction drift** (outputs), **concept drift** (the relationship), and **pipeline bugs**. Input and score monitors are early warnings but cannot see concept drift; labels are eventually required.
# - **PSI** (effect size, convention: 0.1 / 0.25), **KS** (numeric distributions) and **chi-square** (categorical mix) are the standard tools. With big samples p-values flag trivial shifts, so use effect sizes plus minimum sample sizes.
# - Measure your **false-alarm rate** on healthy batches, require **consecutive** alarms, and watch the few features and scores that matter.
# - With **delayed labels**, the label-free monitors fired weeks before accuracy could be measured; AUC then confirms whether drift actually hurt.
# - Write the response as a policy: suspect the pipeline first (unseen categories, nulls), retrain only when it is the world, and compare any new model with the champion through the registry gate.
#
# **Go deeper:** the Evidently documentation (docs.evidentlyai.com) for production drift reports; Gama et al., "A survey on concept drift adaptation" (ACM Computing Surveys, 2014); the SciPy documentation for `ks_2samp` and `chi2_contingency`; Chip Huyen, *Designing Machine Learning Systems*, chapter on data distribution shifts and monitoring.
#
# **Next:** [CI/CD and testing ML systems](08_ci_cd_and_testing_ml_systems.ipynb)

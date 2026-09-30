# %% [markdown]
# # Capstone 5: Evaluation, Interpretation and the Model Card
#
# **Why this matters.** Until now the test set has been locked away. Opening it is a one-way door: whatever number you see
# is the number you report, and if you then "fix" the model to improve it, the number is no longer honest. This notebook
# opens the door once, then asks the questions a reviewer, a risk officer or a sceptical product manager will ask:
# *how sure are you, are the probabilities believable, who does the model serve badly, and what is it actually using?*
# The answers go into a model card that ships next to the model.
#
# **You will learn**
# - report test performance with uncertainty (bootstrap intervals), not a single number
# - check calibration and decide whether the scores can be read as probabilities
# - compare performance across contract types and senior/non-senior customers, and judge which gaps are noise
# - use permutation importance and a counterfactual probe to see what drives predictions (and what it cannot tell you)
# - assemble a model card with limitations stated plainly
#
# **Prerequisites:** [Notebook 04](04_modelling_tuning_and_threshold_selection.ipynb).

# %%
import sys
import tempfile
from pathlib import Path

SRC = Path("../src").resolve()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score

from churn import train
from churn.config import load_config, resolve
from churn.data import FEATURE_COLUMNS, TARGET, load_dataset

CFG = load_config("../configs/config.yaml")
COSTS = CFG["costs"]
df = load_dataset(resolve(CFG, CFG["paths"]["data"]))
train_df, val_df, test_df = train.split_data(df, CFG)

# Run the real training code into a temp folder. It selects on validation, picks the threshold on
# validation, and evaluates on test exactly once: the same thing the CLI does.
_tmp = tempfile.TemporaryDirectory()
OUT = Path(_tmp.name)
metrics = train.run_training(CFG, output_dir=OUT)
bundle = joblib.load(OUT / train.MODEL_FILE)
pipe, THRESHOLD = bundle["pipeline"], bundle["threshold"]
print("selected:", metrics["selected_model"], "| threshold:", THRESHOLD, "| version:", metrics["model_version"])

X_te, y_te = test_df[FEATURE_COLUMNS], test_df[TARGET].to_numpy()
p_te = pipe.predict_proba(X_te)[:, 1]
assert abs(roc_auc_score(y_te, p_te) - metrics["test"]["roc_auc"]) < 1e-12  # our numbers = the package's numbers

# %% [markdown]
# ## 1. The test result, with error bars
# The package already computed the headline numbers. Read them, then ask how much they would wiggle on a different sample of 1,000 customers.

# %%
t = metrics["test"]
pd.Series({
    "ROC AUC": t["roc_auc"], "average precision": t["average_precision"], "churn rate (test)": t["positive_rate"],
    "share flagged": t["flag_rate"], "precision": t["precision"], "recall": t["recall"],
    "cost / customer (model)": t["cost_per_customer"], "cost / customer (contact nobody)": t["cost_flag_none_per_customer"],
    "cost / customer (contact everybody)": t["cost_flag_all_per_customer"],
}).round(3).to_frame("test")

# %% [markdown]
# **Bootstrap** = resample the test rows with replacement many times and recompute the metric each time; the spread of those values
# approximates how much the metric depends on which customers happened to land in the test set. It does not capture every source of
# uncertainty (for example, a different time period), only sampling noise.

# %%
def bootstrap(y, p, stat, n=500, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        out.append(stat(y[i], p[i]))
    return np.array(out)

def cost_saving(y, p):
    """Saving per customer vs contacting nobody, at the fixed pre-chosen threshold."""
    return (train.cost_per_customer(y, np.zeros(len(y), bool), COSTS) - train.cost_per_customer(y, p >= THRESHOLD, COSTS))

auc_boot = bootstrap(y_te, p_te, roc_auc_score)
sav_boot = bootstrap(y_te, p_te, cost_saving)
for name, b, point in [("ROC AUC", auc_boot, t["roc_auc"]), ("saving vs contacting nobody (per customer)", sav_boot,
                                                              t["cost_flag_none_per_customer"] - t["cost_per_customer"])]:
    lo, hi = np.percentile(b, [2.5, 97.5])
    print(f"{name}: {point:.3f}  (95% bootstrap interval {lo:.3f} to {hi:.3f})")

# %% [markdown]
# *What just happened?* The threshold was fixed **before** resampling, so the interval reflects test-set noise only.
# If the interval on the saving excludes zero, the model beats "contact nobody" on this sample with some confidence. The width tells you
# how precisely you can quote the number: report "about X, give or take Y", not three decimals.

# %% [markdown]
# ## 2. Calibration: can the scores be read as probabilities?
# A model is **calibrated** if, among customers it scores near 0.30, about 30% really churn. Ranking (AUC) can be excellent while calibration is poor, and
# the cost-based threshold *formula* from notebook 01 assumes calibration. The **Brier score** is the mean squared error of the probabilities
# (lower is better); the reference point is a model that always predicts the base rate.

# %%
base_rate = train_df[TARGET].mean()  # base rate from TRAIN only, as a deployed model would know it
brier = brier_score_loss(y_te, p_te)
brier_ref = brier_score_loss(y_te, np.full(len(y_te), base_rate))
print(f"Brier: model {brier:.4f} vs always-predict-base-rate {brier_ref:.4f}  (skill = {1 - brier / brier_ref:.1%})")

frac_pos, mean_pred = calibration_curve(y_te, p_te, n_bins=8, strategy="quantile")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot([0, 1], [0, 1], "k--", label="perfect calibration")
ax[0].plot(mean_pred, frac_pos, "o-", label="model (8 quantile bins)")
ax[0].set_xlabel("mean predicted probability in bin")
ax[0].set_ylabel("observed churn rate in bin")
ax[0].set_title("Calibration curve (test)")
ax[0].legend(fontsize=8)
ax[1].hist(p_te, bins=30, color="gray")
ax[1].axvline(THRESHOLD, color="black", linestyle="--", label=f"threshold {THRESHOLD:.2f}")
ax[1].set_xlabel("predicted P(churn)")
ax[1].set_ylabel("customers")
ax[1].set_title("Score distribution (test)")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# Read the left plot: points on the dashed line mean honest probabilities; points above it mean the model under-predicts in that range, below means it over-predicts.
# With about 1,000 rows and 8 bins each point rests on roughly 125 customers, so small wiggles are noise; look for a *systematic* bend.
# If you found one, you would recalibrate (Platt scaling / isotonic regression, fitted on validation, never test) before quoting scores as probabilities.

# %% [markdown]
# ## 3. Subgroup performance
# An average can hide a segment that the model serves badly. The package already computes per-group metrics for `contract` and `senior_citizen`
# (`train.subgroup_report`). We add two things it cannot show: **how noisy each group's AUC is**, and **calibration-in-the-large** (mean predicted vs actual churn rate).

# %%
def group_table(col):
    rows = []
    for level, part in test_df.groupby(col):
        idx = test_df.index.get_indexer(part.index)
        y, p = y_te[idx], p_te[idx]
        b = bootstrap(y, p, roc_auc_score, n=300)
        rows.append({"group": level, "n": len(y), "actual churn": y.mean(), "mean predicted": p.mean(),
                     "AUC": roc_auc_score(y, p), "AUC 2.5%": np.percentile(b, 2.5), "AUC 97.5%": np.percentile(b, 97.5),
                     "recall": ((p >= THRESHOLD) & (y == 1)).sum() / max((y == 1).sum(), 1),
                     "flag rate": (p >= THRESHOLD).mean()})
    return pd.DataFrame(rows).set_index("group").round(3)

by_contract, by_senior = group_table("contract"), group_table("senior_citizen")
by_contract

# %%
by_senior

# %% [markdown]
# Three things to read from these tables, in order:
# 1. **Base rates differ a lot by contract.** Month-to-month customers churn far more often, so they are flagged far more often. That is the model working, not bias.
# 2. **AUC inside a group is lower than overall AUC for a structural reason**: part of the overall ranking comes from separating the groups from each other.
# 3. **Intervals overlap when groups are small.** A gap smaller than the interval is not evidence of unequal treatment. Do not "fix" noise.
# 4. **Calibration by group**: compare `mean predicted` with `actual churn`. The printed gaps below show the model is least well calibrated for the smallest, lowest-risk
#    group (it over-predicts there). With a couple of hundred customers that may be partly noise, and it is worth re-checking on fresh data before acting.
#
# Also cross-check with the package's own report to make sure the notebook and the model card describe the same numbers:

# %%
gap = (by_contract["mean predicted"] - by_contract["actual churn"])
print("mean predicted minus actual churn rate, by contract:", gap.round(3).to_dict())
pkg = {r["group"]: r for r in metrics["subgroups"]["contract"]}
for level, row in by_contract.iterrows():
    assert abs(pkg[level]["roc_auc"] - row["AUC"]) < 1e-3 and pkg[level]["n"] == row["n"]
print("notebook subgroup numbers match metrics.json")

# %% [markdown]
# ## 4. What is the model using?
# **Permutation importance**: shuffle one *input* column at a time on held-out data and measure how much the score drops. A big drop means the model leans on that column.
# We permute the raw columns **before** the pipeline (so one-hot columns of the same feature move together) and use **validation** data, keeping test untouched for reporting.

# %%
X_va, y_va = val_df[FEATURE_COLUMNS], val_df[TARGET]
imp = permutation_importance(pipe, X_va, y_va, scoring="roc_auc", n_repeats=15, random_state=42, n_jobs=1)
importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "auc_drop": imp.importances_mean, "std": imp.importances_std}) \
    .sort_values("auc_drop")
plt.figure(figsize=(7, 4))
plt.barh(importance["feature"], importance["auc_drop"], xerr=importance["std"], color="gray")
plt.xlabel("drop in validation ROC AUC when the column is shuffled")
plt.ylabel("input column")
plt.title("Permutation importance (15 repeats)")
plt.tight_layout()
plt.show()
importance.sort_values("auc_drop", ascending=False).round(4).reset_index(drop=True)

# %% [markdown]
# *What just happened?* Columns at the top are what the model's ranking depends on. Two caveats that trip people up:
# - **Correlated columns can share credit.** `tenure_months`, `monthly_charges` and `total_charges` are nearly collinear (notebook 02). Shuffling one leaves the others to
#   compensate, so it can look less important than the *group* really is. Notice `total_charges` scores about zero: the model gets the same information from tenure and
#   monthly charges. Exercise 3 checks whether the group effect differs from the sum of its parts for this model.
# - **Importance is not causation.** It says what the model uses to predict, not what would happen if you changed that column for a real customer.

# %% [markdown]
# ### A counterfactual probe on a sensitive column
# `senior_citizen` is a personal attribute. Does the model use it? Take every test customer, flip the flag, and look at how the score moves. Everything else stays the same.

# %%
flipped = X_te.copy()
flipped["senior_citizen"] = 1 - flipped["senior_citizen"]
shift = pipe.predict_proba(flipped)[:, 1] - p_te
direction = np.where(X_te["senior_citizen"] == 1, -shift, shift)  # change when going non-senior -> senior
print(f"mean change in P(churn) when a customer is made a senior: {direction.mean():+.4f}; largest absolute change: {np.abs(shift).max():.4f}")
print("flag decision changes for", int(((p_te >= THRESHOLD) != (pipe.predict_proba(flipped)[:, 1] >= THRESHOLD)).sum()), "of", len(p_te), "customers")

# %% [markdown]
# A small effect is not the same as "no effect", and it does not settle the ethics question: `senior_citizen` may also act through correlated columns, and the real question
# is whether the **retention offers** are distributed fairly. Section 6 lists what we would need to check with real data.

# %% [markdown]
# ## 5. Write the model card
# A **model card** is the one-page answer to "what is this, how good is it, where does it fail". `train.render_model_card` writes the facts straight from
# `metrics.json` so the card cannot drift from the numbers. We append the extra evidence from this notebook, computed above.

# %%
def extra_evidence():
    lo, hi = np.percentile(auc_boot, [2.5, 97.5])
    slo, shi = np.percentile(sav_boot, [2.5, 97.5])
    worst = by_contract["AUC"].idxmin()
    top = importance.sort_values("auc_drop", ascending=False).head(3)["feature"].tolist()
    return f"""
## Additional evaluation (notebook 05)
* ROC AUC 95% bootstrap interval on test: {lo:.3f} to {hi:.3f}.
* Saving per customer versus contacting nobody, 95% bootstrap interval: {slo:.2f} to {shi:.2f}.
* Brier score {brier:.4f} versus {brier_ref:.4f} for always predicting the training base rate.
* Lowest within-group AUC among contract types: {worst} ({by_contract.loc[worst, 'AUC']:.3f}, n={int(by_contract.loc[worst, 'n'])}).
* Permutation importance (validation) ranks {', '.join(f'`{c}`' for c in top)} highest.
* Counterfactual probe: flipping `senior_citizen` changes P(churn) by {direction.mean():+.4f} on average.

## Ethics and misuse
* `senior_citizen` is a personal attribute and is an input to the model. Before any real deployment, decide with legal and product whether it should be,
  and audit who receives retention offers, not just who is scored.
* A high score is a reason to offer help, never a reason to degrade service, raise prices, or make a customer harder to leave.
* Customers with short histories (new customers) carry less information; the model is least trustworthy for them.
"""

card = train.render_model_card(metrics, CFG) + extra_evidence()
(OUT / "model_card_extended.md").write_text(card, encoding="utf-8")
for heading in ("Intended use", "Decision rule", "Test performance", "Performance by subgroup", "Limitations", "Ethics and misuse"):
    assert f"## {heading}" in card, heading
print(card[:1400], "...\n[card continues]")

# %% [markdown]
# ## 6. Limitations we state plainly
# - **Synthetic data.** The signal was planted by a script. Real churn is messier: the same AUC here says nothing about a real telecom.
# - **No timestamps**, so we never tested "train on the past, predict the future". A real project must, because customer behaviour drifts (notebook 07).
# - **The cost matrix is an assumption.** The threshold is only as good as the 40% save rate and the 200 churner value, and notebook 04 showed how far it moves.
# - **One test set, one seed.** The bootstrap captures sampling noise, not everything else.
# - **Prediction is not treatment effect.** The model finds who is *likely to leave*, not who is *persuadable*. Some high-risk customers will leave no matter what
#   you offer, and some would have stayed anyway. Measuring that needs a randomised holdout group in production.
# - **Fairness was only probed**, not certified: two slices and one flip test are the start of an audit.

# %% [markdown]
# ## Common mistakes
#
# 1. **Looking at test, adjusting, looking again.** Each peek leaks information; the score becomes a validation score in disguise.
# 2. **One AUC figure with no interval**, then celebrating a 0.005 difference.
# 3. **Reading importance as causation** ("support calls drive churn, so cut calls"). Importance describes the model, not the world.
# 4. **Judging subgroups by eye on tiny n.** Put an interval on it, then decide.
# 5. **Calling raw scores probabilities** without a calibration check, then feeding them into an expected-value calculation.

# %% [markdown]
# ## Interview corner
#
# **1. Your model has AUC 0.83. How do you know it is calibrated?**
# <details><summary>Answer</summary>
# You do not; AUC only measures ranking. Plot a calibration curve and compute the Brier score on held-out data. If the curve bends away from the diagonal, recalibrate
# with Platt scaling or isotonic regression fitted on validation data.
# </details>
#
# **2. Why permute inputs instead of reading logistic regression coefficients?**
# <details><summary>Answer</summary>
# Permutation importance works for any model and measures the effect on held-out performance. Coefficients depend on feature scaling and correlated features, and say nothing
# about how much the score relies on a column. Both are descriptive, neither is causal.
# </details>
#
# **3. A subgroup has lower recall. Is the model biased?**
# <details><summary>Answer</summary>
# Not necessarily. First check the group's size and an interval, then its base rate, then whether the gap remains after accounting for legitimate differences. Decide which
# fairness criterion fits the use (equal recall, equal precision, calibration within groups); they cannot all hold at once when base rates differ.
# </details>
#
# **4. Why bootstrap the test set?**
# <details><summary>Answer</summary>
# A single test split gives a single number. Resampling shows how much of that number is luck of the draw, so you quote an interval and avoid over-interpreting small differences.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: How much does a wider interval cost you?
# Using `bootstrap(y_te, p_te, roc_auc_score, n=500, seed=0)` you already have `auc_boot`. Compute `auc_ci`, the tuple `(lower, upper)` of its 2.5th and 97.5th percentiles,
# then `auc_width = upper - lower`. Then bootstrap again on only the **first 250** test rows (`y_te[:250]`, `p_te[:250]`, same settings) and store its width in `auc_width_small`.
# Hint: the small sample should be noisier.

# %% tags=["starter"]
auc_ci, auc_width, auc_width_small = ..., ..., ...

# %%
_lo, _hi = np.percentile(bootstrap(y_te, p_te, roc_auc_score, n=500, seed=0), [2.5, 97.5])
assert np.allclose(auc_ci, (_lo, _hi)) and abs(auc_width - (_hi - _lo)) < 1e-12, "use the 2.5 and 97.5 percentiles of auc_boot"
_small = np.percentile(bootstrap(y_te[:250], p_te[:250], roc_auc_score, n=500, seed=0), [2.5, 97.5])
assert abs(auc_width_small - (_small[1] - _small[0])) < 1e-12
assert auc_width_small > auc_width, "a quarter of the data should give a wider interval"
print(f"Exercise 1 passed: width {auc_width:.3f} with n={len(y_te)} vs {auc_width_small:.3f} with n=250")

# %% tags=["solution"]
auc_ci = tuple(np.percentile(auc_boot, [2.5, 97.5]))
auc_width = auc_ci[1] - auc_ci[0]
small = bootstrap(y_te[:250], p_te[:250], roc_auc_score, n=500, seed=0)
auc_width_small = np.percentile(small, 97.5) - np.percentile(small, 2.5)

# %% [markdown]
# ### Exercise 2: Where is calibration worst?
# Sort the test customers by predicted probability and split them into **5 equal-sized groups** with `np.array_split` (lowest scores first). For each group compute the mean predicted
# probability and the actual churn rate. Store the five absolute gaps in `gaps` (a list of floats, lowest-score group first) and the index of the worst group in `worst_bin`.

# %% tags=["starter"]
gaps, worst_bin = ..., ...

# %%
_order = np.argsort(p_te, kind="stable")
_expected = [abs(p_te[i].mean() - y_te[i].mean()) for i in np.array_split(_order, 5)]
assert len(gaps) == 5 and np.allclose(gaps, _expected), "split the score-sorted indices into 5 parts with np.array_split"
assert worst_bin == int(np.argmax(_expected))
print("Exercise 2 passed: gaps", np.round(gaps, 3), "worst bin:", worst_bin)

# %% tags=["solution"]
order = np.argsort(p_te, kind="stable")
gaps = [float(abs(p_te[i].mean() - y_te[i].mean())) for i in np.array_split(order, 5)]
worst_bin = int(np.argmax(gaps))

# %% [markdown]
# ### Exercise 3: Importance of a correlated group
# Permute `tenure_months`, `monthly_charges` and `total_charges` **together** (the same row shuffle for all three) on the validation set and measure the drop in AUC,
# averaged over 15 repeats with `np.random.default_rng(42)`. Store it in `group_drop`. Also store in `solo_drops` the dict of the three *individual* drops from the `importance` table.
# Is the group drop larger than the sum of the parts, or smaller? Put `True` in `group_bigger_than_sum` if larger.

# %% tags=["starter"]
group_drop, solo_drops, group_bigger_than_sum = ..., ..., ...

# %%
_cols = ["tenure_months", "monthly_charges", "total_charges"]
_rng = np.random.default_rng(42)
_base = roc_auc_score(y_va, pipe.predict_proba(X_va)[:, 1])
_drops = []
for _ in range(15):
    _perm = _rng.permutation(len(X_va))
    _X = X_va.copy()
    _X[_cols] = X_va[_cols].to_numpy()[_perm]
    _drops.append(_base - roc_auc_score(y_va, pipe.predict_proba(_X)[:, 1]))
assert abs(group_drop - np.mean(_drops)) < 1e-9, "shuffle the three columns with ONE shared permutation per repeat"
assert set(solo_drops) == set(_cols)
for _c in _cols:
    assert abs(solo_drops[_c] - importance.set_index("feature").loc[_c, "auc_drop"]) < 1e-12
assert group_bigger_than_sum == bool(group_drop > sum(solo_drops.values()))
print(f"Exercise 3 passed: group drop {group_drop:.4f} vs sum of solo drops {sum(solo_drops.values()):.4f}")

# %% tags=["solution"]
cols = ["tenure_months", "monthly_charges", "total_charges"]
rng = np.random.default_rng(42)
base = roc_auc_score(y_va, pipe.predict_proba(X_va)[:, 1])
drops = []
for _ in range(15):
    perm = rng.permutation(len(X_va))
    Xp = X_va.copy()
    Xp[cols] = X_va[cols].to_numpy()[perm]
    drops.append(base - roc_auc_score(y_va, pipe.predict_proba(Xp)[:, 1]))
group_drop = float(np.mean(drops))
solo_drops = {c: float(importance.set_index("feature").loc[c, "auc_drop"]) for c in cols}
group_bigger_than_sum = bool(group_drop > sum(solo_drops.values()))

# %% [markdown]
# ### Exercise 4 (open-ended): Write the limitation a stakeholder needs
# Pick the one finding from this notebook that would change a decision (a calibration bend, a weak subgroup, a dominant feature, a wide interval) and write two sentences
# for the model card: what the finding is, and what the retention team should do differently because of it. Then add it as a bullet to the **Limitations** section of `card`.

# %% [markdown]
# ## Recap
# - Open the test set once, and report intervals, not bare point estimates.
# - AUC is about ranking; check calibration separately before treating scores as probabilities.
# - Judge subgroup gaps against their noise and base rates, and say what you did not test.
# - Permutation importance describes the model (watch correlated columns); it is not causal.
# - The model card is generated from the same metrics file, then extended with evidence and plainly stated limitations.
#
# **Go deeper:** scikit-learn user guide sections "Probability calibration" and "Permutation feature importance"; Mitchell et al., "Model Cards for Model Reporting" (2019).
#
# **Next:** [Packaging, serving and testing](06_packaging_serving_and_testing.ipynb)

# %%
_tmp.cleanup()  # remove the temporary model folder

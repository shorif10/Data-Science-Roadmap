# %% [markdown]
# # Capstone: An End-to-End Churn Model You Can Defend
#
# **Why this matters.** A hiring manager, or your own future self, does not care that you can call `fit`. They care whether the number you report will hold up in production and whether
# you can justify every choice: the split, the baseline, the model, the threshold, the assumptions about money. This notebook takes one raw CSV to a decision, in the order a careful practitioner
# would do it, and ends with a **model card** that you fill in. Nothing here is new machinery; the point is discipline and the order of operations.
#
# **You will learn**
# - run a whole modelling project in the right order: brief, split first, EDA on training data only, baselines, pipelines, tuning, threshold, **one** final test evaluation
# - compare a tuned gradient-boosting model with a logistic baseline on identical folds and let the evidence, not habit, pick the winner
# - derive a decision threshold from explicit, challengeable business assumptions, using no test data
# - quantify uncertainty on the final number, check where the model fails by segment, and write the model card
#
# **Prerequisites:** all earlier notebooks of this module, especially 02 (leakage), 07 (cross-validation), 08 (pipelines), 09 (tuning), 12 (thresholds) and 13 (interpretation).
#
# > Data used: `datasets/churn.csv` (synthetic, 5,000 telecom customers). Some raw-data traps are real: inconsistent text casing, a category literally called `None`, and missing values.

# %% [markdown]
# ## 0. The brief (write this before any code)
#
# **Request from the retention team:** "Tell us which customers are likely to leave so we can send them an offer."
#
# | Item | Decision |
# |---|---|
# | Prediction target | `churned` = 1 (customer left) |
# | Unit / when | one row per customer, scored once a month |
# | Metric for model choice | ROC-AUC and PR-AUC on cross-validation (ranking quality), because the team will contact the top of a ranked list |
# | Metric for the decision | expected **net value** of the campaign, from the assumptions below |
# | Baselines | predict the prior (no skill); a one-feature "contract type" rule; a logistic regression |
# | Assumption: customer value | retaining a customer is worth **300** (margin over the next year) |
# | Assumption: offer success | an offer rescues a customer who would have left **30%** of the time |
# | Assumption: offer cost | **20** per offer sent (discount + handling), paid even if the customer would have stayed |
#
# These three money numbers are **invented for the course**. In a real project they come from finance and from an A/B test of the offer. Keep them in one place, because stakeholders will argue about them and the threshold depends on them.

# %%
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path("../datasets")
VALUE_RETAINED, P_SUCCESS, OFFER_COST = 300.0, 0.30, 20.0
GAIN_TP = P_SUCCESS * VALUE_RETAINED - OFFER_COST  # expected net gain when we flag a real churner
LOSS_FP = OFFER_COST  # cost when we flag someone who would have stayed
print(f"expected net gain of flagging a real churner: {GAIN_TP:+.0f}; cost of flagging a loyal customer: -{LOSS_FP:.0f}")

# %% [markdown]
# ## 1. Load the raw file and look before you touch
#
# `keep_default_na=False, na_values=[""]` because the text `None` in `internet_service` is a real category (no internet plan), not a missing value.

# %%
raw = pd.read_csv(DATA / "churn.csv", keep_default_na=False, na_values=[""])
print("shape:", raw.shape, "| duplicate customer ids:", int(raw["customer_id"].duplicated().sum()), "| churn rate:", f"{raw['churned'].mean():.1%}")
print("contract labels as stored:", raw["contract"].value_counts().to_dict())
print("missing values:", raw.isna().sum()[lambda s: s > 0].to_dict())
print("tenure == 0 rows:", int((raw["tenure_months"] == 0).sum()))

# %% [markdown]
# Three problems to handle, **inside the pipeline** so they are applied identically to future data: (1) `contract` has mixed casing (six labels for three contracts), (2) `internet_service` has a few real blanks, (3) `total_charges` is missing for some customers. Since total charges is approximately
# `tenure_months * monthly_charges` (notebook 13), a domain-informed fill is better than a blind median. Write the cleaning as a small transformer:

# %%
class Cleaner(BaseEstimator, TransformerMixin):
    """Row-wise cleaning with no fitted state, so it cannot leak anything from the training data."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["contract"] = X["contract"].str.lower().str.strip()
        X["internet_service"] = X["internet_service"].fillna("Unknown")
        X["total_charges"] = X["total_charges"].fillna(X["tenure_months"] * X["monthly_charges"])
        return X


num_cols = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "senior_citizen"]
cat_cols = ["contract", "internet_service", "paperless_billing", "autopay"]
X_full = raw.drop(columns=["customer_id", "churned"]).astype({c: float for c in num_cols})  # floats so partial-dependence plots work later
y_full = raw["churned"]

# %% [markdown]
# ## 2. Split first, and seal the test set
#
# Before any exploration that could shape a decision, set aside 25% of the customers as a **test set we will evaluate exactly once**. Everything else (exploration, tuning, threshold choice) happens on the training part. We stratify so both parts keep the churn rate.

# %%
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.25, stratify=y_full, random_state=42)
print(f"train {X_train.shape}, churn rate {y_train.mean():.3f} | test {X_test.shape}, churn rate {y_test.mean():.3f}")
print("customers in both sets:", len(set(X_train.index) & set(X_test.index)))

# %% [markdown]
# ## 3. Exploratory analysis on the training set only
#
# The aim of EDA here is to find out what to expect and what could go wrong, not to hunt for a model. Use the cleaned training data:

# %%
eda = Cleaner().transform(X_train).assign(churned=y_train)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
eda.groupby("contract")["churned"].mean().sort_values().plot(kind="barh", ax=axes[0])
axes[0].set_title("Churn rate by contract")
axes[0].set_xlabel("churn rate")
axes[0].set_ylabel("")
eda.groupby(pd.cut(eda["tenure_months"], [-1, 6, 12, 24, 48, 72]), observed=True)["churned"].mean().plot(kind="bar", ax=axes[1])
axes[1].set_title("Churn rate by tenure (months)")
axes[1].set_xlabel("tenure bucket")
axes[1].set_ylabel("churn rate")
eda.groupby("support_calls")["churned"].mean().plot(marker="o", ax=axes[2])
axes[2].set_title("Churn rate by support calls")
axes[2].set_xlabel("support calls")
axes[2].set_ylabel("churn rate")
plt.tight_layout()
plt.show()
print("correlation of the billing columns:", eda[["tenure_months", "monthly_charges", "total_charges"]].corr().round(2).loc["total_charges"].to_dict())

# %% [markdown]
# *What to take from this.* Contract type, tenure and support calls all carry a strong signal in the direction a domain expert would expect (short contracts, new customers and many support calls churn more), which is reassuring. `total_charges` is highly correlated with tenure (and is nearly
# `tenure * monthly_charges`), so expect it to add little. No leakage candidate stands out (nothing recorded *after* the churn event). We will not drop the redundant column by hand; the models will tell us whether it matters.

# %% [markdown]
# ## 4. Pipelines, baselines and cross-validation
#
# Two candidate pipelines share the cleaner, then differ only in the final step. We score everything with the same 5-fold stratified cross-validation **on the training set**, starting with baselines so there is something to beat:

# %%
def preprocessor(scale):
    steps = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([("num", Pipeline(steps), num_cols), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])


logit_pipe = Pipeline([("clean", Cleaner()), ("pre", preprocessor(True)), ("clf", LogisticRegression(max_iter=2000))])
gbm_pipe = Pipeline([("clean", Cleaner()), ("pre", preprocessor(False)), ("clf", HistGradientBoostingClassifier(random_state=42))])
contract_only = Pipeline([("clean", Cleaner()), ("pre", ColumnTransformer([("c", OneHotEncoder(), ["contract"])])), ("clf", LogisticRegression(max_iter=1000))])
prior = DummyClassifier(strategy="prior")

cv = StratifiedKFold(5, shuffle=True, random_state=42)
rows = []
for name, m in [("prior (no skill)", prior), ("contract-only rule", contract_only), ("logistic, default", logit_pipe), ("gradient boosting, defaults", gbm_pipe)]:
    roc = cross_val_score(m, X_train, y_train, cv=cv, scoring="roc_auc")
    pr = cross_val_score(m, X_train, y_train, cv=cv, scoring="average_precision")
    rows.append({"model": name, "ROC-AUC": roc.mean(), "ROC std": roc.std(), "PR-AUC": pr.mean(), "PR std": pr.std()})
baselines = pd.DataFrame(rows).set_index("model")
print(f"prevalence (PR-AUC floor): {y_train.mean():.3f}")
baselines.round(3)

# %% [markdown]
# A single categorical feature already gets more than half of the way from the no-skill 0.5 to the full logistic model, so the question for every richer model is how much it adds over that. Next, tune both candidates **fairly** (a small budget each, same folds; the test set stays sealed):

# %%
logit_search = GridSearchCV(logit_pipe, {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]}, cv=cv, scoring="roc_auc").fit(X_train, y_train)
gbm_search = RandomizedSearchCV(
    gbm_pipe,
    {
        "clf__learning_rate": loguniform(0.02, 0.3),
        "clf__max_depth": [2, 3, 4],
        "clf__max_iter": randint(50, 300),
        "clf__min_samples_leaf": randint(20, 100),
        "clf__l2_regularization": loguniform(1e-3, 10),
    },
    n_iter=12, cv=cv, scoring="roc_auc", random_state=42,
).fit(X_train, y_train)
print(f"logistic: best C = {logit_search.best_params_['clf__C']}, CV AUC {logit_search.best_score_:.4f}")
print(f"boosting: CV AUC {gbm_search.best_score_:.4f} with", {k.split('__')[1]: (round(float(v), 3) if not isinstance(v, int) else v) for k, v in gbm_search.best_params_.items()})
print("(both scores are slightly optimistic: they are the maximum over tried settings, see notebook 09)")

# %% [markdown]
# ## 5. Which model? A paired comparison, not a leaderboard
#
# The two scores above are close, and each is a maximum over several tries. A fairer comparison scores both tuned models on **identical** repeated folds and looks at the *paired differences*: if the difference per fold is consistently positive, the challenger is better; if it wobbles around zero, it is not.
# Our decision rule, written down in advance: *choose the gradient-boosting model only if its mean advantage exceeds two standard errors of the paired difference; otherwise choose the simpler logistic model.*

# %%
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
cv_logit = cross_val_score(logit_search.best_estimator_, X_train, y_train, cv=rcv, scoring="roc_auc")
cv_gbm = cross_val_score(gbm_search.best_estimator_, X_train, y_train, cv=rcv, scoring="roc_auc")
diff = cv_gbm - cv_logit
se = diff.std(ddof=1) / np.sqrt(len(diff))
print("per-fold AUC difference (boosting - logistic):", np.round(diff, 3))
print(f"mean difference {diff.mean():+.4f}, standard error {se:.4f}, rule: boosting needs > {2 * se:.4f}")
chosen_name = "gradient boosting" if diff.mean() > 2 * se else "logistic regression"
chosen = gbm_search.best_estimator_ if chosen_name == "gradient boosting" else logit_search.best_estimator_
print("chosen model:", chosen_name)

# %% [markdown]
# *What just happened?* In this run boosting is behind logistic regression on every paired fold, so the rule keeps the simpler model. The rule turned a judgement call into a procedure. (The folds overlap, so the standard error is somewhat optimistic: treat "two standard errors" as a rough bar, not a p-value.) Whatever the outcome, the conclusion is useful: a tuned boosted model has to earn its extra complexity, and
# "the simple model is as good" is a perfectly defensible result. From here on we use the chosen pipeline, still trained on the training set only.

# %% [markdown]
# ## 6. The decision threshold, from explicit costs (still no test data)
#
# A ranking is not a decision. Flagging a customer with churn probability `p` has expected net value `p * P_SUCCESS * VALUE - OFFER_COST`, so it is worth it when `p > OFFER_COST / (P_SUCCESS * VALUE)`. That formula assumes the probabilities are calibrated, so we also
# choose the threshold empirically on **out-of-fold training predictions** and check calibration. Net value at threshold `t` is `GAIN_TP * (true churners flagged) - LOSS_FP * (loyal customers flagged)`.

# %%
oof = cross_val_predict(chosen, X_train, y_train, cv=cv, method="predict_proba")[:, 1]


def net_value(y_true, p, t):
    flag = p >= t
    y_true = np.asarray(y_true)
    return GAIN_TP * np.sum(flag & (y_true == 1)) - LOSS_FP * np.sum(flag & (y_true == 0))


grid_t = np.linspace(0.05, 0.9, 86)
oof_value = np.array([net_value(y_train, oof, t) for t in grid_t])
t_formula = OFFER_COST / (P_SUCCESS * VALUE_RETAINED)
t_star = float(grid_t[oof_value.argmax()])

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].plot(grid_t, oof_value / len(y_train) * 1000)
axes[0].axvline(t_formula, color="grey", ls=":", label=f"formula {t_formula:.2f}")
axes[0].axvline(t_star, color="tab:red", ls="--", label=f"out-of-fold best {t_star:.2f}")
axes[0].set_title("Campaign net value vs threshold (out-of-fold, training set)")
axes[0].set_xlabel("decision threshold")
axes[0].set_ylabel("net value per 1,000 customers")
axes[0].legend()
frac, mean_pred = calibration_curve(y_train, oof, n_bins=8, strategy="quantile")
axes[1].plot([0, 1], [0, 1], "k:")
axes[1].plot(mean_pred, frac, "o-")
axes[1].set_title("Calibration of out-of-fold probabilities")
axes[1].set_xlabel("mean predicted probability")
axes[1].set_ylabel("observed churn rate")
plt.tight_layout()
plt.show()
print(f"threshold: formula {t_formula:.3f}, empirical {t_star:.2f} | out-of-fold Brier {brier_score_loss(y_train, oof):.3f} (a constant prior would score {brier_score_loss(y_train, np.full(len(y_train), y_train.mean())):.3f})")

# %% [markdown]
# The two thresholds should be close if the model is calibrated (they are here, and the calibration plot hugs the diagonal). Look at the left plot: where the value curve is flat near its top, nearby thresholds are about equally good, which means the choice is **robust** and you can pick a round, explainable number. We use the empirical optimum
# `t_star`; if the assumptions change (a pricier offer, a lower success rate) rerun this cell, not the model.

# %% [markdown]
# ## 7. The one final evaluation on the sealed test set
#
# Everything is frozen: pipeline, hyper-parameters, threshold. The chosen pipeline was already refit on all training rows by the search. Now we look at the test set, once, and compare the model's value with simple policies a manager could run without machine learning.

# %%
p_test = chosen.predict_proba(X_test)[:, 1]
auc, ap, brier = roc_auc_score(y_test, p_test), average_precision_score(y_test, p_test), brier_score_loss(y_test, p_test)
flag = p_test >= t_star
tp, fp = int((flag & (y_test == 1)).sum()), int((flag & (y_test == 0)).sum())
fn = int((~flag & (y_test == 1)).sum())
per_1000 = 1000 / len(y_test)
mtm = (X_test["contract"].str.lower().str.strip() == "month-to-month").to_numpy()
policies = {
    "contact nobody": 0.0,
    "contact everyone": net_value(y_test, np.ones(len(y_test)), 0.5),
    "contact month-to-month customers": GAIN_TP * np.sum(mtm & (y_test == 1)) - LOSS_FP * np.sum(mtm & (y_test == 0)),
    f"model, threshold {t_star:.2f}": net_value(y_test, p_test, t_star),
}
print(f"test ROC-AUC {auc:.3f} | PR-AUC {ap:.3f} (floor {y_test.mean():.3f}) | Brier {brier:.3f}")
print(f"at threshold {t_star:.2f}: flagged {flag.mean():.0%} of customers, precision {tp / max(tp + fp, 1):.2f}, recall {tp / (tp + fn):.2f}")
pd.Series({k: v * per_1000 for k, v in policies.items()}, name="net value per 1,000 customers").round(0).to_frame()

# %% [markdown]
# Two honest remarks. The test score should be close to the cross-validated one (if it is far lower, something leaked or the split was unlucky). And the **ordering of policies** matters more than the decimals: the model earns its keep only if it beats the no-ML policies, particularly the contract-only rule, under the stated assumptions. Notice how strong "contact everyone" is: with a cheap offer relative to the expected gain, most customers are worth contacting, and the model's value is the margin it adds by skipping the safest ones. With a more expensive offer that margin grows; rerun with different numbers to see.
# A number without an uncertainty is a rumour, so bootstrap the test set: resample customers with replacement and recompute the AUC.

# %%
rng = np.random.default_rng(0)
y_arr = y_test.to_numpy()
boot = []
for _ in range(300):
    idx = rng.integers(0, len(y_arr), len(y_arr))
    boot.append(roc_auc_score(y_arr[idx], p_test[idx]))
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"test ROC-AUC {auc:.3f}, bootstrap 95% interval [{lo:.3f}, {hi:.3f}] (300 resamples of {len(y_arr)} customers)")

# %% [markdown]
# ## 8. Where does it fail? Segment check
#
# An overall number hides segments where the model is worse. Check recall and ranking quality for the main segments (still only reporting, not tuning):

# %%
seg = pd.DataFrame({"contract": X_test["contract"].str.lower().str.strip(), "senior": X_test["senior_citizen"].astype(int), "y": y_test, "p": p_test, "flag": flag})
rows = []
for col in ["contract", "senior"]:
    for level, g in seg.groupby(col):
        rows.append({"segment": f"{col} = {level}", "n": len(g), "churn rate": g["y"].mean(), "flagged": g["flag"].mean(),
                     "recall": g.loc[g["y"] == 1, "flag"].mean(), "AUC": roc_auc_score(g["y"], g["p"])})
pd.DataFrame(rows).set_index("segment").round(2)

# %% [markdown]
# Read this table for big gaps. Here the AUC is similar across segments, but the two-year-contract segment has far lower recall: the same ranking quality with a much lower base rate means few of those customers clear a single global threshold. Whether to use segment-specific thresholds is a business policy question, not a modelling one. More generally, a segment with much lower recall or AUC is where retention will quietly fail, and a segment flagged far more often than its churn rate suggests is where offers will be wasted. Small segments give noisy numbers; say so in the model card rather than over-reading them.

# %% [markdown]
# ## 9. Interpretation: what does the model use?
#
# Using what we learned in notebook 13: permutation importance on the test set (now that all decisions are made, this cannot leak into them), and partial dependence for two features. Remember these describe the **model**, not causes.

# %%
pi = permutation_importance(chosen, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=42)
imp = pd.Series(pi.importances_mean, index=X_test.columns).sort_values(ascending=False)
print(imp.round(4).head(6).to_dict())
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
PartialDependenceDisplay.from_estimator(chosen, X_test, ["tenure_months", "support_calls"], ax=ax)
ax[0].set_ylabel("average predicted churn probability")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. The model card
#
# A model card is one page that lets someone else decide whether and how to use the model. The facts below come from this run; the code cell assembles them so the card cannot drift from the numbers. **Your job (Exercise 5) is to fill in the parts that need judgement.**

# %%
card = {
    "model": chosen_name,
    "training rows": len(X_train),
    "test rows": len(X_test),
    "test ROC-AUC (95% bootstrap)": f"{auc:.3f} [{lo:.3f}, {hi:.3f}]",
    "test PR-AUC vs prevalence": f"{ap:.3f} vs {y_test.mean():.3f}",
    "threshold": f"{t_star:.2f} (formula: {t_formula:.2f})",
    "assumed economics": f"value {VALUE_RETAINED:.0f}, success {P_SUCCESS:.0%}, offer cost {OFFER_COST:.0f}",
    "flagged share / precision / recall at threshold": f"{flag.mean():.0%} / {tp / max(tp + fp, 1):.2f} / {tp / (tp + fn):.2f}",
    "top features by permutation importance": list(imp.index[:3]),
}
for k, v in card.items():
    print(f"{k:<48}{v}")

# %% [markdown]
# > **MODEL CARD (fill in the blanks in your copy)**
# >
# > **Intended use:** ______ (who uses the scores, how often, for what action).
# > **Not intended for:** ______ (for example, pricing decisions, or customers outside the training population).
# > **Training data:** synthetic telecom customers, 5,000 rows, ______ (period, known gaps, population it does not represent).
# > **Evaluation:** single sealed test split; report the AUC interval above. **Known weak spots:** ______ (segments from section 8).
# > **Assumptions behind the threshold:** ______ (the three money numbers; who owns them; how you would measure the real offer success rate).
# > **Risks and monitoring:** ______ (data drift, changing offer, feedback loop: customers we contact stop being representative; what you would monitor and how often).
# > **Retraining trigger:** ______.
# > **Explanation limits:** importance shows what the model uses, not what causes churn.

# %% [markdown]
# ## Common mistakes (capstone edition)
#
# 1. **Exploring the whole dataset before splitting**, then "discovering" features that happen to work on the test rows.
# 2. **Cleaning outside the pipeline**, so production data is treated differently from training data (or statistics leak across the split).
# 3. **Choosing the model by the best single number** instead of a paired comparison on identical folds.
# 4. **Looking at the test set more than once** (for model choice, threshold, then "just checking"), which turns it into a validation set.
# 5. **Reporting AUC as if it were the business outcome**: the decision needs a threshold and costs.
# 6. **Skipping the simple baseline**; it is often most of the value (here, the contract-only row).
# 7. **Treating the cost assumptions as facts**; they are hypotheses to be measured.

# %% [markdown]
# ## Interview corner
#
# **1. Walk me through how you would build a churn model from a raw CSV.**
# <details><summary>Answer</summary>
# Frame the target, horizon, metric and costs; split (stratified, or by time if the data is temporal) before exploring; explore on the training part; put cleaning and encoding in a pipeline; score baselines; tune candidates with cross-validation; compare by paired folds; choose a threshold from costs on out-of-fold
# predictions; evaluate once on the sealed test set with an uncertainty interval; check segments; interpret carefully; document in a model card; plan monitoring.
# </details>
#
# **2. The tuned gradient boosting model beats logistic regression by 0.003 AUC. Do you ship it?**
# <details><summary>Answer</summary>
# Probably not on that evidence alone. Check the paired fold differences and their spread, look at PR-AUC and business value at the operating threshold, and weigh the cost of complexity (latency, monitoring, explainability). A gain well inside the noise does not justify it.
# </details>
#
# **3. How did you choose the threshold, and what if the business costs change?**
# <details><summary>Answer</summary>
# From an explicit cost table: flag when the expected benefit exceeds the cost (a threshold of cost / (success x value) for calibrated probabilities), confirmed by sweeping out-of-fold predictions. If costs change, recompute the threshold; the model stays the same. The assumptions, especially the offer success rate, should be measured with an experiment.
# </details>
#
# **4. How will you know the model has stopped working in production?**
# <details><summary>Answer</summary>
# Monitor input drift (feature distributions), score distribution, and, when outcomes arrive, realised AUC, calibration and campaign value by cohort; compare with the test interval. Watch for feedback effects (contacted customers behave differently) and keep a small random holdout that receives no offer to keep measuring the baseline churn.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Audit the raw file independently of pandas
# pandas can hide data quirks, so check the raw text with Python's `csv` module (`csv.DictReader`) on `DATA / "churn.csv"`. Store in `n_contract_labels_raw` the number of distinct `contract` strings as stored, in `n_contract_labels_clean` the number after `.lower().strip()`,
# and in `n_internet_none` the number of rows whose `internet_service` is the literal text `None`. Then store in `n_blank_internet` the number of rows where it is an empty string.

# %% tags=["starter"]
import csv

n_contract_labels_raw = ...
n_contract_labels_clean = ...
n_internet_none = ...
n_blank_internet = ...

# %%
with open(DATA / "churn.csv", newline="") as _fh:
    _rows = list(csv.DictReader(_fh))
assert n_contract_labels_raw == len({r["contract"] for r in _rows}), "distinct contract strings exactly as stored"
assert n_contract_labels_clean == len({r["contract"].lower().strip() for r in _rows}), "distinct after lower().strip()"
assert n_internet_none == sum(r["internet_service"] == "None" for r in _rows), "rows with the literal text None"
assert n_blank_internet == sum(r["internet_service"] == "" for r in _rows), "rows with an empty string"
assert n_internet_none == int((raw["internet_service"] == "None").sum()) and n_blank_internet == int(raw["internet_service"].isna().sum()), "the keep_default_na=False read should agree with the csv module"
print(f"Exercise 1 passed: {n_contract_labels_raw} raw labels -> {n_contract_labels_clean}; 'None' plan: {n_internet_none}; truly blank: {n_blank_internet}")

# %% tags=["solution"]
import csv

with open(DATA / "churn.csv", newline="") as fh:
    records = list(csv.DictReader(fh))
n_contract_labels_raw = len({r["contract"] for r in records})
n_contract_labels_clean = len({r["contract"].lower().strip() for r in records})
n_internet_none = sum(r["internet_service"] == "None" for r in records)
n_blank_internet = sum(r["internet_service"] == "" for r in records)

# %% [markdown]
# ### Exercise 2: Apply the decision rule yourself
# Using the arrays `cv_logit` and `cv_gbm` (paired AUC scores on identical folds), compute the mean paired difference `mean_diff` (boosting minus logistic), its standard error `se_diff` (sample standard deviation with `ddof=1` divided by the square root of the number of pairs),
# and `boosting_wins` = whether `mean_diff > 2 * se_diff`. Then store in `n_folds_boosting_ahead` the number of folds in which boosting scored higher than logistic.

# %% tags=["starter"]
mean_diff = ...
se_diff = ...
boosting_wins = ...
n_folds_boosting_ahead = ...

# %%
_d = np.asarray(cv_gbm) - np.asarray(cv_logit)
_se = _d.std(ddof=1) / np.sqrt(len(_d))
assert abs(mean_diff - _d.mean()) < 1e-12, "mean of (cv_gbm - cv_logit)"
assert abs(se_diff - _se) < 1e-12, "std(ddof=1) / sqrt(number of pairs)"
assert bool(boosting_wins) == bool(_d.mean() > 2 * _se), "mean_diff > 2 * se_diff"
assert n_folds_boosting_ahead == int((_d > 0).sum()), "count folds where cv_gbm > cv_logit"
assert bool(boosting_wins) == (chosen_name == "gradient boosting"), "this must agree with the decision the notebook made"
print(f"Exercise 2 passed: boosting ahead in {n_folds_boosting_ahead} of {len(_d)} folds; wins the rule: {boosting_wins}")

# %% tags=["solution"]
paired = np.asarray(cv_gbm) - np.asarray(cv_logit)
mean_diff = float(paired.mean())
se_diff = float(paired.std(ddof=1) / np.sqrt(len(paired)))
boosting_wins = bool(mean_diff > 2 * se_diff)
n_folds_boosting_ahead = int((paired > 0).sum())

# %% [markdown]
# ### Exercise 3: New economics, new threshold
# The retention team negotiates a cheaper offer: cost **10** instead of 20, with the same success rate (30%) and customer value (300). Using the out-of-fold probabilities `oof` and the grid `grid_t`, store in `t_formula_cheap` the formula threshold `cost / (success * value)`, in `t_best_cheap` the grid
# threshold that maximises **out-of-fold** net value under the new numbers, and in `test_value_cheap_per_1000` the **test-set** net value per 1,000 customers of using `t_best_cheap` (with `p_test`).
# *Hint: the expected gain for a flagged churner is `success * value - cost`; do not overwrite `GAIN_TP` or `LOSS_FP`.*

# %% tags=["starter"]
t_formula_cheap = ...
t_best_cheap = ...
test_value_cheap_per_1000 = ...

# %%
_cost = 10.0
_gain = P_SUCCESS * VALUE_RETAINED - _cost


def _nv(y_true, p, t):
    _f = p >= t
    _y = np.asarray(y_true)
    return _gain * np.sum(_f & (_y == 1)) - _cost * np.sum(_f & (_y == 0))


_vals = np.array([_nv(y_train, oof, t) for t in grid_t])
_tb = float(grid_t[_vals.argmax()])
assert abs(t_formula_cheap - _cost / (P_SUCCESS * VALUE_RETAINED)) < 1e-12, "cost / (success * value)"
assert abs(t_best_cheap - _tb) < 1e-9, f"maximise out-of-fold net value with gain {_gain:.0f} per true churner flagged and cost 10 per loyal customer flagged; expected {_tb:.2f}"
assert t_best_cheap < t_star, "a cheaper offer should lower the threshold (contact more people)"
assert abs(test_value_cheap_per_1000 - _nv(y_test, p_test, _tb) * 1000 / len(y_test)) < 1e-9, "apply the threshold to p_test once; per 1,000 customers"
print(f"Exercise 3 passed: threshold {t_star:.2f} -> {t_best_cheap:.2f}; test value {test_value_cheap_per_1000:,.0f} per 1,000 customers")

# %% tags=["solution"]
cheap_cost = 10.0
cheap_gain = P_SUCCESS * VALUE_RETAINED - cheap_cost


def cheap_value(y_true, p, t):
    flagged = p >= t
    yy = np.asarray(y_true)
    return cheap_gain * np.sum(flagged & (yy == 1)) - cheap_cost * np.sum(flagged & (yy == 0))


t_formula_cheap = cheap_cost / (P_SUCCESS * VALUE_RETAINED)
t_best_cheap = float(grid_t[np.array([cheap_value(y_train, oof, t) for t in grid_t]).argmax()])
test_value_cheap_per_1000 = float(cheap_value(y_test, p_test, t_best_cheap) * 1000 / len(y_test))

# %% [markdown]
# ### Exercise 4: A bootstrap interval for PR-AUC
# Repeat the bootstrap of section 7 for **PR-AUC** (`average_precision_score`) instead of ROC-AUC: `rng = np.random.default_rng(0)`, 300 resamples, each `idx = rng.integers(0, n, n)` with `n = len(y_test)`. Store the 2.5th and 97.5th percentiles in `ap_lo` and `ap_hi`, and set `floor_inside` to whether
# the prevalence floor `y_test.mean()` lies **inside** the interval (it should not, if the model has real skill).

# %% tags=["starter"]
ap_lo = ...
ap_hi = ...
floor_inside = ...

# %%
_r = np.random.default_rng(0)
_y = y_test.to_numpy()
_b = []
for _ in range(300):
    _i = _r.integers(0, len(_y), len(_y))
    _b.append(average_precision_score(_y[_i], p_test[_i]))
_lo, _hi = np.percentile(_b, [2.5, 97.5])
assert abs(ap_lo - _lo) < 1e-12 and abs(ap_hi - _hi) < 1e-12, "percentiles 2.5 and 97.5 of the 300 bootstrap PR-AUCs, rng = default_rng(0), idx = rng.integers(0, n, n)"
assert bool(floor_inside) == bool(_lo <= y_test.mean() <= _hi), "is y_test.mean() between ap_lo and ap_hi?"
assert not floor_inside, "a model with real skill should sit clearly above the prevalence floor"
print(f"Exercise 4 passed: PR-AUC interval [{ap_lo:.3f}, {ap_hi:.3f}] vs floor {y_test.mean():.3f}")

# %% tags=["solution"]
rng_b = np.random.default_rng(0)
yy = y_test.to_numpy()
boot_ap = []
for _ in range(300):
    idx = rng_b.integers(0, len(yy), len(yy))
    boot_ap.append(average_precision_score(yy[idx], p_test[idx]))
ap_lo, ap_hi = (float(v) for v in np.percentile(boot_ap, [2.5, 97.5]))
floor_inside = bool(ap_lo <= y_test.mean() <= ap_hi)

# %% [markdown]
# ### Exercise 5 (open-ended): Fill in the model card
# Copy the model-card template into a markdown cell of your own and complete every blank with specifics from this notebook's numbers: intended use, what it must not be used for, known weak spots from the segment table, the assumptions behind the threshold and how you would measure the offer success rate,
# what you would monitor and how often, and your retraining trigger. Then write one paragraph for the retention manager that begins "My recommendation is..." and says whether the model should be used, how, and what could make you change your mind.

# %% [markdown]
# ## Recap
# - Order of operations is the method: brief, split first, EDA on training only, baselines, pipelines with cleaning inside, tuning, paired comparison, threshold from costs on out-of-fold data, **one** test evaluation with an interval, segments, interpretation, model card.
# - A tuned gradient-boosting model must earn its place against a simple logistic baseline (and a one-feature rule) on identical folds.
# - The decision, not the AUC, is what the business buys: derive the threshold from explicit, challengeable assumptions and keep them with the model.
# - Report uncertainty and failure modes, not just a headline number.
#
# **Go deeper:** "Model Cards for Model Reporting" (Mitchell et al., 2019) for the model-card idea; scikit-learn user guide on cross-validation, pipelines, and "Tuning the decision threshold for class prediction" (scikit-learn.org/stable/); module 20_MLOps for packaging, tracking and monitoring what you built here.
#
# **Next:** continue with the Deep Learning module when you are ready to leave tabular data: [13_Deep_Learning](../13_Deep_Learning/README.md)

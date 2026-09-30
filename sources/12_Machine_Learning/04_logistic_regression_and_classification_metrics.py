# %% [markdown]
# # Logistic Regression and Classification Metrics
#
# **Why this matters.** "Will this customer churn / default / click?" is the most common question a data scientist gets, and the answer a business needs is
# rarely a hard yes/no. It needs a *probability* and a *decision rule* that reflects what mistakes cost. Logistic regression is the standard baseline
# for it, and choosing between precision, recall, ROC-AUC and PR-AUC correctly is one of the most reliably tested interview topics.
#
# **You will learn**
# - how a linear score becomes a probability (sigmoid) and how to reproduce `predict_proba` by hand
# - build a confusion matrix and compute precision, recall and F1 from it
# - choose a decision threshold from a business cost instead of defaulting to 0.5
# - compare ROC-AUC and PR-AUC, and see exactly when ROC-AUC flatters a model
# - interpret coefficients as odds ratios
#
# **Prerequisites:** [ML problem framing](01_ml_problem_framing_and_workflow.ipynb) (the cost assumptions come from there) and
# [Train/test split and leakage](02_train_test_split_and_data_leakage.ipynb).
#
# > Data used: `datasets/churn.csv` (synthetic, `churned` = 1 if the customer left).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# In the CSV, internet_service is the literal text "None" (no internet plan) for many rows and truly blank for a few.
# pandas would turn BOTH into NaN by default; keep_default_na=False + na_values=[""] keeps "None" as a real category.
churn = pd.read_csv(Path("../datasets") / "churn.csv", keep_default_na=False, na_values=[""])
churn["contract"] = churn["contract"].str.lower()          # fix the planted casing bug
churn["internet_service"] = churn["internet_service"].fillna("Unknown")
print(churn["internet_service"].value_counts().to_dict())

# %% [markdown]
# We build a small numeric design matrix: numeric columns as they are, categories as 0/1 dummy columns with one level dropped as the reference (`drop_first=True`).
# `total_charges` is left out on purpose: it is almost `tenure * monthly_charges` (redundant) and has missing values; notebook 08 handles messy columns properly with pipelines.

# %%
num_cols = ["tenure_months", "monthly_charges", "support_calls", "senior_citizen"]
X = pd.get_dummies(
    churn[num_cols + ["contract", "internet_service", "paperless_billing", "autopay"]],
    columns=["contract", "internet_service", "paperless_billing", "autopay"],
    drop_first=True,
    dtype=float,
)
y = churn["churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
print(list(X.columns))

# %% [markdown]
# ## 1. From a linear score to a probability
#
# A linear score `z = b0 + b1*x1 + ...` can be any real number, but a probability must sit in [0, 1]. The **sigmoid** squashes any `z` into that range:
# `p = 1 / (1 + exp(-z))`. Equivalently, the model is linear in the **log-odds**: `log(p / (1 - p)) = z`.

# %%
z = np.linspace(-6, 6, 200)
plt.figure(figsize=(5, 3.2))
plt.plot(z, 1 / (1 + np.exp(-z)))
plt.axhline(0.5, color="grey", ls=":")
plt.axvline(0, color="grey", ls=":")
plt.title("Sigmoid: linear score to probability")
plt.xlabel("linear score z (log-odds)")
plt.ylabel("probability of churn")
plt.show()

# %% [markdown]
# `z = 0` means 50/50; `z = 2` is about 88%; each +1 in `z` multiplies the *odds* by e (~2.72). Fit the model, then reproduce its probabilities by hand:

# %%
logit = LogisticRegression(max_iter=2000).fit(X_train, y_train)
z_test = X_test.to_numpy() @ logit.coef_[0] + logit.intercept_[0]
p_manual = 1 / (1 + np.exp(-z_test))
p_sklearn = logit.predict_proba(X_test)[:, 1]
print("manual sigmoid equals predict_proba:", np.allclose(p_manual, p_sklearn))

# %% [markdown]
# *What just happened?* `predict_proba` is nothing more than the sigmoid of a dot product. Training chooses the coefficients that minimise **log loss**
# (cross-entropy): a confident wrong answer is punished very heavily, which is what makes the outputs behave like honest probabilities. Verify the formula:

# %%
p = np.clip(p_sklearn, 1e-15, 1 - 1e-15)
manual_ll = -np.mean(y_test * np.log(p) + (1 - y_test) * np.log(1 - p))
print("manual log loss:", round(manual_ll, 5), "| sklearn:", round(log_loss(y_test, p_sklearn), 5))

# %% [markdown]
# ## 2. Threshold, confusion matrix, precision and recall
#
# To act, we convert probabilities to labels with a **threshold**: predict churn when `p >= t`. The default 0.5 is a convention, not a law. The
# **confusion matrix** counts what happened:
#
# | | predicted stay | predicted churn |
# |---|---|---|
# | **actually stayed** | true negative (TN) | false positive (FP) |
# | **actually churned** | false negative (FN) | true positive (TP) |
#
# - **Precision** = TP / (TP + FP): of the customers we flagged, how many really churn? (Cost of wasted offers.)
# - **Recall** (sensitivity) = TP / (TP + FN): of the real churners, how many did we catch? (Cost of missed customers.)
# - **F1** = harmonic mean of the two; it punishes a big imbalance between them.
# - **Accuracy** = (TP + TN) / everything; it hides all of the above when classes are imbalanced.

# %%
def report(y_true, proba, t):
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {"threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": tp / (tp + fp), "recall": tp / (tp + fn), "F1": f1_score(y_true, pred),
            "accuracy": (tp + tn) / len(y_true)}

pd.DataFrame([report(y_test, p_sklearn, t) for t in [0.2, 0.35, 0.5, 0.65]]).round(3)

# %% [markdown]
# Lower thresholds flag more people: recall rises, precision falls. There is **no threshold that maximises both**; the trade-off is the point. Check that our hand-made metrics
# agree with scikit-learn's, and look at the matrix at 0.5:

# %%
r = report(y_test, p_sklearn, 0.5)
pred50 = (p_sklearn >= 0.5).astype(int)
print("precision matches:", np.isclose(r["precision"], precision_score(y_test, pred50)), "| recall matches:", np.isclose(r["recall"], recall_score(y_test, pred50)))
ConfusionMatrixDisplay.from_predictions(y_test, pred50, display_labels=["stayed", "churned"], colorbar=False)
plt.title("Confusion matrix at threshold 0.5 (test set)")
plt.show()

# %% [markdown]
# ## 3. Choose the threshold from the cost
#
# In notebook 01 we *assumed* a correct offer is worth +40 and a wasted one costs -10, giving a break-even probability of `10 / (40 + 10) = 0.2`. Sweep thresholds and
# compute the campaign value on the test set:

# %%
GAIN_TP, COST_FP = 40.0, -10.0

def value_at(t, y_true=y_test, proba=p_sklearn):
    flag = proba >= t
    tp = (flag & (y_true == 1)).sum()
    fp = (flag & (y_true == 0)).sum()
    return GAIN_TP * tp + COST_FP * fp

ts = np.linspace(0.05, 0.9, 86)
vals = np.array([value_at(t) for t in ts])
best_t = ts[vals.argmax()]
plt.figure(figsize=(6, 3.4))
plt.plot(ts, vals)
plt.axvline(0.2, color="grey", ls=":", label="cost-based threshold 0.2")
plt.axvline(0.5, color="red", ls=":", label="default 0.5")
plt.title("Campaign value vs decision threshold (assumed costs)")
plt.xlabel("threshold")
plt.ylabel("value on test set (currency units)")
plt.legend()
plt.show()
print(f"best threshold on this test set: {best_t:.2f}; value {vals.max():,.0f} vs {value_at(0.5):,.0f} at 0.5")

# %% [markdown]
# The best value sits near the theoretical 0.2 *because* the probabilities are reasonably calibrated: when the model says 20%, about 20% of those customers churn.
# That is worth checking rather than assuming, with a reliability table (bin predictions, compare average predicted vs actual rate):

# %%
cal = pd.DataFrame({"p": p_sklearn, "y": y_test.to_numpy()})
cal["bin"] = pd.qcut(cal["p"], 5, duplicates="drop")
cal.groupby("bin", observed=True).agg(mean_predicted=("p", "mean"), actual_rate=("y", "mean"), n=("y", "size")).round(3)

# %% [markdown]
# Note the optimum is picked using the *test* set here for teaching. In real work, choose the threshold on validation data (or from a cost formula) and only report on test.

# %% [markdown]
# ## 4. ROC-AUC vs PR-AUC
#
# Both summarise ranking quality across **all** thresholds, without committing to one.
#
# - **ROC curve**: true-positive rate (recall) against false-positive rate (FP / all negatives). **ROC-AUC** = probability that a random churner gets a higher
#   score than a random stayer (0.5 = coin flip, 1 = perfect ranking). Let us verify that interpretation directly:

# %%
pos = p_sklearn[y_test.to_numpy() == 1]
neg = p_sklearn[y_test.to_numpy() == 0]
pairwise = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
print("ROC-AUC by pairwise comparison:", round(pairwise, 4), "| roc_auc_score:", round(roc_auc_score(y_test, p_sklearn), 4))

# %% [markdown]
# - **Precision-recall curve**: precision against recall. Its summary, **average precision (PR-AUC)**, is judged against a baseline equal to the **positive rate**
#   (a random ranker has precision equal to the prevalence), not against 0.5.
#
# Why two? Because the false-positive *rate* divides by the number of negatives. When negatives are abundant, a huge number of false positives barely moves that
# rate, so ROC-AUC can look great while precision is poor. See it by keeping the same model and the same ranking, but making churn rare in the evaluation set
# (keep all the stayers, but only as many churners as make up about 5% of the set):

# %%
rng = np.random.default_rng(42)
yt, pt = y_test.to_numpy(), p_sklearn
idx_pos, idx_neg = np.flatnonzero(yt == 1), np.flatnonzero(yt == 0)
n_keep = len(idx_neg) // 20        # positives will be ~5% of the evaluation set
roc_draws, pr_draws, rate_draws = [], [], []
for _ in range(200):               # repeat: with only ~40 positives a single draw is noisy
    keep = np.concatenate([idx_neg, rng.choice(idx_pos, size=n_keep, replace=False)])
    roc_draws.append(roc_auc_score(yt[keep], pt[keep]))
    pr_draws.append(average_precision_score(yt[keep], pt[keep]))
    rate_draws.append(yt[keep].mean())
rows = [
    ("original test set", yt.mean(), roc_auc_score(yt, pt), average_precision_score(yt, pt)),
    ("rare positives (~5%), mean of 200 draws", np.mean(rate_draws), np.mean(roc_draws), np.mean(pr_draws)),
]
pd.DataFrame(rows, columns=["evaluation set", "positive rate", "ROC-AUC", "PR-AUC (avg precision)"]).round(3)

# %% [markdown]
# *What just happened?* Same model, same ranking rule: ROC-AUC stays essentially the same on average (it does not depend on the class mix; a single draw with only ~40 positives is noisy, which is why we averaged 200), but PR-AUC collapses toward the (now tiny) prevalence, which is what the business
# would feel as a flood of false alarms. **Rule of thumb:** with a rare positive class where you care about the positives (fraud, faults, rare disease), report PR-AUC
# and precision/recall at the operating threshold; ROC-AUC is fine for comparing rankings on roughly balanced data, and is insensitive to class prevalence
# (a feature when the mix will change, a trap when it will not).
#
# PR-AUC also has failure modes: it depends on prevalence, so you cannot compare it across datasets with different positive rates, and it ignores true negatives entirely.

# %%
fpr, tpr, _ = roc_curve(y_test, p_sklearn)
prec, rec, _ = precision_recall_curve(y_test, p_sklearn)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(fpr, tpr)
ax[0].plot([0, 1], [0, 1], "k:")
ax[0].set(title="ROC curve (test set)", xlabel="false-positive rate", ylabel="true-positive rate (recall)")
ax[1].plot(rec, prec)
ax[1].axhline(y_test.mean(), color="k", ls=":")
ax[1].set(title="Precision-recall curve (dotted: random)", xlabel="recall", ylabel="precision")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Interpreting coefficients as odds ratios
#
# Each coefficient is the change in log-odds per unit of the feature; `exp(coef)` is the **odds ratio**: a value of 1.5 means the odds of churn are 50% higher
# per unit, holding the other features fixed. Coefficients here are on raw units (per month, per dollar, per support call, versus the dropped reference category),
# so they are directly readable, but they are also shrunk slightly by the default L2 penalty (`C=1.0`).

# %%
odds = pd.Series(np.exp(logit.coef_[0]), index=X.columns).sort_values(ascending=False)
odds.round(3).to_frame("odds ratio")

# %% [markdown]
# Reading it: values above 1 raise churn odds, below 1 lower them. Contract dummies compare against the dropped reference (month-to-month), so a value well below 1 for
# a longer contract means those customers churn far less. Remember: odds ratios describe **association in this data**, not causal effect (moving a customer to
# a two-year contract would not necessarily reduce churn by that factor), and they are only comparable across features measured in comparable units.

# %% [markdown]
# ## Common mistakes
#
# 1. **Reporting accuracy** on imbalanced data (see the majority baseline in notebook 01).
# 2. **Using 0.5 by default** instead of a cost-derived or validation-derived threshold.
# 3. **Calling `predict` and then computing AUC**: AUC needs scores/probabilities (`predict_proba`), not hard labels.
# 4. **Reading `predict_proba` as calibrated** for models that are not (trees, boosted models and SVMs often need calibration; check a reliability table).
# 5. **Tuning the threshold on the test set** and then reporting that test number.
# 6. **Interpreting coefficients causally**, or comparing raw coefficients across different units.

# %%
# Mistake 3 in code: AUC from hard labels throws away the ranking information
print("AUC from probabilities :", round(roc_auc_score(y_test, p_sklearn), 3))
print("AUC from 0/1 predictions:", round(roc_auc_score(y_test, pred50), 3))

# %% [markdown]
# ## Interview corner
#
# **1. Precision vs recall: give an example where each is more important.**
# <details><summary>Answer</summary>
# Recall matters more when a miss is expensive (screening for a serious disease, fraud that costs far more than a review). Precision matters more when a false alarm is
# expensive (blocking legitimate customers, sending costly interventions). The right balance is set by the ratio of the two costs.
# </details>
#
# **2. What does ROC-AUC mean in plain words, and when is it misleading?**
# <details><summary>Answer</summary>
# The probability that a randomly chosen positive is scored above a randomly chosen negative. It is threshold-free and insensitive to prevalence, which makes it misleading
# for rare-positive problems, where a good-looking ROC-AUC can coexist with poor precision. Use PR-AUC or precision/recall at the operating point.
# </details>
#
# **3. How do you choose the classification threshold?**
# <details><summary>Answer</summary>
# From the cost matrix (break-even probability = cost of FP / (cost of FP + gain of TP) when probabilities are calibrated), or by maximising a business metric on a validation
# set, or by a capacity constraint (contact the top k). Never from the test set, and re-check when prevalence drifts.
# </details>
#
# **4. Why not use linear regression on a 0/1 target?**
# <details><summary>Answer</summary>
# Predictions leave [0, 1], squared error is a poor fit to a Bernoulli outcome, and the errors are heteroscedastic. Logistic regression models the log-odds and is fit by maximum
# likelihood, so outputs are valid probabilities.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Metrics by hand
# At threshold **0.3** on the test probabilities `p_sklearn`, compute `precision_30` and `recall_30` using only NumPy counts of TP, FP, FN (no sklearn metric functions).
# *Hint: `flag = p_sklearn >= 0.3`; TP = flag & (y == 1).*

# %% tags=["starter"]
precision_30 = ...
recall_30 = ...

# %%
_flag = (p_sklearn >= 0.3).astype(int)
assert abs(precision_30 - precision_score(y_test, _flag)) < 1e-9, "precision = TP / (TP + FP)"
assert abs(recall_30 - recall_score(y_test, _flag)) < 1e-9, "recall = TP / (TP + FN)"
print(f"Exercise 1 passed: precision {precision_30:.3f}, recall {recall_30:.3f}")

# %% tags=["solution"]
_yt = y_test.to_numpy()
_fl = p_sklearn >= 0.3
_tp = int((_fl & (_yt == 1)).sum())
_fp = int((_fl & (_yt == 0)).sum())
_fn = int((~_fl & (_yt == 1)).sum())
precision_30 = _tp / (_tp + _fp)
recall_30 = _tp / (_tp + _fn)

# %% [markdown]
# ### Exercise 2: A different cost structure
# The retention team gets a cheaper discount: a wasted offer costs only **-2** (gain for a correct offer stays **+40**). What is the break-even threshold `t_cheap` (a number)
# and, among thresholds in `ts` (the array defined in section 3), which one maximises test-set value under the new costs? Store it in `best_t_cheap`.
# *Hint: recompute value per threshold with the new `COST_FP`; do not overwrite the notebook's `value_at`.*

# %% tags=["starter"]
t_cheap = ...
best_t_cheap = ...

# %%
assert abs(t_cheap - 2 / 42) < 1e-9, "break-even p: cost_fp / (cost_fp + gain) = 2 / (2 + 40)"
def _v(t):
    f = p_sklearn >= t
    return 40 * (f & (y_test.to_numpy() == 1)).sum() - 2 * (f & (y_test.to_numpy() == 0)).sum()
_best = ts[int(np.argmax([_v(t) for t in ts]))]
assert abs(best_t_cheap - _best) < 1e-9, f"expected {_best:.2f}"
print(f"Exercise 2 passed: break-even {t_cheap:.3f}, best swept threshold {best_t_cheap:.2f} (vs {best_t:.2f} with the original costs)")

# %% tags=["solution"]
t_cheap = 2 / (2 + 40)
def _vals(t):
    f = p_sklearn >= t
    return 40 * (f & (y_test.to_numpy() == 1)).sum() - 2 * (f & (y_test.to_numpy() == 0)).sum()
best_t_cheap = ts[int(np.argmax([_vals(t) for t in ts]))]

# %% [markdown]
# ### Exercise 3: Which feature moves the odds most?
# From the `odds` Series (section 5), store in `strongest_up` the feature with the highest odds ratio, and in `strongest_down` the feature with the lowest.
# Then store in `contract_two_year_effect` the **percentage change in churn odds** for a two-year contract versus month-to-month, as `(odds_ratio - 1) * 100`
# (a negative number means lower odds). *Hint: look for the dummy column name in `X.columns`.*

# %% tags=["starter"]
strongest_up = ...
strongest_down = ...
contract_two_year_effect = ...

# %%
assert strongest_up == odds.idxmax(), "highest odds ratio"
assert strongest_down == odds.idxmin(), "lowest odds ratio"
_col = [c for c in X.columns if "two year" in c][0]
assert abs(contract_two_year_effect - (odds[_col] - 1) * 100) < 1e-9, "(odds_ratio - 1) * 100 for the two-year dummy"
print(f"Exercise 3 passed: up={strongest_up}, down={strongest_down}, two-year contract changes odds by {contract_two_year_effect:.0f}%")

# %% tags=["solution"]
strongest_up = odds.idxmax()
strongest_down = odds.idxmin()
_c = [c for c in X.columns if "two year" in c][0]
contract_two_year_effect = float((odds[_c] - 1) * 100)

# %% [markdown]
# ### Exercise 4: Does class weighting change the ranking or the threshold?
# Fit `LogisticRegression(max_iter=2000, class_weight="balanced")` on the training data. Store its test ROC-AUC in `auc_balanced` and its test **recall at threshold 0.5** in `recall_balanced`.
# Then set `ranking_changed_much = True` if `abs(auc_balanced - roc_auc_score(y_test, p_sklearn)) > 0.01`, else `False`.
# (Class weights mostly shift the probabilities, so they act like moving the threshold, rather than teaching the model a new ranking.)

# %% tags=["starter"]
auc_balanced = ...
recall_balanced = ...
ranking_changed_much = ...

# %%
_m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_train, y_train)
_p = _m.predict_proba(X_test)[:, 1]
assert abs(auc_balanced - roc_auc_score(y_test, _p)) < 1e-9, "roc_auc_score on the balanced model's probabilities"
assert abs(recall_balanced - recall_score(y_test, (_p >= 0.5).astype(int))) < 1e-9, "recall at threshold 0.5"
assert ranking_changed_much == (abs(roc_auc_score(y_test, _p) - roc_auc_score(y_test, p_sklearn)) > 0.01), "compare AUCs using the 0.01 rule"
print(f"Exercise 4 passed: AUC {auc_balanced:.3f} vs {roc_auc_score(y_test, p_sklearn):.3f}; recall at 0.5 {recall_balanced:.3f} vs {r['recall']:.3f}")

# %% tags=["solution"]
_mb = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_train, y_train)
_pb = _mb.predict_proba(X_test)[:, 1]
auc_balanced = float(roc_auc_score(y_test, _pb))
recall_balanced = float(recall_score(y_test, (_pb >= 0.5).astype(int)))
ranking_changed_much = bool(abs(auc_balanced - roc_auc_score(y_test, p_sklearn)) > 0.01)

# %% [markdown]
# ### Exercise 5 (open-ended): Add a feature, measure the change
# Add an interaction feature, for example support calls times month-to-month status (`X["mtm_x_calls"] = X["support_calls"] * ((X["contract_one year"] == 0) & (X["contract_two year"] == 0))`), or any idea from the odds ratios; refit,
# and report the change in test ROC-AUC and PR-AUC. Write two sentences: did it help, and is that difference bigger than the split-to-split noise you saw in notebook 02?

# %% [markdown]
# ## Recap
# - Logistic regression = sigmoid of a linear score; coefficients are log-odds, `exp(coef)` are odds ratios (associations, not causes).
# - Confusion matrix -> precision, recall, F1; they trade off through the threshold. Accuracy hides this on imbalanced data.
# - Set the threshold from costs (break-even `cost_fp / (cost_fp + gain_tp)` for calibrated probabilities) and validate calibration.
# - ROC-AUC is prevalence-insensitive and can flatter rare-positive problems; PR-AUC exposes them but depends on prevalence.
#
# **Go deeper:** scikit-learn user guide "Metrics and scoring" and "Probability calibration" (scikit-learn.org/stable/user_guide.html).
#
# **Next:** [Decision trees and overfitting](05_decision_trees_and_overfitting.ipynb)

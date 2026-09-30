# %% [markdown]
# # ML Problem Framing and the Standard Workflow
#
# **Why this matters.** Most failed ML projects do not fail because someone picked the wrong algorithm.
# They fail because the target was vague ("predict churn" - over what horizon? known when?), the metric had no
# link to money, or nobody checked whether a trivial rule already did most of the job. A senior data scientist
# spends the first hour on those three questions, and the first model they train is a *baseline* that is
# deliberately dumb.
#
# **You will learn**
# - tell supervised, unsupervised and other problem types apart, and choose classification vs regression
# - turn a business request into a precise target and a metric tied to a cost
# - build majority-class and mean/median baselines and read them honestly
# - run the standard workflow (frame, split, baseline, model, evaluate, iterate) once end to end
#
# **Prerequisites:** [Loading and inspecting data](../05_Pandas/01_loading_and_inspecting_data.ipynb) and basic
# probability from [Statistics and Probability](../03_Statistics_and_Probability/).
#
# > Data used: `datasets/churn.csv` (classification) and `datasets/housing.csv` (regression). Both are synthetic
# > (see `docs/DATASETS.md`), with real signal planted in them.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("../datasets")
churn = pd.read_csv(DATA / "churn.csv")
housing = pd.read_csv(DATA / "housing.csv")
print(churn.shape, housing.shape)

# %% [markdown]
# ## 1. What kind of problem is this?
#
# | Family | You have | You want | Example in this course |
# |---|---|---|---|
# | **Supervised: classification** | features + a *category* label | predict the category | will this customer churn? |
# | **Supervised: regression** | features + a *number* label | predict the number | what will this house sell for? |
# | **Unsupervised** | features only | find structure | segment customers (notebook 11) |
# | **Anomaly detection** | mostly normal data | flag the odd one | sensor faults (notebook 15) |
# | **Forecasting** | a time series | predict future values | hourly power demand (notebook 14) |
#
# "Supervised" just means every training row comes with the answer, so we can *measure* how wrong a model is.
# The choice between classification and regression is decided by the **target's type and by how the prediction
# will be used**: you could also frame churn as "months until churn" (regression or survival analysis), but if
# the business only acts on a yes/no list, classification is simpler and easier to evaluate.

# %% [markdown]
# ## 2. Defining the target precisely
#
# Look at what `churned` actually is in our table, and ask what a model could legitimately know at prediction time.

# %%
churn.head(3)

# %% [markdown]
# Questions to write down *before* modelling:
#
# 1. **Unit of prediction.** One row per customer. Fine.
# 2. **Horizon.** "Churned" means what, and by when? A real project pins this down ("cancelled within 30 days after the
#    snapshot date"). Our synthetic file has no dates, so we treat it as given, but you would insist on it at work.
# 3. **Availability at prediction time.** Every feature must be known *before* the outcome. `total_charges` looks like
#    `tenure_months * monthly_charges`, so it may just be derived from features we already have. Is it also a
#    leak (recorded only at cancellation)? Here we can check with data instead of guessing:

# %%
ratio = churn["total_charges"] / (churn["tenure_months"].clip(lower=1) * churn["monthly_charges"])
print(ratio.groupby(churn["churned"]).describe().round(2)[["mean", "25%", "50%", "75%"]])

# %% [markdown]
# *What just happened?* We compared the ratio `total / (tenure * monthly)` for churners and stayers. If `total_charges` were a
# leak, the two groups would differ in some telling way; they look alike, so it is just redundant with tenure and
# monthly charge (and, with the missing values, an imputation problem, not a leak). Checking a suspicious feature against the target and its neighbours is a habit worth having.
#
# **The class balance** tells you which metrics make sense:

# %%
churn["churned"].value_counts(normalize=True).round(3)

# %% [markdown]
# ## 3. Metric = business cost, written down
#
# Accuracy is a fine metric only when every mistake costs the same. In churn they do not:
#
# - a **false negative** (we ignore someone who then leaves) loses a customer;
# - a **false positive** (we send a retention offer to someone who would have stayed) wastes a discount.
#
# Suppose finance tells us (these numbers are **assumptions for this exercise**, not facts from the data):
# a retention offer to a customer who *would* have churned is worth `+40` on average after the discount, and an
# offer to someone who would have stayed costs `-10`. Then the value of contacting a set of customers is
# `40 * (true positives) - 10 * (false positives)`.

# %%
GAIN_TP = 40.0   # assumed value of correctly contacting a churner
COST_FP = -10.0  # assumed cost of contacting a customer who would have stayed


def campaign_value(y_true, contacted):
    """Total value of contacting the customers where `contacted` is True."""
    y_true = np.asarray(y_true).astype(bool)
    contacted = np.asarray(contacted).astype(bool)
    tp = (contacted & y_true).sum()
    fp = (contacted & ~y_true).sum()
    return GAIN_TP * tp + COST_FP * fp


y_all = churn["churned"]
print("contact nobody  :", campaign_value(y_all, np.zeros(len(y_all))))
print("contact everyone:", campaign_value(y_all, np.ones(len(y_all))))

# %% [markdown]
# Even "contact everyone" is profitable under these assumptions, because a large share of the list are real churners
# and a hit is worth four misses. A model has to beat *that* to be worth building. A useful consequence: contacting someone
# is worth it when `p * 40 - (1 - p) * 10 > 0`, i.e. `p > 10 / (40 + 10) = 0.2`. The **decision threshold comes from the
# cost ratio**, not from a default 0.5. We use that again in [notebook 04](04_logistic_regression_and_classification_metrics.ipynb).

# %% [markdown]
# ## 4. Baselines first
#
# A baseline is the best *trivial* predictor. It gives your metric a floor: a model that cannot beat it has learned nothing,
# and a model that beats it by a hair may not justify its complexity. First we split (details in
# [notebook 02](02_train_test_split_and_data_leakage.ipynb)) so that every score is measured on rows the model never saw.

# %%
churn["contract"] = churn["contract"].str.lower()  # quick fix of the casing bug; notebook 08 does it inside a pipeline
num_cols = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "senior_citizen"]
X = churn[num_cols]
y = churn["churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
pred = majority.predict(X_test)
print("majority-class accuracy:", round(accuracy_score(y_test, pred), 3))
print("majority-class AUC     :", roc_auc_score(y_test, majority.predict_proba(X_test)[:, 1]))
print("campaign value (contacts nobody):", campaign_value(y_test, pred))

# %% [markdown]
# *What just happened?* Predicting "nobody churns" is right whenever the customer stays, so its accuracy equals the
# share of stayers: **that accuracy is the floor, not a good score**. The dummy model's AUC is exactly 0.5 (no ranking
# ability at all), and as a business decision it earns zero. A claim like "our model is 70% accurate" is meaningless
# without this number next to it.
#
# ### A regression baseline
# For a numeric target the equivalent is "always predict the training mean (or median)". The scores are in the target's own units,
# so MAE (mean absolute error) is easy to explain to a stakeholder.

# %%
# The housing file contains a handful of absurd prices (negative, or millions for a small house). Notebook 03 treats
# them carefully; here we just flag them (price per sq ft far outside the normal range) and count them.
ppsf = housing["price"] / housing["area_sqft"]
sane = ppsf.between(20, 500)
print("rows flagged as absurd (price per sq ft outside 20-500):", int((~sane).sum()))

hX = housing[["area_sqft", "bedrooms", "bathrooms", "garage_spaces", "distance_to_center_km"]]
hy = housing["price"]
hX_train, hX_test, hy_train, hy_test = train_test_split(hX, hy, test_size=0.25, random_state=42)

for strategy in ["mean", "median"]:
    d = DummyRegressor(strategy=strategy).fit(hX_train, hy_train)
    print(f"{strategy:6s} predictor  MAE on test = {mean_absolute_error(hy_test, d.predict(hX_test)):,.0f}")

# %% [markdown]
# On this data the two constants score almost the same MAE, because house prices here are only mildly skewed. In theory the
# **median** minimises MAE and the **mean** minimises squared error, and on heavily skewed targets (incomes, claim sizes) the gap is large.
# Pick the baseline constant that matches your metric.

# %% [markdown]
# ## 5. The standard workflow, once, end to end
#
# 1. **Frame**: target, horizon, unit, cost of errors (done above).
# 2. **Split** off a test set you will touch *once* at the end.
# 3. **Baseline**: dummy model, then a simple interpretable model.
# 4. **Model + iterate**: engineer features, try stronger models, tune. Use cross-validation on the training part only.
# 5. **Evaluate once** on the test set; report the metric with its uncertainty and the baseline beside it.
# 6. **Ship and monitor** (the MLOps module): a model that is not monitored drifts silently.
#
# Here is a first "simple model" pass for churn. A `Pipeline` bundles preprocessing and the model so that the
# imputer and scaler are learned on training rows only (notebook 08 goes deep).

# %%
simple = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
simple.fit(X_train, y_train)
proba = simple.predict_proba(X_test)[:, 1]
print("logistic AUC:", round(roc_auc_score(y_test, proba), 3), "(baseline 0.5)")

threshold = 10 / (40 + 10)
print("value, contact everyone   :", campaign_value(y_test, np.ones(len(y_test))))
print("value, contact if p > 0.2 :", campaign_value(y_test, proba > threshold))

# %% [markdown]
# Reading this honestly: the model ranks customers above chance (AUC above the dummy's 0.5). Whether targeting only
# those above the cost-derived threshold beats contacting everyone is a *measured* result on this one test set; compare the
# two values above. What we have **not** done: quantify uncertainty (one split is one draw; see
# [notebook 07](07_cross_validation_and_model_evaluation.ipynb)), or check that predicted probabilities are calibrated
# (a threshold of 0.2 assumes they are).

# %% [markdown]
# ## Common mistakes
#
# 1. **Starting with a fancy model** and no baseline. You cannot tell 0.72 AUC from 0.70 without knowing what nothing scores.
# 2. **Quoting accuracy on imbalanced data.** A 63/37 split already makes "predict the majority" look decent.
# 3. **Choosing the metric after seeing results.** Decide it from the business cost first, or you will pick whichever number looks best.
# 4. **Vague targets.** "Churn" without a horizon and snapshot date cannot be turned into a leak-free training set.
# 5. **Using a feature that will not exist at prediction time** (covered in the next notebook).

# %%
# Mistake 2 in numbers: accuracy of a model that never finds a single churner
never = DummyClassifier(strategy="constant", constant=0).fit(X_train, y_train)
print("accuracy:", round(accuracy_score(y_test, never.predict(X_test)), 3), "| churners found:", int(never.predict(X_test).sum()))

# %% [markdown]
# ## Interview corner
#
# **1. Why train a dummy model?**
# <details><summary>Answer</summary>
# It sets the floor every real model must beat and exposes trivial problems (like a 95/5 class split where 95% accuracy is free).
# It also validates the evaluation pipeline itself: a dummy scoring far above chance means you have a leak.
# </details>
#
# **2. A stakeholder asks for "the most accurate churn model". What do you ask back?**
# <details><summary>Answer</summary>
# What action follows a prediction, what a wrong contact and a missed churner each cost, over what horizon, and how many
# customers the team can realistically contact. That converts "accurate" into a cost-based metric and a threshold or a top-k budget.
# </details>
#
# **3. Classification or regression for "how long until this customer leaves"?**
# <details><summary>Answer</summary>
# It can be either: a regression on time-to-event, a classification on "leaves within 30 days", or survival analysis, which handles
# customers who have not left *yet* (censoring). Regression on only the churners would be biased. Pick by how the output is used.
# </details>
#
# **4. Which constant minimises MAE and which minimises MSE?**
# <details><summary>Answer</summary>
# The median minimises mean absolute error; the mean minimises mean squared error. So the right regression baseline depends on the metric.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: The bar to clear
# Using the `y_train` / `y_test` split above, store in `baseline_acc` the test accuracy of always predicting the **training** majority class
# (compute it by hand from `y_train` and `y_test`, without a Dummy model). *Hint: `y_train.mode()[0]`.*

# %% tags=["starter"]
baseline_acc = ...

# %%
maj = y_train.value_counts().idxmax()
expected = float((y_test == maj).mean())
assert abs(baseline_acc - expected) < 1e-9, f"expected {expected:.4f}"
print("Exercise 1 passed:", round(baseline_acc, 3))

# %% tags=["solution"]
baseline_acc = float((y_test == y_train.mode()[0]).mean())

# %% [markdown]
# ### Exercise 2: What does a threshold change?
# Using `proba` (the logistic model's test probabilities) compute `value_at_050`: the campaign value if we only contact customers with
# probability above **0.5**, and `best_threshold`: which threshold among `[0.1, 0.2, 0.3, 0.4, 0.5]` gives the highest value.
# *Hint: loop over thresholds and call `campaign_value`.*

# %% tags=["starter"]
value_at_050 = ...
best_threshold = ...

# %%
vals = {t: campaign_value(y_test, proba > t) for t in [0.1, 0.2, 0.3, 0.4, 0.5]}
assert value_at_050 == vals[0.5], "use proba > 0.5 and campaign_value(y_test, ...)"
assert best_threshold == max(vals, key=vals.get), f"best is {max(vals, key=vals.get)}; values: {vals}"
print("Exercise 2 passed:", vals)

# %% tags=["solution"]
vals = {t: campaign_value(y_test, proba > t) for t in [0.1, 0.2, 0.3, 0.4, 0.5]}
value_at_050 = vals[0.5]
best_threshold = max(vals, key=vals.get)

# %% [markdown]
# ### Exercise 3: A regression baseline that respects the metric
# On the housing split (`hy_train`, `hy_test`) find the test **RMSE** of the training-mean predictor (`rmse_mean`) and the test **MAE** of the
# training-median predictor (`mae_median`). Compute them with NumPy, no `DummyRegressor`.

# %% tags=["starter"]
rmse_mean = ...
mae_median = ...

# %%
exp_rmse = float(np.sqrt(np.mean((hy_test - hy_train.mean()) ** 2)))
exp_mae = float(np.mean(np.abs(hy_test - hy_train.median())))
assert abs(rmse_mean - exp_rmse) < 1e-6, "RMSE = sqrt(mean((y_test - train_mean)**2))"
assert abs(mae_median - exp_mae) < 1e-6, "MAE = mean(abs(y_test - train_median))"
print(f"Exercise 3 passed: RMSE(mean)={rmse_mean:,.0f}  MAE(median)={mae_median:,.0f}")

# %% tags=["solution"]
rmse_mean = float(np.sqrt(np.mean((hy_test - hy_train.mean()) ** 2)))
mae_median = float(np.mean(np.abs(hy_test - hy_train.median())))

# %% [markdown]
# ### Exercise 4: A rule-of-thumb baseline
# Business people often already have a rule. Build one for churn: predict churn for **month-to-month** customers and no one else.
# Store its test accuracy in `rule_acc` and the number of churners it finds (true positives) in `rule_tp`, using the *same* `X_test`
# rows (`churn.loc[X_test.index, "contract"]`). *Hint: the contract column was lower-cased above.*

# %% tags=["starter"]
rule_acc = ...
rule_tp = ...

# %%
c = churn.loc[X_test.index, "contract"]
rp = (c == "month-to-month").astype(int)
assert abs(rule_acc - accuracy_score(y_test, rp)) < 1e-9, "predict 1 where contract == 'month-to-month', else 0"
assert rule_tp == int(((rp == 1) & (y_test == 1)).sum()), "rule_tp = predicted 1 and actually 1"
print(f"Exercise 4 passed: rule accuracy {rule_acc:.3f}, finds {rule_tp} of {int(y_test.sum())} churners")

# %% tags=["solution"]
c = churn.loc[X_test.index, "contract"]
rp = (c == "month-to-month").astype(int)
rule_acc = float(accuracy_score(y_test, rp))
rule_tp = int(((rp == 1) & (y_test == 1)).sum())

# %% [markdown]
# ### Exercise 5 (open-ended): Frame a new problem
# Pick one of: predicting `support_calls` (count), predicting `internet_service` (which is missing for many rows), or predicting `price` from
# housing. In a markdown cell write: the problem type, the exact target definition, one feature you would have to *exclude* because it would not
# exist at prediction time, the metric you would report, and the baseline you would beat.

# %% [markdown]
# ## Recap
# - Frame the problem first: type, target with horizon, unit, cost of each kind of error.
# - Tie the metric to money; the decision threshold follows from the cost ratio (`10/(40+10)` here, under *assumed* costs).
# - A baseline is your floor: majority class for classification, mean or median for regression, plus a business rule.
# - Workflow: frame, split, baseline, model, cross-validate, evaluate once on test, monitor.
#
# **Go deeper:** the scikit-learn user guide chapters "Metrics and scoring" and "Dummy estimators"
# (scikit-learn.org/stable/user_guide.html).
#
# **Next:** [Train/test split and data leakage](02_train_test_split_and_data_leakage.ipynb)

# %% [markdown]
# # Ensembles: Random Forests and Gradient Boosting
#
# **Why this matters.** On tabular business data (churn, credit, pricing, demand), tuned tree ensembles are the default "strong model" that more exotic methods
# have to beat. Interviewers expect you to explain *why* averaging many weak trees works, what differs between bagging and boosting, and what the main knobs do,
# not just to type `RandomForestClassifier()`.
#
# **You will learn**
# - why averaging high-variance trees helps (bagging), and build a bagged ensemble by hand
# - how boosting works, by writing a tiny gradient-boosting loop yourself
# - use random forests (incl. the out-of-bag score) and `HistGradientBoosting`, and see the learning-rate / number-of-trees trade-off
# - compare every model against the baselines honestly, and know when an ensemble is *not* worth it
#
# **Prerequisites:** [Decision trees and overfitting](05_decision_trees_and_overfitting.ipynb).
#
# > Data used: `datasets/churn.csv` (classification) and `datasets/housing.csv` (regression), synthetic.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

DATA = Path("../datasets")
churn = pd.read_csv(DATA / "churn.csv", keep_default_na=False, na_values=[""])
churn["contract"] = churn["contract"].str.lower()
churn["internet_service"] = churn["internet_service"].fillna("Unknown")
X = pd.get_dummies(
    churn[["tenure_months", "monthly_charges", "support_calls", "senior_citizen", "contract", "internet_service", "paperless_billing", "autopay"]],
    columns=["contract", "internet_service", "paperless_billing", "autopay"], dtype=float,
)
y = churn["churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
print(X_train.shape, X_test.shape)

# %% [markdown]
# ## 1. Bagging: average many noisy trees
#
# A single deep tree has low bias but high variance: its predictions swing with the training sample (notebook 05). If you average `B` models whose errors are
# *not perfectly correlated*, the noise partly cancels while the signal remains. **Bagging** ("bootstrap aggregating") manufactures such models: train each
# tree on a different bootstrap resample of the rows, then average their predictions. Build it by hand:

# %%
def bagged_probs(n_trees, min_leaf=5, seed=0):
    rng = np.random.default_rng(seed)
    probs = []
    for _ in range(n_trees):
        idx = rng.integers(0, len(X_train), len(X_train))
        t = DecisionTreeClassifier(min_samples_leaf=min_leaf, random_state=42).fit(X_train.iloc[idx], y_train.iloc[idx])
        probs.append(t.predict_proba(X_test)[:, 1])
    return np.array(probs)

P = bagged_probs(50)
single_auc = np.mean([roc_auc_score(y_test, p) for p in P])
bag_auc = roc_auc_score(y_test, P.mean(axis=0))
print(f"average AUC of one tree: {single_auc:.3f}   AUC of the average of 50 trees: {bag_auc:.3f}")

# %% [markdown]
# *What just happened?* Every individual tree is mediocre and noisy, but their *average* ranks customers noticeably better: that gain is pure variance reduction. A **random forest** adds one more
# trick to make the trees less correlated: at each split only a random subset of features (`max_features`) is considered, which stops every tree from copying the same dominant split.
# Gains flatten out as you add trees; more trees never overfit in the classic sense, they just cost time.

# %% [markdown]
# ## 2. Random forest in practice, and the out-of-bag score
#
# Each bootstrap sample leaves out about a third of the rows (roughly 36.8%, since `(1 - 1/n)^n -> 1/e`). Those **out-of-bag (OOB)** rows are a free validation set for that tree.
# Scoring every row using only the trees that did not see it gives a generalisation estimate without a separate split:

# %%
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, oob_score=True, random_state=42, n_jobs=1).fit(X_train, y_train)
oob_auc = roc_auc_score(y_train, rf.oob_decision_function_[:, 1])
test_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print(f"OOB accuracy {rf.oob_score_:.3f} | test accuracy {accuracy_score(y_test, rf.predict(X_test)):.3f}")
print(f"OOB AUC      {oob_auc:.3f} | test AUC      {test_auc:.3f}")

# %% [markdown]
# OOB and test scores land close together, so OOB is a cheap sanity check (not a replacement for proper cross-validation when you compare many models). Now see how the forest size matters:

# %%
rows = []
for n in [1, 5, 25, 100, 300]:
    m = RandomForestClassifier(n_estimators=n, min_samples_leaf=5, random_state=42, n_jobs=1).fit(X_train, y_train)
    rows.append({"n_estimators": n, "test AUC": roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])})
pd.DataFrame(rows).round(3)

# %% [markdown]
# A big jump from 1 tree to a few dozen, then diminishing returns. The other knobs that matter most: `min_samples_leaf` / `max_depth` (capacity), `max_features` (decorrelation;
# `"sqrt"` is the classification default), and `class_weight` for imbalance (notebook 12).

# %% [markdown]
# ## 3. Boosting: fix the mistakes sequentially
#
# Bagging averages independent *deep* trees to cut variance. **Boosting** goes the other way: it adds many *shallow* trees **one after another**, each trained to correct what the
# ensemble so far still gets wrong, which cuts bias. For squared error the recipe (gradient boosting) is short:
#
# 1. Start with a constant prediction (the mean).
# 2. Compute the **residuals** (actual minus current prediction).
# 3. Fit a small tree to the residuals.
# 4. Add `learning_rate * tree_prediction` to the ensemble. Repeat from 2.
#
# The learning rate shrinks each step so no single tree dominates. Write it from scratch on a toy curve:

# %%
rng = np.random.default_rng(42)
xs = np.sort(rng.uniform(0, 6, 200))
ys = np.sin(xs) + rng.normal(0, 0.3, xs.size)
Xg = xs.reshape(-1, 1)
xg_test = np.linspace(0, 6, 300).reshape(-1, 1)


def boost(X_, y_, n_rounds, lr=0.1, depth=2):
    pred = np.full(len(y_), y_.mean())
    trees = []
    for _ in range(n_rounds):
        resid = y_ - pred
        t = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(X_, resid)
        pred = pred + lr * t.predict(X_)
        trees.append(t)
    return y_.mean(), trees

def boost_predict(base, trees, X_, lr=0.1):
    return base + lr * sum(t.predict(X_) for t in trees)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
for ax, n in zip(axes, [1, 10, 200]):
    base, trees = boost(Xg, ys, n)
    ax.scatter(xs, ys, s=8, alpha=0.5)
    ax.plot(xg_test[:, 0], boost_predict(base, trees, xg_test), "r")
    ax.set(title=f"{n} boosting round(s)", xlabel="x")
axes[0].set_ylabel("y")
plt.tight_layout()
plt.show()

# %% [markdown]
# One round is a crude step function; ten rounds trace the curve; two hundred rounds fit the training points more and more closely, which is where noise-chasing begins. **Number of rounds is a capacity dial**, like tree depth.
# Quantify it: training error keeps falling, while the error against the *true* curve (sin x, known here because the data is synthetic) bottoms out and then creeps up.

# %%
rows = []
for n in [1, 10, 50, 200, 1000]:
    base, trees = boost(Xg, ys, n)
    rows.append({"rounds": n,
                 "train MSE": np.mean((ys - boost_predict(base, trees, Xg)) ** 2),
                 "MSE vs true curve": np.mean((np.sin(xg_test[:, 0]) - boost_predict(base, trees, xg_test)) ** 2)})
pd.DataFrame(rows).round(4)

# %% [markdown]
# To prove we wrote real gradient boosting, compare with scikit-learn's implementation using the same settings:

# %%
base, trees = boost(Xg, ys, 50, lr=0.1, depth=2)
mine = boost_predict(base, trees, xg_test)
ref = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=2, random_state=0).fit(Xg, ys).predict(xg_test)
print("max difference vs GradientBoostingRegressor:", float(np.abs(mine - ref).max()))

# %% [markdown]
# They agree (up to floating-point/tie-breaking noise). Real libraries add regularisation, subsampling and clever histogram-based split finding, but the loop above is the core idea.

# %% [markdown]
# ## 4. HistGradientBoosting and the learning-rate trade-off
#
# `HistGradientBoostingClassifier` is scikit-learn's fast implementation (it bins features into histograms, handles missing values natively, and supports early stopping).
# The two central knobs interact: **a smaller `learning_rate` needs more iterations** but usually generalises a bit better. Early stopping picks the number of iterations
# by watching a held-out validation fraction of the *training* data:

# %%
rows = []
for lr in [0.5, 0.1, 0.03]:
    m = HistGradientBoostingClassifier(learning_rate=lr, max_iter=500, early_stopping=True, validation_fraction=0.15,
                                       n_iter_no_change=15, random_state=42).fit(X_train, y_train)
    rows.append({"learning_rate": lr, "iterations used": m.n_iter_, "test AUC": roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])})
pd.DataFrame(rows).round(3)

# %% [markdown]
# *What just happened?* Each learning rate stopped when validation loss stopped improving. The large rate needed far fewer iterations; the small one needed many more.
# The test AUCs may differ only slightly: the trade-off is mostly *compute vs a small accuracy gain*. Main knobs in practice: `learning_rate`, `max_iter` (use early stopping),
# `max_leaf_nodes` / `max_depth`, `min_samples_leaf`, `l2_regularization`.
#
# **Bagging vs boosting in one table**
#
# | | Random forest (bagging) | Gradient boosting |
# |---|---|---|
# | Trees | deep-ish, independent, in parallel | shallow, sequential |
# | Reduces | variance | bias (and some variance via shrinkage) |
# | Overfitting with more trees | essentially no (plateaus) | yes, needs early stopping / small rate |
# | Tuning sensitivity | forgiving | more sensitive |
# | Noisy labels | robust | can chase noise |

# %% [markdown]
# ## 5. Compare everything against the baselines
#
# Never judge an ensemble alone. Put it next to the dummy baseline and the simple models on the same split. Churn first (metric: test ROC-AUC; 0.5 = no skill):

# %%
models = {
    "dummy (prior)": DummyClassifier(strategy="prior"),
    "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "tree depth 4": DecisionTreeClassifier(max_depth=4, random_state=42),
    "tree unpruned": DecisionTreeClassifier(random_state=42),
    "random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=1),
    "hist gradient boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, early_stopping=True, random_state=42),
}
res = {name: roc_auc_score(y_test, m.fit(X_train, y_train).predict_proba(X_test)[:, 1]) for name, m in models.items()}
pd.Series(res).sort_values(ascending=False).round(3).to_frame("test ROC-AUC")

# %% [markdown]
# Read this honestly. The unpruned tree is far *worse* than a short tree, and the ensembles clearly beat both single trees: that is the variance reduction working.
# But look at the top of the table: **logistic regression beats both ensembles on this dataset**. That is a real and common outcome, not a bug. A likely reason (a hypothesis you could test by adding
# interactions to the logistic model) is that the planted churn signal is close to additive on the log-odds scale, which a linear model represents exactly and
# trees only approximate with steps. Ensembles shine when there are strong interactions and non-linearities; when there are not, they add cost and lose interpretability for nothing.
# One split is also a noisy referee (notebook 02), so notebook 07 turns this comparison into a proper one with repeated cross-validation.
#
# Now regression. Housing, with the absurd-price rule from notebook 03 (price per sq ft outside 20-500), metric test MAE (lower is better). The forest and the boosted model read missing values natively;
# the linear model, the single tree and the dummy get a median imputer:

# %%
h = pd.read_csv(DATA / "housing.csv")
h = h[(h["price"] / h["area_sqft"]).between(20, 500)]
hX = pd.get_dummies(h.drop(columns="price"), columns=["neighborhood"], dtype=float)
hy = h["price"]
hX_train, hX_test, hy_train, hy_test = train_test_split(hX, hy, test_size=0.25, random_state=42)

reg_models = {
    "dummy (median)": DummyRegressor(strategy="median"),
    "linear regression": make_pipeline(SimpleImputer(strategy="median"), LinearRegression()),
    "tree (leaf>=20)": DecisionTreeRegressor(min_samples_leaf=20, random_state=42),
    "random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=1),
    "hist gradient boosting": HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, random_state=42),
}
maes = {}
for name, m in reg_models.items():
    if name.startswith(("tree", "dummy")):
        m = make_pipeline(SimpleImputer(strategy="median"), m)   # single trees + dummy: impute for simplicity
    maes[name] = mean_absolute_error(hy_test, m.fit(hX_train, hy_train).predict(hX_test))
pd.Series(maes).sort_values().round(0).to_frame("test MAE ($)")

# %% [markdown]
# Again the ensembles do not run away with it. Boosting is only a little better than the straight-line model, and the random forest is *worse* than it: price here rises smoothly with area and
# quality, and a forest approximates a smooth ramp with steps built from finite data. Boosting also overfits if you give it too many rounds (try `max_iter=1000` with no early stopping and watch the test MAE rise).
# **The right conclusion comes from measuring on your data**, not from folklore that "boosting always wins".

# %% [markdown]
# ## Common mistakes
#
# 1. **Reporting only the ensemble's score** with no simple baseline next to it.
# 2. **Tuning `n_estimators` of a boosted model on the test set.** Use early stopping on a validation split.
# 3. **Assuming a random forest's `feature_importances_` are reliable** (same bias as a single tree; use permutation importance).
# 4. **Treating `predict_proba` from forests/boosting as calibrated.** Check a reliability table; calibrate if thresholds matter.
# 5. **Forgetting cost:** a 500-tree forest is bigger and slower to serve than a logistic regression; the gain has to be worth it.
# 6. **Expecting forests to extrapolate**, like single trees they return values seen in training.

# %%
# Mistake 4 in code: forest probabilities are averages of leaf frequencies; check whether "0.2" really means 20%
rf_p = models["random forest"].predict_proba(X_test)[:, 1]
cal = pd.DataFrame({"p": rf_p, "y": y_test.to_numpy()})
cal["bin"] = pd.qcut(cal["p"], 5, duplicates="drop")
cal.groupby("bin", observed=True).agg(mean_predicted=("p", "mean"), actual_rate=("y", "mean")).round(3)

# %% [markdown]
# ## Interview corner
#
# **1. Bagging vs boosting?**
# <details><summary>Answer</summary>
# Bagging trains models independently on bootstrap samples and averages them (mainly reduces variance; random forests add feature subsampling to decorrelate trees).
# Boosting trains models sequentially, each one focusing on the errors of the current ensemble (mainly reduces bias), using shallow trees and a learning rate.
# </details>
#
# **2. Why does a random forest choose a random subset of features at each split?**
# <details><summary>Answer</summary>
# To decorrelate the trees. If one feature dominates, every bootstrap tree would split on it first and look alike; averaging correlated models removes little variance.
# Random feature subsets force diversity, so the average has lower variance.
# </details>
#
# **3. What is the out-of-bag score?**
# <details><summary>Answer</summary>
# Each bootstrap sample omits roughly 36.8% of rows. Predicting each row using only trees that did not train on it gives an honest generalisation estimate for free, without a separate validation set.
# </details>
#
# **4. Your gradient-boosting model overfits. What do you change?**
# <details><summary>Answer</summary>
# Lower the learning rate and use early stopping on a validation set, reduce tree depth / leaf count, raise <code>min_samples_leaf</code> and L2 regularisation, subsample rows/columns (in libraries that support it),
# and check for leakage or label noise. More data helps more than anything.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Does bagging really help?
# Using `bagged_probs(n_trees, seed=0)` from section 1 (which uses `min_leaf=5`), compute `auc_1` = test ROC-AUC of a **single** bagged tree (the first row of `bagged_probs(1)`)
# and `auc_25` = test ROC-AUC of the **average** of 25 trees (`bagged_probs(25).mean(axis=0)`). Store `gain = auc_25 - auc_1`.

# %% tags=["starter"]
auc_1 = ...
auc_25 = ...
gain = ...

# %%
_p1 = bagged_probs(1)[0]
_p25 = bagged_probs(25).mean(axis=0)
assert abs(auc_1 - roc_auc_score(y_test, _p1)) < 1e-9, "auc_1 uses the single tree from bagged_probs(1)"
assert abs(auc_25 - roc_auc_score(y_test, _p25)) < 1e-9, "auc_25 averages the probabilities of 25 trees, then scores"
assert abs(gain - (auc_25 - auc_1)) < 1e-9 and gain > 0, "averaging should beat a single tree"
print(f"Exercise 1 passed: {auc_1:.3f} -> {auc_25:.3f} (gain {gain:.3f})")

# %% tags=["solution"]
auc_1 = float(roc_auc_score(y_test, bagged_probs(1)[0]))
auc_25 = float(roc_auc_score(y_test, bagged_probs(25).mean(axis=0)))
gain = auc_25 - auc_1

# %% [markdown]
# ### Exercise 2: Trust the OOB score?
# Fit `RandomForestClassifier(n_estimators=200, min_samples_leaf=5, oob_score=True, random_state=42, n_jobs=1)` on `X_train`, `y_train`. Store `oob_acc` (its `oob_score_`), `test_acc` (accuracy on `X_test`),
# and `close_enough = True` if the two differ by less than 0.03, else `False`.

# %% tags=["starter"]
oob_acc = ...
test_acc = ...
close_enough = ...

# %%
_m = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, oob_score=True, random_state=42, n_jobs=1).fit(X_train, y_train)
assert abs(oob_acc - _m.oob_score_) < 1e-9, "oob_acc is model.oob_score_"
assert abs(test_acc - accuracy_score(y_test, _m.predict(X_test))) < 1e-9, "test_acc is accuracy on X_test"
assert close_enough == bool(abs(_m.oob_score_ - accuracy_score(y_test, _m.predict(X_test))) < 0.03), "compare |oob - test| with 0.03"
print(f"Exercise 2 passed: OOB {oob_acc:.3f} vs test {test_acc:.3f}")

# %% tags=["solution"]
_rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, oob_score=True, random_state=42, n_jobs=1).fit(X_train, y_train)
oob_acc = float(_rf.oob_score_)
test_acc = float(accuracy_score(y_test, _rf.predict(X_test)))
close_enough = bool(abs(oob_acc - test_acc) < 0.03)

# %% [markdown]
# ### Exercise 3: Learning rate versus iterations
# For each `lr` in `[0.3, 0.1, 0.03]` fit `HistGradientBoostingClassifier(learning_rate=lr, max_iter=500, early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, random_state=42)`
# on `X_train`, `y_train`. Store the dict `{lr: n_iter_}` in `iters`, and in `slowest_lr` the learning rate that needed the most iterations.

# %% tags=["starter"]
iters = ...
slowest_lr = ...

# %%
_exp = {}
for _lr in [0.3, 0.1, 0.03]:
    _exp[_lr] = HistGradientBoostingClassifier(learning_rate=_lr, max_iter=500, early_stopping=True, validation_fraction=0.15,
                                               n_iter_no_change=15, random_state=42).fit(X_train, y_train).n_iter_
assert iters == _exp, f"expected {_exp}"
assert slowest_lr == max(_exp, key=_exp.get), "the learning rate with the largest n_iter_"
print("Exercise 3 passed:", iters)

# %% tags=["solution"]
iters = {}
for lr in [0.3, 0.1, 0.03]:
    iters[lr] = HistGradientBoostingClassifier(learning_rate=lr, max_iter=500, early_stopping=True, validation_fraction=0.15,
                                               n_iter_no_change=15, random_state=42).fit(X_train, y_train).n_iter_
slowest_lr = max(iters, key=iters.get)

# %% [markdown]
# ### Exercise 4: Is boosting worth it for house prices?
# Using the results in `maes` from section 5, compute `hgb_vs_linear_pct`: how many percent lower (positive number; negative if worse) the histogram-boosting MAE is compared with
# linear regression, i.e. `100 * (mae_linear - mae_hgb) / mae_linear`. Then store `worth_it = True` if that improvement is above 10%, else `False`.

# %% tags=["starter"]
hgb_vs_linear_pct = ...
worth_it = ...

# %%
_exp = 100 * (maes["linear regression"] - maes["hist gradient boosting"]) / maes["linear regression"]
assert abs(hgb_vs_linear_pct - _exp) < 1e-9, "100 * (mae_linear - mae_hgb) / mae_linear"
assert worth_it == bool(_exp > 10), "True only if the improvement exceeds 10%"
print(f"Exercise 4 passed: boosting is {hgb_vs_linear_pct:.1f}% better than linear in MAE -> worth_it={worth_it}")

# %% tags=["solution"]
hgb_vs_linear_pct = 100 * (maes["linear regression"] - maes["hist gradient boosting"]) / maes["linear regression"]
worth_it = bool(hgb_vs_linear_pct > 10)

# %% [markdown]
# ### Exercise 5 (open-ended): Stress the boosting model
# Flip 15% of the **training** labels at random (`rng = np.random.default_rng(0)`, flip `y_train` only) and refit the logistic regression, the random forest and the histogram boosting model from section 5.
# Compare their *test* AUCs on the clean test set with their clean-label AUCs. Which model degrades least, and does it match the "noisy labels" row of the bagging-vs-boosting table? Write your observation in a markdown cell.

# %% [markdown]
# ## Recap
# - Bagging = average decorrelated deep trees (variance down); random forests add feature subsampling and give a free OOB estimate.
# - Boosting = add shallow trees sequentially to fit residuals with a learning rate (bias down); `n_estimators` and `learning_rate` trade off, and early stopping picks the number of rounds.
# - Always compare against a dummy and a simple model: here logistic regression beat both ensembles on churn, and boosting only slightly beat linear regression on housing. Measure, do not assume.
# - Costs of ensembles: size, latency, interpretability, uncalibrated probabilities.
#
# **Go deeper:** scikit-learn user guide "Ensembles: Gradient boosting, random forests, bagging, voting, stacking" (scikit-learn.org/stable/modules/ensemble.html).
#
# **Next:** [Cross-validation and model evaluation](07_cross_validation_and_model_evaluation.ipynb)

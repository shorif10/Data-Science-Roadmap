# %% [markdown]
# # Decision Trees and Overfitting
#
# **Why this matters.** A decision tree is the only mainstream model a non-technical stakeholder can read out loud ("if the contract is month-to-month and
# the customer has called support 4+ times, ..."). It also needs almost no preprocessing and captures non-linear effects and interactions automatically. And it is
# the perfect specimen for learning **overfitting**, because one dial (depth) takes it from underfitting to memorising. Random forests and gradient boosting
# (next notebook) are built from trees, so this is the foundation for the models that win most tabular-data problems.
#
# **You will learn**
# - how a tree chooses each split (Gini impurity / entropy), by computing it yourself
# - read a fitted tree as rules, and see how depth trades bias against variance
# - control overfitting with `max_depth`, `min_samples_leaf` and cost-complexity pruning
# - see why single trees are unstable, and what trees cannot do (extrapolate)
#
# **Prerequisites:** [Logistic regression and metrics](04_logistic_regression_and_classification_metrics.ipynb).
#
# > Data used: `datasets/churn.csv` (classification) and `datasets/housing.csv` (regression), both synthetic.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree

DATA = Path("../datasets")
churn = pd.read_csv(DATA / "churn.csv", keep_default_na=False, na_values=[""])
churn["contract"] = churn["contract"].str.lower()
churn["internet_service"] = churn["internet_service"].fillna("Unknown")
X = pd.get_dummies(
    churn[["tenure_months", "monthly_charges", "support_calls", "senior_citizen", "contract", "internet_service", "paperless_billing", "autopay"]],
    columns=["contract", "internet_service", "paperless_billing", "autopay"],
    dtype=float,
)
y = churn["churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
print(X_train.shape, "features:", list(X.columns))

# %% [markdown]
# ## 1. A tree is a sequence of yes/no questions
#
# Think of "20 questions". Each internal node asks about one feature ("tenure <= 12?"), sends rows left or right, and repeats. A leaf holds the rows that reached it, and
# the prediction is their majority class (or class proportions, which gives `predict_proba`). Fit a shallow tree and print it as rules:

# %%
tree3 = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
print(export_text(tree3, feature_names=list(X.columns), show_weights=True))

# %% [markdown]
# `value = [stayed, churned]` counts training rows in each node. Each line is a rule a human can check: the leaf at the end of a path gives the
# stayed/churned counts for customers who satisfy all the conditions on the way down. That readability is a real advantage over boosted ensembles and neural nets.

# %% [markdown]
# ## 2. How a split is chosen: impurity
#
# A node is **pure** when all its rows have the same label. **Gini impurity** of a node with class proportions `p_k` is `1 - sum(p_k^2)`: 0 when pure, 0.5 for a
# 50/50 two-class node. **Entropy** is `-sum(p_k * log2(p_k))`: 0 when pure, 1 bit for 50/50. For every candidate (feature, threshold) the tree computes the
# *weighted average impurity of the two children* and picks the split that lowers impurity the most. It is **greedy**: it never looks ahead, only picks the best split *now*.

# %%
def gini(labels):
    p = np.bincount(labels, minlength=2) / len(labels)
    return 1 - np.sum(p**2)

def entropy(labels):
    p = np.bincount(labels, minlength=2) / len(labels)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

yt = y_train.to_numpy()
print(f"root node: churn rate {yt.mean():.3f}, Gini {gini(yt):.3f}, entropy {entropy(yt):.3f}")

# %% [markdown]
# Now search every possible threshold on **one** feature, `tenure_months`, and pick the best by weighted Gini. Then ask scikit-learn for a depth-1 tree (a "stump") on
# the same single feature and compare:

# %%
ten = X_train["tenure_months"].to_numpy()
values = np.unique(ten)
best = (None, np.inf)
for t in values[:-1]:                        # split rule: tenure <= t goes left
    left, right = yt[ten <= t], yt[ten > t]
    weighted = (len(left) * gini(left) + len(right) * gini(right)) / len(yt)
    if weighted < best[1]:
        best = (t, weighted)
t_best, g_best = best
nxt = values[values > t_best][0]

stump = DecisionTreeClassifier(max_depth=1, random_state=42).fit(X_train[["tenure_months"]], y_train)
print(f"by hand: best threshold tenure <= {t_best} (weighted Gini {g_best:.4f}); sklearn puts the boundary at {stump.tree_.threshold[0]}")
print("sklearn threshold is the midpoint of the two neighbouring values:", stump.tree_.threshold[0] == (t_best + nxt) / 2)

# %% [markdown]
# *What just happened?* Same answer: the tree tries every boundary between consecutive sorted values, scores it by impurity and keeps the winner
# (scikit-learn reports the midpoint between the two values). The drop from the parent's Gini to the weighted child Gini is that split's **impurity decrease**,
# also called information gain when using entropy. The tree then repeats the search inside each child, until it hits a stopping rule or every leaf is pure.
# In practice Gini and entropy give very similar trees; Gini is the default and slightly cheaper.

# %% [markdown]
# ## 3. Depth: from underfitting to memorising
#
# With no stopping rule a tree keeps splitting until every leaf is pure, which means it memorises the training set, noise included. Plot the training and cross-validated
# score as depth grows (`validation_curve` runs 5-fold CV on the training set only; notebook 07 covers CV properly):

# %%
depths = np.arange(1, 21)
tr, va = validation_curve(
    DecisionTreeClassifier(random_state=42), X_train, y_train,
    param_name="max_depth", param_range=depths, cv=5, scoring="roc_auc",
)
plt.figure(figsize=(7, 3.6))
plt.plot(depths, tr.mean(1), "o-", label="train")
plt.plot(depths, va.mean(1), "o-", label="validation (5-fold mean)")
plt.fill_between(depths, va.mean(1) - va.std(1), va.mean(1) + va.std(1), alpha=0.2)
plt.title("Decision tree: ROC-AUC vs maximum depth")
plt.xlabel("max_depth")
plt.ylabel("ROC-AUC")
plt.legend()
plt.show()
best_depth_cv = int(depths[va.mean(1).argmax()])
print("best depth by validation AUC:", best_depth_cv, "| train AUC there:", round(tr.mean(1)[best_depth_cv - 1], 3), "| validation AUC:", round(va.mean(1).max(), 3))
print("depth 20            : train AUC", round(tr.mean(1)[-1], 3), "| validation AUC", round(va.mean(1)[-1], 3))

# %% [markdown]
# Read the curves. Small depth: both scores are low, the model is too simple (**underfitting**, high bias). As depth grows the training score keeps rising toward 1.0, but
# the validation score peaks and then *falls*: the tree is fitting noise (**overfitting**, high variance). The gap between the curves is the signature. The right depth is
# where validation peaks, not where training is best.

# %% [markdown]
# ## 4. Other ways to hold a tree back
#
# `max_depth` is blunt. More targeted controls:
#
# - `min_samples_leaf`: every leaf must contain at least this many rows, so no rule is based on a handful of customers. Often the most effective single knob.
# - `min_samples_split`, `max_leaf_nodes`: similar capacity limits.
# - `ccp_alpha`: **cost-complexity pruning**. Grow a big tree, then cut branches whose impurity gain does not justify their extra leaves.

# %%
rows = []
for leaf in [1, 5, 20, 50, 100]:
    m = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=42).fit(X_train, y_train)
    rows.append({"min_samples_leaf": leaf, "leaves": m.get_n_leaves(), "depth": m.get_depth(),
                 "train AUC": roc_auc_score(y_train, m.predict_proba(X_train)[:, 1]),
                 "test AUC": roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])})
pd.DataFrame(rows).round(3)

# %% [markdown]
# Larger leaves mean fewer rules, a smaller train/test gap, and typically a better test score up to a point. (Here we peek at the test set to *show* the pattern; when choosing the
# value in real work, use cross-validation.) Also, note the probabilities: with `min_samples_leaf=1`, leaves are pure so probabilities are only 0 or 1: terrible for ranking and for threshold decisions.

# %% [markdown]
# ## 5. Instability: small data changes, different tree
#
# Trees are **high-variance** learners. Because splits are greedy, a small change in the data can change a split and therefore everything below it. Fit 30
# depth-4 trees, each on a different bootstrap sample of the training rows (sampling with replacement), and see how similar they are:

# %%
rng = np.random.default_rng(42)
roots, used, preds = [], [], []
for _ in range(30):
    idx = rng.integers(0, len(X_train), len(X_train))
    m = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train.iloc[idx], y_train.iloc[idx])
    roots.append(X.columns[m.tree_.feature[0]])
    used.append(tuple(sorted(set(X.columns[m.tree_.feature[m.tree_.feature >= 0]]))))
    preds.append(m.predict_proba(X_test)[:, 1])
print("root feature across 30 bootstrap trees:", pd.Series(roots).value_counts().to_dict())
print("distinct sets of features used anywhere in the tree:", len(set(used)), "out of 30 trees")
preds = np.array(preds)
print("std of predicted probability across trees, averaged over test rows:", round(float(preds.std(axis=0).mean()), 3))
print("share of test rows where some trees say >0.5 and others <=0.5:", round(float(((preds > 0.5).any(0) & (preds <= 0.5).any(0)).mean()), 3))

# %% [markdown]
# *What just happened?* The very top split is stable when one feature dominates (the contract type), but the deeper structure and the
# predictions are not: the trees use different feature combinations, and many customers get contradictory verdicts from different trees. Averaging many such trees cancels
# the noise; that is exactly the idea behind **bagging and random forests** in the next notebook.

# %% [markdown]
# ## 6. Two more things to know
#
# **Trees cannot extrapolate.** Predictions are constants per leaf. Fit the housing data (with the same "absurd price" rule from notebook 03: price per sq ft outside 20-500) and ask about a huge house:

# %%
hs = pd.read_csv(DATA / "housing.csv")
hs = hs[(hs["price"] / hs["area_sqft"]).between(20, 500)]
hcols = ["area_sqft", "bathrooms", "quality_score"]
hs = hs.dropna(subset=hcols)
reg = DecisionTreeRegressor(min_samples_leaf=20, random_state=42).fit(hs[hcols], hs["price"])
probe = pd.DataFrame({"area_sqft": [2500, 4900, 10_000, 40_000], "bathrooms": 2.0, "quality_score": 6.0})
probe["tree_prediction"] = reg.predict(probe[hcols]).round(0)
probe

# %% [markdown]
# Beyond the largest area seen in training the tree keeps returning the same value (a linear model would keep climbing, as in notebook 03). That is sometimes *safer* than a
# nonsense extrapolation, but it means trees are poor for trends outside the training range.
#
# **Impurity-based feature importance is biased.** `feature_importances_` totals impurity decreases per feature, and it favours continuous, high-cardinality features because
# they offer more split points. A pure-noise numeric column gets non-zero importance:

# %%
Xn = X_train.copy()
Xn["random_noise"] = np.random.default_rng(0).normal(size=len(Xn))
mn = DecisionTreeClassifier(min_samples_leaf=20, random_state=42).fit(Xn, y_train)
imp = pd.Series(mn.feature_importances_, index=Xn.columns).sort_values(ascending=False)
imp.head(6).round(3).to_frame("importance")

# %% [markdown]
# Compare the noise column with the real features above: it is not zero, and here it even outranks a genuine feature. Treat these as a rough guide only; permutation importance on held-out data (notebook 13) is more trustworthy.

# %% [markdown]
# ## Common mistakes
#
# 1. **Leaving trees unpruned** and reporting a great training score.
# 2. **Using an unconstrained tree's `predict_proba`** as a probability: pure leaves give 0/1.
# 3. **Standardising features "for the tree"**: it is pointless (splits are order-based), harmless but a sign the pipeline is cargo-culted.
# 4. **Trusting `feature_importances_` blindly** (bias toward continuous features; splits credit among correlated features arbitrarily).
# 5. **Expecting extrapolation** or smooth relationships: trees give step functions.
# 6. **Comparing depths on the test set.** Use validation data.

# %%
# Mistake 3: monotonic rescaling of a feature does not change the tree's predictions at all
sc = X_train.copy(); sc["monthly_charges"] = np.log1p(sc["monthly_charges"])
sc_test = X_test.copy(); sc_test["monthly_charges"] = np.log1p(sc_test["monthly_charges"])
a = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train).predict_proba(X_test)
b = DecisionTreeClassifier(max_depth=4, random_state=42).fit(sc, y_train).predict_proba(sc_test)
print("identical predictions after a monotonic transform:", bool(np.allclose(a, b)))

# %% [markdown]
# ## Interview corner
#
# **1. How does a decision tree decide where to split?**
# <details><summary>Answer</summary>
# For each feature and candidate threshold it computes the impurity (Gini or entropy) of the resulting children, weighted by their size, and picks the split with the largest
# decrease from the parent's impurity. It is greedy and recursive; it stops on depth/leaf-size limits or when nodes are pure.
# </details>
#
# **2. Why do decision trees overfit, and how do you prevent it?**
# <details><summary>Answer</summary>
# Unconstrained, a tree can isolate every training row in its own leaf (zero bias, high variance). Limit capacity (<code>max_depth</code>, <code>min_samples_leaf</code>, <code>max_leaf_nodes</code>),
# prune (<code>ccp_alpha</code>), tune by cross-validation, or average many trees (bagging/forests).
# </details>
#
# **3. Do trees need feature scaling or one-hot encoding?**
# <details><summary>Answer</summary>
# No scaling (splits depend on order only). Scikit-learn's trees need numeric inputs, so categoricals must be encoded, and one-hot on a high-cardinality feature dilutes it across many
# sparse columns. Missing values are supported natively by recent scikit-learn trees, but check the docs for your version and criterion.
# </details>
#
# **4. What is the bias-variance trade-off, using trees as the example?**
# <details><summary>Answer</summary>
# Shallow trees are simple: high bias (systematic error), low variance (stable under resampling). Deep trees are flexible: low bias but high variance (different data, different tree).
# Total test error is roughly bias squared plus variance plus irreducible noise, so the best depth is a compromise; ensembles reduce variance without adding much bias.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: One split by hand
# Consider splitting the training rows on "contract is month-to-month" (that is, both `contract_one year` and `contract_two year` are 0). Using the `gini` function above,
# store in `parent_gini` the Gini of all of `y_train`, in `child_gini` the *weighted* Gini of the two children, and in `impurity_decrease` the difference (parent minus children).

# %% tags=["starter"]
parent_gini = ...
child_gini = ...
impurity_decrease = ...

# %%
_mtm = ((X_train["contract_one year"] == 0) & (X_train["contract_two year"] == 0)).to_numpy()
_yt = y_train.to_numpy()
_exp_child = (_mtm.sum() * gini(_yt[_mtm]) + (~_mtm).sum() * gini(_yt[~_mtm])) / len(_yt)
assert abs(parent_gini - gini(_yt)) < 1e-9, "parent_gini = gini(y_train.to_numpy())"
assert abs(child_gini - _exp_child) < 1e-9, "weight each child's Gini by its share of the rows"
assert abs(impurity_decrease - (gini(_yt) - _exp_child)) < 1e-9 and impurity_decrease > 0, "parent - weighted children (positive means a useful split)"
print(f"Exercise 1 passed: {parent_gini:.4f} -> {child_gini:.4f} (decrease {impurity_decrease:.4f})")

# %% tags=["solution"]
_m = ((X_train["contract_one year"] == 0) & (X_train["contract_two year"] == 0)).to_numpy()
_y = y_train.to_numpy()
parent_gini = gini(_y)
child_gini = (_m.sum() * gini(_y[_m]) + (~_m).sum() * gini(_y[~_m])) / len(_y)
impurity_decrease = parent_gini - child_gini

# %% [markdown]
# ### Exercise 2: Pick a simpler depth honestly
# Using `va` (the 5-fold validation AUC array from section 3, one row per depth in `depths`), apply a **parsimony rule**: find the **smallest** depth whose mean validation AUC is within
# **0.03** of the best mean validation AUC, and store it in `simple_depth`. (A simpler tree that is statistically indistinguishable from the best is easier to explain and less likely to be overfit.)
# Then fit that tree on all of `X_train` and store its **test** AUC in `test_auc`. Do not just return `best_depth_cv`: the check verifies that your rule picked a *different* (shallower) depth.
# *Hint: `np.flatnonzero(mask)[0]` gives the first True index.*

# %% tags=["starter"]
simple_depth = ...
test_auc = ...

# %%
_mv = va.mean(1)
_exp = int(depths[np.flatnonzero(_mv >= _mv.max() - 0.03)[0]])
assert simple_depth != best_depth_cv, "that is the depth with the best validation AUC; the rule asks for the SMALLEST depth within 0.03 of it"
assert simple_depth == _exp, f"smallest depth whose mean validation AUC >= best - 0.03; expected {_exp}"
assert _exp < best_depth_cv, "the parsimony rule should pick a shallower tree here"
_m = DecisionTreeClassifier(max_depth=_exp, random_state=42).fit(X_train, y_train)
assert abs(test_auc - roc_auc_score(y_test, _m.predict_proba(X_test)[:, 1])) < 1e-9, "fit on X_train with random_state=42 and score AUC on X_test"
print(f"Exercise 2 passed: depth {simple_depth} (best was {best_depth_cv}); test AUC {test_auc:.3f}")

# %% tags=["solution"]
_mv = va.mean(1)
simple_depth = int(depths[np.flatnonzero(_mv >= _mv.max() - 0.03)[0]])
_m = DecisionTreeClassifier(max_depth=simple_depth, random_state=42).fit(X_train, y_train)
test_auc = float(roc_auc_score(y_test, _m.predict_proba(X_test)[:, 1]))

# %% [markdown]
# ### Exercise 3: Pruning path
# Fit `DecisionTreeClassifier(ccp_alpha=0.002, random_state=42)` on `X_train`. Store the number of leaves in `n_leaves_pruned`, and the number of leaves of the **unpruned**
# tree (`ccp_alpha=0`) in `n_leaves_full`. *Hint: `.get_n_leaves()`.*

# %% tags=["starter"]
n_leaves_pruned = ...
n_leaves_full = ...

# %%
_a = DecisionTreeClassifier(ccp_alpha=0.002, random_state=42).fit(X_train, y_train).get_n_leaves()
_b = DecisionTreeClassifier(random_state=42).fit(X_train, y_train).get_n_leaves()
assert n_leaves_pruned == _a and n_leaves_full == _b, f"expected {_a} pruned and {_b} full leaves"
assert n_leaves_pruned < n_leaves_full, "pruning should remove leaves"
print(f"Exercise 3 passed: {n_leaves_full} leaves -> {n_leaves_pruned} after pruning")

# %% tags=["solution"]
n_leaves_pruned = DecisionTreeClassifier(ccp_alpha=0.002, random_state=42).fit(X_train, y_train).get_n_leaves()
n_leaves_full = DecisionTreeClassifier(random_state=42).fit(X_train, y_train).get_n_leaves()

# %% [markdown]
# ### Exercise 4: Quantify instability
# Repeat the bootstrap experiment of section 5 but with **depth-3** trees: `rng = np.random.default_rng(0)`, 20 trees, each fit on `idx = rng.integers(0, len(X_train), len(X_train))`
# (`random_state=42` for the tree). For each tree collect the *set of feature indices it splits on* (`m.tree_.feature`, ignoring the leaf marker -2). Store in `n_distinct_sets` the number of distinct
# sets seen across the 20 trees and in `top_share` the fraction of trees that use the most common set.

# %% tags=["starter"]
n_distinct_sets = ...
top_share = ...

# %%
_r = np.random.default_rng(0)
_sets = []
for _ in range(20):
    _i = _r.integers(0, len(X_train), len(X_train))
    _f = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train.iloc[_i], y_train.iloc[_i]).tree_.feature
    _sets.append(tuple(sorted(set(_f[_f >= 0]))))
_vc = pd.Series([str(t) for t in _sets]).value_counts()
assert n_distinct_sets == len(_vc), f"number of distinct feature sets; expected {len(_vc)}"
assert abs(top_share - _vc.iloc[0] / 20) < 1e-9, "fraction of trees using the most common feature set"
print(f"Exercise 4 passed: {n_distinct_sets} distinct feature sets in 20 trees; the most common covers {top_share:.0%}")

# %% tags=["solution"]
_rng = np.random.default_rng(0)
_used = []
for _ in range(20):
    _idx = _rng.integers(0, len(X_train), len(X_train))
    _ft = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train.iloc[_idx], y_train.iloc[_idx]).tree_.feature
    _used.append(str(tuple(sorted(set(_ft[_ft >= 0])))))
_counts = pd.Series(_used).value_counts()
n_distinct_sets = int(len(_counts))
top_share = float(_counts.iloc[0] / 20)

# %% [markdown]
# ### Exercise 5 (open-ended): Read a tree as a business rule
# Fit `DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=42)` and print it with `export_text`. Pick the leaf with the **highest** churn rate and rewrite its path as one
# sentence a retention manager could act on ("Customers on ... with ... churn about X% of the time; there are N of them"). Mention one limitation of trusting that rule.

# %% [markdown]
# ## Recap
# - A tree = greedy sequence of splits chosen to reduce Gini/entropy; you can reproduce the choice by hand.
# - Depth is the bias-variance dial: training score always rises, validation score peaks then falls.
# - Control capacity with `min_samples_leaf`, `max_depth`, `ccp_alpha`; select by cross-validation, not by the test set.
# - Single trees are unstable and can't extrapolate; impurity importances are biased. Averaging many trees fixes the variance: next notebook.
#
# **Go deeper:** scikit-learn user guide "Decision Trees" (scikit-learn.org/stable/modules/tree.html), including the section on minimal cost-complexity pruning.
#
# **Next:** [Ensembles: random forests and gradient boosting](06_ensembles_random_forest_and_gradient_boosting.ipynb)

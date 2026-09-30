# %% [markdown]
# # Hyperparameter Tuning: Grid, Random and Successive Halving
#
# **Why this matters.** Every model has knobs (tree depth, learning rate, regularisation strength) that the
# data cannot set for you. On the job you will be asked "did you tune it?" and, more importantly, "how do you
# know the tuned score is real?". Tuning is cheap to start and easy to do wrong: the more configurations you
# try, the more your best score is flattered by luck.
#
# **You will learn**
# - tell parameters (learned) from hyperparameters (chosen), and name them inside a `Pipeline`
# - run `GridSearchCV`, `RandomizedSearchCV` and `HalvingRandomSearchCV` and read `cv_results_`
# - count fits so you can budget a search before you launch it
# - see with your own eyes how over-tuning inflates the validation score, and fix the estimate with nested CV
#
# **Prerequisites:** [Cross-validation](07_cross_validation_and_model_evaluation.ipynb) and
# [Feature engineering and pipelines](08_feature_engineering_and_pipelines.ipynb).
#
# > Data used: `datasets/churn.csv` (synthetic telecom customers, `churned` is the target) plus a pure-noise
# > table we generate ourselves. See `docs/DATASETS.md`.

# %%
import os

# Small tables + many CPU threads = mostly thread overhead. Capping threads keeps timings sane and comparable.
os.environ.setdefault("OMP_NUM_THREADS", "2")

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401  (required to import Halving*)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    HalvingRandomSearchCV,
    ParameterGrid,
    ParameterSampler,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA = Path("../datasets")
df = pd.read_csv(DATA / "churn.csv", keep_default_na=False, na_values=[""])  # "None" is a real category, not missing
df["contract"] = df["contract"].str.lower().str.strip()  # fix the planted casing bug ('One year' vs 'one year')
df["internet_service"] = df["internet_service"].fillna("Unknown")  # the few genuinely blank values get their own level

X = df.drop(columns=["customer_id", "churned"])
y = df["churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("train:", X_train.shape, " test:", X_test.shape, " (the test set stays sealed until the very end)")

# %% [markdown]
# ## 1. Parameters vs hyperparameters
#
# A **parameter** is something the model *learns* from data (a regression coefficient, a split threshold).
# A **hyperparameter** is something *you* set before training that controls how learning happens
# (`C` in logistic regression, `max_depth` in a tree, `learning_rate` in boosting). The data cannot tell you the
# right value directly, so we **try several and compare them with cross-validation**.
#
# We will reuse one preprocessing recipe throughout: impute + scale numbers, one-hot encode categories.

# %%
num_cols = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "senior_citizen"]
cat_cols = ["contract", "internet_service", "paperless_billing", "autopay"]


def make_preprocessor():
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )


logreg = Pipeline([("pre", make_preprocessor()), ("clf", LogisticRegression(max_iter=1000))])
print(list(logreg.get_params().keys())[:6], "...")
print("tunable names for the classifier:", [k for k in logreg.get_params() if k.startswith("clf__")][:5], "...")

# %% [markdown]
# *What just happened?* Inside a `Pipeline`, every hyperparameter has a full name: `<step>__<parameter>`
# (double underscore). `clf__C` means "the `C` of the step called `clf`". You can even tune preprocessing
# (`pre__num__impute__strategy`). `get_params()` lists every legal name, and a typo raises an error rather than
# being silently ignored.

# %% [markdown]
# ## 2. Grid search: try every combination
#
# `GridSearchCV` takes a dict of lists and evaluates the full cross product with cross-validation. We tune the
# regularisation strength `C` (smaller = stronger shrinkage) and, as a preprocessing example, the imputation strategy.

# %%
grid = {
    "clf__C": [0.01, 0.1, 1, 10],
    "pre__num__impute__strategy": ["median", "mean"],
}
gs = GridSearchCV(logreg, grid, cv=5, scoring="roc_auc")
gs.fit(X_train, y_train)

res = pd.DataFrame(gs.cv_results_)
cols = ["param_clf__C", "param_pre__num__impute__strategy", "mean_test_score", "std_test_score", "rank_test_score"]
print("best:", gs.best_params_, f"CV AUC = {gs.best_score_:.4f}")
res[cols].sort_values("rank_test_score").round(4)

# %% [markdown]
# *What just happened?* 4 x 2 = 8 candidates, each scored on 5 folds. `cv_results_` is the full audit trail;
# always look at it, not just `best_params_`. Two things to read:
# 1. The **spread** (`std_test_score`) across folds. If the top candidates differ by less than the std, the
#    "winner" is a coin flip, so prefer the simpler/faster one.
# 2. By default `refit=True`: after the search, sklearn retrains the best configuration on **all** of `X_train`
#    and stores it in `gs.best_estimator_`, ready for `predict`.

# %% [markdown]
# ### Budgeting: count the fits *before* you launch
#
# Total fits = (number of candidates) x (number of CV folds), plus one refit. Grids explode multiplicatively:
# add a fifth hyperparameter with 5 values and the cost multiplies by 5.

# %%
n_candidates = len(ParameterGrid(grid))
print(f"{n_candidates} candidates x 5 folds = {n_candidates * 5} fits (+1 refit)")
big = {"a": range(5), "b": range(5), "c": range(5), "d": range(5), "e": range(5)}
print(f"a 5-hyperparameter grid with 5 values each: {len(ParameterGrid(big))} candidates -> {len(ParameterGrid(big)) * 5} fits")

# %% [markdown]
# ## 3. Random search: sample instead of enumerate
#
# `RandomizedSearchCV` draws a fixed number (`n_iter`) of configurations from **distributions**. You choose the
# budget directly, and continuous hyperparameters are sampled from a range rather than a hand-picked list.
#
# Why is that often better than a grid? Usually only a couple of hyperparameters really matter. A 3x3 grid tests
# just **3 distinct values of each**, while 9 random draws test **9 distinct values of each**. Let us count it:

# %%
space = {"learning_rate": loguniform(0.01, 0.3), "l2": loguniform(1e-3, 10)}
grid_pts = pd.DataFrame(list(ParameterGrid({"learning_rate": [0.01, 0.05, 0.3], "l2": [1e-3, 0.1, 10]})))
rand_pts = pd.DataFrame(list(ParameterSampler(space, n_iter=9, random_state=42)))
print("distinct learning_rate values  grid:", grid_pts["learning_rate"].nunique(), " random:", rand_pts["learning_rate"].nunique())

fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(grid_pts["learning_rate"], grid_pts["l2"], s=70, label="grid (9 points)")
ax.scatter(rand_pts["learning_rate"], rand_pts["l2"], s=70, marker="x", label="random (9 points)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("learning_rate (log scale)")
ax.set_ylabel("l2 regularisation (log scale)")
ax.set_title("Same budget, different coverage")
ax.legend()
plt.show()

# %% [markdown]
# *What just happened?* Same 9 evaluations, but random search probes 9 different learning rates. If learning
# rate is what matters, the grid wasted two thirds of its budget re-testing the same three values.
# Two practical rules: sample scale-like hyperparameters (`C`, learning rate, regularisation) with `loguniform`,
# and integers (`max_leaf_nodes`) with `randint`.
#
# Now a real one: gradient boosting on churn. `HistGradientBoostingClassifier` is sklearn's fast histogram-based
# booster. We keep `max_iter` fixed at 100 trees to keep the run short.

# %%
hgb = Pipeline([("pre", make_preprocessor()), ("clf", HistGradientBoostingClassifier(max_iter=100, random_state=42))])
hgb_space = {
    "clf__learning_rate": loguniform(0.01, 0.3),
    "clf__max_leaf_nodes": randint(4, 64),
    "clf__min_samples_leaf": randint(10, 200),
    "clf__l2_regularization": loguniform(1e-3, 10),
}

t0 = time.perf_counter()
rs = RandomizedSearchCV(hgb, hgb_space, n_iter=12, cv=3, scoring="roc_auc", random_state=42)
rs.fit(X_train, y_train)
t_random = time.perf_counter() - t0
print(f"random search: {len(rs.cv_results_['params'])} candidates, {t_random:.1f}s, best CV AUC = {rs.best_score_:.4f}")
print({k.replace("clf__", ""): (round(float(v), 4) if isinstance(v, float) else int(v)) for k, v in rs.best_params_.items()})

# %% [markdown]
# Compare with the untuned default and with the tuned logistic regression from before. Do not assume the fancy
# model wins: on tabular data with mostly simple effects a regularised linear model is often hard to beat.

# %%
default_hgb = cross_val_score(hgb, X_train, y_train, cv=3, scoring="roc_auc").mean()
print(f"HGB defaults  CV AUC: {default_hgb:.4f}")
print(f"HGB random    CV AUC: {rs.best_score_:.4f}")
print(f"LogReg grid   CV AUC: {gs.best_score_:.4f}   (different fold count, so only roughly comparable)")

# %% [markdown]
# ## 4. Successive halving: spend little on losers
#
# Idea from tournaments: start many candidates on a **small slice of the data**, keep the best third, give the
# survivors 3x more data, repeat. Only a few finalists ever see all the rows.
# `HalvingRandomSearchCV` implements it (import `enable_halving_search_cv` first; sklearn still marks it experimental).

# %%
t0 = time.perf_counter()
hs = HalvingRandomSearchCV(
    hgb, hgb_space, n_candidates=24, factor=3, min_resources=400, cv=3,
    scoring="roc_auc", random_state=42,
)
hs.fit(X_train, y_train)
t_halving = time.perf_counter() - t0
print(f"halving: {t_halving:.1f}s   best CV AUC (on the last rung) = {hs.best_score_:.4f}")
print("rung  candidates  training rows per fit")
for i, (c, r) in enumerate(zip(hs.n_candidates_, hs.n_resources_)):
    print(f"{i:>4}  {c:>10}  {r:>10}")

# %% [markdown]
# *What just happened?* 24 candidates started with 400 rows each; after every rung only the top third survived
# and the data tripled. Note that `best_score_` comes from the **last rung only** and is not directly comparable
# with a full-data CV score.
#
# **Honest caveat.** Halving saves time when a single fit is expensive. Here fits take a fraction of a second, so
# overheads dominate and the saving may be small or absent. Compare the timings above with the random search
# (which only tried 12 candidates) rather than trusting the folklore. Also, early rungs judge on tiny samples,
# so a slow-starting but ultimately good configuration can be eliminated.

# %% [markdown]
# ## 5. Over-tuning: the best score of many tries is a biased score
#
# Every CV score is a noisy estimate. If you evaluate many configurations and report the **maximum**, you have
# selected the luckiest noise, not just the best model. To make this impossible to miss, we build data with
# **no signal at all**: random features and coin-flip labels. The honest AUC is 0.5, whatever we do.

# %%
rng = np.random.default_rng(42)
X_noise = rng.normal(size=(1200, 10))
y_noise = rng.integers(0, 2, size=1200)
X_small, y_small = X_noise[:150], y_noise[:150]  # what we tune on
X_hold, y_hold = X_noise[150:], y_noise[150:]  # fresh data the search never sees

tree_space = {
    "max_depth": randint(1, 12),
    "min_samples_leaf": randint(1, 30),
    "max_features": [0.2, 0.5, 1.0],
    "criterion": ["gini", "entropy"],
    "random_state": randint(0, 1000),  # different seeds = more ways to get lucky
}
rows = []
for n_iter in [1, 5, 20, 100, 300]:
    s = RandomizedSearchCV(DecisionTreeClassifier(), tree_space, n_iter=n_iter, cv=5, scoring="roc_auc", random_state=42)
    s.fit(X_small, y_small)
    rows.append((n_iter, s.best_score_, roc_auc_score(y_hold, s.predict_proba(X_hold)[:, 1])))
overtune = pd.DataFrame(rows, columns=["configs_tried", "best_cv_auc", "holdout_auc"])
overtune.round(3)

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(overtune["configs_tried"], overtune["best_cv_auc"], "o-", label="best CV AUC (what the search reports)")
ax.plot(overtune["configs_tried"], overtune["holdout_auc"], "s-", label="fresh holdout AUC (the truth)")
ax.axhline(0.5, color="grey", linestyle=":", label="pure chance")
ax.set_xscale("log")
ax.set_xlabel("configurations tried (log scale)")
ax.set_ylabel("ROC AUC")
ax.set_title("Tuning on pure noise: the reported score climbs, reality does not")
ax.legend(fontsize=8)
plt.show()

# %% [markdown]
# *What just happened?* The search reports a CV AUC well above chance on data that contains nothing to
# learn, and the more configurations it tries, the higher that number gets. The holdout score stays around 0.5.
# Real data has real signal, so the effect is smaller, but the mechanism is identical, and it is worst with
# **small data, big search spaces and noisy metrics**.
#
# ### The fix: nested cross-validation
# Treat "search + refit" as *part of the model* and evaluate that whole procedure on outer folds the search never touched.

# %%
inner = RandomizedSearchCV(DecisionTreeClassifier(), tree_space, n_iter=40, cv=3, scoring="roc_auc", random_state=42)
nested = cross_val_score(inner, X_small, y_small, cv=5, scoring="roc_auc")
inner.fit(X_small, y_small)
print(f"non-nested (search's own best_score_): {inner.best_score_:.3f}")
print(f"nested CV estimate (outer folds)      : {nested.mean():.3f}  (truth is 0.5)")

# %% [markdown]
# On the noise data the nested estimate lands closer to chance than `best_score_` (it can even dip a little below 0.5: it is an estimate too), while `best_score_` is optimistic.
# Nested CV costs `outer_folds x` the search, so use it to **validate the tuning procedure** or to compare model
# families fairly, not for every experiment.

# %% [markdown]
# ## 6. Cheaper knobs: early stopping
#
# Some hyperparameters can be found *during* training instead of searched. Boosting's number of trees is the classic
# case: hold out a slice of the training data internally and stop when the score stops improving.

# %%
es = HistGradientBoostingClassifier(
    max_iter=1000, early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, random_state=42
)
es_pipe = Pipeline([("pre", make_preprocessor()), ("clf", es)]).fit(X_train, y_train)
print("asked for up to 1000 trees, early stopping kept:", es_pipe.named_steps["clf"].n_iter_)

# %% [markdown]
# *What just happened?* One hyperparameter (`max_iter`) removed from the search space for free. The internal
# validation slice is carved from the training data only, so the test set is still untouched.

# %% [markdown]
# ## 7. Finishing: evaluate the tuned model on the test set once

# %%
best = rs.best_estimator_  # already refit on all of X_train
test_auc = roc_auc_score(y_test, best.predict_proba(X_test)[:, 1])
print(f"tuned HGB   CV AUC {rs.best_score_:.4f}  ->  test AUC {test_auc:.4f}")
print("A small gap is normal. A large drop means the search overfit the validation folds.")

# %% [markdown]
# ## Common mistakes
#
# 1. **Tuning on the test set** (or peeking at it "just once" per experiment). The test set is for one final number.
# 2. **Fitting the scaler/imputer outside the pipeline** and then searching: preprocessing sees validation folds. Put it in the `Pipeline`.
# 3. **Reading `best_score_` as your expected performance.** It is the maximum of noisy estimates (section 5).
# 4. **Grids on log-scale hyperparameters with linear spacing** (`C` in `[1, 2, 3, 4]`). Use powers of 10 or `loguniform`.
# 5. **Ignoring the edge of the range.** If the best value sits on the boundary of your grid, extend the grid and rerun.
# 6. **Tuning a metric that is not your business metric** (accuracy on an imbalanced problem).

# %%
# A typo in a hyperparameter name fails loudly, which is what you want.
try:
    GridSearchCV(logreg, {"clf__Cee": [1]}, cv=2, error_score="raise").fit(X_train, y_train)
except ValueError as e:
    print("ValueError:", str(e).splitlines()[-1][:110])

# %% [markdown]
# ## Interview corner
#
# **1. Grid vs random search: when would you choose each?**
# <details><summary>Answer</summary>
# Grid for 1-2 hyperparameters with a few sensible values, when you want a complete, reproducible picture. Random for
# 3 or more, or continuous ranges: you set the budget, and each draw tests a fresh value of every hyperparameter, which
# matters because typically only a few hyperparameters dominate performance.
# </details>
#
# **2. Your tuned model scored 0.85 in CV but 0.79 in production. Name two causes related to tuning.**
# <details><summary>Answer</summary>
# Selection bias: the best of many noisy CV scores is optimistic (fix: nested CV or a final untouched test set). And
# leakage or distribution shift: preprocessing fitted outside the folds, or production data differing from training data.
# </details>
#
# **3. What does successive halving trade away to save compute?**
# <details><summary>Answer</summary>
# Early elimination is decided on small samples and few resources, so candidates that only shine with more data can be
# dropped. It also makes scores from different rungs non-comparable. It pays off when single fits are expensive.
# </details>
#
# **4. What is nested cross-validation and when is it worth its cost?**
# <details><summary>Answer</summary>
# An outer CV loop evaluates a procedure that includes an inner CV search. The outer score is an honest estimate of
# "tune, then fit" because outer test folds never influenced the choice. Worth it with small data, large search spaces, or
# when comparing model families that were tuned differently.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Tune for the metric you actually care about
# Using the `logreg` pipeline and the same `clf__C` values `[0.01, 0.1, 1, 10]`, run a `GridSearchCV` with `cv=5` that
# optimises **average precision** (`scoring="average_precision"`) on `X_train, y_train`. Store the fitted search in `ap_search`
# and the winning `C` in `best_C`.
# *Hint: the search object's `best_params_` is a dict with the prefixed key.*

# %% tags=["starter"]
ap_search = ...
best_C = ...

# %%
manual = {C: cross_val_score(logreg.set_params(clf__C=C), X_train, y_train, cv=5, scoring="average_precision").mean()
          for C in [0.01, 0.1, 1, 10]}
expected_C = max(manual, key=manual.get)
assert best_C == expected_C, f"expected C={expected_C} (checked with cross_val_score by hand), got {best_C}"
assert abs(ap_search.best_score_ - manual[expected_C]) < 1e-9, "best_score_ should be the mean CV average precision"
print(f"Exercise 1 passed (best C = {best_C}, AP = {ap_search.best_score_:.4f})")

# %% tags=["solution"]
ap_search = GridSearchCV(logreg, {"clf__C": [0.01, 0.1, 1, 10]}, cv=5, scoring="average_precision").fit(X_train, y_train)
best_C = ap_search.best_params_["clf__C"]

# %% [markdown]
# ### Exercise 2: Budget a search
# A colleague proposes a grid over `max_depth` in `[2, 4, 6, 8, 10]`, `min_samples_leaf` in `[5, 10, 20, 50]` and
# `criterion` in `["gini", "entropy"]`, with 5-fold CV. How many model fits will it run **excluding** the final refit?
# Store it in `n_fits`. Then say how many fits a `RandomizedSearchCV(n_iter=15, cv=5)` runs, in `n_fits_random`.
# *Hint: `ParameterGrid` will count candidates for you.*

# %%
colleague_grid = {"max_depth": [2, 4, 6, 8, 10], "min_samples_leaf": [5, 10, 20, 50], "criterion": ["gini", "entropy"]}

# %% tags=["starter"]
n_fits = ...
n_fits_random = ...

# %%
probe = GridSearchCV(DecisionTreeClassifier(random_state=42), colleague_grid, cv=5, refit=False, scoring="roc_auc")
probe.fit(X_train.select_dtypes("number").fillna(0), y_train)
expected = len(probe.cv_results_["params"]) * 5
assert n_fits == expected, f"expected {expected} fits (candidates x folds), got {n_fits}"
assert n_fits_random == 15 * 5, "random search runs n_iter x cv fits"
print(f"Exercise 2 passed ({n_fits} vs {n_fits_random})")

# %% tags=["solution"]
n_fits = len(ParameterGrid(colleague_grid)) * 5
n_fits_random = 15 * 5

# %% [markdown]
# ### Exercise 3: Random search over a tree pipeline
# Build a pipeline `tree_pipe` = `make_preprocessor()` + `DecisionTreeClassifier(random_state=42)` (step names `"pre"`, `"clf"`).
# Run `RandomizedSearchCV` named `tree_search` with `n_iter=15`, `cv=3`, `scoring="roc_auc"`, `random_state=42`, searching
# `clf__max_depth` in `randint(2, 15)` and `clf__min_samples_leaf` in `randint(5, 100)`, fitted on `X_train, y_train`.
# *Hint: names are `<step>__<param>`; `random_state=42` in the search makes the draws reproducible.*

# %% tags=["starter"]
tree_pipe = ...
tree_search = ...

# %%
assert isinstance(tree_pipe, Pipeline) and list(tree_pipe.named_steps) == ["pre", "clf"], "steps must be named 'pre' and 'clf'"
results = pd.DataFrame(tree_search.cv_results_)
assert len(results) == 15, "n_iter=15 should give 15 candidates"
assert set(tree_search.best_params_) == {"clf__max_depth", "clf__min_samples_leaf"}, "search only those two hyperparameters"
assert tree_search.best_score_ == results["mean_test_score"].max(), "best_score_ should be the top mean_test_score"
assert 2 <= tree_search.best_params_["clf__max_depth"] < 15
top = results.sort_values("rank_test_score").iloc[0]
print(f"Exercise 3 passed (best depth {tree_search.best_params_['clf__max_depth']}, AUC {top['mean_test_score']:.3f} +/- {top['std_test_score']:.3f})")

# %% tags=["solution"]
tree_pipe = Pipeline([("pre", make_preprocessor()), ("clf", DecisionTreeClassifier(random_state=42))])
tree_search = RandomizedSearchCV(
    tree_pipe,
    {"clf__max_depth": randint(2, 15), "clf__min_samples_leaf": randint(5, 100)},
    n_iter=15, cv=3, scoring="roc_auc", random_state=42,
).fit(X_train, y_train)

# %% [markdown]
# ### Exercise 4: Quantify the optimism
# Reproduce the optimism gap on the noise data for a **different search size**. Using `X_small, y_small`, run a
# `RandomizedSearchCV(DecisionTreeClassifier(), tree_space, n_iter=60, cv=5, scoring="roc_auc", random_state=7)`
# and store the search's own score in `reported` and the AUC on `X_hold, y_hold` in `honest`. Then store
# `gap = reported - honest`. The check requires that the gap is positive: tuning on noise must look better than it is.

# %% tags=["starter"]
reported = ...
honest = ...
gap = ...

# %%
chk = RandomizedSearchCV(DecisionTreeClassifier(), tree_space, n_iter=60, cv=5, scoring="roc_auc", random_state=7).fit(X_small, y_small)
assert abs(reported - chk.best_score_) < 1e-9, "reported should be the search's best_score_ (use random_state=7)"
assert abs(honest - roc_auc_score(y_hold, chk.predict_proba(X_hold)[:, 1])) < 1e-9, "honest is the holdout AUC of the refit best model"
assert gap > 0, f"gap was {gap:.3f}; on noise, the best-of-many CV score should exceed the holdout"
print(f"Exercise 4 passed (reported {reported:.3f}, honest {honest:.3f}, gap {gap:.3f})")

# %% tags=["solution"]
s4 = RandomizedSearchCV(DecisionTreeClassifier(), tree_space, n_iter=60, cv=5, scoring="roc_auc", random_state=7).fit(X_small, y_small)
reported = s4.best_score_
honest = roc_auc_score(y_hold, s4.predict_proba(X_hold)[:, 1])
gap = reported - honest

# %% [markdown]
# ### Exercise 5 (open-ended): Is the tuned booster worth it?
# Compare `gs.best_score_` (tuned logistic regression), `default_hgb` and `rs.best_score_`. Write two or three sentences:
# is the extra complexity of the booster justified on this dataset? Would a difference this size matter for the business?
# What would you check next (fold std, calibration, training time, interpretability)?

# %% [markdown]
# ## Recap
# - Hyperparameters are set by you; find them with cross-validation on the **training** data, inside a `Pipeline`.
# - Grid search enumerates and explodes; random search fixes the budget and tests more distinct values; halving spends
#   little on losers but only pays off for expensive fits.
# - Count fits (candidates x folds) before launching, and read `cv_results_`, including the fold spread.
# - The best of many noisy scores is optimistic. Nested CV, or one untouched test set, gives the honest number.
# - Use early stopping for "how many trees/epochs" instead of searching for it.
#
# **Go deeper:** the scikit-learn user guide chapter "Tuning the hyper-parameters of an estimator" and the example
# "Nested versus non-nested cross-validation" (scikit-learn.org/stable/).
#
# **Next:** [k-NN, SVM and Naive Bayes](10_knn_svm_and_naive_bayes.ipynb)

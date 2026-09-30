# %% [markdown]
# # Train/Test Split and Data Leakage
#
# **Why this matters.** A model's score on data it was trained on tells you almost nothing: a model can simply memorise. The whole
# point of evaluation is to estimate how the model behaves on *new* customers next month. Data leakage is any way that information
# from outside the training set sneaks into training, so the estimate becomes a fantasy. Leaks are the most common reason a model that
# "had 95% AUC in the notebook" is useless in production, and reviewers will probe for them in every interview.
#
# **You will learn**
# - split data with `train_test_split`, including `stratify`, and see how much a single split can vary
# - recognise and *measure* four classic leaks: preprocessing before the split, target-derived features, duplicates and time leakage
# - apply the safe pattern: split first, fit everything on the training part only
#
# **Prerequisites:** [ML problem framing and workflow](01_ml_problem_framing_and_workflow.ipynb).
#
# > Data used: `datasets/churn.csv` and `datasets/sensors.csv` (synthetic, see `docs/DATASETS.md`).

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA = Path("../datasets")
churn = pd.read_csv(DATA / "churn.csv")
churn["total_charges"] = churn["total_charges"].fillna(churn["total_charges"].median())  # crude fill, fixed properly in notebook 08
NUM = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "senior_citizen"]
X, y = churn[NUM], churn["churned"]

# %% [markdown]
# ## 1. Why we split at all
#
# Give a model enough flexibility and it will fit the training data perfectly, noise included. An unlimited-depth decision tree is the
# extreme case (notebook 05 explains trees). Compare its score on the rows it saw vs rows it did not:

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
print("train accuracy:", round(tree.score(X_train, y_train), 3))
print("test  accuracy:", round(tree.score(X_test, y_test), 3))
print("always-predict-majority on test:", round(1 - y_test.mean(), 3))

# %% [markdown]
# *What just happened?* The tree scores 1.0 on training rows (it memorised them) but on unseen rows does worse than a model that predicts "nobody churns".
# The training score measured memory, not skill. **The test set is a stand-in for the future**, so it must play no role in any decision
# about the model, including preprocessing.
#
# ### `train_test_split` options that matter
# - `test_size`: 0.2-0.25 is typical; with little data, use cross-validation instead (notebook 07).
# - `random_state`: makes the split reproducible. It is *not* a hyperparameter to tune (choosing the seed that gives the best test score is leakage by hand).
# - `stratify=y`: keeps the class proportions equal in both parts. Matters most with small or imbalanced data.

# %%
small = churn.sample(200, random_state=0)
for strat in [None, small["churned"]]:
    rates = [
        train_test_split(small["churned"], test_size=0.2, stratify=strat, random_state=seed)[1].mean()
        for seed in range(200)
    ]
    print(f"stratify={'yes' if strat is not None else 'no ':3s}: test churn rate ranges {min(rates):.3f} to {max(rates):.3f} across 200 seeds (overall {small['churned'].mean():.3f})")

# %% [markdown]
# Without `stratify` the test churn rate wobbles from seed to seed; with it, it barely moves. And even so, **one split gives one noisy estimate**:

# %%
accs = []
for seed in range(30):
    a, b, c, d = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    accs.append(DecisionTreeClassifier(max_depth=3, random_state=42).fit(a, c).score(b, d))
print(f"depth-3 tree, 30 different splits: mean {np.mean(accs):.3f}, std {np.std(accs):.3f}, range {min(accs):.3f} to {max(accs):.3f}")

# %% [markdown]
# Differences smaller than that spread between two models are not evidence of anything. Cross-validation (notebook 07) is the systematic answer.
#
# **Three-way split.** When you tune hyperparameters, you need a *validation* set (or CV inside the training set) so the test set stays untouched:
# train (fit) -> validation (choose) -> test (report once).

# %% [markdown]
# ## 2. Leak #1: preprocessing fitted before the split
#
# The classic recipe error: `scaler.fit_transform(X)` on *all* data, then split. The scaler's mean and standard deviation now contain
# information about the test rows. For a plain scaler the effect is usually tiny; the habit is wrong regardless, and with **supervised**
# steps like feature selection the effect is enormous. First see the tiny one:

# %%
scaler_all = StandardScaler().fit(X)  # WRONG: sees the test rows
scaler_train = StandardScaler().fit(X_train)  # right
print("mean of monthly_charges seen by scaler | all data:", round(scaler_all.mean_[1], 3), "| train only:", round(scaler_train.mean_[1], 3), "| test rows:", round(X_test["monthly_charges"].mean(), 3))

# %% [markdown]
# The wrong scaler's statistics are pulled slightly toward the test rows. Now the dramatic one. Here the data is **pure noise**: 100 rows,
# 5,000 random features, random labels. No model can genuinely do better than chance, about 50% accuracy.

# %%
rng = np.random.default_rng(42)
X_noise = rng.normal(size=(100, 5000))
y_noise = rng.integers(0, 2, size=100)

# WRONG: pick the 20 features that correlate best with y using ALL rows, then cross-validate
X_selected = SelectKBest(f_classif, k=20).fit_transform(X_noise, y_noise)
leaky = cross_val_score(LogisticRegression(), X_selected, y_noise, cv=5).mean()

# RIGHT: selection lives inside the pipeline, so it is re-fitted on each training fold only
honest = cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression()), X_noise, y_noise, cv=5).mean()
print(f"leaky selection accuracy: {leaky:.2f}   honest accuracy: {honest:.2f}")

# %% [markdown]
# *What just happened?* With 5,000 candidates, some features correlate with random labels by luck. Selecting them using every row's label bakes the
# test folds' answers into the features, so the "cross-validated" score is far above chance on data that contains no signal at all.
# The honest version, which lets the pipeline select inside each training fold, lands near chance, as it must.
# **Rule: anything that learns from data (imputer, scaler, encoder, selector, PCA) is part of the model and is fitted on training rows only.**
# Notebook 08 builds this rule into pipelines so you cannot forget it.

# %% [markdown]
# ## 3. Leak #2: a feature derived from the target
#
# Suppose a data engineer adds `cancellation_requested`, a flag that is set when a customer calls to cancel. It is not filled in for most
# stayers and it is recorded *before* the churn is finalised. It looks like a great predictor because it is a consequence of the target.
# We simulate it here (this column does not exist in the dataset; we create it from the label to show what happens):

# %%
leaky_df = churn.copy()
leaky_df["cancellation_requested"] = ((leaky_df["churned"] == 1) & (np.random.default_rng(0).random(len(leaky_df)) < 0.8)).astype(int)

def test_auc(df, cols):
    a, b, c, d = train_test_split(df[cols], df["churned"], test_size=0.25, stratify=df["churned"], random_state=42)
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(a, c)
    return roc_auc_score(d, m.predict_proba(b)[:, 1])

print("AUC without the leaky feature:", round(test_auc(leaky_df, NUM), 3))
print("AUC with the leaky feature   :", round(test_auc(leaky_df, NUM + ["cancellation_requested"]), 3))

# %% [markdown]
# A jump like that should make you suspicious, not happy. A quick **leak detector**: score each feature *alone*. A single feature with
# near-perfect AUC in a problem where nothing is that predictable is almost always a leak or a mislabel.

# %%
single = {}
for col in NUM + ["cancellation_requested"]:
    auc = roc_auc_score(leaky_df["churned"], leaky_df[col])
    single[col] = max(auc, 1 - auc)  # direction does not matter
pd.Series(single).sort_values(ascending=False).round(3)

# %% [markdown]
# The fix is not statistical, it is organisational: for every feature ask *"would I have this value at the moment I make the prediction?"*
# and if a table has timestamps, filter features to values recorded **before** the snapshot date.

# %% [markdown]
# ## 4. Leak #3: duplicates and related rows across the split
#
# If the same customer (or a near-copy) lands in both train and test, the model is being tested on something it has seen.
# This happens after bad joins (row fan-out), repeated snapshots of one customer, or images from the same patient. Simulate a join fan-out that
# doubles every row, then split randomly:

# %%
dup = pd.concat([churn, churn], ignore_index=True)  # each customer appears twice
a, b, c, d = train_test_split(dup[NUM], dup["churned"], test_size=0.25, random_state=42)
t_dup = DecisionTreeClassifier(random_state=42).fit(a, c)
print("test accuracy with duplicated rows across the split:", round(t_dup.score(b, d), 3))

# Correct: split by customer, so both copies of a customer stay on the same side
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
tr_idx, te_idx = next(gss.split(dup, groups=dup["customer_id"]))
t_grp = DecisionTreeClassifier(random_state=42).fit(dup.loc[tr_idx, NUM], dup.loc[tr_idx, "churned"])
print("test accuracy with a group-aware split            :", round(t_grp.score(dup.loc[te_idx, NUM], dup.loc[te_idx, "churned"]), 3))

# %% [markdown]
# *What just happened?* The memorising tree looked great on the first split because most test rows had an identical twin in training.
# Splitting **by group** (`GroupShuffleSplit`, or `GroupKFold` for CV) removed the leak and the score dropped to roughly the honest unseen-data level of section 1.
# Two habits: check `df.duplicated()` and key uniqueness before splitting, and ask "what is the real unit of independence?" (customer, patient, device, store).

# %% [markdown]
# ## 5. Leak #4: time
#
# If the model will predict the *future*, a random split lets it train on hours after the ones it is tested on: it interpolates instead of forecasting.
# Take the hourly temperature sensor (dropouts removed) and predict temperature from the timestamp alone, using a 3-nearest-neighbours model.

# %%
s = pd.read_csv(DATA / "sensors.csv", parse_dates=["timestamp"]).dropna(subset=["temperature_c"]).sort_values("timestamp")
s["hours"] = (s["timestamp"] - s["timestamp"].min()).dt.total_seconds() / 3600
Xs, ys = s[["hours"]], s["temperature_c"]

a, b, c, d = train_test_split(Xs, ys, test_size=0.25, random_state=42)
mae_random = mean_absolute_error(d, KNeighborsRegressor(3).fit(a, c).predict(b))

cut = int(len(s) * 0.75)  # first 75% of time = train, last 25% = test
mae_time = mean_absolute_error(ys[cut:], KNeighborsRegressor(3).fit(Xs[:cut], ys[:cut]).predict(Xs[cut:]))
print(f"random split MAE: {mae_random:.2f} C   chronological split MAE: {mae_time:.2f} C")

# %% [markdown]
# The random split looks accurate because each test hour is sandwiched between training hours with nearly the same temperature. The chronological split
# is the honest estimate of forecasting the future, and it is much worse. **If predictions will be made forward in time, validate forward in time**
# (`TimeSeriesSplit` in notebook 07, full treatment in notebook 14).

# %% [markdown]
# ## Common mistakes
#
# 1. **`fit_transform` on the full dataset, then split.** Fit on train, `transform` on test (or use a `Pipeline`).
# 2. **Tuning on the test set**: trying ten models and reporting the best test score turns the test set into a validation set. Use CV, keep test for one final look.
# 3. **Re-running the split until it looks good.** The seed is not a hyperparameter.
# 4. **Ignoring group structure** (repeat customers, patients, devices).
# 5. **Trusting a score that looks too good.** Treat surprising jumps as a bug until proven otherwise.

# %%
# Mistake 1 as code you should recognise in someone else's notebook
bad_scaled = StandardScaler().fit_transform(X)           # <- fitted on everything
bad_train, bad_test = train_test_split(bad_scaled, test_size=0.25, random_state=42)  # split AFTER: leak
print("looks fine, runs fine, still wrong:", bad_train.shape, bad_test.shape)

# %% [markdown]
# ## Interview corner
#
# **1. What is data leakage? Give three examples.**
# <details><summary>Answer</summary>
# Information available at training or evaluation time that will not be available when the model is used, or that leaks the answer from the evaluation data into training.
# Examples: fitting a scaler/imputer/selector on all rows before splitting, a feature computed from the target or recorded after the outcome, the same entity in train and test,
# random splitting of time-ordered data.
# </details>
#
# **2. Your model has 0.99 AUC on a hard problem. What do you do?**
# <details><summary>Answer</summary>
# Assume a leak. Check the single-feature importances/AUCs, look for identifiers or post-outcome fields, verify duplicates and group overlap, check how the split
# was made, and compare against a baseline. Then re-run with the suspicious feature removed.
# </details>
#
# **3. Why fit the scaler on the training set only?**
# <details><summary>Answer</summary>
# In production the scaler will only ever have seen historical data, so the test set must be treated like future data. Fitting on all data lets test statistics influence
# training. The effect is small for scalers but large for supervised preprocessing such as feature selection or target encoding.
# </details>
#
# **4. When do you use `stratify`, and when a group or time split?**
# <details><summary>Answer</summary>
# Stratify for class balance (classification, especially imbalanced). Group split when rows are not independent (same customer/patient/device). Time split when the model predicts the future.
# You can combine them in CV (`StratifiedGroupKFold`).
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Stratify or not?
# Take the first 500 rows of `churn` (`small = churn.head(500)`). Split with `test_size=0.1`, `random_state=0` and `stratify=small["churned"]`. Store the churn rate
# of the test part in `strat_rate` and the number of test rows in `n_test`. *Hint: `train_test_split(small, small["churned"], ...)` returns four objects.*

# %%
small = churn.head(500)

# %% tags=["starter"]
strat_rate = ...
n_test = ...

# %%
assert n_test == round(0.1 * len(small)), "test_size=0.1 of the rows"
assert abs(strat_rate - small["churned"].mean()) <= 1 / 50, "stratified test rate should be within one row of the overall rate"
print(f"Exercise 1 passed: test rate {strat_rate:.3f} vs overall {small['churned'].mean():.3f}")

# %% tags=["solution"]
_, _, _, y_te = train_test_split(small, small["churned"], test_size=0.1, random_state=0, stratify=small["churned"])
strat_rate = float(y_te.mean())
n_test = len(y_te)

# %% [markdown]
# ### Exercise 2: A scaler that does not peek
# With the existing `X_train` / `X_test`, fit a `StandardScaler` on the **training** rows only. Store in `test_scaled_mean` the mean of the scaled
# `monthly_charges` column (column index 1) **over the test rows**. It will not be exactly 0, and that is correct: the test rows are new data.
# *Hint: `scaler.transform(X_test)[:, 1].mean()`.*

# %% tags=["starter"]
test_scaled_mean = ...

# %%
mu, sd = X_train["monthly_charges"].mean(), X_train["monthly_charges"].std(ddof=0)
expected = float(((X_test["monthly_charges"] - mu) / sd).mean())
assert abs(test_scaled_mean - expected) < 1e-9, "fit on X_train only, then transform X_test (do not refit)"
print("Exercise 2 passed:", round(test_scaled_mean, 4))

# %% tags=["solution"]
sc = StandardScaler().fit(X_train)
test_scaled_mean = float(sc.transform(X_test)[:, 1].mean())

# %% [markdown]
# ### Exercise 3: Leak detective
# The DataFrame `suspect` below has the churn features plus a mystery column `mystery`. Using the single-feature AUC idea (direction-free: `max(auc, 1 - auc)`),
# store the name of the column that looks like a leak in `leaky_feature`, and its direction-free AUC in `leaky_auc`.

# %%
_r = np.random.default_rng(7)
suspect = churn[NUM + ["churned"]].copy()
suspect["mystery"] = suspect["churned"] * 2.0 + _r.normal(0, 0.6, len(suspect))  # looks like an ordinary numeric column

# %% tags=["starter"]
leaky_feature = ...
leaky_auc = ...

# %%
aucs = {c: max(roc_auc_score(suspect["churned"], suspect[c]), 1 - roc_auc_score(suspect["churned"], suspect[c])) for c in suspect.columns if c != "churned"}
best = max(aucs, key=aucs.get)
assert leaky_feature == best, f"score every feature on its own against churned; the top one is {best!r}"
assert abs(leaky_auc - aucs[best]) < 1e-9, "leaky_auc is max(auc, 1 - auc) for that column"
print(f"Exercise 3 passed: {leaky_feature} alone gives AUC {leaky_auc:.3f}; next best is {sorted(aucs.values())[-2]:.3f}")

# %% tags=["solution"]
aucs = {}
for col in suspect.columns.drop("churned"):
    a_ = roc_auc_score(suspect["churned"], suspect[col])
    aucs[col] = max(a_, 1 - a_)
leaky_feature = max(aucs, key=aucs.get)
leaky_auc = aucs[leaky_feature]

# %% [markdown]
# ### Exercise 4: Forward-in-time evaluation
# Using the sensor arrays `Xs`, `ys` and `cut` from section 5, compute the MAE of a **mean-of-training-data baseline** (always predict the mean temperature of the training part)
# on the chronological test part, `mae_baseline`, and store `beats_baseline = True` if the 3-NN chronological MAE (`mae_time`) is lower than it, else `False`.
# The answer tells you whether the forward-in-time model has any skill at all beyond a constant.

# %% tags=["starter"]
mae_baseline = ...
beats_baseline = ...

# %%
exp = float(np.mean(np.abs(ys[cut:] - ys[:cut].mean())))
assert abs(mae_baseline - exp) < 1e-9, "baseline predicts ys[:cut].mean() for every test row"
assert beats_baseline == (mae_time < exp), "compare mae_time with the baseline MAE"
print(f"Exercise 4 passed: baseline MAE {mae_baseline:.2f} vs 3-NN {mae_time:.2f} (random-split MAE was {mae_random:.2f})")

# %% tags=["solution"]
mae_baseline = float(np.mean(np.abs(ys[cut:] - ys[:cut].mean())))
beats_baseline = bool(mae_time < mae_baseline)

# %% [markdown]
# ### Exercise 5 (open-ended): Audit a feature list
# For the churn columns `tenure_months, contract, internet_service, monthly_charges, total_charges, support_calls, paperless_billing, autopay, senior_citizen`,
# write in a markdown cell which ones you would want to double-check for leakage in a real deployment, and *what question* you would ask the data owner
# (for example: when is `support_calls` counted, and does it include calls made while cancelling?).

# %% [markdown]
# ## Recap
# - Score on unseen rows: training accuracy measures memory. A single split is noisy; `stratify=y` reduces one source of noise.
# - Anything that learns from data (scaler, imputer, selector, encoder) is fitted on training rows only. Pipelines enforce this.
# - Leaks come from four places: preprocessing before the split, target-derived/post-outcome features, duplicates or shared entities, and time order.
# - Too-good results are a smell; single-feature AUC is a fast detector.
# - Match the split to reality: stratified, grouped, or chronological.
#
# **Go deeper:** the scikit-learn user guide "Common pitfalls and recommended practices" (data leakage section) at scikit-learn.org/stable/common_pitfalls.html.
#
# **Next:** [Linear regression and regularization](03_linear_regression_and_regularization.ipynb)

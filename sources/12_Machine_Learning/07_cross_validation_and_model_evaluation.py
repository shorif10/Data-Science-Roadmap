# %% [markdown]
# # Cross-Validation and Model Evaluation
#
# **Why this matters.** "Model B scored 0.79 and model A scored 0.78, so B wins" is the most common unsupported claim in data science. A single train/test split gives
# one noisy number. Cross-validation (CV) gives many numbers, so you can report a mean *and* a spread, choose between models fairly, tune hyperparameters without touching
# the test set, and pick a splitting scheme that matches how the model will be used. This is the skill that separates a model that "works in the notebook" from one you can defend.
#
# **You will learn**
# - implement k-fold CV by hand and with `cross_val_score` / `cross_validate`
# - choose the right splitter: stratified, repeated, group-aware, time-aware
# - read learning curves to decide between "more data" and "more complexity"
# - compare two models fairly with paired folds, and understand what nested CV is for
#
# **Prerequisites:** [Train/test split and leakage](02_train_test_split_and_data_leakage.ipynb), [ensembles](06_ensembles_random_forest_and_gradient_boosting.ipynb).
#
# > Data used: `datasets/churn.csv` and `datasets/sensors.csv` (synthetic).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    cross_validate,
    learning_curve,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA = Path("../datasets")
churn = pd.read_csv(DATA / "churn.csv", keep_default_na=False, na_values=[""])
churn["contract"] = churn["contract"].str.lower()
churn["internet_service"] = churn["internet_service"].fillna("Unknown")
X = pd.get_dummies(
    churn[["tenure_months", "monthly_charges", "support_calls", "senior_citizen", "contract", "internet_service", "paperless_billing", "autopay"]],
    columns=["contract", "internet_service", "paperless_billing", "autopay"], dtype=float,
)
y = churn["churned"]
# Hold out a final test set FIRST. All CV below happens inside the training part.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(X_train.shape, X_test.shape)

# %% [markdown]
# ## 1. k-fold cross-validation by hand
#
# Split the training data into `k` folds. For each fold in turn: fit on the other `k-1` folds, score on the held-out fold. Every row is used for validation exactly once,
# and you get `k` scores. Tiny example first, to see the index bookkeeping:

# %%
for i, (tr, va) in enumerate(KFold(n_splits=5).split(np.arange(10))):
    print(f"fold {i}: train {tr.tolist()}  validate {va.tolist()}")

# %% [markdown]
# Now the real thing on churn, written as an explicit loop, and checked against `cross_val_score`:

# %%
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
kf = KFold(n_splits=5, shuffle=True, random_state=42)

manual = []
for tr, va in kf.split(X_train):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_train.iloc[tr], y_train.iloc[tr])
    manual.append(roc_auc_score(y_train.iloc[va], m.predict_proba(X_train.iloc[va])[:, 1]))
auto = cross_val_score(model, X_train, y_train, cv=kf, scoring="roc_auc")
print("manual   :", np.round(manual, 4))
print("automatic:", np.round(auto, 4), "| identical:", np.allclose(manual, auto))
print(f"report as: AUC {auto.mean():.3f} +/- {auto.std():.3f} (mean +/- std over 5 folds)")

# %% [markdown]
# *What just happened?* `cross_val_score` is that loop, and because the scaler lives **inside the pipeline** it is refitted on each training fold: no leakage
# (notebook 02). **Always report the mean and the spread**; a difference between models that is smaller than the fold-to-fold spread is not evidence.
# `cross_validate` returns more: fit time, several metrics, and training scores (useful to spot overfitting):

# %%
cv_out = cross_validate(model, X_train, y_train, cv=kf, scoring=["roc_auc", "average_precision", "accuracy"], return_train_score=True)
pd.DataFrame(cv_out).drop(columns=["fit_time", "score_time"]).agg(["mean", "std"]).round(3)

# %% [markdown]
# ## 2. Choosing the right splitter
#
# **Stratified k-fold** keeps the class ratio in every fold. It is what scikit-learn uses automatically when you pass an integer `cv` to a classifier. Plain shuffled `KFold` can
# drift on small or imbalanced data. Compare the positive rate per fold on a small sample where the effect is visible:

# %%
small = churn.sample(150, random_state=1)["churned"]
Xs_small = np.zeros((len(small), 1))
for name, splitter in [("KFold", KFold(5, shuffle=True, random_state=42)), ("StratifiedKFold", StratifiedKFold(5, shuffle=True, random_state=42))]:
    rates = [small.iloc[va].mean() for _, va in splitter.split(Xs_small, small)]
    print(f"{name:16s} positive rate per fold: {np.round(rates, 3)}")

# %% [markdown]
# **Repeated CV.** One 5-fold run depends on which rows happen to share a fold. `RepeatedStratifiedKFold` reruns it with new shuffles and gives a distribution, which is
# especially valuable when data is small or models differ by little:

# %%
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=42)
rep_scores = cross_val_score(model, X_train, y_train, cv=rskf, scoring="roc_auc")
print(f"20 scores: mean {rep_scores.mean():.3f}, std {rep_scores.std():.3f}, min {rep_scores.min():.3f}, max {rep_scores.max():.3f}")

# %% [markdown]
# **Group-aware CV.** If several rows come from the same customer (or patient, device, store), a random split puts some of them on each side and the model is tested on
# things it has effectively seen (notebook 02, leak #3). Use `GroupKFold` so that each group lives entirely in one fold. Here, each customer appears twice (a join fan-out), and we use a
# memorising model (unpruned tree) so the leak is visible:

# %%
dup = pd.concat([churn, churn], ignore_index=True)
Xd = pd.get_dummies(dup[["tenure_months", "monthly_charges", "support_calls", "senior_citizen", "contract", "internet_service"]], columns=["contract", "internet_service"], dtype=float)
yd, groups = dup["churned"], dup["customer_id"]
tree = DecisionTreeClassifier(random_state=42)
plain = cross_val_score(tree, Xd, yd, cv=KFold(5, shuffle=True, random_state=42), scoring="roc_auc").mean()
grouped = cross_val_score(tree, Xd, yd, cv=GroupKFold(5), groups=groups, scoring="roc_auc").mean()
print(f"plain KFold AUC {plain:.3f}   GroupKFold AUC {grouped:.3f}")

# %% [markdown]
# The plain scheme rewards memorisation; the grouped one tells the truth. `StratifiedGroupKFold` combines both when you need class balance too.
#
# **Time-aware CV.** When the model predicts the future, folds must move forward in time: train on the past, validate on the next block. `TimeSeriesSplit` does exactly that
# (folds are *expanding windows*). Look at the fold boundaries on the sensor data, and compare with a shuffled scheme on a task where the neighbours in time leak the answer:

# %%
s = pd.read_csv(DATA / "sensors.csv", parse_dates=["timestamp"]).dropna(subset=["temperature_c"]).sort_values("timestamp")
s["hours"] = (s["timestamp"] - s["timestamp"].min()).dt.total_seconds() / 3600
Xs, ys = s[["hours"]], s["temperature_c"]
for i, (tr, va) in enumerate(TimeSeriesSplit(n_splits=4).split(Xs)):
    print(f"fold {i}: train rows 0..{tr.max()}  validate rows {va.min()}..{va.max()}")

knn = KNeighborsRegressor(3)
mae_shuffled = -cross_val_score(knn, Xs, ys, cv=KFold(4, shuffle=True, random_state=42), scoring="neg_mean_absolute_error").mean()
mae_forward = -cross_val_score(knn, Xs, ys, cv=TimeSeriesSplit(n_splits=4), scoring="neg_mean_absolute_error").mean()
print(f"shuffled KFold MAE {mae_shuffled:.2f} C   TimeSeriesSplit MAE {mae_forward:.2f} C")

# %% [markdown]
# Same leak as notebook 02: shuffling makes forecasting look easy. Match the CV scheme to the deployment scenario.

# %% [markdown]
# ## 3. Learning curves: more data or more complexity?
#
# A **learning curve** plots training and validation score against the number of training rows. Two patterns matter:
# - a big, persistent gap between train and validation: **high variance** (overfitting) -> more data, more regularisation or a simpler model may help;
# - both curves converge at a mediocre score: **high bias** (underfitting) -> more data will not help; you need better features or a more flexible model.

# %%
sizes, tr_sc, va_sc = learning_curve(
    model, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 6), cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc",
)
forest = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=1)
sizes_f, tr_f, va_f = learning_curve(
    forest, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 6), cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc",
)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for a, (sz, tr_, va_, title) in zip(ax, [(sizes, tr_sc, va_sc, "Logistic regression"), (sizes_f, tr_f, va_f, "Random forest")]):
    a.plot(sz, tr_.mean(1), "o-", label="train")
    a.plot(sz, va_.mean(1), "o-", label="validation")
    a.fill_between(sz, va_.mean(1) - va_.std(1), va_.mean(1) + va_.std(1), alpha=0.2)
    a.set(title=f"Learning curve: {title}", xlabel="training rows", ylabel="ROC-AUC")
    a.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Compare the two panels: the logistic curves hug each other (little gap: its limit is model capacity, not data volume), while the forest's training score sits far above its
# validation score (a big gap: variance). At the right edge, look at the slope of the validation curve: if it is still rising, more data would help that model.

# %% [markdown]
# ## 4. Comparing models fairly
#
# Three rules: (1) **same folds for every model** (fix the splitter's `random_state`) so differences are *paired*; (2) look at the **per-fold differences**, not two separate means;
# (3) with repeated CV the scores are not independent (folds share training rows), so naive t-tests are overconfident. A commonly used fix is the *corrected resampled t-test*
# (Nadeau & Bengio), which inflates the variance by `1/J + n_test/n_train`, where J is the number of scores.

# %%
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
candidates = {
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "hist GB": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=150, random_state=42),
    "forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=1),
}
cv_scores = {name: cross_val_score(m, X_train, y_train, cv=cv, scoring="roc_auc") for name, m in candidates.items()}
pd.DataFrame(cv_scores).agg(["mean", "std"]).round(4)

# %%
def corrected_paired_ttest(a, b, n_train, n_test):
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * d.var(ddof=1))
    return d.mean(), t, 2 * stats.t.sf(abs(t), df=J - 1)

n_tr = int(len(X_train) * 0.8)
n_te = len(X_train) - n_tr
n_scores = len(cv_scores['logistic'])
for other in ["hist GB", "forest"]:
    md, t, pval = corrected_paired_ttest(cv_scores["logistic"], cv_scores[other], n_tr, n_te)
    wins = int((cv_scores["logistic"] > cv_scores[other]).sum())
    print(f"logistic minus {other:8s}: mean diff {md:+.4f}, corrected t {t:+.2f}, p {pval:.4f}, logistic better in {wins}/{n_scores} folds")

# %% [markdown]
# *What just happened?* Because all models saw identical folds we can look at differences per fold. A p-value from a corrected test is a rough guide, not a verdict;
# also ask whether the *size* of the difference matters for the business (0.005 AUC rarely does). Also remember multiple comparisons: if you compare twenty models, one will look best by luck.

# %% [markdown]
# ## 5. Nested CV: honest estimates when you also tune
#
# If you tune hyperparameters with CV and then report *that same CV score*, the number is optimistic: you picked the setting that happened to score best on those folds.
# **Nested CV** wraps the whole tuning procedure (inner CV) inside an outer CV, so each outer test fold is never used for choosing hyperparameters. It estimates the
# performance of *the procedure "tune, then fit"*.

# %%
grid = {"max_depth": [2, 3, 4, 5, 6, 8], "min_samples_leaf": [5, 20, 50]}
inner = StratifiedKFold(3, shuffle=True, random_state=1)
outer = StratifiedKFold(5, shuffle=True, random_state=2)

search = GridSearchCV(DecisionTreeClassifier(random_state=42), grid, cv=inner, scoring="roc_auc")
search.fit(X_train, y_train)
print(f"tuned CV score (optimistic): {search.best_score_:.4f}  with {search.best_params_}")

nested = cross_val_score(GridSearchCV(DecisionTreeClassifier(random_state=42), grid, cv=inner, scoring="roc_auc"), X_train, y_train, cv=outer, scoring="roc_auc")
print(f"nested CV score            : {nested.mean():.4f} +/- {nested.std():.4f}")

# %% [markdown]
# The optimism is often small when the grid is small and data is plentiful (as here), and grows with a bigger search space and smaller data. Nested CV is expensive (folds x grid x folds),
# so use it to *compare tuning procedures* or to report an honest estimate on small datasets; for big data a single held-out test set is usually enough. Tuning tools are the focus of notebook 09.

# %% [markdown]
# ## 6. The final report
#
# 1. Choose the model by CV on the **training** part only.
# 2. Refit it on all training rows.
# 3. Score **once** on the held-out test set we set aside at the beginning:

# %%
final = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_train, y_train)
final_auc = roc_auc_score(y_test, final.predict_proba(X_test)[:, 1])
print(f"CV estimate {cv_scores['logistic'].mean():.3f} +/- {cv_scores['logistic'].std():.3f}   |   held-out test AUC {final_auc:.3f}")

# %% [markdown]
# A test score inside (or near) the CV range is reassuring. A test score far *below* it would signal leakage in CV, drift between the sets, or CV overfitting from too many comparisons.

# %% [markdown]
# ## Common mistakes
#
# 1. **Preprocessing outside the CV loop** (scaling, imputing, selecting features on all data). Put it in a pipeline.
# 2. **Reporting only the mean.** The spread is half of the result.
# 3. **Different splits for different models**, then attributing noise to model quality.
# 4. **Random folds for grouped or time-ordered data.**
# 5. **Selecting on CV, then reporting that CV score** after a large search (use nested CV or a held-out test set).
# 6. **Using the test set repeatedly** as if it were another validation set.

# %%
# Mistake 1 shown: selecting the 5 "best" features from noise on ALL data and then cross-validating fools you (see notebook 02)
from sklearn.feature_selection import SelectKBest, f_classif

rng = np.random.default_rng(0)
Xn, yn = rng.normal(size=(120, 2000)), rng.integers(0, 2, 120)
Xsel = SelectKBest(f_classif, k=5).fit_transform(Xn, yn)
print("leaky CV accuracy on pure noise:", round(cross_val_score(LogisticRegression(), Xsel, yn, cv=5).mean(), 3), "(honest value is about 0.5)")

# %% [markdown]
# ## Interview corner
#
# **1. Why use k-fold CV instead of one train/test split? What is the downside?**
# <details><summary>Answer</summary>
# It uses all data for both training and validation and gives a distribution of scores (mean and spread) rather than one noisy number. The downside is k times the compute,
# and (for repeated or overlapping training sets) scores that are correlated, so naive significance tests are overconfident.
# </details>
#
# **2. How do you choose k?**
# <details><summary>Answer</summary>
# 5 or 10 are standard. Larger k means less pessimistic bias (each model trains on more data) but higher compute and more correlated fold models; leave-one-out is the extreme and is
# rarely worth it. With small data use repeated CV rather than very large k.
# </details>
#
# **3. What is nested cross-validation for?**
# <details><summary>Answer</summary>
# To get an unbiased estimate of performance when hyperparameters are tuned with CV. The inner loop chooses hyperparameters; the outer loop scores the whole procedure on data the tuner never saw.
# </details>
#
# **4. Your CV AUC is 0.85 but the AUC after deployment is 0.70. Possible causes?**
# <details><summary>Answer</summary>
# Leakage in the CV set-up (preprocessing outside folds, target-derived features, duplicates or groups across folds), time-ordered data validated with random folds, distribution shift between
# training data and production, or a training/serving feature mismatch. Check each, starting with how the folds were built.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Cross-validate by hand
# Using `StratifiedKFold(n_splits=4, shuffle=True, random_state=0)` on `X_train`/`y_train` and the pipeline `make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))`,
# write the fold loop yourself (fit on the training fold, `roc_auc_score` on the validation fold). Store the list of 4 scores in `fold_aucs` and their mean in `cv_mean`.

# %% tags=["starter"]
fold_aucs = ...
cv_mean = ...

# %%
_ref = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X_train, y_train,
                       cv=StratifiedKFold(4, shuffle=True, random_state=0), scoring="roc_auc")
assert len(fold_aucs) == 4, "one score per fold"
assert np.allclose(fold_aucs, _ref), "fold scores should match cross_val_score with the same splitter"
assert abs(cv_mean - _ref.mean()) < 1e-9, "cv_mean is the mean of fold_aucs"
print("Exercise 1 passed:", np.round(fold_aucs, 4))

# %% tags=["solution"]
fold_aucs = []
for tr_i, va_i in StratifiedKFold(4, shuffle=True, random_state=0).split(X_train, y_train):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_train.iloc[tr_i], y_train.iloc[tr_i])
    fold_aucs.append(roc_auc_score(y_train.iloc[va_i], m.predict_proba(X_train.iloc[va_i])[:, 1]))
cv_mean = float(np.mean(fold_aucs))

# %% [markdown]
# ### Exercise 2: How much does stratification matter?
# On the 150-row sample `small` (from section 2), compute the **largest absolute deviation** between a fold's positive rate and the overall rate, for
# `KFold(5, shuffle=True, random_state=42)` (`dev_plain`) and `StratifiedKFold(5, shuffle=True, random_state=42)` (`dev_strat`).

# %% tags=["starter"]
dev_plain = ...
dev_strat = ...

# %%
def _dev(splitter):
    return max(abs(small.iloc[va].mean() - small.mean()) for _, va in splitter.split(Xs_small, small))
assert abs(dev_plain - _dev(KFold(5, shuffle=True, random_state=42))) < 1e-9, "max |fold rate - overall rate| for plain KFold"
assert abs(dev_strat - _dev(StratifiedKFold(5, shuffle=True, random_state=42))) < 1e-9, "same for StratifiedKFold (pass `small` as y to split)"
assert dev_strat < dev_plain, "stratified folds should be closer to the overall rate"
print(f"Exercise 2 passed: max deviation {dev_plain:.3f} (plain) vs {dev_strat:.3f} (stratified)")

# %% tags=["solution"]
overall = small.mean()
dev_plain = max(abs(small.iloc[va].mean() - overall) for _, va in KFold(5, shuffle=True, random_state=42).split(Xs_small, small))
dev_strat = max(abs(small.iloc[va].mean() - overall) for _, va in StratifiedKFold(5, shuffle=True, random_state=42).split(Xs_small, small))

# %% [markdown]
# ### Exercise 3: Expose a leaky CV
# With the doubled data `Xd`, `yd`, `groups` from section 2, cross-validate a `DecisionTreeClassifier(max_depth=10, random_state=42)` with ROC-AUC using (a) `KFold(5, shuffle=True, random_state=42)` -> `auc_leaky`
# and (b) `GroupKFold(5)` with the customer groups -> `auc_grouped`. Store `inflation = auc_leaky - auc_grouped`.

# %% tags=["starter"]
auc_leaky = ...
auc_grouped = ...
inflation = ...

# %%
_t = DecisionTreeClassifier(max_depth=10, random_state=42)
_a = cross_val_score(_t, Xd, yd, cv=KFold(5, shuffle=True, random_state=42), scoring="roc_auc").mean()
_b = cross_val_score(_t, Xd, yd, cv=GroupKFold(5), groups=groups, scoring="roc_auc").mean()
assert abs(auc_leaky - _a) < 1e-9 and abs(auc_grouped - _b) < 1e-9, "use scoring='roc_auc', mean over folds, and groups=groups for GroupKFold"
assert abs(inflation - (_a - _b)) < 1e-9 and inflation > 0, "the leaky scheme should score higher"
print(f"Exercise 3 passed: leaky {auc_leaky:.3f} vs grouped {auc_grouped:.3f} (inflation {inflation:.3f})")

# %% tags=["solution"]
_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
auc_leaky = float(cross_val_score(_tree, Xd, yd, cv=KFold(5, shuffle=True, random_state=42), scoring="roc_auc").mean())
auc_grouped = float(cross_val_score(_tree, Xd, yd, cv=GroupKFold(5), groups=groups, scoring="roc_auc").mean())
inflation = auc_leaky - auc_grouped

# %% [markdown]
# ### Exercise 4: A paired comparison
# Use `RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=7)` on `X_train`/`y_train` to score two pipelines with ROC-AUC: `lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))` and
# `rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=1)`. Store the mean paired difference (lr minus rf) in `mean_diff` and the number of the 10 folds
# in which the logistic model scored higher in `lr_wins`.

# %% tags=["starter"]
mean_diff = ...
lr_wins = ...

# %%
_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=7)
_lr = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X_train, y_train, cv=_cv, scoring="roc_auc")
_rf = cross_val_score(RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=1), X_train, y_train, cv=_cv, scoring="roc_auc")
assert abs(mean_diff - (_lr - _rf).mean()) < 1e-9, "mean of per-fold differences (lr - rf)"
assert lr_wins == int((_lr > _rf).sum()), "count folds where lr > rf"
print(f"Exercise 4 passed: mean diff {mean_diff:+.4f}; logistic wins {lr_wins}/10 folds")

# %% tags=["solution"]
_cv_ = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=7)
lr_scores = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X_train, y_train, cv=_cv_, scoring="roc_auc")
rf_scores = cross_val_score(RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=1), X_train, y_train, cv=_cv_, scoring="roc_auc")
mean_diff = float((lr_scores - rf_scores).mean())
lr_wins = int((lr_scores > rf_scores).sum())

# %% [markdown]
# ### Exercise 5 (open-ended): Learning curve for boosting
# Draw a learning curve (as in section 3) for `HistGradientBoostingClassifier(learning_rate=0.05, max_iter=150, random_state=42)`. Write two sentences: is the gap between train and validation large or small, and
# would you expect more data to help? Then say which model of the three in section 4 you would ship and *why*, mentioning both the score difference and one non-accuracy factor (latency, interpretability, maintenance).

# %% [markdown]
# ## Recap
# - CV = fit on k-1 folds, score on the held-out fold, repeat; report **mean and spread**. Keep preprocessing inside a pipeline.
# - Match the splitter to reality: stratified (class balance), repeated (small data), group (shared entities), time-series (forecasting).
# - Learning curves separate high variance (gap) from high bias (converged and low).
# - Compare models on identical folds, use paired differences, and remember CV scores are correlated; nested CV gives an honest estimate when tuning.
# - Final recipe: choose by CV on train, refit, evaluate once on the held-out test.
#
# **Go deeper:** scikit-learn user guide "Cross-validation: evaluating estimator performance" and "Tuning the hyper-parameters of an estimator" (scikit-learn.org/stable/modules/cross_validation.html).
#
# **Next:** [Feature engineering and pipelines](08_feature_engineering_and_pipelines.ipynb)

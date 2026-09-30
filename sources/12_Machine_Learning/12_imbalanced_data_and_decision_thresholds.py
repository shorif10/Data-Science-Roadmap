# %% [markdown]
# # Imbalanced Data and Decision Thresholds
#
# **Why this matters.** Fraud, machine failure, a customer who will leave this month: the events you care about are
# usually rare. A model that says "nothing will happen" is 95% accurate and completely useless, and a tutorial trick such as
# "just oversample" can silently wreck your validation score. Interviewers love this topic because the
# right answer is mostly about *evaluation and decisions*, not about fancy resampling algorithms.
#
# **You will learn**
# - see why accuracy and ROC-AUC flatter a rare-event model, and use PR-AUC against its prevalence floor
# - compare class weights, oversampling and plain training, and know what each one changes
# - resample **only inside the training part of each CV fold**, and measure the leak when you do not
# - turn an explicit cost table into a decision threshold, chosen without touching the test set
# - read a calibration curve and fix probabilities with `CalibratedClassifierCV`
#
# **Prerequisites:** [Logistic regression and metrics](04_logistic_regression_and_classification_metrics.ipynb) and
# [Cross-validation](07_cross_validation_and_model_evaluation.ipynb).
#
# > Data used: `datasets/churn.csv` (synthetic). The real churn rate is about 37%, which is not rare, so we
# > **deliberately make it rare**: keep every non-churner but only a random 10% of churners. It is a simulation of "a rare event",
# > not a statement about real churn.

# %%
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path("../datasets")
raw = pd.read_csv(DATA / "churn.csv", keep_default_na=False, na_values=[""])
raw["contract"] = raw["contract"].str.lower().str.strip()
raw["internet_service"] = raw["internet_service"].fillna("Unknown")

rare = pd.concat([raw[raw["churned"] == 0], raw[raw["churned"] == 1].sample(frac=0.10, random_state=42)])
rare = rare.sample(frac=1, random_state=42).reset_index(drop=True)
X = rare.drop(columns=["customer_id", "churned"])
y = rare["churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
print(f"rows {len(rare)}, positive rate {y.mean():.1%}; test set has {int(y_test.sum())} positives out of {len(y_test)}")

# %% [markdown]
# Note the test set has only a few dozen positives. That is realistic for rare events, and it means every test metric below is **noisy**; we
# will keep the decisions on the training side and look at the test set once.

# %% [markdown]
# ## 1. The accuracy trap and the prevalence floor
#
# Fit a model that always predicts the majority class, and compare with a real model.

# %%
num_cols = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "senior_citizen"]
cat_cols = ["contract", "internet_service", "paperless_billing", "autopay"]


def make_model(**kw):
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, **kw))])


dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
plain = make_model().fit(X_train, y_train)
p_plain = plain.predict_proba(X_test)[:, 1]

print(f"always 'no churn': accuracy {accuracy_score(y_test, dummy.predict(X_test)):.3f}, recall {recall_score(y_test, dummy.predict(X_test)):.2f}")
print(f"logistic @0.5    : accuracy {accuracy_score(y_test, p_plain >= 0.5):.3f}, recall {recall_score(y_test, p_plain >= 0.5):.2f}")
print(f"ROC-AUC {roc_auc_score(y_test, p_plain):.3f} | PR-AUC (average precision) {average_precision_score(y_test, p_plain):.3f} | prevalence floor {y_test.mean():.3f}")

# %% [markdown]
# *What just happened?* The do-nothing model scores about the same accuracy as the real one, and its recall is exactly 0. That is the accuracy trap.
# **ROC-AUC** looks comfortable because the huge negative class makes the false-positive *rate* tiny even when many alarms are wrong.
# **PR-AUC** (average precision) is anchored to the **prevalence**: a random ranking scores about the positive rate, so you must always report it
# next to that floor. The honest summary is "PR-AUC is X times the floor", and whatever you choose, measure it on a test set big enough to mean something.

# %% [markdown]
# ## 2. Three ways to "handle" imbalance
#
# 1. **Do nothing, choose the threshold well.** The model's ranking is usually fine; 0.5 is what is wrong.
# 2. **Class weights** (`class_weight="balanced"`): each positive row counts more in the loss. No new rows are created.
# 3. **Resampling**: duplicate minority rows (oversample) or drop majority rows (undersample) before training.
#
# `imbalanced-learn` (SMOTE and friends) is not installed here, and plain random oversampling is enough to show all the
# important issues. Write it in six lines:

# %%
def oversample(X_df, y_ser, seed=42):
    """Duplicate random positive rows until both classes are the same size. Returns new (X, y)."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y_ser.to_numpy() == 1)
    neg = np.flatnonzero(y_ser.to_numpy() == 0)
    extra = rng.choice(pos, size=len(neg) - len(pos), replace=True)
    idx = np.concatenate([np.arange(len(y_ser)), extra])
    return X_df.iloc[idx].reset_index(drop=True), y_ser.iloc[idx].reset_index(drop=True)


Xo, yo = oversample(X_train, y_train)
print("train positives before/after oversampling:", int(y_train.sum()), "->", int(yo.sum()), "of", len(yo))

variants = {
    "plain": make_model().fit(X_train, y_train),
    "class_weight=balanced": make_model(class_weight="balanced").fit(X_train, y_train),
    "oversampled": make_model().fit(Xo, yo),
}
rows = []
for name, m in variants.items():
    p = m.predict_proba(X_test)[:, 1]
    rows.append({"variant": name, "ROC-AUC": roc_auc_score(y_test, p), "PR-AUC": average_precision_score(y_test, p),
                 "recall@0.5": recall_score(y_test, p >= 0.5), "share flagged @0.5": float((p >= 0.5).mean())})
pd.DataFrame(rows).set_index("variant").round(3)

# %% [markdown]
# *What just happened?* The **ranking** quality (ROC-AUC, PR-AUC) barely moves. What changes is where the probabilities sit: weights and oversampling push them up,
# so a fixed 0.5 threshold flags many more customers and recall rises. In other words, for a model like logistic regression, reweighting is mostly a
# **threshold shift in disguise**, and it makes the probabilities lie (section 6). If you only need a good yes/no decision, you can often skip it
# and pick the threshold from costs. Weights and resampling matter more for flexible models, where minority rows otherwise barely influence the fit.

# %% [markdown]
# ## 3. Resample inside the folds, never before them
#
# The classic mistake: oversample the whole training set, *then* cross-validate. Duplicated positives land in both the training and validation parts of a fold, so
# a flexible model is being tested on rows it has memorised. Measure the damage with a random forest, which happily memorises duplicates. We compare the
# wrong order with the right one (resample the **training part of each fold only**, evaluate on the untouched fold):

# %%
Xd = pd.get_dummies(X_train, columns=cat_cols, dtype=float)  # one-hot with no fitted statistics; the forest handles NaN natively
skf = StratifiedKFold(5, shuffle=True, random_state=42)
rf = lambda: RandomForestClassifier(n_estimators=100, min_samples_leaf=1, random_state=42, n_jobs=1)

# WRONG: oversample first, then split
Xd_o, yd_o = oversample(Xd, y_train)
wrong = []
for tr, va in skf.split(Xd_o, yd_o):
    m = rf().fit(Xd_o.iloc[tr], yd_o.iloc[tr])
    wrong.append(average_precision_score(yd_o.iloc[va], m.predict_proba(Xd_o.iloc[va])[:, 1]))

# RIGHT: split first, oversample the training part of each fold only
right = []
for tr, va in skf.split(Xd, y_train):
    Xf, yf = oversample(Xd.iloc[tr], y_train.iloc[tr])
    m = rf().fit(Xf, yf)
    right.append(average_precision_score(y_train.iloc[va], m.predict_proba(Xd.iloc[va])[:, 1]))

print(f"PR-AUC, oversample THEN cross-validate : {np.mean(wrong):.3f}")
print(f"PR-AUC, oversample INSIDE each fold    : {np.mean(right):.3f}  (prevalence floor {y_train.mean():.3f})")

# %% [markdown]
# *What just happened?* The "wrong" number is wildly optimistic: it would make you ship a model that looks great and fails in production. It is the same lesson as notebook 02 (fit anything
# data-dependent on training data only), applied to resampling. The safest habit is to put the sampler **inside** the pipeline so the cross-validation machinery applies it to each training
# fold only. That is what `imblearn.pipeline.Pipeline` does. Without that library, the explicit loop above is the honest fallback. Also note the validation folds keep
# the **natural** class ratio: never rebalance what you evaluate on.

# %% [markdown]
# ## 4. Choose the threshold from an explicit cost table
#
# A rare-event model is useful only with a decision rule attached. Write your assumptions down (invented for this course; replace them with real numbers in real life):
#
# | outcome | cost |
# |---|---|
# | miss a customer who would have churned (FN) | **100** (lost margin) |
# | send a retention offer to someone who would have stayed (FP) | **10** |
# | correct flag (TP) or correct silence (TN) | 0 (we ignore the offer cost that saves a customer, to keep the maths simple) |
#
# If probabilities are calibrated, flag a customer when `p * 100 > (1 - p) * 10`, i.e. `p > 10 / (100 + 10)`, about 0.09. But our probabilities may not be calibrated, so we also
# find the best threshold empirically on **out-of-fold predictions for the training set** (no test data involved) and compare.

# %%
COST_FN, COST_FP = 100.0, 10.0
breakeven = COST_FP / (COST_FP + COST_FN)


def total_cost(y_true, p, t):
    pred = p >= t
    y_true = np.asarray(y_true)
    return COST_FN * np.sum(~pred & (y_true == 1)) + COST_FP * np.sum(pred & (y_true == 0))


oof = cross_val_predict(make_model(), X_train, y_train, cv=skf, method="predict_proba")[:, 1]
grid_t = np.linspace(0.02, 0.8, 79)
oof_cost = np.array([total_cost(y_train, oof, t) for t in grid_t])
t_star = float(grid_t[oof_cost.argmin()])

plt.figure(figsize=(7, 3.6))
plt.plot(grid_t, oof_cost)
plt.axvline(breakeven, color="grey", ls=":", label=f"break-even p = {breakeven:.2f}")
plt.axvline(t_star, color="tab:red", ls="--", label=f"best on out-of-fold = {t_star:.2f}")
plt.title("Total cost on out-of-fold training predictions vs threshold")
plt.xlabel("decision threshold")
plt.ylabel("total cost (assumed currency units)")
plt.legend()
plt.show()

for label, t in [("default 0.5", 0.5), (f"break-even {breakeven:.2f}", breakeven), (f"out-of-fold best {t_star:.2f}", t_star)]:
    print(f"test cost at {label:<24}: {total_cost(y_test, p_plain, t):8.0f}")
print(f"test cost of never flagging anyone   : {COST_FN * int(y_test.sum()):8.0f}")

# %% [markdown]
# *What just happened?* The threshold was chosen on data the test set never touched, then applied once. A low threshold wins because a miss is ten times costlier than a wasted offer, even though precision
# at that threshold is low. If the cost ratio changes, the threshold moves; the model does not need retraining. Keep the cost table next to the model so a stakeholder can challenge it. The test
# costs are noisy (few positives): in the run above the textbook break-even threshold even beat the empirically chosen one on the test set, which says more about 47 test positives than about either method.
# Both low thresholds beat the default 0.5, and that is the robust finding.

# %% [markdown]
# ## 5. Precision-recall curves
#
# A PR curve shows every achievable (recall, precision) pair as the threshold sweeps. Compare the plain and class-weighted logistic models, plus the prevalence floor:

# %%
p_bal = variants["class_weight=balanced"].predict_proba(X_test)[:, 1]
fig, ax = plt.subplots(figsize=(6, 4))
for label, p in [("plain", p_plain), ("balanced weights", p_bal)]:
    pr, rc, _ = precision_recall_curve(y_test, p)
    ax.plot(rc, pr, label=f"{label} (AP {average_precision_score(y_test, p):.2f})")
ax.axhline(y_test.mean(), color="grey", ls=":", label="random (prevalence)")
ax.set_title("Precision-recall curves on the test set")
ax.set_xlabel("recall")
ax.set_ylabel("precision")
ax.legend()
plt.show()

# %% [markdown]
# The two curves almost coincide: weighting did not create a better ranking, as section 2 said. A PR curve is also how you answer "if we want to catch 80% of churners, how many
# alarms will be wrong?": read the precision at recall 0.8. It is a **curve** you discuss with the business, not a single number.

# %% [markdown]
# ## 6. Calibration: are the probabilities honest?
#
# A model is **calibrated** if, among customers scored 0.3, about 30% really churn. Thresholds derived from costs assume this. A reliability diagram bins predictions and plots
# the observed rate against the mean prediction. We compare the plain model, the class-weighted one and a version wrapped in `CalibratedClassifierCV`:

# %%
cal = CalibratedClassifierCV(make_model(class_weight="balanced"), method="sigmoid", cv=5).fit(X_train, y_train)
p_cal = cal.predict_proba(X_test)[:, 1]

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot([0, 1], [0, 1], "k:", label="perfectly calibrated")
for label, p in [("plain", p_plain), ("balanced weights", p_bal), ("balanced + calibrated", p_cal)]:
    frac, mean_pred = calibration_curve(y_test, p, n_bins=6, strategy="quantile")
    ax.plot(mean_pred, frac, "o-", label=label)
ax.set_title("Reliability diagram (test set, quantile bins)")
ax.set_xlabel("mean predicted probability")
ax.set_ylabel("observed churn rate")
ax.legend()
plt.show()
print("Brier score (lower is better):", {n: round(brier_score_loss(y_test, p), 4) for n, p in [("plain", p_plain), ("balanced", p_bal), ("balanced+calibrated", p_cal)]})

# %% [markdown]
# *What just happened?* The plain model sits close to the diagonal (logistic regression is usually reasonably calibrated); the class-weighted one overstates risk, so its curve lies below the diagonal. Re-fitting
# a mapping on top (`CalibratedClassifierCV`, Platt scaling via `method="sigmoid"`, or `"isotonic"` for more data) pulls it back and the **Brier score** (mean squared error of the probabilities) falls. With only a few dozen test
# positives, the bins are noisy, so look at the overall pattern, not single points. Takeaway: **if you reweight or resample, recalibrate** before you use probabilities for costs or for ranking by expected value.

# %% [markdown]
# ## Common mistakes
#
# 1. **Reporting accuracy** (or ROC-AUC alone) for a rare event. Always show PR-AUC next to its floor, plus recall/precision at the operating threshold.
# 2. **Oversampling before splitting** (section 3): duplicated rows leak into validation.
# 3. **Resampling the validation or test set.** They must keep the real class ratio.
# 4. **Using 0.5 by default**, or tuning the threshold on the test set, then reporting that same test score.
# 5. **Using reweighted probabilities as if they were real probabilities** (expected-value maths, "this customer has a 70% risk").
# 6. **Stratifying only the final split**, then getting folds with 3 positives. Use `StratifiedKFold`, and with very few positives prefer repeated CV.

# %%
# Mistake 6: without stratification, folds can be badly unbalanced when positives are rare
from sklearn.model_selection import KFold

plain_folds = [int(y_train.iloc[va].sum()) for _, va in KFold(10, shuffle=True, random_state=1).split(X_train)]
strat_folds = [int(y_train.iloc[va].sum()) for _, va in StratifiedKFold(10, shuffle=True, random_state=1).split(X_train, y_train)]
print("positives per validation fold, plain KFold     :", plain_folds)
print("positives per validation fold, StratifiedKFold :", strat_folds)

# %% [markdown]
# ## Interview corner
#
# **1. Your fraud model has 99% accuracy. Is it good?**
# <details><summary>Answer</summary>
# Not necessarily: if 99% of transactions are legitimate, predicting "legitimate" always gives 99%. Ask for the prevalence, then look at precision and recall at the operating threshold, PR-AUC against
# the prevalence floor, and the cost of each error type.
# </details>
#
# **2. Is SMOTE or oversampling always a good idea?**
# <details><summary>Answer</summary>
# No. It must be applied only to training data inside each CV fold (otherwise validation leaks), it changes the predicted probabilities (so recalibrate if you use them), and for ranking-based models it often gives
# no gain over simply tuning the threshold. It can help flexible models that otherwise ignore the minority class, but test it with honest validation.
# </details>
#
# **3. How do you pick a classification threshold?**
# <details><summary>Answer</summary>
# From the business costs: flag when expected cost of ignoring exceeds expected cost of acting. With calibrated probabilities that is <code>p &gt; C_FP / (C_FP + C_FN)</code>; otherwise sweep thresholds on
# out-of-fold or validation predictions and minimise total cost. Never pick it on the test set.
# </details>
#
# **4. ROC-AUC or PR-AUC for a rare positive class?**
# <details><summary>Answer</summary>
# PR-AUC is more informative when positives are rare and you care about the positive class, because precision reflects the flood of false positives that ROC hides. ROC-AUC is insensitive to prevalence, which is good for
# comparing ranking ability across datasets but can make a weak rare-event model look fine.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: How much better than luck?
# Using `p_plain` and `y_test`, store the model's test PR-AUC (`average_precision_score`) in `ap`, the prevalence floor (share of positives in `y_test`) in `floor`, and their ratio
# in `lift_over_floor`. Also store in `dummy_accuracy` the accuracy of always predicting 0 on `y_test` (compute it from `y_test`, not from the dummy model).

# %% tags=["starter"]
ap = ...
floor = ...
lift_over_floor = ...
dummy_accuracy = ...

# %%
_ap = average_precision_score(y_test, p_plain)
assert abs(ap - _ap) < 1e-9, "ap = average_precision_score(y_test, p_plain)"
assert abs(floor - y_test.mean()) < 1e-9, "floor = share of positives in y_test"
assert abs(lift_over_floor - _ap / y_test.mean()) < 1e-9, "lift = ap / floor"
assert abs(dummy_accuracy - (1 - y_test.mean())) < 1e-9, "always predicting 0 is right on every negative row"
print(f"Exercise 1 passed: PR-AUC is {lift_over_floor:.1f}x the floor, while 'never churn' gets {dummy_accuracy:.1%} accuracy")

# %% tags=["solution"]
ap = float(average_precision_score(y_test, p_plain))
floor = float(y_test.mean())
lift_over_floor = ap / floor
dummy_accuracy = float((y_test == 0).mean())

# %% [markdown]
# ### Exercise 2: A different cost table, a different threshold
# The retention team finds that an offer costs **25** (not 10) while a missed churner still costs **100**. Using the out-of-fold probabilities `oof` and `grid_t` from section 4 (do not use the test set to choose),
# store the break-even probability in `t_breakeven_new`, the threshold on `grid_t` that minimises total out-of-fold cost under the new costs in `t_best_new`, and the **test-set** cost of using
# that threshold with `p_plain` in `test_cost_new`.
# *Hint: `total_cost` reads the globals `COST_FN`/`COST_FP`, so write your own cost inside the exercise instead of overwriting them.*

# %% tags=["starter"]
t_breakeven_new = ...
t_best_new = ...
test_cost_new = ...

# %%
def _cost(y_true, p, t, fn=100.0, fp=25.0):
    pred, yy = p >= t, np.asarray(y_true)
    return fn * np.sum(~pred & (yy == 1)) + fp * np.sum(pred & (yy == 0))


_costs = np.array([_cost(y_train, oof, t) for t in grid_t])
_best = float(grid_t[_costs.argmin()])
assert abs(t_breakeven_new - 25 / 125) < 1e-9, "break-even = C_FP / (C_FP + C_FN)"
assert abs(t_best_new - _best) < 1e-9, f"minimise out-of-fold cost under the new costs; expected {_best:.2f}"
assert t_best_new != t_star, "the new cost table must move the threshold compared with section 4"
assert abs(test_cost_new - _cost(y_test, p_plain, _best)) < 1e-9, "apply your chosen threshold to the test predictions once"
print(f"Exercise 2 passed: threshold moves from {t_star:.2f} to {t_best_new:.2f}; test cost {test_cost_new:.0f}")

# %% tags=["solution"]
def new_cost(y_true, p, t, fn=100.0, fp=25.0):
    pred, yy = p >= t, np.asarray(y_true)
    return fn * np.sum(~pred & (yy == 1)) + fp * np.sum(pred & (yy == 0))


t_breakeven_new = 25 / (25 + 100)
new_costs = np.array([new_cost(y_train, oof, t) for t in grid_t])
t_best_new = float(grid_t[new_costs.argmin()])
test_cost_new = float(new_cost(y_test, p_plain, t_best_new))

# %% [markdown]
# ### Exercise 3: Oversample inside the folds
# Write the correct evaluation for a plain **logistic regression pipeline** (`make_model()`): for each split of `skf.split(X_train, y_train)`, oversample the *training part only* with `oversample(...)`, fit,
# and compute the average precision on the *untouched* validation part. Store the list of five fold scores in `fold_ap` and their mean in `mean_ap`.

# %% tags=["starter"]
fold_ap = ...
mean_ap = ...

# %%
_ref = []
for _tr, _va in skf.split(X_train, y_train):
    _xf, _yf = oversample(X_train.iloc[_tr], y_train.iloc[_tr])
    _m = make_model().fit(_xf, _yf)
    _ref.append(average_precision_score(y_train.iloc[_va], _m.predict_proba(X_train.iloc[_va])[:, 1]))
assert len(fold_ap) == 5, "one score per fold"
assert np.allclose(fold_ap, _ref), "oversample only X_train.iloc[tr]; score on X_train.iloc[va] unchanged"
assert abs(mean_ap - np.mean(_ref)) < 1e-9, "mean_ap = mean of the five fold scores"
print(f"Exercise 3 passed: honest PR-AUC {mean_ap:.3f} (floor {y_train.mean():.3f})")

# %% tags=["solution"]
fold_ap = []
for tr, va in skf.split(X_train, y_train):
    Xf, yf = oversample(X_train.iloc[tr], y_train.iloc[tr])
    m = make_model().fit(Xf, yf)
    fold_ap.append(average_precision_score(y_train.iloc[va], m.predict_proba(X_train.iloc[va])[:, 1]))
mean_ap = float(np.mean(fold_ap))

# %% [markdown]
# ### Exercise 4: Does calibration help?
# Fit `CalibratedClassifierCV(make_model(class_weight="balanced"), method="isotonic", cv=5)` on the training data and store the **test** Brier score in `brier_isotonic`. Store the Brier score of the uncalibrated
# balanced model (`p_bal`) in `brier_balanced`, and set `isotonic_better` to whether the calibrated score is lower.

# %% tags=["starter"]
brier_isotonic = ...
brier_balanced = ...
isotonic_better = ...

# %%
_iso = CalibratedClassifierCV(make_model(class_weight="balanced"), method="isotonic", cv=5).fit(X_train, y_train)
_b = brier_score_loss(y_test, _iso.predict_proba(X_test)[:, 1])
assert abs(brier_isotonic - _b) < 1e-9, "Brier score of the calibrated model's positive-class probability on the test set"
assert abs(brier_balanced - brier_score_loss(y_test, p_bal)) < 1e-9, "brier_score_loss(y_test, p_bal)"
assert bool(isotonic_better) == bool(_b < brier_score_loss(y_test, p_bal)), "compare the two scores"
print(f"Exercise 4 passed: Brier {brier_balanced:.4f} -> {brier_isotonic:.4f} (calibrated is better: {isotonic_better})")

# %% tags=["solution"]
iso = CalibratedClassifierCV(make_model(class_weight="balanced"), method="isotonic", cv=5).fit(X_train, y_train)
brier_isotonic = float(brier_score_loss(y_test, iso.predict_proba(X_test)[:, 1]))
brier_balanced = float(brier_score_loss(y_test, p_bal))
isotonic_better = bool(brier_isotonic < brier_balanced)

# %% [markdown]
# ### Exercise 5 (open-ended): Make your own recommendation
# Write a 4-line recommendation for the retention manager: which variant (plain, weighted, oversampled, calibrated), which threshold and why, what PR-AUC versus the floor you can honestly claim,
# and one thing about the small number of test positives that limits your confidence. What would you do to get a tighter estimate (repeated CV, more data, bootstrap interval)?

# %% [markdown]
# ## Recap
# - Accuracy and ROC-AUC flatter rare-event models; report PR-AUC next to the prevalence floor and recall/precision at the operating threshold.
# - Weights and oversampling mainly shift probabilities (a threshold change in disguise) for linear models; they do not reliably improve ranking.
# - Resample only inside training folds and keep validation and test sets at the natural class ratio.
# - Derive the threshold from an explicit cost table, select it on out-of-fold data, and look at the test set once.
# - If you reweight or resample, check calibration before using probabilities for expected-value decisions.
#
# **Go deeper:** scikit-learn user guide sections "Tuning the decision threshold for class prediction", "Probability calibration" and "Metrics and scoring"
# (scikit-learn.org/stable/); the `imbalanced-learn` documentation if you need SMOTE-style samplers with pipeline support (not installed here).
#
# **Next:** [Model interpretation and explainability](13_model_interpretation_and_explainability.ipynb)

# %% [markdown]
# # Capstone 4: Modelling, Tuning and Threshold Selection
#
# **Why this matters.** A model outputs a probability; the business needs a *decision*. Turning one into the other is where
# the money is made or lost, and it is usually done with a default (`0.5`) that nobody chose. Here you compare candidate models honestly,
# tune with restraint, and pick the threshold that minimises the cost matrix from notebook 01.
#
# **You will learn**
# - compare a linear model and gradient boosting on the same split, and judge whether the gap is real
# - tune hyper-parameters with cross-validation on train only (never on test)
# - pick the decision threshold by minimising expected cost, and see how noisy that choice is
# - watch the threshold, and the value of the model itself, move when the cost assumptions change
#
# **Prerequisites:** [Notebook 03](03_features_and_baseline_model.ipynb).

# %%
import sys
import tempfile
from pathlib import Path

SRC = Path("../src").resolve()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from churn import train
from churn.config import load_config, resolve
from churn.data import FEATURE_COLUMNS, TARGET, load_dataset
from churn.features import build_pipeline, make_estimator

CFG = load_config("../configs/config.yaml")
COSTS = CFG["costs"]
df = load_dataset(resolve(CFG, CFG["paths"]["data"]))
train_df, val_df, _test_df = train.split_data(df, CFG)  # test stays untouched until notebook 05
X_tr, y_tr = train_df[FEATURE_COLUMNS], train_df[TARGET]
X_va, y_va = val_df[FEATURE_COLUMNS], val_df[TARGET]
CV = StratifiedKFold(5, shuffle=True, random_state=42)

# %% [markdown]
# ## 1. Two candidates, one validation set
# The config lists two candidates. Fit both on train, score both on validation.

# %%
def fit_score(name, params=None):
    pipe = build_pipeline(make_estimator(name, params or CFG["models"][name], 42)).fit(X_tr, y_tr)
    p = pipe.predict_proba(X_va)[:, 1]
    return pipe, p, {"val_roc_auc": roc_auc_score(y_va, p), "val_avg_precision": average_precision_score(y_va, p),
                     "val_brier": brier_score_loss(y_va, p)}

logit, p_logit, s_logit = fit_score("logistic")
hgb, p_hgb, s_hgb = fit_score("hgb")
pd.DataFrame({"logistic": s_logit, "hgb (config params)": s_hgb}).T.round(4)

# %% [markdown]
# On this data the two are close, and the linear model is not behind. Is the gap even meaningful? Bootstrap the validation set and look at the
# spread of the **difference in AUC**.

# %%
rng = np.random.default_rng(42)
diffs = []
for _ in range(300):
    i = rng.integers(0, len(y_va), len(y_va))
    yb = y_va.to_numpy()[i]
    diffs.append(roc_auc_score(yb, p_hgb[i]) - roc_auc_score(yb, p_logit[i]))
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"AUC(hgb) - AUC(logistic): 95% bootstrap interval [{lo:.3f}, {hi:.3f}]")

# %% [markdown]
# If the interval contains 0, you cannot say one model is better. When two models tie, **pick the simpler one** (cheaper to serve, easier to explain, better calibrated by default).

# %% [markdown]
# ## 2. Tuning with restraint
# Tuning uses cross-validation **inside the training set**; the validation set stays a clean referee. Logistic regression has one knob that matters (`C`, the inverse regularisation strength).

# %%
logit_grid = GridSearchCV(build_pipeline(make_estimator("logistic", {"max_iter": 1000}, 42)),
                          {"model__C": [0.01, 0.1, 1, 10]}, cv=CV, scoring="roc_auc").fit(X_tr, y_tr)
pd.DataFrame(logit_grid.cv_results_)[["param_model__C", "mean_test_score", "std_test_score"]].round(4)

# %% [markdown]
# Boosting has many knobs; a small **random search** covers them better than a coarse grid for the same budget. We keep the budget small on purpose:
# more trials on the same data mostly finds noise.

# %%
hgb_search = RandomizedSearchCV(
    build_pipeline(make_estimator("hgb", {"max_iter": 150}, 42)),
    {"model__learning_rate": loguniform(0.02, 0.2), "model__max_depth": randint(2, 6),
     "model__min_samples_leaf": randint(20, 120), "model__l2_regularization": loguniform(0.1, 10)},
    n_iter=8, cv=StratifiedKFold(3, shuffle=True, random_state=42), scoring="roc_auc", random_state=42, n_jobs=1,
).fit(X_tr, y_tr)
print("best CV AUC:", round(hgb_search.best_score_, 4), hgb_search.best_params_)
p_tuned = hgb_search.predict_proba(X_va)[:, 1]
print("tuned hgb  val AUC:", round(roc_auc_score(y_va, p_tuned), 4), "   config hgb val AUC:", round(s_hgb["val_roc_auc"], 4),
      "   logistic val AUC:", round(s_logit["val_roc_auc"], 4))

# %% [markdown]
# *What just happened?* Search picked the best of 8 random settings by 3-fold CV on train; then we looked at validation **once**. Compare the three validation AUCs:
# tuning may move boosting by a point or so, and it stays within a few points of the linear model. The honest summary is "no meaningful winner, so keep the simple
# logistic pipeline that `configs/config.yaml` already selects". We do not write tuned values into the config just because they exist.

# %% [markdown]
# ## 3. The decision threshold from costs
# We use the logistic model's validation probabilities. `train.cost_curve` computes the expected cost per customer at every threshold from 0.02 to 0.98.

# %%
curve = train.cost_curve(y_va, p_logit, COSTS)
t_best = train.choose_threshold(y_va, p_logit, COSTS)
t_theory = train.theoretical_threshold(COSTS)
t_f1 = max(train.THRESHOLD_GRID, key=lambda t: f1_score(y_va, p_logit >= t))

def cost_at(t):
    return train.cost_per_customer(y_va, p_logit >= t, COSTS)

plt.figure(figsize=(7, 4))
plt.plot(curve["threshold"], curve["cost_per_customer"], color="black")
for t, label, style in [(t_best, "cost-optimal", "-"), (t_theory, "formula (calibrated)", "--"), (0.5, "default 0.5", ":"), (t_f1, "F1-optimal", "-.")]:
    plt.axvline(t, linestyle=style, label=f"{label}: {t:.2f} (cost {cost_at(t):.1f})", color=None)
plt.xlabel("decision threshold on P(churn)")
plt.ylabel("expected cost per customer (validation)")
plt.title("Cost as a function of the threshold")
plt.legend(fontsize=8)
plt.show()

# %% [markdown]
# Compare the four thresholds. The **cost-optimal** one is the lowest point by construction; the formula value from notebook 01 lands close to it because logistic
# regression is roughly calibrated; **0.5** and the **F1-optimal** threshold are worse *for this business*, because F1 treats a false positive and a false negative as equally bad.
# Never present a threshold without the cost matrix that justifies it.

# %% [markdown]
# ### How stable is that threshold?
# The curve is flat near its minimum, so the exact argmin on 1,000 validation rows is partly luck. Bootstrap the validation set and re-choose the threshold each time.

# %%
rng = np.random.default_rng(0)
boot = []
for _ in range(200):
    i = rng.integers(0, len(y_va), len(y_va))
    boot.append(train.choose_threshold(y_va.to_numpy()[i], p_logit[i], COSTS))
print(f"chosen {t_best:.2f}; bootstrap 5th-95th percentile: {np.percentile(boot, 5):.2f} to {np.percentile(boot, 95):.2f}")

# %% [markdown]
# Expect a spread of several hundredths. The right reading: "flag customers above roughly 0.2 to 0.3", not "0.26 is exactly optimal". Because the cost curve is flat there, the
# cost difference between thresholds inside that band is small, which is why a slightly imprecise threshold is not a disaster. More validation data (or cross-validated predictions) narrows it.

# %% [markdown]
# ## 4. When do the costs change the answer?
# Same model, five different retention offers. The offer cost changes `fp`; the churner value and save rate stay fixed. `tp = offer + 0.6 * 200`.

# %%
rows = []
for offer in (5, 10, 20, 40, 60):
    c = {"tn": 0.0, "fp": float(offer), "fn": 200.0, "tp": offer + 0.6 * 200.0}
    t = train.choose_threshold(y_va, p_logit, c)
    model_cost = train.cost_per_customer(y_va, p_logit >= t, c)
    none = train.cost_per_customer(y_va, np.zeros(len(y_va), bool), c)
    everyone = train.cost_per_customer(y_va, np.ones(len(y_va), bool), c)
    rows.append({"offer_cost": offer, "formula_threshold": train.theoretical_threshold(c), "chosen_threshold": t,
                 "flag_rate": (p_logit >= t).mean(), "model_cost": model_cost, "contact_nobody": none,
                 "contact_everybody": everyone, "saving_vs_best_dumb_policy": min(none, everyone) - model_cost})
sensitivity = pd.DataFrame(rows).set_index("offer_cost")
sensitivity.round(3)

# %% [markdown]
# Two lessons in one table:
# 1. **The threshold rises with the offer cost**: an expensive offer means you can only afford to contact people who are quite likely to leave.
# 2. **The model's value depends on the costs, not only on its AUC.** When offers are cheap, "contact everybody" is almost as good as any model
#    (look at the last column shrink). The model earns its keep when contacting has a real cost. A stakeholder who says "just call everyone" may be right at the cheap end.
#    Report this sensitivity, not one number.

# %% [markdown]
# ## 5. The package does the same thing
# `python -m churn.train` runs exactly this logic (fit candidates on train, select on validation, threshold from costs, test once). Run it into a temporary folder and compare
# with what we computed by hand.

# %%
with tempfile.TemporaryDirectory() as tmp:
    m = train.run_training(CFG, output_dir=tmp)
print("selected:", m["selected_model"], "| threshold:", m["threshold"]["value"], "| formula:", round(m["threshold"]["theoretical_if_calibrated"], 3))
assert m["selected_model"] == "logistic"
assert abs(m["candidates"]["logistic"]["val_roc_auc"] - s_logit["val_roc_auc"]) < 1e-9
assert m["threshold"]["value"] == t_best, "package and notebook must agree on the threshold"
print("hand-built and packaged results match")

# %% [markdown]
# ## Common mistakes
#
# 1. **Tuning on the test set** (or on validation, then reporting validation). Tune with CV on train; use validation to choose; report test once.
# 2. **Choosing the model with the best F1 at 0.5.** Compare ranking (AUC / AP) first, then apply the cost-based threshold.
# 3. **Resampling or class weights *and* a cost threshold.** Both shift the probabilities; the formula and the empirical curve then no longer mean what you think.
# 4. **A threshold picked on the test set.** It becomes part of the model and the test score is optimistic.
# 5. **Believing a tiny AUC gap.** Bootstrap the difference before you celebrate.

# %% [markdown]
# ## Interview corner
#
# **1. How do you choose a threshold for a fraud / churn / medical model?**
# <details><summary>Answer</summary>
# From the cost (or utility) of each outcome, evaluated on data the model was not trained on. Sweep thresholds, compute expected cost, pick the minimum, and report the sensitivity
# to the cost assumptions. Check calibration, because the closed-form break-even assumes calibrated probabilities.
# </details>
#
# **2. Grid search vs random search?**
# <details><summary>Answer</summary>
# Random search explores more distinct values of each important hyper-parameter for the same budget, which usually matters because only a few parameters drive performance.
# Grid search is fine for one or two knobs (like `C`).
# </details>
#
# **3. Two models have AUC 0.80 and 0.81. Which do you ship?**
# <details><summary>Answer</summary>
# Test whether the difference exceeds noise (bootstrap or paired CV). If not, ship the simpler, cheaper-to-serve one. Also compare on the business metric and calibration.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: What does 0.5 cost you?
# Using the logistic validation probabilities `p_logit` and `COSTS`, store the expected cost per customer at threshold **0.5** in `cost_half` and at the cost-optimal
# threshold in `cost_opt`. Store the money left on the table per customer in `waste_per_customer` (`cost_half - cost_opt`).

# %% tags=["starter"]
cost_half, cost_opt, waste_per_customer = ..., ..., ...

# %%
def _manual(t):
    flag = p_logit >= t
    y = y_va.to_numpy().astype(bool)
    return ((~y & flag).sum() * 20 + (y & ~flag).sum() * 200 + (y & flag).sum() * 140) / len(y)

assert abs(cost_half - _manual(0.5)) < 1e-9, "cost_half: flag customers with p >= 0.5"
assert abs(cost_opt - _manual(t_best)) < 1e-9, "cost_opt: use train.choose_threshold"
assert abs(waste_per_customer - (cost_half - cost_opt)) < 1e-9 and waste_per_customer >= 0
print(f"Exercise 1 passed: 0.5 wastes {waste_per_customer:.2f} per customer on validation")

# %% tags=["solution"]
cost_half = train.cost_per_customer(y_va, p_logit >= 0.5, COSTS)
cost_opt = train.cost_per_customer(y_va, p_logit >= train.choose_threshold(y_va, p_logit, COSTS), COSTS)
waste_per_customer = cost_half - cost_opt

# %% [markdown]
# ### Exercise 2: An expensive offer
# The retention team switches to a premium offer costing **60** (`fp = 60`, `tp = 60 + 0.6 * 200 = 180`, `fn = 200`, `tn = 0`). Build `premium_costs`, store the new
# cost-optimal threshold in `t_premium` and the old one in `t_old` (the original `COSTS`). Then store in `flag_more_customers` whether the premium offer makes you flag **more** customers on validation.

# %% tags=["starter"]
premium_costs = ...
t_premium, t_old, flag_more_customers = ..., ..., ...

# %%
assert premium_costs == {"tn": 0.0, "fp": 60.0, "fn": 200.0, "tp": 180.0}
def _brute(c):
    best_t, best = None, np.inf
    for t in np.round(np.arange(0.02, 0.981, 0.01), 2):
        flag = p_logit >= t
        y = y_va.to_numpy().astype(bool)
        cost = ((~y & flag).sum() * c["fp"] + (y & ~flag).sum() * c["fn"] + (y & flag).sum() * c["tp"]) / len(y)
        if cost < best - 1e-12:
            best_t, best = float(t), cost
    return best_t
assert t_premium == _brute(premium_costs) and t_old == _brute(COSTS)
assert t_premium > t_old, "an expensive offer should raise the threshold"
assert flag_more_customers == ((p_logit >= t_premium).sum() > (p_logit >= t_old).sum()) and flag_more_customers is False
print(f"Exercise 2 passed (threshold {t_old:.2f} -> {t_premium:.2f})")

# %% tags=["solution"]
premium_costs = {"tn": 0.0, "fp": 60.0, "fn": 200.0, "tp": 60.0 + 0.6 * 200.0}
t_premium = train.choose_threshold(y_va, p_logit, premium_costs)
t_old = train.choose_threshold(y_va, p_logit, COSTS)
flag_more_customers = bool((p_logit >= t_premium).sum() > (p_logit >= t_old).sum())

# %% [markdown]
# ### Exercise 3: Losing a customer gets more expensive
# For `fn` in `[100, 200, 400, 800]` (keep `tn = 0`, `fp = 20`, `tp = 140`; note `fn` must stay above `tp`, so start the list at 200 if `check_costs` complains and explain why)
# compute the cost-optimal threshold on validation. Store the list in `thresholds_by_fn`, using `None` for any `fn` that `check_costs` rejects. What happens as `fn` grows?

# %% tags=["starter"]
thresholds_by_fn = ...

# %%
def _expected():
    out = []
    for fn in (100, 200, 400, 800):
        c = {"tn": 0.0, "fp": 20.0, "fn": float(fn), "tp": 140.0}
        out.append(train.choose_threshold(y_va, p_logit, c) if fn > 140 else None)
    return out
exp = _expected()
assert thresholds_by_fn == exp, f"expected {exp}"
valid = [t for t in thresholds_by_fn if t is not None]
assert valid == sorted(valid, reverse=True), "a costlier miss should never raise the threshold"
assert thresholds_by_fn[0] is None, "fn=100 < tp=140 means acting on a churner is worse than doing nothing: the matrix is invalid"
print("Exercise 3 passed:", thresholds_by_fn)

# %% tags=["solution"]
thresholds_by_fn = []
for fn in (100, 200, 400, 800):
    c = {"tn": 0.0, "fp": 20.0, "fn": float(fn), "tp": 140.0}
    try:
        train.check_costs(c)
        thresholds_by_fn.append(train.choose_threshold(y_va, p_logit, c))
    except ValueError:
        thresholds_by_fn.append(None)

# %% [markdown]
# ### Exercise 4 (open-ended): When should we stop calling people?
# Using the sensitivity table in section 4 (or extending it), find roughly at what offer cost the saving over the best dumb policy becomes too small to justify running a model
# in production (define "too small" yourself and say what maintenance cost you assumed). Write two sentences a product manager could act on. Stretch: change `fn` (churner value) in the loop too.

# %% [markdown]
# ## Recap
# - Compare models on validation with an eye on noise (bootstrap the difference); ties go to the simpler model.
# - Tune with CV on train only, keep the budget small, and do not overwrite the config with noise-level gains.
# - Choose the threshold by minimising cost; the closed-form value is a sanity check that assumes calibration; 0.5 and F1 are not business-aware.
# - The threshold is a noisy estimate and moves with the cost matrix, and the model's value depends on the costs. Report the sensitivity.
#
# **Go deeper:** scikit-learn "Tuning the decision threshold" (`TunedThresholdClassifierCV`), "Tuning the hyper-parameters of an estimator", and the `HistGradientBoostingClassifier` docs.
#
# **Next:** [Evaluation, interpretation and the model card](05_evaluation_interpretation_and_model_card.ipynb)

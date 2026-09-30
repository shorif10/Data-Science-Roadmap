# %% [markdown]
# # Model Interpretation and Explainability
#
# **Why this matters.** Sooner or later someone asks "why does the model think this customer will leave?" or "which inputs matter most?", and your answer steers
# real money (who gets the discount) and sometimes real regulation. The standard tools each answer a *different* question, and each has a failure mode that produces a confident,
# wrong story. Knowing which tool to trust for which question is the skill.
#
# **You will learn**
# - read logistic-regression coefficients correctly (scaling, odds ratios) and say what they do not mean
# - compute **permutation importance** on held-out data, by hand and with `sklearn.inspection`
# - draw **partial dependence (PDP)** and **ICE** curves, and build a PDP from scratch
# - see what correlated features do to importance and to PDPs, and how to respond
# - separate "what the model uses" from "what causes churn"
#
# **Prerequisites:** [Logistic regression](04_logistic_regression_and_classification_metrics.ipynb), [Ensembles](06_ensembles_random_forest_and_gradient_boosting.ipynb)
# and [Pipelines](08_feature_engineering_and_pipelines.ipynb).
#
# > Data used: `datasets/churn.csv` (synthetic). The `shap` library is not installed here, so we use scikit-learn's own tools; they cover the core ideas and you can add SHAP later.

# %%
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path("../datasets")
df = pd.read_csv(DATA / "churn.csv", keep_default_na=False, na_values=[""])
df["contract"] = df["contract"].str.lower().str.strip()
df["internet_service"] = df["internet_service"].fillna("Unknown")

num_cols = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "senior_citizen"]
cat_cols = ["contract", "internet_service", "paperless_billing", "autopay"]
X = df[num_cols + cat_cols].astype({c: float for c in num_cols})  # PDP requires float columns, not int
y = df["churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)


def preprocessor(num=num_cols, scale=True):
    steps = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([("num", Pipeline(steps), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])


logit = Pipeline([("pre", preprocessor()), ("clf", LogisticRegression(max_iter=2000))]).fit(X_train, y_train)
gbm = Pipeline([("pre", preprocessor(scale=False)), ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=42))]).fit(X_train, y_train)
for name, m in [("logistic", logit), ("gradient boosting", gbm)]:
    print(f"{name:<18} test ROC-AUC {roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]):.3f}")

# %% [markdown]
# Interpretation only makes sense for a model that works; both are decent, so we can ask what they rely on. Keep the **question** in mind for each tool:
# *global* ("what matters overall?") or *local* ("why this customer?"), and *model behaviour* ("what does the model use?") or *reality* ("what drives churn?"). Tools only ever answer the first of each pair about the **model**.

# %% [markdown]
# ## 1. Coefficients: readable, but only if you scale and say it right
#
# In logistic regression, `coef` is the change in **log-odds** of churn per unit of the feature, holding the others fixed. `exp(coef)` is the **odds ratio**. Because we standardised the numeric inputs, each
# numeric coefficient is "per one standard deviation", so magnitudes are comparable across features.

# %%
names = logit.named_steps["pre"].get_feature_names_out()
coefs = pd.Series(logit.named_steps["clf"].coef_[0], index=[n.split("__")[1] for n in names])
coef_table = pd.DataFrame({"coef (log-odds)": coefs, "odds ratio": np.exp(coefs)}).sort_values("coef (log-odds)", key=abs, ascending=False)
coef_table.round(3)

# %% [markdown]
# *What just happened?* Read an odds ratio as "multiplies the odds of churning by this, per one SD of the feature, other features held fixed". Two traps. First, **units**: a coefficient on raw `monthly_charges` (dollars)
# and on raw `tenure_months` are not comparable; standardise first (as here) or convert explicitly. Second, **collinearity**: `total_charges` is almost `tenure_months * monthly_charges`,
# so the model can trade weight between them, and the individual signs/sizes are unstable even when predictions are fine. The one-hot columns (`contract_...`) are relative to *all* the other levels because we kept every
# level (with regularisation that is fine, but the numbers are not "versus a reference level").

# %% [markdown]
# ## 2. Permutation importance: what happens to the score if I scramble a feature?
#
# Idea: take **held-out** data, shuffle one column (breaking its link to the target while keeping its distribution), and see how much the metric drops. A big drop means the model relied on that feature.
# It works for any model. Implement it by hand for one feature first:

# %%
def perm_drop(model, X_df, y_ser, col, seed=0):
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y_ser, model.predict_proba(X_df)[:, 1])
    Xp = X_df.copy()
    Xp[col] = X_df[col].to_numpy()[rng.permutation(len(X_df))]
    return base - roc_auc_score(y_ser, model.predict_proba(Xp)[:, 1])


print("AUC drop when shuffling contract on the test set:", round(perm_drop(gbm, X_test, y_test, "contract"), 4))
print("AUC drop when shuffling autopay                 :", round(perm_drop(gbm, X_test, y_test, "autopay"), 4))

# %% [markdown]
# Now the library version does this for every column with repeats (so you get a spread, not a single lucky shuffle), on the **test set**, using the **whole pipeline** so it shuffles raw columns:

# %%
pi = permutation_importance(gbm, X_test, y_test, scoring="roc_auc", n_repeats=15, random_state=42)
pi_table = pd.DataFrame({"mean AUC drop": pi.importances_mean, "std": pi.importances_std}, index=X_test.columns).sort_values("mean AUC drop", ascending=False)
pi_table.round(4)

# %% [markdown]
# *What just happened?* Features with a mean drop many times larger than the std are genuinely used; ones near zero are either useless or **redundant** with another feature (section 4). Two rules: compute it on
# **held-out data** (on training data it also rewards memorising noise), and remember it measures reliance *by this model*, not truth about the world. Compare it with the impurity-based
# `feature_importances_` of a random forest, which is computed on training data and inflated for continuous features (notebook 05):

# %%
rf = Pipeline([("pre", preprocessor(scale=False)), ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=1))]).fit(X_train, y_train)
rf_imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=[n.split("__")[1] for n in rf.named_steps["pre"].get_feature_names_out()])
rf_perm = pd.Series(permutation_importance(rf, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=42).importances_mean, index=X_test.columns)
cmp = pd.DataFrame({"impurity importance (train, per encoded column)": rf_imp.filter(num_cols), "permutation drop (test)": rf_perm.filter(num_cols)})
cmp.round(3)

# %% [markdown]
# Compare the columns. Impurity importance gives `total_charges` roughly the same credit as `tenure_months` and `monthly_charges`, because continuous columns offer many split points and the forest uses them on the training data. Permutation importance on held-out data says `total_charges`
# adds very little once the other two are present (it is nearly their product: redundant). Prefer the permutation version for decisions; the two numbers are on different scales.

# %% [markdown]
# ## 3. Partial dependence and ICE: what does the model do as a feature changes?
#
# **Partial dependence (PDP)**: pick a feature, set it to each value on a grid **for every row**, predict, and average. The curve shows the model's average response. **ICE** (individual conditional expectation)
# draws one such curve per row, so you can see whether the average hides differences. Build a PDP by hand (it is just a loop), then compare with scikit-learn:

# %%
grid = np.array([1, 12, 24, 48, 72])
manual_pdp = []
for v in grid:
    Xv = X_test.copy()
    Xv["tenure_months"] = float(v)
    manual_pdp.append(gbm.predict_proba(Xv)[:, 1].mean())
sk = partial_dependence(gbm, X_test, ["tenure_months"], grid_resolution=50, kind="average")
sk_at_grid = np.interp(grid, sk["grid_values"][0], sk["average"][0])
print("manual PDP at tenure", grid.tolist(), "=", np.round(manual_pdp, 3))
print("sklearn (interpolated)              =", np.round(sk_at_grid, 3))

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
PartialDependenceDisplay.from_estimator(gbm, X_test, ["tenure_months"], kind="both", subsample=60, random_state=0, ax=axes[0],
                                        ice_lines_kw={"alpha": 0.25, "color": "tab:blue"}, pd_line_kw={"color": "tab:red", "lw": 3})
axes[0].set_title("ICE (thin) and PDP (thick): tenure")
axes[0].set_ylabel("predicted churn probability")
PartialDependenceDisplay.from_estimator(gbm, X_test, ["monthly_charges"], kind="both", subsample=60, random_state=0, ax=axes[1],
                                        ice_lines_kw={"alpha": 0.25, "color": "tab:blue"}, pd_line_kw={"color": "tab:red", "lw": 3})
axes[1].set_title("ICE (thin) and PDP (thick): monthly charges")
axes[1].set_ylabel("predicted churn probability")
plt.tight_layout()
plt.show()

# %% [markdown]
# *What just happened?* The manual loop agrees with the library (the library evaluates a finer grid, hence the interpolation). The thick line is the average effect; the thin lines are individual customers. If the thin lines are roughly parallel, the PDP is a faithful summary; if they
# fan out or cross, the feature **interacts** with others and the average hides it (for example, high monthly charges may matter only for month-to-month customers). Always draw ICE before writing "the effect of X is ...".

# %% [markdown]
# ## 4. The correlated-feature pitfalls
#
# **Pitfall A: importance is split (or hidden) among correlated features.** To see the mechanism cleanly, give the model a *near-duplicate* of `tenure_months` (the same value plus a little noise, like a second sensor measuring
# the same thing). Predictions barely change, but where does the importance go?

# %%
def with_twin(d, seed):
    d = d.copy()
    d["tenure_twin"] = d["tenure_months"] + np.random.default_rng(seed).normal(0, 1.5, len(d))
    return d


Xt_train, Xt_test = with_twin(X_train, 1), with_twin(X_test, 2)
num_tw = num_cols + ["tenure_twin"]
gbm_tw = Pipeline([("pre", preprocessor(num_tw, scale=False)), ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=42))]).fit(Xt_train, y_train)
pi_tw = permutation_importance(gbm_tw, Xt_test, y_test, scoring="roc_auc", n_repeats=15, random_state=42)
imp_tw = pd.Series(pi_tw.importances_mean, index=Xt_test.columns)
print("test ROC-AUC with the twin:", round(roc_auc_score(y_test, gbm_tw.predict_proba(Xt_test)[:, 1]), 3), "(was", round(roc_auc_score(y_test, gbm.predict_proba(X_test)[:, 1]), 3), ")")
print(f"permutation importance of tenure_months: {pi_table.loc['tenure_months', 'mean AUC drop']:.4f} alone  ->  {imp_tw['tenure_months']:.4f} with a twin present (twin: {imp_tw['tenure_twin']:.4f})")


def group_drop(model, X_df, y_ser, cols, seed=0):
    """Shuffle several columns with the SAME row permutation, so the group is destroyed as a unit."""
    perm = np.random.default_rng(seed).permutation(len(X_df))
    Xp = X_df.copy()
    for c in cols:
        Xp[c] = X_df[c].to_numpy()[perm]
    return roc_auc_score(y_ser, model.predict_proba(X_df)[:, 1]) - roc_auc_score(y_ser, model.predict_proba(Xp)[:, 1])


print("shuffling the pair together:", round(group_drop(gbm_tw, Xt_test, y_test, ["tenure_months", "tenure_twin"]), 4))

# %% [markdown]
# *What just happened?* Shuffling one twin leaves the other to carry the signal, so **the individual numbers understate the information** (compare the two printed importances with the group shuffle in the last line). A naive reader would say "tenure barely matters"
# and might drop it. Remedies: shuffle correlated features **as a group** (last line), remove redundant features, or report importance for feature *groups* ("tenure information", "contract information"). The real data has
# the same problem at a smaller scale: `total_charges` versus `tenure_months` and `monthly_charges`.
#
# **Pitfall B: PDP/ICE visit impossible rows.** Setting `tenure_months` to 72 for *everyone* keeps each customer's original `total_charges`, producing customers who have been around six years but have paid almost nothing. The model has never seen those
# and its answers there are extrapolation. Measure how unrealistic the manufactured rows are:

# %%
ratio = lambda d: d["total_charges"] / (d["tenure_months"] * d["monthly_charges"]).replace(0, np.nan)
real_ok = ratio(X_test).between(0.9, 1.1).mean()
forced = X_test.assign(tenure_months=72.0)
forced_ok = ratio(forced).between(0.9, 1.1).mean()
print(f"rows where total_charges is consistent with tenure*monthly (+-10%): real data {real_ok:.0%}, after forcing tenure=72 {forced_ok:.0%}")

# %% [markdown]
# Real rows follow the billing identity almost always; the forced rows almost never do. Mitigations: drop the redundant feature (here `total_charges` adds little once the other two are present, as permutation importance showed), use ICE to look for odd regions, or restrict the plot to the range where data is dense.
# Compare the tenure PDP with and without the redundant column:

# %%
no_total = [c for c in num_cols if c != "total_charges"]
gbm_nt = Pipeline([("pre", preprocessor(no_total, scale=False)), ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=42))]).fit(X_train, y_train)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for label, m in [("with total_charges", gbm), ("without total_charges", gbm_nt)]:
    pdp = partial_dependence(m, X_test, ["tenure_months"], grid_resolution=40, kind="average")
    ax.plot(pdp["grid_values"][0], pdp["average"][0], label=label)
ax.set_title("Partial dependence on tenure, with and without the redundant column")
ax.set_xlabel("tenure (months)")
ax.set_ylabel("average predicted churn probability")
ax.legend()
plt.show()
print("test AUC without total_charges:", round(roc_auc_score(y_test, gbm_nt.predict_proba(X_test)[:, 1]), 3), "| with:", round(roc_auc_score(y_test, gbm.predict_proba(X_test)[:, 1]), 3))

# %% [markdown]
# The curves can differ where the forced rows are unrealistic, while the model's accuracy is about the same. When two features tell the same story, the simpler model gives the more trustworthy explanation.

# %% [markdown]
# ## 5. Explaining one customer (local explanation)
#
# For a linear model the log-odds is a sum: `intercept + sum(coef_j * x_j)`. Relative to an "average customer", each feature contributes `coef_j * (x_j - mean_j)`, and the contributions add up *exactly* to the
# difference in log-odds. That is the simplest local explanation, and it is what SHAP generalises to arbitrary models (using Shapley values from game theory; SHAP is not installed here, so we stay with the linear case).

# %%
pre = logit.named_steps["pre"]
Z_train, Z_test = pre.transform(X_train), pre.transform(X_test)
clf = logit.named_steps["clf"]
risk = logit.predict_proba(X_test)[:, 1]
i = int(risk.argmax())
contrib = pd.Series(clf.coef_[0] * (Z_test[i] - Z_train.mean(axis=0)), index=[n.split("__")[1] for n in names])
baseline_logodds = clf.intercept_[0] + clf.coef_[0] @ Z_train.mean(axis=0)
print(f"highest-risk test customer: predicted churn probability {risk[i]:.2f}")
print("check: baseline + contributions == model log-odds:", bool(np.isclose(baseline_logodds + contrib.sum(), clf.decision_function(Z_test[i:i + 1])[0])))
contrib.sort_values(key=abs, ascending=False).head(5).round(2).to_frame("contribution to log-odds")

# %% [markdown]
# Note what this explanation is and is not: it says *why the model scored this customer high relative to the average customer*. It does not say that changing those inputs would change the customer's behaviour.

# %% [markdown]
# ## 6. What the model uses is not what causes churn
#
# Every tool above describes the **model**. If `support_calls` is important, it means "calls carry information about churn", not "reduce calls and churn will fall". Calls may be a *symptom* (unhappy customers call more), a consequence of
# high bills (monthly charges and calls are correlated here), or both. Only an experiment (like the A/B test in module 03) or a careful causal analysis can say what happens when you intervene. Use importance to debug models, find leaks, sanity-check against domain knowledge and pick
# who to target; use experiments to decide what to *do*.

# %% [markdown]
# ## Common mistakes
#
# 1. **Comparing raw coefficients** of differently scaled features, or reading them as effects when features are correlated.
# 2. **Computing permutation importance on the training set** (rewards overfitting) or on a single shuffle (noisy; use repeats).
# 3. **Trusting impurity-based `feature_importances_`** for decisions.
# 4. **Concluding a feature is useless** because its importance is low when a correlated twin exists; shuffle groups.
# 5. **Reading a PDP as a causal effect** or as truth in regions with no data; always add ICE and check the data density.
# 6. **Explaining a bad model.** An elegant explanation of a model with no signal is still nonsense; check the score first.

# %%
# Mistake 2: permutation importance on training data overstates what generalises. Fit an over-flexible model and compare.
deep = Pipeline([("pre", preprocessor(scale=False)), ("clf", RandomForestClassifier(n_estimators=100, min_samples_leaf=1, random_state=42, n_jobs=1))]).fit(X_train, y_train)
on_train = permutation_importance(deep, X_train, y_train, scoring="roc_auc", n_repeats=5, random_state=0).importances_mean
on_test = permutation_importance(deep, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=0).importances_mean
print(pd.DataFrame({"on train": on_train, "on test": on_test}, index=X_test.columns).loc[["total_charges", "monthly_charges", "tenure_months"]].round(3))
print("train AUC", round(roc_auc_score(y_train, deep.predict_proba(X_train)[:, 1]), 3), "| test AUC", round(roc_auc_score(y_test, deep.predict_proba(X_test)[:, 1]), 3))

# %% [markdown]
# ## Interview corner
#
# **1. How do you find the most important features of a model?**
# <details><summary>Answer</summary>
# Prefer permutation importance on held-out data (model-agnostic, measures the drop in the metric), with repeats for a spread. Impurity importances from trees are biased toward high-cardinality continuous features
# and come from training data. Always say what the number means and check for correlated features.
# </details>
#
# **2. What is the difference between a PDP and ICE curves?**
# <details><summary>Answer</summary>
# A PDP is the average prediction as one feature is varied over a grid while the other features keep their observed values. ICE draws the curve for each row separately. If the ICE lines differ in shape
# (interactions), the PDP average can hide that; if they are parallel, the PDP summarises well.
# </details>
#
# **3. Two features are highly correlated; how does that affect interpretation?**
# <details><summary>Answer</summary>
# Coefficients become unstable, permutation and impurity importance get split or hidden among the twins, and PDPs evaluate unrealistic combinations. Remedies: drop or combine redundant features, permute groups together,
# use regularisation, and be explicit that explanations are about the model.
# </details>
#
# **4. The model says feature X is the strongest driver of churn. Should we change X?**
# <details><summary>Answer</summary>
# Not on that evidence alone. Importance is predictive association, possibly through confounders or reverse causality. Run an experiment or a causal analysis before intervening.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Permutation importance by hand
# Write it yourself instead of calling `perm_drop`: with `rng = np.random.default_rng(0)` and `perm = rng.permutation(len(X_test))`, shuffle **only** the `support_calls` column of a copy of `X_test` using `perm`,
# and store the drop in test ROC-AUC of the gradient-boosting pipeline `gbm` (baseline AUC minus shuffled AUC) in `support_drop`. Also store in `support_more_than_autopay` whether this drop is larger than the same computation for `autopay`.

# %% tags=["starter"]
support_drop = ...
support_more_than_autopay = ...

# %%
def _drop(col):
    _perm = np.random.default_rng(0).permutation(len(X_test))
    _Xp = X_test.copy()
    _Xp[col] = X_test[col].to_numpy()[_perm]
    return roc_auc_score(y_test, gbm.predict_proba(X_test)[:, 1]) - roc_auc_score(y_test, gbm.predict_proba(_Xp)[:, 1])


assert abs(support_drop - _drop("support_calls")) < 1e-9, "shuffle only support_calls with perm; drop = baseline AUC - shuffled AUC"
assert bool(support_more_than_autopay) == bool(_drop("support_calls") > _drop("autopay")), "compare with the autopay drop computed the same way"
print(f"Exercise 1 passed: shuffling support_calls costs {support_drop:.4f} AUC")

# %% tags=["solution"]
def drop_for(col):
    perm = np.random.default_rng(0).permutation(len(X_test))
    Xp = X_test.copy()
    Xp[col] = X_test[col].to_numpy()[perm]
    return roc_auc_score(y_test, gbm.predict_proba(X_test)[:, 1]) - roc_auc_score(y_test, gbm.predict_proba(Xp)[:, 1])


support_drop = float(drop_for("support_calls"))
support_more_than_autopay = bool(support_drop > drop_for("autopay"))

# %% [markdown]
# ### Exercise 2: A partial dependence curve from scratch
# For the pipeline `gbm`, compute the partial dependence of churn probability on `monthly_charges` at the values `[20, 50, 80, 110]` using the test set: for each value, set `monthly_charges` to it for **every** row of a copy of
# `X_test` (as a float), predict churn probabilities, and average. Store the four numbers (a NumPy array) in `pdp_monthly` and the difference between the largest and smallest in `pdp_spread`.

# %% tags=["starter"]
pdp_monthly = ...
pdp_spread = ...

# %%
_ref = []
for _v in [20, 50, 80, 110]:
    _Xv = X_test.copy()
    _Xv["monthly_charges"] = float(_v)
    _ref.append(gbm.predict_proba(_Xv)[:, 1].mean())
assert np.allclose(np.asarray(pdp_monthly, dtype=float), _ref), "set the column to the grid value for ALL rows, predict, then average the probabilities"
assert abs(pdp_spread - (max(_ref) - min(_ref))) < 1e-9, "pdp_spread = max - min of the four values"
print(f"Exercise 2 passed: PDP {np.round(pdp_monthly, 3)} (spread {pdp_spread:.3f})")

# %% tags=["solution"]
vals = []
for v in [20, 50, 80, 110]:
    Xv = X_test.copy()
    Xv["monthly_charges"] = float(v)
    vals.append(gbm.predict_proba(Xv)[:, 1].mean())
pdp_monthly = np.array(vals)
pdp_spread = float(pdp_monthly.max() - pdp_monthly.min())

# %% [markdown]
# ### Exercise 3: Permute a correlated pair together
# `total_charges` and `tenure_months` are strongly related. With `perm = np.random.default_rng(0).permutation(len(X_test))`, compute the AUC drop for `gbm` when (a) only `tenure_months`, (b) only `total_charges`, and (c) **both together** using the same
# `perm` are shuffled. Store them in `drop_tenure`, `drop_total`, `drop_both`, and set `both_exceeds_sum` to whether the joint drop is larger than the sum of the two single drops.

# %% tags=["starter"]
drop_tenure = ...
drop_total = ...
drop_both = ...
both_exceeds_sum = ...

# %%
_perm = np.random.default_rng(0).permutation(len(X_test))
_base = roc_auc_score(y_test, gbm.predict_proba(X_test)[:, 1])


def _gd(cols):
    _Xp = X_test.copy()
    for _c in cols:
        _Xp[_c] = X_test[_c].to_numpy()[_perm]
    return _base - roc_auc_score(y_test, gbm.predict_proba(_Xp)[:, 1])


assert abs(drop_tenure - _gd(["tenure_months"])) < 1e-9, "shuffle tenure_months only, with the shared perm"
assert abs(drop_total - _gd(["total_charges"])) < 1e-9, "shuffle total_charges only, with the shared perm"
assert abs(drop_both - _gd(["tenure_months", "total_charges"])) < 1e-9, "shuffle both columns with the SAME perm"
assert bool(both_exceeds_sum) == bool(_gd(["tenure_months", "total_charges"]) > _gd(["tenure_months"]) + _gd(["total_charges"])), "compare drop_both with drop_tenure + drop_total"
print(f"Exercise 3 passed: {drop_tenure:.4f} + {drop_total:.4f} vs together {drop_both:.4f}; joint exceeds sum: {both_exceeds_sum}")

# %% tags=["solution"]
perm = np.random.default_rng(0).permutation(len(X_test))
base = roc_auc_score(y_test, gbm.predict_proba(X_test)[:, 1])


def shuffled_drop(cols):
    Xp = X_test.copy()
    for c in cols:
        Xp[c] = X_test[c].to_numpy()[perm]
    return base - roc_auc_score(y_test, gbm.predict_proba(Xp)[:, 1])


drop_tenure = float(shuffled_drop(["tenure_months"]))
drop_total = float(shuffled_drop(["total_charges"]))
drop_both = float(shuffled_drop(["tenure_months", "total_charges"]))
both_exceeds_sum = bool(drop_both > drop_tenure + drop_total)

# %% [markdown]
# ### Exercise 4: One more support call
# Using the **logistic** pipeline `logit`, what happens to the odds of churn for one extra support call? The scaler inside `logit` standardised `support_calls`, so you must convert the coefficient back to raw units. Store in `or_per_call` the odds ratio
# for **one additional call**, `exp(coef / scale)`. Then store in `extra_call_prob_change` the average change in predicted churn probability across `X_test` when every customer's `support_calls` is increased by 1.
# *Hint: the numeric block of the ColumnTransformer is named `num` and its scaler step is `sc`; the order of `num_cols` gives each column's index in `coef_` and `scale_`.*

# %% tags=["starter"]
or_per_call = ...
extra_call_prob_change = ...

# %%
_sc = logit.named_steps["pre"].named_transformers_["num"].named_steps["sc"]
_coef = logit.named_steps["clf"].coef_[0][num_cols.index("support_calls")]
_or = float(np.exp(_coef / _sc.scale_[num_cols.index("support_calls")]))
_plus = X_test.assign(support_calls=X_test["support_calls"] + 1.0)
_chg = float((logit.predict_proba(_plus)[:, 1] - logit.predict_proba(X_test)[:, 1]).mean())
assert abs(or_per_call - _or) < 1e-9, "exp(standardised coef / scaler.scale_ for support_calls)"
assert abs(extra_call_prob_change - _chg) < 1e-9, "average of p(support_calls+1) - p(support_calls) over X_test"
assert or_per_call > 1 and extra_call_prob_change > 0, "more calls should raise predicted churn in this model"
print(f"Exercise 4 passed: odds x{or_per_call:.2f} per call; average probability change {extra_call_prob_change:+.3f}")

# %% tags=["solution"]
scaler = logit.named_steps["pre"].named_transformers_["num"].named_steps["sc"]
idx = num_cols.index("support_calls")
or_per_call = float(np.exp(logit.named_steps["clf"].coef_[0][idx] / scaler.scale_[idx]))
plus_one = X_test.assign(support_calls=X_test["support_calls"] + 1.0)
extra_call_prob_change = float((logit.predict_proba(plus_one)[:, 1] - logit.predict_proba(X_test)[:, 1]).mean())

# %% [markdown]
# ### Exercise 5 (open-ended): Write the explanation you would give a retention manager
# Using the coefficient table, permutation importance and one PDP/ICE plot, write five sentences: the three things that most influence the model's churn scores, the direction of each, one caveat about correlated features,
# and one sentence that explicitly says what you can and cannot conclude about *causes* of churn. Then propose one experiment that could test a causal claim.

# %% [markdown]
# ## Recap
# - Standardised logistic coefficients and odds ratios explain a *linear* model, per unit of a feature with the others held fixed; correlated inputs make them unstable.
# - Permutation importance on held-out data is the general-purpose "what does the model rely on" tool; use repeats and prefer it over impurity importance.
# - PDPs show the average response and ICE shows the spread; build one by hand once so it stops being magic.
# - Correlated features split importance and push PDPs into unrealistic regions; permute groups, drop redundant columns.
# - All of this explains the **model**, not the world: do not turn importance into causal advice.
#
# **Go deeper:** scikit-learn user guide "Inspection" (permutation feature importance, partial dependence and ICE plots); the SHAP documentation (shap.readthedocs.io) once you install it; Christoph Molnar's free book "Interpretable Machine Learning".
#
# **Next:** [Time-series forecasting baselines](14_time_series_forecasting_baselines.ipynb)

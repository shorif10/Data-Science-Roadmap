# %% [markdown]
# # Linear Regression and Regularization
#
# **Why this matters.** Linear regression is the workhorse of applied prediction: pricing, demand, risk scores, and the first model any
# reviewer expects as a baseline. It is fast, its coefficients are explainable ("each extra 100 sq ft adds about X"), and understanding
# *how* it fits, and how Ridge/Lasso restrain it, gives you the vocabulary for every model that comes after (overfitting, bias/variance, shrinkage).
#
# **You will learn**
# - fit a line by least squares, both from the closed-form formula and with scikit-learn, and confirm they agree
# - read coefficients correctly (units, scaling, one-hot categories) and inspect residuals
# - decide what to do with absurd target values without fooling yourself
# - see Ridge, Lasso and ElasticNet shrink coefficients, and see polynomial features overfit
#
# **Prerequisites:** [Train/test split and leakage](02_train_test_split_and_data_leakage.ipynb); the linear algebra notebooks in
# [Math for Data Science](../02_Math_for_Data_Science/).
#
# > Data used: `datasets/housing.csv` (3,000 synthetic house sales; price rises non-linearly with area and depends on neighborhood).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

DATA = Path("../datasets")
housing = pd.read_csv(DATA / "housing.csv")
housing.head(3)

# %% [markdown]
# ## 1. The absurd prices: decide, document, do not hide
#
# `DATASETS.md` warns that a few prices are absurd. Do not delete "outliers" by feel; find a *reason*. Price per square foot is a natural
# sanity check because it removes the effect of house size:

# %%
housing["ppsf"] = housing["price"] / housing["area_sqft"]
print(housing["ppsf"].describe(percentiles=[0.01, 0.5, 0.99]).round(1).to_dict())
absurd = ~housing["ppsf"].between(20, 500)
print("flagged as absurd:", int(absurd.sum()), "of", len(housing))
housing.loc[absurd, ["area_sqft", "quality_score", "price", "ppsf"]].sort_values("ppsf").round(0).head(8)

# %% [markdown]
# A negative price, or a 1,700 sq ft house sold for a few thousand dollars, cannot be a real sale; at the other end, small houses priced in the millions
# are just as implausible. All of them sit far outside the bulk of the distribution (compare the 1st to 99th percentile of price per sq ft above). The rule `20 <= price/sqft <= 500` is a judgement call,
# and it is one: write it down, apply it identically to train and test (they are the same kind of error), and in production reject such inputs at the
# data-validation step. What we must **not** do is drop rows because the model predicts them badly.
#
# How much does the choice matter? Fit a plain linear model with and without them:

# %%
FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years", "garage_spaces", "distance_to_center_km", "quality_score"]


def fit_eval(df, seed=42):
    a, b, c, d = train_test_split(df[FEATURES], df["price"], test_size=0.25, random_state=seed)
    m = make_pipeline(SimpleImputer(strategy="median"), LinearRegression()).fit(a, c)
    p = m.predict(b)
    return m, mean_absolute_error(d, p), r2_score(d, p)


_, mae_dirty, r2_dirty = fit_eval(housing)
m_clean, mae_clean, r2_clean = fit_eval(housing[~absurd])
print(f"with absurd rows   : MAE {mae_dirty:,.0f}  R2 {r2_dirty:.3f}")
print(f"without absurd rows: MAE {mae_clean:,.0f}  R2 {r2_clean:.3f}")

# %% [markdown]
# *What just happened?* A few bad rows out of 3,000 worsen both metrics, and, worse, a test score computed on dirty rows is partly made of errors nobody can predict.
# For the rest of the notebook we work with the cleaned table (`hc`). Note the honest caveat: numbers on `hc` are what the model does on
# *plausible* houses, not on garbage input.

# %%
hc = housing.loc[~absurd].drop(columns="ppsf").reset_index(drop=True)
hc = pd.get_dummies(hc, columns=["neighborhood"], dtype=float)  # one-hot: one 0/1 column per neighborhood (details in notebook 08)
NBH = [c for c in hc.columns if c.startswith("neighborhood_")]
X_all, y_all = hc.drop(columns="price"), hc["price"]
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.25, random_state=42)
print(X_train.shape, X_test.shape)

# %% [markdown]
# ## 2. What "least squares" means
#
# A linear model predicts `price = b0 + b1*area + b2*bedrooms + ...`. Of all possible coefficient vectors, least squares picks the one minimising
# the sum of squared errors on the training rows. With a design matrix `X` (a column of ones for the intercept plus the features) and target `y`, the answer has
# a closed form, the **normal equations**: `beta = (X'X)^-1 X'y`. Let us compute it ourselves with NumPy on two features and compare with scikit-learn.

# %%
cols2 = ["area_sqft", "quality_score"]
train2 = X_train[cols2].copy()
train2 = train2.fillna(train2.median())  # quality_score has missing values; median-fill on train
A = np.column_stack([np.ones(len(train2)), train2.to_numpy()])   # intercept column + features
beta = np.linalg.solve(A.T @ A, A.T @ y_train.to_numpy())       # solve, never invert explicitly

sk = LinearRegression().fit(train2, y_train)
print("closed form :", beta.round(2))
print("scikit-learn:", np.r_[sk.intercept_, sk.coef_].round(2))
print("same answer :", np.allclose(beta, np.r_[sk.intercept_, sk.coef_]))

# %% [markdown]
# *What just happened?* `LinearRegression` is exactly this linear-algebra problem (scikit-learn uses a more numerically stable solver, `lstsq`, rather than
# forming `X'X`). Practical notes: `np.linalg.solve` beats computing an inverse; when features are nearly collinear `X'X` is close to singular and the answer
# becomes unstable, which is the problem Ridge fixes below. Interpretation: each extra square foot adds `beta[1]` dollars *holding quality constant*.

# %% [markdown]
# ## 3. The full model, coefficients and residuals
#
# Impute missing values with the training median, fit on all features including neighborhood, and score on the test set:

# %%
lin = make_pipeline(SimpleImputer(strategy="median"), LinearRegression()).fit(X_train, y_train)
pred = lin.predict(X_test)
print(f"MAE {mean_absolute_error(y_test, pred):,.0f}   RMSE {root_mean_squared_error(y_test, pred):,.0f}   R2 {r2_score(y_test, pred):.3f}")
print(f"median-price baseline MAE for comparison: {mean_absolute_error(y_test, np.full(len(y_test), y_train.median())):,.0f}")

coef = pd.Series(lin[-1].coef_, index=X_all.columns)
coef.round(0).to_frame("dollars per unit")

# %% [markdown]
# **Reading coefficients.** `area_sqft` is dollars per square foot; `quality_score` is dollars per quality point; each `neighborhood_X` is a *shift relative to the
# others*, and because the dummies sum to 1 (a redundancy called the dummy-variable trap) their absolute values are only meaningful relative to each other,
# since scikit-learn resolves the redundancy with a minimum-norm solution. You cannot compare `area_sqft` (range ~500-5000) with `quality_score` (1-10) by the
# raw numbers because their units differ. Standardise the features (subtract mean, divide by std) and coefficients become "dollars per one standard deviation":

# %%
std_model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()).fit(X_train, y_train)
std_coef = pd.Series(std_model[-1].coef_, index=X_all.columns)
std_coef.reindex(std_coef.abs().sort_values(ascending=False).index).round(0).head(6).to_frame("dollars per 1 std")

# %% [markdown]
# Now the residuals (`actual - predicted`). A good linear fit leaves residuals that look like structureless noise around zero. Two things to check:
# a curve (the model is missing non-linearity) and a funnel (errors grow with the prediction, i.e. non-constant variance).

# %%
resid = y_test.to_numpy() - pred
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].scatter(pred, resid, s=8, alpha=0.5)
ax[0].axhline(0, color="k", lw=1)
ax[0].set(title="Residuals vs predicted price", xlabel="predicted price ($)", ylabel="residual: actual - predicted ($)")
ax[1].hist(resid, bins=40)
ax[1].set(title="Residual distribution", xlabel="residual ($)", ylabel="count of houses")
plt.tight_layout()
plt.show()

order = np.argsort(pred)
thirds = np.array_split(resid[order], 3)
print("residual std by predicted-price third (low, mid, high):", [round(float(t.std())) for t in thirds])

# %% [markdown]
# Compare the three spreads printed above: if the high third has a clearly larger spread, a plain linear model in dollars is under-fitting the expensive end
# and a log-target or a tree model (next notebooks) is worth trying. That is the usual reading of a funnel: **residuals tell you what to try next**.

# %% [markdown]
# ## 4. Shrinking coefficients: Ridge, Lasso, ElasticNet
#
# Least squares has no brakes: with many or correlated features it will pick large, offsetting coefficients that fit noise. **Regularization** adds a penalty
# on coefficient size to the loss:
#
# | Model | Penalty | Effect |
# |---|---|---|
# | Ridge | `alpha * sum(b^2)` (L2) | shrinks all coefficients smoothly toward zero; never exactly zero; handles collinearity |
# | Lasso | `alpha * sum(abs(b))` (L1) | shrinks *and* sets some coefficients exactly to zero, i.e. feature selection |
# | ElasticNet | mix of both (`l1_ratio`) | Lasso-like sparsity with Ridge-like stability for correlated groups |
#
# Because the penalty depends on coefficient size, **features must be standardised first** (otherwise a feature measured in tiny units is punished more).
# `alpha` is the strength: 0 gives plain least squares. Trace the coefficients as `alpha` grows:

# %%
def path(estimator_cls, alphas, **kw):
    rows = []
    for a in alphas:
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator_cls(alpha=a, max_iter=100_000, **kw) if estimator_cls is not Ridge else Ridge(alpha=a))
        m.fit(X_train, y_train)
        rows.append(m[-1].coef_)
    return pd.DataFrame(rows, index=alphas, columns=X_all.columns)


alphas = np.logspace(-1, 4.3, 30)
paths = {"Ridge": path(Ridge, alphas), "Lasso": path(Lasso, alphas), "ElasticNet": path(ElasticNet, alphas, l1_ratio=0.5)}

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
for ax, (name, df_) in zip(axes, paths.items()):
    for c in df_.columns:
        ax.plot(df_.index, df_[c], label=c)
    ax.set_xscale("log")
    ax.set(title=f"{name} coefficient path", xlabel="alpha (log scale)")
axes[0].set_ylabel("standardised coefficient ($ per 1 std)")
axes[2].legend(fontsize=6, loc="upper right")
plt.tight_layout()
plt.show()

print("nonzero coefficients at the largest alpha ->", {k: int((v.iloc[-1] != 0).sum()) for k, v in paths.items()})

# %% [markdown]
# *What just happened?* As `alpha` grows, Ridge coefficients slide toward zero together but are never exactly zero; Lasso and ElasticNet drop the weakest features to
# exactly zero one by one (the counts printed above at the largest `alpha`). The order in which Lasso zeroes features is a rough importance ranking.
#
# Does it *help* predictions here? Check the test error along the way:

# %%
for a in [0.1, 100, 1000, 10000]:
    r = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=a)).fit(X_train, y_train)
    print(f"Ridge alpha={a:>7}: test MAE {mean_absolute_error(y_test, r.predict(X_test)):>9,.0f}")

# %% [markdown]
# Honest finding: with ~2,200 training rows and only 13 features, moderate regularization gives at most a small gain and heavy regularization *hurts* (it biases
# every coefficient toward zero). Regularization pays off when features are many relative to rows, or highly correlated. Here is a correlated case: give the model
# a second, noisy copy of `area_sqft` (like two sensors measuring the same thing) and compare how coefficients are shared:

# %%
rng = np.random.default_rng(42)
Xc_train, Xc_test = X_train.copy(), X_test.copy()
Xc_train["area_copy"] = Xc_train["area_sqft"] + rng.normal(0, 30, len(Xc_train))
Xc_test["area_copy"] = Xc_test["area_sqft"] + rng.normal(0, 30, len(Xc_test))
for name, est in [("OLS", LinearRegression()), ("Ridge(alpha=100)", Ridge(alpha=100))]:
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est).fit(Xc_train, y_train)
    cf = pd.Series(m[-1].coef_, index=Xc_train.columns)
    print(f"{name:17s} area_sqft {cf['area_sqft']:>9,.0f}   area_copy {cf['area_copy']:>9,.0f}   sum {cf['area_sqft'] + cf['area_copy']:>9,.0f}   test MAE {mean_absolute_error(y_test, m.predict(Xc_test)):,.0f}")

# %% [markdown]
# The *sum* of the two area coefficients is stable, but how OLS splits it between the two near-duplicates is arbitrary and sensitive to noise, so individual coefficients
# become uninterpretable. Ridge shares the weight roughly evenly. If you care about explaining coefficients, correlated features are a warning sign.

# %% [markdown]
# ## 5. Polynomial features and overfitting
#
# A line cannot bend. `PolynomialFeatures(degree=d)` adds squares, cubes and interaction terms (`area*quality`, ...), letting a linear model draw curved surfaces.
# The number of columns explodes, so with few training rows the model can memorise. Use only 60 training rows to make that visible:

# %%
X_small, y_small = X_train.iloc[:60], y_train.iloc[:60]
rows = []
for deg in [1, 2, 3, 4]:
    m = make_pipeline(SimpleImputer(strategy="median"), PolynomialFeatures(deg), StandardScaler(), LinearRegression()).fit(X_small, y_small)
    n_feat = m[1].n_output_features_
    rows.append({"degree": deg, "n_features": n_feat, "train R2": r2_score(y_small, m.predict(X_small)), "test R2": r2_score(y_test, m.predict(X_test))})
pd.DataFrame(rows).round(3)

# %% [markdown]
# The train R2 climbs to 1.0 (60 rows, hundreds of columns: a perfect fit is guaranteed), while the test R2 typically *falls*. That gap **is** overfitting. Ridge can reduce the damage:

# %%
rows = []
for a in [0.1, 1, 10, 100, 1000]:
    m = make_pipeline(SimpleImputer(strategy="median"), PolynomialFeatures(2), StandardScaler(), Ridge(alpha=a)).fit(X_small, y_small)
    rows.append({"alpha": a, "train R2": r2_score(y_small, m.predict(X_small)), "test R2": r2_score(y_test, m.predict(X_test))})
lin1 = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()).fit(X_small, y_small)
print("plain degree-1 linear model on the same 60 rows: test R2 =", round(r2_score(y_test, lin1.predict(X_test)), 3))
pd.DataFrame(rows).round(3)

# %% [markdown]
# Read this carefully: a well-chosen `alpha` improves the degree-2 model compared with unregularised degree 2, but compare with the straight-line number printed above: the simple model still wins.
# **Regularization reduces overfitting; it does not turn a needlessly complex model into a better one than a simple one.** Choose `alpha` by validation
# (cross-validation: notebook 07, `RidgeCV` / `LassoCV` do it internally), never by the test set.

# %% [markdown]
# ## Common mistakes
#
# 1. **Regularizing unscaled features.** The penalty then depends on units, not importance. Standardise first, inside a pipeline.
# 2. **Reading raw coefficients as importance** when features have different scales or are correlated.
# 3. **Deleting outliers because the model misses them**, instead of using a documented validity rule.
# 4. **Reporting R2 on the training set.** It only ever goes up as you add columns.
# 5. **Extrapolating**: a line keeps going; predicted prices for a 20,000 sq ft house are meaningless if training stopped at 5,000.

# %%
# Mistake 5: a linear model happily returns a nonsense number outside its training range
print("predicted price of a 40,000 sq ft, otherwise typical house:")
typical = X_train.median().to_frame().T
typical["area_sqft"] = 40_000
print(f"  ${lin.predict(typical)[0]:,.0f} (training max area was {X_train['area_sqft'].max():,.0f} sq ft)")

# %% [markdown]
# ## Interview corner
#
# **1. State the assumptions of linear regression. Which matter for prediction versus inference?**
# <details><summary>Answer</summary>
# Linearity of the mean in the features, independent errors, constant error variance, roughly normal errors, no perfect collinearity. For pure prediction, linearity and
# not-overfitting matter most; for confidence intervals and p-values on coefficients, independence, constant variance (and normality for small samples) matter as well.
# </details>
#
# **2. Ridge vs Lasso: when do you use which?**
# <details><summary>Answer</summary>
# Ridge when most features carry some signal or are correlated (it shrinks them together). Lasso when you believe only a few matter and want a sparse, explainable model.
# ElasticNet when you want sparsity but have groups of correlated features (Lasso alone picks one from a group arbitrarily). Tune <code>alpha</code> by cross-validation.
# </details>
#
# **3. Why standardise before Ridge/Lasso?**
# <details><summary>Answer</summary>
# The penalty is on coefficient magnitude, and the magnitude depends on the feature's units. Without scaling, features with small numeric ranges get large coefficients and are
# penalised more, for reasons unrelated to usefulness. (The intercept is not penalised.)
# </details>
#
# **4. Your model's R2 is 0.9 on train and 0.5 on test. Diagnose and fix.**
# <details><summary>Answer</summary>
# Overfitting (high variance). Options: more data, fewer or simpler features, stronger regularization, a lower polynomial degree, or a model with built-in capacity control.
# Verify with a learning curve (notebook 07) and check for leakage differences between the splits.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Slope from linear algebra
# Using only `area_sqft` (median-filled if needed; it has no missing values) and `y_train`, compute with `np.linalg.lstsq` the intercept and slope of the
# simple regression `price ~ area_sqft`. Store them in `intercept` and `slope`. *Hint: build `A = np.column_stack([np.ones(n), area])` and use `lstsq(A, y, rcond=None)[0]`.*

# %% tags=["starter"]
intercept = ...
slope = ...

# %%
sk1 = LinearRegression().fit(X_train[["area_sqft"]], y_train)
assert abs(slope - sk1.coef_[0]) < 1e-6, f"slope should be {sk1.coef_[0]:.4f}"
assert abs(intercept - sk1.intercept_) < 1e-3, f"intercept should be {sk1.intercept_:.2f}"
print(f"Exercise 1 passed: each extra sq ft adds ${slope:,.1f} (ignoring everything else)")

# %% tags=["solution"]
area = X_train["area_sqft"].to_numpy()
A_ = np.column_stack([np.ones(len(area)), area])
intercept, slope = np.linalg.lstsq(A_, y_train.to_numpy(), rcond=None)[0]

# %% [markdown]
# ### Exercise 2: What matters most?
# Using `std_coef` (standardised coefficients from section 3), store in `top_feature` the name of the feature with the largest **absolute** standardised coefficient,
# and in `n_negative` how many standardised coefficients are negative.

# %% tags=["starter"]
top_feature = ...
n_negative = ...

# %%
assert top_feature == std_coef.abs().idxmax(), "use the absolute value, then idxmax"
assert n_negative == int((std_coef < 0).sum()), "count coefficients below zero"
print(f"Exercise 2 passed: top feature {top_feature}, {n_negative} negative coefficients")

# %% tags=["solution"]
top_feature = std_coef.abs().idxmax()
n_negative = int((std_coef < 0).sum())

# %% [markdown]
# ### Exercise 3: Pick alpha with a validation split
# Using the 60-row set `X_small`/`y_small` and a degree-2 polynomial Ridge pipeline (`SimpleImputer(median)`, `PolynomialFeatures(2)`, `StandardScaler`, `Ridge`),
# split `X_small`/`y_small` 70/30 with `random_state=0`, choose the `alpha` from `[0.1, 1, 10, 100, 1000]` with the lowest **validation MAE**, and store it in `best_alpha`.
# Then store in `test_mae` the MAE of that model (refit on the same 70% part) on the real `X_test`. The test set is used once, only for the final report.

# %% tags=["starter"]
best_alpha = ...
test_mae = ...

# %%
a_, b_, c_, d_ = train_test_split(X_small, y_small, test_size=0.3, random_state=0)
def _make(al):
    return make_pipeline(SimpleImputer(strategy="median"), PolynomialFeatures(2), StandardScaler(), Ridge(alpha=al)).fit(a_, c_)
scores = {al: mean_absolute_error(d_, _make(al).predict(b_)) for al in [0.1, 1, 10, 100, 1000]}
exp_alpha = min(scores, key=scores.get)
assert best_alpha == exp_alpha, f"the lowest validation MAE is at alpha={exp_alpha}; scores: { {k: round(v) for k, v in scores.items()} }"
assert abs(test_mae - mean_absolute_error(y_test, _make(exp_alpha).predict(X_test))) < 1e-6, "refit on the 70% part, then score on X_test"
print(f"Exercise 3 passed: alpha={best_alpha}, test MAE {test_mae:,.0f}")

# %% tags=["solution"]
a_, b_, c_, d_ = train_test_split(X_small, y_small, test_size=0.3, random_state=0)
def _make(al):
    return make_pipeline(SimpleImputer(strategy="median"), PolynomialFeatures(2), StandardScaler(), Ridge(alpha=al)).fit(a_, c_)
scores = {al: mean_absolute_error(d_, _make(al).predict(b_)) for al in [0.1, 1, 10, 100, 1000]}
best_alpha = min(scores, key=scores.get)
test_mae = float(mean_absolute_error(y_test, _make(best_alpha).predict(X_test)))

# %% [markdown]
# ### Exercise 4: How sparse is sparse?
# Fit a `Lasso(alpha=10_000, max_iter=100_000)` in a pipeline (median imputer, `StandardScaler`, Lasso) on `X_train`, `y_train`. Store the **names** of features whose coefficient is
# exactly 0 in `dropped` (a list), and the test-set R2 in `lasso_r2`.

# %% tags=["starter"]
dropped = ...
lasso_r2 = ...

# %%
_l = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Lasso(alpha=10_000, max_iter=100_000)).fit(X_train, y_train)
exp_drop = [c for c, w in zip(X_all.columns, _l[-1].coef_) if w == 0]
assert sorted(dropped) == sorted(exp_drop), f"expected {exp_drop}"
assert abs(lasso_r2 - r2_score(y_test, _l.predict(X_test))) < 1e-9, "lasso_r2 is R2 on X_test"
print(f"Exercise 4 passed: Lasso dropped {dropped}; test R2 {lasso_r2:.3f}")

# %% tags=["solution"]
_l = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Lasso(alpha=10_000, max_iter=100_000)).fit(X_train, y_train)
dropped = [c for c, w in zip(X_all.columns, _l[-1].coef_) if w == 0]
lasso_r2 = float(r2_score(y_test, _l.predict(X_test)))

# %% [markdown]
# ### Exercise 5 (open-ended): Log the target
# Prices are positive and grow multiplicatively. Fit the same linear pipeline on `np.log(y_train)`, convert predictions back with `np.exp`, and compare MAE in dollars with the
# plain model. Write two sentences: which is better, and why might a log target help (or hurt) for an expensive-end funnel in the residuals? Remember that `exp(mean of log)` estimates a *median*, not a mean.

# %% [markdown]
# ## Recap
# - Least squares has a closed form (`solve(X'X, X'y)`); scikit-learn solves the same problem more stably.
# - Coefficients are in the feature's units; standardise to compare, and beware correlated features.
# - Residual plots tell you what to try next; absurd targets get a written rule, not a quiet deletion.
# - Ridge shrinks, Lasso selects, ElasticNet mixes; scale first, tune `alpha` on validation data, and expect small gains when rows greatly exceed features.
# - Polynomial features can overfit dramatically on small samples; regularization mitigates but simplicity often wins.
#
# **Go deeper:** the scikit-learn user guide "Linear Models" (scikit-learn.org/stable/modules/linear_model.html).
#
# **Next:** [Logistic regression and classification metrics](04_logistic_regression_and_classification_metrics.ipynb)

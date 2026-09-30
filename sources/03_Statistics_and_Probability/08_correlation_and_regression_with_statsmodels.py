# %% [markdown]
# # Correlation and Regression with statsmodels
#
# **Why this matters.** "Which factors drive price?" and "which customers are likely to churn, and by how much does each factor matter?" are the bread and butter of applied analytics. A regression answers them *and* tells you how
# uncertain each answer is. Unlike scikit-learn's prediction-first API, `statsmodels` gives the full inference table (standard errors, p-values, confidence intervals), which is what stakeholders and interviewers ask you to interpret.
#
# **You will learn**
# - compute and interpret Pearson and Spearman correlation, and know why correlation is not causation
# - fit and read an OLS regression (coefficients, p-values, R-squared) with the formula API
# - check residual diagnostics: patterns, non-constant variance, influential points, multicollinearity
# - fit a logistic regression for churn and interpret **odds ratios**
#
# **Prerequisites:** [Hypothesis testing](06_hypothesis_testing.ipynb) and [Distributions](03_distributions.ipynb).
#
# > Data used: `housing.csv` (price rows outside 10,000 to 1,000,000 removed as data errors) and `churn.csv` (synthetic, see `docs/DATASETS.md`).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

DATA = Path("../datasets")
rng = np.random.default_rng(42)

housing = pd.read_csv(DATA / "housing.csv")
h = housing[(housing["price"] > 10_000) & (housing["price"] < 1_000_000)].copy()
h["price_k"] = h["price"] / 1000                       # thousand dollars, easier to read in tables
churn = pd.read_csv(DATA / "churn.csv")
churn["contract"] = churn["contract"].str.lower()
churn["internet_service"] = churn["internet_service"].fillna("No internet")     # pandas reads the text "None" as missing; here it means "no internet"
print(f"{len(h)} houses, {len(churn)} customers")

# %% [markdown]
# ## 1. Correlation: strength of a *linear* relationship
#
# **Pearson's r** ranges from -1 to 1: +1 is a perfect increasing line, 0 no *linear* relationship. **Spearman's rho** is Pearson on the ranks: it detects any consistently increasing/decreasing (monotonic) relationship and is robust to outliers.

# %%
cols = ["price_k", "area_sqft", "bedrooms", "bathrooms", "age_years", "garage_spaces", "distance_to_center_km", "quality_score"]
corr = h[cols].corr()                                        # Pearson; pairwise deletion of missing values
fig, ax = plt.subplots(figsize=(6.5, 5.3))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right"); ax.set_yticks(range(len(cols)), cols)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, label="Pearson r"); ax.set_title("Correlation matrix of the housing data")
plt.tight_layout(); plt.show()

# %% [markdown]
# Area has the strongest link with price. Now three cases where **r alone misleads** (all synthetic so that we know the truth):

# %%
x = np.linspace(-3, 3, 200)
y_curve = x**2 + rng.normal(0, 0.3, 200)                       # perfect dependence, but U-shaped
x_out = np.append(rng.normal(0, 1, 50), 8.0); y_out = np.append(rng.normal(0, 1, 50), 8.0)    # no relation, plus one outlier
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
axes[0].scatter(x, y_curve, s=8); axes[0].set_title(f"U-shape: Pearson r = {stats.pearsonr(x, y_curve)[0]:.2f}")
axes[1].scatter(x_out, y_out, s=8); axes[1].set_title(f"One outlier: r = {stats.pearsonr(x_out, y_out)[0]:.2f}, Spearman = {stats.spearmanr(x_out, y_out)[0]:.2f}")
xs = rng.uniform(0, 4, 200); ys = np.exp(xs) + rng.normal(0, 1, 200)
axes[2].scatter(xs, ys, s=8); axes[2].set_title(f"Monotonic curve: Pearson {stats.pearsonr(xs, ys)[0]:.2f}, Spearman {stats.spearmanr(xs, ys)[0]:.2f}")
for ax in axes:
    ax.set_xlabel("x"); ax.set_ylabel("y")
plt.tight_layout(); plt.show()

# %% [markdown]
# A perfect but U-shaped dependence gives `r` near 0; a single outlier manufactures a strong `r`; a monotonic curve scores lower on Pearson than Spearman. **Always plot the data.**
#
# ### Correlation is not causation: confounding
# If a third variable `Z` drives both `X` and `Y`, they correlate although neither causes the other. Simulate (say `Z` = temperature, `X` = ice-cream sales, `Y` = swimming accidents):

# %%
z = rng.normal(size=2000)
x = z + rng.normal(0, 0.5, 2000)
y = z + rng.normal(0, 0.5, 2000)
print(f"corr(X, Y) = {np.corrcoef(x, y)[0, 1]:.2f}  <- strong, yet X does not affect Y")
resid_x = x - np.polyval(np.polyfit(z, x, 1), z); resid_y = y - np.polyval(np.polyfit(z, y, 1), z)
print(f"partial correlation after removing Z = {np.corrcoef(resid_x, resid_y)[0, 1]:.2f}  <- vanishes")

# %% [markdown]
# The same pattern shows up in real-looking data: does having more **bedrooms** raise price? Compare the simple slope with the slope after also controlling for area (which regression makes easy):

# %%
m_simple = smf.ols("price_k ~ bedrooms", data=h).fit()
m_ctrl = smf.ols("price_k ~ bedrooms + area_sqft", data=h).fit()
print(f"bedrooms alone         : +{m_simple.params['bedrooms']:.1f} thousand $ per bedroom")
print(f"bedrooms, given area   : +{m_ctrl.params['bedrooms']:.1f} thousand $ per bedroom")

# %% [markdown]
# Most of the apparent "bedroom effect" was really **size**: bigger houses have more bedrooms *and* cost more. Regression is the tool that lets you ask "the effect of X *holding other variables fixed*". (It only adjusts for variables you include and measure; it does not turn observational data into an experiment.)

# %% [markdown]
# ## 2. Simple linear regression with the formula API
#
# `smf.ols("y ~ x", data=df)` uses R-style formulas: `~` reads "modelled by", `+` adds terms, `C(col)` treats a column as categorical, `np.log(col)` transforms inline. `.fit()` returns a results object.

# %%
m1 = smf.ols("price_k ~ area_sqft", data=h).fit()
print(m1.summary().tables[1])
print(f"R-squared = {m1.rsquared:.3f}")

# %% [markdown]
# **How to read the table.**
# - `coef`: the slope: each extra square foot adds about that many thousand dollars on average (multiply by 1000 for dollars).
# - `std err`: standard error of the coefficient: how much it would wobble across samples (same idea as the standard error of a mean).
# - `t` = coef / std err, and `P>|t|` tests the **null hypothesis that the true coefficient is 0**. Small p means the slope is unlikely to be pure noise. It says nothing about whether the effect is big.
# - `[0.025, 0.975]`: 95% confidence interval for the coefficient. Prefer it to the p-value: it shows plausible magnitudes.
# - **R-squared**: the share of the variance of price explained by the model (0 to 1). It does not tell you the model is correct or the coefficients causal.
#
# **Assumptions behind these p-values and intervals:** (1) the mean of `y` is linear in the predictors, (2) residuals have constant variance (homoscedasticity), (3) residuals are independent, (4) residuals are roughly normal (matters mostly for small samples; the CLT covers large ones).
# We test them in section 4.

# %% [markdown]
# ## 3. Multiple regression with categories, and a log model

# %%
formula = "price_k ~ area_sqft + bedrooms + bathrooms + age_years + garage_spaces + distance_to_center_km + quality_score + C(neighborhood)"
m2 = smf.ols(formula, data=h).fit()
print(m2.summary().tables[1])
print(f"n used = {int(m2.nobs)} of {len(h)} (rows with missing values are dropped), R-squared = {m2.rsquared:.3f}, adjusted = {m2.rsquared_adj:.3f}")

# %% [markdown]
# **Reading a multiple regression:** each coefficient is the change in price for a one-unit increase in that variable **holding the others constant**. For example, each extra year of age is associated with about
# `-0.8` thousand dollars, each quality point with about `+14`. `C(neighborhood)[T.Eastgate]` is the difference from the reference neighborhood (alphabetically first, Downtown) after adjusting for everything else.
# Notice `bedrooms` dropped to a small coefficient once area and bathrooms were in the model.

# %% [markdown]
# ### Diagnostics: is the linear model on raw price trustworthy?

# %%
def diagnostics(model, title):
    fitted, resid = model.fittedvalues, model.resid
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].scatter(fitted, resid, s=5, alpha=0.4); axes[0].axhline(0, color="k")
    axes[0].set_xlabel("fitted value"); axes[0].set_ylabel("residual"); axes[0].set_title(f"{title}: residuals vs fitted")
    stats.probplot(resid, dist="norm", plot=axes[1]); axes[1].set_title("QQ plot of residuals")
    axes[2].hist(resid, bins=50, color="lightsteelblue"); axes[2].set_xlabel("residual"); axes[2].set_ylabel("count"); axes[2].set_title("Residual histogram")
    plt.tight_layout(); plt.show()
    lm, lm_p, _, _ = het_breuschpagan(resid, model.model.exog)
    print(f"Breusch-Pagan test (H0: constant variance): p = {lm_p:.4f}")

diagnostics(m2, "price")

# %% [markdown]
# *What to look for:* residuals vs fitted should be a formless cloud around 0. A **funnel** (spread growing with the fitted value) means **heteroscedasticity**: the model is less certain about expensive houses, and the standard
# errors and p-values above are unreliable. The **Breusch-Pagan test** formalises this: **null = constant variance**, assumptions = a correctly specified mean model and independent errors. A small p flags non-constant variance
# (with thousands of rows it also flags mild problems, so read the plot too). The usual fix for positive, right-skewed targets is to model **log(price)**: multiplicative effects become additive and the spread evens out.

# %%
formula_log = "np.log(price_k) ~ area_sqft + bedrooms + bathrooms + age_years + garage_spaces + distance_to_center_km + quality_score + C(neighborhood)"
m3 = smf.ols(formula_log, data=h).fit()
diagnostics(m3, "log(price)")
print(f"R-squared, price model: {m2.rsquared:.3f}   log-price model: {m3.rsquared:.3f}   (NOT comparable: they explain variance of different targets)")

# %% [markdown]
# The log model gets a formless residual cloud and the Breusch-Pagan test no longer rejects, at the cost of a lower (but not comparable) R-squared. A model with a higher R-squared is not automatically the better one; validity of the assumptions matters for inference.
# In a log model a coefficient `b` means "+1 unit multiplies price by `exp(b)`", i.e. about `100*b` percent for small `b`:

# %%
coefs = m3.params.drop("Intercept").to_frame("coef")
coefs["% change in price per unit"] = (np.exp(coefs["coef"]) - 1) * 100
ci = m3.conf_int().drop("Intercept")
coefs["95% CI low %"] = (np.exp(ci[0]) - 1) * 100; coefs["95% CI high %"] = (np.exp(ci[1]) - 1) * 100
coefs.drop(columns="coef").round(2)

# %% [markdown]
# For instance, each extra garage space is associated with roughly the percentage shown in the table, holding the other variables fixed.
#
# ### Influential points and multicollinearity
# **Cook's distance** measures how much the fit would change if one observation were removed. **VIF** (variance inflation factor) measures how well the other predictors explain one predictor; large values (rule of thumb above 5 to 10)
# mean the coefficient is poorly determined because predictors overlap (**multicollinearity**): it inflates standard errors and makes coefficients unstable, though predictions may be fine.

# %%
cooks = m3.get_influence().cooks_distance[0]
print(f"largest Cook's distance = {cooks.max():.3f}; observations above 4/n: {(cooks > 4 / len(cooks)).sum()} of {len(cooks)}")
X = m2.model.exog
vif = pd.Series([variance_inflation_factor(X, i) for i in range(X.shape[1])], index=m2.model.exog_names).drop("Intercept")
vif.round(2).sort_values(ascending=False).head(4)

# %% [markdown]
# VIFs are modest here (`bedrooms` is highest because it moves with area and bathrooms). If two predictors were near-duplicates (say area in sqft and area in square metres) the VIF would explode and the coefficients would become meaningless individually.

# %% [markdown]
# ## 4. Logistic regression: a probability for a yes/no outcome
#
# For a 0/1 outcome such as churn, linear regression can predict impossible probabilities. **Logistic regression** models the **log-odds**: `log(p / (1-p)) = b0 + b1*x1 + ...`. **Odds** are `p / (1-p)`
# (a 50% chance is odds of 1; 75% is odds of 3). Exponentiating a coefficient gives an **odds ratio (OR)**: the multiplicative change in the odds for a one-unit increase, holding the others fixed. OR = 1 means no effect; above 1 raises the odds; below 1 lowers them.
# We fit churn on contract, tenure, price, support calls and a few others, using month-to-month as the reference contract:

# %%
f_logit = ("churned ~ tenure_months + C(contract, Treatment(reference='month-to-month')) + monthly_charges + support_calls"
           " + senior_citizen + C(internet_service) + C(autopay) + C(paperless_billing)")
logit = smf.logit(f_logit, data=churn).fit(disp=0)

ors = pd.DataFrame({"odds ratio": np.exp(logit.params), "CI low": np.exp(logit.conf_int()[0]), "CI high": np.exp(logit.conf_int()[1]), "p": logit.pvalues}).drop("Intercept")
ors.index = [i.replace("C(contract, Treatment(reference='month-to-month'))", "contract").replace("C(", "").replace(")", "") for i in ors.index]
ors.round(3)

# %% [markdown]
# **Interpretation.** Holding the other variables fixed: a two-year contract multiplies the churn odds by the tiny odds ratio shown (much lower than month-to-month), each extra support call multiplies them by about 1.3
# (a **30% increase in odds**, not in probability), each extra month of tenure by about 0.96. `senior_citizen` has an interval around 1 and a large p-value: no detectable effect once the others are included.
# Odds ratios are not probability ratios; for common outcomes (churn is 37%) the odds change more than the probability. To talk in probabilities, predict:

# %%
profile = pd.DataFrame({"tenure_months": [6, 6], "contract": ["month-to-month"] * 2, "monthly_charges": [70.0, 70.0], "support_calls": [1, 5],
                        "senior_citizen": [0, 0], "internet_service": ["Fiber"] * 2, "autopay": ["No"] * 2, "paperless_billing": ["Yes"] * 2})
print("predicted churn probability, 1 vs 5 support calls:", logit.predict(profile).round(3).tolist())

from sklearn.metrics import roc_auc_score
print(f"in-sample AUC = {roc_auc_score(churn['churned'], logit.predict(churn)):.3f}  (optimistic: same data used to fit and score; model evaluation is a later module)")

# %% [markdown]
# **Assumptions for logistic regression:** independent observations, a linear relationship between the predictors and the *log-odds*, no severe multicollinearity, and enough events per predictor. (Note `total_charges` was left out on purpose:
# it is roughly `tenure * monthly_charges`, so it would be almost perfectly collinear with two other predictors.)

# %% [markdown]
# ## Common mistakes
#
# 1. **Reading correlation as causation**, or a coefficient as causal when the data is observational and confounders may be missing.
# 2. **Trusting p-values from a model whose residuals show a funnel or curve.** Diagnose first.
# 3. **Comparing R-squared across different targets** (price vs log price) or celebrating a high R-squared on a badly specified model.
# 4. **Reading an odds ratio as a probability ratio.**
# 5. **Extrapolating** beyond the range of the data (a 20,000 sqft house).
# 6. **Ignoring dropped rows**: statsmodels silently drops rows with missing values in the formula (see `nobs`).
# 7. **Interpreting a coefficient's p-value as its importance.** Compare effects on a common scale (standardise, or look at intervals in meaningful units).

# %%
print(f"rows in data: {len(h)}, rows used by the full housing model: {int(m2.nobs)}  -> {len(h) - int(m2.nobs)} silently dropped for missing age or quality")

# %% [markdown]
# ## Interview corner
#
# **1. How do you interpret a regression coefficient?**
# <details><summary>Answer</summary>
# The expected change in the outcome for a one-unit increase in that predictor, holding the other predictors fixed, with a standard error and confidence interval. In a log-outcome model it is an approximate percentage change; in logistic regression its exponential is an odds ratio. It is a statement about association unless the design supports causation.
# </details>
#
# **2. What are the assumptions of linear regression and how do you check them?**
# <details><summary>Answer</summary>
# Linearity of the mean, independent errors, constant error variance, roughly normal errors, and no extreme multicollinearity or influential points. Check residuals vs fitted, a QQ plot, the Breusch-Pagan test, VIFs and Cook's distance.
# </details>
#
# **3. What is multicollinearity and why does it matter?**
# <details><summary>Answer</summary>
# Predictors that are strongly linearly related. Coefficient estimates become unstable with inflated standard errors (VIF measures this), so individual effects are hard to separate although overall predictions can remain good. Fix by dropping or combining predictors, or using regularisation.
# </details>
#
# **4. A logistic regression gives an odds ratio of 1.3 for `support_calls`. What does that mean?**
# <details><summary>Answer</summary>
# Each additional support call multiplies the odds of churn by 1.3 (a 30% increase in odds), holding the other variables fixed. The change in probability depends on the starting probability.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Pearson vs Spearman
# Using rows of `h` with no missing `distance_to_center_km`, compute `r_pearson` and `r_spearman` between `distance_to_center_km` and `price_k`. Store `weak` as True if `abs(r_pearson) < 0.3`.
# Then compute `r_spearman_area` between `area_sqft` and `price_k`.

# %% tags=["starter"]
r_pearson = ...
r_spearman = ...
weak = ...
r_spearman_area = ...

# %%
d_ = h.dropna(subset=["distance_to_center_km"])
assert abs(r_pearson - np.corrcoef(d_["distance_to_center_km"], d_["price_k"])[0, 1]) < 1e-9, "Pearson between distance and price_k"
assert abs(r_spearman - stats.spearmanr(d_["distance_to_center_km"], d_["price_k"])[0]) < 1e-9, "Spearman (rank correlation)"
assert bool(weak) == (abs(r_pearson) < 0.3)
assert abs(r_spearman_area - h["area_sqft"].corr(h["price_k"], method="spearman")) < 1e-9, "Spearman between area_sqft and price_k"
print(f"Exercise 1 passed: distance r = {r_pearson:.3f}, rho = {r_spearman:.3f}; area rho = {r_spearman_area:.3f}")

# %% tags=["solution"]
_d = h.dropna(subset=["distance_to_center_km"])
r_pearson = float(np.corrcoef(_d["distance_to_center_km"], _d["price_k"])[0, 1])
r_spearman = float(stats.spearmanr(_d["distance_to_center_km"], _d["price_k"])[0])
weak = bool(abs(r_pearson) < 0.3)
r_spearman_area = float(h["area_sqft"].corr(h["price_k"], method="spearman"))

# %% [markdown]
# ### Exercise 2: Interpret a log-price coefficient
# Fit `np.log(price_k) ~ area_sqft + quality_score + age_years` on `h` with `smf.ols` (store the fitted results as `model_ex`). Store in `pct_per_100sqft` the percentage change in price associated with **100 extra square feet**, computed exactly as
# `(exp(100 * b_area) - 1) * 100`, and in `r2_ex` the model's R-squared.

# %% tags=["starter"]
model_ex = ...
pct_per_100sqft = ...
r2_ex = ...

# %%
d2 = h.dropna(subset=["area_sqft", "quality_score", "age_years"])
Xd = np.column_stack([np.ones(len(d2)), d2["area_sqft"], d2["quality_score"], d2["age_years"]])
beta = np.linalg.lstsq(Xd, np.log(d2["price_k"]), rcond=None)[0]
assert int(model_ex.nobs) == len(d2), "the model should use the rows without missing values in those three columns (formula API drops them)"
assert abs(pct_per_100sqft - (np.exp(100 * beta[1]) - 1) * 100) < 1e-6, "compute from the area_sqft coefficient"
fit_ = Xd @ beta; y_ = np.log(d2["price_k"])
assert abs(r2_ex - (1 - ((y_ - fit_) ** 2).sum() / ((y_ - y_.mean()) ** 2).sum())) < 1e-9, "R-squared of your model"
print(f"Exercise 2 passed: +100 sqft is associated with {pct_per_100sqft:.1f}% higher price, R2 = {r2_ex:.3f}")

# %% tags=["solution"]
model_ex = smf.ols("np.log(price_k) ~ area_sqft + quality_score + age_years", data=h).fit()
pct_per_100sqft = float((np.exp(100 * model_ex.params["area_sqft"]) - 1) * 100)
r2_ex = float(model_ex.rsquared)

# %% [markdown]
# ### Exercise 3: An odds ratio with a confidence interval
# Fit a logistic regression `churned ~ support_calls + tenure_months + C(contract, Treatment(reference='month-to-month'))` on `churn` (store as `logit_ex`). Store in `or_support` the odds ratio for one extra support call, and in
# `or_ci` its 95% interval as a tuple `(low, high)`. Store `excludes_one` as whether that interval excludes 1.

# %% tags=["starter"]
logit_ex = ...
or_support = ...
or_ci = ...
excludes_one = ...

# %%
from sklearn.linear_model import LogisticRegression
Xs = pd.get_dummies(churn[["support_calls", "tenure_months", "contract"]], columns=["contract"], drop_first=False).drop(columns="contract_month-to-month").astype(float)
skl = LogisticRegression(C=1e10, max_iter=5000).fit(Xs, churn["churned"])
skl_or = float(np.exp(skl.coef_[0][list(Xs.columns).index("support_calls")]))
assert abs(or_support - skl_or) < 0.01, f"odds ratio should be about {skl_or:.3f} (an independent fit gives this)"
assert len(or_ci) == 2 and or_ci[0] < or_support < or_ci[1], "the interval must bracket the point estimate"
assert abs(or_ci[0] - np.exp(logit_ex.conf_int().loc["support_calls", 0])) < 1e-9, "use exp of the coefficient interval from conf_int()"
assert bool(excludes_one) == (or_ci[0] > 1 or or_ci[1] < 1)
print(f"Exercise 3 passed: OR = {or_support:.3f}, 95% CI [{or_ci[0]:.3f}, {or_ci[1]:.3f}]")

# %% tags=["solution"]
logit_ex = smf.logit("churned ~ support_calls + tenure_months + C(contract, Treatment(reference='month-to-month'))", data=churn).fit(disp=0)
or_support = float(np.exp(logit_ex.params["support_calls"]))
_ci = logit_ex.conf_int().loc["support_calls"]
or_ci = (float(np.exp(_ci[0])), float(np.exp(_ci[1])))
excludes_one = bool(or_ci[0] > 1 or or_ci[1] < 1)

# %% [markdown]
# ### Exercise 4 (open-ended): Explain the model to a stakeholder
# Take the full logistic model from section 4. In a markdown cell, write five plain-language sentences for a retention manager: which factors most increase churn risk, which reduce it, one factor with no detectable effect, and one caveat
# (association vs cause, the missing `internet_service` category, or in-sample evaluation). Support at least two sentences with a predicted probability computed by `logit.predict` for a specific customer profile.

# %% [markdown]
# ## Recap
# - Pearson measures linear association, Spearman monotonic; both hide non-linearity and outliers, so plot. Confounders create correlation without causation.
# - OLS coefficients are "change in y per unit of x, holding the others fixed", with SEs, p-values (null: coefficient is 0) and confidence intervals.
# - Diagnose residuals (funnel, curve, QQ), test constant variance (Breusch-Pagan), check Cook's distance and VIF; a log target often fixes skewed, funnel-shaped residuals.
# - Logistic regression models log-odds; exponentiated coefficients are odds ratios, not probability ratios.
# - The formula API silently drops rows with missing values; count `nobs`.
#
# **Go deeper:** the statsmodels documentation ("Linear Regression", "Generalized Linear Models" and the regression diagnostics page); "An Introduction to Statistical Learning" (free PDF) chapters 3 and 4.
#
# **Next:** [Bayesian thinking: an introduction](09_bayesian_thinking_intro.ipynb)

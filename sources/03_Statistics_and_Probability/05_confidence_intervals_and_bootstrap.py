# %% [markdown]
# # Confidence Intervals and the Bootstrap
#
# **Why this matters.** "Churn is 37%" is a claim about a *sample*. Stakeholders need to know how much to trust it: is it 37 plus or minus 1, or plus or minus 10? A confidence interval
# is the honest way to report an estimate. The **bootstrap** gives you one for almost any statistic (median, ratio, 90th percentile) using nothing but a `for` loop, which is why it is a daily tool in
# data-science interviews and in practice.
#
# **You will learn**
# - build analytic confidence intervals for a proportion (Wilson) and a mean (t-interval)
# - interpret "95% confidence" correctly (a statement about the *procedure*) and test it by simulation
# - build bootstrap intervals for the mean, median and any custom statistic
# - know when each method fails: small samples, extreme proportions, skew, max/min statistics
#
# **Prerequisites:** [Sampling, CLT and standard error](04_sampling_clt_and_standard_error.ipynb).
#
# > Data used: `churn.csv` and `ab_test.csv` (synthetic, see `docs/DATASETS.md`).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

DATA = Path("../datasets")
rng = np.random.default_rng(42)
churn = pd.read_csv(DATA / "churn.csv")
churn["contract"] = churn["contract"].str.lower()
ab = pd.read_csv(DATA / "ab_test.csv")

# %% [markdown]
# ## 1. A point estimate is not enough
#
# Our best guess for the churn rate is the sample proportion. But a different sample would give a different number. An **interval estimate** reports a range of plausible values.
# The recipe for a large-sample interval: `estimate +/- z * SE`, with `z = 1.96` for 95%.

# %%
n = len(churn)
p_hat = churn["churned"].mean()
se = np.sqrt(p_hat * (1 - p_hat) / n)
lo, hi = p_hat - 1.96 * se, p_hat + 1.96 * se
print(f"churn rate = {p_hat:.3f}, SE = {se:.4f}, 95% CI = [{lo:.3f}, {hi:.3f}]  (n = {n})")

# %% [markdown]
# On 5,000 customers we pin the rate down to roughly plus or minus 1.3 points. Now take just 100 customers:

# %%
small = churn["churned"].sample(100, random_state=42)
ph = small.mean(); s_e = np.sqrt(ph * (1 - ph) / 100)
print(f"n = 100: churn rate = {ph:.2f}, 95% CI = [{ph - 1.96 * s_e:.2f}, {ph + 1.96 * s_e:.2f}]  -> much wider")

# %% [markdown]
# ## 2. What does "95% confidence" actually mean?
#
# It does **not** mean "there is a 95% probability that the true value is in this interval" (the true value is a fixed number; it either is or is not). It means: *if we repeated the whole procedure many times, about
# 95% of the intervals we build would contain the true value.* We can test that claim, because here we know the truth (we treat the 5,000-customer file as the full population).

# %%
true_p = churn["churned"].mean()
reps, size = 100, 200
samples = rng.binomial(1, true_p, size=(reps, size))
ph = samples.mean(axis=1)
half = 1.96 * np.sqrt(ph * (1 - ph) / size)
covered = (ph - half <= true_p) & (true_p <= ph + half)

fig, ax = plt.subplots(figsize=(6, 6))
for i in range(reps):
    ax.plot([ph[i] - half[i], ph[i] + half[i]], [i, i], color="tab:blue" if covered[i] else "tab:red", lw=1.5)
ax.axvline(true_p, color="k", ls="--", label="true rate")
ax.set_xlabel("churn rate"); ax.set_ylabel("repeat experiment #"); ax.set_title(f"100 repeated 95% CIs (n=200 each): {covered.sum()} contain the truth")
ax.legend(); plt.show()

# %% [markdown]
# Each horizontal line is one experiment's interval; the red ones missed the true value. About 5 in 100 miss, as advertised. A given interval either covers the truth or not; the 95% describes how reliable the *method* is.

# %% [markdown]
# ## 3. Better intervals for proportions: Wilson
#
# The interval above (the **Wald** interval) is easy but unreliable for small `n` or for proportions near 0 or 1: it can even fall below 0. The **Wilson** interval fixes this and should be your default.
# Compare their real coverage when `n = 30` and the true rate is 5%:

# %%
def coverage(method, true_p, size, reps=4000):
    x = rng.binomial(size, true_p, size=reps)
    lo_, hi_ = proportion_confint(x, size, alpha=0.05, method=method)
    return np.mean((lo_ <= true_p) & (true_p <= hi_))

print(f"Wald   coverage (nominal 95%): {coverage('normal', 0.05, 30):.3f}")
print(f"Wilson coverage (nominal 95%): {coverage('wilson', 0.05, 30):.3f}")
print("e.g. 0 successes out of 30:", "Wald =", np.round(proportion_confint(0, 30, method="normal"), 3), " Wilson =", np.round(proportion_confint(0, 30, method="wilson"), 3))

# %% [markdown]
# *What just happened?* The Wald interval covers noticeably less than 95% for these settings and collapses to a zero-width interval when there are zero successes, which is absurd. Wilson stays close to nominal.
# Use `statsmodels.stats.proportion.proportion_confint(count, nobs, method="wilson")`.

# %%
mtm = churn[churn["contract"] == "month-to-month"]["churned"]
lo, hi = proportion_confint(mtm.sum(), len(mtm), method="wilson")
print(f"month-to-month churn: {mtm.mean():.3f}, 95% Wilson CI = [{lo:.3f}, {hi:.3f}]  (n = {len(mtm)})")

# %% [markdown]
# ## 4. Confidence interval for a mean: the t-interval
#
# `mean +/- t * s / sqrt(n)` where `t` comes from the Student t distribution with `n-1` degrees of freedom. It is wider than the `1.96` version for small `n` because estimating `sigma` from the same data adds uncertainty.
# For `n` in the hundreds, `t` is nearly 1.96.

# %%
for dof in [4, 9, 29, 99, 999]:
    print(f"n = {dof + 1:4d}: t multiplier for 95% = {stats.t.ppf(0.975, dof):.3f}")

# %%
rev = ab["revenue"].to_numpy()
m, s_e = rev.mean(), stats.sem(rev)
t_lo, t_hi = stats.t.interval(0.95, len(rev) - 1, loc=m, scale=s_e)
print(f"mean revenue per user = {m:.3f}, t-interval = [{t_lo:.3f}, {t_hi:.3f}]")

# %% [markdown]
# ## 5. The bootstrap: simulate the sampling process from your own data
#
# You have one sample and cannot draw new ones from the population. The **bootstrap** idea: treat the sample as a stand-in for the population and **resample from it with replacement**, same size as the original.
# Compute the statistic on each resample; the spread of those values approximates its sampling distribution. The **percentile interval** simply takes the 2.5th and 97.5th percentiles.

# %%
def bootstrap_ci(data, stat=np.mean, reps=5000, alpha=0.05, seed=0):
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(data), size=(reps, len(data)))
    boots = np.array([stat(data[i]) for i in idx]) if stat not in (np.mean, np.median) else stat(data[idx], axis=1)
    return np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)]), boots

(b_lo, b_hi), boots = bootstrap_ci(rev, np.mean)
print(f"bootstrap CI for mean revenue: [{b_lo:.3f}, {b_hi:.3f}]   t-interval: [{t_lo:.3f}, {t_hi:.3f}]")

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.hist(boots, bins=60, color="lightsteelblue")
ax.axvline(b_lo, color="tab:red"); ax.axvline(b_hi, color="tab:red")
ax.set_xlabel("mean revenue per user in a resample ($)"); ax.set_ylabel("resamples"); ax.set_title("Bootstrap distribution of the mean")
plt.show()

# %% [markdown]
# With 8,000 users the bootstrap and the t-interval nearly coincide. The bootstrap earns its keep on statistics with **no simple formula**:

# %%
paying = rev[rev > 0]
(m_lo, m_hi), _ = bootstrap_ci(paying, np.median)
(q_lo, q_hi), _ = bootstrap_ci(paying, lambda a: np.percentile(a, 90))
print(f"{len(paying)} paying users: median order value = {np.median(paying):.2f}, 95% CI [{m_lo:.2f}, {m_hi:.2f}]")
print(f"90th percentile = {np.percentile(paying, 90):.2f}, 95% CI [{q_lo:.2f}, {q_hi:.2f}]")

# %% [markdown]
# Or a difference between two groups, which is what an A/B test needs. Bootstrap each group separately, then subtract:

# %%
ctrl = ab.loc[ab["group"] == "control", "converted"].to_numpy()
trt = ab.loc[ab["group"] == "treatment", "converted"].to_numpy()
r = np.random.default_rng(1)
diffs = np.array([r.choice(trt, len(trt)).mean() - r.choice(ctrl, len(ctrl)).mean() for _ in range(5000)])
lo_d, hi_d = np.percentile(diffs, [2.5, 97.5])
print(f"conversion lift (treatment - control) = {trt.mean() - ctrl.mean():+.4f}, bootstrap 95% CI = [{lo_d:+.4f}, {hi_d:+.4f}]")

# %% [markdown]
# The interval includes 0: the data cannot rule out "no effect", and also cannot rule out a lift of over a percentage point. The interval shows the *range of effects consistent with the data*, which is more informative than a yes/no verdict.
# SciPy also ships `scipy.stats.bootstrap`, which offers a better-behaved variant (BCa):

# %%
res = stats.bootstrap((paying,), np.median, confidence_level=0.95, n_resamples=5000, random_state=0, method="BCa")
print("scipy BCa CI for the paying-user median:", np.round([res.confidence_interval.low, res.confidence_interval.high], 2))

# %% [markdown]
# ## 6. When intervals fail
#
# - **Wald for small n / extreme p:** shown above; use Wilson.
# - **t-interval for skewed data with small n:** the CLT has not kicked in. Check coverage by simulation (Exercise 3).
# - **Bootstrap with tiny samples (n < ~20) or for extremes** (max, min, very high quantiles): resamples cannot produce values outside the data, so the interval is too narrow.
# - **Dependent data** (time series, users with several rows): naive resampling of rows breaks the dependence and gives intervals that are too narrow; resample whole users or blocks.
# - **A CI is not a prediction interval.** It says where the *mean* likely is, not where the next customer's value falls.

# %%
sample_max = np.array([3.0, 5.0, 6.0, 9.0, 10.0, 12.0, 13.0, 20.0])
r = np.random.default_rng(3)
boot_max = np.array([r.choice(sample_max, len(sample_max)).max() for _ in range(5000)])
print("bootstrap 95% CI for the population maximum:", np.percentile(boot_max, [2.5, 97.5]), "-> its upper end can never exceed the sample max of", sample_max.max())

# %% [markdown]
# ## Common mistakes
#
# 1. **Saying "95% probability the true mean is in the interval".** The correct reading is about the long-run behaviour of the method (a Bayesian *credible* interval does make the probability statement; see notebook 09).
# 2. **Using the Wald interval** with small samples or rare events.
# 3. **Comparing two groups by checking whether their separate CIs overlap.** Overlap does not imply "not significant"; build a CI for the *difference*.
# 4. **Bootstrapping without a seed** (irreproducible) or with too few resamples (a few thousand is a fine default).
# 5. **Reading a narrow interval as proof the estimate is right.** It only reflects sampling noise, not bias or a flawed measurement.

# %%
g = ab.groupby("group")["converted"].agg(["sum", "count"])
for name, row in g.iterrows():
    lo_, hi_ = proportion_confint(row["sum"], row["count"], method="wilson")
    print(f"{name:9s} conversion 95% CI: [{lo_:.4f}, {hi_:.4f}]")
print("the two intervals overlap heavily. Overlap is a crude check; the proper comparison is the interval for the difference (section 5)")

# %% [markdown]
# ## Interview corner
#
# **1. How do you interpret a 95% confidence interval?**
# <details><summary>Answer</summary>
# If we repeated the sampling and interval construction many times, about 95% of the intervals would contain the true parameter. Any single interval either contains it or not; it is not a 95% probability statement about the parameter (that is what a Bayesian credible interval gives, under a prior).
# </details>
#
# **2. How would you get a confidence interval for the median of order value?**
# <details><summary>Answer</summary>
# Bootstrap: resample the orders with replacement many times, compute the median of each resample, and take the 2.5th and 97.5th percentiles (or BCa). No simple closed-form standard error exists for a median.
# </details>
#
# **3. What makes an interval narrower?**
# <details><summary>Answer</summary>
# More data (width scales as 1/sqrt(n)), less variable data, or a lower confidence level (which trades coverage for width). Better measurement or a stratified design reduces variance too.
# </details>
#
# **4. When should you not trust the bootstrap?**
# <details><summary>Answer</summary>
# Very small samples, statistics that depend on the extremes (max, min, extreme quantiles), heavy-tailed data with infinite variance, and dependent data unless you resample in blocks or clusters.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: A Wilson interval by hand and by library
# For customers on **two-year** contracts (casing already fixed, contract equals `"two year"`), compute `ci_lib`, the 95% Wilson interval `(low, high)` from `proportion_confint`, and `ci_wald`, the hand-built Wald interval
# `p +/- 1.96*sqrt(p(1-p)/n)` as a tuple. Store `wilson_narrower` as whether the Wilson interval is narrower than the Wald one.

# %% tags=["starter"]
ci_lib = ...
ci_wald = ...
wilson_narrower = ...

# %%
tw = churn.loc[churn["contract"] == "two year", "churned"]
e_lo, e_hi = proportion_confint(tw.sum(), len(tw), alpha=0.05, method="wilson")
assert abs(ci_lib[0] - e_lo) < 1e-9 and abs(ci_lib[1] - e_hi) < 1e-9, "use proportion_confint(count, nobs, method='wilson')"
ph_ = tw.mean(); h_ = 1.96 * np.sqrt(ph_ * (1 - ph_) / len(tw))
assert abs(ci_wald[0] - (ph_ - h_)) < 1e-9 and abs(ci_wald[1] - (ph_ + h_)) < 1e-9, "Wald: p +/- 1.96*sqrt(p(1-p)/n)"
assert bool(wilson_narrower) == ((e_hi - e_lo) < (2 * h_))
print(f"Exercise 1 passed: n = {len(tw)}, Wilson [{ci_lib[0]:.4f}, {ci_lib[1]:.4f}], Wald [{ci_wald[0]:.4f}, {ci_wald[1]:.4f}]")

# %% tags=["solution"]
_tw = churn.loc[churn["contract"] == "two year", "churned"]
ci_lib = tuple(proportion_confint(_tw.sum(), len(_tw), alpha=0.05, method="wilson"))
_p = _tw.mean(); _h = 1.96 * np.sqrt(_p * (1 - _p) / len(_tw))
ci_wald = (_p - _h, _p + _h)
wilson_narrower = bool((ci_lib[1] - ci_lib[0]) < 2 * _h)

# %% [markdown]
# ### Exercise 2: Bootstrap the median monthly charge
# Bootstrap the **median of `monthly_charges`** for churned customers (`churned == 1`): 3,000 resamples using `np.random.default_rng(5)`. Store the percentile 95% interval in `ci_median` as a tuple `(low, high)`.
# Also store `point` (the sample median). *Hint: `idx = rng.integers(0, n, size=(3000, n))`, then `np.median(data[idx], axis=1)`.*

# %% tags=["starter"]
ci_median = ...
point = ...

# %%
d_ = churn.loc[churn["churned"] == 1, "monthly_charges"].to_numpy()
assert abs(point - np.median(d_)) < 1e-9, "point is the plain sample median"
assert ci_median[0] < point < ci_median[1] or ci_median[0] <= point <= ci_median[1], "the interval should contain the point estimate"
ref = stats.bootstrap((d_,), np.median, n_resamples=3000, random_state=99, method="percentile").confidence_interval
w_mine, w_ref = ci_median[1] - ci_median[0], ref.high - ref.low
assert 0.6 < w_mine / w_ref < 1.6, f"width {w_mine:.2f} should be similar to an independent bootstrap ({w_ref:.2f}); resample with replacement, same size as the data"
print(f"Exercise 2 passed: median {point:.2f}, CI [{ci_median[0]:.2f}, {ci_median[1]:.2f}]")

# %% tags=["solution"]
_d = churn.loc[churn["churned"] == 1, "monthly_charges"].to_numpy()
_r = np.random.default_rng(5)
_idx = _r.integers(0, len(_d), size=(3000, len(_d)))
_meds = np.median(_d[_idx], axis=1)
ci_median = tuple(np.percentile(_meds, [2.5, 97.5]))
point = float(np.median(_d))

# %% [markdown]
# ### Exercise 3: Does the t-interval really cover 95%?
# The revenue data is very skewed. Repeat 2,000 times: draw a sample of size `n` from `rev` (with replacement, `np.random.default_rng(11)`), build the 95% t-interval for the mean, and check if it contains the population mean `rev.mean()`.
# Store the fraction covered as `coverage_30` (n = 30) and `coverage_1000` (n = 1000). *Hint: vectorise with a `(2000, n)` index array; `s = std(ddof=1)`, `t = stats.t.ppf(0.975, n-1)`.*

# %% tags=["starter"]
coverage_30 = ...
coverage_1000 = ...

# %%
def cover(n, seed=11, reps=2000):
    r_ = np.random.default_rng(seed)
    x = rev[r_.integers(0, len(rev), size=(reps, n))]
    m_, s_ = x.mean(axis=1), x.std(axis=1, ddof=1)
    h_ = stats.t.ppf(0.975, n - 1) * s_ / np.sqrt(n)
    return np.mean((m_ - h_ <= rev.mean()) & (rev.mean() <= m_ + h_))

assert abs(coverage_30 - cover(30)) < 0.03, f"n=30 coverage should be about {cover(30):.3f}"
assert abs(coverage_1000 - cover(1000)) < 0.03, f"n=1000 coverage should be about {cover(1000):.3f}"
assert coverage_30 < coverage_1000, "coverage should be worse for the small sample of skewed data"
print(f"Exercise 3 passed: coverage at n=30: {coverage_30:.3f}, at n=1000: {coverage_1000:.3f} (nominal 0.95)")

# %% tags=["solution"]
def _cover(n, seed=11, reps=2000):
    r_ = np.random.default_rng(seed)
    x = rev[r_.integers(0, len(rev), size=(reps, n))]
    m_, s_ = x.mean(axis=1), x.std(axis=1, ddof=1)
    h_ = stats.t.ppf(0.975, n - 1) * s_ / np.sqrt(n)
    return float(np.mean((m_ - h_ <= rev.mean()) & (rev.mean() <= m_ + h_)))

coverage_30 = _cover(30)
coverage_1000 = _cover(1000)

# %% [markdown]
# ### Exercise 4 (open-ended): Revenue per *converter*
# Using only users with `converted == 1`, build a bootstrap 95% CI for the mean revenue in each group and for the **difference** (treatment - control). Report the three intervals in a markdown cell and say whether you would conclude that
# treatment changes order value among converters, and what sample size you would want to say more.

# %% [markdown]
# ## Recap
# - A confidence interval reports the range of plausible values; "95%" is a property of the method under repeated sampling, and you can verify it by simulation.
# - Use Wilson (not Wald) for proportions and the t-interval for means; both scale with `1/sqrt(n)`.
# - The bootstrap (resample with replacement, recompute, take percentiles) gives intervals for almost any statistic; prefer BCa via `scipy.stats.bootstrap` when unsure.
# - Intervals fail for tiny samples, extremes, dependent data and skew with small `n`; a narrow interval says nothing about bias.
# - For comparisons, build an interval for the *difference*, not two separate intervals.
#
# **Go deeper:** the statsmodels docs for `proportion_confint`; the SciPy docs for `scipy.stats.bootstrap`; Efron and Tibshirani's "An Introduction to the Bootstrap" is the classic text.
#
# **Next:** [Hypothesis testing](06_hypothesis_testing.ipynb)

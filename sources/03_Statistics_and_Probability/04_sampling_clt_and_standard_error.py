# %% [markdown]
# # Sampling, the Central Limit Theorem and Standard Error
#
# **Why this matters.** You almost never measure everyone: you survey 500 users, run an experiment on 8,000 visitors, inspect 100 parts. The question
# behind every dashboard number and A/B test is *"how much would this number change if I had drawn a different sample?"* The **standard error** answers it, and the **central
# limit theorem** explains why the answer has a simple, bell-shaped form even when your data is nothing like a bell. This is the bridge from descriptive statistics to inference.
#
# **You will learn**
# - distinguish population, sample, parameter and statistic
# - simulate the **sampling distribution** of the mean and watch the CLT emerge from skewed data
# - compute standard errors for means and proportions and see the `1/sqrt(n)` law
# - recognise when the CLT is slow or fails (very skewed, heavy tails) and when more data cannot rescue a biased sample
#
# **Prerequisites:** [Distributions](03_distributions.ipynb).
#
# > Data used: `ab_test.csv` (revenue per user is very skewed) and `housing.csv`, `churn.csv`. For the simulations we *pretend the full file is the entire population* and repeatedly sample from it.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DATA = Path("../datasets")
rng = np.random.default_rng(42)
ab = pd.read_csv(DATA / "ab_test.csv")
churn = pd.read_csv(DATA / "churn.csv")
housing = pd.read_csv(DATA / "housing.csv")

# %% [markdown]
# ## 1. Population vs sample
#
# - **Population**: everyone you care about (all 8,000 visitors to the page, past and future).
# - **Sample**: the ones you actually measured.
# - **Parameter**: a number describing the population (true mean revenue per user). Fixed, usually unknown.
# - **Statistic**: a number computed from a sample (sample mean). Changes from sample to sample.
#
# Here revenue per user is extremely skewed: most users spend nothing, a few spend a lot.

# %%
revenue = ab["revenue"].to_numpy()
print(f"share of users with zero revenue: {(revenue == 0).mean():.1%}")
print(f"population mean = {revenue.mean():.3f}, std = {revenue.std():.3f}, skewness = {stats.skew(revenue):.2f}")

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.hist(revenue, bins=60, color="lightsteelblue")
ax.set_yscale("log"); ax.set_xlabel("revenue per user ($)"); ax.set_ylabel("users (log scale)"); ax.set_title("Population: a spike at zero and a long right tail")
plt.show()

# %% [markdown]
# ## 2. The sampling distribution of the mean
#
# Draw a random sample of size `n`, compute its mean, and repeat 5,000 times. The 5,000 means form the **sampling distribution** of the mean: they show how much a sample mean wobbles by luck alone.

# %%
def sample_means(data, n, reps=5000):
    idx = rng.integers(0, len(data), size=(reps, n))       # sample with replacement so that n can exceed the file size
    return data[idx].mean(axis=1)

sizes = [2, 10, 50, 500]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharex=True)
for ax, n in zip(axes, sizes):
    m = sample_means(revenue, n)
    ax.hist(m, bins=50, density=True, color="lightsteelblue")
    xs = np.linspace(revenue.min(), 14, 300)
    ax.plot(xs, stats.norm.pdf(xs, revenue.mean(), revenue.std() / np.sqrt(n)), color="tab:red", lw=1.2)
    ax.set_title(f"n = {n}   skew of means = {stats.skew(m):.2f}"); ax.set_xlabel("sample mean ($)")
axes[0].set_ylabel("density")
plt.suptitle("Means of samples drawn from a very skewed population (red: normal predicted by the CLT)", y=1.03)
plt.tight_layout(); plt.show()

# %% [markdown]
# *What just happened?* For `n = 2` the means inherit the skew of the population. As `n` grows the distribution of the *mean* becomes narrower, more symmetric and eventually matches the red normal curve
# with centre `mu` and width `sigma / sqrt(n)`. That is the **central limit theorem**: for almost any population with a finite variance, the mean of `n` independent draws is approximately normal for large `n`.
# The population itself never became normal; only the *average* did.

# %% [markdown]
# ## 3. Standard error: the wobble of a statistic
#
# The **standard error (SE)** of the mean is the standard deviation of its sampling distribution, and the CLT tells us it equals `sigma / sqrt(n)`. Do not confuse it with the standard deviation `sigma` of the data:
# SD describes how spread out *individuals* are; SE describes how precisely the *mean* is pinned down. Verify the formula:

# %%
rows = []
for n in [10, 40, 160, 640, 2560]:
    m = sample_means(revenue, n)
    rows.append({"n": n, "empirical SE (std of 5000 means)": m.std(), "formula sigma/sqrt(n)": revenue.std() / np.sqrt(n)})
pd.DataFrame(rows).round(4)

# %% [markdown]
# They agree. Each time `n` is multiplied by 4, the SE halves. **Precision costs quadratically**: to get 10x more accurate you need 100x the data. This one fact drives sample-size planning for every survey and A/B test.
#
# In practice you do not know `sigma`, so you estimate it from the sample: `SE = s / sqrt(n)` (`scipy.stats.sem` does it).

# %%
one_sample = rng.choice(revenue, size=400, replace=False)
print(f"one sample of 400: mean = {one_sample.mean():.3f}, SE = {stats.sem(one_sample):.3f}  (true population mean {revenue.mean():.3f})")
print(f"so the estimate is typically within about {2 * stats.sem(one_sample):.2f} of the truth (2 SE)")

# %% [markdown]
# ## 4. Standard error of a proportion
#
# For a 0/1 variable (churned or not) the mean is the proportion `p`, and the SD is `sqrt(p(1-p))`, so `SE = sqrt(p(1-p)/n)`.
# Check on churn with samples of 200 customers:

# %%
p = churn["churned"].mean()
n = 200
props = sample_means(churn["churned"].to_numpy().astype(float), n)
print(f"true p = {p:.3f}; std of 5000 sample proportions = {props.std():.4f}; formula sqrt(p(1-p)/n) = {np.sqrt(p * (1 - p) / n):.4f}")
print(f"a sample of 200 customers typically lands within {2 * np.sqrt(p * (1 - p) / n):.3f} of the true churn rate (2 SE)")

# %% [markdown]
# The SE of a proportion is largest at `p = 0.5` and shrinks near 0 or 1. Note that it depends on `n`, not on the size of the population: a sample of 1,000 is about as precise whether the population has 10,000 or 10 million members
# (as long as the sample is a small part of it).

# %% [markdown]
# ## 5. Two groups: the SE of a difference
#
# A/B tests compare two sample means. The two samples are independent, so their variances add: `SE(diff) = sqrt(SE1^2 + SE2^2)`. Conversion rate in the A/B data:

# %%
g = ab.groupby("group")["converted"].agg(["mean", "count"])
p1, n1 = g.loc["control"]; p2, n2 = g.loc["treatment"]
se1, se2 = np.sqrt(p1 * (1 - p1) / n1), np.sqrt(p2 * (1 - p2) / n2)
se_diff = np.sqrt(se1**2 + se2**2)
print(f"control {p1:.4f} (SE {se1:.4f}), treatment {p2:.4f} (SE {se2:.4f})")
print(f"difference = {p2 - p1:+.4f}, SE of difference = {se_diff:.4f}, difference / SE = {(p2 - p1) / se_diff:.2f}")

# %% [markdown]
# The observed lift is smaller than one standard error of the difference: the noise in the two conversion rates is as large as the difference we see. This "difference divided by its SE" ratio is exactly the **z statistic**
# that hypothesis tests use; we return to it in the testing notebooks.

# %% [markdown]
# ## 6. When the CLT is slow, fails, or cannot help
#
# **(a) Heavy skew needs a larger `n`.** How large before the sample means of `revenue` look symmetric? Measure skewness of the sampling distribution:

# %%
for n in [5, 20, 100, 500, 2000]:
    print(f"n = {n:5d}  skewness of sample means = {stats.skew(sample_means(revenue, n)):.2f}")

# %% [markdown]
# Rules of thumb like "n >= 30 is enough" are only guidelines; for strongly skewed data (rare, big purchases) you need hundreds. You can always *measure* it with a simulation like this.
#
# **(b) Heavy tails can break it.** The CLT needs a finite variance. The Cauchy distribution has none: the running mean never settles.

# %%
cauchy = stats.cauchy.rvs(size=100_000, random_state=42)
normal = stats.norm.rvs(size=100_000, random_state=42)
running = lambda a: np.cumsum(a) / np.arange(1, len(a) + 1)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(running(normal), label="normal: settles near 0"); ax.plot(running(cauchy), label="Cauchy: keeps jumping")
ax.set_xscale("log"); ax.set_ylim(-3, 3); ax.set_xlabel("sample size (log scale)"); ax.set_ylabel("running mean"); ax.set_title("No finite variance, no CLT")
ax.legend(); plt.show()

# %% [markdown]
# **(c) A biased sample stays biased.** The SE shrinks with `n`, but it only measures *random* error. Suppose we estimate the churn rate by "surveying" only customers with short tenure (say, those who recently called us). More of them will not fix the answer:

# %%
true_rate = churn["churned"].mean()
biased_pool = churn.loc[churn["tenure_months"] < 12, "churned"].to_numpy().astype(float)
for n in [50, 500, 1500]:
    m = sample_means(biased_pool, n, reps=2000)
    print(f"n = {n:5d}: mean of estimates {m.mean():.3f}, spread (SE) {m.std():.3f}   | true churn rate {true_rate:.3f}")

# %% [markdown]
# The estimates become more *precise* (SE falls) but stay wrong by the same amount. Precision is not accuracy. Random sampling, not sample size, is what protects you from bias.

# %% [markdown]
# ## Common mistakes
#
# 1. **Confusing SD and SE.** "Mean = 4.29, SD = 14.4" describes users; "mean = 4.29, SE = 0.16" describes how well you know the mean.
# 2. **Believing the data becomes normal with more rows.** Only the *distribution of the mean* does; the data keep their shape.
# 3. **Applying "n >= 30" blindly** to skewed or heavy-tailed data.
# 4. **Using SE for individuals.** A 95% range for the mean is not a range where 95% of individual values fall.
# 5. **Expecting more data to fix bias** (self-selected surveys, survivorship).

# %%
print(f"revenue per user: SD = {revenue.std():.2f} (spread of individuals), SE of the mean = {stats.sem(revenue):.3f} (precision of the average)")

# %% [markdown]
# ## Interview corner
#
# **1. State the central limit theorem and say why it matters.**
# <details><summary>Answer</summary>
# For independent, identically distributed data with finite variance, the distribution of the sample mean approaches a normal with mean mu and standard deviation sigma/sqrt(n) as n grows, regardless of the data's own shape.
# It is why z- and t-based confidence intervals and tests work for means of non-normal data, given enough observations.
# </details>
#
# **2. Standard deviation vs standard error?**
# <details><summary>Answer</summary>
# SD measures the spread of individual observations. SE is the SD of a statistic's sampling distribution, e.g. s/sqrt(n) for the mean. SE shrinks with more data; SD does not.
# </details>
#
# **3. You want to halve the margin of error of a survey. What happens to the required sample size?**
# <details><summary>Answer</summary>
# It quadruples, since SE is proportional to 1/sqrt(n).
# </details>
#
# **4. When does the CLT fail or work poorly?**
# <details><summary>Answer</summary>
# Infinite variance (very heavy tails), strong dependence between observations (time series, clustered users), very small n with strong skew, and for statistics like the maximum that are not averages.
# Bias from non-random sampling is a separate problem that no CLT fixes.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Standard errors from formulas
# Compute `se_charge`: the standard error of the mean of `churn["monthly_charges"]` (use the sample SD with `ddof=1`), and `se_churn`: the standard error of the churn proportion `sqrt(p(1-p)/n)`.
# Then compute `n_for_half`: the sample size needed to make `se_charge` **half as large** (round up to an int).

# %% tags=["starter"]
se_charge = ...
se_churn = ...
n_for_half = ...

# %%
mc = churn["monthly_charges"]
assert abs(se_charge - mc.std(ddof=1) / np.sqrt(len(mc))) < 1e-9, "SE = sample std (ddof=1) / sqrt(n)"
pp = churn["churned"].mean()
assert abs(se_churn - np.sqrt(pp * (1 - pp) / len(churn))) < 1e-9, "SE of a proportion = sqrt(p(1-p)/n)"
assert n_for_half == int(np.ceil(4 * len(mc))), "halving the SE needs 4x the sample size"
print(f"Exercise 1 passed: SE mean = {se_charge:.4f}, SE proportion = {se_churn:.4f}, n for half = {n_for_half}")

# %% tags=["solution"]
se_charge = churn["monthly_charges"].std(ddof=1) / np.sqrt(len(churn))
_p = churn["churned"].mean()
se_churn = float(np.sqrt(_p * (1 - _p) / len(churn)))
n_for_half = int(np.ceil(4 * len(churn)))

# %% [markdown]
# ### Exercise 2: Verify the SE by simulation on housing prices
# Take clean prices `pr = housing.price between 10,000 and 1,000,000` (in dollars). Draw 4,000 samples of size 100 with replacement (use `np.random.default_rng(0)`), compute each mean, and store the standard deviation
# of those 4,000 means in `emp_se`. Store the formula value `pr.std() / sqrt(100)` (population SD, `ddof=0`) in `formula_se`. The two should be within 5%.

# %%
pr = housing.loc[(housing["price"] > 10_000) & (housing["price"] < 1_000_000), "price"].to_numpy()

# %% tags=["starter"]
emp_se = ...
formula_se = ...

# %%
assert abs(formula_se - pr.std() / 10) < 1e-6, "formula_se = pr.std() / sqrt(100)"
assert abs(emp_se / formula_se - 1) < 0.05, f"empirical {emp_se:.0f} vs formula {formula_se:.0f}: use 4000 sample means of size 100"
print(f"Exercise 2 passed: empirical {emp_se:,.0f} vs formula {formula_se:,.0f}")

# %% tags=["solution"]
_r = np.random.default_rng(0)
_means = pr[_r.integers(0, len(pr), size=(4000, 100))].mean(axis=1)
emp_se = float(_means.std())
formula_se = float(pr.std() / np.sqrt(100))

# %% [markdown]
# ### Exercise 3: How many customers to estimate churn within +/- 2 points?
# A margin of error of about 2 SE should be at most 0.02 (two percentage points). Using the observed churn rate `p_hat = churn["churned"].mean()`, find the smallest integer `n_needed` with `2 * sqrt(p_hat*(1-p_hat)/n) <= 0.02`.
# Also store `n_worst_case`, the same computation using the worst-case `p = 0.5`.
# *Hint: solve the inequality for n: `n >= 4 * p*(1-p) / 0.02**2`; `np.ceil`.*

# %%
p_hat = churn["churned"].mean()

# %% tags=["starter"]
n_needed = ...
n_worst_case = ...

# %%
def _ok(n, p): return 2 * np.sqrt(p * (1 - p) / n) <= 0.02
assert _ok(n_needed, p_hat) and not _ok(n_needed - 1, p_hat), "n_needed must be the smallest n meeting the margin"
assert _ok(n_worst_case, 0.5) and not _ok(n_worst_case - 1, 0.5), "n_worst_case must be the smallest n for p = 0.5"
assert n_worst_case >= n_needed, "p = 0.5 is the worst case, so it needs the most data"
print(f"Exercise 3 passed: n = {n_needed} (observed p), {n_worst_case} (worst case)")

# %% tags=["solution"]
n_needed = int(np.ceil(4 * p_hat * (1 - p_hat) / 0.02**2))
n_worst_case = int(np.ceil(4 * 0.5 * 0.5 / 0.02**2))

# %% [markdown]
# ### Exercise 4 (open-ended): Bias versus noise
# Reuse `sample_means` with the `churned` values of customers on month-to-month contracts only (clean the casing first with `.str.lower()`). Compare the mean and SE of the estimates for n = 100 and n = 2000 against the
# true all-customer churn rate. In a markdown cell write two sentences on which error shrank and which did not, and give one real-world example of this trap (for instance, surveying only people who answer emails).

# %% [markdown]
# ## Recap
# - A statistic (sample mean) is random; its spread across hypothetical repeated samples is the standard error.
# - CLT: means of many independent draws are approximately normal with SD `sigma/sqrt(n)`, whatever the data's shape, provided the variance is finite and `n` is large enough.
# - `SE(mean) = s/sqrt(n)`, `SE(proportion) = sqrt(p(1-p)/n)`, `SE(difference) = sqrt(SE1^2 + SE2^2)`; 4x the data halves the SE.
# - Skewed data needs bigger `n`, heavy tails can break the CLT, and no amount of data repairs a biased sampling method.
#
# **Go deeper:** the SciPy docs for `scipy.stats.sem` and `scipy.stats.bootstrap` (used in the next notebook); the OpenIntro Statistics textbook chapter on foundations for inference (free online).
#
# **Next:** [Confidence intervals and the bootstrap](05_confidence_intervals_and_bootstrap.ipynb)

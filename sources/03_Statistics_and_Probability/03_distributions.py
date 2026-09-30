# %% [markdown]
# # Probability Distributions: Normal, Binomial, Poisson, Exponential, Lognormal
#
# **Why this matters.** A distribution is a model of how a variable behaves: how many customers churn out of 50, how many orders arrive per day, how long until the next failure,
# how prices are spread. Choose the right family and you get probabilities, thresholds and simulations for free ("what is the chance we get 10+ orders in an hour?"). Choose wrongly and every p-value and
# confidence interval built on it is off. This notebook shows how to *fit* a distribution to data and how to check whether it actually fits.
#
# **You will learn**
# - use `scipy.stats` (`pmf`/`pdf`, `cdf`, `ppf`, `rvs`, `fit`) for any distribution
# - recognise the story behind each family: normal (sums), binomial (successes in n trials), Poisson (counts), exponential (waiting times), lognormal (products)
# - fit distributions to housing prices, sensor readings and event counts
# - check a fit with histograms and **QQ plots**, and know why a p-value from a normality test is not enough
#
# **Prerequisites:** [Probability and simulation](02_probability_and_simulation.ipynb).
#
# > Data used: `housing.csv`, `churn.csv` and `sensors.csv` (synthetic, see `docs/DATASETS.md`).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DATA = Path("../datasets")
rng = np.random.default_rng(42)
housing = pd.read_csv(DATA / "housing.csv")
churn = pd.read_csv(DATA / "churn.csv")
sensors = pd.read_csv(DATA / "sensors.csv", parse_dates=["timestamp"])

# %% [markdown]
# ## 0. The scipy.stats toolbox
#
# Every distribution object has the same methods, which is why one API covers dozens of families:
#
# | Method | Answers | Example |
# |---|---|---|
# | `pmf(k)` / `pdf(x)` | probability of exactly `k` (discrete) / density at `x` (continuous) | `binom.pmf(3, 10, 0.3)` |
# | `cdf(x)` | P(X <= x) | `norm.cdf(1.96)` |
# | `sf(x)` | P(X > x) = 1 - cdf | `norm.sf(1.96)` |
# | `ppf(q)` | the value below which a fraction `q` lies (inverse cdf) | `norm.ppf(0.975)` |
# | `rvs(size, random_state)` | random draws | `norm.rvs(size=5, random_state=42)` |
# | `.fit(data)` | maximum-likelihood parameters | `norm.fit(x)` |
#
# For a continuous variable the **density** `pdf` is not a probability (it can exceed 1); probability is the *area* under it: `P(a < X < b) = cdf(b) - cdf(a)`.

# %%
print("P(Z <= 1.96)       =", round(stats.norm.cdf(1.96), 4))
print("z with 97.5% below =", round(stats.norm.ppf(0.975), 4))
print("P(exactly 3 of 10, p=0.3) =", round(stats.binom.pmf(3, 10, 0.3), 4))

# %% [markdown]
# ## 1. Normal: what you get when many small effects add up
#
# Bell-shaped, defined by a mean `mu` and standard deviation `sigma`. It arises from sums and averages of many independent contributions (the central limit theorem, next-but-one notebook).
# The **68-95-99.7 rule** says about that share of values lies within 1, 2 and 3 standard deviations of the mean. Sensor power draw looks like a natural candidate, so fit a normal to it and test how well it actually fits.

# %%
power = sensors["power_kw"]
mu, sigma = stats.norm.fit(power)
print(f"fitted normal: mu = {mu:.1f} kW, sigma = {sigma:.1f} kW")
for k in (1, 2, 3):
    inside = ((power - mu).abs() <= k * sigma).mean()
    print(f"within {k} sigma: observed {inside:.3f}  vs normal {stats.norm.cdf(k) - stats.norm.cdf(-k):.3f}")

# %%
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.hist(power, bins=60, density=True, color="lightsteelblue", label="observed")
xs = np.linspace(power.min(), power.max(), 300)
ax.plot(xs, stats.norm.pdf(xs, mu, sigma), color="tab:red", label="fitted normal")
ax.set_xlabel("power (kW)"); ax.set_ylabel("density"); ax.set_title("Hourly power draw vs fitted normal"); ax.legend()
plt.show()

# %% [markdown]
# A histogram overlay is a first look, but our eyes are poor judges of tails. The **QQ plot** (quantile-quantile) is sharper: it plots the sorted data against the quantiles a perfect normal would give.
# Points on the straight line = a good fit; a bend = skew; an S-shape = heavy or light tails.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
stats.probplot(power, dist="norm", plot=axes[0])
axes[0].set_title("QQ plot: power_kw vs normal"); axes[0].set_xlabel("theoretical quantiles (z)"); axes[0].set_ylabel("observed power (kW)")
stats.probplot(housing["price"].dropna(), dist="norm", plot=axes[1])
axes[1].set_title("QQ plot: house price vs normal"); axes[1].set_xlabel("theoretical quantiles (z)"); axes[1].set_ylabel("price ($)")
plt.tight_layout(); plt.show()

# %% [markdown]
# *What just happened?* Power draw is **not** normal, even though its mean and std are fine summaries. The QQ plot is S-shaped, which means lighter tails than a normal, and the histogram above shows two humps, so this is a mixture (think of two operating regimes) rather than one bell. The
# 68-95-99.7 table already hinted at it: 62% within 1 sigma instead of 68%, 99% within 2 sigma instead of 95%. A few isolated points at the extremes are the planted sensor anomalies.
# House prices curve sharply upward at the right: a much heavier right tail than a normal, i.e. skewed.
# Do not lean on a formal normality test alone: with thousands of rows, tests such as Shapiro-Wilk reject for deviations too small to matter, and with tiny samples they cannot detect real problems. Look at the plot.

# %% [markdown]
# ## 2. Binomial: number of successes in n independent trials
#
# `Binomial(n, p)`: each of `n` independent trials succeeds with probability `p`; count the successes. Mean `n*p`, variance `n*p*(1-p)`.
# Story: among 50 randomly chosen customers, how many churn? Use the data's churn rate as `p`, then compare the theory with repeated random samples from the actual table.

# %%
p_churn = churn["churned"].mean()
n = 50
counts = np.array([churn["churned"].sample(n, random_state=int(s)).sum() for s in rng.integers(0, 1_000_000, 4000)])

ks = np.arange(0, 41)
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.hist(counts, bins=np.arange(-0.5, 41.5), density=True, alpha=0.6, label="4000 random samples of 50 customers")
ax.plot(ks, stats.binom.pmf(ks, n, p_churn), "o-", ms=3, color="tab:red", label=f"Binomial(50, {p_churn:.3f})")
ax.set_xlabel("churners among 50 customers"); ax.set_ylabel("probability"); ax.set_title("Binomial predicts the sampling behaviour"); ax.legend()
plt.show()
print(f"mean of counts {counts.mean():.2f} vs n*p = {n * p_churn:.2f};  var {counts.var():.2f} vs n*p*(1-p) = {n * p_churn * (1 - p_churn):.2f}")
print(f"P(20 or more churners in 50) = {stats.binom.sf(19, n, p_churn):.3f}  (sf(19) = P(X > 19) = P(X >= 20))")

# %% [markdown]
# Binomial assumptions: **fixed `n`, same `p` for every trial, independent trials**. If customers in the same household churn together, the counts vary *more* than the binomial predicts (overdispersion).

# %% [markdown]
# ## 3. Poisson: counts of rare events in a fixed window
#
# `Poisson(lambda)`: the number of events in a fixed interval when events occur independently at a constant average rate `lambda`. **Mean = variance = lambda**, a fingerprint you can check.
# The sensors file flags anomalous hours. How many anomalies per day?

# %%
daily = sensors.set_index("timestamp")["is_anomaly"].resample("D").sum()
lam = daily.mean()
print(f"{len(daily)} days, mean = {lam:.3f}, variance = {daily.var():.3f}  (Poisson: these should be similar)")

obs = daily.value_counts().reindex(range(0, 5), fill_value=0)
exp = stats.poisson.pmf(range(0, 5), lam) * len(daily)
pd.DataFrame({"observed days": obs, "expected days (Poisson)": exp.round(1)})

# %% [markdown]
# The match is good: anomalies behave like independent random events at about 0.14 per day. Now a **counter-example**: total orders per day in the shop data. It is a count, but is it Poisson?

# %%
orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
per_day = orders.groupby("order_date").size()
print(f"orders per day: mean {per_day.mean():.1f}, variance {per_day.var():.1f}, variance/mean = {per_day.var() / per_day.mean():.1f}")
print("by month, mean orders/day:", orders.groupby(orders["order_date"].dt.month).size().div(orders.groupby(orders["order_date"].dt.month)["order_date"].nunique()).round(1).to_dict())

# %% [markdown]
# Variance is several times the mean: **overdispersion**. The rate is not constant (orders peak in Nov/Dec, see `DATASETS.md`), and a mixture of different rates is more variable than a single Poisson.
# Lesson: check `variance/mean` before assuming Poisson; if it is much above 1, model the rate (seasonality) or use a negative binomial.

# %% [markdown]
# ## 4. Exponential: waiting time between Poisson events
#
# If events arrive as a Poisson process, the *gaps* between them follow an **exponential** distribution with mean `1/lambda`. Its signature is **memorylessness**: having waited 100 hours tells you nothing about how much longer you must wait.
# Its standard deviation equals its mean (another checkable fingerprint). Gaps between anomalous hours:

# %%
idx = np.flatnonzero(sensors["is_anomaly"].to_numpy())
gaps = np.diff(idx)                                   # hours between consecutive anomalies
print(f"{len(gaps)} gaps: mean {gaps.mean():.0f} h, std {gaps.std(ddof=1):.0f} h, median {np.median(gaps):.0f} h")
print(f"exponential theory: median = mean * ln 2 = {gaps.mean() * np.log(2):.0f} h")

fig, ax = plt.subplots(figsize=(6.5, 3.8))
t = np.sort(gaps)
ax.step(t, 1 - np.arange(1, len(t) + 1) / len(t), where="post", label="observed P(gap > t)")
ax.plot(t, stats.expon.sf(t, scale=gaps.mean()), color="tab:red", label="exponential")
ax.set_yscale("log"); ax.set_xlabel("hours since previous anomaly"); ax.set_ylabel("P(gap > t), log scale")
ax.set_title("Survival curve: a straight line on a log axis means exponential"); ax.legend()
plt.show()

# %% [markdown]
# With only 24 gaps we cannot be sure of much, but mean, std and median are consistent with an exponential, matching the Poisson story from the counts. This is what "anomalies occur at random" looks like.

# %% [markdown]
# ## 5. Lognormal: what you get when many effects multiply
#
# If `log(X)` is normal, `X` is **lognormal**: positive, right-skewed. Multiplicative growth (a house is 10% bigger, 5% better located, ...) produces it. Prices, incomes, file sizes and session lengths are common examples.
# Fit house prices (after removing the impossible ones) and compare normal vs lognormal by QQ plot:

# %%
p = housing["price"]
p = p[(p > 10_000) & (p < 1_000_000)].to_numpy()
shape, loc, scale = stats.lognorm.fit(p, floc=0)               # fix location at 0 so that the two parameters are the log-mean and log-std
print(f"lognormal fit: sigma (of log price) = {shape:.3f}, median = scale = ${scale:,.0f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
stats.probplot(p, dist="norm", plot=axes[0]); axes[0].set_title("Price vs normal: curved (poor fit)")
axes[0].set_xlabel("theoretical quantiles (z)"); axes[0].set_ylabel("price ($)")
stats.probplot(p, dist=stats.lognorm(shape, loc=0, scale=scale), plot=axes[1]); axes[1].set_title("Price vs fitted lognormal: closer to a line")
axes[1].set_xlabel("theoretical quantiles"); axes[1].set_ylabel("price ($)")
plt.tight_layout(); plt.show()

# %% [markdown]
# The lognormal is far better than the normal but not perfect: the plot bends below the line at the top, meaning the fitted lognormal predicts more very expensive houses than the data contains (our 1M cap plays a part).
# Every model is an approximation; the question is whether it is good enough for the decision at hand.

# %% [markdown]
# ## 6. Which distribution? A cheat sheet
#
# | You are modelling | Try | Sanity check |
# |---|---|---|
# | successes out of `n` fixed trials | binomial | variance = `n p (1-p)` |
# | events per time window | Poisson | variance = mean |
# | waiting time between random events | exponential | std = mean, memoryless |
# | sums/averages of many things; measurement noise | normal | QQ plot straight |
# | positive, right-skewed, multiplicative | lognormal | `log(x)` looks normal |
# | proportions between 0 and 1 | beta | (used in the Bayesian notebook) |

# %% [markdown]
# ## Common mistakes
#
# 1. **Reading a `pdf` value as a probability.** Densities can exceed 1; take areas with `cdf` differences.
# 2. **`binom.cdf(k)` vs `sf(k)` off-by-one**: `cdf(k) = P(X <= k)`; "at least k" is `sf(k - 1)`, not `sf(k)`.
# 3. **Forcing a normal fit on skewed data** and then quoting "the 99th percentile" from it. Check the QQ plot.
# 4. **Assuming Poisson without checking variance/mean** (overdispersion is the rule in real count data).
# 5. **Fitting `lognorm` without `floc=0`**, which adds a third free parameter and often gives odd fits.

# %%
print("P(X >= 3), X~Bin(10, 0.3): correct sf(2) =", round(stats.binom.sf(2, 10, 0.3), 4), "| off-by-one sf(3) =", round(stats.binom.sf(3, 10, 0.3), 4))
print("pdf can exceed 1:", stats.norm.pdf(0, 0, 0.1).round(2), "(normal with sigma = 0.1 at its peak)")

# %% [markdown]
# ## Interview corner
#
# **1. Which distribution would you use for the number of support tickets per hour, and how would you check it?**
# <details><summary>Answer</summary>
# Start with Poisson; check that variance is close to the mean and that the rate is stable across hours. Real ticket data usually has a time-of-day cycle and bursts (overdispersion), so you may need a rate that varies with time or a negative binomial.
# </details>
#
# **2. What is the difference between a PMF, a PDF and a CDF?**
# <details><summary>Answer</summary>
# PMF gives probabilities for discrete values; PDF gives a density for continuous ones (probability is area under it); CDF gives P(X &lt;= x) for either and is the running total of the PMF or the integral of the PDF.
# </details>
#
# **3. What does a QQ plot show and how do you read it?**
# <details><summary>Answer</summary>
# It plots sample quantiles against theoretical quantiles. A straight line means the shape matches; a systematic curve means skew; an S-shape means heavier or lighter tails than the reference; isolated points off the line at the ends are outliers.
# </details>
#
# **4. Why would prices be modelled as lognormal instead of normal?**
# <details><summary>Answer</summary>
# Prices are positive and right-skewed, and factors act multiplicatively (percent changes), so log(price) is often approximately normal. A normal model would assign probability to negative prices and understate the upper tail.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Normal probabilities for power draw
# Using the fitted `mu` and `sigma`, compute `p_normal`: the model probability that an hour's power lies **between 180 and 240 kW**, and `p_empirical`: the observed fraction in that range.
# Store `gap` as their absolute difference. *Hint: `stats.norm.cdf(b, mu, sigma) - stats.norm.cdf(a, mu, sigma)`.*

# %% tags=["starter"]
p_normal = ...
p_empirical = ...
gap = ...

# %%
exp_n = stats.norm.cdf(240, mu, sigma) - stats.norm.cdf(180, mu, sigma)
exp_e = ((power >= 180) & (power <= 240)).mean()
assert abs(p_normal - exp_n) < 1e-9, f"expected {exp_n:.4f} from the normal cdf"
assert abs(p_empirical - exp_e) < 1e-9, f"expected {exp_e:.4f} observed"
assert abs(gap - abs(exp_n - exp_e)) < 1e-9
print(f"Exercise 1 passed: model {p_normal:.3f}, observed {p_empirical:.3f}, gap {gap:.4f}")

# %% tags=["solution"]
p_normal = float(stats.norm.cdf(240, mu, sigma) - stats.norm.cdf(180, mu, sigma))
p_empirical = float(((power >= 180) & (power <= 240)).mean())
gap = abs(p_normal - p_empirical)

# %% [markdown]
# ### Exercise 2: A binomial decision
# A retention team can only handle a batch if **at least 22 of 50** randomly selected customers churn (imagine an alert threshold). Compute `p_alert` = the binomial probability of that
# (at least 22), using the overall churn rate `p_churn`. Then estimate the same probability by simulation with `rng2 = np.random.default_rng(1)` and 200,000 batches (`p_sim`), and store
# whether they agree within 0.005 in `agree`.

# %% tags=["starter"]
p_alert = ...
p_sim = ...
agree = ...

# %%
assert abs(p_alert - (1 - stats.binom.cdf(21, 50, p_churn))) < 1e-9, "at least 22 means P(X >= 22) = 1 - cdf(21) = sf(21)"
assert abs(p_sim - p_alert) < 0.01, f"simulation {p_sim:.4f} should be near the exact {p_alert:.4f}"
assert bool(agree) == (abs(p_sim - p_alert) < 0.005)
print(f"Exercise 2 passed: exact {p_alert:.4f}, simulated {p_sim:.4f}")

# %% tags=["solution"]
p_alert = float(stats.binom.sf(21, 50, p_churn))
rng2 = np.random.default_rng(1)
p_sim = float((rng2.binomial(50, p_churn, size=200_000) >= 22).mean())
agree = bool(abs(p_sim - p_alert) < 0.005)

# %% [markdown]
# ### Exercise 3: Lognormal percentiles
# Using the fitted `shape` and `scale` from section 5, compute `p90_model`, the 90th percentile of house price predicted by the fitted lognormal (`stats.lognorm.ppf`), and `p90_data`, the
# empirical 90th percentile of `p` (`np.percentile`). Store `rel_err = abs(p90_model - p90_data) / p90_data`. Then set `good_enough` to whether `rel_err < 0.10`.

# %% tags=["starter"]
p90_model = ...
p90_data = ...
rel_err = ...
good_enough = ...

# %%
e_model = stats.lognorm.ppf(0.9, shape, loc=0, scale=scale)
e_data = np.percentile(p, 90)
assert abs(p90_model - e_model) < 1e-6 * e_model, "use lognorm.ppf(0.9, shape, loc=0, scale=scale)"
assert abs(p90_data - e_data) < 1e-6 * e_data, "np.percentile(p, 90)"
assert abs(rel_err - abs(e_model - e_data) / e_data) < 1e-9
assert bool(good_enough) == (rel_err < 0.10)
print(f"Exercise 3 passed: model {p90_model:,.0f} vs data {p90_data:,.0f} (error {rel_err:.1%})")

# %% tags=["solution"]
p90_model = float(stats.lognorm.ppf(0.9, shape, loc=0, scale=scale))
p90_data = float(np.percentile(p, 90))
rel_err = abs(p90_model - p90_data) / p90_data
good_enough = bool(rel_err < 0.10)

# %% [markdown]
# ### Exercise 4 (open-ended): Does temperature look normal?
# Take `sensors["temperature_c"]` (60 values are missing), plot its histogram and a QQ plot against a normal, and write two sentences: is a single normal a good description? The temperature has a daily cycle.
# Compare the histogram of the full column with the histogram of just the readings between 12:00 and 15:00, and explain what you see.

# %% [markdown]
# ## Recap
# - `scipy.stats` gives every family the same methods: `pmf/pdf`, `cdf`, `sf`, `ppf`, `rvs`, `fit`.
# - Normal for sums/noise, binomial for successes in `n` trials, Poisson for counts (mean = variance), exponential for waiting times, lognormal for positive skewed multiplicative data.
# - Check the fingerprint (variance vs mean, std vs mean) and the QQ plot; real data is often over-dispersed or a mixture.
# - Densities are not probabilities; "at least k" for a discrete variable is `sf(k-1)`.
#
# **Go deeper:** the SciPy docs page "Statistical functions (scipy.stats)" and its tutorial on probability distributions; statsmodels has `qqplot` as well.
#
# **Next:** [Sampling, the CLT and standard error](04_sampling_clt_and_standard_error.ipynb)

# %% [markdown]
# # Bayesian Thinking: An Introduction with the Beta-Binomial Model
#
# **Why this matters.** Stakeholders rarely ask "can we reject the null?" They ask "how likely is it that B beats A, and by how much?" Bayesian inference answers that directly, combines
# prior knowledge with data, and updates naturally as data arrives. The conversion-rate model needs only NumPy and SciPy, so you can see every moving part.
#
# **You will learn**
# - combine a **prior** and the data's **likelihood** into a **posterior** (Beta prior + binomial data = Beta posterior)
# - compute a credible interval and `P(treatment > control)` by sampling
# - contrast a credible interval with a frequentist confidence interval
# - test **prior sensitivity**: when the prior matters and when the data overwhelm it
#
# **Prerequisites:** [Confidence intervals and the bootstrap](05_confidence_intervals_and_bootstrap.ipynb) and [A/B testing](07_ab_testing_end_to_end.ipynb).
#
# > Data used: `datasets/ab_test.csv` (synthetic, see `docs/DATASETS.md`).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

DATA = Path("../datasets")
rng = np.random.default_rng(42)
ab = pd.read_csv(DATA / "ab_test.csv")
g = ab.groupby("group")["converted"].agg(["sum", "count"])
x_c, n_c = int(g.loc["control", "sum"]), int(g.loc["control", "count"])
x_t, n_t = int(g.loc["treatment", "sum"]), int(g.loc["treatment", "count"])
print(f"control {x_c}/{n_c}, treatment {x_t}/{n_t}")

# %% [markdown]
# ## 1. Bayes' rule for a parameter
#
# `posterior ~ prior x likelihood`. The unknown is the conversion rate `p` (between 0 and 1). The **prior** says which values were plausible *before* the data. The **likelihood** says how probable the observed
# data are for each `p`. Multiply and renormalise to get the **posterior**: your updated belief. A grid makes this concrete.

# %%
p_grid = np.linspace(0.001, 0.999, 999)
prior = stats.beta.pdf(p_grid, 1, 1)                          # flat prior: every rate equally plausible
likelihood = stats.binom.pmf(x_c, n_c, p_grid)
posterior = prior * likelihood
posterior /= np.trapezoid(posterior, p_grid)                   # normalise to a density
print(f"posterior mean from the grid = {np.trapezoid(p_grid * posterior, p_grid):.4f}   (sample rate {x_c / n_c:.4f})")

# %% [markdown]
# ## 2. The conjugate shortcut: Beta prior, binomial data
#
# A `Beta(a, b)` distribution lives on [0, 1], so it is a natural prior for a rate; `a` and `b` act like pseudo-counts of past successes and failures. With `x` successes in `n` trials the posterior is exactly
# **`Beta(a + x, b + n - x)`**: no integration needed. Check it against the grid:

# %%
a0, b0 = 1, 1
post = stats.beta(a0 + x_c, b0 + n_c - x_c)
print(f"Beta posterior mean = {post.mean():.4f}, sd = {post.std():.4f}")
print("max difference from the grid posterior:", np.abs(post.pdf(p_grid) - posterior).max().round(6))

# %% [markdown]
# ## 3. Posterior for both arms, credible interval, and P(B > A)
# A **95% credible interval** is the central range that holds 95% of the posterior: given the model and prior, there *is* a 95% probability that the rate lies inside. That is the sentence people wrongly say about confidence intervals.

# %%
post_c = stats.beta(1 + x_c, 1 + n_c - x_c)
post_t = stats.beta(1 + x_t, 1 + n_t - x_t)
fig, ax = plt.subplots(figsize=(7, 3.8))
xs = np.linspace(0.07, 0.14, 400)
ax.plot(xs, post_c.pdf(xs), label="control"); ax.plot(xs, post_t.pdf(xs), label="treatment")
ax.set_xlabel("conversion rate"); ax.set_ylabel("posterior density"); ax.set_title("Posterior beliefs about each arm's conversion rate"); ax.legend()
plt.show()

for name, pst in [("control", post_c), ("treatment", post_t)]:
    lo, hi = pst.ppf([0.025, 0.975])
    print(f"{name:9s} mean {pst.mean():.4f}, 95% credible interval [{lo:.4f}, {hi:.4f}]")

# %% [markdown]
# Now the question the business asks. Draw samples from both posteriors and compare them draw by draw:

# %%
s_c = post_c.rvs(200_000, random_state=1); s_t = post_t.rvs(200_000, random_state=2)
diff = s_t - s_c
print(f"P(treatment > control) = {(diff > 0).mean():.3f}")
print(f"posterior mean lift = {diff.mean():+.4f}, 95% credible interval [{np.percentile(diff, 2.5):+.4f}, {np.percentile(diff, 97.5):+.4f}]")
print(f"P(lift > 1 percentage point) = {(diff > 0.01).mean():.3f}")
print(f"expected loss if we ship treatment: {np.mean(np.maximum(s_c - s_t, 0)):.5f}  (average conversion given up in the draws where control is better)")

# %% [markdown]
# *What just happened?* Instead of a yes/no verdict we got a probability that treatment is better, the plausible size of the lift, and the **expected loss** of each decision. The lesson matches the frequentist analysis: the result is
# inconclusive, now stated as "treatment is probably (not certainly) ahead, and a lift of a full percentage point is unlikely". Note the number `P(treatment > control)` is *not* a p-value and cannot be compared with 0.05.

# %% [markdown]
# ## 4. Credible interval vs confidence interval
# Here the two intervals are numerically close because the data are plentiful and the prior is flat. Their *meanings* differ:

# %%
lo, hi = proportion_confint(x_c, n_c, method="wilson")
clo, chi = post_c.ppf([0.025, 0.975])
print(f"control Wilson 95% confidence interval : [{lo:.4f}, {hi:.4f}]  (long-run coverage of the method)")
print(f"control 95% credible interval (flat)    : [{clo:.4f}, {chi:.4f}]  (probability statement about p, given prior and data)")

# %% [markdown]
# They diverge when data are scarce or when a meaningful prior is used. With 3 conversions out of 20 visitors:

# %%
for a, b, name in [(1, 1, "flat Beta(1,1)"), (10, 90, "informative Beta(10,90)")]:
    pst = stats.beta(a + 3, b + 17)
    print(f"{name:24s} posterior mean {pst.mean():.3f}, 95% CrI [{pst.ppf(0.025):.3f}, {pst.ppf(0.975):.3f}]")
print("Wilson CI (ignores any prior):", np.round(proportion_confint(3, 20, method="wilson"), 3))

# %% [markdown]
# ## 5. Prior sensitivity: how much does the prior matter?
# Honest Bayesian work checks this. Consider three priors for the control conversion rate: flat; a **weakly informative** `Beta(10, 90)` (centred at 10%, worth about 100 past visitors); and a **strong but wrong** `Beta(300, 700)`
# (centred at 30%, worth 1,000 visitors). Posterior for the full data, then for just the first 100 users:

# %%
priors = {"flat Beta(1,1)": (1, 1), "weak Beta(10,90)": (10, 90), "strong wrong Beta(300,700)": (300, 700)}
ctrl = ab.loc[ab["group"] == "control", "converted"].to_numpy()
for label, n_use in [("all data", len(ctrl)), ("first 100 users", 100)]:
    print(f"--- {label} (sample rate {ctrl[:n_use].mean():.3f}) ---")
    for name, (a, b) in priors.items():
        pst = stats.beta(a + ctrl[:n_use].sum(), b + n_use - ctrl[:n_use].sum())
        print(f"  {name:28s} posterior mean {pst.mean():.3f}  95% CrI [{pst.ppf(0.025):.3f}, {pst.ppf(0.975):.3f}]")

# %% [markdown]
# With 3,959 users the flat and weak priors agree, while the stubborn wrong prior is still visibly biased (its 1,000 pseudo-observations weigh as much as a quarter of our data). With 100 users the strong prior completely dominates the posterior.
# Practical rule: a prior is a modelling assumption, like any other. Use weakly informative priors grounded in history, **report a sensitivity check**, and never choose a prior because it yields the answer you want.

# %% [markdown]
# ## 6. Sequential updating
# Yesterday's posterior is today's prior, so Bayesian updating handles data arriving in batches without recomputing. Watch the belief about control's rate sharpen:

# %%
a, b = 1, 1
fig, ax = plt.subplots(figsize=(7, 3.8))
xs = np.linspace(0, 0.3, 500)
for k, upto in enumerate([0, 50, 200, 1000, len(ctrl)]):
    lo_i = [0, 0, 50, 200, 1000][k]
    chunk = ctrl[lo_i:upto]
    a, b = a + chunk.sum(), b + len(chunk) - chunk.sum()
    ax.plot(xs, stats.beta.pdf(xs, a, b), label=f"after {upto} users")
ax.set_xlabel("conversion rate"); ax.set_ylabel("posterior density"); ax.set_title("Each batch narrows the posterior"); ax.legend()
plt.show()
print("sequential result equals batch result:", (a, b) == (1 + x_c, 1 + n_c - x_c))

# %% [markdown]
# Bayesian posteriors are often said to be immune to peeking. That is only partly true: the posterior itself stays coherent, but a *decision rule* such as "stop when `P(B > A) > 0.95`" still has frequentist error rates that
# grow with repeated looks. Choose stopping rules deliberately either way.

# %% [markdown]
# ## Common mistakes
#
# 1. **Treating `P(B > A)` as a p-value**, or assuming a threshold like 95% carries the same guarantees.
# 2. **Hiding an informative prior** or not testing sensitivity.
# 3. **Calling a confidence interval a probability statement**, or a credible interval a coverage guarantee: they answer different questions.
# 4. **A very strong prior with little data**: the answer is mostly your prior.
# 5. **Ignoring the model**: the Beta-binomial assumes independent users with one common rate per arm; segments or time effects break it.

# %%
print("strong wrong prior, 100 users: posterior mean", round(stats.beta(300 + ctrl[:100].sum(), 700 + 100 - ctrl[:100].sum()).mean(), 3), "vs sample rate", ctrl[:100].mean())

# %% [markdown]
# ## Interview corner
#
# **1. Credible interval vs confidence interval?**
# <details><summary>Answer</summary>
# A 95% credible interval contains the parameter with 95% posterior probability given the prior and data. A 95% confidence interval comes from a procedure that covers the true value in 95% of repeated samples; the parameter is fixed and any one interval either covers it or not.
# </details>
#
# **2. What is a conjugate prior?**
# <details><summary>Answer</summary>
# A prior whose posterior is in the same family given the likelihood. Beta with binomial data gives Beta(a + successes, b + failures), so updating is addition and no sampling is required.
# </details>
#
# **3. How would you present an A/B test to a non-statistician using a Bayesian approach?**
# <details><summary>Answer</summary>
# Give the probability that B beats A, the credible interval for the lift, and expected loss for each decision, plus a prior sensitivity note. Say what prior you used and why.
# </details>
#
# **4. When does the prior matter most?**
# <details><summary>Answer</summary>
# With little data or a strongly informative prior. With abundant data the likelihood dominates and reasonable priors give nearly the same posterior.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Posterior for the mobile conversion rate
# Using a flat `Beta(1,1)` prior, compute the posterior for the conversion rate of **mobile** users (both groups pooled). Store its mean in `mob_mean` and the 95% credible interval as a tuple `(low, high)` in `mob_ci`.

# %% tags=["starter"]
mob_mean = ...
mob_ci = ...

# %%
mob = ab.loc[ab["device"] == "mobile", "converted"]
pp = stats.beta(1 + mob.sum(), 1 + len(mob) - mob.sum())
assert abs(mob_mean - pp.mean()) < 1e-9, "posterior mean = (1 + x) / (2 + n)"
assert abs(mob_ci[0] - pp.ppf(0.025)) < 1e-9 and abs(mob_ci[1] - pp.ppf(0.975)) < 1e-9, "use the 2.5% and 97.5% posterior quantiles"
print(f"Exercise 1 passed: mean {mob_mean:.4f}, CrI [{mob_ci[0]:.4f}, {mob_ci[1]:.4f}]")

# %% tags=["solution"]
_m = ab.loc[ab["device"] == "mobile", "converted"]
_p = stats.beta(1 + _m.sum(), 1 + len(_m) - _m.sum())
mob_mean = float(_p.mean())
mob_ci = (float(_p.ppf(0.025)), float(_p.ppf(0.975)))

# %% [markdown]
# ### Exercise 2: Desktop vs mobile
# Draw 100,000 samples from the posterior of each device (flat priors, `random_state=10` for desktop and `11` for mobile) and store `p_desktop_better` = the fraction of draws where the desktop rate exceeds the mobile rate.
# Store `credible_diff` = the mean of (desktop - mobile) draws.

# %% tags=["starter"]
p_desktop_better = ...
credible_diff = ...

# %%
def _post(dev):
    d = ab.loc[ab["device"] == dev, "converted"]
    return stats.beta(1 + d.sum(), 1 + len(d) - d.sum())
sd = _post("desktop").rvs(100_000, random_state=10); sm_ = _post("mobile").rvs(100_000, random_state=11)
assert abs(p_desktop_better - (sd > sm_).mean()) < 1e-9, "fraction of draws with desktop > mobile"
assert abs(credible_diff - (sd - sm_).mean()) < 1e-9, "mean of the paired differences"
assert 0.9 < p_desktop_better <= 1.0, "with thousands of users per device the rates are well separated"
print(f"Exercise 2 passed: P(desktop > mobile) = {p_desktop_better:.4f}, mean difference {credible_diff:+.4f}")

# %% tags=["solution"]
def _pp(dev):
    d = ab.loc[ab["device"] == dev, "converted"]
    return stats.beta(1 + d.sum(), 1 + len(d) - d.sum())
_sd = _pp("desktop").rvs(100_000, random_state=10)
_sm = _pp("mobile").rvs(100_000, random_state=11)
p_desktop_better = float((_sd > _sm).mean())
credible_diff = float((_sd - _sm).mean())

# %% [markdown]
# ### Exercise 3: Prior sensitivity on a small sample
# Take the **first 40 users of the treatment group** (file order). Compute the posterior mean under a flat prior (`mean_flat`) and under `Beta(20, 180)` (`mean_informative`, centred at 10%).
# Store `prior_matters` = True if the two means differ by more than 0.01.

# %% tags=["starter"]
mean_flat = ...
mean_informative = ...
prior_matters = ...

# %%
t40 = ab.loc[ab["group"] == "treatment", "converted"].to_numpy()[:40]
ef = (1 + t40.sum()) / (2 + 40)
ei = (20 + t40.sum()) / (200 + 40)
assert abs(mean_flat - ef) < 1e-9 and abs(mean_informative - ei) < 1e-9, "posterior mean = (a + x) / (a + b + n)"
assert bool(prior_matters) == (abs(ef - ei) > 0.01)
print(f"Exercise 3 passed: flat {mean_flat:.4f}, informative {mean_informative:.4f}, sample rate {t40.mean():.4f}, matters: {prior_matters}")

# %% tags=["solution"]
_t40 = ab.loc[ab["group"] == "treatment", "converted"].to_numpy()[:40]
mean_flat = float((1 + _t40.sum()) / (2 + 40))
mean_informative = float((20 + _t40.sum()) / (200 + 40))
prior_matters = bool(abs(mean_flat - mean_informative) > 0.01)

# %% [markdown]
# ### Exercise 4 (open-ended): Would you ship?
# Using the posterior samples `diff` from section 3, assume shipping has a cost equivalent to 0.5 percentage points of conversion. Compute the probability that the lift exceeds that cost, and the expected net gain.
# Write a short markdown recommendation and state how your answer changes under the prior `Beta(10, 90)` for both arms.

# %% [markdown]
# ## Recap
# - Posterior is prior times likelihood; Beta prior with binomial data gives a Beta posterior by adding successes and failures.
# - Sampling from posteriors answers business questions directly: `P(B > A)`, credible interval of the lift, expected loss.
# - A credible interval is a probability statement about the parameter given a prior; a confidence interval is a statement about a procedure.
# - Always run a prior sensitivity check; with lots of data reasonable priors agree, with little data they dominate.
#
# **Go deeper:** "Think Bayes" by Allen Downey (free online) and "Bayesian Data Analysis" by Gelman et al.; `scipy.stats.beta` docs.
#
# **Next:** [Statistical fallacies](10_statistical_fallacies.ipynb)

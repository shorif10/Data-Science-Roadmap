# %% [markdown]
# # A/B Testing End to End
#
# **Why this matters.** Almost every product company decides on features by experiment, and "design, run and read an A/B test" is a standard data-science interview task. The statistics is not the hard part; the
# hard parts are planning the sample size *before* looking, checking the experiment was run correctly, and resisting the temptations that inflate false positives: peeking early and slicing the results twenty ways until something is "significant".
# This notebook walks the whole lifecycle with code.
#
# **You will learn**
# - design a test: metric, unit of randomisation, minimum detectable effect, and the sample size for the power you want (`statsmodels`)
# - run sanity checks (sample-ratio mismatch, covariate balance) before trusting a result
# - analyse a conversion experiment: two-proportion z-test, confidence interval for the lift, effect size
# - show by simulation that **peeking** inflates false positives, and that testing many metrics needs a **multiple-comparison correction**
#
# **Prerequisites:** [Hypothesis testing](06_hypothesis_testing.ipynb).
#
# > Data used: `datasets/ab_test.csv` (8,000 users; `group`, `device`, `converted`, `revenue`; synthetic, see `docs/DATASETS.md`).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_effectsize, proportions_ztest

DATA = Path("../datasets")
rng = np.random.default_rng(42)
ab = pd.read_csv(DATA / "ab_test.csv")
power_calc = NormalIndPower()

# %% [markdown]
# ## 1. Design before you look
#
# Write these down *before* launching (a mini pre-registration):
#
# | Decision | Our example |
# |---|---|
# | Hypothesis | The new page raises conversion. |
# | Primary metric | conversion rate (share of users who buy). One primary metric, to avoid multiple-testing games. |
# | Unit of randomisation | the **user** (not the page view), so that one person never sees both versions and rows are independent. |
# | Guardrail metrics | revenue per user, page load time: things that must not get worse. |
# | Alpha / power | 0.05 (two-sided) / 0.80 |
# | Minimum detectable effect (MDE) | the smallest lift worth shipping; here a **10% relative** lift. |
# | Sample size and duration | computed from the above (next section); do not stop early. |
#
# The baseline conversion rate comes from history. In our data the control group converts at:

# %%
base = ab.loc[ab["group"] == "control", "converted"].mean()
print(f"baseline conversion (control) = {base:.4f}")

# %% [markdown]
# ## 2. Power and sample size
#
# Power is the probability the test detects a real effect of the planned size. It depends on four linked quantities: effect size, sample size, alpha, power. Fix three, solve for the fourth.
# For proportions the standard effect size is **Cohen's h**: `h = 2*asin(sqrt(p2)) - 2*asin(sqrt(p1))`, which makes the calculation depend only on `h`.

# %%
p1 = 0.10
p2 = p1 * 1.10                                 # 10% relative lift: 10.0% -> 11.0%
h = proportion_effectsize(p2, p1)
n_per_group = power_calc.solve_power(effect_size=h, alpha=0.05, power=0.80, ratio=1.0, alternative="two-sided")
print(f"Cohen's h = {h:.4f}; users needed per group = {int(np.ceil(n_per_group)):,}")

# cross-check with the textbook formula n = (z_(1-a/2) + z_power)^2 * (p1 q1 + p2 q2) / (p2 - p1)^2
z_a, z_b = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
n_formula = (z_a + z_b) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2
print(f"classic formula gives {n_formula:,.0f} per group (close; the two use slightly different variance approximations)")

# %% [markdown]
# So detecting a **1 percentage point** lift on a 10% baseline needs about 15,000 users *per group*, roughly four times what this experiment had (about 4,000 per group). Sample size depends on effect size squared, so smaller lifts get expensive fast:

# %%
rows = []
for rel in [0.05, 0.10, 0.20, 0.30]:
    hh = proportion_effectsize(p1 * (1 + rel), p1)
    rows.append({"relative lift": f"{rel:.0%}", "target rate": f"{p1 * (1 + rel):.1%}", "users per group": int(np.ceil(power_calc.solve_power(hh, alpha=0.05, power=0.8)))})
pd.DataFrame(rows)

# %% [markdown]
# We can also turn the question around: with the sample we actually have, what is the smallest lift we could reliably detect (80% power)?

# %%
n_actual = int(ab["group"].value_counts().min())
h_mde = power_calc.solve_power(effect_size=None, nobs1=n_actual, alpha=0.05, power=0.80)
p2_mde = np.sin(np.arcsin(np.sqrt(p1)) + h_mde / 2) ** 2              # invert Cohen's h
print(f"with {n_actual:,} users per group the minimum detectable lift is {p1:.1%} -> {p2_mde:.1%} ({p2_mde / p1 - 1:.0%} relative)")

# %% [markdown]
# Let us trust but verify: simulate many experiments with the true rates 10% and 11% at the computed sample size and count how often the test rejects.

# %%
def sim_power(p_ctrl, p_trt, n, reps=3000, seed=0):
    r = np.random.default_rng(seed)
    x1, x2 = r.binomial(n, p_ctrl, reps), r.binomial(n, p_trt, reps)
    pooled = (x1 + x2) / (2 * n)
    z = (x2 - x1) / n / np.sqrt(pooled * (1 - pooled) * 2 / n)
    return (2 * stats.norm.sf(np.abs(z)) < 0.05).mean()

print(f"simulated power at n = {int(np.ceil(n_per_group)):,} per group: {sim_power(p1, p2, int(np.ceil(n_per_group))):.3f}  (target 0.80)")

# %% [markdown]
# ## 3. Sanity checks before analysis
#
# **Sample ratio mismatch (SRM).** We intended a 50/50 split. If the observed counts deviate more than chance allows, something broke (a bug in assignment, bots, a redirect that loses users) and the results cannot be trusted.
# Test the counts against the intended ratio with a chi-square goodness-of-fit test. Practitioners use a strict threshold (about 0.001) because the check is cheap and a failure is serious.

# %%
counts = ab["group"].value_counts().sort_index()
chi2, p_srm = stats.chisquare(counts.to_numpy(), f_exp=[counts.sum() / 2] * 2)
print(counts.to_dict(), f"-> chi2 = {chi2:.2f}, p = {p_srm:.3f}  (no evidence of a broken split)")

# %% [markdown]
# **Covariate balance.** Randomisation should equalise everything else too. Check that the device mix is the same in both arms (a chi-square test of independence):

# %%
dev = pd.crosstab(ab["group"], ab["device"])
print(dev)
print(f"device mix by group: chi-square p = {stats.chi2_contingency(dev)[1]:.3f}")

# %% [markdown]
# Both checks pass. (With many covariates, a few will look unbalanced by chance; that is what the multiple-comparisons section is about.)

# %% [markdown]
# ## 4. The analysis
#
# **Null:** conversion probability is the same in both arms. **Assumptions:** users are independent and randomised; counts are large enough for the normal approximation (yes: hundreds of conversions per arm).

# %%
c = ab.groupby("group")["converted"].agg(["sum", "count"])
x_t, n_t = c.loc["treatment"]; x_c, n_c = c.loc["control"]
z, p = proportions_ztest([x_t, x_c], [n_t, n_c])
lo, hi = confint_proportions_2indep(x_t, n_t, x_c, n_c, method="newcomb", compare="diff")
h_obs = proportion_effectsize(x_t / n_t, x_c / n_c)
print(f"control {x_c / n_c:.4f}  treatment {x_t / n_t:.4f}  absolute lift {x_t / n_t - x_c / n_c:+.4f}  relative {(x_t / n_t) / (x_c / n_c) - 1:+.1%}")
print(f"z = {z:.2f}, p = {p:.3f}; 95% CI for the difference: [{lo:+.4f}, {hi:+.4f}]; Cohen's h = {h_obs:.3f}")

# %% [markdown]
# **How to read this.** `p = 0.54`: no evidence against "no difference". The interval is the more useful part: the data are compatible with anything from about a percentage point *worse* to almost two points *better*. That range includes
# the 1-point lift we were hoping for, so this test is **inconclusive**, not negative. It was underpowered (section 2). The honest recommendations are "extend the experiment to the planned sample size" or "accept that any effect is smaller than
# what we can measure and decide on other grounds". It would be wrong to write "the new page does not work".
#
# **Secondary metric: revenue per user.** It includes many zeros and a long tail, so use Welch's t-test (mean is the business quantity) and check with a bootstrap:

# %%
rt, rc = ab.loc[ab["group"] == "treatment", "revenue"], ab.loc[ab["group"] == "control", "revenue"]
t, p_rev = stats.ttest_ind(rt, rc, equal_var=False)
boot = np.array([rng.choice(rt.to_numpy(), len(rt)).mean() - rng.choice(rc.to_numpy(), len(rc)).mean() for _ in range(3000)])
print(f"revenue per user: treatment {rt.mean():.2f} vs control {rc.mean():.2f}; Welch p = {p_rev:.3f}; bootstrap 95% CI for the difference: [{np.percentile(boot, 2.5):+.2f}, {np.percentile(boot, 97.5):+.2f}]")

# %% [markdown]
# ## 5. Peeking: why you must not stop when p dips below 0.05
#
# The tempting workflow: check the dashboard every day and stop as soon as `p < 0.05`. But every look is another chance for luck to cross the line. The p-value of a *fixed-sample* test is valid only if you look **once**, at the planned end.
# Simulate **A/A tests** (both arms identical, so any "win" is a false positive), looking 10 times at equally spaced sample sizes:

# %%
def peeking_sim(looks=10, per_look=1000, p=0.10, reps=4000, alpha=0.05, seed=1):
    r = np.random.default_rng(seed)
    a = np.cumsum(r.binomial(per_look, p, (reps, looks)), axis=1)     # cumulative conversions after each look
    b = np.cumsum(r.binomial(per_look, p, (reps, looks)), axis=1)
    n = per_look * np.arange(1, looks + 1)
    pooled = (a + b) / (2 * n)
    z = (b - a) / n / np.sqrt(pooled * (1 - pooled) * 2 / n)
    pv = 2 * stats.norm.sf(np.abs(z))
    return pv

pv = peeking_sim()
print(f"false positives if you look ONCE at the end        : {(pv[:, -1] < 0.05).mean():.3f}")
print(f"false positives if you stop at the first p < 0.05  : {(pv.min(axis=1) < 0.05).mean():.3f}   (10 looks)")
print(f"with a Bonferroni-adjusted threshold 0.05/10       : {(pv.min(axis=1) < 0.005).mean():.3f}")

# %%
fig, ax = plt.subplots(figsize=(7.5, 3.8))
for i in range(12):
    ax.plot(range(1, 11), pv[i], lw=1, alpha=0.8)
ax.axhline(0.05, color="k", ls="--"); ax.set_yscale("log")
ax.set_xlabel("look number (each look adds 1,000 users per group)"); ax.set_ylabel("p-value (log scale)")
ax.set_title("12 A/A tests: identical arms, yet several dip below 0.05")
plt.show()

# %% [markdown]
# *What just happened?* With ten looks the chance of at least one "significant" moment is several times the nominal 5%. Options: (1) decide `n` in advance and look once; (2) use a **sequential design** (group-sequential
# boundaries or always-valid methods) that is built for repeated looks; (3) a crude fix like Bonferroni across looks (conservative but valid, shown above). Many experimentation platforms implement option 2.

# %% [markdown]
# ## 6. Multiple comparisons: many metrics, many segments
#
# Test 20 metrics or segments and, if nothing is going on, the chance that at least one has `p < 0.05` is `1 - 0.95^20`. Simulate 20 independent null tests per "experiment":

# %%
print(f"theory: P(at least one false positive among 20 null tests) = {1 - 0.95 ** 20:.3f}")
m = 20
pvals = np.random.default_rng(5).uniform(size=(5000, m))              # p-values are uniform under H0
print(f"simulation, uncorrected            : {(pvals.min(axis=1) < 0.05).mean():.3f}")
print(f"simulation, Bonferroni (alpha/{m})   : {(pvals.min(axis=1) < 0.05 / m).mean():.3f}")
rej_holm = np.array([multipletests(row, alpha=0.05, method="holm")[0].any() for row in pvals[:1000]])
print(f"simulation, Holm (first 1000 rows)  : {rej_holm.mean():.3f}")

# %% [markdown]
# **Bonferroni** compares each p-value with `alpha/m`: simple, controls the chance of *any* false positive (family-wise error rate), conservative. **Holm** is uniformly more powerful and equally valid; prefer it.
# **Benjamini-Hochberg (BH)** controls the *false discovery rate* (the expected share of false positives among your discoveries): the usual choice when you are screening many hypotheses and can tolerate a few false leads.
# Example with one real effect hidden among 19 nulls:

# %%
r = np.random.default_rng(3)
n_each = 4000
pv_list = []
for k in range(20):
    lift = 0.03 if k == 0 else 0.0                                       # metric 0 has a real +3pt effect
    x1, x2 = r.binomial(n_each, 0.10), r.binomial(n_each, 0.10 + lift)
    pv_list.append(proportions_ztest([x2, x1], [n_each, n_each])[1])
pv_arr = np.array(pv_list)
for method in ["bonferroni", "holm", "fdr_bh"]:
    rej = multipletests(pv_arr, alpha=0.05, method=method)[0]
    print(f"{method:11s}: rejects {rej.sum()} hypotheses; includes the real effect (metric 0): {bool(rej[0])}")
print(f"uncorrected : rejects {(pv_arr < 0.05).sum()} hypotheses")

# %% [markdown]
# ## A reporting template
#
# 1. **Question and decision** the test informs. 2. **Design**: primary metric, MDE, alpha, power, planned `n`, dates. 3. **Checks**: SRM and balance passed. 4. **Result**: effect (absolute and relative) with a 95% interval, p-value, effect size.
# 5. **Guardrails** and secondary metrics (with multiple-comparison handling). 6. **Recommendation** that reflects the interval, not only the p-value; state limitations (novelty effects, short duration, single population).

# %% [markdown]
# ## Common mistakes
#
# 1. **No power calculation**, then reading a non-significant result as "no effect".
# 2. **Peeking** and stopping at the first significant p-value.
# 3. **Randomising by page view instead of user** (one user counted many times, p-values too small).
# 4. **Skipping the SRM check**: a mismatch means the result is untrustworthy, whatever the p-value.
# 5. **Slicing after the fact** (by device, country, plan...) and announcing the one segment that "won" without correction.
# 6. **Changing the primary metric after seeing results.**
# 7. **Running for a partial cycle**, e.g. only weekdays, when behaviour differs on weekends.

# %%
# Mistake 5 in action: the *actual* segments of this experiment
for dev_name, sub in ab.groupby("device"):
    cc = sub.groupby("group")["converted"].agg(["sum", "count"])
    zz, pp = proportions_ztest([cc.loc["treatment", "sum"], cc.loc["control", "sum"]], [cc.loc["treatment", "count"], cc.loc["control", "count"]])
    print(f"{dev_name:8s} lift = {cc.loc['treatment', 'sum'] / cc.loc['treatment', 'count'] - cc.loc['control', 'sum'] / cc.loc['control', 'count']:+.4f}, p = {pp:.3f}")
print("if one of these had crept under 0.05 we would still need to correct for having looked at 2 segments (and any others we tried)")

# %% [markdown]
# ## Interview corner
#
# **1. How do you decide the sample size for an A/B test?**
# <details><summary>Answer</summary>
# Fix alpha (0.05), power (0.8), the baseline rate and the minimum detectable effect that would change the decision; solve for `n` per group (statsmodels `NormalIndPower` with Cohen's h, or the classic formula). Then convert to days using traffic and round up to full weekly cycles.
# </details>
#
# **2. What is peeking and why is it a problem?**
# <details><summary>Answer</summary>
# Repeatedly testing an accumulating experiment and stopping when p &lt; 0.05. Each look gives luck another chance, so the false-positive rate grows well beyond alpha. Use a fixed horizon, or a sequential/alpha-spending method designed for interim looks.
# </details>
#
# **3. What is a sample ratio mismatch and what do you do about it?**
# <details><summary>Answer</summary>
# Observed group sizes differ from the designed split by more than chance (chi-square goodness of fit, strict threshold). It signals an assignment, logging or filtering bug. Do not interpret the metric results; find and fix the cause, then rerun.
# </details>
#
# **4. You test 15 metrics and one is significant at p = 0.03. Do you ship?**
# <details><summary>Answer</summary>
# Not on that alone. With 15 tests, a p-value of 0.03 is unremarkable (Bonferroni would need 0.0033). Was it the pre-registered primary metric? Correct for multiplicity, look for corroborating metrics and mechanism, and consider a confirmatory rerun.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Size a new experiment
# A checkout has a baseline conversion of 8%. You want to detect an increase to **9.2%** with 90% power at alpha = 0.05 (two-sided, equal groups). Store the required users per group (rounded up to an int) in `n_needed`,
# and store in `n_needed_80` the same for 80% power. *Hint: `proportion_effectsize(0.092, 0.08)` then `NormalIndPower().solve_power`.*

# %% tags=["starter"]
n_needed = ...
n_needed_80 = ...

# %%
hh = proportion_effectsize(0.092, 0.08)
e90 = int(np.ceil(NormalIndPower().solve_power(hh, alpha=0.05, power=0.9)))
e80 = int(np.ceil(NormalIndPower().solve_power(hh, alpha=0.05, power=0.8)))
assert n_needed == e90, f"expected {e90} per group for 90% power"
assert n_needed_80 == e80, f"expected {e80} per group for 80% power"
assert n_needed > n_needed_80, "more power needs more users"
assert sim_power(0.08, 0.092, n_needed, reps=3000, seed=4) > 0.85, "a simulation at your n should reach roughly 90% power"
print(f"Exercise 1 passed: {n_needed:,} users per group for 90% power, {n_needed_80:,} for 80%")

# %% tags=["solution"]
_h = proportion_effectsize(0.092, 0.08)
n_needed = int(np.ceil(NormalIndPower().solve_power(_h, alpha=0.05, power=0.9)))
n_needed_80 = int(np.ceil(NormalIndPower().solve_power(_h, alpha=0.05, power=0.8)))

# %% [markdown]
# ### Exercise 2: A suspicious split
# An experiment intended a 50/50 split but the logs show 10,300 users in A and 9,700 in B. Run the SRM check (chi-square goodness of fit against 50/50) and store the p-value in `p_srm_ex`;
# store `srm_flag = True` if `p < 0.001` (the usual strict SRM threshold).

# %% tags=["starter"]
p_srm_ex = ...
srm_flag = ...

# %%
e_p = stats.chisquare([10300, 9700], f_exp=[10000, 10000]).pvalue
assert abs(p_srm_ex - e_p) < 1e-12, "stats.chisquare([10300, 9700], f_exp=[10000, 10000])"
assert bool(srm_flag) == bool(e_p < 0.001)
print(f"Exercise 2 passed: p = {p_srm_ex:.2e}, SRM flagged: {srm_flag}")

# %% tags=["solution"]
p_srm_ex = float(stats.chisquare([10300, 9700], f_exp=[10000, 10000]).pvalue)
srm_flag = bool(p_srm_ex < 0.001)

# %% [markdown]
# ### Exercise 3: How bad is checking 5 times?
# Using `peeking_sim(looks=5, per_look=2000, reps=4000, seed=9)` (A/A tests), store in `fp_peek5` the fraction of experiments where **any** of the 5 looks has `p < 0.05`,
# in `fp_final5` the fraction where the **last** look is significant, and in `fp_bonf5` the fraction with any look below `0.05 / 5`.

# %% tags=["starter"]
fp_peek5 = ...
fp_final5 = ...
fp_bonf5 = ...

# %%
pv5 = peeking_sim(looks=5, per_look=2000, reps=4000, seed=9)
assert abs(fp_peek5 - (pv5.min(axis=1) < 0.05).mean()) < 1e-12, "any look below 0.05"
assert abs(fp_final5 - (pv5[:, -1] < 0.05).mean()) < 1e-12, "only the last column"
assert abs(fp_bonf5 - (pv5.min(axis=1) < 0.01).mean()) < 1e-12, "threshold 0.05 / 5 = 0.01"
assert fp_peek5 > 0.09 and fp_bonf5 < 0.06, "peeking should be clearly inflated and Bonferroni close to or below 5%"
print(f"Exercise 3 passed: any-look {fp_peek5:.3f}, final-only {fp_final5:.3f}, Bonferroni {fp_bonf5:.3f}")

# %% tags=["solution"]
_pv5 = peeking_sim(looks=5, per_look=2000, reps=4000, seed=9)
fp_peek5 = float((_pv5.min(axis=1) < 0.05).mean())
fp_final5 = float((_pv5[:, -1] < 0.05).mean())
fp_bonf5 = float((_pv5.min(axis=1) < 0.05 / 5).mean())

# %% [markdown]
# ### Exercise 4 (open-ended): Write the experiment report
# Using the template above, write a half-page recommendation for this A/B test for a product manager: the effect with its interval, why it is inconclusive, the sample size that would settle a 10% relative lift,
# how many more days that is if traffic is about 1,000 users per day split evenly, and the guardrail result for revenue. Compute every number in code.

# %% [markdown]
# ## Recap
# - Design first: primary metric, unit of randomisation, MDE, alpha, power, then solve for `n` (sample size grows with `1/effect^2`).
# - Check SRM and covariate balance before reading any result.
# - Report the effect, its confidence interval and effect size; a non-significant underpowered test is inconclusive, not negative.
# - Peeking inflates false positives; use a fixed horizon or a sequential method. Many metrics/segments need Bonferroni/Holm (any false positive) or Benjamini-Hochberg (false discovery rate).
#
# **Go deeper:** the statsmodels docs for `NormalIndPower`, `proportion_effectsize`, `multipletests`; "Trustworthy Online Controlled Experiments" (Kohavi, Tang, Xu) is the standard practitioner book.
#
# **Next:** [Correlation and regression with statsmodels](08_correlation_and_regression_with_statsmodels.ipynb)

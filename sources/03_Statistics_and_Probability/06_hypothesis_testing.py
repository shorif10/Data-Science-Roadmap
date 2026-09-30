# %% [markdown]
# # Hypothesis Testing: p-values, t-test, chi-square, Mann-Whitney, and Two Kinds of Error
#
# **Why this matters.** "Is the new checkout better, or did we get lucky?" "Do churners really have shorter tenure?" Hypothesis tests are the standard tool for separating signal from noise, and
# they are misused more than any other statistical tool: p-values are misread, tests are applied where their assumptions are broken, and "significant" gets confused with "important". In this notebook
# you will build the logic from scratch with a permutation test, then use the standard tests, and, for each one, **state its null hypothesis, its assumptions, and watch it break when those assumptions fail**.
#
# **You will learn**
# - state a null and alternative hypothesis, and define a p-value precisely (and list what it is *not*)
# - run a permutation test, a two-proportion z-test, Welch's t-test, a chi-square test and a Mann-Whitney U test
# - explain type I and type II errors and see them by simulation
# - separate statistical significance from practical importance, and recognise invalid test setups
#
# **Prerequisites:** [Confidence intervals and the bootstrap](05_confidence_intervals_and_bootstrap.ipynb).
#
# > Data used: `ab_test.csv` and `churn.csv` (synthetic, see `docs/DATASETS.md`).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

DATA = Path("../datasets")
rng = np.random.default_rng(42)
ab = pd.read_csv(DATA / "ab_test.csv")
churn = pd.read_csv(DATA / "churn.csv")
churn["contract"] = churn["contract"].str.lower()

# %% [markdown]
# ## 1. The logic of a test
#
# A hypothesis test is a proof by contradiction with probabilities:
#
# 1. **Null hypothesis H0**: the boring explanation ("treatment and control convert at the same rate; any difference is luck").
# 2. **Alternative H1**: what you suspect ("the rates differ"; or one-sided: "treatment is higher").
# 3. **Test statistic**: one number summarising the evidence (difference in conversion rates).
# 4. **p-value**: *assuming H0 is true*, the probability of seeing a result at least as extreme as the one observed.
# 5. Compare to a pre-chosen **significance level alpha** (usually 0.05). If `p < alpha`, we "reject H0"; otherwise we "fail to reject" it (which is **not** the same as proving H0 true).
#
# Observed data in our A/B test:

# %%
g = ab.groupby("group")["converted"].agg(["sum", "count", "mean"])
print(g.round(4))
obs_diff = g.loc["treatment", "mean"] - g.loc["control", "mean"]
print(f"observed lift (treatment - control) = {obs_diff:+.4f}")

# %% [markdown]
# ## 2. The permutation test: the p-value with no formulas
#
# **Null:** group labels are irrelevant to conversion. If that is true, shuffling the labels should produce differences as large as the observed one reasonably often. So: shuffle the labels thousands of times, compute the difference each
# time, and count how often the shuffled difference is at least as extreme as the real one. That fraction *is* the (two-sided) p-value.

# %%
converted = ab["converted"].to_numpy()
is_treat = (ab["group"] == "treatment").to_numpy()
perm_diffs = np.empty(10_000)
for i in range(len(perm_diffs)):
    shuffled = rng.permutation(is_treat)
    perm_diffs[i] = converted[shuffled].mean() - converted[~shuffled].mean()

p_perm = (np.abs(perm_diffs) >= abs(obs_diff)).mean()
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.hist(perm_diffs, bins=60, color="lightsteelblue")
ax.axvline(obs_diff, color="tab:red", label=f"observed {obs_diff:+.4f}"); ax.axvline(-obs_diff, color="tab:red")
ax.set_xlabel("difference in conversion rate under shuffled labels"); ax.set_ylabel("permutations"); ax.set_title("What luck alone produces (H0)")
ax.legend(); plt.show()
print(f"permutation p-value = {p_perm:.3f}")

# %% [markdown]
# *What just happened?* Luck alone produces differences this large in roughly half of the shuffles, so our observed lift is entirely unremarkable under H0. We **fail to reject** the null. It does *not* prove the treatment does nothing:
# a true small lift would also be hard to see with this much noise (power, section 8).

# %% [markdown]
# ## 3. What a p-value is *not*
#
# | Misreading | Truth |
# |---|---|
# | "p = 0.03 means a 3% chance H0 is true" | It is P(data this extreme \| H0), not P(H0 \| data). Getting P(H0 \| data) needs a prior (Bayes). |
# | "p > 0.05 means no effect" | It means the data are not surprising under H0: absence of evidence, not evidence of absence. |
# | "p < 0.05 means the effect is big / important" | p mixes effect size and sample size. A huge sample makes trivial effects significant. |
# | "A smaller p-value means a bigger effect" | Same reason. |
# | "p = 0.049 and p = 0.051 are opposites" | They are essentially the same evidence; 0.05 is a convention. |
#
# One property worth *seeing*: if H0 is true, the p-value is uniformly distributed between 0 and 1, so a true null still gives `p < 0.05` in 5% of experiments.

# %%
def two_group_pvalues(effect, n=50, reps=3000, seed=0):
    r = np.random.default_rng(seed)
    a = r.normal(0, 1, (reps, n)); b = r.normal(effect, 1, (reps, n))
    return stats.ttest_ind(a, b, axis=1).pvalue

p_null, p_real = two_group_pvalues(0.0), two_group_pvalues(0.5)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
axes[0].hist(p_null, bins=20, range=(0, 1)); axes[0].set_title("H0 true: p-values are uniform")
axes[1].hist(p_real, bins=20, range=(0, 1)); axes[1].set_title("Real effect (0.5 sd, n=50/group): p piles up near 0")
for ax in axes:
    ax.set_xlabel("p-value"); ax.set_ylabel("experiments")
plt.tight_layout(); plt.show()
print(f"share with p < 0.05 when H0 is true: {(p_null < 0.05).mean():.3f}   when the effect is real: {(p_real < 0.05).mean():.3f}")

# %% [markdown]
# ## 4. Two proportions: z-test and chi-square (they are the same test)
#
# **Null:** the two groups have the same conversion probability. **Test statistic:** the difference in proportions divided by its standard error (under H0 with a *pooled* rate).
# **Assumptions:** (1) each user is independent of the others and appears once, (2) users are randomly assigned, (3) expected counts in every cell of the 2x2 table are large enough (a common rule: at least 5), so the normal approximation holds.

# %%
count = g["sum"].loc[["treatment", "control"]].to_numpy()
nobs = g["count"].loc[["treatment", "control"]].to_numpy()
z, p_z = proportions_ztest(count, nobs)
print(f"two-proportion z-test: z = {z:.3f}, p = {p_z:.3f}")

table = pd.crosstab(ab["group"], ab["converted"])
chi2, p_chi, dof, expected = stats.chi2_contingency(table, correction=False)
print(f"chi-square (no continuity correction): chi2 = {chi2:.3f}, p = {p_chi:.3f};  z^2 = {z**2:.3f}  -> identical")
print("expected counts under H0:\n", pd.DataFrame(expected, index=table.index, columns=table.columns).round(1))

# %% [markdown]
# The permutation p-value and both formula-based p-values agree (all around 0.5). **When the assumptions fail:** with tiny counts the normal approximation is unreliable and you should use **Fisher's exact test**:

# %%
tiny = np.array([[3, 1], [1, 4]])
print("expected counts:\n", stats.chi2_contingency(tiny, correction=False)[3].round(2), "(some below 5: chi-square is unreliable here)")
print("chi-square p =", round(stats.chi2_contingency(tiny, correction=False)[1], 3), "| Fisher exact p =", round(stats.fisher_exact(tiny)[1], 3))

# %% [markdown]
# ## 5. Chi-square test of independence for a larger table
#
# **Null:** contract type and churn are independent. **Assumptions:** independent rows, and expected counts of at least about 5 per cell. It tells you *whether* there is an association, not how strong; add an effect size
# such as **Cramer's V** (0 = none, 1 = perfect).

# %%
ct = pd.crosstab(churn["contract"], churn["churned"])
chi2, p, dof, expd = stats.chi2_contingency(ct)
cramers_v = np.sqrt(chi2 / (ct.to_numpy().sum() * (min(ct.shape) - 1)))
print(ct.assign(churn_rate=(ct[1] / ct.sum(axis=1)).round(3)))
print(f"chi2 = {chi2:.1f}, dof = {dof}, p = {p:.2e}, Cramer's V = {cramers_v:.2f}, smallest expected count = {expd.min():.0f}")

# %% [markdown]
# ## 6. Comparing means: Welch's t-test
#
# **Null:** the two groups have the same population mean. **Assumptions:** independent observations; the *sampling distribution of the mean* is roughly normal (fine for moderate `n`, per the CLT; for very small skewed samples it is not).
# **Use Welch's version** (`equal_var=False`): it does not assume equal variances. The classic pooled t-test does, and can badly misbehave when group sizes and variances both differ:

# %%
ch = churn[churn["churned"] == 1]["tenure_months"]; st = churn[churn["churned"] == 0]["tenure_months"]
t, p_t = stats.ttest_ind(ch, st, equal_var=False)
print(f"mean tenure: churned {ch.mean():.1f} vs stayed {st.mean():.1f} months;  Welch t = {t:.1f}, p = {p_t:.1e}")

# %%
r = np.random.default_rng(0)
small_noisy = r.normal(0, 5, (5000, 10))        # 10 observations, sd 5
large_calm = r.normal(0, 1, (5000, 100))        # 100 observations, sd 1; same true mean, so H0 is TRUE
fp_pooled = (stats.ttest_ind(small_noisy, large_calm, axis=1, equal_var=True).pvalue < 0.05).mean()
fp_welch = (stats.ttest_ind(small_noisy, large_calm, axis=1, equal_var=False).pvalue < 0.05).mean()
print(f"false-positive rate when H0 is true (nominal 5%): pooled t-test {fp_pooled:.1%}, Welch {fp_welch:.1%}")

# %% [markdown]
# *What just happened?* With unequal variances *and* unequal sizes, the pooled test declares a "significant" difference about half the time even though the true means are identical. The test is **invalid** there, and nothing in its output warns you.
# Welch stays near 5%. Default to Welch.

# %% [markdown]
# ## 7. When the mean is the wrong summary: Mann-Whitney U
#
# The **Mann-Whitney U test** (also called the Wilcoxon rank-sum test) compares two groups using *ranks*. **Null:** a random value from group A is equally likely to be larger or smaller than a random value from group B
# (`P(A > B) = 0.5`). It is robust to outliers and skew. **Caveat:** it is a test about *stochastic dominance*, and only a test of *medians* if the two distributions have the same shape.
# Revenue among users who converted, treatment vs control:

# %%
conv = ab[ab["converted"] == 1]
rt, rc = conv.loc[conv["group"] == "treatment", "revenue"], conv.loc[conv["group"] == "control", "revenue"]
u, p_u = stats.mannwhitneyu(rt, rc, alternative="two-sided")
_, p_w = stats.ttest_ind(rt, rc, equal_var=False)
print(f"n = {len(rt)} / {len(rc)}; means {rt.mean():.1f} vs {rc.mean():.1f}, medians {rt.median():.1f} vs {rc.median():.1f}")
print(f"Mann-Whitney p = {p_u:.3f}    Welch t-test p = {p_w:.3f}   (neither detects a difference in order value)")
print(f"common-language effect size P(treatment > control) = {u / (len(rt) * len(rc)):.3f}   (0.5 = no difference)")

# %% [markdown]
# ## 8. Two ways to be wrong: type I and type II errors, and power
#
# | | H0 true (no effect) | H1 true (real effect) |
# |---|---|---|
# | **Reject H0** | **Type I error** (false positive), probability = alpha | correct: probability = **power** |
# | **Fail to reject** | correct | **Type II error** (false negative), probability = beta = 1 - power |
#
# alpha is set by you. **Power** depends on the true effect size, the noise, alpha and the sample size, and it is why "not significant" often means "not enough data". Simulate an A/B test with a true 12% relative lift
# (10% -> 11.2%), which is the size the dataset description says was planted, at several sample sizes:

# %%
def sim_power(p_ctrl, p_trt, n, reps=2000, seed=0):
    r = np.random.default_rng(seed)
    x1 = r.binomial(n, p_ctrl, reps); x2 = r.binomial(n, p_trt, reps)
    pooled = (x1 + x2) / (2 * n)
    z = (x2 / n - x1 / n) / np.sqrt(pooled * (1 - pooled) * 2 / n)
    return (2 * stats.norm.sf(np.abs(z)) < 0.05).mean()

print(f"{'users per group':>16} {'false-positive rate (no effect)':>34} {'power (10% -> 11.2%)':>22}")
for n in [1000, 4000, 16000, 64000]:
    print(f"{n:16,d} {sim_power(0.10, 0.10, n):34.3f} {sim_power(0.10, 0.112, n):22.3f}")

# %% [markdown]
# The false-positive rate stays near 5% whatever `n` is (that is what alpha means), while power climbs with `n`. With 4,000 users per group (roughly our experiment) power is far below the conventional 80%, so
# an inconclusive result was to be expected: the test could not have reliably seen a lift of that size. Planning `n` in advance is the subject of the next notebook.

# %% [markdown]
# ## 9. Statistically significant is not the same as important
# Given enough data, any nonzero difference becomes "significant". Conversion 10.00% vs 10.10% with four million users per group:

# %%
n_big = 4_000_000
z_big, p_big = proportions_ztest([int(0.1010 * n_big), int(0.1000 * n_big)], [n_big, n_big])
print(f"difference of 0.1 percentage points, n = {n_big:,} per group: z = {z_big:.2f}, p = {p_big:.1e}")

# %% [markdown]
# `p` is far below 0.05, yet the lift is a tenth of a percentage point: whether that pays for the engineering effort is a business judgement, not a statistical one. Always report the **effect size and a confidence interval**, not only the p-value.

# %% [markdown]
# ## Test chooser
#
# | Question | Test | Null | Check |
# |---|---|---|---|
# | Two proportions | z-test / chi-square (Fisher if counts small) | equal rates | expected counts >= 5, independent units |
# | Two means | Welch t-test | equal means | independence, means roughly normal (n, skew) |
# | Two distributions, outliers/skew | Mann-Whitney U | P(A>B) = 0.5 | independence; same shape if you want a median claim |
# | Category vs category | chi-square independence | independent | expected counts >= 5 |
# | Same subjects before/after | paired t-test / Wilcoxon signed-rank | mean difference 0 | pairs are the unit |

# %% [markdown]
# ## Common mistakes
#
# 1. **Reading p as "probability the null is true"** or "no significance means no effect".
# 2. **Treating repeated rows from the same user as independent observations** (pseudo-replication): your effective `n` is the number of users, and p-values become far too small.
# 3. **Choosing the test after seeing the data** (try t-test, then Mann-Whitney, keep the smaller p): that inflates false positives.
# 4. **Using the pooled t-test** when variances or sizes differ.
# 5. **Chi-square with tiny expected counts**, or applied to percentages instead of raw counts.
# 6. **Ignoring effect size and power.** A significant tiny effect and a non-significant underpowered test both mislead.

# %%
# Mistake 5: chi-square on percentages instead of counts gives a different (wrong) answer
counts = np.array([[425, 3616], [400, 3559]])
print("chi-square p on counts     :", round(stats.chi2_contingency(counts, correction=False)[1], 3))
print("chi-square p on percentages:", round(stats.chi2_contingency(counts / counts.sum(axis=1, keepdims=True) * 100, correction=False)[1], 3), "(meaningless: the sample size was thrown away)")

# %% [markdown]
# ## Interview corner
#
# **1. Explain a p-value to a non-technical stakeholder.**
# <details><summary>Answer</summary>
# "If the new design made no difference, how often would we see a gap at least this big just by chance? A p-value of 0.03 means about 3 in 100 experiments. It does not tell us how likely the design is to work, or how large the benefit is."
# </details>
#
# **2. What are type I and type II errors? How does sample size affect them?**
# <details><summary>Answer</summary>
# Type I: rejecting a true null (rate = alpha, chosen by you). Type II: missing a real effect (rate = beta). Increasing n reduces type II errors (raises power) without changing alpha.
# </details>
#
# **3. When would you use Mann-Whitney over a t-test?**
# <details><summary>Answer</summary>
# Small samples of strongly skewed or outlier-prone data, or ordinal data. With large samples the t-test on means is usually fine by the CLT, and if your business metric is the mean (revenue per user) the mean-based test answers the right question.
# </details>
#
# **4. Your A/B test is not significant. What do you conclude?**
# <details><summary>Answer</summary>
# Only that you have not detected a difference. Report the confidence interval for the effect, check the power to detect the smallest lift you care about, and decide whether to collect more data or stop.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: A permutation test on tenure
# **Null:** tenure is unrelated to churn status. Use `mean(tenure | churned) - mean(tenure | stayed)` as the statistic; shuffle the churn labels 5,000 times using `np.random.default_rng(1)`, and store the two-sided permutation p-value in `p_perm_tenure`
# and the observed difference in `obs_tenure_diff`. *Hint: the Welch test above suggests the answer will be extreme: the smallest possible p-value with 5,000 shuffles is 0, so report `(count + 1) / (5000 + 1)` for a safe estimate.*

# %% tags=["starter"]
p_perm_tenure = ...
obs_tenure_diff = ...

# %%
ten = churn["tenure_months"].to_numpy(dtype=float); lab = churn["churned"].to_numpy().astype(bool)
exp_diff = ten[lab].mean() - ten[~lab].mean()
assert abs(obs_tenure_diff - exp_diff) < 1e-9, "obs = mean tenure(churned) - mean tenure(stayed)"
assert abs(p_perm_tenure - 1 / 5001) < 1e-12, "with this strong an effect no shuffle should reach the observed difference, so (0 + 1) / 5001"
print(f"Exercise 1 passed: observed difference {obs_tenure_diff:.2f} months, permutation p = {p_perm_tenure:.5f}")

# %% tags=["solution"]
_ten = churn["tenure_months"].to_numpy(dtype=float)
_lab = churn["churned"].to_numpy().astype(bool)
obs_tenure_diff = float(_ten[_lab].mean() - _ten[~_lab].mean())
_r = np.random.default_rng(1)
_cnt = 0
for _ in range(5000):
    _s = _r.permutation(_lab)
    _cnt += abs(_ten[_s].mean() - _ten[~_s].mean()) >= abs(obs_tenure_diff)
p_perm_tenure = (_cnt + 1) / (5000 + 1)

# %% [markdown]
# ### Exercise 2: Paperless billing and churn
# Test whether `paperless_billing` is associated with `churned`. Store the chi-square p-value (no continuity correction) in `p_paperless`, Cramer's V in `v_paperless`, and `significant_but_small` = True if `p < 0.05` **and** `V < 0.1`.
# *Hint: `pd.crosstab`, `stats.chi2_contingency(table, correction=False)`; V = sqrt(chi2 / (n * (min(rows, cols) - 1))).*

# %% tags=["starter"]
p_paperless = ...
v_paperless = ...
significant_but_small = ...

# %%
tb = pd.crosstab(churn["paperless_billing"], churn["churned"])
c2, pp, _, _ = stats.chi2_contingency(tb, correction=False)
vv = np.sqrt(c2 / (tb.to_numpy().sum() * (min(tb.shape) - 1)))
assert abs(p_paperless - pp) < 1e-12, "use correction=False"
assert abs(v_paperless - vv) < 1e-9, "Cramer's V = sqrt(chi2 / (n * (min(r, c) - 1)))"
assert bool(significant_but_small) == bool(pp < 0.05 and vv < 0.1)
print(f"Exercise 2 passed: p = {p_paperless:.4f}, V = {v_paperless:.3f}, significant but small: {significant_but_small}")

# %% tags=["solution"]
_tb = pd.crosstab(churn["paperless_billing"], churn["churned"])
_c2, p_paperless, _, _ = stats.chi2_contingency(_tb, correction=False)
v_paperless = float(np.sqrt(_c2 / (_tb.to_numpy().sum() * (min(_tb.shape) - 1))))
significant_but_small = bool(p_paperless < 0.05 and v_paperless < 0.1)

# %% [markdown]
# ### Exercise 3: Show that the pooled t-test lies
# Simulate 4,000 experiments (`np.random.default_rng(3)`) in which **H0 is true**: group A has 8 observations from `Normal(0, 4)`, group B has 80 observations from `Normal(0, 1)`. For each, run the pooled t-test (`equal_var=True`) and
# Welch's test. Store the fraction with `p < 0.05` for each as `fp_pooled_ex` and `fp_welch_ex`. *Hint: `stats.ttest_ind(A, B, axis=1, equal_var=...)` on arrays of shape `(4000, n)`.*

# %% tags=["starter"]
fp_pooled_ex = ...
fp_welch_ex = ...

# %%
assert 0.03 <= fp_welch_ex <= 0.075, f"Welch's false-positive rate should be near 0.05, got {fp_welch_ex:.3f}"
assert fp_pooled_ex > 0.10, f"the pooled test should be badly inflated here, got {fp_pooled_ex:.3f}: check group sizes and standard deviations"
print(f"Exercise 3 passed: pooled {fp_pooled_ex:.1%} vs Welch {fp_welch_ex:.1%} (nominal 5%)")

# %% tags=["solution"]
_r = np.random.default_rng(3)
_A = _r.normal(0, 4, (4000, 8))
_B = _r.normal(0, 1, (4000, 80))
fp_pooled_ex = float((stats.ttest_ind(_A, _B, axis=1, equal_var=True).pvalue < 0.05).mean())
fp_welch_ex = float((stats.ttest_ind(_A, _B, axis=1, equal_var=False).pvalue < 0.05).mean())

# %% [markdown]
# ### Exercise 4 (open-ended): Mobile vs desktop
# In `ab_test.csv`, does the conversion rate differ between `mobile` and `desktop` users (ignoring the treatment)? Write down H0, choose a test, check its assumptions (expected counts, independence), report the p-value,
# the two rates and a Wilson interval for each, and say in a markdown cell whether the difference is practically meaningful. Then repeat within the control group only: does the conclusion change?

# %% [markdown]
# ## Recap
# - A test asks: if H0 were true, how surprising is this data? The p-value is that probability, **not** the probability that H0 is true.
# - Permutation tests need almost no assumptions and show the logic; z/chi-square for proportions, Welch for means, Mann-Whitney for ranks each have stated assumptions that you must check.
# - The pooled t-test and chi-square with small counts can be badly wrong; Welch and Fisher are the safe defaults.
# - alpha controls type I errors; power (type II) depends on effect size, noise and `n`; "not significant" often means "underpowered".
# - Report effect sizes and intervals with the p-value; significance is not importance.
#
# **Go deeper:** the SciPy docs for `scipy.stats.permutation_test`, `mannwhitneyu` and `chi2_contingency`; "Statistical Rethinking" and the American Statistical Association's 2016 statement on p-values are widely cited discussions of interpretation.
#
# **Next:** [A/B testing end to end](07_ab_testing_end_to_end.ipynb)

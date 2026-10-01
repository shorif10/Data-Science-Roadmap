# %% [markdown]
# # Descriptive Statistics: Center, Spread, Shape and Outliers
#
# **Why this matters.** "What is the typical order value?" sounds simple until one whale customer drags the average to a number nobody
# has ever actually paid. Choosing the right summary (mean or median, standard deviation or IQR) and spotting outliers before they leak into a
# model or a dashboard is the first statistical skill an analyst is judged on.
#
# **You will learn**
# - summarise a variable's centre (mean, median, mode), spread (std, IQR, MAD) and shape (skewness)
# - choose *robust* statistics when data has outliers, and measure how much each summary moves
# - detect outliers with the IQR rule and z-scores, and know when each fails
# - use a log transform to tame right-skewed data such as prices
#
# **Prerequisites:** [pandas basics](../05_Pandas/01_loading_and_inspecting_data.ipynb) and the [NumPy notebooks](../04_NumPy/).
#
# > Data used: `datasets/housing.csv` and `datasets/churn.csv` (synthetic, see `docs/DATASETS.md`).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DATA = Path("../datasets")
housing = pd.read_csv(DATA / "housing.csv")
churn = pd.read_csv(DATA / "churn.csv")
price = housing["price"] / 1000            # thousand dollars, raw, including the planted bad values
print(f"{len(housing)} houses, {len(churn)} customers")

# %% [markdown]
# ## 1. Center: what is "typical"?
#
# - **Mean**: add up, divide by count. Uses every value, so it is pulled by extreme ones.
# - **Median**: the middle value once sorted. Half the data is below it. Ignores how extreme the extremes are.
# - **Mode**: the most frequent value; the only one that makes sense for categories.

# %%
print(f"mean   = {price.mean():.1f}")
print(f"median = {price.median():.1f}")
print("mode of bedrooms:", housing["bedrooms"].mode().tolist(), "| mode of contract:", churn["contract"].str.lower().mode().tolist())

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.hist(price, bins=80, color="lightsteelblue")
ax.axvline(price.mean(), color="tab:red", label=f"mean {price.mean():.0f}")
ax.axvline(price.median(), color="tab:green", label=f"median {price.median():.0f}")
ax.set_xlabel("price (thousand $)"); ax.set_ylabel("number of houses"); ax.set_title("House prices are right-skewed: mean > median")
ax.legend()
plt.show()

# %% [markdown]
# *What just happened?* A long right tail (a few expensive houses) pulls the mean above the median. When mean and median disagree by a lot, the distribution is skewed or has outliers;
# report both, or use the median. Income, prices, session durations and order values are classic right-skewed variables.

# %% [markdown]
# ## 2. Spread: how much do values vary?
#
# Two cities can have the same average temperature and very different weather. Spread measures how far values sit from the centre.
#
# | Measure | What it is | Robust? |
# |---|---|---|
# | range | `max - min` | no: two values decide it |
# | variance / std | average squared distance from the mean (std = square root, same units as data) | no |
# | IQR | 75th percentile minus 25th percentile: the width of the middle half | yes |
# | MAD | median of `abs(x - median)` | yes |

# %%
x = price.dropna()
q1, q3 = x.quantile([0.25, 0.75])
mad = (x - x.median()).abs().median()
print(f"range = {x.max() - x.min():.0f}   std = {x.std():.1f}   IQR = {q3 - q1:.1f}   MAD = {mad:.1f}")
print(f"std predicted from IQR if normal: {0.7413 * (q3 - q1):.1f}   from MAD if normal: {1.4826 * mad:.1f}")

# %% [markdown]
# For normal data you would expect `std ~ 0.74 x IQR ~ 1.48 x MAD`; the cell above computes what that predicts. The actual std is noticeably larger: extreme rows inflate it, and a mismatch like this is a sign of a non-normal shape.
#
# **A classic trap: `n` vs `n-1`.** The *population* variance divides by `n`; the *sample* variance divides by `n-1` (Bessel's correction) because a sample's values sit closer to their own mean than to
# the true mean, so dividing by `n` underestimates the spread. pandas defaults to `n-1` (`ddof=1`), NumPy to `n` (`ddof=0`).

# %%
small = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
print("np.std (ddof=0)    :", small.std())
print("np.std (ddof=1)    :", small.std(ddof=1))
print("pandas .std()      :", pd.Series(small).std())

# check that ddof=1 is the unbiased choice: average the estimates over many small samples from a population with std 10
rng = np.random.default_rng(42)
draws = rng.normal(0, 10, size=(200_000, 5))
print("true variance = 100 | mean of ddof=0 estimates:", round(draws.var(axis=1, ddof=0).mean(), 1),
      "| mean of ddof=1 estimates:", round(draws.var(axis=1, ddof=1).mean(), 1))

# %% [markdown]
# With samples of size 5, dividing by `n` is off by about 20% on average (expected 100 x 4/5 = 80); `n-1` fixes it. With thousands of rows the difference is negligible, but the default matters when you compare NumPy and pandas output.

# %% [markdown]
# ## 3. Shape: skewness, and the log transform
#
# **Skewness** measures asymmetry: 0 is symmetric, positive means a long right tail. **Kurtosis** measures tail heaviness. We will use skewness.
# Right-skewed positive data often becomes roughly symmetric on a log scale, because multiplicative effects (a house is 20% bigger, therefore about 20% pricier) become additive.

# %%
pos = price[price > 0]
print(f"skewness of price       : {stats.skew(pos):.2f}")
print(f"skewness of log(price)  : {stats.skew(np.log(pos)):.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].hist(pos, bins=60, color="lightsteelblue"); axes[0].set_xlabel("price (thousand $)"); axes[0].set_ylabel("houses"); axes[0].set_title("Raw price")
axes[1].hist(np.log(pos), bins=60, color="lightsteelblue"); axes[1].set_xlabel("log(price in thousand $)"); axes[1].set_ylabel("houses"); axes[1].set_title("Log price: far less skewed (cheap tail now on the left)")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4. Robust statistics: summaries that survive bad data
#
# We contaminate the data on purpose and see which summaries move. Start with a clean copy (prices between 10k and 1M), then inject 30 absurd values.

# %%
clean = price[(price > 10) & (price < 1000)].to_numpy()
dirty = np.concatenate([clean, np.full(30, 50_000.0)])       # 30 entries with a data-entry error (an extra zero or more)

def summarise(a):
    return pd.Series({
        "mean": a.mean(),
        "median": np.median(a),
        "10% trimmed mean": stats.trim_mean(a, 0.10),
        "std": a.std(ddof=1),
        "IQR": np.subtract(*np.percentile(a, [75, 25])),
        "MAD": np.median(np.abs(a - np.median(a))),
    })

comparison = pd.DataFrame({"clean": summarise(clean), "with 30 bad rows": summarise(dirty)})
comparison["change %"] = (comparison["with 30 bad rows"] / comparison["clean"] - 1) * 100
comparison.round(1)

# %% [markdown]
# *What just happened?* About 1% of the rows were bad. The mean and the std exploded; the median, trimmed mean, IQR and MAD barely moved. The **breakdown point** of a statistic is the fraction of
# contamination it can absorb: 0% for the mean, 25% for the IQR, 50% for the median. The **trimmed mean** discards the top and bottom 10% before averaging: it keeps the efficiency of averaging while being
# safe against modest contamination. That is not a licence to delete data you dislike: always *look* at the outliers, because they are sometimes the most valuable rows (fraud, a viral product).

# %% [markdown]
# ## 5. Quantiles and outlier rules
#
# The `q`-th **quantile** is the value below which a fraction `q` of the data lies. Five of them make up the "five number summary": min, Q1, median, Q3, max. A box plot draws exactly that.
#
# Two common outlier rules:
# - **IQR rule (Tukey fences):** flag values below `Q1 - 1.5*IQR` or above `Q3 + 1.5*IQR`.
# - **z-score rule:** flag values with `abs(x - mean) / std > 3`.

# %%
def iqr_flags(a, k=1.5):
    a = pd.Series(a); q1, q3 = a.quantile([0.25, 0.75]); iqr = q3 - q1
    return (a < q1 - k * iqr) | (a > q3 + k * iqr)

def z_flags(a, thresh=3):
    a = pd.Series(a); return ((a - a.mean()) / a.std()).abs() > thresh

for name, a in [("clean prices", clean), ("prices + 30 bad rows", dirty)]:
    print(f"{name:22s} IQR rule flags {int(iqr_flags(a).sum()):4d}   z-score rule flags {int(z_flags(a).sum()):4d}")

# %% [markdown]
# With 1% contamination the z-score rule still catches the 30 bad rows (but notice it no longer flags any of the genuinely expensive houses it flagged before, because the inflated std raised
# its fence). With heavier contamination it fails outright: this is **masking**, where the outliers inflate the mean and std so much that they no longer look extreme.

# %%
heavy = np.concatenate([clean, np.full(450, 3000.0)])          # about 13% bad rows this time
z_hit = z_flags(heavy).to_numpy()[-450:].sum()
iqr_hit = iqr_flags(heavy).to_numpy()[-450:].sum()
mad_ = np.median(np.abs(heavy - np.median(heavy)))
rz_hit = (np.abs(0.6745 * (heavy - np.median(heavy)) / mad_) > 3.5)[-450:].sum()
print(f"bad rows caught out of 450 -> z-score: {z_hit}, IQR rule: {iqr_hit}, robust z: {rz_hit}")

# %% [markdown]
# The z-score rule misses them all while the quartile-based rules still work (their breakdown points are 25% and 50%). A robust z-score uses `0.6745 * (x - median) / MAD` instead of the mean and std. Back on the 1%-contaminated data:

# %%
robust_z = 0.6745 * (dirty - np.median(dirty)) / np.median(np.abs(dirty - np.median(dirty)))
flag_rz = np.abs(robust_z) > 3.5
print(f"robust z-score (>3.5) flags {int(flag_rz.sum())} rows, of which {int(flag_rz[-30:].sum())} are the 30 injected bad rows")
print("the rest are real expensive houses in the long tail; the IQR rule on skewed CLEAN prices also flags", int(iqr_flags(clean).sum()), "of them")

# %% [markdown]
# So an outlier *rule* only produces candidates. On right-skewed data apply the rule to the log scale, then ask whether each flagged row is an error, a rare-but-real case, or a different population.

# %% [markdown]
# ## 6. Summaries by group: the workhorse of analysis
# Add the grouping and most analytics questions appear: how do churners differ from stayers?

# %%
churn["contract"] = churn["contract"].str.lower()
summary = churn.groupby("churned")[["tenure_months", "monthly_charges", "support_calls"]].agg(["mean", "median", "std"])
summary.round(2)

# %% [markdown]
# Compare mean and median within each column: for `tenure_months` the mean sits well above the median (right-skewed), for churners' `monthly_charges` the mean sits below the median (left-skewed). Numbers alone hide this, so look at a picture. A box plot compares medians, IQRs and outliers per group at a glance:

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, col, unit in zip(axes, ["tenure_months", "monthly_charges", "support_calls"], ["months", "$ / month", "calls"]):
    ax.boxplot([churn.loc[churn["churned"] == 0, col].dropna(), churn.loc[churn["churned"] == 1, col].dropna()], tick_labels=["stayed", "churned"])
    ax.set_ylabel(f"{col} ({unit})"); ax.set_title(col)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Common mistakes
#
# 1. **Reporting only the mean** of a skewed variable. Show the median and a histogram too.
# 2. **`np.std` vs `pd.Series.std` confusion**: they default to different `ddof`. State which you used.
# 3. **Deleting every point beyond 1.5 IQR** automatically. On skewed data that removes real, valuable rows.
# 4. **Using z-scores to find outliers in data with outliers** (masking) or on a variable that is not roughly bell-shaped.
# 5. **Averaging averages.** The mean of group means is not the overall mean unless the groups have equal size.

# %%
g = pd.DataFrame({"group": ["A"] * 90 + ["B"] * 10, "value": [10.0] * 90 + [100.0] * 10})
print("mean of the two group means:", g.groupby("group")["value"].mean().mean(), "| true overall mean:", g["value"].mean())

# %% [markdown]
# ## Interview corner
#
# **1. Mean vs median: when do you use which?**
# <details><summary>Answer</summary>
# Use the mean for roughly symmetric data without outliers, or when totals matter (revenue = mean x count). Use the median for skewed data or when you want the "typical" case. Reporting both signals the skew.
# </details>
#
# **2. Why divide by n-1 for a sample variance?**
# <details><summary>Answer</summary>
# Sample values sit closer to the sample mean than to the true mean, so dividing by n underestimates the variance. Dividing by n-1 makes the estimator unbiased for the population variance.
# </details>
#
# **3. How do you detect outliers and what do you do with them?**
# <details><summary>Answer</summary>
# IQR fences, robust z-scores (median/MAD), or model-based methods. Then investigate: data error (fix or remove), real but rare (keep, maybe use robust methods or a log transform), or another population (model separately). Never delete silently.
# </details>
#
# **4. What is a robust statistic?**
# <details><summary>Answer</summary>
# One that changes little when a fraction of the data is arbitrary. The median (breakdown point 50%) and the MAD are robust; the mean and standard deviation (breakdown point 0%) are not.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Typical monthly charge for churners and stayers
# Build `charge_summary`: a DataFrame indexed by `churned` (0 and 1) with two columns, `median` and `iqr`, for `monthly_charges`.
# *Hint: `groupby("churned")["monthly_charges"].agg(...)` with a lambda for the IQR, or two separate aggregations.*

# %% tags=["starter"]
charge_summary = ...

# %%
assert list(charge_summary.columns) == ["median", "iqr"], "columns must be ['median', 'iqr']"
for k in (0, 1):
    v = churn.loc[churn["churned"] == k, "monthly_charges"]
    assert abs(charge_summary.loc[k, "median"] - v.median()) < 1e-9, f"wrong median for churned={k}"
    assert abs(charge_summary.loc[k, "iqr"] - (v.quantile(0.75) - v.quantile(0.25))) < 1e-9, f"wrong IQR for churned={k}"
print("Exercise 1 passed")
print(charge_summary.round(2))

# %% tags=["solution"]
charge_summary = churn.groupby("churned")["monthly_charges"].agg(
    median="median",
    iqr=lambda s: s.quantile(0.75) - s.quantile(0.25),
)

# %% [markdown]
# ### Exercise 2: How many houses look like outliers?
# Using the raw `price` (thousand dollars, NaN-free here) count how many houses the **IQR rule (k = 1.5)** flags on the raw scale (`n_out_raw`) and on the **log scale**
# (`n_out_log`, apply the rule to `np.log` of the *positive* prices only; also store how many prices were not positive as `n_nonpositive`).
# Then set `log_flags_fewer` to whether the log scale flags fewer than the raw scale.

# %% tags=["starter"]
n_out_raw = ...
n_out_log = ...
n_nonpositive = ...
log_flags_fewer = ...

# %%
p_ = price.dropna()
qa, qb = p_.quantile([0.25, 0.75])
exp_raw = int(((p_ < qa - 1.5 * (qb - qa)) | (p_ > qb + 1.5 * (qb - qa))).sum())
lp = np.log(p_[p_ > 0])
qa, qb = lp.quantile([0.25, 0.75])
exp_log = int(((lp < qa - 1.5 * (qb - qa)) | (lp > qb + 1.5 * (qb - qa))).sum())
assert n_out_raw == exp_raw, f"raw-scale IQR rule should flag {exp_raw}, you have {n_out_raw}"
assert n_out_log == exp_log, f"log-scale IQR rule should flag {exp_log}, you have {n_out_log}"
assert n_nonpositive == int((p_ <= 0).sum()), "count prices <= 0"
assert bool(log_flags_fewer) == (exp_log < exp_raw), "compare the two counts"
print(f"Exercise 2 passed: raw {n_out_raw}, log {n_out_log}, non-positive {n_nonpositive}")

# %% tags=["solution"]
def _iqr_count(s):
    a, b = s.quantile([0.25, 0.75])
    return int(((s < a - 1.5 * (b - a)) | (s > b + 1.5 * (b - a))).sum())

_p = price.dropna()
n_out_raw = _iqr_count(_p)
n_out_log = _iqr_count(np.log(_p[_p > 0]))
n_nonpositive = int((_p <= 0).sum())
log_flags_fewer = bool(n_out_log < n_out_raw)

# %% [markdown]
# ### Exercise 3: How much does the mean move when the top 1% is removed?
# Drop the highest 1% of `clean` prices (the values above the 99th percentile). Store in `mean_shift_pct` the percentage change of the **mean** and in `median_shift_pct` the
# percentage change of the **median** (relative to their original values, absolute value). Store `median_more_stable` as a bool.

# %% tags=["starter"]
mean_shift_pct = ...
median_shift_pct = ...
median_more_stable = ...

# %%
c_ = pd.Series(clean)
t_ = c_[c_ <= c_.quantile(0.99)]
e_mean = abs(t_.mean() / c_.mean() - 1) * 100
e_med = abs(t_.median() / c_.median() - 1) * 100
assert abs(mean_shift_pct - e_mean) < 1e-9, f"expected {e_mean:.3f}"
assert abs(median_shift_pct - e_med) < 1e-9, f"expected {e_med:.3f}"
assert bool(median_more_stable) == (e_med < e_mean)
print(f"Exercise 3 passed: mean moved {mean_shift_pct:.2f}%, median {median_shift_pct:.2f}%")

# %% tags=["solution"]
_c = pd.Series(clean)
_t = _c[_c <= _c.quantile(0.99)]
mean_shift_pct = float(abs(_t.mean() / _c.mean() - 1) * 100)
median_shift_pct = float(abs(_t.median() / _c.median() - 1) * 100)
median_more_stable = bool(median_shift_pct < mean_shift_pct)

# %% [markdown]
# ### Exercise 4 (open-ended): Profile one variable
# Pick `total_charges` from `churn` (it has missing values). Report in a markdown cell its mean, median, std, IQR, skewness and the number of missing rows; state whether you would
# describe it as symmetric or skewed and which centre and spread you would put on a dashboard. Add one plot.

# %% [markdown]
# ## Recap
# - The mean is pulled by extremes; the median is not. When they disagree, the data is skewed or has outliers.
# - Std and variance are outlier-sensitive; IQR and MAD are robust. `ddof=1` (pandas) vs `ddof=0` (NumPy) is a real difference for small samples.
# - Log transforms tame right-skewed positive data.
# - Outlier rules (IQR, z-score, robust z) produce *candidates*; investigate before deleting. z-scores can be masked by the outliers they hunt.
#
# **Go deeper:** the SciPy docs for `scipy.stats.describe`, `trim_mean`, `median_abs_deviation`; the pandas user guide section "Computational tools".
#
# **Next:** [Probability and simulation](02_probability_and_simulation.ipynb)

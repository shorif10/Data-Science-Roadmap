# %% [markdown]
# # Seaborn: Relationships and Categorical Plots
#
# **Why this matters.** After "what does each variable look like?" comes "how do they relate?" and "which groups differ?": does price rise with area, which contract type churns most, is the gap real or noise?
# Seaborn's scatter, regression and bar plots answer these, and its bar plots quietly compute **confidence intervals**. If you do not know what those error bars mean, you will misread (or mis-present) your own chart.
#
# **You will learn**
# - show relationships with `scatterplot` (colour, size, style as extra dimensions) and fit lines with `regplot`/`lmplot`
# - plot group summaries with `barplot`, `pointplot`, `countplot` and `catplot`
# - explain **exactly** what the error bars on a seaborn bar mean (and verify them numerically)
# - spot when a straight line is the wrong model
#
# **Prerequisites:** [Distributions](01_distributions.ipynb).
#
# > Data used: `datasets/housing.csv` and `datasets/churn.csv` (synthetic; see `docs/DATASETS.md`). Local CSVs only.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams["figure.max_open_warning"] = 0
sns.set_theme(style="whitegrid")
DATA = Path("../datasets")

housing = pd.read_csv(DATA / "housing.csv")
homes = housing[(housing.price / housing.area_sqft).between(20, 500)].copy()
churn = pd.read_csv(DATA / "churn.csv")
churn["contract"] = churn.contract.str.lower().map({"month-to-month": "Month-to-month", "one year": "One year", "two year": "Two year"})
churn["senior"] = np.where(churn.senior_citizen == 1, "senior", "non-senior")
print(len(homes), "homes;", len(churn), "customers")

# %% [markdown]
# ## 1. Scatter plots with more dimensions
# Two numbers on x and y; add up to three more channels: `hue` (colour), `size`, `style` (marker shape). Use them sparingly: each extra encoding makes the chart harder to read.

# %%
fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
sns.scatterplot(data=homes, x="area_sqft", y="price", hue="neighborhood", alpha=0.45, s=14, ax=ax)
ax.set(title="Price vs area by neighbourhood", xlabel="Living area (sq ft)", ylabel="Price (USD)")
ax.legend(title="Neighbourhood", fontsize=8, ncols=2)
plt.show()

# %% [markdown]
# *What just happened?* Colour adds the neighbourhood. Notice the clouds overlap heavily: six categorical colours are near the limit of what people can separate, so when the colour story matters use small multiples (notebook 03) instead.
# Overplotting is handled by `alpha` and small `s`, as in Matplotlib.
#
# **Check the relationship with a number as well as a picture:**

# %%
print(f"correlation area-price overall: {homes.area_sqft.corr(homes.price):.2f}")
print(homes.groupby("neighborhood").apply(lambda d: d.area_sqft.corr(d.price), include_groups=False).round(2).to_dict())

# %% [markdown]
# ## 2. Fitting a line: `regplot` and `lmplot`
# `regplot` adds a fitted regression line and a shaded band: the **95% confidence interval for the line** (estimated by bootstrap), not the spread of the points. `lmplot` is the figure-level version and can facet.

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
sns.regplot(data=homes, x="area_sqft", y="price", scatter_kws={"s": 6, "alpha": 0.25}, line_kws={"color": "crimson"}, ax=axs[0])
axs[0].set(title="Straight-line fit (order=1)", xlabel="Living area (sq ft)", ylabel="Price (USD)")
sns.regplot(data=homes, x="area_sqft", y="price", order=2, scatter_kws={"s": 6, "alpha": 0.25}, line_kws={"color": "crimson"}, ax=axs[1])
axs[1].set(title="Quadratic fit (order=2)", xlabel="Living area (sq ft)", ylabel="Price (USD)")
plt.show()

# %% [markdown]
# The data curve upward (price grows faster than area), so the straight line misses systematically (see the residuals below); the quadratic follows the cloud more closely. A fit line is a **model**, and the
# confidence band only says how well the *line* is pinned down, not that a line is the right shape. Look at the residuals to be sure:

# %%
slope, intercept = np.polyfit(homes.area_sqft, homes.price, 1)
resid = homes.price - (slope * homes.area_sqft + intercept)
bins = pd.qcut(homes.area_sqft, 5)
print(f"straight-line slope: {slope:,.0f} USD per extra sq ft")
print("mean residual by area quintile (should hover near 0 if a line fits):")
print(resid.groupby(bins, observed=True).mean().round(0).to_string())

# %% [markdown]
# Residuals (actual minus fitted) that are not scattered around 0 but differ systematically by area band (look at the smallest-area quintile versus the middle ones) are the signature of a wrong functional form. The slope is also an *average* over all homes: read "USD per sq ft" as a description of this sample, not a causal price list.
# `lmplot` does the same per group in one call:

# %%
g = sns.lmplot(data=homes, x="area_sqft", y="price", col="neighborhood", col_wrap=3, height=2.6, scatter_kws={"s": 5, "alpha": 0.25}, line_kws={"color": "crimson"})
g.set_axis_labels("Living area (sq ft)", "Price (USD)")
g.set_titles("{col_name}")
plt.show()

# %% [markdown]
# ## 3. Bar plots that aggregate, and what the error bars mean
# Matplotlib's `bar` draws heights you give it. Seaborn's `barplot` takes **raw rows**, groups them, computes a **mean** per group, and draws an **error bar**. By default that error bar is a **95% confidence interval of the mean**, computed by
# bootstrapping (resampling the rows many times and looking at the spread of the recomputed mean). Let us see it for the churn rate per contract (the mean of a 0/1 column is a proportion).

# %%
fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
sns.barplot(data=churn, x="contract", y="churned", order=["Month-to-month", "One year", "Two year"], errorbar=("ci", 95), seed=42, ax=ax)
ax.set(title="Churn rate by contract type (bars: mean, whiskers: 95% CI)", xlabel="Contract type", ylabel="Churn rate (fraction of customers)")
plt.show()

stats = churn.groupby("contract").churned.agg(["mean", "size"])
stats["normal_CI_low"] = stats["mean"] - 1.96 * np.sqrt(stats["mean"] * (1 - stats["mean"]) / stats["size"])
stats["normal_CI_high"] = stats["mean"] + 1.96 * np.sqrt(stats["mean"] * (1 - stats["mean"]) / stats["size"])
drawn = pd.DataFrame([ln.get_ydata() for ln in ax.lines], index=["Month-to-month", "One year", "Two year"], columns=["drawn_low", "drawn_high"])
pd.concat([stats.round(3), drawn.round(3)], axis=1)

# %% [markdown]
# We cross-checked against an independent formula (normal approximation for a proportion, `p +/- 1.96 sqrt(p(1-p)/n)`): the drawn whiskers match it closely, because with thousands of rows the bootstrap and the formula agree.
# **What the whiskers are NOT:** they are not the spread of individual customers (half of month-to-month customers churn, so individuals are clearly 0 or 1, not 0.5 +/- 0.02). They describe how well the **mean** is estimated.
# That is why they shrink as `n` grows, and why `errorbar="sd"` (standard deviation of the data) looks so different:

# %%
fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), layout="constrained", sharey=True)
for ax, eb, name in zip(axs, (("ci", 95), "se", "sd"), ("95% CI of the mean", "standard error", "standard deviation of the data")):
    sns.barplot(data=churn, x="contract", y="monthly_charges", order=["Month-to-month", "One year", "Two year"], errorbar=eb, seed=42, ax=ax)
    ax.set(title=name, xlabel="Contract type", ylabel="Monthly charges (USD)")
plt.show()

# %% [markdown]
# Same data, three very different whiskers: the CI and SE are tiny (we know the *mean* precisely) while the SD is wide (individual customers vary a lot). **Always say which one you show.** Two more honest caveats:
# - Non-overlapping 95% CIs suggest a real difference; overlapping CIs do **not** prove there is none (that needs a proper test or a CI for the *difference*).
# - A CI describes sampling uncertainty only. It says nothing about bias in how the data were collected.
#
# Also, a bar chart of a **mean** hides the distribution; if shape matters (monthly charges here are about the same across contracts), use box/violin/ECDF from the previous notebook.

# %% [markdown]
# ## 4. Counts, points and `catplot`
# `countplot` counts rows per category (a histogram for categories). `pointplot` shows the same estimate and CI as a dot, which is better for comparing **many** groups or two hues (less ink than bars, and it need not start at 0 because it does not encode value by length).
# `catplot` is the figure-level wrapper with `col=`/`row=`.

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
sns.countplot(data=churn, x="contract", order=["Month-to-month", "One year", "Two year"], ax=axs[0])
axs[0].set(title="Customers by contract type", xlabel="Contract type", ylabel="Customers (count)")
sns.pointplot(data=churn, x="support_calls", y="churned", errorbar=("ci", 95), seed=42, ax=axs[1])
axs[1].set(title="Churn rate by number of support calls", xlabel="Support calls (count)", ylabel="Churn rate (fraction)")
plt.show()
print(churn.groupby("support_calls").churned.size().to_dict())

# %% [markdown]
# Look at the right panel's whiskers: they widen sharply at high call counts because very few customers make 7-8 calls (see the printed group sizes). The wide CI is the chart honestly telling you "not enough data here", which a bare bar would hide.
# A **count** chart shows volume, a **rate** chart shows risk: for a retention decision you usually need both.

# %%
g = sns.catplot(data=churn, x="contract", y="churned", col="senior", kind="bar", order=["Month-to-month", "One year", "Two year"], errorbar=("ci", 95), seed=42, height=3.2, aspect=1.1)
g.set_axis_labels("Contract type", "Churn rate (fraction)")
g.set_titles("{col_name}")
plt.show()

# %% [markdown]
# ## Common mistakes
#
# 1. **Reading bar whiskers as the spread of the data.** The default is a CI of the **mean**; use `errorbar="sd"` if you want spread, and label it.
# 2. **Concluding "no difference" because CIs overlap**, or "significant" because they do not, without a test.
# 3. **A straight line through curved data**, then quoting the slope as a universal rate. Check residuals or try `order=2`/`lowess=True`.
# 4. **Ignoring group size**: a rate over 4 customers looks as tall as a rate over 4,000. Print `n` or read the CI width.
# 5. **Too many hues** (more than about 5 colours) or redundant encodings.
# 6. **Bars for a numeric-x relationship** (support calls 0-8 is ordered: points/lines fit; purely nominal categories fit bars).
# 7. **Forgetting bootstrap randomness**: CIs move slightly between runs unless you set `seed=`.

# %%
# Mistake 4 in numbers
small = churn.groupby("support_calls").churned.agg(["mean", "size"]).round(2)
print(small.tail(3).to_string())

# %% [markdown]
# ## Interview corner
#
# **1. What do the error bars in a seaborn barplot represent by default?**
# <details><summary>Answer</summary>
# A 95% bootstrap confidence interval of the mean of each group (<code>errorbar=("ci", 95)</code>): the uncertainty of the estimated mean, not the spread of individual values. <code>errorbar="sd"</code> shows the standard deviation of the data and <code>"se"</code> the standard error.
# </details>
#
# **2. Two groups have overlapping 95% confidence intervals. Are they not different?**
# <details><summary>Answer</summary>
# Not necessarily. Overlapping intervals of the two *means* do not prove equality; the right check is a test or an interval for the difference. Non-overlap is a fairly strong hint of a difference, but it is not a formal test.
# </details>
#
# **3. <code>regplot</code> draws a line and a band. What is the band?**
# <details><summary>Answer</summary>
# A 95% confidence interval for the fitted regression line (how much the line might move if you resampled), not a prediction interval for individual points. It does not validate the model's shape.
# </details>
#
# **4. When would you prefer <code>pointplot</code> over <code>barplot</code>?**
# <details><summary>Answer</summary>
# Many groups, two grouping variables, or ordered x-values where you care about differences between estimates; bars encode value by length and so need a zero baseline and more ink.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Colour the scatter
# Draw a scatter of `area_sqft` (x) vs `price` (y) for `homes` on `fig1, ax1` using `sns.scatterplot`, coloured by `neighborhood`, with `alpha <= 0.5`. Title and axis labels with units. The legend must list all neighbourhoods.

# %% tags=["starter"]
fig1, ax1 = plt.subplots()
# your code here

# %%
assert len(ax1.collections) >= 1 and ax1.collections[0].get_offsets().shape[0] == len(homes), "plot every plausible listing in one scatterplot call"
assert np.allclose(np.sort(ax1.collections[0].get_offsets()[:, 0]), np.sort(homes.area_sqft.values)), "x should be area_sqft"
assert ax1.collections[0].get_alpha() is not None and ax1.collections[0].get_alpha() <= 0.5
labels = {t.get_text() for t in ax1.get_legend().get_texts()}
assert set(homes.neighborhood.unique()) <= labels, f"legend should include all neighbourhoods; has {labels}"
assert ax1.get_title() and "sq ft" in ax1.get_xlabel().lower() and "usd" in ax1.get_ylabel().lower()
print("Exercise 1 passed")

# %% tags=["solution"]
fig1, ax1 = plt.subplots(figsize=(7, 4.2), layout="constrained")
sns.scatterplot(data=homes, x="area_sqft", y="price", hue="neighborhood", alpha=0.4, s=14, ax=ax1)
ax1.set(title="Price vs area by neighbourhood", xlabel="Living area (sq ft)", ylabel="Price (USD)")

# %% [markdown]
# ### Exercise 2: Which contract churns most, and how sure are we?
# Draw a seaborn `barplot` of churn rate (`churned`) by `contract` with the **95% CI** and `seed=42` on `fig2, ax2` (order: Month-to-month, One year, Two year). Then set `worst_contract` to the contract with the highest churn rate
# and `ci_width` to the **width of the whiskers** (`high - low`) drawn for that contract, taken from the figure (`ax2.lines`).

# %% tags=["starter"]
fig2, ax2 = plt.subplots()
worst_contract = ...
ci_width = ...

# %%
order = ["Month-to-month", "One year", "Two year"]
rate = churn.groupby("contract").churned.mean()
assert len(ax2.patches) == 3
assert np.allclose([p.get_height() for p in ax2.patches], rate.loc[order].values), "bar heights should be the churn rate per contract in the given order"
assert len(ax2.lines) == 3, "three error bars expected (a 95% CI per bar)"
assert worst_contract == rate.idxmax()
i = order.index(worst_contract)
y = ax2.lines[i].get_ydata()
assert abs(ci_width - (max(y) - min(y))) < 1e-9, "ci_width must be taken from the drawn whisker of that bar"
p, n = rate[worst_contract], int((churn.contract == worst_contract).sum())
assert abs(ci_width - 2 * 1.96 * np.sqrt(p * (1 - p) / n)) < 0.01, "sanity: width should be close to the normal-approximation CI width"
assert ax2.get_title() and ax2.get_xlabel() and "churn" in ax2.get_ylabel().lower()
print(f"Exercise 2 passed ({worst_contract}: rate {p:.3f}, CI width {ci_width:.3f})")

# %% tags=["solution"]
order = ["Month-to-month", "One year", "Two year"]
fig2, ax2 = plt.subplots(figsize=(6, 3.6), layout="constrained")
sns.barplot(data=churn, x="contract", y="churned", order=order, errorbar=("ci", 95), seed=42, ax=ax2)
ax2.set(title="Churn rate by contract type", xlabel="Contract type", ylabel="Churn rate (fraction of customers)")
worst_contract = churn.groupby("contract").churned.mean().idxmax()
y = ax2.lines[order.index(worst_contract)].get_ydata()
ci_width = float(max(y) - min(y))

# %% [markdown]
# ### Exercise 3: A slope you can defend
# Use `sns.regplot` (straight line, `order=1`) for `distance_to_center_km` (x) vs `price` (y) on `homes`, on `fig3, ax3`. Then set `slope_per_km` to the slope of the fitted line **as drawn** (from `ax3.lines[0]`), and `price_drop_per_km_positive` to `True` if
# the slope is negative (price falls with distance) else `False`.

# %% tags=["starter"]
fig3, ax3 = plt.subplots()
slope_per_km = ...
price_drop_per_km_positive = ...

# %%
xl, yl = ax3.lines[0].get_xdata(), ax3.lines[0].get_ydata()
assert len(ax3.lines) == 1 and len(ax3.collections) == 2, "regplot draws one line plus scatter points and a CI band"
assert ax3.collections[0].get_offsets().shape[0] == len(homes)
drawn_slope = (yl[-1] - yl[0]) / (xl[-1] - xl[0])
ref = np.polyfit(homes.distance_to_center_km, homes.price, 1)[0]
assert abs(drawn_slope - ref) < 1e-6 * max(1, abs(ref)), "the drawn line should be the least-squares line"
assert abs(slope_per_km - ref) < 1e-6 * max(1, abs(ref)), f"slope_per_km should be {ref:,.1f} (USD per km)"
assert price_drop_per_km_positive == (ref < 0)
assert ax3.get_title() and "km" in ax3.get_xlabel().lower() and "usd" in ax3.get_ylabel().lower()
print(f"Exercise 3 passed (slope {slope_per_km:,.0f} USD per km)")

# %% tags=["solution"]
fig3, ax3 = plt.subplots(figsize=(6.5, 4), layout="constrained")
sns.regplot(data=homes, x="distance_to_center_km", y="price", order=1, scatter_kws={"s": 6, "alpha": 0.25}, line_kws={"color": "crimson"}, ax=ax3)
ax3.set(title="Price vs distance from the city centre", xlabel="Distance to city centre (km)", ylabel="Price (USD)")
xl, yl = ax3.lines[0].get_xdata(), ax3.lines[0].get_ydata()
slope_per_km = float((yl[-1] - yl[0]) / (xl[-1] - xl[0]))
price_drop_per_km_positive = bool(slope_per_km < 0)

# %% [markdown]
# ### Exercise 4: Faceted churn rates
# Make `g4 = sns.catplot(...)`: a bar chart of churn rate by `contract` with **one panel per `autopay` value** (`col="autopay"`), 95% CI, `seed=42`, contracts in the usual order. Panel titles should show just the autopay value; axis labels with units.
# Then set `autopay_gap` to (overall churn rate of `autopay == "No"`) minus (overall churn rate of `autopay == "Yes"`).

# %% tags=["starter"]
g4 = ...
autopay_gap = ...

# %%
assert type(g4).__name__ == "FacetGrid" and g4.axes.shape == (1, 2), "two panels (autopay No / Yes)"
assert [a.get_title() for a in g4.axes.flat] == ["No", "Yes"], "titles should be 'No' and 'Yes' (g.set_titles('{col_name}'))"
order = ["Month-to-month", "One year", "Two year"]
for ax, val in zip(g4.axes.flat, ["No", "Yes"]):
    rates = churn[churn.autopay == val].groupby("contract").churned.mean().loc[order]
    assert np.allclose([p.get_height() for p in ax.patches], rates.values), f"panel {val}: wrong bar heights or order"
assert g4.axes[0, 0].get_xlabel() and g4.axes[0, 0].get_ylabel() and g4.axes[0, 1].get_xlabel(), "set axis labels with g.set_axis_labels"
g_ = churn.groupby("autopay").churned.mean()
assert abs(autopay_gap - (g_["No"] - g_["Yes"])) < 1e-9, "autopay_gap = churn(No) - churn(Yes)"
print(f"Exercise 4 passed (gap {autopay_gap:+.3f})")

# %% tags=["solution"]
g4 = sns.catplot(data=churn, x="contract", y="churned", col="autopay", col_order=["No", "Yes"], kind="bar",
                 order=["Month-to-month", "One year", "Two year"], errorbar=("ci", 95), seed=42, height=3.2, aspect=1.1)
g4.set_titles("{col_name}")
g4.set_axis_labels("Contract type", "Churn rate (fraction)")
g_ = churn.groupby("autopay").churned.mean()
autopay_gap = float(g_["No"] - g_["Yes"])

# %% [markdown]
# ### Exercise 5 (open-ended): Is the pattern real?
# Pick a numeric predictor and the `churned` outcome. Plot churn rate against it with a `pointplot` (bin it with `pd.cut` if continuous) and read the CI widths. In a few sentences: (a) what pattern do you see, (b) where is the evidence thin, and (c) what analysis would you run before telling the business?

# %% [markdown]
# ## Recap
# - `scatterplot` + `hue`/`size`/`style` adds dimensions (use sparingly); `regplot`/`lmplot` add fitted lines whose band is a CI for the **line**.
# - A fit line is a model: check residuals or try a curve before quoting a slope.
# - `barplot` aggregates raw rows to a mean with a **95% bootstrap CI by default**: uncertainty of the mean, not spread; choose `errorbar` deliberately and label it.
# - Overlapping CIs are not a test; small groups produce wide CIs (good, that is honest).
# - `countplot` for volume, `pointplot` for many groups/ordered x, `catplot` to facet.
#
# **Go deeper:** seaborn docs "Visualizing categorical data", "Statistical estimation and error bars" and "Visualizing statistical relationships" (seaborn.pydata.org).
#
# **Next:** [Matrix and grid plots](03_matrix_and_grid_plots.ipynb)

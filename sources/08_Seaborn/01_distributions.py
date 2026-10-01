# %% [markdown]
# # Seaborn: Looking at Distributions
#
# **Why this matters.** Before you model anything you need to know what your variables *look like*: where most values sit, how skewed they are, whether groups differ, whether two humps hide two populations.
# Seaborn sits on top of Matplotlib and turns these questions into one-liners that take a DataFrame and column names, with sensible statistics built in. It is what most analysts reach for during exploratory data analysis (EDA).
#
# **You will learn**
# - draw histograms, KDE curves, ECDFs, box and violin plots with `seaborn`, and pick the right one for your question
# - split a distribution by a category with `hue`, and avoid the traps (`common_norm`, overlapping curves)
# - tell **axes-level** functions (draw on an `Axes`) from **figure-level** functions (make a whole `FacetGrid`)
# - read an ECDF to answer "what fraction is below X?"
#
# **Prerequisites:** [Matplotlib core chart types](../07_Matplotlib/02_core_chart_types_and_when_to_use_them.ipynb). Seaborn returns Matplotlib objects, so everything you learned there still applies.
#
# > Data used: `datasets/housing.csv` (synthetic house listings, with 15 planted errors; the simple price-per-sqft filter below flags a few more, a reminder that such rules are judgement calls) and `datasets/churn.csv` (synthetic telecom customers; `contract` has inconsistent casing, cleaned below). We use local CSVs only: `seaborn.load_dataset` downloads data, so we never call it.

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
homes = housing[(housing.price / housing.area_sqft).between(20, 500)].copy()        # drop implausible price-per-sqft listings
print(f"housing: kept {len(homes)} of {len(housing)} rows")

churn = pd.read_csv(DATA / "churn.csv")
churn["contract"] = churn.contract.str.lower().map({"month-to-month": "Month-to-month", "one year": "One year", "two year": "Two year"})
churn["status"] = np.where(churn.churned == 1, "churned", "stayed")                 # text labels make better legends than 0/1
print(churn.contract.value_counts().to_dict())

# %% [markdown]
# ## 1. Two kinds of seaborn functions
#
# | | **Axes-level** | **Figure-level** |
# |---|---|---|
# | examples | `histplot`, `kdeplot`, `ecdfplot`, `boxplot`, `violinplot`, `scatterplot`, `barplot` | `displot`, `catplot`, `relplot`, `pairplot`, `lmplot` |
# | draws on | one Matplotlib `Axes` (you may pass `ax=`) | its **own** figure, returns a `FacetGrid` |
# | facets (`col=`, `row=`) | no | yes |
# | sizing | `plt.subplots(figsize=...)` | `height=`, `aspect=` |
#
# Use axes-level functions inside your own layouts (and for exact control); use figure-level ones for quick small multiples. Let us see the difference in the return type:

# %%
fig, ax = plt.subplots(figsize=(6, 3.2), layout="constrained")
r1 = sns.histplot(data=homes, x="price", ax=ax)
ax.set(title="Axes-level: histplot draws on OUR axes", xlabel="Price (USD)", ylabel="Listings (count)")
plt.show()
g = sns.displot(data=homes, x="price")
r2 = g
g.set(title="Figure-level: displot made its own figure", xlabel="Price (USD)", ylabel="Listings (count)")
plt.show()
print(type(r1).__name__, "vs", type(r2).__name__)

# %% [markdown]
# *What just happened?* `histplot` returned the Matplotlib `Axes` we gave it, so all your `ax.set(...)` knowledge applies. `displot` returned a `FacetGrid` object that owns the figure (`g.figure`) and the axes (`g.axes`).

# %% [markdown]
# ## 2. Histograms
# Same idea as Matplotlib's, with better defaults: a **bin rule** that adapts to the data, and a `stat` option for what the bar height means: `count`, `percent`, `density` (area = 1).

# %%
fig, axs = plt.subplots(1, 3, figsize=(12, 3.2), layout="constrained")
sns.histplot(data=homes, x="price", bins=30, ax=axs[0])
sns.histplot(data=homes, x="price", bins=30, stat="percent", ax=axs[1])
sns.histplot(data=homes, x="price", bins=30, stat="density", kde=True, ax=axs[2])
for ax, ylab in zip(axs, ("Listings (count)", "Share of listings (%)", "Density (1/USD)")):
    ax.set(xlabel="Price (USD)", ylabel=ylab)
axs[0].set_title("stat='count'")
axs[1].set_title("stat='percent'")
axs[2].set_title("stat='density' + KDE line")
plt.show()

# %% [markdown]
# Use `count` for one group; use `percent` or `density` when **comparing groups of different sizes**, otherwise the bigger group always looks "taller".
# Prices are right-skewed (a long right tail). A **log scale** spreads the body of the data and often pulls the tail in and makes such a shape more symmetric (compare the skewness numbers below; a log transform can even overshoot):

# %%
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
sns.histplot(data=homes, x="price", bins=30, ax=axs[0])
axs[0].set(title="Linear axis", xlabel="Price (USD)", ylabel="Listings (count)")
sns.histplot(data=homes, x="price", log_scale=True, bins=30, ax=axs[1])
axs[1].set(title="Log axis", xlabel="Price (USD, log scale)", ylabel="Listings (count)")
plt.show()
print(f"skew linear: {homes.price.skew():.2f}   skew of log(price): {np.log(homes.price).skew():.2f}")

# %% [markdown]
# ## 3. Comparing groups with `hue`
# `hue="status"` splits by category and colours each. The defaults can mislead: by default each group is normalised **together** (`common_norm=True`), so the bigger group dominates. Compare the two:

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained", sharex=True)
sns.histplot(data=churn, x="tenure_months", hue="status", stat="density", common_norm=True, element="step", ax=axs[0])
axs[0].set(title="common_norm=True (shares one total)", xlabel="Tenure (months)", ylabel="Density")
sns.histplot(data=churn, x="tenure_months", hue="status", stat="density", common_norm=False, element="step", ax=axs[1])
axs[1].set(title="common_norm=False (each group sums to 1)", xlabel="Tenure (months)", ylabel="Density")
plt.show()
print(churn.status.value_counts().to_dict())

# %% [markdown]
# When the groups have different sizes, **`common_norm=False`** answers "how is tenure distributed *within* churners versus *within* stayers?", which is the usual question. The clear message: churners cluster at low tenure.
# `element="step"` draws outlines so overlapping groups stay readable (the default filled bars hide each other).

# %% [markdown]
# ## 4. KDE: a smooth density curve
# A **kernel density estimate** replaces bars by a smooth curve: each point contributes a small bump and the bumps are added. It shows shape without bin edges. The key parameter is the bandwidth (`bw_adjust`):
# too small looks spiky, too large blurs real humps.

# %%
fig, axs = plt.subplots(1, 3, figsize=(12, 3), layout="constrained", sharey=True)
for ax, bw in zip(axs, (0.2, 1, 3)):
    sns.kdeplot(data=churn, x="monthly_charges", bw_adjust=bw, ax=ax, fill=True)
    ax.set(title=f"bw_adjust={bw}", xlabel="Monthly charges (USD)", ylabel="Density")
plt.show()
print(f"monthly_charges range in the data: {churn.monthly_charges.min():.1f} to {churn.monthly_charges.max():.1f}")

# %% [markdown]
# Caveats: a KDE can spill **beyond the data range** (look at the tails: there is density below the minimum), so for bounded variables use `cut=0` or a histogram; and it needs enough points, since a bump from 10 points is mostly an artefact of the bandwidth.

# %% [markdown]
# ## 5. ECDF: the honest distribution plot
# The **empirical cumulative distribution function** plots, for each value `x`, the fraction of data at or below `x`. No bins, no bandwidth, every point plotted: nothing to tune, and you can read **percentiles** directly.
# Medians and quantiles are where the curve crosses 0.5, 0.9 and so on; differences between groups show as horizontal shifts.

# %%
fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
sns.ecdfplot(data=churn, x="monthly_charges", hue="contract", ax=ax)
ax.axhline(0.5, color="gray", linestyle=":")
ax.set(title="ECDF of monthly charges by contract type", xlabel="Monthly charges (USD)", ylabel="Fraction of customers at or below")
plt.show()
med = churn.groupby("contract").monthly_charges.median().round(1).to_dict()
print("median monthly charge by contract:", med)

# %% [markdown]
# Read the dotted line: where each curve crosses 0.5 is that group's median, and it matches the printed numbers. Curves that overlap mean the groups have about the same distribution (here, contract type hardly changes the price distribution).
# Question answered by the ECDF directly: "what share of **Month-to-month** customers pay $50 or less?":

# %%
mm = churn.loc[churn.contract == "Month-to-month", "monthly_charges"]
print(f"share paying <= 50: {(mm <= 50).mean():.1%}  (the y-value of the Month-to-month curve at x = 50)")

# %% [markdown]
# ## 6. Box, violin and strip plots: distributions by category
# Put the category on x and the number on y. **Box**: five-number summary plus outliers. **Violin**: a mirrored KDE, shows shape. **Strip**: every point, jittered. Combining violin and strip (or box and strip) gives shape *and* evidence.

# %%
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained", sharey=True)
sns.boxplot(data=churn, x="status", y="tenure_months", ax=axs[0])
axs[0].set(title="Box", xlabel="Customer status", ylabel="Tenure (months)")
sns.violinplot(data=churn, x="status", y="tenure_months", cut=0, ax=axs[1])
axs[1].set(title="Violin (cut=0: stay inside the data range)", xlabel="Customer status", ylabel="Tenure (months)")
sns.stripplot(data=churn.sample(600, random_state=42), x="status", y="tenure_months", alpha=0.4, size=2.5, ax=axs[2])
axs[2].set(title="Strip (600-row sample)", xlabel="Customer status", ylabel="Tenure (months)")
plt.show()

# %% [markdown]
# All three show churners have shorter tenure. The box gives a clean median comparison; the violin reveals shape (for example a pile-up near zero); the strip shows how many points support each shape (here a sample, since 5,000 jittered points would be a blob).
# Verify what the box says with numbers:

# %%
print(churn.groupby("status").tenure_months.describe()[["count", "25%", "50%", "75%"]])

# %% [markdown]
# ## 7. Small multiples from a figure-level function
# `displot(..., col="contract")` creates one panel per category with shared axes. Good for quick comparisons across 2-6 groups.

# %%
g = sns.displot(data=churn, x="monthly_charges", col="contract", hue="status", stat="density", common_norm=False, element="step", height=3, aspect=1.1)
g.set_axis_labels("Monthly charges (USD)", "Density")
g.set_titles("{col_name}")
g.figure.suptitle("Monthly charges by contract and churn status", y=1.05)
plt.show()
print("grid shape:", g.axes.shape)

# %% [markdown]
# ## Common mistakes
#
# 1. **Comparing groups of different size with `stat="count"`**, or with `common_norm=True` by accident. Use `percent`/`density` and `common_norm=False`.
# 2. **Trusting one bin width or bandwidth.** Try several; prefer the ECDF when in doubt.
# 3. **Using a 0/1 column as `hue`**: seaborn treats it as numeric and picks a sequential light-to-dark palette instead of distinct category colours. Convert to text labels.
# 4. **Dirty categories** (the `contract` casing) produce extra groups. Clean first, then plot.
# 5. **Mixing axes-level and figure-level functions** (`ax=` is not accepted by `displot`; it makes its own figure).
# 6. **Leaving outliers/typos in** so the plot is unreadable; inspect, then decide openly.
# 7. **KDE on bounded data** (negative charges): it leaks past the limits. Use `cut=0`.

# %%
# Mistakes 3 and 4 in numbers: what the raw file would have given us
raw = pd.read_csv(DATA / "churn.csv")
print("groups the raw contract column would create:", raw.contract.nunique(), "instead of", churn.contract.nunique())
try:
    sns.displot(data=homes, x="price", ax=plt.gca())
except TypeError as e:
    print("TypeError:", str(e)[:90])
plt.close("all")

# %% [markdown]
# ## Interview corner
#
# **1. Histogram, KDE or ECDF: which do you use to compare two groups?**
# <details><summary>Answer</summary>
# ECDFs need no tuning and show all quantiles at once, so they compare groups most reliably. Overlaid normalised histograms (<code>element="step"</code>) or KDEs are more intuitive for a non-technical audience. Always normalise within groups when sizes differ.
# </details>
#
# **2. What is the difference between axes-level and figure-level functions in seaborn?**
# <details><summary>Answer</summary>
# Axes-level functions draw on one Matplotlib Axes and accept <code>ax=</code>; figure-level functions (<code>displot</code>, <code>catplot</code>, <code>relplot</code>) build their own figure through a <code>FacetGrid</code>, support <code>col</code>/<code>row</code> faceting and are sized with <code>height</code>/<code>aspect</code>.
# </details>
#
# **3. A variable is heavily right-skewed. How do you visualise and handle it?**
# <details><summary>Answer</summary>
# Plot on a log axis (<code>log_scale=True</code>) or transform it, compare mean with median, and report medians/quantiles rather than means. The transform choice should be stated on the chart.
# </details>
#
# **4. What does a violin plot add over a box plot, and what is the risk?**
# <details><summary>Answer</summary>
# It shows the full shape (multi-modality, skew). The risk is that the smoothed KDE can suggest data where there is none (tails past the min/max) or hide small samples, so check <code>cut=0</code> and show sample sizes.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Shares, not counts
# Draw a histogram of `homes.price` (in USD) with **25 bins** where the bar heights are the **percentage** of listings (they must add up to 100). Use `ax=` on your own axes `fig1, ax1`, and label the axes with units.
# *Hint: `sns.histplot(data=homes, x="price", bins=25, stat="percent", ax=ax1)`.*

# %% tags=["starter"]
fig1, ax1 = plt.subplots()
# your code here

# %%
assert len(ax1.patches) == 25, f"expected 25 bars, found {len(ax1.patches)}"
assert abs(sum(p.get_height() for p in ax1.patches) - 100) < 1e-6, "bar heights should be percentages adding up to 100 (stat='percent')"
lo, hi = min(p.get_x() for p in ax1.patches), max(p.get_x() + p.get_width() for p in ax1.patches)
assert lo <= homes.price.min() + 1e-6 and hi >= homes.price.max() - 1e-6, "the bars must cover the full price range (use homes, not housing)"
assert ax1.get_title() and "usd" in ax1.get_xlabel().lower() and ("%" in ax1.get_ylabel() or "percent" in ax1.get_ylabel().lower()), "title, x label with USD, y label mentioning % or percent"
print("Exercise 1 passed")

# %% tags=["solution"]
fig1, ax1 = plt.subplots(figsize=(6, 3.2), layout="constrained")
sns.histplot(data=homes, x="price", bins=25, stat="percent", ax=ax1)
ax1.set(title="Distribution of house prices", xlabel="Price (USD)", ylabel="Share of listings (%)")

# %% [markdown]
# ### Exercise 2: Price distribution by neighbourhood (ECDF)
# Draw ECDFs of `homes.price` with one curve per `neighborhood` on `fig2, ax2`. Then set `priciest` to the neighbourhood with the **highest median price** and `frac_below_250k` to the fraction of **Downtown** listings priced at or below 250,000.

# %% tags=["starter"]
fig2, ax2 = plt.subplots()
priciest = ...
frac_below_250k = ...

# %%
assert len(ax2.lines) == homes.neighborhood.nunique(), f"expected one ECDF line per neighbourhood ({homes.neighborhood.nunique()})"
assert ax2.get_title() and "usd" in ax2.get_xlabel().lower() and ax2.get_ylabel()
med = homes.groupby("neighborhood").price.median()
assert priciest == med.idxmax(), f"highest median price is in {med.idxmax()}"
d = homes[homes.neighborhood == "Downtown"].price
assert abs(frac_below_250k - (d <= 250_000).mean()) < 1e-9, "fraction of Downtown listings with price <= 250000"
print(f"Exercise 2 passed ({priciest}; {frac_below_250k:.1%} of Downtown listings <= 250k)")

# %% tags=["solution"]
fig2, ax2 = plt.subplots(figsize=(7, 3.8), layout="constrained")
sns.ecdfplot(data=homes, x="price", hue="neighborhood", ax=ax2)
ax2.set(title="House price ECDF by neighbourhood", xlabel="Price (USD)", ylabel="Fraction of listings at or below")
priciest = homes.groupby("neighborhood").price.median().idxmax()
frac_below_250k = float((homes.loc[homes.neighborhood == "Downtown", "price"] <= 250_000).mean())

# %% [markdown]
# ### Exercise 3: Does churn relate to monthly charges?
# Draw a **box plot** (`fig3, ax3`) of `monthly_charges` by `status` (x = status, y = monthly charges) with axis labels. Then set `median_gap` to `median(monthly_charges | churned) - median(monthly_charges | stayed)`
# (a number, in USD) and `higher_for` to `"churned"` or `"stayed"` depending on who pays more at the median (if equal, `"stayed"`).

# %% tags=["starter"]
fig3, ax3 = plt.subplots()
median_gap = ...
higher_for = ...

# %%
assert len(ax3.patches) == 2, "two boxes (stayed, churned)"
assert {t.get_text() for t in ax3.get_xticklabels()} == {"stayed", "churned"}, "x should be the status column"
assert ax3.get_title() and ax3.get_xlabel() and "usd" in ax3.get_ylabel().lower()
m = churn.groupby("status").monthly_charges.median()
assert abs(median_gap - (m["churned"] - m["stayed"])) < 1e-9, "median_gap = median(churned) - median(stayed)"
assert higher_for == ("churned" if m["churned"] > m["stayed"] else "stayed")
print(f"Exercise 3 passed (gap {median_gap:.2f} USD; higher for {higher_for})")

# %% tags=["solution"]
fig3, ax3 = plt.subplots(figsize=(5, 3.6), layout="constrained")
sns.boxplot(data=churn, x="status", y="monthly_charges", ax=ax3)
ax3.set(title="Monthly charges by customer status", xlabel="Customer status", ylabel="Monthly charges (USD)")
m = churn.groupby("status").monthly_charges.median()
median_gap = float(m["churned"] - m["stayed"])
higher_for = "churned" if m["churned"] > m["stayed"] else "stayed"

# %% [markdown]
# ### Exercise 4: Small multiples with a figure-level call
# Use `sns.displot` to make a grid `g4` with **one panel per contract type** (`col="contract"`) of the `tenure_months` histogram, `kind="hist"`, `stat="density"`. Set the panel titles to just the contract name and label the axes with units.
# *Hint: `g.set_titles("{col_name}")` and `g.set_axis_labels(x, y)`.*

# %% tags=["starter"]
g4 = ...

# %%
assert type(g4).__name__ == "FacetGrid", "displot returns a FacetGrid"
assert g4.axes.shape == (1, 3), f"one row, three columns expected; got {g4.axes.shape}"
assert [ax.get_title() for ax in g4.axes.flat] == sorted(churn.contract.unique()), "titles should be the contract names (alphabetical)"
assert all("month" in ax.get_xlabel().lower() for ax in g4.axes.flat), "x label should mention months"
for ax, name in zip(g4.axes.flat, sorted(churn.contract.unique())):
    n = int((churn.contract == name).sum())
    area = sum(p.get_height() * p.get_width() for p in ax.patches)
    assert abs(area - 1) < 1e-6, f"{name}: density bars should have total area 1 (stat='density')"
print("Exercise 4 passed")

# %% tags=["solution"]
g4 = sns.displot(data=churn, x="tenure_months", col="contract", kind="hist", stat="density", common_norm=False, col_order=sorted(churn.contract.unique()), height=3, aspect=1.1)
g4.set_titles("{col_name}")
g4.set_axis_labels("Tenure (months)", "Density")

# %% [markdown]
# ### Exercise 5 (open-ended): Report one distribution finding
# Choose any numeric column in `housing` or `churn` and any category. Make the plot you think best answers "does this number differ by group, and how?" (histogram, ECDF, box, or violin), justify the choice in one sentence,
# and write the finding with one supporting number (a median, a quantile, or an ECDF reading) computed in code.

# %% [markdown]
# ## Recap
# - **Histogram** for shape (try several bin counts, use `stat="percent"/"density"` when comparing), **KDE** for a smooth shape (mind bandwidth and tails), **ECDF** for bin-free comparison and reading percentiles, **box/violin/strip** for groups.
# - With `hue`, normalise within groups (`common_norm=False`) and use `element="step"` for overlap.
# - Log scale for right-skewed positive data.
# - Axes-level functions draw on your `ax`; figure-level functions (`displot`, `catplot`) make a `FacetGrid` and can facet with `col=`/`row=`.
# - Clean categories and convert 0/1 flags to labels before plotting.
#
# **Go deeper:** the seaborn documentation "Visualizing distributions of data" and "Overview of seaborn plotting functions" (seaborn.pydata.org).
#
# **Next:** [Relationships and categorical plots](02_relationships_and_categorical_plots.ipynb)

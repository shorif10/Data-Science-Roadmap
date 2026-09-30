# %% [markdown]
# # Seaborn: Matrix and Grid Plots
#
# **Why this matters.** With 8 numeric columns there are 28 pairs of variables; with 30 columns, 435. You cannot eyeball that many scatter plots one at a time.
# Correlation heatmaps, pair plots and small-multiple grids give you the whole landscape on one screen, so you can decide which relationships deserve a closer look (and which columns are redundant before modelling).
#
# **You will learn**
# - build a **correlation heatmap** that is readable: masked triangle, a diverging palette centred on zero, fixed colour limits
# - explain what correlation does and does not say (linear only, outlier-sensitive, not causal) and when to use Spearman
# - turn a pivot table into an annotated heatmap (rates by two categories)
# - use `pairplot`, `PairGrid` and `FacetGrid` for grids of plots, and know their cost
#
# **Prerequisites:** [Relationships and categorical plots](02_relationships_and_categorical_plots.ipynb).
#
# > Data used: `datasets/housing.csv` and `datasets/churn.csv` (synthetic; see `docs/DATASETS.md`).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams["figure.max_open_warning"] = 0
sns.set_theme(style="white")
DATA = Path("../datasets")

housing = pd.read_csv(DATA / "housing.csv")
homes = housing[(housing.price / housing.area_sqft).between(20, 500)].copy()
num_cols = ["area_sqft", "bedrooms", "bathrooms", "age_years", "garage_spaces", "distance_to_center_km", "quality_score", "price"]

churn = pd.read_csv(DATA / "churn.csv")
churn["contract"] = churn.contract.str.lower().map({"month-to-month": "Month-to-month", "one year": "One year", "two year": "Two year"})
print(len(homes), "homes;", len(churn), "customers")

# %% [markdown]
# ## 1. The correlation matrix
# The **Pearson correlation** between two columns is a number from -1 to +1 measuring how well a *straight line* describes their relationship (+1 perfect upward line, -1 perfect downward, 0 no linear relationship).
# `DataFrame.corr()` computes all pairs; it ignores missing values pair by pair (`age_years` and `quality_score` have some).

# %%
corr = homes[num_cols].corr()
corr.round(2)

# %% [markdown]
# A table of 64 numbers hides the pattern. Colour it, but do it carefully:
# - **Mask the upper triangle**: the matrix is symmetric, half the cells repeat the other half, and the diagonal is always 1.
# - Use a **diverging palette centred at 0** (`center=0`, `cmap="vlag"`) so sign is obvious, with **fixed limits** `vmin=-1, vmax=1` so colours mean the same thing on every chart.
# - Annotate the values.

# %%
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
fig, ax = plt.subplots(figsize=(7.5, 6), layout="constrained")
sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, center=0, cmap="vlag", annot=True, fmt=".2f", annot_kws={"size": 8}, linewidths=0.5, square=True,
            cbar_kws={"label": "Pearson correlation (-1 to +1)"}, ax=ax)
ax.set(title="Correlation of housing variables")
plt.show()

# %% [markdown]
# *What just happened?* Strong red cells: `price` with `area_sqft` (and through it with `bedrooms` and `bathrooms`); `bedrooms`-`bathrooms` are correlated with each other (**multicollinearity**: two columns carrying much the same information, a known problem for linear models);
# near-zero cells: `garage_spaces`, `quality_score` hardly relate to area. Confirm the strongest pair with code rather than by eye:

# %%
pairs = corr.where(~mask & ~np.eye(len(corr), dtype=bool)).stack().rename("r")
print(pairs.abs().sort_values(ascending=False).head(3).round(2).to_string())

# %% [markdown]
# **Reading a correlation matrix responsibly**
# - Pearson measures **linear** association only. A perfect U-shape has correlation near 0. Always look at the scatter for pairs you care about.
# - It is sensitive to **outliers** (that is why we removed the implausible listings first) and is **not causal**.
# - For monotonic-but-curved relationships, **Spearman** (rank correlation) is more faithful. Compare on the curved price-area link:

# %%
for method in ("pearson", "spearman"):
    print(f"{method:8s} area vs price: {homes.area_sqft.corr(homes.price, method=method):.3f}")

# %% [markdown]
# Spearman is a bit higher because price rises monotonically but curves upward, and ranks ignore the curvature. Only read colours as "strength", never as "importance": a column with low correlation can still matter in a model through interactions.

# %% [markdown]
# ## 2. A heatmap of a pivot table
# Heatmaps are not only for correlations: any **two categories (or bins) by one number** fits. Churn rate by contract type and tenure band:

# %%
churn["tenure_band"] = pd.cut(churn.tenure_months, [-0.1, 6, 12, 24, 48, 72], labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
rate = churn.pivot_table(index="contract", columns="tenure_band", values="churned", aggfunc="mean", observed=True)
n = churn.pivot_table(index="contract", columns="tenure_band", values="churned", aggfunc="size", observed=True)

fig, ax = plt.subplots(figsize=(7, 3.2), layout="constrained")
sns.heatmap(rate, annot=True, fmt=".0%", cmap="rocket_r", vmin=0, vmax=1, cbar_kws={"label": "Churn rate (fraction of customers)"}, ax=ax)
ax.set(title="Churn rate by contract and tenure", xlabel="Tenure band (months)", ylabel="Contract type")
plt.show()
n

# %% [markdown]
# Use a **sequential** palette (light to dark) for one-directional quantities like rates (and a diverging one only when there is a meaningful midpoint like 0). The second table is the **group size** behind each cell: a dark cell built on
# 12 customers deserves less trust than one built on 900. Good heatmap habit: print or show `n` next to the rates.

# %% [markdown]
# ## 3. `pairplot`: every pair at once
# `pairplot` draws a scatter for each pair of columns and the distribution of each column on the diagonal. Use it on **a handful of columns** and a **sample**: the cost grows with the square of the number of columns.

# %%
cols = ["area_sqft", "age_years", "distance_to_center_km", "price"]
g = sns.pairplot(homes.sample(500, random_state=42), vars=cols, corner=True, diag_kind="kde", plot_kws={"s": 8, "alpha": 0.4}, height=2)
g.figure.suptitle("Pair plot of four housing variables (500-row sample)", y=1.02)
plt.show()
print("panels drawn:", sum(a is not None for a in g.axes.flat), "for", len(cols), "columns (corner=True draws the lower triangle only)")

# %% [markdown]
# Read the bottom row: how price relates to each variable. Area clearly rises with price; age and distance show weak, noisy trends, matching the heatmap's small numbers. `corner=True` drops the redundant upper triangle.
# Add `hue="neighborhood"` for a categorical split, but with six colours and overlapping clouds it gets busy fast.
# **Always sample:** a pair plot of 3,000 rows x 8 columns is 64 panels with 192,000 points.

# %% [markdown]
# ## 4. Grids you control: `FacetGrid` and `PairGrid`
# `FacetGrid` splits the data by a category into panels and then draws **any** plotting function on each. `col_wrap` wraps long rows; panels share axes by default, which is what makes small multiples comparable.

# %%
g = sns.FacetGrid(homes, col="neighborhood", col_wrap=3, height=2.4, aspect=1.2)
g.map_dataframe(sns.histplot, x="price", bins=25)
overall_median = homes.price.median()
for ax in g.axes.flat:
    ax.axvline(overall_median, color="crimson", linestyle="--", linewidth=1)          # the same reference line in every panel
g.set_titles("{col_name}")
g.set_axis_labels("Price (USD)", "Listings (count)")
g.figure.suptitle("Price by neighbourhood (dashed = overall median)", y=1.03)
plt.show()

# %% [markdown]
# The dashed line is the same in every panel, so you see at a glance which neighbourhoods sit above or below the overall median (Downtown right of the line, Eastgate left). That **shared reference** is what turns six histograms into a comparison.
# Verify what the picture suggests with numbers:

# %%
print((homes.groupby("neighborhood").price.median() - overall_median).round(0).sort_values().to_string())

# %% [markdown]
# **`catplot`** is a `FacetGrid` specialised for categories (`kind="box"`, `"violin"`, `"bar"`, `"strip"`, `"count"`). **`PairGrid`** is the customisable `pairplot`: pick different plot types for the lower triangle, diagonal and upper triangle.

# %%
pg = sns.PairGrid(homes.sample(500, random_state=42), vars=["area_sqft", "distance_to_center_km", "price"], height=2.2)
pg.map_lower(sns.scatterplot, s=8, alpha=0.4)
pg.map_diag(sns.histplot, bins=20)
pg.map_upper(sns.kdeplot, levels=5)
pg.figure.suptitle("PairGrid: scatter below, density contours above", y=1.03)
plt.show()

# %% [markdown]
# ## Common mistakes
#
# 1. **A full symmetric heatmap** with a rainbow colormap: redundant half, and colours that do not map to magnitude. Mask and use a diverging palette centred on 0.
# 2. **No fixed `vmin`/`vmax`**: the colour scale stretches to the data, so a weak 0.3 looks like a strong signal.
# 3. **Reading correlation as causation or as "importance".**
# 4. **Trusting Pearson on curved or outlier-laden data**; check the scatter, consider Spearman.
# 5. **A pairplot of many columns and thousands of rows**: slow and unreadable. Sample and select.
# 6. **Heatmap rates without group sizes.**
# 7. **Correlating categorical codes** (for example encoding `neighborhood` as 1-6 and correlating it): the numbers are labels, not quantities.

# %%
# Mistake 4 in action: a perfect but non-linear relationship
x = np.linspace(-3, 3, 200)
df_u = pd.DataFrame({"x": x, "y": x ** 2})
print(f"y = x^2 is a perfect relationship, yet Pearson r = {df_u.x.corr(df_u.y):.3f}")

# %% [markdown]
# ## Interview corner
#
# **1. What does a correlation of 0 mean?**
# <details><summary>Answer</summary>
# No *linear* association. Variables can still be strongly related non-linearly (for example y = x squared, or a U-shape). Plot the pair.
# </details>
#
# **2. Why mask half of a correlation heatmap and fix vmin/vmax?**
# <details><summary>Answer</summary>
# The matrix is symmetric so half the cells are duplicates; masking reduces clutter. Fixed limits (-1 to 1) and a diverging palette centred at 0 keep colours comparable across charts and make sign and strength legible.
# </details>
#
# **3. Two features have a 0.95 correlation. Why might that matter for modelling?**
# <details><summary>Answer</summary>
# Multicollinearity: coefficients of linear models become unstable and hard to interpret, and you pay for redundant features. Consider dropping one, combining them, or using regularisation; tree models are less affected in predictive accuracy but importances get split.
# </details>
#
# **4. When do you use Spearman instead of Pearson?**
# <details><summary>Answer</summary>
# When the relationship is monotonic but not linear, the data are ordinal, or outliers are a concern; Spearman correlates the ranks, so it is insensitive to monotone transformations and extreme values.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: A readable correlation heatmap
# Draw a correlation heatmap of `homes[num_cols]` on `fig1, ax1` with: the **upper triangle masked** (diagonal kept), `vmin=-1, vmax=1`, a diverging cmap with `center=0`, annotations, and a title.
# Then set `top_pair` to the tuple `(col_a, col_b)` (alphabetical order inside the tuple) of the two **different** columns with the largest absolute correlation.
# *Hint: `mask = np.triu(np.ones_like(corr, dtype=bool), k=1)`.*

# %% tags=["starter"]
fig1, ax1 = plt.subplots()
top_pair = ...

# %%
corr_ = homes[num_cols].corr()
k = len(corr_)
mesh = ax1.collections[0]
arr = mesh.get_array()
assert arr.shape == (k, k), "heatmap should cover the full k x k matrix (with a mask), not a subset"
assert int(np.ma.getmaskarray(arr).sum()) == k * (k - 1) // 2, "mask exactly the upper triangle (np.triu(..., k=1)); keep the diagonal"
assert mesh.get_clim() == (-1.0, 1.0), "set vmin=-1 and vmax=1"
assert len(ax1.texts) >= k * (k + 1) // 2, "annotate the visible cells (annot=True)"
assert ax1.get_title(), "add a title"
off = corr_.where(~np.eye(k, dtype=bool)).abs().stack()
best = tuple(sorted(off.idxmax()))
assert tuple(top_pair) == best, f"the strongest pair is {best}"
print(f"Exercise 1 passed: {best} with |r| = {off.max():.2f}")

# %% tags=["solution"]
corr_ = homes[num_cols].corr()
mask_ = np.triu(np.ones_like(corr_, dtype=bool), k=1)
fig1, ax1 = plt.subplots(figsize=(7.5, 6), layout="constrained")
sns.heatmap(corr_, mask=mask_, vmin=-1, vmax=1, center=0, cmap="vlag", annot=True, fmt=".2f", annot_kws={"size": 8}, square=True,
            cbar_kws={"label": "Pearson correlation"}, ax=ax1)
ax1.set(title="Correlation of housing variables")
off = corr_.where(~np.eye(len(corr_), dtype=bool)).abs().stack()
top_pair = tuple(sorted(off.idxmax()))

# %% [markdown]
# ### Exercise 2: Where is churn highest?
# Build `rate` = churn rate by `contract` (rows) and `tenure_band` (columns) as in section 2 and draw it as an annotated heatmap on `fig2, ax2` with `vmin=0, vmax=1`.
# Then set `riskiest` to the `(contract, tenure_band)` cell with the highest rate **among cells with at least 100 customers** (as strings).

# %% tags=["starter"]
fig2, ax2 = plt.subplots()
riskiest = ...

# %%
rate_ = churn.pivot_table(index="contract", columns="tenure_band", values="churned", aggfunc="mean", observed=True)
n_ = churn.pivot_table(index="contract", columns="tenure_band", values="churned", aggfunc="size", observed=True)
arr = ax2.collections[0].get_array()
assert arr.shape == rate_.shape, f"heatmap should be {rate_.shape} (contract x tenure band)"
assert np.allclose(np.asarray(arr, dtype=float), rate_.values, equal_nan=True), "cell values should be churn rates"
assert ax2.collections[0].get_clim() == (0.0, 1.0), "vmin=0, vmax=1"
assert len(ax2.texts) == rate_.size, "annotate every cell"
assert ax2.get_title() and ax2.get_xlabel() and ax2.get_ylabel()
ok = rate_.where(n_ >= 100).stack()
assert len(ok) > 0
assert tuple(map(str, riskiest)) == tuple(map(str, ok.idxmax())), f"expected {ok.idxmax()} (highest rate among cells with n >= 100)"
print(f"Exercise 2 passed: {ok.idxmax()} at {ok.max():.1%}")

# %% tags=["solution"]
rate_ = churn.pivot_table(index="contract", columns="tenure_band", values="churned", aggfunc="mean", observed=True)
n_ = churn.pivot_table(index="contract", columns="tenure_band", values="churned", aggfunc="size", observed=True)
fig2, ax2 = plt.subplots(figsize=(7, 3.2), layout="constrained")
sns.heatmap(rate_, annot=True, fmt=".0%", cmap="rocket_r", vmin=0, vmax=1, cbar_kws={"label": "Churn rate"}, ax=ax2)
ax2.set(title="Churn rate by contract and tenure", xlabel="Tenure band (months)", ylabel="Contract type")
riskiest = tuple(map(str, rate_.where(n_ >= 100).stack().idxmax()))

# %% [markdown]
# ### Exercise 3: A sampled corner pair plot
# Draw a `pairplot` of `area_sqft`, `bedrooms`, `price` on a **400-row random sample** of `homes` (`random_state=42`), `corner=True`. Store the `PairGrid` in `g3`. Then set `n_panels` to the number of panels actually drawn (count the Axes that are not `None`).

# %% tags=["starter"]
g3 = ...
n_panels = ...

# %%
assert type(g3).__name__ == "PairGrid", "pairplot returns a PairGrid"
assert g3.axes.shape == (3, 3)
drawn = sum(a is not None for a in g3.axes.flat)
assert n_panels == drawn == 6, "corner=True on 3 variables draws 3 + 2 + 1 = 6 panels"
pts = [a for a in g3.axes.flat if a is not None and len(a.collections) == 1]
assert pts and all(a.collections[0].get_offsets().shape[0] == 400 for a in pts), "each scatter panel should show the 400 sampled rows"
print("Exercise 3 passed")

# %% tags=["solution"]
g3 = sns.pairplot(homes.sample(400, random_state=42), vars=["area_sqft", "bedrooms", "price"], corner=True, plot_kws={"s": 8, "alpha": 0.4}, height=2.2)
n_panels = sum(a is not None for a in g3.axes.flat)

# %% [markdown]
# ### Exercise 4: Six histograms, one reference line
# Build a `FacetGrid` `g4` of `homes` with `col="neighborhood"`, `col_wrap=3`, draw a histogram of `price` with **20 bins** in every panel (`map_dataframe`), add a **vertical dashed line at the overall median price** to every panel, set titles to the neighbourhood names, and label axes with units.

# %% tags=["starter"]
g4 = ...

# %%
assert g4.axes.shape == (6,), "six panels (col_wrap=3 gives 2 rows x 3)"
med = homes.price.median()
assert [a.get_title() for a in g4.axes.flat] == sorted(homes.neighborhood.unique()), "titles should be the neighbourhood names"
for a in g4.axes.flat:
    assert len(a.patches) == 20, "20 bins in every panel"
    lines = [ln for ln in a.lines if np.allclose(ln.get_xdata(), med)]
    assert len(lines) == 1 and lines[0].get_linestyle() == "--", "every panel needs a dashed vertical line at the overall median"
assert "usd" in g4.axes.flat[-1].get_xlabel().lower() and g4.axes.flat[0].get_ylabel(), "axis labels with units"
print("Exercise 4 passed")

# %% tags=["solution"]
g4 = sns.FacetGrid(homes, col="neighborhood", col_order=sorted(homes.neighborhood.unique()), col_wrap=3, height=2.4, aspect=1.2)
g4.map_dataframe(sns.histplot, x="price", bins=20)
for a in g4.axes.flat:
    a.axvline(homes.price.median(), color="crimson", linestyle="--", linewidth=1)
g4.set_titles("{col_name}")
g4.set_axis_labels("Price (USD)", "Listings (count)")

# %% [markdown]
# ### Exercise 5 (open-ended): From 28 pairs to one question
# Use the correlation heatmap of `churn`'s numeric columns (`tenure_months`, `monthly_charges`, `support_calls`, `senior_citizen`, `churned`; drop rows with missing values first) to choose **one** pair worth a closer look.
# Plot it properly (scatter, or rate plot if one is binary), and state in two sentences what the correlation does and does not tell you here.

# %% [markdown]
# ## Recap
# - Correlation heatmap: mask the redundant triangle, diverging palette centred at 0, `vmin=-1, vmax=1`, annotate; verify the strongest pairs in code.
# - Pearson = linear only and outlier-sensitive; Spearman for monotonic curves; correlation is not causation; watch multicollinearity.
# - Pivot table + heatmap for rates across two categories; show group sizes.
# - `pairplot` (sample, few columns, `corner=True`), `PairGrid` for custom triangles, `FacetGrid`/`catplot` for small multiples with shared axes and shared reference lines.
#
# **Go deeper:** seaborn docs "Plotting with heatmaps", "Building structured multi-plot grids" and `clustermap` (seaborn.pydata.org); pandas `DataFrame.corr` (methods pearson, spearman, kendall).
#
# **Next:** [Visual storytelling](04_visual_storytelling.ipynb)

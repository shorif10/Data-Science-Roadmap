# %% [markdown]
# # Core Chart Types and When to Use Them
#
# **Why this matters.** Choosing the wrong chart is not a style problem, it is a *communication* bug: a pie with nine slices hides the ranking, a truncated bar chart makes a 3% gap look like a 3x gap,
# a line across unordered categories invents a trend. Data scientists get asked "can you plot this?" daily, and the skill is to start from the **question** ("compare", "trend", "distribution", "relationship"), not from the chart menu.
#
# **You will learn**
# - map a question to a chart: **line** (trend), **bar** (compare), **histogram** (distribution), **box** (compare distributions), **scatter** (relationship)
# - draw each with the object-oriented API and read it correctly (what a box plot's box, whiskers and dots mean)
# - spot and fix common wrong-chart mistakes: truncated bars, pie overload, lines across categories, too few or too many histogram bins, overplotted scatter
#
# **Prerequisites:** [Anatomy of a figure](01_anatomy_figure_axes.ipynb).
#
# > Data used: `datasets/orders.csv`, `order_items.csv`, `products.csv` (synthetic shop) and `datasets/housing.csv` (synthetic house listings, with 15 planted data-entry errors; the simple filter below flags a few more, because any such rule is a judgement call). See `docs/DATASETS.md`.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.max_open_warning"] = 0
DATA = Path("../datasets")

orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
items = pd.read_csv(DATA / "order_items.csv")
products = pd.read_csv(DATA / "products.csv")
housing = pd.read_csv(DATA / "housing.csv")

# The housing file contains absurd prices (typos). A simple sanity rule: price per square foot between 20 and 500.
# This is a judgement call, not a law; a real project would ask the data owner.
housing["price_per_sqft"] = housing.price / housing.area_sqft
plausible = housing.price_per_sqft.between(20, 500)
homes = housing[plausible].copy()
print(f"kept {len(homes)} of {len(housing)} listings; flagged {(~plausible).sum()} as implausible price per sqft")

# %% [markdown]
# ## 0. Start from the question
#
# | Your question | Chart | Watch out for |
# |---|---|---|
# | How did X change over time? | **line** | uneven time steps; too many series |
# | How do categories compare? | **bar** (horizontal if names are long), sorted | y-axis must start at 0 |
# | What does the distribution of one number look like? | **histogram** (or KDE/ECDF, next module) | bin width changes the story |
# | How do distributions compare across groups? | **box** (or violin) | hides shape (two humps look like one box) |
# | Are two numbers related? | **scatter** | overplotting; correlation is not causation |
# | What share does each part have? | **stacked bar** or a sorted bar of shares, rarely a pie | more than ~4 slices is unreadable |

# %% [markdown]
# ## 1. Line chart: change over time
# A line joins points **in order**, so the x-axis must be ordered and continuous (time is the classic case). The slope is the message.

# %%
monthly = orders.groupby(orders.order_date.dt.to_period("M").dt.to_timestamp()).size()

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(monthly.index, monthly.values, marker="o", markersize=4)
ax.set(title="Orders per month", xlabel="Month", ylabel="Orders (count)")
ax.grid(alpha=0.3)
fig.autofmt_xdate()
plt.show()

# %% [markdown]
# *What just happened?* One ordered series, dots on the real observations, light grid. The data rises with a big December peak each year: a **seasonal** pattern. A line lets you see it; a table would not.
# The first and last months of a dataset are often partial: check before you interpret an end-of-line drop.

# %% [markdown]
# ## 2. Bar chart: compare categories
# A bar's **length** encodes the value, so the baseline must be **zero**. Sort by value (unless the categories have a natural order) and use horizontal bars when labels are long.

# %%
rev = (items.merge(products, on="product_id").merge(orders[["order_id", "status"]], on="order_id")
       .query("status != 'cancelled'").assign(revenue=lambda d: d.quantity * d.unit_price)
       .groupby("category").revenue.sum().sort_values())

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.barh(rev.index, rev.values / 1000)
ax.set(title="Revenue by category (non-cancelled orders, current prices)", xlabel="Revenue (USD thousands)", ylabel="Category")
for y, v in enumerate(rev.values / 1000):
    ax.text(v, y, f" {v:,.0f}k", va="center", fontsize=8)
ax.set_xlim(0, rev.max() / 1000 * 1.15)
plt.show()

# %% [markdown]
# `barh` draws the **first** item at the bottom, so we sorted ascending to get the largest on top. Value labels at the bar ends save the reader from squinting at gridlines.
#
# **Wrong-chart example 1: the truncated bar.** Same numbers, but the y-axis starts near the smallest value. Watch how much taller the top bar looks compared with the truth:

# %%
top2 = rev.sort_values(ascending=False).iloc[:2]
fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
for ax_, ymin, ttl in ((a, 0, "Honest: axis starts at 0"), (b, top2.min() * 0.9, "Misleading: axis truncated")):
    ax_.bar(top2.index, top2.values / 1000)
    ax_.set_ylim(ymin / 1000, None)
    ax_.set(title=ttl, xlabel="Category", ylabel="Revenue (USD thousands)")
plt.show()
true_ratio = top2.iloc[0] / top2.iloc[1]
seen_ratio = (top2.iloc[0] - top2.min() * 0.9) / (top2.iloc[1] - top2.min() * 0.9)
print(f"true ratio of the two bars: {true_ratio:.2f}x   ratio the truncated chart makes you SEE: {seen_ratio:.2f}x")

# %% [markdown]
# ## 3. Histogram: the distribution of one number
# A histogram cuts the number line into **bins** and counts how many values fall in each. It answers: where are most values, how spread out, is it skewed, are there outliers or two humps?
# The most important (and most abused) setting is the **number of bins**.

# %%
fig, axs = plt.subplots(1, 3, figsize=(11, 3), layout="constrained", sharey=False)
for ax_, bins in zip(axs, (4, 40, 400)):
    ax_.hist(homes.price / 1000, bins=bins)
    ax_.set(title=f"{bins} bins", xlabel="Price (USD thousands)", ylabel="Listings (count)")
fig.suptitle("Same data, three bin counts")
plt.show()

# %% [markdown]
# Too few bins (left) hide the shape; too many (right) show noise. A useful default is to try a few values around what `bins="auto"` picks, and to prefer round bin widths ("every 25k") when you must explain the chart.
# Prices are **right-skewed** (a long tail of expensive homes): the mean is pulled above the median. Check with numbers, not only with the picture:

# %%
print(f"mean {homes.price.mean():,.0f}   median {homes.price.median():,.0f}   skewness {homes.price.skew():.2f}")

# %% [markdown]
# **Wrong-chart example 2: leaving errors in.** Plotting *all* listings, including the typos, squeezes the real data into a few bars at the left because the axis must stretch to reach the largest absurd value.

# %%
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3), layout="constrained")
a.hist(housing.price / 1000, bins=40)
a.set(title="With implausible prices", xlabel="Price (USD thousands)", ylabel="Listings (count)")
b.hist(homes.price / 1000, bins=40)
b.set(title="After the sanity filter", xlabel="Price (USD thousands)", ylabel="Listings (count)")
plt.show()

# %% [markdown]
# Always plot raw data *first*: a histogram is one of the fastest ways to discover data-quality problems. But do not delete points silently: say how many you removed and why, as we printed at the top.

# %% [markdown]
# ## 4. Box plot: compare distributions across groups
# A box plot shows five numbers per group: the **box** spans the 25th to 75th percentile (Q1 to Q3, the interquartile range, IQR), the line inside is the **median**, the **whiskers** reach the furthest
# point within 1.5 x IQR of the box, and points beyond are drawn individually as outliers. It is compact, so it lets you compare many groups; it hides the *shape* (two humps look like one box).

# %%
order_ = homes.groupby("neighborhood").price.median().sort_values().index.tolist()
groups = [homes.loc[homes.neighborhood == n, "price"] / 1000 for n in order_]

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.boxplot(groups, tick_labels=order_)
ax.set(title="House price by neighbourhood (sorted by median)", xlabel="Neighbourhood", ylabel="Price (USD thousands)")
plt.show()

# verify the box edges are what we said they are
q1, med, q3 = groups[0].quantile([0.25, 0.5, 0.75])
print(f"{order_[0]}: Q1 {q1:.0f}k, median {med:.0f}k, Q3 {q3:.0f}k, IQR {q3 - q1:.0f}k, upper whisker limit {q3 + 1.5 * (q3 - q1):.0f}k")

# %% [markdown]
# Sorting groups by median makes the ranking obvious; alphabetical order rarely is meaningful. In `boxplot`, the labels parameter is called `tick_labels` in current Matplotlib (older code uses `labels=`, now deprecated).

# %% [markdown]
# ## 5. Scatter plot: relationship between two numbers
# Each point is one row. Look for direction (up or down), shape (line or curve), spread, and clusters.

# %%
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.scatter(homes.area_sqft, homes.price / 1000, s=8, alpha=0.3)
ax.set(title="Price rises with living area", xlabel="Living area (sq ft)", ylabel="Price (USD thousands)")
plt.show()
print(f"correlation area vs price: {homes.area_sqft.corr(homes.price):.2f}")

# %% [markdown]
# With 3,000 points, `alpha=0.3` (transparency) lets dense regions look darker: that is the fix for **overplotting**. Compare with fully opaque large dots, where everything is a blob:

# %%
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
a.scatter(homes.area_sqft, homes.price / 1000, s=40)
a.set(title="Opaque, large markers: blob", xlabel="Living area (sq ft)", ylabel="Price (USD thousands)")
b.hexbin(homes.area_sqft, homes.price / 1000, gridsize=30, mincnt=1)
b.set(title="Hexbin: density is visible", xlabel="Living area (sq ft)", ylabel="Price (USD thousands)")
plt.show()

# %% [markdown]
# A scatter shows association, not cause: bigger homes cost more, but a scatter of ice-cream sales and drowning deaths would look just as neat. Add a third variable with colour or size, not with a 3D chart.

# %% [markdown]
# ## 6. Wrong-chart gallery
#
# **Pie with many slices.** Humans compare angles poorly. Same shares as a sorted bar:

# %%
share = orders.groupby("status").size().sort_values(ascending=False)
mix = homes.groupby("neighborhood").size().sort_values(ascending=False)     # six slices of near-equal size: the pie's worst case
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
a.pie(mix.values, labels=mix.index)
a.set_title("Pie: which neighbourhood is biggest?")
b.barh(mix.index[::-1], mix.values[::-1])
b.set(title="Sorted bar: the ranking is obvious", xlabel="Listings (count)", ylabel="Neighbourhood")
plt.show()

# %% [markdown]
# **Line across unordered categories.** Connecting Downtown to Eastgate to Hillcrest implies a "trend" from one neighbourhood to the next, but the order is alphabetical, arbitrary. Use bars (points are fine, lines are not):

# %%
avg = homes.groupby("neighborhood").price.mean() / 1000
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained", sharey=True)
a.plot(avg.index, avg.values, marker="o")
a.set(title="Wrong: a line implies order", xlabel="Neighbourhood", ylabel="Mean price (USD thousands)")
b.bar(avg.index, avg.values)
b.set(title="Right: separate bars", xlabel="Neighbourhood")
plt.show()

# %% [markdown]
# ## Common mistakes
#
# 1. **Bar chart with a non-zero baseline** (shown above).
# 2. **Pie/donut charts** with many slices or where slices differ little; use sorted bars.
# 3. **Lines across unordered categories.**
# 4. **A histogram with the default bins accepted blindly**, or with outliers left in so the data hides in one bar.
# 5. **Overplotted scatter** (fix: `alpha`, smaller `s`, `hexbin`, or sample).
# 6. **Sorting categories alphabetically** when ranking is the point.
# 7. **Missing units** in axis labels (USD thousands, sq ft, count).
# 8. **Dual y-axes** that make unrelated series look correlated by choosing the two scales freely; avoid unless the audience needs it (notebook 04).

# %%
# Mistake 4 in action, in numbers: with 4 bins, how much of the story is left?
counts, edges = np.histogram(homes.price, bins=4)
print("4 bins:", counts.tolist(), "-> one bar holds", f"{counts.max() / counts.sum():.0%}", "of the data")

# %% [markdown]
# ## Interview corner
#
# **1. When would you use a histogram instead of a box plot?**
# <details><summary>Answer</summary>
# To see the *shape* of one distribution: skew, several peaks, gaps. A box plot compresses a distribution to five numbers plus outliers, which is ideal for comparing many groups but can hide multimodality.
# </details>
#
# **2. Why must a bar chart start at zero but a line chart need not?**
# <details><summary>Answer</summary>
# Bars encode value by length/area, so cutting the baseline distorts ratios. A line encodes value by position and slope; the reader compares shape and change, and a zero baseline can flatten real movement. State the axis range clearly.
# </details>
#
# **3. What do the whiskers of a box plot show?**
# <details><summary>Answer</summary>
# By default, the most extreme data points within 1.5 x IQR of the quartiles (Q1 minus 1.5 IQR, Q3 plus 1.5 IQR). Points beyond are plotted individually as outliers. It is a convention, not a statistical test.
# </details>
#
# **4. Your scatter plot of 500,000 points is a black blob. What do you do?**
# <details><summary>Answer</summary>
# Reduce marker size and add transparency, use a 2-D density plot (hexbin or 2-D histogram), or plot a random sample; consider aggregating or binning the x variable.
# </details>

# %% [markdown]
# ## Exercises
# The checks read the **figure objects**: patches (bars), collections (scatter), lines, labels.

# %% [markdown]
# ### Exercise 1: Rank the neighbourhoods
# Draw a **horizontal bar chart** of the **mean price** (in USD thousands) of `homes` per neighbourhood, with the **most expensive neighbourhood at the top**. Store `fig, ax`. Title, an x label with the unit, and a y label are required.
# *Hint: `barh` puts the first bar at the bottom, so pass values sorted ascending.*

# %% tags=["starter"]
fig, ax = plt.subplots()
# your code here

# %%
exp = (homes.groupby("neighborhood").price.mean() / 1000).sort_values()
assert len(ax.patches) == len(exp) == 6, "one bar per neighbourhood (6)"
widths = [p.get_width() for p in ax.patches]
assert np.allclose(widths, exp.values), "bar lengths should be mean price in USD thousands, ordered so the largest is at the top (sorted ascending in barh)"
assert [t.get_text() for t in ax.get_yticklabels()] == exp.index.tolist(), "y tick labels must be the neighbourhood names in the same order"
assert ax.get_title() and ax.get_ylabel() and "usd" in ax.get_xlabel().lower(), "title, y label, and an x label with the unit (USD) are required"
assert ax.get_xlim()[0] == 0, "bars need a zero baseline"
print(f"Exercise 1 passed: most expensive = {exp.index[-1]} ({exp.iloc[-1]:,.0f}k)")

# %% tags=["solution"]
exp = (homes.groupby("neighborhood").price.mean() / 1000).sort_values()
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.barh(exp.index, exp.values)
ax.set(title="Mean house price by neighbourhood", xlabel="Mean price (USD thousands)", ylabel="Neighbourhood")
ax.set_xlim(0, None)

# %% [markdown]
# ### Exercise 2: Histogram with the median marked
# Draw a histogram of `homes.price` in USD thousands with **exactly 40 bins**, and a vertical dashed line at the **median** (label it `"median"`, add a legend). Store `fig2, ax2`. Label axes with units.

# %% tags=["starter"]
fig2, ax2 = plt.subplots()
# your code here

# %%
assert len(ax2.patches) == 40, f"expected 40 bars, found {len(ax2.patches)}"
assert int(round(sum(p.get_height() for p in ax2.patches))) == len(homes), "the bars must count every plausible listing"
med = homes.price.median() / 1000
vl = [ln for ln in ax2.lines if np.allclose(ln.get_xdata(), med)]
assert len(vl) == 1, f"add a vertical line at the median ({med:.1f})"
assert vl[0].get_linestyle() == "--" and vl[0].get_label() == "median", "dashed, labelled 'median'"
assert ax2.get_legend() is not None, "show the legend"
assert ax2.get_title() and "usd" in ax2.get_xlabel().lower() and ax2.get_ylabel(), "title, x label with unit, y label"
print("Exercise 2 passed")

# %% tags=["solution"]
fig2, ax2 = plt.subplots(figsize=(7, 3.4))
ax2.hist(homes.price / 1000, bins=40)
ax2.axvline(homes.price.median() / 1000, linestyle="--", color="black", label="median")
ax2.set(title="Distribution of house prices", xlabel="Price (USD thousands)", ylabel="Listings (count)")
ax2.legend()

# %% [markdown]
# ### Exercise 3: Scatter without the blob
# Scatter `distance_to_center_km` (x) against `price / 1000` (y) for `homes`. Use marker size `s <= 15` and transparency `alpha <= 0.5` so dense areas are readable. Store `fig3, ax3`. Then set `dist_corr` to the correlation coefficient between distance and price, and
# `direction` to `"negative"` or `"positive"` according to its sign.

# %% tags=["starter"]
fig3, ax3 = plt.subplots()
dist_corr = ...
direction = ...

# %%
assert len(ax3.collections) == 1, "one scatter collection"
coll = ax3.collections[0]
assert coll.get_offsets().shape[0] == len(homes), "plot every plausible listing"
assert np.allclose(coll.get_offsets()[:, 0], homes.distance_to_center_km.values), "x should be distance to centre"
assert np.allclose(coll.get_offsets()[:, 1], homes.price.values / 1000), "y should be price in USD thousands"
assert coll.get_alpha() is not None and coll.get_alpha() <= 0.5, "alpha <= 0.5"
assert np.max(coll.get_sizes()) <= 15, "marker size s <= 15"
assert ax3.get_title() and "km" in ax3.get_xlabel().lower() and "usd" in ax3.get_ylabel().lower(), "labels with units (km, USD)"
exp_corr = homes.distance_to_center_km.corr(homes.price)
assert abs(dist_corr - exp_corr) < 1e-9, "dist_corr should be the Pearson correlation"
assert direction == ("negative" if exp_corr < 0 else "positive")
print(f"Exercise 3 passed (r = {exp_corr:.2f})")

# %% tags=["solution"]
fig3, ax3 = plt.subplots(figsize=(6.5, 4))
ax3.scatter(homes.distance_to_center_km, homes.price / 1000, s=8, alpha=0.3)
ax3.set(title="Price vs distance from the city centre", xlabel="Distance to city centre (km)", ylabel="Price (USD thousands)")
dist_corr = homes.distance_to_center_km.corr(homes.price)
direction = "negative" if dist_corr < 0 else "positive"

# %% [markdown]
# ### Exercise 4: Compare price spread by bedrooms
# Draw a **box plot** of price (USD thousands) for homes with **1 to 5 bedrooms** (five boxes, in that order, tick labels `"1"`..`"5"`). Store `fig4, ax4`. Then set `widest_box` to the bedroom count (an int from 1-5) whose **IQR** (Q3 minus Q1) is largest.
# *Hint: `ax.boxplot(list_of_series, tick_labels=[...])`; compute IQRs with `quantile`.*

# %% tags=["starter"]
fig4, ax4 = plt.subplots()
widest_box = ...

# %%
sub = homes[homes.bedrooms.between(1, 5)]
data = {b: sub.loc[sub.bedrooms == b, "price"] / 1000 for b in range(1, 6)}
assert [t.get_text() for t in ax4.get_xticklabels()] == ["1", "2", "3", "4", "5"], "tick labels must be '1'..'5'"
assert len(ax4.lines) == 5 * 7, f"expected 5 boxes (7 line artists each), found {len(ax4.lines)} lines; did you draw exactly five boxes?"
medians = {round(float(np.median(v)), 6) for v in data.values()}
drawn = {round(float(ln.get_ydata()[0]), 6) for ln in ax4.lines if len(ln.get_ydata()) == 2 and ln.get_ydata()[0] == ln.get_ydata()[1]}
assert medians <= drawn, "median lines are missing or use different data (USD thousands, homes only)"
assert ax4.get_title() and ax4.get_xlabel() and "usd" in ax4.get_ylabel().lower()
iqr = {b: v.quantile(0.75) - v.quantile(0.25) for b, v in data.items()}
assert widest_box == max(iqr, key=iqr.get), f"widest IQR belongs to {max(iqr, key=iqr.get)} bedrooms"
print(f"Exercise 4 passed (widest spread: {widest_box} bedrooms)")

# %% tags=["solution"]
sub = homes[homes.bedrooms.between(1, 5)]
data = [sub.loc[sub.bedrooms == b, "price"] / 1000 for b in range(1, 6)]
fig4, ax4 = plt.subplots(figsize=(7, 3.6))
ax4.boxplot(data, tick_labels=[str(b) for b in range(1, 6)])
ax4.set(title="House price by number of bedrooms", xlabel="Bedrooms", ylabel="Price (USD thousands)")
iqr = [v.quantile(0.75) - v.quantile(0.25) for v in data]
widest_box = int(np.argmax(iqr)) + 1

# %% [markdown]
# ### Exercise 5 (open-ended): Fix a bad chart
# Pick one chart from the wrong-chart gallery (or one you have seen at work) and write, in a markdown cell: (a) what question it was trying to answer, (b) why the chart type misleads, (c) the better chart. Then implement (c) for `orders.status`.

# %% [markdown]
# ## Recap
# - Choose by question: **line** for trend, **bar** for comparison (zero baseline, sorted), **histogram** for one distribution (test several bin counts), **box** for comparing distributions, **scatter** for relationships (mind overplotting).
# - Inspect the data before trusting a chart; a histogram exposes outliers and typos quickly, and you should report what you removed.
# - Avoid pies with many slices, lines across categories, truncated bars.
# - Every plot: title, labelled axes, units.
#
# **Go deeper:** Matplotlib docs "Plot types" gallery (matplotlib.org/stable/plot_types) and the `hist`, `boxplot` and `scatter` API pages.
#
# **Next:** [Subplots, layout, annotations and styling](03_subplots_layout_annotations_styling.ipynb)

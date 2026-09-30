# %% [markdown]
# # Seaborn: Visual Storytelling
#
# **Why this matters.** A chart that is *correct* can still fail: the analyst sends five coloured lines, the VP asks "so what?", and the decision gets made on a hunch. Your job at work is rarely "make a plot", it is
# "make the **one** point this data supports impossible to miss, and honest about its uncertainty". This notebook practises that loop: start from a messy default chart, name the finding, and redraw so the chart states it.
#
# **You will learn**
# - turn a cluttered chart into a clear one by **naming the finding first**, then applying a short list of design moves
# - pick a chart type from the **question** being asked (compare, trend, distribution, relationship, composition)
# - use grey-plus-one-accent colour, a title that states the finding, direct labels and sensible ordering
# - show uncertainty and group sizes so the clear chart is also the honest chart
# - audit a chart programmatically for missing titles, labels and overloaded legends
#
# **Prerequisites:** [Matrix and grid plots](03_matrix_and_grid_plots.ipynb) (and, for the matplotlib side, [Anatomy of a figure](../07_Matplotlib/01_anatomy_figure_axes.ipynb)).
#
# > Data used: the synthetic shop tables (`orders`, `order_items`, `products`) and `churn.csv` (see `docs/DATASETS.md`). Nothing is downloaded.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams["figure.max_open_warning"] = 0
sns.set_theme(style="whitegrid")
DATA = Path("../datasets")
GREY, ACCENT = "#aab4be", "#d1495b"

orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
items = pd.read_csv(DATA / "order_items.csv")
products = pd.read_csv(DATA / "products.csv")
lines = items.merge(orders, on="order_id").merge(products, on="product_id")
lines["revenue"] = lines["quantity"] * lines["unit_price"]
done = lines[lines["status"].isin(["delivered", "shipped"])].copy()   # completed orders only
done["month"] = done["order_date"].dt.to_period("M").dt.to_timestamp()

churn = pd.read_csv(DATA / "churn.csv")
churn["contract"] = churn["contract"].str.lower().map({"month-to-month": "Month-to-month", "one year": "One year", "two year": "Two year"})
print(f"{len(done):,} completed order lines, {done['month'].nunique()} months; {len(churn):,} customers")

# %% [markdown]
# ## 1. The method: finding first, chart second
# Before touching code, write the **sentence the chart must prove**. Then every design decision is a test: does it make the sentence clearer or louder than the rest?
#
# | Step | Question to ask | Typical move |
# |---|---|---|
# | 1. Finding | What is the one thing I want the reader to conclude? | Write it as the title |
# | 2. Question type | Compare, trend, distribution, relationship or composition? | Choose the chart type (section 5) |
# | 3. Emphasis | What should the eye hit first? | Grey for context, **one** accent colour for the finding |
# | 4. Remove | What is decoration or repeated? | Drop redundant legend, gridlines, borders, 3-D, rainbow palettes |
# | 5. Label | Can the reader skip the legend and axis guessing? | Direct labels, units on axes, sorted categories |
# | 6. Honesty | Does it show uncertainty and sample size? | Error bars, `n` in labels, zero baseline for bars |
#
# A tiny helper lets us *measure* a chart's hygiene instead of arguing about it. It inspects a matplotlib `Axes`:

# %%
def audit(ax):
    """Return simple hygiene facts about an Axes (title, labels, legend size, distinct bar colours)."""
    legend = ax.get_legend()
    bar_colors = {tuple(np.round(p.get_facecolor(), 3)) for p in ax.patches if isinstance(p, plt.Rectangle) and p.get_height() != 0 and p.get_width() != 0}
    return {
        "title": bool(ax.get_title()),
        "x_label": bool(ax.get_xlabel()),
        "y_label": bool(ax.get_ylabel()),
        "legend_entries": len(legend.get_texts()) if legend else 0,
        "distinct_bar_colors": len(bar_colors),
        "pie_wedges": sum(not isinstance(p, plt.Rectangle) for p in ax.patches),
    }

# %% [markdown]
# ## 2. Case 1: from five lines to one message (trend)
# **The messy default.** Monthly revenue by category: five coloured lines, a legend the eye has to decode, no title, no units.

# %%
by_cat = done.pivot_table(index="month", columns="category", values="revenue", aggfunc="sum")
long = by_cat.reset_index().melt(id_vars="month", var_name="category", value_name="revenue")

fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
sns.lineplot(data=long, x="month", y="revenue", hue="category", ax=ax)
plt.show()
print("BEFORE:", audit(ax))

# %% [markdown]
# *What is wrong?* You can see that everything spikes at some point, but which category? By how much? The chart makes the reader do the analysis. Find the finding in numbers first:

# %%
total = done.groupby("month")["revenue"].sum()
peak_share = total[total.index.month.isin([11, 12])].sum() / total.sum()
peak_months = int(total.index.month.isin([11, 12]).sum())
typical = total[~total.index.month.isin([11, 12])].mean()
peak_avg = total[total.index.month.isin([11, 12])].mean()
print(f"Nov+Dec are {peak_months} of {len(total)} months but {peak_share:.0%} of revenue; average peak month = {peak_avg / typical:.1f}x the average other month")

# %% [markdown]
# **The finding:** *"Two months a year (Nov and Dec) bring a disproportionate share of revenue"* (the exact share is in the title, computed above). The category split is a distraction for this message, so the clear chart shows **total** revenue by month, with Nov/Dec in the accent colour and everything else grey.

# %%
def peak_bars(total, figsize=(9, 3.8)):
    """Monthly revenue bars: Nov/Dec in the accent colour, other months grey; returns (fig, ax)."""
    months = total.index.strftime("%b %y")
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    sns.barplot(x=months, y=total.values / 1000, color=GREY, ax=ax)
    for patch, month in zip(ax.patches, total.index):
        if month.month in (11, 12):
            patch.set_facecolor(ACCENT)
    ax.set_xticks(range(0, len(months), 2))
    ax.set_xticklabels(months[::2], rotation=45, ha="right")
    ax.set(title=f"Nov and Dec bring {peak_share:.0%} of revenue in {peak_months} of {len(total)} months",
           xlabel="Month", ylabel="Revenue (USD thousands)")
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)
    return fig, ax

fig, ax = peak_bars(total)
plt.show()
print("AFTER: ", audit(ax))

# %% [markdown]
# *What just happened?* The title **is** the finding (a reader who only reads titles still gets it). One accent colour marks the evidence, there is no legend to decode, the bars start at zero (length encodes value honestly), axes carry units, and the top/right frame is gone.
# We *removed* the category detail because it did not serve this message. It is not lost: a second chart (or a filter in a dashboard) can answer "which categories drive the peak?".

# %% [markdown]
# ## 3. Case 2: a pie answering the wrong question (comparison)
# **Question:** *which contract type is the churn problem?* The messy attempt: a pie of where churned customers come from.

# %%
churners = churn[churn["churned"] == 1]
fig, ax = plt.subplots(figsize=(4.5, 4), layout="constrained")
ax.pie(churners["contract"].value_counts(), labels=churners["contract"].value_counts().index, autopct="%1.0f%%")
plt.show()
print("BEFORE:", audit(ax))

# %% [markdown]
# A pie of *churners* shows **composition of the lost customers**, which mostly mirrors how many customers each contract has. The question was about **risk**, which needs a **rate**: of the customers on each contract, what fraction left?
# Compare the two readings:

# %%
summary = churn.groupby("contract").agg(customers=("churned", "size"), churners=("churned", "sum"), rate=("churned", "mean"))
summary["share_of_churners"] = summary["churners"] / summary["churners"].sum()
summary["share_of_customers"] = summary["customers"] / summary["customers"].sum()
summary.round(3)

# %% [markdown]
# Month-to-month holds a large share of the churners partly because it holds a large share of *customers*; the rate column is what compares risk. A horizontal bar, sorted, with the value printed at the end of each bar, plus the **overall churn rate as a reference line** so the reader sees "above or below average":

# %%
def rate_bars(summary, overall):
    order = summary.sort_values("rate", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 2.8), layout="constrained")
    sns.barplot(x=order["rate"] * 100, y=order.index, color=GREY, ax=ax)
    ax.patches[0].set_facecolor(ACCENT)                                   # the highest bar is the finding
    for patch, (name, row) in zip(ax.patches, order.iterrows()):
        ax.text(patch.get_width() + 1, patch.get_y() + patch.get_height() / 2, f"{row['rate']:.0%}  (n={int(row['customers']):,})", va="center")
    ax.axvline(overall * 100, color="black", linestyle="--", linewidth=1)
    ax.text(overall * 100 + 0.5, len(order) - 0.45, f"overall {overall:.0%}", fontsize=8, va="bottom")
    top = order.index[0]
    ax.set(title=f"{top} customers churn at {order['rate'].iloc[0]:.0%}, more than {int(order['rate'].iloc[0] / order['rate'].iloc[-1])}x the {order.index[-1].lower()} rate",
           xlabel="Customers who churned (%)", ylabel="", xlim=(0, 75))
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True)
    return fig, ax

fig, ax = rate_bars(summary, churn["churned"].mean())
plt.show()
print("AFTER: ", audit(ax))

# %% [markdown]
# *What just happened?* We changed the **question type** (composition to comparison of rates), sorted the categories, labelled bars directly (including `n`), and added a baseline. Sorting is a cheap, powerful move: the order alone tells the ranking.
# Why not keep the pie? Humans compare lengths along a common baseline far more accurately than angles, and with three slices a bar is already clearer.

# %% [markdown]
# ## 4. Case 3: overplotted dots hiding a relationship (uncertainty)
# **Question:** *do customers who call support more often churn more?* `churned` is 0 or 1, so a scatter stacks thousands of dots on two horizontal lines.

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.4), layout="constrained")
sns.scatterplot(data=churn, x="support_calls", y="churned", ax=ax)
plt.show()
print("BEFORE:", audit(ax), "| points drawn:", len(churn))

# %% [markdown]
# Every customer with the same call count sits on exactly the same pixel, so the picture says almost nothing. Summarise: the **mean of a 0/1 column is a rate**. Bin the rare high counts together (a group of 4 customers has a meaningless rate) and show a confidence interval and the group sizes:

# %%
churn["calls_band"] = churn["support_calls"].clip(upper=5).map(lambda v: "5+" if v >= 5 else str(v))
order_calls = ["0", "1", "2", "3", "4", "5+"]
counts = churn["calls_band"].value_counts().reindex(order_calls)
print("customers per band:", counts.to_dict())

fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
sns.pointplot(data=churn, x="calls_band", y="churned", order=order_calls, errorbar=("ci", 95), seed=42, color=ACCENT, ax=ax)
ax.set_xticks(range(len(order_calls)))
ax.set_xticklabels([f"{b}\n(n={counts[b]:,})" for b in order_calls])
ax.axhline(churn["churned"].mean(), color="grey", linestyle="--", linewidth=1)
ax.text(len(order_calls) - 0.6, churn["churned"].mean() + 0.01, "overall", color="grey", fontsize=8, ha="right")
ax.set(title="Customers with more support calls churn more, and the gap is well beyond noise",
       xlabel="Support calls (customers per group below)", ylabel="Churn rate (fraction of customers)", ylim=(0, 1))
sns.despine(ax=ax)
plt.show()
print("AFTER: ", audit(ax))

# %% [markdown]
# *What just happened?* Aggregating turned an unreadable cloud into a trend; the **95% bootstrap interval** shows how sure we are (intervals are wider where `n` is small); printing `n` under each tick stops anyone over-reading a thin group.
# Careful with the wording, though: this is an **association**. Customers with problems may call more *and* leave more, and the chart cannot tell us that calls *cause* churn. A title can state a finding without claiming a cause.

# %% [markdown]
# ## 5. Choose the chart by the question
# Start from what the reader wants to know, not from what the library can draw.
#
# | Question | Good default | Why | Avoid |
# |---|---|---|---|
# | How do categories **compare**? | sorted bar (horizontal if labels are long) | length on a shared baseline is easiest to judge | pie with many slices, 3-D |
# | How does something **change over time**? | line (or bars for few, discrete periods) | slope shows direction and speed | many overlapping lines; start with one |
# | What is the **distribution** of one number? | histogram, KDE, ECDF or box/violin per group | shape, spread, outliers | a bar of the mean alone |
# | How do two numbers **relate**? | scatter (hexbin/alpha when dense), optionally a fit line | pattern, clusters, outliers | correlation coefficient without the plot |
# | What is the **composition** of a whole? | stacked bar over a few parts, or a sorted bar of shares | parts sum to 100% | pie charts with small differences |
# | How do **many groups** differ across **two categories**? | heatmap with annotations and `n` | dense, sortable | rainbow palettes |
#
# One figure with four of these, each titled with its *question*:

# %%
housing = pd.read_csv(DATA / "housing.csv")
homes = housing[(housing["price"] / housing["area_sqft"]).between(20, 500)]
order_value = done.groupby("order_id")["revenue"].sum()
cat_rev = done.groupby("category")["revenue"].sum().sort_values(ascending=False) / 1000

fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), layout="constrained")
sns.barplot(x=cat_rev.values, y=cat_rev.index, color=GREY, ax=axes[0, 0])
axes[0, 0].set(title="Compare: which category earns most?", xlabel="Revenue (USD thousands)", ylabel="")
sns.lineplot(x=total.index, y=total.values / 1000, color=ACCENT, ax=axes[0, 1])
axes[0, 1].set(title="Trend: how does revenue move over time?", xlabel="Month", ylabel="Revenue (USD thousands)")
sns.histplot(order_value, bins=40, color=GREY, ax=axes[1, 0])
axes[1, 0].set(title="Distribution: what does an order look like?", xlabel="Order value (USD)", ylabel="Orders (count)")
axes[1, 1].hexbin(homes["area_sqft"], homes["price"] / 1000, gridsize=30, cmap="mako_r", mincnt=1)
axes[1, 1].set(title="Relationship: does price rise with area?", xlabel="Area (sq ft)", ylabel="Price (USD thousands)")
axes[0, 1].tick_params(axis="x", rotation=30)
plt.show()
print(f"order values: median {order_value.median():.0f}, mean {order_value.mean():.0f} (right-skewed: the mean sits above the median)")

# %% [markdown]
# ## 6. The honesty checklist
# Clear is not the same as persuasive-at-any-cost. Before sending a chart:
# - **Bars start at zero.** Truncating the axis turns a 5% difference into a giant gap. (Lines and dots may use a non-zero baseline, but say so.)
# - **Uncertainty is visible** when groups are small or differences are modest (error bars, `n`).
# - **The denominator is named**: "% of customers" is not "% of churners".
# - **Emphasis matches evidence.** Highlight what the data supports, not what you hoped for.
# - **Colour is not the only carrier of meaning**: accent vs grey also differs in position and label, so it survives a greyscale print or colour-blind reader.
#
# A quick demonstration of the first rule, using our own numbers: the same two bars with an honest and a truncated axis.

# %%
cat2 = done.groupby("category")["revenue"].sum()
a, b = cat2["Home"], cat2["Sports"]
print(f"Sports / Home revenue ratio = {b / a:.2f}  (a {b / a - 1:.0%} difference)")
fig, axes = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
for ax, ylim, label in zip(axes, [(0, a * 1.2 if a > b else b * 1.2), (min(a, b) * 0.95, max(a, b) * 1.02)], ["Honest: starts at 0", "Misleading: truncated axis"]):
    sns.barplot(x=["Home", "Sports"], y=[a / 1000, b / 1000], color=GREY, ax=ax)
    ax.set(ylim=(ylim[0] / 1000, ylim[1] / 1000), title=label, ylabel="Revenue (USD thousands)", xlabel="Category")
plt.show()

# %% [markdown]
# ## Common mistakes
#
# 1. **Starting with the chart type instead of the question.** (Five lines because the data had five categories.)
# 2. **A title that names the axes** ("Revenue by month") instead of the finding ("Nov and Dec bring 30% of revenue", with the number computed).
# 3. **Rainbow / default palettes for ordered or highlighted data.** Use grey plus one accent, or a sequential palette for ordered values.
# 4. **Pies and stacked areas for comparisons** that need precise reading.
# 5. **Unsorted categories**: alphabetical order is rarely the order of interest.
# 6. **Truncated bar axes** and **dual y-axes** that let you fake a correlation.
# 7. **Hiding sample size**: a bold bar on 12 customers.
# 8. **Causal titles for observational data** ("calls cause churn").

# %%
# Mistake 7: how wide is the uncertainty when n is small?
rng = np.random.default_rng(42)
p = churn["churned"].mean()
for n in (12, 120, 1200):
    sample = rng.choice(churn["churned"].to_numpy(), size=n, replace=False)
    se = np.sqrt(sample.mean() * (1 - sample.mean()) / n)
    print(f"n={n:5d}: observed rate {sample.mean():.0%}, approx. 95% margin of error +/- {1.96 * se:.0%}")

# %% [markdown]
# ## Interview corner
#
# **1. You have five categories over 24 months. Do you draw five lines?**
# <details><summary>Answer</summary>
# Only if the question is about comparing all five trends. Usually I first decide the message: one highlighted line with the others in grey, small multiples with a shared y-axis, or a total plus a separate breakdown chart. Five coloured lines put the analysis on the reader.
# </details>
#
# **2. Why are pie charts discouraged?**
# <details><summary>Answer</summary>
# Readers judge angles and areas less accurately than lengths on a common baseline, slices are hard to rank when similar, and labels crowd. A sorted bar with direct labels is almost always clearer. A pie is tolerable for two or three very different parts of one whole.
# </details>
#
# **3. A stakeholder wants the y-axis to start at 95 "so the difference is visible". What do you say?**
# <details><summary>Answer</summary>
# For bar charts, a non-zero baseline distorts because bar length encodes the value. Better: show the difference explicitly (a delta label, or plot the change), use a dot or line chart where a non-zero baseline is acceptable and is labelled, or show the confidence interval so the reader can judge whether a small gap matters.
# </details>
#
# **4. How do you make sure a chart does not overclaim?**
# <details><summary>Answer</summary>
# Show uncertainty (intervals) and sample sizes, name the denominator, avoid causal wording for observational data, use zero baselines for bars, and do not highlight a difference the interval cannot support.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Highlight the peak months
# Using `peak_bars`-style logic, build the chart for the **top 3 revenue months** instead: a bar per month of `total` with only the three highest months in `ACCENT` and the rest in `GREY`.
# Store the figure's axes in `ax1`, and the list of the three top months (as `"%b %y"` strings, highest first) in `top3`. The title must state the finding and the axis labels must have units.
# *Hint: `total.nlargest(3)`; the bar patches are in the same order as `total.index`.*

# %% tags=["starter"]
ax1 = ...
top3 = ...

# %%
_m = total.nlargest(3)
_exp = list(_m.index.strftime("%b %y"))
assert list(top3) == _exp, f"expected top three months {_exp}"
_bars = [p for p in ax1.patches if p.get_height() > 0]
assert len(_bars) == len(total), f"one bar per month ({len(total)})"
_acc = [i for i, p in enumerate(_bars) if np.allclose(p.get_facecolor()[:3], plt.matplotlib.colors.to_rgb(ACCENT))]
assert sorted(total.index[_acc]) == sorted(_m.index), "only the three highest months should use ACCENT"
_a = audit(ax1)
assert _a["title"] and _a["x_label"] and _a["y_label"], "title and both axis labels are required"
assert "usd" in ax1.get_ylabel().lower(), "y label needs units"
assert _a["legend_entries"] == 0, "no legend needed"
print("Exercise 1 passed:", _exp)

# %% tags=["solution"]
_top = total.nlargest(3)
top3 = list(_top.index.strftime("%b %y"))
_fig, ax1 = plt.subplots(figsize=(9, 3.8), layout="constrained")
sns.barplot(x=total.index.strftime("%b %y"), y=total.values / 1000, color=GREY, ax=ax1)
for _p, _mo in zip(ax1.patches, total.index):
    if _mo in _top.index:
        _p.set_facecolor(ACCENT)
ax1.set_xticks(range(0, len(total), 2))
ax1.set_xticklabels(total.index.strftime("%b %y")[::2], rotation=45, ha="right")
ax1.set(title=f"The three best months ({', '.join(top3)}) are all in the Nov-Dec peak" if all(m.month in (11, 12) for m in _top.index) else f"Top three months: {', '.join(top3)}",
        xlabel="Month", ylabel="Revenue (USD thousands)")
sns.despine(ax=ax1)

# %% [markdown]
# ### Exercise 2: Risk by channel, sorted and labelled
# **Business question:** *which sales channel has the highest cancellation-or-return rate?* Use `orders` (column `status`; a "problem" order has status `cancelled` or `returned`).
# Compute `problem_rate` (a Series indexed by `channel`, fraction of orders, **sorted descending**), then draw a horizontal bar chart on `ax2` with the highest bar in `ACCENT`, the rate printed next to each bar, and a title that states the finding.

# %% tags=["starter"]
problem_rate = ...
ax2 = ...

# %%
_pr = orders["status"].isin(["cancelled", "returned"]).groupby(orders["channel"]).mean().sort_values(ascending=False)
assert isinstance(problem_rate, pd.Series) and np.allclose(problem_rate.values, _pr.values) and list(problem_rate.index) == list(_pr.index), \
    "problem_rate should be the share of cancelled/returned orders per channel, sorted descending"
_bars = [p for p in ax2.patches if p.get_width() > 0]
assert len(_bars) == len(_pr), "one bar per channel"
assert [t.get_text() for t in ax2.get_yticklabels()] == list(_pr.index), "channels should run top to bottom from highest to lowest rate"
assert np.allclose(_bars[0].get_facecolor()[:3], plt.matplotlib.colors.to_rgb(ACCENT)), "highlight the highest bar only"
assert not np.allclose(_bars[-1].get_facecolor()[:3], plt.matplotlib.colors.to_rgb(ACCENT)), "other bars must not use the accent colour"
assert len(ax2.texts) >= len(_pr), "print each rate next to its bar (ax.text)"
assert ax2.get_title() and ax2.get_xlabel(), "title and x label required"
print(f"Exercise 2 passed: highest problem rate is {_pr.index[0]} at {_pr.iloc[0]:.1%}")

# %% tags=["solution"]
problem_rate = orders["status"].isin(["cancelled", "returned"]).groupby(orders["channel"]).mean().sort_values(ascending=False)
_fig, ax2 = plt.subplots(figsize=(6.5, 2.6), layout="constrained")
sns.barplot(x=problem_rate.values * 100, y=problem_rate.index, color=GREY, ax=ax2)
ax2.patches[0].set_facecolor(ACCENT)
for _p, _v in zip(ax2.patches, problem_rate.values):
    ax2.text(_p.get_width() + 0.2, _p.get_y() + _p.get_height() / 2, f"{_v:.1%}", va="center")
ax2.set(title=f"{problem_rate.index[0].title()} has the highest cancelled-or-returned rate ({problem_rate.iloc[0]:.1%})",
        xlabel="Orders cancelled or returned (%)", ylabel="")
sns.despine(ax=ax2, left=True)

# %% [markdown]
# ### Exercise 3: Choose the chart, then draw it with uncertainty
# **Business question:** *does churn rise with monthly charges?* `monthly_charges` is numeric and `churned` is binary. Create `charge_band` = `pd.qcut(churn["monthly_charges"], 5, labels=False)` (0 = cheapest fifth to 4 = most expensive fifth; drop rows with missing charges first, working on a copy named `c3`).
# Store `rate_by_band` (mean churn per band as a Series indexed 0..4) and draw a `sns.pointplot` of churn by band with a 95% confidence interval on `ax3`. Set `rises` to `True` if the churn rate of band 4 is above band 0.
# Title, both axis labels (with units or a description of the band) required.

# %% tags=["starter"]
c3 = ...
rate_by_band = ...
ax3 = ...
rises = ...

# %%
_c = churn.dropna(subset=["monthly_charges"]).copy()
_c["b"] = pd.qcut(_c["monthly_charges"], 5, labels=False)
_r = _c.groupby("b")["churned"].mean()
assert len(c3) == len(_c), "drop rows with missing monthly_charges (and only those)"
assert np.allclose(rate_by_band.sort_index().values, _r.sort_index().values), "rate_by_band: mean churned per quintile"
assert list(rate_by_band.index) == [0, 1, 2, 3, 4]
assert rises == bool(_r.loc[4] > _r.loc[0]), "rises compares band 4 with band 0"
assert np.allclose(np.asarray(ax3.lines[0].get_ydata(), dtype=float), _r.values), "draw sns.pointplot of churn by band (the plotted points should be the band means)"
assert len(ax3.lines) >= 1 + 5, "pointplot draws the means line plus one interval per band (keep errorbar on)"
assert ax3.get_title() and ax3.get_xlabel() and ax3.get_ylabel()
print("Exercise 3 passed:", {int(k): round(float(v), 3) for k, v in _r.items()}, "| rises:", rises)

# %% tags=["solution"]
c3 = churn.dropna(subset=["monthly_charges"]).copy()
c3["charge_band"] = pd.qcut(c3["monthly_charges"], 5, labels=False)
rate_by_band = c3.groupby("charge_band")["churned"].mean()
_fig, ax3 = plt.subplots(figsize=(6.5, 3.4), layout="constrained")
sns.pointplot(data=c3, x="charge_band", y="churned", errorbar=("ci", 95), seed=42, color=ACCENT, ax=ax3)
ax3.set(title="Churn rate by monthly-charge quintile", xlabel="Monthly charge band (0 = cheapest fifth, 4 = most expensive)", ylabel="Churn rate (fraction of customers)", ylim=(0, 1))
rises = bool(rate_by_band.loc[4] > rate_by_band.loc[0])

# %% [markdown]
# ### Exercise 4 (open-ended): Your own before and after
# Pick one question about the shop data (revenue by country, by channel, premium vs non-premium customers, returns by category, ...).
# 1. Draw the **messy first attempt** (default palette, no title) and run `audit` on it.
# 2. Write the finding as one sentence, back it with a computed number.
# 3. Draw the **clear version** following section 1's checklist and run `audit` again.
# 4. In two sentences, say which design moves mattered most and what the chart deliberately leaves out.

# %% [markdown]
# ## Recap
# - **Finding first.** Write the sentence, use it as the title, and let each design choice serve it.
# - **Choose the chart from the question**: compare (sorted bar), trend (line), distribution (hist/ECDF), relationship (scatter/hexbin), composition (few-part bar).
# - **Emphasis**: grey for context, one accent for the evidence; remove legends, borders and decoration you do not need; direct-label what you can.
# - **Honesty**: zero baseline for bars, intervals and `n` for groups, name the denominator, association is not causation.
# - Measure hygiene (`audit`) instead of debating it, and keep before/after pairs: they are the best portfolio pieces.
#
# **Go deeper:** seaborn docs, "Controlling figure aesthetics" and "Choosing color palettes" (seaborn.pydata.org); the matplotlib documentation on `Axes.annotate` and `Axes.text`; Claus Wilke, *Fundamentals of Data Visualization* (free online book); Edward Tufte, *The Visual Display of Quantitative Information*.
#
# **Next:** this completes the seaborn module. Continue with [Snowflake](../09_Snowflake/) or go back and redraw a chart from your own work using this checklist. Real public data to practise on: Kaggle retail datasets, the UCI Machine Learning Repository, and `sklearn.datasets` bundled sets (for example `load_wine`).

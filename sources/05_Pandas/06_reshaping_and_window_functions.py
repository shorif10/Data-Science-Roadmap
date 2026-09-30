# %% [markdown]
# # Reshaping and Window Functions
#
# **Why this matters.** Two very different jobs, one notebook. *Reshaping* (pivot, melt, crosstab) turns a long list of transactions into the
# month-by-category grid your manager wants in a report, or back into the tidy long format a plotting library needs. *Window functions*
# (rolling, expanding, shift, rank, cumulative) answer "compared with the previous period", "3-month average", "running total" and "top N in each group",
# which is most of business analytics.
#
# **You will learn**
# - build cross-tabulations with `pivot_table` and `crosstab`, and go back to long format with `melt`
# - move index levels with `stack` / `unstack`
# - compute rolling and expanding statistics, `shift` / `diff` / `pct_change`, ranks and cumulative sums
# - avoid look-ahead mistakes with windows
#
# **Prerequisites:** [Merge, join and concat](05_merge_join_concat.ipynb).
#
# > Data used: the shop tables (synthetic, see `docs/DATASETS.md`). Revenue here means `quantity * unit_price` of order lines
# > **excluding cancelled orders** (a cancelled order produced no sale). Returned orders are kept as *gross* revenue.

# %%
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("../datasets")
customers = pd.read_csv(DATA / "customers.csv")
products = pd.read_csv(DATA / "products.csv")
orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
order_items = pd.read_csv(DATA / "order_items.csv")

# one row per order line, with product, order and customer attributes (see the merge notebook for the checks)
sales = (
    order_items.merge(products, on="product_id", how="left", validate="many_to_one")
    .merge(orders, on="order_id", how="left", validate="many_to_one")
    .merge(customers[["customer_id", "country", "is_premium"]], on="customer_id", how="left", validate="many_to_one")
)
assert len(sales) == len(order_items) and sales["unit_price"].notna().all()
sales = sales.assign(revenue=sales["quantity"] * sales["unit_price"], month=sales["order_date"].dt.to_period("M"))
sold = sales[sales["status"] != "cancelled"]          # boolean filter: a new frame, `sales` is untouched
print(sales.shape, "->", sold.shape, "after dropping cancelled orders")

# %% [markdown]
# ## 1. `pivot_table`: long to wide with aggregation
# `pivot_table(index=..., columns=..., values=..., aggfunc=...)` groups by two keys and lays one of them out as columns. It is a `groupby` with a
# presentation step. `fill_value` replaces empty combinations, and `margins=True` adds totals.

# %%
pt = sold.pivot_table(index="channel", columns="category", values="revenue", aggfunc="sum", fill_value=0, margins=True, margins_name="Total")
pt.round(0).astype(int)

# %% [markdown]
# *What just happened?* Each cell is the revenue of one (channel, category) pair; the `Total` row and column are the margins. Cross-check: the grand total must equal the plain sum.

# %%
assert abs(pt.loc["Total", "Total"] - sold["revenue"].sum()) < 1e-6
print("grand total matches the direct sum:", round(pt.loc["Total", "Total"], 2))

# %% [markdown]
# `pivot` (without "table") does **no aggregation** and raises an error if an (index, column) pair appears more than once. That error is a feature: it tells you the data is not at the grain you thought.

# %%
try:
    sold.pivot(index="channel", columns="category", values="revenue")
except ValueError as e:
    print("ValueError:", str(e)[:70])

# %% [markdown]
# ## 2. `melt`: wide back to long
# Plotting libraries and `groupby` prefer **long (tidy)** data: one row per observation, one column per variable. `melt` unpivots wide tables.

# %%
month_cat = sold.pivot_table(index="month", columns="category", values="revenue", aggfunc="sum", fill_value=0)
print(month_cat.shape, "(months x categories)")
long = month_cat.reset_index().melt(id_vars="month", var_name="category", value_name="revenue")
print(long.shape, "= months x categories rows")
print(long.head(3).to_string(index=False))
assert abs(long["revenue"].sum() - month_cat.to_numpy().sum()) < 1e-6, "melt must not change the totals"

# %% [markdown]
# ## 3. `stack` and `unstack`: moving index levels
# `unstack` pivots the **innermost index level into columns**; `stack` does the reverse. They are the index-based cousins of pivot and melt.

# %%
by_two = sold.groupby(["channel", "status"])["revenue"].sum()      # Series with a 2-level index
print(by_two.head(4).round(0).to_string())
wide = by_two.unstack("status", fill_value=0)
print(wide.round(0).to_string())
print(wide.stack().shape, "-> back to a 2-level Series of length", len(wide.stack()))

# %% [markdown]
# ## 4. `crosstab`: counts and proportions
# `crosstab` is the fastest way to answer "how are two categorical columns related?". `normalize="index"` turns counts into **row percentages**
# (each row sums to 1), which is what you usually want when comparing groups of different sizes.

# %%
counts = pd.crosstab(orders["channel"], orders["status"])
print(counts.to_string())
share = pd.crosstab(orders["channel"], orders["status"], normalize="index")
print((share * 100).round(1).to_string())
print("rows sum to 1:", bool(np.allclose(share.sum(axis=1), 1)))

# %% [markdown]
# Always report the counts next to the percentages: a difference in return rate between two channels means little if one channel has 30 orders.

# %% [markdown]
# ## 5. Rolling windows and time comparisons
# First a monthly revenue series (one value per month) to practise on.

# %%
monthly = sold.groupby("month")["revenue"].sum()
print(len(monthly), "months from", monthly.index.min(), "to", monthly.index.max())
print((monthly / 1000).round(0).astype(int).head(6).to_string())

# %% [markdown]
# **`rolling(window)`** computes a statistic over the last N rows. The first N-1 results are `NaN` unless you allow `min_periods`.
# A 3-month rolling mean smooths noise; compare it with the raw series.

# %%
roll = pd.DataFrame({
    "revenue": monthly,
    "roll3_mean": monthly.rolling(3).mean(),
    "roll3_loose": monthly.rolling(3, min_periods=1).mean(),
})
print((roll / 1000).round(0).head(4).to_string())

# %% [markdown]
# **`shift(k)`** moves values down k rows (so each row sees the *previous* period); `diff()` is `x - x.shift(1)`; `pct_change()` is `x / x.shift(1) - 1`.
# `shift(12)` on monthly data gives the same month last year: year-over-year growth. Seasonal businesses should compare with last year, not last month.

# %%
growth = pd.DataFrame({
    "revenue": monthly,
    "mom_pct": monthly.pct_change() * 100,
    "yoy_pct": (monthly / monthly.shift(12) - 1) * 100,
})
print(growth.round(1).iloc[[0, 1, 5, 6, 12, 17, 18]].to_string())
print("first month of history has no previous month -> NaN; first 12 months have no year-ago comparison")

# %% [markdown]
# Which month-on-month changes are the biggest? Compare with what a seasonal (holiday) pattern would predict:

# %%
top_mom = growth["mom_pct"].sort_values(ascending=False).head(3).round(1)
print(top_mom.to_string())

# %% [markdown]
# **`expanding()`** uses all rows so far (a growing window): running max, running average. **`cumsum`**, **`cummax`** are the cumulative shortcuts.

# %%
run = pd.DataFrame({"revenue": monthly, "cum_revenue": monthly.cumsum(), "best_so_far": monthly.expanding().max()})
print((run / 1000).round(0).iloc[[0, 5, 11, 23]].to_string())
assert np.isclose(run["cum_revenue"].iloc[-1], monthly.sum())

# %% [markdown]
# **Rank** answers "what position is each row in?" Use `groupby(...).rank()` for "rank within group"; `method` decides ties (`"average"`, `"min"`, `"first"`, `"dense"`).

# %%
cat_month = long.copy()
cat_month["rank_in_month"] = cat_month.groupby("month")["revenue"].rank(ascending=False, method="min").astype(int)
leaders = cat_month[cat_month["rank_in_month"] == 1]["category"].value_counts()
print("how many months each category ranked #1:")
print(leaders.to_string())

# %% [markdown]
# **Time-based windows.** With a `DatetimeIndex`, `rolling("7D")` means "the last seven calendar days", which stays correct when days are missing (unlike `rolling(7)`, which means seven *rows*).

# %%
daily = sold.set_index("order_date").sort_index()["revenue"].resample("D").sum()
print(len(daily), "calendar days")
print(daily.rolling("7D").mean().tail(3).round(0).to_string())

# %% [markdown]
# ## Common mistakes
#
# 1. **Look-ahead with `center=True`**: a centred window uses *future* values. Fine for smoothing a chart, wrong for anything you will use to predict.
# 2. **`shift` direction confusion.** `shift(1)` gives the previous row's value (shifts data *down*). Use `shift(-1)` for the *next* row.
# 3. **Computing `pct_change` on unsorted or gap-filled data.** Sort first; missing months make "previous row" the wrong month. Reindex to a full calendar first.
# 4. **`pivot` on non-unique keys** (raises) or `pivot_table` with the default `aggfunc="mean"` when you wanted sums.
# 5. **Reading percentages without counts** in `crosstab`.
# 6. **Ranking with ties and ignoring `method`.**
# 7. **Rolling across groups.** `df["x"].rolling(3).mean()` on stacked groups mixes them at the boundaries; use `groupby(...).rolling` or `transform`.

# %%
# Mistake 7 in action: two "customers" stacked in one column
stacked = pd.DataFrame({"cust": ["a", "a", "a", "b", "b", "b"], "x": [1, 2, 3, 100, 200, 300]})
print("naive rolling(2):", stacked["x"].rolling(2).mean().tolist())
print("per group       :", stacked.groupby("cust")["x"].transform(lambda s: s.rolling(2).mean()).tolist())

# %% [markdown]
# ## Interview corner
#
# **1. Difference between `pivot` and `pivot_table`?**
# <details><summary>Answer</summary>
# <code>pivot</code> only reshapes and fails when an index/column pair repeats; <code>pivot_table</code> aggregates duplicates with <code>aggfunc</code> (default mean), supports <code>fill_value</code> and margins.
# </details>
#
# **2. How do you compute year-over-year growth for monthly data?**
# <details><summary>Answer</summary>
# Make sure the series is sorted with one row per month (no gaps), then <code>s / s.shift(12) - 1</code>. Explicit shifting is easy to verify; the first 12 months come out as NaN.
# </details>
#
# **3. Get the top-3 rows within each group.**
# <details><summary>Answer</summary>
# <code>df.sort_values("v", ascending=False).groupby("g").head(3)</code>, or rank within the group and filter <code>rank &lt;= 3</code> (choose the tie <code>method</code> deliberately).
# </details>
#
# **4. `rolling(7)` versus `rolling("7D")`?**
# <details><summary>Answer</summary>
# The integer window counts rows; the string window is time-based (needs a DatetimeIndex) and covers seven calendar days regardless of missing rows.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Channel by category, as shares
# From `sold`, build `mix`: a DataFrame with `channel` as index and `category` as columns, values = **that channel's revenue share** by category (each row sums to 1).
# *Hint: `pivot_table` with sum, then divide each row by its own total (`div(..., axis=0)`).*

# %% tags=["starter"]
mix = ...

# %%
assert set(mix.index) == set(sold["channel"]) and set(mix.columns) == set(sold["category"]), "index = channel, columns = category"
assert np.allclose(mix.sum(axis=1), 1), "each row must sum to 1"
ch, cat = "web", "Electronics"
exp = sold[(sold["channel"] == ch) & (sold["category"] == cat)]["revenue"].sum() / sold[sold["channel"] == ch]["revenue"].sum()
assert abs(mix.loc[ch, cat] - exp) < 1e-9, f"web/Electronics share should be {exp:.4f}"
print("Exercise 1 passed")
print(mix.round(3).to_string())

# %% tags=["solution"]
totals = sold.pivot_table(index="channel", columns="category", values="revenue", aggfunc="sum", fill_value=0)
mix = totals.div(totals.sum(axis=1), axis=0)

# %% [markdown]
# ### Exercise 2: Which channel has the highest return rate?
# Using `crosstab` on `orders` (all statuses), compute `return_rate` (Series indexed by channel: the share of that channel's orders with status `returned`) and store the name of the channel with the highest rate in `worst_channel`.

# %% tags=["starter"]
return_rate = ...
worst_channel = ...

# %%
exp = {c: (orders.loc[orders["channel"] == c, "status"] == "returned").mean() for c in orders["channel"].unique()}
assert set(return_rate.index) == set(exp), "one value per channel"
for c, v in exp.items():
    assert abs(return_rate[c] - v) < 1e-9, f"{c}: expected {v:.4f}"
assert worst_channel == max(exp, key=exp.get), f"expected {max(exp, key=exp.get)}"
print(f"Exercise 2 passed (highest: {worst_channel}, {return_rate[worst_channel]:.1%})")

# %% tags=["solution"]
ct = pd.crosstab(orders["channel"], orders["status"], normalize="index")
return_rate = ct["returned"]
worst_channel = return_rate.idxmax()

# %% [markdown]
# ### Exercise 3: Smooth the trend
# Using the `monthly` revenue Series, compute `smooth`: the 3-month rolling **mean** (first two values `NaN`) and `best_smooth_month`: the month (a `Period`) with the highest smoothed value.

# %% tags=["starter"]
smooth = ...
best_smooth_month = ...

# %%
vals = monthly.to_numpy()
manual = np.array([np.nan, np.nan] + [vals[i - 2:i + 1].mean() for i in range(2, len(vals))])
assert np.allclose(smooth.to_numpy(), manual, equal_nan=True), "3-month trailing mean; first two NaN"
assert best_smooth_month == monthly.index[int(np.nanargmax(manual))], "month with the highest smoothed revenue"
print(f"Exercise 3 passed (best smoothed month: {best_smooth_month})")

# %% tags=["solution"]
smooth = monthly.rolling(3).mean()
best_smooth_month = smooth.idxmax()

# %% [markdown]
# ### Exercise 4: Year-over-year for the holiday months
# Compute `yoy` (year-over-year growth as a fraction, e.g. 0.25 for +25%) for each month that has a year-ago month, and store the growth of **December** in `dec_growth`
# (December of the later year compared with the December before).
# *Hint: `shift(12)`, then `dropna`.*

# %% tags=["starter"]
yoy = ...
dec_growth = ...

# %%
idx = list(monthly.index)
exp_pairs = {idx[i]: monthly.iloc[i] / monthly.iloc[i - 12] - 1 for i in range(12, len(idx))}
assert set(yoy.index) == set(exp_pairs), "only months with a year-ago comparison (drop the first 12)"
assert all(abs(yoy[m] - v) < 1e-9 for m, v in exp_pairs.items()), "yoy = revenue / revenue a year earlier - 1"
dec = [m for m in exp_pairs if m.month == 12]
assert len(dec) == 1 and abs(dec_growth - exp_pairs[dec[0]]) < 1e-9, "dec_growth is the December value of yoy"
print(f"Exercise 4 passed (December {dec[0]}: {dec_growth:+.1%})")

# %% tags=["solution"]
yoy = (monthly / monthly.shift(12) - 1).dropna()
dec_growth = yoy[yoy.index.month == 12].iloc[0]

# %% [markdown]
# ### Exercise 5 (open-ended): Seasonal or growing?
# Revenue in the last months of each year jumps. Is that seasonality or just a growing customer base? Use `monthly`, `yoy` and the rolling mean, plus the customer signup counts
# (`customers["signup_date"]`), to argue one way or the other in a markdown cell. Name the one extra piece of data you would want to be sure.

# %% [markdown]
# ## Recap
# - `pivot_table` = groupby + layout (with `fill_value`, `margins`); `pivot` requires unique pairs; `melt` goes back to long; `unstack` / `stack` move index levels.
# - `crosstab(normalize="index")` for row percentages; always keep the counts.
# - `rolling(n)` trailing windows (`NaN` at the start), `expanding()` running statistics, `shift` / `diff` / `pct_change` for period comparisons, `shift(12)` for monthly YoY.
# - `groupby(...).rank(method=...)` for within-group positions; time windows (`"7D"`) survive missing days.
# - Watch for look-ahead (`center=True`), unsorted or gappy data, and windows that cross group boundaries.
#
# **Go deeper:** pandas user guide "Reshaping and pivot tables" and "Windowing operations" (pandas.pydata.org/docs/user_guide).
#
# **Next:** [Time series in pandas](07_time_series_in_pandas.ipynb)

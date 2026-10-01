# %% [markdown]
# # Tableau: Calculated Fields and LOD Expressions
#
# **Why this matters.** The questions stakeholders actually ask ("what share of each customer's lifetime spend is Books?", "revenue by the year a customer first bought",
# "average revenue per customer per category") are not answerable by dragging a field onto a chart. They need a calculation at a **different level of detail** than the chart. Tableau's answer is
# the **LOD expression** (`FIXED`, `INCLUDE`, `EXCLUDE`). They are also the most common Tableau interview topic, because they expose whether you understand *grain*.
#
# **You will learn**
# - tell row-level, aggregate and table calculations apart, and why `SUM(a)/SUM(b)` is not `AVG(a/b)`
# - write `FIXED`, `INCLUDE` and `EXCLUDE` LODs and know the `groupby`/`transform` each one corresponds to
# - explain Tableau's **order of operations** and predict when a filter will or will not affect a LOD
# - avoid the averaging-at-the-wrong-grain trap
#
# **Prerequisites:** [Tableau concepts and data prep](01_tableau_concepts_and_data_prep.ipynb) and pandas `groupby` / `transform`.
#
# > **Candour box.** Tableau was **not run** here. Tableau calculation syntax is in `no-run` cells, written from the documented language (verify function names and the order-of-operations diagram in
# > Tableau Help: help.tableau.com). Every LOD is paired with a **pandas equivalent that runs**, which gives the numbers you should see in a Tableau view built on `exports/order_lines_flat.csv`
# > (from notebook 01). Data: synthetic shop tables (`docs/DATASETS.md`).

# %%
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("../datasets")
customers = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
products = pd.read_csv(DATA / "products.csv")
orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
items = pd.read_csv(DATA / "order_items.csv")

lines = (
    items.merge(orders, on="order_id").merge(products, on="product_id").merge(customers, on="customer_id")
    .assign(revenue=lambda d: d["quantity"] * d["unit_price"], cost=lambda d: d["quantity"] * d["unit_cost"])
)
# Data source filter: completed orders only (delivered or shipped). In Tableau this is applied before everything else below.
done = lines[lines["status"].isin(["delivered", "shipped"])].copy()
print("line rows after the data source filter:", len(done), "of", len(lines))

# %% [markdown]
# ## 1. Three kinds of calculation
#
# | Kind | Evaluated | Example (Tableau syntax) | pandas twin |
# |---|---|---|---|
# | **Row-level** | once per row of the data, *before* aggregation | `[Quantity] * [Unit Price]` | `df["q"] * df["p"]` |
# | **Aggregate** | after aggregating to the view's grain | `SUM([Revenue]) / SUM([Cost])` | `g["rev"].sum() / g["cost"].sum()` |
# | **Table calculation** | on the *query result* shown in the view (window, running, rank) | `RUNNING_SUM(SUM([Revenue]))` | `cumsum`, `rank`, window ops |
#
# The distinction matters because **order changes the answer**. A margin ratio computed per row and then averaged differs from the ratio of the summed components:

# %%
ratio_of_sums = done["revenue"].sum() / done["cost"].sum()                  # aggregate calc: SUM([Revenue]) / SUM([Cost])
avg_of_ratios = (done["revenue"] / done["cost"]).mean()                     # row-level calc [Revenue]/[Cost], then AVG(...)
print(f"SUM(Revenue)/SUM(Cost) = {ratio_of_sums:.4f}  vs  AVG(Revenue/Cost) = {avg_of_ratios:.4f}")

# %% tags=["no-run"]
TABLEAU_CALCS = """
// row-level (one value per data row; then you aggregate it)
Line Profit:   [Quantity] * ([Unit Price] - [Unit Cost])

// aggregate (evaluated at the view's level of detail; do not mix with non-aggregated fields)
Margin %:      (SUM([Revenue]) - SUM([Cost])) / SUM([Revenue])
Orders:        COUNTD([Order ID])
AOV:           SUM([Revenue]) / COUNTD([Order ID])

// table calculations (computed over the results in the view; 'Compute using' decides partition and direction)
Running Rev:   RUNNING_SUM(SUM([Revenue]))
Rev Rank:      RANK(SUM([Revenue]))
% of Total:    SUM([Revenue]) / TOTAL(SUM([Revenue]))

// a few frequently used functions
Month:         DATETRUNC('month', [Order Date])
Safe num:      ZN([Some Measure])            // null -> 0
Cond:          IF [Status] = 'returned' THEN 'Lost' ELSE 'Kept' END
"""

# %% [markdown]
# *What just happened?* The two results differ: the row-level average treats every line alike, while the ratio of sums weights by size. Use **aggregate calculations** for ratios.
# A well-known error message in Tableau, paraphrased, says you cannot mix aggregate and non-aggregate arguments in one expression (for example `SUM([Revenue]) / [Cost]`): decide the level first, then aggregate
# everything consistently (or wrap a dimension in `ATTR()` when you know it has one value per mark).
#
# **Table calculations** operate on what the view already returned and depend on how you partition. Same idea in pandas, with two "Compute using" choices:

# %%
monthly = (done.assign(m=done["order_date"].dt.to_period("M")).groupby("m", as_index=False)["revenue"].sum())
monthly["year"] = monthly["m"].dt.year
monthly["running_all"] = monthly["revenue"].cumsum()                                 # RUNNING_SUM across the whole table
monthly["running_in_year"] = monthly.groupby("year")["revenue"].cumsum()             # restarts each year (partition by Year)
monthly["rank"] = monthly["revenue"].rank(ascending=False, method="min").astype(int)
monthly["pct_of_total"] = monthly["revenue"] / monthly["revenue"].sum()
monthly.iloc[[0, 6, 7, 18, 23]].round(3)

# %% [markdown]
# A table calc needs the **view's rows** to exist: if a filter removes a month from the view, the running total and the percent-of-total change. That is exactly why they sit *late* in the order of operations (section 3).

# %% [markdown]
# ## 2. Level of Detail (LOD) expressions
#
# The **view's level of detail** is the set of dimensions on the shelves (and in the marks card). A regular aggregate is computed at that level. A **LOD expression** computes at a level *you specify*, then the result is
# attached to each row/mark, so you can mix grains in one view. Three forms:
#
# | Form | Meaning | pandas |
# |---|---|---|
# | `{ FIXED [dims] : AGG }` | compute at exactly `dims`, **independent of the view** | `groupby(dims)[x].transform(agg)` on the *unfiltered (except context) data* |
# | `{ INCLUDE [dims] : AGG }` | view's level **plus** `dims` | aggregate at `view + dims`, then aggregate up to the view |
# | `{ EXCLUDE [dims] : AGG }` | view's level **minus** `dims` | aggregate at `view - dims` and repeat it across the dropped dimension |
#
# `{ AGG }` with no dimensions is shorthand for a FIXED at the whole table.

# %% tags=["no-run"]
TABLEAU_LODS = """
Customer Lifetime Revenue:   { FIXED [Customer ID] : SUM([Revenue]) }
First Order Date:            { FIXED [Customer ID] : MIN([Order Date]) }
Cohort Year:                 YEAR({ FIXED [Customer ID] : MIN([Order Date]) })
Orders per Customer:         { FIXED [Customer ID] : COUNTD([Order ID]) }
Revenue per Customer (view): AVG({ INCLUDE [Customer ID] : SUM([Revenue]) })     // view by Category -> avg customer spend in each category
Category Revenue on Product rows:  { EXCLUDE [Product Name] : SUM([Revenue]) }   // view by Category, Product Name
Share of Category:           SUM([Revenue]) / MAX({ EXCLUDE [Product Name] : SUM([Revenue]) })
Grand Total (any view):      { SUM([Revenue]) }
"""

# %% [markdown]
# ### FIXED: customer lifetime revenue
# Whatever the chart looks like (by category, by month, by country), `Customer Lifetime Revenue` is that customer's *total*.

# %%
done["ltv"] = done.groupby("customer_id")["revenue"].transform("sum")           # { FIXED [Customer ID] : SUM([Revenue]) }
ltv = done.groupby("customer_id")["revenue"].sum()
assert np.allclose(done.drop_duplicates("customer_id").set_index("customer_id")["ltv"].sort_index(), ltv.sort_index())
print("customers with revenue:", len(ltv), "| median lifetime revenue:", round(ltv.median(), 2), "| top customer:", int(ltv.idxmax()), round(ltv.max(), 2))

# %% [markdown]
# Useful consequences: **cohorts** (group by the year of a customer's first order, a `FIXED MIN`) and **frequency** (`FIXED COUNTD(order)`), both common "customer" analyses.

# %%
done["first_order"] = done.groupby("customer_id")["order_date"].transform("min")      # { FIXED [Customer ID] : MIN([Order Date]) }
done["cohort_year"] = done["first_order"].dt.year
cohort_rev = done.groupby("cohort_year")["revenue"].sum()
print(cohort_rev.round(2).to_dict(), "| sums to total:", bool(abs(cohort_rev.sum() - done["revenue"].sum()) < 1e-6))

# %% [markdown]
# ### INCLUDE: average customer spend *per category*
# The view has one mark per category. `AVG(SUM(Revenue) per customer)` needs a grain **finer** than the view (customer within category). The naive `AVG([Revenue])` averages *lines*, a different quantity.

# %%
per_cust_cat = done.groupby(["category", "customer_id"])["revenue"].sum()            # { INCLUDE [Customer ID] : SUM([Revenue]) }
avg_customer_spend = per_cust_cat.groupby("category").mean()                          # AVG over those, per category
naive_avg_line = done.groupby("category")["revenue"].mean()                            # AVG([Revenue]): per line
pd.DataFrame({"AVG(INCLUDE customer SUM)": avg_customer_spend, "naive AVG(line revenue)": naive_avg_line}).round(2)

# %% [markdown]
# ### EXCLUDE: share within category
# The view has Category and Product rows. The denominator must be the **category total** (drop Product from the level of detail), repeated on each product row.

# %%
view = done.groupby(["category", "product_name"], as_index=False)["revenue"].sum()
view["cat_total"] = view.groupby("category")["revenue"].transform("sum")             # { EXCLUDE [Product Name] : SUM([Revenue]) }
view["share_of_category"] = view["revenue"] / view["cat_total"]
assert np.allclose(view.groupby("category")["share_of_category"].sum(), 1.0), "shares within each category sum to 1"
view.sort_values("share_of_category", ascending=False).head(3).round(3)

# %% [markdown]
# *What just happened?* Each LOD maps to a `groupby` at a particular key set: FIXED = that key, then `transform` back to rows; INCLUDE = aggregate at finer grain then re-aggregate; EXCLUDE = aggregate at coarser grain and broadcast.
# The skill is **naming the grain** of the numerator and the denominator out loud before writing the calculation.

# %% [markdown]
# ## 3. Order of operations: why a filter sometimes "does nothing"
#
# Tableau applies things in a fixed sequence (verify the official diagram, titled "Order of Operations", in Tableau Help). The part that matters for LODs:
#
# 1. Extract filters, then **data source filters**
# 2. **Context filters**
# 3. **`FIXED` LODs** are computed here
# 4. **Dimension filters** (ordinary filters on dimension fields; sets, conditional and Top N filters have their own slots around here: check the official diagram)
# 5. **`INCLUDE` / `EXCLUDE` LODs**, then aggregation of the view
# 6. Measure filters
# 7. **Table calculations**, then table-calc filters
#
# Consequences: a normal dimension filter is applied **after** `FIXED`, so it **does not change** a FIXED value. `INCLUDE`/`EXCLUDE` are computed after dimension filters, so they **do** respond. To make a filter affect a FIXED expression, add it **to context**.
#
# Demonstration with "share of a customer's lifetime revenue spent on Books". The numerator follows the filter; the FIXED denominator must not:

# %%
books_view = done[done["category"] == "Books"]                                   # dimension filter: Category = Books (applied after FIXED)
num = books_view.groupby("customer_id")["revenue"].sum()                          # SUM([Revenue]) in the filtered view
den = ltv.reindex(num.index)                                                      # { FIXED [Customer ID] : SUM([Revenue]) } : unaffected by the filter
share_books = (num / den)
naive = (num / num)                                                               # no LOD: denominator also filtered -> always 100%
print(f"{len(share_books)} customers bought Books; mean share of their own spend: {share_books.mean():.1%}; max share: {share_books.max():.1%}")
print(f"without the FIXED denominator every customer shows {naive.mean():.0%}: the filter shrank the denominator too")

# If Category = Books were a CONTEXT filter, FIXED would be computed after it, so the lifetime revenue would shrink to Books only:
ltv_in_context = books_view.groupby("customer_id")["revenue"].sum()
c = int(num.index[0])
print(f"customer {c}: Books revenue {num[c]:.2f} | FIXED lifetime (dimension filter) {ltv[c]:.2f} | FIXED lifetime (Books as context filter) {ltv_in_context[c]:.2f}")

# %% [markdown]
# *What just happened?* Same formula, different filter *kind*, different result. When a FIXED number "ignores my filter", it is working as designed. Decide deliberately: dimension filter (LOD keeps the whole-customer view) or context filter (LOD recomputed inside the filtered data).
# Context filters also make Tableau build a temporary subset first, which can cost performance: use them when you need this behaviour, not by default.

# %% [markdown]
# ## 4. The grain trap: averaging a lower-grain number over higher-grain rows
#
# In a line-level table, `{ FIXED [Order ID] : SUM([Revenue]) }` gives every *line* its order's total. If you then take `AVG` of that field, an order with 6 lines counts 6 times and an order with 1 line counts once, so big orders are over-weighted.

# %%
done["order_revenue"] = done.groupby("order_id")["revenue"].transform("sum")         # { FIXED [Order ID] : SUM([Revenue]) }
wrong_aov = done["order_revenue"].mean()                                              # AVG([Order Revenue]) over LINE rows
right_aov = done["revenue"].sum() / done["order_id"].nunique()                        # SUM([Revenue]) / COUNTD([Order ID])
per_order = done.groupby("order_id")["revenue"].sum().mean()                          # view at Order ID level, then average
assert abs(right_aov - per_order) < 1e-9
print(f"AVG of a FIXED order total over lines: {wrong_aov:.2f}   correct AOV: {right_aov:.2f}   (overstated by {wrong_aov / right_aov - 1:+.1%})")

# %% [markdown]
# The right pattern is `SUM([Revenue]) / COUNTD([Order ID])`. Whenever you average a value that was computed at a coarser grain than the rows, stop and ask: *what is one row?*

# %% [markdown]
# ## Common mistakes
#
# 1. **Expecting a dimension filter to change a `FIXED`**: use context, or use INCLUDE/EXCLUDE, or restructure.
# 2. **Averaging a FIXED (coarser-grain) value over finer-grain rows** (above).
# 3. **Mixing aggregate and non-aggregate fields** in one calculation. Aggregate everything consistently.
# 4. **Row-level ratio then AVG** instead of ratio of sums.
# 5. **`COUNT` instead of `COUNTD`** on ids in joined data (notebook 01).
# 6. **Using FIXED where a simple aggregate would do**: LODs add complexity and can slow dashboards. Use them when the grain differs from the view.

# %%
# Demonstration of mistake 6: a FIXED with the same dimensions as the view just repeats the aggregate
by_cat = done.groupby("category")["revenue"].sum()
fixed_same = done.groupby("category")["revenue"].transform("sum").groupby(done["category"]).first()
print("FIXED [Category] in a view by Category equals plain SUM:", bool(np.allclose(by_cat.sort_index(), fixed_same.sort_index())))

# %% [markdown]
# ## Interview corner
#
# **1. Difference between FIXED, INCLUDE and EXCLUDE?**
# <details><summary>Answer</summary>
# FIXED computes at exactly the listed dimensions regardless of the view; INCLUDE computes at the view's dimensions plus the listed ones (finer grain); EXCLUDE computes at the view's dimensions minus the listed ones (coarser grain).
# FIXED is also evaluated before dimension filters, so ordinary dimension filters do not affect it; INCLUDE and EXCLUDE come after dimension filters.
# </details>
#
# **2. A FIXED calculation ignores my filter. Why, and what are the options?**
# <details><summary>Answer</summary>
# Order of operations: FIXED LODs are computed after context filters and before dimension filters. Options: promote the filter to a context filter, switch to INCLUDE/EXCLUDE, or include the filtered dimension in the FIXED expression's dimension list when appropriate.
# </details>
#
# **3. Row-level calculation vs aggregate calculation vs table calculation?**
# <details><summary>Answer</summary>
# Row-level: evaluated per data row before aggregation. Aggregate: evaluated on aggregated values at the view's grain. Table calculation: evaluated on the query result in the view (running totals, ranks, percent of total), depending on partitioning and direction ("compute using") and applied late, after most filters.
# </details>
#
# **4. How would you compute average order value correctly on line-level data?**
# <details><summary>Answer</summary>
# <code>SUM([Revenue]) / COUNTD([Order ID])</code>, or build the view at Order ID level and average the order totals. Averaging a per-order FIXED total over line rows over-weights orders with many lines.
# </details>

# %% [markdown]
# ## Exercises
# `done` (completed lines; columns include `revenue`, `customer_id`, `order_id`, `category`, `order_date`) and `ltv` are defined above. Pandas stands in for the Tableau calculation.

# %% [markdown]
# ### Exercise 1: Top customer by lifetime revenue
# Emulate `{ FIXED [Customer ID] : SUM([Revenue]) }`: store in `top_customer_id` the customer with the highest lifetime revenue and in `top_ltv` that revenue. Also store `n_above_avg` = how many customers have lifetime revenue **above the
# average customer's** lifetime revenue (mean across customers, not across lines).

# %% tags=["starter"]
top_customer_id = ...
top_ltv = ...
n_above_avg = ...

# %%
_s = done.groupby("customer_id")["revenue"].sum()
assert top_customer_id == _s.idxmax(), f"expected customer {_s.idxmax()}"
assert abs(top_ltv - _s.max()) < 1e-9
assert n_above_avg == int((_s > _s.mean()).sum()), f"expected {int((_s > _s.mean()).sum())}"
print(f"Exercise 1 passed (customer {top_customer_id}: {top_ltv:,.2f}; {n_above_avg} of {len(_s)} above average)")

# %% tags=["solution"]
_ltv = done.groupby("customer_id")["revenue"].sum()
top_customer_id = _ltv.idxmax()
top_ltv = _ltv.max()
n_above_avg = int((_ltv > _ltv.mean()).sum())

# %% [markdown]
# ### Exercise 2: Revenue by acquisition cohort
# Emulate `YEAR({ FIXED [Customer ID] : MIN([Order Date]) })` (the year of each customer's **first completed order**) and store in `rev_by_cohort` a dict `{cohort_year: revenue}` over **all** completed lines.
# Also store `share_2023_cohort` = the share of total revenue coming from customers whose first order was in 2023.

# %% tags=["starter"]
rev_by_cohort = ...
share_2023_cohort = ...

# %%
_first = done.groupby("customer_id")["order_date"].min().dt.year
_rev = done.groupby("customer_id")["revenue"].sum()
_exp = _rev.groupby(_first).sum()
assert set(rev_by_cohort) == set(_exp.index)
assert all(abs(rev_by_cohort[y] - _exp[y]) < 1e-6 for y in _exp.index)
assert abs(sum(rev_by_cohort.values()) - done["revenue"].sum()) < 1e-6, "cohorts must partition total revenue"
assert abs(share_2023_cohort - _exp[2023] / _exp.sum()) < 1e-12
print(f"Exercise 2 passed (2023 cohort carries {share_2023_cohort:.1%} of revenue)")

# %% tags=["solution"]
_f = done.groupby("customer_id")["order_date"].transform("min").dt.year
_g = done.assign(cohort=_f).groupby("cohort")["revenue"].sum()
rev_by_cohort = {int(k): float(v) for k, v in _g.items()}
share_2023_cohort = rev_by_cohort[2023] / sum(rev_by_cohort.values())

# %% [markdown]
# ### Exercise 3: Average customer spend per channel (INCLUDE)
# Emulate `AVG({ INCLUDE [Customer ID] : SUM([Revenue]) })` in a view by **Channel**. Store the result as a Series `avg_spend_by_channel` (index = channel). Then store `naive_by_channel`, the Series of plain `AVG([Revenue])` per channel, and
# `include_is_larger` = `True` if the INCLUDE average is larger than the naive one for **every** channel.

# %% tags=["starter"]
avg_spend_by_channel = ...
naive_by_channel = ...
include_is_larger = ...

# %%
_pc = done.groupby(["channel", "customer_id"])["revenue"].sum().groupby("channel").mean()
_nv = done.groupby("channel")["revenue"].mean()
assert np.allclose(avg_spend_by_channel.sort_index().values, _pc.sort_index().values)
assert np.allclose(naive_by_channel.sort_index().values, _nv.sort_index().values)
assert include_is_larger == bool((_pc > _nv).all())
print("Exercise 3 passed:", avg_spend_by_channel.round(2).to_dict())

# %% tags=["solution"]
avg_spend_by_channel = done.groupby(["channel", "customer_id"])["revenue"].sum().groupby("channel").mean()
naive_by_channel = done.groupby("channel")["revenue"].mean()
include_is_larger = bool((avg_spend_by_channel.sort_index() > naive_by_channel.sort_index()).all())

# %% [markdown]
# ### Exercise 4: Quantify the grain trap
# Compute AOV per **category** two ways on the line-level table: `aov_wrong` = `AVG` of the FIXED order total over lines, grouped by category; `aov_right` = `SUM(Revenue) / COUNTD(Order ID)` per category. Both as Series indexed by category.
# Store `worst_category` = the category where the wrong value overstates the right one the most **in relative terms**.
# *Hint: `done["order_revenue"]` already holds the FIXED order total. Note an order can span categories; `COUNTD` per category counts the order once in each.*

# %% tags=["starter"]
aov_wrong = ...
aov_right = ...
worst_category = ...

# %%
_w = done.groupby("category")["order_revenue"].mean()
_r = done.groupby("category")["revenue"].sum() / done.groupby("category")["order_id"].nunique()
assert np.allclose(aov_wrong.sort_index().values, _w.sort_index().values)
assert np.allclose(aov_right.sort_index().values, _r.sort_index().values)
assert worst_category == (_w / _r).idxmax(), f"expected {(_w / _r).idxmax()}"
print(f"Exercise 4 passed (largest distortion: {worst_category}, wrong/right = {(_w / _r).max():.2f}x)")

# %% tags=["solution"]
aov_wrong = done.groupby("category")["order_revenue"].mean()
aov_right = done.groupby("category")["revenue"].sum() / done.groupby("category")["order_id"].nunique()
worst_category = (aov_wrong / aov_right).idxmax()

# %% [markdown]
# ### Exercise 5 (open-ended): Design a LOD
# Write, in Tableau syntax in a markdown cell, the calculation for **"customers whose lifetime revenue is above the average customer's lifetime revenue"** (hint: a FIXED at customer level, compared with a table-wide aggregate of it),
# say which filter kind would (or would not) change it, and give the pandas computation of the expected count that Exercise 1 already produced.

# %% [markdown]
# ## Recap
# - Row-level, aggregate and table calculations are evaluated at different stages; ratios belong in aggregate calculations (`SUM/SUM`).
# - `FIXED` = exact grain, independent of the view (before dimension filters); `INCLUDE` = finer than the view; `EXCLUDE` = coarser. Each is a `groupby`/`transform` you can test in pandas.
# - Order of operations: context > FIXED > dimension filters > INCLUDE/EXCLUDE > measure filters > table calculations. A filter "not working" on a FIXED is usually this.
# - Watch the grain: averaging a coarser-grain value over finer rows over-weights big groups.
#
# **Go deeper:** Tableau Help, "Create Level of Detail Expressions in Tableau" and "Tableau's Order of Operations" (help.tableau.com).
#
# **Next:** [Charts, dashboards and interactivity](03_charts_dashboards_and_interactivity.ipynb)

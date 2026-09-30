# %% [markdown]
# # Tableau: Charts, Dashboards and Interactivity
#
# **Why this matters.** In a review meeting nobody asks "which mark type did you use?". They ask "why is this number different from finance's?" and "what happens when I click the US?". A dashboard earns trust when
# every chart has a clear question, the numbers behind it are known in advance, and clicking or filtering changes the *right* things by the *right* amount. This notebook computes the numbers behind each proposed Tableau view
# so you can check your build, and simulates filter interactions so you can predict what a click will do.
#
# **You will learn**
# - pick a chart for a question (compare, trend, relationship, distribution, part-to-whole) and compute the numbers behind it
# - reproduce table-calculation style views in pandas: Pareto / cumulative share, percent of row, moving average, year-over-year
# - build **Top N + Other** correctly so the pieces still add up
# - simulate dashboard **filter actions** and see which numbers add up across slices and which do not (`COUNTD` is not additive)
# - review a dashboard against a layout and performance checklist
#
# **Prerequisites:** [Tableau concepts](01_tableau_concepts_and_data_prep.ipynb) and [calculated fields and LODs](02_calculated_fields_and_lod_expressions.ipynb).
#
# > **Candour box (read this).** Tableau is a licensed GUI product and was **not run** while writing this module. Everything about Tableau's interface (card names, menus, dashboard object types, action types, sizing options)
# > is described from general knowledge and **must be verified in the current Tableau documentation** (help.tableau.com). Cells tagged `no-run` were not executed. The **pandas/SQLite cells below do run**: they give the numbers
# > your Tableau views should show, which is the real verification you can do offline. Data: synthetic shop tables (`docs/DATASETS.md`).

# %%
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("../datasets")
customers = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
products = pd.read_csv(DATA / "products.csv")
orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
items = pd.read_csv(DATA / "order_items.csv")

lines = (
    items.merge(orders, on="order_id").merge(products, on="product_id").merge(customers[["customer_id", "country", "is_premium"]], on="customer_id")
    .assign(revenue=lambda d: d["quantity"] * d["unit_price"], cost=lambda d: d["quantity"] * d["unit_cost"])
)
done = lines[lines["status"].isin(["delivered", "shipped"])].copy()            # data source filter: completed orders
done["month"] = done["order_date"].dt.to_period("M").dt.to_timestamp()
print(f"{len(done):,} completed lines, total revenue {done['revenue'].sum():,.2f}, {done['order_id'].nunique():,} orders, {done['customer_id'].nunique():,} customers")

# %% [markdown]
# ## 1. Start from the question, then the mark
# Tableau has a feature that suggests chart types for the fields you selected (often called *Show Me*; verify the name in your version). Treat it as a suggestion: **you** decide the question.
#
# | Question | Tableau view (typical build) | Numbers to check |
# |---|---|---|
# | Which categories earn the most, and how concentrated is revenue? | sorted bar; a Pareto adds a cumulative-share table calculation | revenue and cumulative share per category |
# | How does revenue move over time? | continuous month line; optional moving average; year-over-year | monthly totals, growth |
# | Do expensive products sell fewer units? | scatter (one mark per product) with average reference lines | price, units, quadrant counts |
# | Where is the mix different? | highlight table / heatmap of country by category with percent of row | row-wise shares |
# | What does an order look like? | histogram (bins of a measure) | counts per bin, and how bin size changes the story |
# | Who are the top N and what is everyone else? | Top N filter or set, plus an "Other" group | pieces sum to the total |
#
# Each section below computes the "Numbers to check" column.

# %% [markdown]
# ## 2. Compare: sorted bar and Pareto (cumulative share)
# Sorting is a property of the view, not the data: in Tableau you sort the dimension by the measure. The Pareto's cumulative line is a **table calculation** (running total of percent of total), so it depends on the sort order.
# pandas twin: `sort_values`, `cumsum`.

# %%
cat = done.groupby("category")["revenue"].sum().sort_values(ascending=False).to_frame("revenue")
cat["share"] = cat["revenue"] / cat["revenue"].sum()                   # % of Total
cat["cum_share"] = cat["share"].cumsum()                               # RUNNING_SUM(% of Total), sorted descending
cat.round(3)

# %% tags=["no-run"]
TABLEAU_PARETO = """
// Rows: Category (sorted descending by SUM([Revenue])). Columns: SUM([Revenue]).
// Second measure on the same view:
% of Total:       SUM([Revenue]) / TOTAL(SUM([Revenue]))
Cumulative %:     RUNNING_SUM([% of Total])        // 'Compute using' = Category (the sorted dimension)
// Put Cumulative % on a dual axis as a line; both axes should be fixed/synchronised sensibly, see section 8.
"""

# %% [markdown]
# *What just happened?* The top category carries a large share, and the cumulative column tells you how many categories it takes to reach (say) 80%. The same idea at product level (60 products) is the classic "a few products make most of the money" chart;
# Exercise 1 asks you to count them. Caution: a Pareto across *categories* with only five bars is thin; the technique shines with many items.

# %% [markdown]
# ## 3. Trend: continuous month line, moving average and year-over-year
# Use a **continuous** month (a truncated date that draws a real time axis), not a discrete "Jan, Feb, ..." header, unless you deliberately want months stacked across years. Our data runs June 2023 to May 2025: two complete June-to-May years.

# %%
monthly = done.groupby("month")["revenue"].sum().to_frame("revenue")
monthly["ma3"] = monthly["revenue"].rolling(3).mean()                          # WINDOW_AVG(SUM([Revenue]), -2, 0)
first_year, second_year = monthly["revenue"].iloc[:12], monthly["revenue"].iloc[12:]
print("months:", monthly.index.min().strftime("%b %Y"), "to", monthly.index.max().strftime("%b %Y"), f"({len(monthly)} months)")
print(f"Year 1 (Jun 2023-May 2024): {first_year.sum():,.0f}  Year 2 (Jun 2024-May 2025): {second_year.sum():,.0f}  growth {second_year.sum() / first_year.sum() - 1:+.1%}")
monthly.assign(yoy=monthly["revenue"].pct_change(12)).round(3).iloc[[0, 5, 11, 12, 17, 23]]

# %% [markdown]
# `pct_change(12)` compares each month with the same month a year earlier (a table calculation with a fixed offset: `LOOKUP(SUM([Revenue]), -12)` in Tableau, verify). The first 12 months have no prior year, so the result is null there; do not show zeros. The year-over-year figures here are huge (the first months of this synthetic shop are tiny and the business ramps up), so read them as a demonstration of the mechanics, not as a realistic growth rate.
# Note the seasonality: November and December are the peak. A moving average smooths the series but **lags** and hides the peaks, so show it *with* the raw line, not instead of it.

# %% [markdown]
# ## 4. Relationship: scatter of products with reference lines
# One mark per product needs the product on the **Detail** shelf of the marks card (otherwise Tableau aggregates all products into a single dot). The numbers: price, units sold and revenue per product; average reference lines split four quadrants.

# %%
prod = done.groupby(["product_id", "product_name", "category"], as_index=False).agg(units=("quantity", "sum"), revenue=("revenue", "sum")).merge(products[["product_id", "unit_price"]], on="product_id")
avg_price, avg_units = prod["unit_price"].mean(), prod["units"].mean()
prod["quadrant"] = np.where(prod["unit_price"] >= avg_price, "high price", "low price") + " / " + np.where(prod["units"] >= avg_units, "high units", "low units")
print(f"{len(prod)} products; average price {avg_price:.2f}, average units {avg_units:.1f}; correlation(price, units) = {prod['unit_price'].corr(prod['units']):.2f}")
prod["quadrant"].value_counts().to_frame("products")

# %% [markdown]
# Two honest caveats for this chart: the reference lines are *averages of products*, not of units sold; and a scatter of 60 points can show a pattern that a correlation number summarises badly, so look at it (and colour by category to see clusters).

# %% [markdown]
# ## 5. Part-to-whole across two categories: percent of row
# A highlight table (or heatmap) of country by category. The question is *mix*, so the useful number is **percent of row** (each country's categories sum to 100%), not raw revenue, which just reproduces "the US is big".

# %%
cc = done.pivot_table(index="country", columns="category", values="revenue", aggfunc="sum")
row_pct = cc.div(cc.sum(axis=1), axis=0)                                      # 'Percent of Total' computed along the table across (row)
assert np.allclose(row_pct.sum(axis=1), 1.0)
spread = (row_pct.max() - row_pct.min()).sort_values(ascending=False)
print("category share range across countries (max - min):", spread.round(3).to_dict())
(row_pct * 100).round(1)

# %% [markdown]
# *What just happened?* The shares move only by a few percentage points across countries (see the spread printed above), so the honest title is closer to "the category mix is similar everywhere", not a dramatic colour story. Colour scales stretch to the data range, so a tiny difference
# can look large: check the numbers before you headline a pattern.

# %% [markdown]
# ## 6. Distribution: histogram and the bin-size choice
# In Tableau a histogram is a **bin** of a measure (you choose the bin size) plus a count. Bin size changes the story, so pick it deliberately.

# %%
order_value = done.groupby("order_id")["revenue"].sum()
for size in (50, 200, 1000):
    bins = (order_value // size * size).astype(int)
    counts = bins.value_counts().sort_index()
    print(f"bin size {size:4d}: {len(counts):3d} bins, tallest bin starts at {counts.idxmax():5d} holding {counts.max() / len(order_value):.0%} of orders")
print(f"median order {order_value.median():.0f}, mean {order_value.mean():.0f}, p95 {order_value.quantile(.95):.0f}, max {order_value.max():.0f} (long right tail)")

# %% [markdown]
# Too small a bin gives a jagged picture, too large hides the shape. With a long right tail, consider capping the axis or showing the median rather than the mean in a KPI card.

# %% [markdown]
# ## 7. Top N + "Other": make the pieces add up
# A Top N filter **removes** the rest, so a bar chart of the top 5 countries no longer sums to the total and percent-of-total calculations silently use the wrong denominator. Stakeholders want to see "everyone else" too.
# Note from notebook 02: a Top N (dimension-type) filter is applied after `FIXED` LODs, so a FIXED denominator is still the whole table.
# pandas: rank, keep the top N, and sum the rest into one row.

# %%
def top_n_with_other(s, n):
    """Top n items by value plus one 'Other' row; total is preserved."""
    s = s.sort_values(ascending=False)
    out = s.iloc[:n].copy()
    if len(s) > n:
        out["Other"] = s.iloc[n:].sum()
    return out

rev_country = done.groupby("country")["revenue"].sum()
t3 = top_n_with_other(rev_country, 3)
assert abs(t3.sum() - rev_country.sum()) < 1e-6, "pieces must add up to the total"
naive_share_top3 = (rev_country.nlargest(3) / rev_country.nlargest(3).sum()).iloc[0]
true_share = rev_country.max() / rev_country.sum()
print((t3 / t3.sum()).round(3).to_dict())
print(f"largest country share: {true_share:.1%} of all revenue, but {naive_share_top3:.1%} if the % of total only sees the Top 3 (wrong denominator)")

# %% tags=["no-run"]
TABLEAU_TOPN_OTHER = """
// One common pattern (verify in current docs): a Parameter [Top N] (integer), a calculated field that ranks the dimension
// using a FIXED LOD or RANK, then a group label:
Country Rank:    RANK(SUM([Revenue]))                          // table calc, depends on view partitioning
Country (Top N): IF [Country Rank] <= [Top N] THEN ATTR([Country]) ELSE 'Other' END
// Exact construction differs by version and by whether you use sets; test that the pieces sum to the total.
"""

# %% [markdown]
# ## 8. Dual axis: when two measures on one chart mislead
# Two lines with separate y-axes can be made to look as if they track each other (or diverge) simply by choosing each axis range. Before a dual-axis chart, compute whether the relationship exists in numbers: here revenue and orders per month.

# %%
mo = done.groupby("month").agg(revenue=("revenue", "sum"), orders=("order_id", "nunique"))
mo["aov"] = mo["revenue"] / mo["orders"]
print(f"correlation revenue vs orders: {mo['revenue'].corr(mo['orders']):.3f}; AOV ranges {mo['aov'].min():.0f} to {mo['aov'].max():.0f} (std/mean = {mo['aov'].std() / mo['aov'].mean():.2f})")
idx = (mo[["revenue", "orders"]] / mo[["revenue", "orders"]].iloc[0] * 100).round(0)
idx.iloc[[0, 5, 6, 11, 17, 23]]

# %% [markdown]
# Revenue and orders move together (high correlation), because most revenue variation comes from order *volume*, while order *value* (AOV) varies much less. Indexing both to 100 at the first month puts them on one honest axis, which is
# often a better alternative to a dual axis. If you do use a dual axis: **synchronise** ranges when units are comparable, label both axes, and never use it to imply causation.

# %% [markdown]
# ## 9. Interactivity: what does a click do? (filter actions, simulated)
# A dashboard **action** lets a selection in one sheet filter others (Tableau describes filter, highlight, URL, parameter and set actions; verify current names and options). The user-facing promise is: *clicking "US" on the country bar
# recomputes every other sheet for the US only*. We can simulate that and check what adds up.

# %%
def dashboard_numbers(df):
    """The numbers behind four sheets (KPIs, by category, monthly total, AOV) for any filtered subset."""
    return {
        "revenue": df["revenue"].sum(),
        "orders": df["order_id"].nunique(),
        "customers": df["customer_id"].nunique(),
        "aov": df["revenue"].sum() / df["order_id"].nunique() if len(df) else np.nan,
        "by_category": df.groupby("category")["revenue"].sum(),
    }

allv = dashboard_numbers(done)
us = dashboard_numbers(done[done["country"] == "US"])                    # action: filter Country = US
print(f"all: revenue {allv['revenue']:,.0f}, orders {allv['orders']:,}, customers {allv['customers']:,}, AOV {allv['aov']:.0f}")
print(f"US : revenue {us['revenue']:,.0f}, orders {us['orders']:,}, customers {us['customers']:,}, AOV {us['aov']:.0f}")

# %% [markdown]
# **A reconciliation test every dashboard should pass:** the pieces from a partition (every country) must add up to the whole, for **additive** measures. Revenue, orders (an order belongs to one customer and so to one country) and customers
# (each customer has one country) all partition by country. Now partition by *category*, where one order or one customer can appear in several categories:

# %%
by_country = {c: dashboard_numbers(g) for c, g in done.groupby("country")}
by_category = {c: dashboard_numbers(g) for c, g in done.groupby("category")}
for name, parts in (("country", by_country), ("category", by_category)):
    print(f"partition by {name:8s}: revenue {sum(p['revenue'] for p in parts.values()) / allv['revenue']:.4f}x  "
          f"orders {sum(p['orders'] for p in parts.values()) / allv['orders']:.4f}x  customers {sum(p['customers'] for p in parts.values()) / allv['customers']:.4f}x of the total")

# %% [markdown]
# *What just happened?* By country everything sums to exactly 1.0000x. By category, revenue still adds up but **orders and customers add up to more than the total**: a distinct count is **not additive** across a dimension that an entity can span.
# In Tableau this shows up as a grand total (a distinct count over everything) that is smaller than the sum of the rows above it. Neither number is wrong; label them, and never add distinct counts across categories.

# %% [markdown]
# ## 10. Dashboard design and performance checklist
# Design (check each against your build):
# - **One purpose, 3-5 questions.** A title that says what the dashboard is for and the period and filter state shown.
# - **Reading order**: headline KPIs top-left, then trend, then breakdowns. Whitespace is a feature.
# - **Consistent encodings**: the same colour means the same category on every sheet; number formats and units agree (revenue always in the same unit).
# - **Interactivity explained**: if clicking filters other sheets, say so (a short caption); provide a visible way to reset filters.
# - **Fixed size or responsive?** Decide the target device first; Tableau offers fixed, range and automatic sizing and separate device layouts (verify current options), and a layout that looks fine on your monitor may not on a laptop or phone.
# - **Accessibility**: do not encode meaning by colour alone; sufficient contrast; alt text where the product supports it (verify).
#
# Performance (general principles; measure in your own workbook rather than trusting folklore):
# - Fewer marks, fewer sheets, fewer quick filters: each one is a query. Filter or aggregate in the data source rather than in every sheet.
# - Prefer **extracts** for modest data sizes (notebook 01) and keep only the columns you use.
# - Context filters and complex LODs add work: use them because you need their semantics (notebook 02), not by habit.
# - Tableau has a built-in performance recording feature (verify the exact name and location in your version) that shows where time is spent: start there before optimising blindly.

# %% tags=["no-run"]
TABLEAU_BUILD_SHEETS = """
Sheet "Category revenue":  Rows Category (sort desc by SUM(Revenue)), Columns SUM(Revenue), label = value
Sheet "Monthly trend":     Columns MONTH(Order Date) as a continuous (truncated) date, Rows SUM(Revenue); add a moving-average table calc if wanted
Sheet "Product scatter":   Columns AVG(Unit Price), Rows SUM(Quantity), Product Name on Detail, Category on Color, reference lines = averages
Sheet "Country x category": Country on Rows, Category on Columns, SUM(Revenue) on Color/Label, table calc 'Percent of Total' computed along Category
Sheet "Order value":       bin of order total (needs an order-level field, e.g. a FIXED LOD on Order ID), Columns bin, Rows count of orders
Dashboard:                 layout the sheets; set the country bar to 'Use as Filter' (verify the control name), add a title and a caption explaining the click behaviour
// All of this is written from general knowledge of the product and was not executed. Compare each sheet with the numbers computed above.
"""

# %% [markdown]
# ## Common mistakes
#
# 1. **Discrete month for a time series** (headers instead of an axis), or a date that mixes years because the year part is missing.
# 2. **A scatter with one dot**: forgetting to put the entity (product) on Detail.
# 3. **Top N without "Other"** and a percent-of-total that uses the wrong denominator.
# 4. **Adding distinct counts across categories** (grand total smaller than the sum of rows).
# 5. **Dual axes with arbitrary ranges**, or a truncated bar axis.
# 6. **Table calculations with the wrong "compute using"** (the percent of row becomes a percent of column, a running total restarts at the wrong place).
# 7. **Overloaded dashboards**: ten sheets, six quick filters, and no reading order.
# 8. **No reconciliation**: nobody ever compared the dashboard with an independent total.

# %%
# Mistake 6 in numbers: the same 'percent of total' computed along the wrong direction
wrong_dir = cc.div(cc.sum(axis=0), axis=1)                        # along the column instead of the row
print("row-wise shares sum to", row_pct.sum(axis=1).round(3).unique(), "but column-wise 'percent' of a row sums to", wrong_dir.sum(axis=1).round(2).head(3).to_list(), "...")

# %% [markdown]
# ## Interview corner
#
# **1. What is the difference between a discrete and a continuous date in Tableau?**
# <details><summary>Answer</summary>
# Discrete dates are separate labelled values that draw headers (for example "Jan", "Feb"), so the same month from different years can stack together. Continuous dates form an unbroken range and draw a real time axis. Line charts over time are usually better continuous.
# </details>
#
# **2. The grand total of distinct customers is smaller than the sum of the category rows. Is it a bug?**
# <details><summary>Answer</summary>
# No. A customer can buy in several categories, so counted per category they appear several times, while the grand total counts each customer once. Distinct counts are not additive across a dimension entities can span. Label the total and do not sum the rows.
# </details>
#
# **3. A stakeholder wants a dual-axis chart of revenue and profit. What do you check?**
# <details><summary>Answer</summary>
# Whether the units are comparable (then use one axis or synchronise), whether the relationship is actually meaningful (compute the correlation or plot an index), whether both axes are labelled, and whether the chosen ranges create a false impression. Indexing to a base period or showing margin % directly is often clearer.
# </details>
#
# **4. How do you check that a dashboard's numbers are right before sharing?**
# <details><summary>Answer</summary>
# Reconcile against an independent source: a SQL query or pandas total on the same filters, for several filter states. Check that partitions of additive measures sum to the whole, that row counts match expectation after joins, and that definitions (net vs gross, completed orders only) are written on the dashboard.
# </details>

# %% [markdown]
# ## Exercises
# `done`, `products`, `orders` and the helper functions `top_n_with_other` and `dashboard_numbers` are defined above.

# %% [markdown]
# ### Exercise 1: How concentrated is product revenue?
# Build the Pareto at **product** level (by `product_name`, completed orders). Store `n_products_half` = the smallest number of products, taken in descending revenue order, whose cumulative revenue reaches **at least 50%** of the total,
# and `top_product` = the name of the highest-revenue product.
# *Hint: `cumsum()` on the sorted series, then `(cum >= 0.5).idxmax()`'s position.*

# %% tags=["starter"]
n_products_half = ...
top_product = ...

# %%
_s = done.groupby("product_name")["revenue"].sum().sort_values(ascending=False)
_cum = _s.cumsum() / _s.sum()
_n = int((_cum < 0.5).sum() + 1)
assert n_products_half == _n, f"expected {_n}"
assert top_product == _s.index[0], f"expected {_s.index[0]}"
assert _cum.iloc[_n - 1] >= 0.5 and (_n == 1 or _cum.iloc[_n - 2] < 0.5), "n is the smallest count reaching 50%"
print(f"Exercise 1 passed: {n_products_half} of {len(_s)} products reach 50% of revenue; top product {top_product}")

# %% tags=["solution"]
_p = done.groupby("product_name")["revenue"].sum().sort_values(ascending=False)
_c = _p.cumsum() / _p.sum()
n_products_half = int((_c < 0.5).sum() + 1)
top_product = _p.index[0]

# %% [markdown]
# ### Exercise 2: Top N + Other for products
# Using `top_n_with_other`, build `top5_products`: the 5 highest-revenue products (completed orders) plus an `"Other"` row. Store `other_share` = the share of total revenue sitting in `"Other"`.
# Then store `naive_top5_share_of_first` = the highest product's share if a percent-of-total were computed only over the five shown products (the wrong-denominator mistake), and `true_first_share` = its share of the whole.

# %% tags=["starter"]
top5_products = ...
other_share = ...
naive_top5_share_of_first = ...
true_first_share = ...

# %%
_r = done.groupby("product_name")["revenue"].sum().sort_values(ascending=False)
assert list(top5_products.index) == list(_r.index[:5]) + ["Other"], "five products then 'Other'"
assert abs(top5_products.sum() - _r.sum()) < 1e-6, "pieces must add up to the total"
assert abs(other_share - _r.iloc[5:].sum() / _r.sum()) < 1e-12
assert abs(true_first_share - _r.iloc[0] / _r.sum()) < 1e-12
assert abs(naive_top5_share_of_first - _r.iloc[0] / _r.iloc[:5].sum()) < 1e-12
assert naive_top5_share_of_first > true_first_share, "the wrong denominator overstates the share"
print(f"Exercise 2 passed: Other = {other_share:.1%}; first product {true_first_share:.1%} true vs {naive_top5_share_of_first:.1%} naive")

# %% tags=["solution"]
_rev = done.groupby("product_name")["revenue"].sum()
top5_products = top_n_with_other(_rev, 5)
other_share = top5_products["Other"] / top5_products.sum()
naive_top5_share_of_first = _rev.max() / _rev.nlargest(5).sum()
true_first_share = _rev.max() / _rev.sum()

# %% [markdown]
# ### Exercise 3: Predict a filter action, verify with SQL
# Simulate clicking **channel = "mobile"** with **category = "Electronics"** also selected: use `dashboard_numbers` on the filtered `done` rows and store the result in `clicked`.
# Then verify it with an **independent method**: run a SQL query on `../datasets/shop.db` (SQLite) for the same filters (statuses delivered and shipped) and store its revenue, orders and customers in `sql_revenue`, `sql_orders`, `sql_customers`.
# *Hint: join `order_items`, `orders`, `products`; revenue is `SUM(quantity * unit_price)`; use `COUNT(DISTINCT ...)`.*

# %% tags=["starter"]
clicked = ...
sql_revenue = ...
sql_orders = ...
sql_customers = ...

# %%
_f = done[(done["channel"] == "mobile") & (done["category"] == "Electronics")]
assert abs(clicked["revenue"] - _f["revenue"].sum()) < 1e-6 and clicked["orders"] == _f["order_id"].nunique() and clicked["customers"] == _f["customer_id"].nunique()
assert abs(sql_revenue - clicked["revenue"]) < 0.01, "SQL and pandas revenue should agree"
assert int(sql_orders) == clicked["orders"] and int(sql_customers) == clicked["customers"], "SQL and pandas counts should agree"
print(f"Exercise 3 passed: revenue {sql_revenue:,.2f}, {sql_orders} orders, {sql_customers} customers")

# %% tags=["solution"]
clicked = dashboard_numbers(done[(done["channel"] == "mobile") & (done["category"] == "Electronics")])
_con = sqlite3.connect(DATA / "shop.db")
_row = _con.execute("""
    SELECT SUM(oi.quantity * p.unit_price), COUNT(DISTINCT o.order_id), COUNT(DISTINCT o.customer_id)
    FROM order_items oi
    JOIN orders o ON o.order_id = oi.order_id
    JOIN products p ON p.product_id = oi.product_id
    WHERE o.status IN ('delivered', 'shipped') AND o.channel = 'mobile' AND p.category = 'Electronics'
""").fetchone()
_con.close()
sql_revenue, sql_orders, sql_customers = _row

# %% [markdown]
# ### Exercise 4 (open-ended): Choose and justify
# Pick **three** of the six questions in section 1 and, for a question of your own about the shop (for example: "do premium customers order more often?"), write: the chart you would build in Tableau, the fields on each shelf,
# the pandas numbers you would check it against, and one way the chart could mislead. Compute the numbers in a cell.

# %% [markdown]
# ## Recap
# - Choose the view from the question; compute the numbers behind it first so you can verify the build.
# - Pareto, percent of row, moving average and year-over-year are table calculations: they depend on the sort order and the "compute using" direction.
# - Top N needs an "Other" row (and the right denominator) or the pieces stop adding up.
# - Distinct counts are not additive: by country they partition, by category they do not.
# - Reconcile every filter state against an independent calculation; interaction design is a checklist, not a mood.
#
# **Go deeper:** Tableau Help (help.tableau.com): table calculations, dashboard actions, and dashboard layout and sizing; Cole Nussbaumer Knaflic, *Storytelling with Data*; Stephen Few, *Information Dashboard Design*.
#
# **Next:** [Storytelling project: the Customer and Revenue dashboard](04_storytelling_project.ipynb)

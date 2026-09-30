# %% [markdown]
# # DAX Fundamentals: Measures, Filter Context and CALCULATE
#
# **Why this matters.** DAX is the formula language behind every number on a Power BI report, and the reason people get *wrong* numbers
# with total confidence is almost always one concept: **filter context**. Once you can predict what filters are active in a cell of a visual,
# and how `CALCULATE` changes them, you can write correct KPIs. This notebook makes that concept concrete by letting you *run* it: we build a tiny
# filter-context engine in pandas so that every DAX measure below has a running twin whose value you can compare with your report.
#
# **You will learn**
# - calculated column vs measure, and when each is right
# - row context vs filter context, and how `CALCULATE` replaces filters
# - `DIVIDE`, `SUMX`/`RELATED`, `ALL`/`ALLEXCEPT`, `VAR ... RETURN`
# - the seven KPIs of this module, each with its expected value from pandas: total revenue, margin %, AOV, premium share, revenue by month and more
#
# **Prerequisites:** [the data model](01_power_bi_overview_and_data_model.ipynb) (you loaded `exports/*.csv` into Power BI and related the tables).
#
# > **Candour box.** DAX runs only inside Power BI (and related engines), and it was **not executed** here. The DAX code is written from documented semantics and tagged `no-run`;
# > **verify syntax in the DAX reference** (learn.microsoft.com/dax) and by comparing your report with the numbers below. The Python "twin" of each measure **does run**. It is a teaching
# > model of filter context (single-column value filters, no `KEEPFILTERS`, no complex relationships), not a DAX engine.
# > Data: the synthetic shop tables (`docs/DATASETS.md`).

# %%
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("../datasets")
customers = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
products = pd.read_csv(DATA / "products.csv")
orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
items = pd.read_csv(DATA / "order_items.csv")

# The model as Power BI sees it, flattened so that one row = one Sales line with related-table columns attached (this is what RELATED() reaches).
sales = (
    items.merge(orders, on="order_id", validate="many_to_one")
    .merge(products, on="product_id", validate="many_to_one")
    .merge(customers[["customer_id", "country", "is_premium"]], on="customer_id", validate="many_to_one")
    .assign(year=lambda d: d["order_date"].dt.year, month_number=lambda d: d["order_date"].dt.month,
            year_month=lambda d: d["order_date"].dt.strftime("%Y-%m"))
)
print(sales.shape)
sales[["item_id", "order_id", "status", "quantity", "category", "unit_price", "unit_cost", "is_premium", "year_month"]].head(3)

# %% [markdown]
# ## 1. Measure or calculated column?
#
# | | Calculated column | Measure |
# |---|---|---|
# | Evaluated | once per row, at **data refresh**, result stored in the model | at **query time**, for every cell of every visual, never stored |
# | Context | **row context** (the current row) | **filter context** (whatever filters the cell carries) |
# | Uses memory | yes (a stored column) | no |
# | Reacts to slicers | no, fixed per row | yes |
# | Typical use | a per-row attribute used to slice or filter (e.g. price band) | any aggregate: totals, ratios, KPIs |
#
# Rule of thumb: **if the answer changes when the user clicks a slicer, it must be a measure.** A very common beginner error is a per-row *ratio* column that is then averaged:

# %%
sales_c = sales[sales["status"].isin(["delivered", "shipped"])].assign(
    line_revenue=lambda d: d["quantity"] * d["unit_price"], line_cost=lambda d: d["quantity"] * d["unit_cost"])
sales_c = sales_c.assign(row_margin=lambda d: (d["line_revenue"] - d["line_cost"]) / d["line_revenue"])

avg_of_ratios = sales_c["row_margin"].mean()                                           # what AVERAGE(Sales[Row Margin %]) would show
ratio_of_sums = (sales_c["line_revenue"].sum() - sales_c["line_cost"].sum()) / sales_c["line_revenue"].sum()   # what the measure DIVIDE(...) shows
print(f"average of row-level margins: {avg_of_ratios:.4%}   |   margin of the totals: {ratio_of_sums:.4%}")

# %% [markdown]
# *What just happened?* A ratio of averages is not an average of ratios: big orders should weigh more than small ones. The **measure** (`DIVIDE(revenue - cost, revenue)` over whatever rows are in context)
# gives the right, weighted answer at every level. Store *components* in the model, compute ratios in measures.

# %% [markdown]
# ## 2. Row context and iterators: `SUMX` and `RELATED`
#
# The price is in `Product`, the quantity in `Sales`. To get revenue you must multiply **line by line**, not multiply totals.
# `SUMX(Sales, expr)` walks each row of `Sales` (creating a **row context**), evaluates `expr` there, and adds the results. `RELATED(Product[unit_price])` follows the relationship
# from the current `Sales` row to its product. A naive `SUM(qty) * AVERAGE(price)` is a classic wrong answer (the size of the error depends on how prices and quantities co-vary; here it is small but nonzero, and it can be large on real data):

# %%
correct = float((sales["quantity"] * sales["unit_price"]).sum())                       # SUMX(Sales, Sales[quantity] * RELATED(Product[unit_price]))
naive = float(sales["quantity"].sum() * sales["unit_price"].mean())                    # SUM(Sales[quantity]) * AVERAGE(Product[unit_price])
print(f"SUMX line by line: {correct:,.2f}   naive SUM x AVERAGE: {naive:,.2f}   (error {naive / correct - 1:+.1%})")

# %% [markdown]
# ## 3. Filter context: what filters does this cell carry?
#
# Every cell of a visual has a **filter context**: the set of filters coming from slicers, page/report filters, and the row/column headers of that cell.
# In a matrix with `category` on rows and `channel` on columns, the cell "Books, web" carries `category = Books` **and** `channel = web`.
# A measure is simply evaluated inside that context. The **grand total** cell carries *no* filters from the headers.
#
# Below is our mini engine. `ctx` is a dict `{column: [allowed values]}`. `rows(ctx)` returns the Sales rows that survive (filters on dimension columns flow to the fact through relationships).
# `calculate(measure, ctx, overrides)` evaluates a measure in a *modified* context, and, exactly like `CALCULATE` with a column filter, **an override replaces any existing filter on that column**.

# %%
TABLE_OF = {"status": "Sales", "channel": "Sales", "order_id": "Sales", "item_id": "Sales", "quantity": "Sales",
            "category": "Product", "product_id": "Product", "product_name": "Product",
            "country": "Customer", "is_premium": "Customer", "customer_id": "Customer",
            "year": "Date", "month_number": "Date", "year_month": "Date"}
BLANK = np.nan


def rows(ctx):
    m = pd.Series(True, index=sales.index)
    for col, vals in ctx.items():
        m &= sales[col].isin(vals)
    return sales[m]


def calculate(measure, ctx, overrides=None, remove_table=None, keep_only=()):
    """CALCULATE: start from ctx, drop filters (ALL / ALLEXCEPT on a table), then apply overrides (which replace same-column filters)."""
    new = dict(ctx)
    if remove_table:
        for col in [c for c, t in TABLE_OF.items() if t == remove_table and c not in keep_only]:
            new.pop(col, None)
    new.update(overrides or {})
    return measure(new)


def divide(a, b):
    """DAX DIVIDE: BLANK (not an error) when the denominator is 0 or blank."""
    return BLANK if (pd.isna(a) or pd.isna(b) or b == 0) else a / b


COMPLETED = {"status": ["delivered", "shipped"]}


# --- measures (each has its DAX definition in the next cell) ---
def sales_amount(ctx):
    r = rows(ctx)
    return float((r["quantity"] * r["unit_price"]).sum()) if len(r) else BLANK


def net_revenue(ctx):
    return calculate(sales_amount, ctx, COMPLETED)


def net_cost(ctx):
    def f(c):
        r = rows(c)
        return float((r["quantity"] * r["unit_cost"]).sum()) if len(r) else BLANK
    return calculate(f, ctx, COMPLETED)


def n_orders(ctx):
    return calculate(lambda c: float(rows(c)["order_id"].nunique()) or BLANK, ctx, COMPLETED)


def margin_pct(ctx):
    rev = net_revenue(ctx)
    return divide(rev - net_cost(ctx), rev)


def aov(ctx):
    return divide(net_revenue(ctx), n_orders(ctx))


def premium_share(ctx):
    return divide(calculate(net_revenue, ctx, {"is_premium": [True]}), net_revenue(ctx))


print(f"Net revenue (no filters): {net_revenue({}):,.2f}")
print(f"Net revenue for Books   : {net_revenue({'category': ['Books']}):,.2f}")

# %% [markdown]
# ## 4. The KPI definitions, with the numbers your report must show
#
# **Business definitions** (write these down before writing DAX; disagreements here cause most "your number is wrong" meetings):
# - *Sales Amount* = quantity x product unit price, for every order line, whatever its status.
# - *Net Revenue* = Sales Amount for **completed** orders only (`status` is `delivered` or `shipped`); `cancelled` and `returned` orders are excluded.
# - *Net Cost* = quantity x unit cost, completed orders. *Margin %* = (Net Revenue - Net Cost) / Net Revenue.
# - *Orders* = distinct completed orders. *AOV* (average order value) = Net Revenue / Orders.
# - *Premium Share* = Net Revenue from premium customers / Net Revenue.

# %% tags=["no-run"]
DAX_MEASURES = """
-- Calculated column on Sales (row context, stored at refresh): use sparingly
Line Revenue = Sales[quantity] * RELATED ( Product[unit_price] )

-- Measures (filter context, computed at query time)
Sales Amount =
    SUMX ( Sales, Sales[quantity] * RELATED ( Product[unit_price] ) )

Net Revenue =
    CALCULATE ( [Sales Amount], Sales[status] IN { "delivered", "shipped" } )

Net Cost =
    CALCULATE (
        SUMX ( Sales, Sales[quantity] * RELATED ( Product[unit_cost] ) ),
        Sales[status] IN { "delivered", "shipped" }
    )

Margin % =
    VAR Rev  = [Net Revenue]
    VAR Cost = [Net Cost]
    RETURN DIVIDE ( Rev - Cost, Rev )

Orders =
    CALCULATE ( DISTINCTCOUNT ( Sales[order_id] ), Sales[status] IN { "delivered", "shipped" } )

AOV =
    DIVIDE ( [Net Revenue], [Orders] )

Premium Share =
    DIVIDE ( CALCULATE ( [Net Revenue], Customer[is_premium] = TRUE () ), [Net Revenue] )
"""

# %% [markdown]
# **Expected values** (format the card visuals like this; a tiny rounding difference from formatting is normal):

# %%
overall = pd.Series({
    "Sales Amount": sales_amount({}),
    "Net Revenue": net_revenue({}),
    "Net Cost": net_cost({}),
    "Margin %": margin_pct({}),
    "Orders": n_orders({}),
    "AOV": aov({}),
    "Premium Share": premium_share({}),
}, name="expected (all data, no slicers)")
overall.map(lambda v: f"{v:,.4f}").to_frame()

# %% [markdown]
# Now cross-check the engine against plain pandas computed a different way (this is what "validate the report against ground truth" means: two independent routes, same number):

# %%
done = sales[sales["status"].isin(["delivered", "shipped"])]
truth = {
    "net_revenue": (done["quantity"] * done["unit_price"]).sum(),
    "net_cost": (done["quantity"] * done["unit_cost"]).sum(),
    "orders": done["order_id"].nunique(),
    "premium_revenue": (done.loc[done["is_premium"], "quantity"] * done.loc[done["is_premium"], "unit_price"]).sum(),
}
assert abs(overall["Net Revenue"] - truth["net_revenue"]) < 1e-6
assert abs(overall["Net Cost"] - truth["net_cost"]) < 1e-6
assert overall["Orders"] == truth["orders"]
assert abs(overall["Premium Share"] - truth["premium_revenue"] / truth["net_revenue"]) < 1e-12
assert abs(overall["AOV"] - truth["net_revenue"] / truth["orders"]) < 1e-9
print("engine matches direct pandas for revenue, cost, orders, AOV and premium share")

# %% [markdown]
# ### Revenue by month (a line chart with `year_month` on the axis)
# Each point on the axis puts one `year_month` into the filter context. Then compare with a straight `groupby`:

# %%
months = sorted(sales["year_month"].unique())
by_month_engine = pd.Series({m: net_revenue({"year_month": [m]}) for m in months}, name="net_revenue")
by_month_pandas = done.assign(rev=done["quantity"] * done["unit_price"]).groupby("year_month")["rev"].sum()
assert np.allclose(by_month_engine.values, by_month_pandas.reindex(months).values), "engine and groupby must agree"
print("first 3 months:", by_month_engine.head(3).round(2).to_dict())
print("best month:", by_month_engine.idxmax(), f"{by_month_engine.max():,.2f}", "| months on the axis:", len(months))

# %% [markdown]
# *What just happened?* The measure is written **once** and the visual supplies the month. That is the payoff of measures: no per-month formulas. The best month above is in the late-year peak (Nov/Dec seasonality is planted in this synthetic data). If your line chart does not show a late-year peak, suspect a relationship or date-table problem.

# %% [markdown]
# ## 5. CALCULATE: change the context, on purpose
#
# `CALCULATE(<expression>, <filter>, <filter>, ...)` evaluates the expression in a **modified filter context**. Three behaviours to internalise:
#
# 1. A filter argument on a column **replaces** any existing filter on that column (so `Net Revenue` ignores a `status` slicer, by design).
# 2. Filters on *different* columns **add** (AND).
# 3. `ALL(table_or_column)` **removes** filters; `ALLEXCEPT(table, col, ...)` removes all filters on the table *except* the listed columns. This is how you get "% of total".
#
# Test 1 with the engine: pretend the user selected `status = returned` in a slicer.

# %%
slicer = {"status": ["returned"]}
print("Sales Amount with status=returned slicer:", f"{sales_amount(slicer):,.2f}")
print("Net Revenue with the same slicer        :", f"{net_revenue(slicer):,.2f}", "(the measure overrides the status filter, so it equals the overall value)")
assert abs(net_revenue(slicer) - net_revenue({})) < 1e-9

# %% [markdown]
# Now "% of total" patterns. For a row `category = Books`, the share of all revenue needs a denominator that **ignores the category filter** (`ALL(Product)`), and the share *within* a category
# needs a denominator that keeps only the category (`ALLEXCEPT(Product, Product[category])`):

# %% tags=["no-run"]
DAX_SHARES = """
Revenue % of All =
    DIVIDE ( [Net Revenue], CALCULATE ( [Net Revenue], ALL ( Product ) ) )

Revenue % of Category =
    DIVIDE ( [Net Revenue], CALCULATE ( [Net Revenue], ALLEXCEPT ( Product, Product[category] ) ) )
"""

# %%
def share_of_all(ctx):
    return divide(net_revenue(ctx), calculate(net_revenue, ctx, remove_table="Product"))


def share_of_category(ctx):
    return divide(net_revenue(ctx), calculate(net_revenue, ctx, remove_table="Product", keep_only=("category",)))


cats = sorted(products["category"].unique())
tbl = pd.DataFrame({
    "net_revenue": [net_revenue({"category": [c]}) for c in cats],
    "% of all": [share_of_all({"category": [c]}) for c in cats],
}, index=cats)
tbl.loc["Total"] = [net_revenue({}), share_of_all({})]
print(tbl.round(4))
assert abs(tbl.loc[cats, "% of all"].sum() - 1) < 1e-9, "category shares must sum to 100%"

hp = {"category": ["Electronics"], "product_name": ["Headphones Pro"]}
print("Headphones Pro: share of Electronics =", round(share_of_category(hp), 4), "| share of all revenue =", round(share_of_all(hp), 4))

# %% [markdown]
# *What just happened?* The **Total** row is `ALL`'s result (100%), each category is its slice, and the product row shows how the *same visual* can hold different denominators.
# `DIVIDE` is preferred to `/` because it returns BLANK rather than an error when the denominator is zero or blank:

# %%
future = {"year_month": ["2025-12"]}          # a month with no orders in our data
print("Net Revenue for 2025-12:", net_revenue(future), "| AOV:", aov(future), "| margin:", margin_pct(future))

# %% [markdown]
# ## 6. Totals are not always sums of the cells
#
# A non-additive measure such as AOV or margin % is computed **fresh in the total cell's context**; it is not the sum (or average) of the cells above it. This is correct, and beginners often think it is a bug.

# %%
channels = sorted(sales["channel"].unique())
cell_aov = pd.Series({ch: aov({"channel": [ch]}) for ch in channels})
print(cell_aov.round(2).to_dict())
print("simple average of the cells:", round(cell_aov.mean(), 2), "| the Total cell (AOV in the empty context):", round(aov({}), 2))

# %% [markdown]
# ## 7. Context transition (one advanced idea, briefly)
#
# Inside an iterator such as `AVERAGEX(VALUES(Customer[customer_id]), [Net Revenue])`, each customer is a **row context**, and calling a *measure* there implicitly wraps it in `CALCULATE`, converting
# the current customer into a **filter** on `customer_id`. That is **context transition**. `AVERAGEX` ignores customers whose value is blank (no completed sales). Result: average revenue per customer *who bought*.

# %% tags=["no-run"]
DAX_AVG_CUSTOMER = """
Avg Revenue per Buying Customer =
    AVERAGEX ( VALUES ( Customer[customer_id] ), [Net Revenue] )
"""

# %%
buyers = customers["customer_id"].tolist()
vals = [net_revenue({"customer_id": [c]}) for c in buyers[:400]]             # 400 customers keeps this quick
vals = [v for v in vals if not pd.isna(v)]
direct = done.assign(rev=done["quantity"] * done["unit_price"]).groupby("customer_id")["rev"].sum()
check = direct.reindex(buyers[:400]).dropna()
assert np.allclose(sorted(vals), sorted(check.values)), "context transition == groupby per customer"
print(f"first 400 customers: {len(vals)} bought; mean revenue per buyer = {np.mean(vals):,.2f}")

# %% [markdown]
# ## Common mistakes
#
# 1. **A calculated column where a measure is needed**: it cannot respond to slicers, and ratios averaged across rows are wrong.
# 2. **`SUM(a) * AVERAGE(b)` instead of `SUMX(table, a * b)`**: shown above: silently wrong, and the error is not bounded.
# 3. **Using `/` instead of `DIVIDE`**: errors or infinities in visuals for empty contexts.
# 4. **Expecting a slicer to affect a measure that overrides that column** (`Net Revenue` and `status`).
# 5. **Summing a non-additive measure** (AOV, margin) yourself for a total.
# 6. **Not validating**: compare card values with a second computation (as in this notebook) before publishing.

# %%
# Demonstration of mistake 5: summing cell margins is meaningless
cell_margin = pd.Series({ch: margin_pct({"channel": [ch]}) for ch in channels})
print("sum of channel margins:", round(cell_margin.sum(), 3), "(meaningless) | overall margin:", round(margin_pct({}), 3))

# %% [markdown]
# ## Interview corner
#
# **1. Row context vs filter context?**
# <details><summary>Answer</summary>
# Row context means "the current row" while iterating (calculated columns, iterator functions such as SUMX) and does not filter other tables by itself. Filter context is the set of filters applied to the model for a given cell (slicers, visual axes, CALCULATE) and determines which rows an aggregation sees. <code>CALCULATE</code> creates or modifies filter context and turns row context into filter context (context transition).
# </details>
#
# **2. Calculated column or measure for profit margin %?**
# <details><summary>Answer</summary>
# A measure: <code>DIVIDE([Profit],[Revenue])</code>. A per-row ratio column averaged later weights every row equally and is wrong at totals; a measure recomputes from summed components in the current context and reacts to slicers.
# </details>
#
# **3. What does <code>CALCULATE([Sales], Product[category] = "Books")</code> do when the visual is already filtered to Electronics?**
# <details><summary>Answer</summary>
# The filter argument replaces the existing filter on <code>Product[category]</code> (unless <code>KEEPFILTERS</code> is used), so the result is Books sales, not an empty intersection. Filters on other columns stay in force.
# </details>
#
# **4. Why use DIVIDE?**
# <details><summary>Answer</summary>
# It handles a zero or blank denominator by returning BLANK (or an alternate value you specify) instead of an error, so visuals stay clean.
# </details>

# %% [markdown]
# ## Exercises
# Engine helpers: `rows`, `calculate`, `divide`, `COMPLETED`, `net_revenue`, `sales_amount`, `n_orders`, `share_of_category`, `BLANK`. Ground-truth frames: `sales`, `done`, `sales_c`.

# %% [markdown]
# ### Exercise 1: Units per order
# DAX target: `Units per Order = DIVIDE ( CALCULATE ( SUM ( Sales[quantity] ), Sales[status] IN {"delivered","shipped"} ), [Orders] )`.
# Write the engine twin `units_per_order(ctx)` (use `calculate`, `rows`, `n_orders`, `divide`). The checker tests it with no filters and with `category = Books`.

# %% tags=["starter"]
def units_per_order(ctx):
    ...

# %%
def _truth(cat=None):
    d = done if cat is None else done[done["category"] == cat]
    return d["quantity"].sum() / d["order_id"].nunique()


assert abs(units_per_order({}) - _truth()) < 1e-9, f"no filter: expected {_truth():.6f}, got {units_per_order({})}"
assert abs(units_per_order({"category": ["Books"]}) - _truth("Books")) < 1e-9, "with category = Books"
print(f"Exercise 1 passed (overall {units_per_order({}):.3f} units per order)")

# %% tags=["solution"]
def units_per_order(ctx):
    units = calculate(lambda c: float(rows(c)["quantity"].sum()) if len(rows(c)) else BLANK, ctx, COMPLETED)
    return divide(units, n_orders(ctx))

# %% [markdown]
# ### Exercise 2: Return rate
# DAX target: `Return Rate = DIVIDE ( CALCULATE ( [Sales Amount], Sales[status] = "returned" ), [Sales Amount] )` (denominator: all statuses).
# Write `return_rate(ctx)`. It is tested overall and for `channel = web`.

# %% tags=["starter"]
def return_rate(ctx):
    ...

# %%
def _rr(mask=None):
    d = sales if mask is None else sales[mask]
    amt = d["quantity"] * d["unit_price"]
    return amt[d["status"] == "returned"].sum() / amt.sum()


assert abs(return_rate({}) - _rr()) < 1e-12, f"overall: expected {_rr():.6f}"
assert abs(return_rate({"channel": ["web"]}) - _rr(sales["channel"] == "web")) < 1e-12, "channel = web"
print(f"Exercise 2 passed (overall return rate {return_rate({}):.2%})")

# %% tags=["solution"]
def return_rate(ctx):
    return divide(calculate(sales_amount, ctx, {"status": ["returned"]}), sales_amount(ctx))

# %% [markdown]
# ### Exercise 3: Which category has the biggest slice of a *channel*?
# For the `web` channel, compute each category's **share of web net revenue** using the engine (`calculate` with `remove_table="Product"`). Store the result as a dict `web_share` `{category: share}` and the winning
# category name in `top_category`. The shares must sum to 1.

# %% tags=["starter"]
web_share = ...
top_category = ...

# %%
_d = done[done["channel"] == "web"].assign(rev=lambda d: d["quantity"] * d["unit_price"])
_exp = (_d.groupby("category")["rev"].sum() / _d["rev"].sum())
assert set(web_share) == set(_exp.index), "one entry per category"
assert all(abs(web_share[c] - _exp[c]) < 1e-9 for c in _exp.index), "shares differ from the groupby computation"
assert abs(sum(web_share.values()) - 1) < 1e-9
assert top_category == _exp.idxmax()
print(f"Exercise 3 passed ({top_category} leads web with {web_share[top_category]:.1%})")

# %% tags=["solution"]
web_share = {c: share_of_all({"channel": ["web"], "category": [c]}) for c in sorted(products["category"].unique())}
top_category = max(web_share, key=web_share.get)

# %% [markdown]
# ### Exercise 4: Ratio of sums or average of ratios?
# Using `sales_c` (completed lines with `row_margin`), compute the margin of **Fashion** two ways: `avg_of_ratios_fashion` (mean of `row_margin`) and `ratio_of_sums_fashion`
# (weighted, from summed revenue and cost). Store `weighted_is_higher` as `True/False` for whether the weighted one is larger.

# %% tags=["starter"]
avg_of_ratios_fashion = ...
ratio_of_sums_fashion = ...
weighted_is_higher = ...

# %%
_f = sales_c[sales_c["category"] == "Fashion"]
_a = _f["row_margin"].mean()
_b = 1 - _f["line_cost"].sum() / _f["line_revenue"].sum()
assert abs(avg_of_ratios_fashion - _a) < 1e-12 and abs(ratio_of_sums_fashion - _b) < 1e-12
assert weighted_is_higher == (_b > _a)
print(f"Exercise 4 passed (avg of ratios {_a:.4%}, ratio of sums {_b:.4%})")

# %% tags=["solution"]
_f = sales_c[sales_c["category"] == "Fashion"]
avg_of_ratios_fashion = _f["row_margin"].mean()
ratio_of_sums_fashion = (_f["line_revenue"].sum() - _f["line_cost"].sum()) / _f["line_revenue"].sum()
weighted_is_higher = bool(ratio_of_sums_fashion > avg_of_ratios_fashion)

# %% [markdown]
# ### Exercise 5 (open-ended): Write and test a new measure
# Pick a KPI: "customers with 2 or more completed orders", "revenue from the store channel", or "average items per order". In a markdown cell write (a) the business definition, (b) the DAX you would enter,
# and (c) a pandas computation of the expected value for the whole data set. Then say which filter (slicer) would make it change and what it should become.

# %% [markdown]
# ## Recap
# - **Measures** answer questions in the current filter context; **calculated columns** store per-row values at refresh. Ratios belong in measures.
# - `SUMX` + `RELATED` handles row-level products across tables; `SUM(a) * AVERAGE(b)` does not.
# - `CALCULATE` replaces same-column filters, adds different-column filters; `ALL`/`ALLEXCEPT` remove filters (for "% of total" style measures); `DIVIDE` avoids errors; `VAR` names intermediate results.
# - Totals of non-additive measures are recomputed in the total cell's context.
# - Validate every KPI by an independent pandas computation: the expected values printed above are your acceptance numbers.
#
# **Go deeper:** the DAX reference on Microsoft Learn (learn.microsoft.com/dax), and the free DAX Guide site by SQLBI for function-level detail.
#
# **Next:** [Time intelligence and visual design](04_time_intelligence_and_visual_design.ipynb)

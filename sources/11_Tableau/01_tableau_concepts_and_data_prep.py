# %% [markdown]
# # Tableau: Concepts and Data Preparation
#
# **Why this matters.** Tableau is fast to *start* and easy to get subtly wrong: a pill that is blue when you wanted green, an "orders" count that is really a count
# of order *lines*, an extract that has not refreshed since Monday. Knowing the vocabulary (dimension/measure, discrete/continuous, relationship/join, live/extract) lets you predict what a view will show and
# prepare data so it behaves. Interviewers for BI roles ask about exactly these.
#
# **You will learn**
# - separate the two independent axes: **dimension vs measure** (what a field *is*) and **discrete vs continuous** (how it is *drawn*)
# - map the workbook structure: data source, worksheet, dashboard, story
# - explain **relationships vs joins** and reproduce the row-multiplication trap with real numbers
# - choose **live vs extract**, and see extract staleness and incremental refresh in a runnable simulation
# - prepare clean, tidy files (including a geographic-role gotcha) and export them to load into Tableau
#
# **Prerequisites:** [pandas joins](../05_Pandas/01_loading_and_inspecting_data.ipynb); the [Power BI notebooks](../10_PowerBI/01_power_bi_overview_and_data_model.ipynb) help but are not required.
#
# > **Candour box (read this).** Tableau is a licensed GUI product and it was **not run** while writing this module. This notebook prepares your data, explains the model precisely, and reproduces in pandas/SQLite
# > the numbers you should see in Tableau. Descriptions of the interface (pill colours, menu names, file types, product editions) come from general knowledge of the product: **verify them in the current
# > Tableau documentation** (help.tableau.com). Cells tagged `no-run` were not executed. You can follow along with Tableau Public or a trial of Tableau Desktop: check current terms yourself. Note that Tableau Public
# > publishes to the public web: **never put confidential data in it.**
# >
# > Data: synthetic shop tables (`docs/DATASETS.md`).

# %%
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("../datasets")
EXPORTS = Path("exports")
EXPORTS.mkdir(exist_ok=True)

customers = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
products = pd.read_csv(DATA / "products.csv")
orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
items = pd.read_csv(DATA / "order_items.csv")

# %% [markdown]
# ## 1. The workbook structure
#
# | Piece | What it is | Typical mistake |
# |---|---|---|
# | **Data source** (tab) | The connection and the logical model: which tables, how they relate, filters, extract or live | Doing data cleaning in every worksheet instead of once here |
# | **Worksheet** | One visualisation: fields dragged onto Rows, Columns, Marks (colour, size, label), Filters | Ten worksheets each with a slightly different filter |
# | **Dashboard** | A layout of worksheets plus objects (text, images, filters, buttons) and actions | Overcrowding; ignoring the target screen size |
# | **Story** | A sequence of "story points", each a worksheet or dashboard with a caption | Using it as a slideshow of unrelated charts |
#
# Editions differ (Desktop authoring, a free public-web edition, and server/cloud hosting for sharing and scheduled refresh). Names and capabilities change: verify in the current documentation.

# %% [markdown]
# ## 2. Dimensions vs measures, discrete vs continuous
#
# Two *independent* properties of a field, and beginners merge them:
#
# | Axis | Choices | Question it answers |
# |---|---|---|
# | **Dimension / Measure** | dimension = a category to slice by (qualitative); measure = a number to aggregate (quantitative) | *What is this field for?* |
# | **Discrete / Continuous** | discrete = separate labelled values (draws **headers**); continuous = an unbroken range (draws an **axis**) | *How should it be drawn?* |
#
# In Tableau's data pane, dimensions are conventionally shown in blue-ish and measures in green-ish, and *pills* in the view are coloured by discrete (blue) versus continuous (green). Verify the colours in your version:
# the point is that **colour on a pill tells you discrete/continuous, not dimension/measure**.
#
# Two consequences to internalise:
# - A **date** can be discrete (`MONTH(Order Date)` = "Jan, Feb, ..." headers, ignoring year unless you add it) or continuous (`MONTH(Order Date)` as a truncated date = a real time axis). Continuous gives correct time axes and forecasts.
# - A **numeric identifier** (`order_id`, `customer_id`) is *numeric* but not a *measure*: summing customer ids is meaningless. Tableau guesses from the data type, so you must re-classify such fields as dimensions.
#
# Let us look at what a *type-only* rule would guess for our fields (Tableau's own defaults are mostly type-based; verify in your version):

# %%
lines = (
    items.merge(orders, on="order_id").merge(products, on="product_id")
    .merge(customers, on="customer_id")
    .assign(revenue=lambda d: d["quantity"] * d["unit_price"], cost=lambda d: d["quantity"] * d["unit_cost"],
            year=lambda d: d["order_date"].dt.year, year_month=lambda d: d["order_date"].dt.strftime("%Y-%m"))
)
guess = pd.DataFrame({
    "pandas dtype": lines.dtypes.astype(str),
    "distinct values": lines.nunique(),
})
guess["Tableau default (by type)"] = np.where(guess["pandas dtype"].str.startswith(("int", "float")), "measure (numeric)",
                                     np.where(guess["pandas dtype"].str.startswith("datetime"), "dimension (date)", "dimension (text)"))
guess.loc[["item_id", "order_id", "customer_id", "product_id", "age", "quantity", "unit_price", "revenue", "order_date", "category"]]

# %% [markdown]
# *What just happened?* By type alone, numeric ids such as `item_id`, `customer_id`, `product_id` look like **measures** (the table above is a type-only rule; Tableau may apply name-based tricks too, so check what your version does), and `SUM(customer_id)` would happily draw a meaningless bar. A better rule uses semantics: is it summable, and is it an identifier?

# %%
def suggest_role(name, s):
    """Heuristic: text/dates and identifiers are dimensions; other numbers are measures."""
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return "dimension"
    if name.endswith("_id") or name in ("year", "month") or s.is_unique:
        return "dimension"          # identifier or calendar part: numeric, but you slice by it, you do not sum it
    return "measure"


roles = pd.Series({c: suggest_role(c, lines[c]) for c in lines.columns})
print(roles[["item_id", "customer_id", "quantity", "unit_price", "revenue", "age", "is_premium", "order_date"]].to_dict())

# %% [markdown]
# `age` is a debatable one (you can average it, but you might also bin it and use it as a dimension). Real judgement beats a heuristic, and **the same field can serve as both**: Tableau lets you convert a measure
# to a dimension or create **bins** of a measure (a histogram is exactly that: a binned dimension and a count).

# %% [markdown]
# ## 3. Relationships vs joins (and the fan-out trap)
#
# **Join** (physical layer): combines tables into *one flat table* before analysis. Every row of the result is a combination, so a one-to-many join **repeats** the "one" side's values.
# **Relationship** (logical layer, the default when you drag a second table into the data source canvas in current Tableau versions; verify): tables stay separate at their **native grain**, and Tableau joins them **at query
# time, based on which fields the current view uses**, aggregating each table appropriately. That protects you from double-counting.
#
# The trap in numbers. We want **orders per channel**. `orders` has one row per order; `order_items` has several lines per order. If we join first (flat) and then `COUNT` rows:

# %%
flat = orders.merge(items, on="order_id")                       # what a JOIN gives you: one row per order LINE
wrong = flat.groupby("channel")["order_id"].count()             # COUNT(order_id) on the flat table = lines
right = flat.groupby("channel")["order_id"].nunique()           # COUNTD(order_id): distinct orders
truth = orders.groupby("channel")["order_id"].count()           # a relationship keeps Orders at order grain
cmp = pd.DataFrame({"COUNT on joined table": wrong, "COUNTD on joined table": right, "orders table (native grain)": truth})
assert (right == truth).all()
print("total 'orders':", int(wrong.sum()), "vs real orders:", int(truth.sum()), f"(overcount x{wrong.sum() / truth.sum():.2f})")
cmp

# %% [markdown]
# *What just happened?* After a join, the fact you thought was "orders" is actually "order lines". With a **join**, use `COUNTD([Order ID])` (or a LOD, notebook 02) to get orders right. With a **relationship** the same view computes
# orders from the `orders` table at its own grain, so plain `COUNT` of that table is correct. The trade-off: relationships are convenient and safe for multi-table models, but some things still need care (for instance
# which table a shared field comes from). Use **joins** when you truly want one flat table (small lookup, or you need a full outer join, or a row-level calculation across tables).
#
# **Join types** (Tableau offers inner, left, right, full outer; **union** stacks tables with the same columns): choose the type by *which rows must survive*. Check row counts before and after every join:

# %%
def join_report(left, right, key, how):
    m = left.merge(right, on=key, how=how)
    return {"how": how, "rows before (left)": len(left), "rows after": len(m)}


pd.DataFrame([join_report(customers, orders[["customer_id", "order_id"]], "customer_id", h) for h in ["inner", "left"]])

# %% [markdown]
# The left join keeps the customers who never ordered (one row each, with a null order); the inner join drops them but multiplies customers by their number of orders. **Both** change the row count: know which you expect.

# %% [markdown]
# ## 4. Live connection vs extract
#
# | | Live | Extract (a snapshot in Tableau's own columnar format) |
# |---|---|---|
# | Data freshness | current at every query | as fresh as the last refresh |
# | Speed | depends on the source database | usually fast (optimised local engine) |
# | Load on source | every interaction queries it | only during refresh |
# | Offline / no source access | no | yes |
# | Best for | fast, well-indexed sources; real-time needs | slow sources, files, and dashboards for many users |
# | Refresh | not needed | scheduled or manual; **full** or **incremental** (only rows newer than the last refresh, using a key or date column) |
#
# Let us simulate the freshness trade-off honestly with SQLite as "the source" and a DataFrame copy as "the extract":

# %%
with tempfile.TemporaryDirectory() as tmp:
    src = Path(tmp) / "source.db"
    con = sqlite3.connect(src)
    con.execute("CREATE TABLE sales (id INTEGER PRIMARY KEY, amount REAL)")
    con.executemany("INSERT INTO sales VALUES (?, ?)", [(i, 10.0) for i in range(1, 1001)])
    con.commit()

    extract = pd.read_sql("SELECT * FROM sales", con)                        # take the extract on Monday
    con.executemany("INSERT INTO sales VALUES (?, ?)", [(i, 10.0) for i in range(1001, 1051)])   # 50 new sales on Tuesday
    con.commit()

    live_total = con.execute("SELECT SUM(amount) FROM sales").fetchone()[0]
    print(f"live query: {live_total:,.0f} | extract (stale): {extract['amount'].sum():,.0f}")

    # incremental refresh: fetch only rows with a key beyond what the extract already has
    new_rows = pd.read_sql("SELECT * FROM sales WHERE id > ?", con, params=(int(extract['id'].max()),))
    extract = pd.concat([extract, new_rows], ignore_index=True)
    print(f"incremental refresh added {len(new_rows)} rows -> extract total {extract['amount'].sum():,.0f}")
    con.close()

# %% [markdown]
# *What just happened?* The extract was **stale** until refreshed; the incremental refresh fetched only the 50 new rows by key. Incremental refresh is fast but has a blind spot: it does **not** notice rows that were *changed* or *deleted* at the
# source (only new keys arrive), so schedule an occasional full refresh. That is a real design decision, not a detail.

# %% [markdown]
# ## 5. Preparing data so Tableau behaves
#
# 1. **Tidy shape**: one row per observation, one column per variable. A sheet with a column per month ("Jan", "Feb", ...) must be **pivoted** (unpivoted to long) before charting: Tableau's data source page has a Pivot option (verify).
# 2. **Clean types**: dates as real dates (ISO `YYYY-MM-DD` is safest), numbers without currency symbols, booleans as True/False.
# 3. **Derived columns you will reuse** (revenue, cost, year-month) can be created once at the source; but calculated fields in Tableau are fine for business logic you want versioned inside the workbook.
# 4. **Geographic roles**: Tableau can plot fields that it recognises as countries, states, cities. Check your codes. Our customers have `country` codes such as `US`, `IN` and `UK`. `UK` is a common
#    everyday abbreviation but the ISO 3166 alpha-2 code for the United Kingdom is `GB`; a tool that expects ISO codes or full names may not recognise `UK`. We add explicit `country_name` and `iso2` columns to be safe (**verify in Tableau** how each is recognised).

# %%
GEO = pd.DataFrame({
    "country": ["US", "IN", "UK", "DE", "CA", "FR", "BD", "AU"],
    "country_name": ["United States", "India", "United Kingdom", "Germany", "Canada", "France", "Bangladesh", "Australia"],
    "iso2": ["US", "IN", "GB", "DE", "CA", "FR", "BD", "AU"],
})
missing = set(customers["country"]) - set(GEO["country"])
assert not missing, f"unmapped country codes: {missing}"
customers_geo = customers.merge(GEO, on="country", how="left", validate="many_to_one")
assert len(customers_geo) == len(customers) and customers_geo["iso2"].notna().all()
customers_geo[["customer_id", "country", "country_name", "iso2"]].head(3)

# %% [markdown]
# **Export the files** for Tableau (`exports/`). Two shapes on purpose: a **flat line-level file** (simplest: connect, then build views) and the **four separate tables**, so you can practise relationships and joins.

# %%
flat_export = lines.merge(GEO, on="country", how="left").assign(
    profit=lambda d: d["revenue"] - d["cost"], order_month=lambda d: d["order_date"].dt.to_period("M").dt.to_timestamp())
flat_export = flat_export[[
    "item_id", "order_id", "customer_id", "order_date", "order_month", "year", "year_month", "status", "channel",
    "product_id", "product_name", "category", "unit_price", "unit_cost", "quantity", "revenue", "cost", "profit",
    "country", "country_name", "iso2", "is_premium", "signup_date", "age",
]].sort_values("item_id")
flat_export.to_csv(EXPORTS / "order_lines_flat.csv", index=False, date_format="%Y-%m-%d")
customers_geo.to_csv(EXPORTS / "customers.csv", index=False, date_format="%Y-%m-%d")
products.to_csv(EXPORTS / "products.csv", index=False)
orders.to_csv(EXPORTS / "orders.csv", index=False, date_format="%Y-%m-%d")
items.to_csv(EXPORTS / "order_items.csv", index=False)
print({p.name: len(pd.read_csv(p)) for p in sorted(EXPORTS.glob("*.csv"))})

# %% [markdown]
# **Row counts to expect** after connecting: `order_lines_flat` should have as many rows as `order_items`. If your Tableau data source shows a different count after adding tables with **joins**, you have found a fan-out.
# (In Tableau: *Data > View Data*, or drag the auto-generated `Number of Records`/`Table (Count)` measure; verify the exact name in your version.)

# %% tags=["no-run"]
TABLEAU_STEPS = """
1. Connect > Text file > exports/order_lines_flat.csv  (or start with the four tables for the relationships exercise)
2. Data source tab: check types. order_date/order_month -> Date, is_premium -> Boolean,
   customer_id/order_id/product_id/item_id -> right-click > Convert to Dimension (or change the default property)
3. Set geographic role on country_name (or iso2) if not detected
4. Choose Live vs Extract (start with Extract for a CSV; it is small either way)
5. Sheet 1: drag Category to Rows, SUM(Revenue) to Columns; sort descending. Compare with the numbers in this notebook.
"""

# %%
# The numbers to compare with your first bar chart in Tableau (all order statuses; filter status in {delivered, shipped} for "net" numbers)
first_view = flat_export.groupby("category")["revenue"].sum().sort_values(ascending=False).round(2)
print("SUM(Revenue) by category, all statuses:")
print(first_view.to_string())
print("total:", round(flat_export["revenue"].sum(), 2))

# %% [markdown]
# ## Common mistakes
#
# 1. **Treating ids as measures** (`SUM(order_id)`). Convert to dimension; count with `COUNTD`.
# 2. **Counting rows of a joined table as orders** (shown above). Use `COUNTD`, a relationship, or a LOD.
# 3. **Confusing blue/green pills**: it is discrete vs continuous, not dimension vs measure. A discrete month gives you headers; a continuous month gives you an axis.
# 4. **Live connection to a slow source** for a dashboard with 200 viewers. Use an extract.
# 5. **Stale extracts**: an unrefreshed dashboard is trusted and wrong. Show the refresh date and schedule refreshes.
# 6. **Publishing sensitive data to the free public edition.** It is public by design.

# %%
# Demonstration of mistake 1: summing an identifier gives a number that means nothing
print("SUM(customer_id) over all lines:", int(lines["customer_id"].sum()), "-> not a business quantity. Distinct customers who ordered:", lines["customer_id"].nunique())

# %% [markdown]
# ## Interview corner
#
# **1. Dimension vs measure, and discrete vs continuous. Are they the same thing?**
# <details><summary>Answer</summary>
# No. Dimension/measure says what a field is (a category you slice by, or a number you aggregate). Discrete/continuous says how it is drawn (separate labelled values that create headers, or an unbroken range that creates an axis). A date can be a dimension used either way; a measure is usually continuous but can be made discrete (for example bins).
# </details>
#
# **2. Relationships vs joins in Tableau?**
# <details><summary>Answer</summary>
# A join merges tables into one flat table up front, so one-to-many joins repeat values and can inflate aggregates. A relationship keeps each table at its own level of detail and combines them at query time based on the fields in the view, which avoids duplication and lets one data source answer questions at different grains. Use joins when you need a flat table or a specific join type.
# </details>
#
# **3. Live or extract?**
# <details><summary>Answer</summary>
# Live gives real-time data but depends on source performance and load; an extract is a fast snapshot that must be refreshed (full or incremental). Choose live for fast sources and real-time needs; choose extracts for speed, offline use, or protecting a slow/operational source. Incremental refresh misses updates and deletes to existing rows.
# </details>
#
# **4. Your dashboard shows twice as many orders as Finance. Where do you look first?**
# <details><summary>Answer</summary>
# The grain: a join to line-level data makes order counts count lines. Check row counts after each join, use COUNTD on the order id, or model with relationships. Then check filters and the definition of "order" (statuses).
# </details>

# %% [markdown]
# ## Exercises
# `lines` (line-level joined data), `orders`, `items`, `customers`, `suggest_role`, `flat_export` are defined above.

# %% [markdown]
# ### Exercise 1: Classify the fields
# Apply `suggest_role` to every column of `flat_export` and store the result in `role_counts`, a dict `{"dimension": n, "measure": m}`.
# Then list in `measures` (a sorted list) the column names classified as measures. Decide: is `profit` among them? (It should be.)

# %% tags=["starter"]
role_counts = ...
measures = ...

# %%
_r = {c: suggest_role(c, flat_export[c]) for c in flat_export.columns}
_exp_measures = sorted(c for c, v in _r.items() if v == "measure")
assert measures == _exp_measures, f"expected {_exp_measures}"
assert role_counts == {"dimension": sum(v == "dimension" for v in _r.values()), "measure": len(_exp_measures)}
assert "profit" in measures and "customer_id" not in measures
print(f"Exercise 1 passed: {role_counts}, measures = {measures}")

# %% tags=["solution"]
_roles = {c: suggest_role(c, flat_export[c]) for c in flat_export.columns}
role_counts = {"dimension": sum(v == "dimension" for v in _roles.values()), "measure": sum(v == "measure" for v in _roles.values())}
measures = sorted(c for c, v in _roles.items() if v == "measure")

# %% [markdown]
# ### Exercise 2: How badly does a join inflate the numbers?
# Join `customers` to `orders` and to `order_items` (in that order, on the natural keys, inner joins). Store in `lines_per_order` the average number of order lines per order (`len(items) / len(orders)`), and in
# `inflation_factor` how many times bigger the row count of the joined table is than the number of customers who have at least one order. Explain in a comment why `COUNT(customer_id)` on the joined table would mislead.

# %% tags=["starter"]
lines_per_order = ...
inflation_factor = ...

# %%
_j = customers.merge(orders, on="customer_id").merge(items, on="order_id")
assert abs(lines_per_order - len(items) / len(orders)) < 1e-12
assert abs(inflation_factor - len(_j) / orders["customer_id"].nunique()) < 1e-9, "inflation = joined rows / distinct customers with orders"
assert inflation_factor > 1
print(f"Exercise 2 passed ({lines_per_order:.2f} lines per order; joined table has {inflation_factor:.2f}x as many rows as buying customers)")

# %% tags=["solution"]
_j = customers.merge(orders, on="customer_id").merge(items, on="order_id")
lines_per_order = len(items) / len(orders)
inflation_factor = len(_j) / _j["customer_id"].nunique()

# %% [markdown]
# ### Exercise 3: Incremental refresh with a date column
# The function below simulates an extract of `orders` taken up to `2024-12-31`. Complete `incremental_refresh(extract, source, date_col)` so it appends **only rows with a date strictly after the extract's latest date** and returns the new extract.
# Store the number of rows added in `n_added`. Then, in a comment, say what a date-based incremental refresh would *not* pick up (think: rows edited or deleted in the past).

# %% tags=["starter"]
source = orders.copy()
extract = source[source["order_date"] <= "2024-12-31"].copy()


def incremental_refresh(extract, source, date_col):
    ...


refreshed = incremental_refresh(extract, source, "order_date")
n_added = ...

# %%
_ex = source[source["order_date"] <= "2024-12-31"]
_new = source[source["order_date"] > _ex["order_date"].max()]
assert len(refreshed) == len(_ex) + len(_new) == len(source), "after refresh the extract has all source rows"
assert n_added == len(_new), f"expected {len(_new)} added rows"
assert refreshed["order_id"].is_unique
print(f"Exercise 3 passed ({n_added} rows added)")

# %% tags=["solution"]
source = orders.copy()
extract = source[source["order_date"] <= "2024-12-31"].copy()


def incremental_refresh(extract, source, date_col):
    cutoff = extract[date_col].max()
    return pd.concat([extract, source[source[date_col] > cutoff]], ignore_index=True)


refreshed = incremental_refresh(extract, source, "order_date")
n_added = len(refreshed) - len(extract)
# not picked up: rows edited or deleted in the past (their dates are not newer than the cutoff)

# %% [markdown]
# ### Exercise 4 (open-ended): Live or extract?
# For each of three situations pick **live** or **extract** (and full or incremental refresh), and justify in one sentence: (a) a dashboard on a 2 GB CSV emailed weekly, (b) a warehouse table updated every minute that a floor manager watches,
# (c) an operational database that must not be hit by 300 dashboard users.

# %% [markdown]
# ## Recap
# - Dimension/measure (what) and discrete/continuous (how drawn) are independent; identifiers are numeric dimensions, not measures.
# - Workbook = data source, worksheets, dashboards, stories. Clean once in the data source.
# - A join flattens and can multiply rows; a relationship keeps native grain. Always check row counts and use `COUNTD` on flat data.
# - Live = fresh but source-dependent; extract = fast snapshot needing refresh (incremental misses edits and deletes).
# - Prepare tidy files, real dates, explicit geographic codes (`UK` vs `GB`). The files are in `exports/`.
#
# **Go deeper:** Tableau Help (help.tableau.com), sections on "Data Sources", "Relationships" and "Extracts". Verify current wording for your version.
#
# **Next:** [Calculated fields and LOD expressions](02_calculated_fields_and_lod_expressions.ipynb)

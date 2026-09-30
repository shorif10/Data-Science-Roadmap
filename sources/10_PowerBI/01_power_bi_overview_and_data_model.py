# %% [markdown]
# # Power BI: Overview and the Data Model
#
# **Why this matters.** Most broken Power BI reports are not broken by bad charts; they are broken by a bad **data model**: totals that
# are too big because a join fanned out, slicers that do nothing because a relationship points the wrong way, a "blank" category
# nobody can explain. If you can design a clean star schema and predict how filters flow through it, DAX (next notebooks) becomes easy,
# and you will be trusted with the numbers.
#
# **You will learn**
# - describe what Power BI Desktop, the Service and the semantic model each do
# - design a **star schema** (fact and dimension tables, grain) for the shop data and export the tables to load into Power BI
# - explain relationship **cardinality** and **filter direction**, and reproduce the effects in pandas
# - detect the three classic model bugs before they reach a dashboard: orphan keys, duplicate dimension keys, and missing date table
#
# **Prerequisites:** [pandas joins](../05_Pandas/01_loading_and_inspecting_data.ipynb) and basic SQL/relational keys.
#
# > **Candour box (read this).** Power BI is a GUI/cloud product and it was **not run** while writing this module. This notebook prepares the
# > data, explains the model, and computes in pandas the numbers your Power BI model must reproduce, so you can validate your report against
# > ground truth. Steps that describe the Power BI interface are written from general knowledge of the product: **menu names and
# > options change between releases, so verify in the current Microsoft documentation** (learn.microsoft.com/power-bi). Code marked `no-run` was not executed.
# >
# > Data: synthetic shop tables in `datasets/` (see `docs/DATASETS.md`).

# %%
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
{name: len(df) for name, df in [("customers", customers), ("products", products), ("orders", orders), ("order_items", items)]}

# %% [markdown]
# ## 1. The Power BI landscape in one table
#
# | Piece | What it is | You use it to |
# |---|---|---|
# | **Power BI Desktop** | Free Windows application for authoring | Connect to data, clean it (Power Query), build the model, write DAX, design report pages |
# | **Power BI Service** | The online (cloud) site | Publish reports, share with colleagues, schedule refresh, set up workspaces and security |
# | **Semantic model** (formerly "dataset") | The tables, relationships and measures behind a report | One model can feed many reports; this is what you are really building |
# | **Report** | Pages of visuals over a model | What people look at |
# | **Dashboard** | A Service-only canvas of tiles pinned from reports | A one-screen summary (do not confuse with "report") |
# | **Gateway** | A bridge that lets the cloud Service refresh data that lives on-premises | Scheduled refresh from internal databases |
#
# Verify current names, licensing and what is included in which licence in Microsoft's documentation: they change, and we do not quote them here.
#
# **Storage modes** (per table): **Import** copies data into the model's in-memory engine (fast visuals, needs refresh); **DirectQuery** sends queries to the source at view time
# (fresher data, slower and source-dependent, some DAX limits); **Live connection** points at an existing published model. Import is the sensible default unless data is too big or must be live.

# %% [markdown]
# ## 2. Star schema: the model shape Power BI is built for
#
# A **fact table** records events at a chosen **grain** (one row per ...). **Dimension tables** describe the things you slice by. In a star schema each dimension relates to the fact by one key,
# so filters flow **from the dimension (one side) down to the fact (many side)**. This shape is what the engine is optimised for, and it makes DAX behave predictably.
#
# Our raw data is a *chain*: `customers 1-* orders 1-* order_items *-1 products`. That is workable, but a cleaner model is:
#
# ```
#            dim_customer            dim_product
#                  \                    /
#                   \                  /
#   dim_date ------- fact_sales  (grain: one row per ORDER LINE)
# ```
#
# We **flatten the order header onto the line** (the order's date, status and channel become columns of the fact), because at line grain they have one value each. Choose the grain first; everything else follows from it.

# %%
fact_sales = (
    items.merge(orders, on="order_id", how="left", validate="many_to_one")
    [["item_id", "order_id", "order_date", "customer_id", "product_id", "status", "channel", "quantity"]]
    .sort_values("item_id", ignore_index=True)
)
dim_customer = customers.copy()
dim_product = products.copy()


def make_dim_date(start, end):
    d = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    d["year"] = d["date"].dt.year
    d["quarter"] = "Q" + d["date"].dt.quarter.astype(str)
    d["month_number"] = d["date"].dt.month
    d["month_name"] = d["date"].dt.strftime("%b")
    d["year_month"] = d["date"].dt.strftime("%Y-%m")
    d["weekday_number"] = d["date"].dt.dayofweek + 1        # Monday = 1
    d["is_weekend"] = d["weekday_number"] >= 6
    return d


# a date table must be CONTINUOUS and cover whole years: from Jan 1 of the first year to Dec 31 of the last
first_year, last_year = fact_sales["order_date"].dt.year.min(), fact_sales["order_date"].dt.year.max()
dim_date = make_dim_date(f"{first_year}-01-01", f"{last_year}-12-31")
print({"fact_sales": fact_sales.shape, "dim_customer": dim_customer.shape, "dim_product": dim_product.shape, "dim_date": dim_date.shape})
fact_sales.head(3)

# %% [markdown]
# *What just happened?* The `validate="many_to_one"` argument makes pandas **raise** if an order id were duplicated in `orders`. That is a cheap, real-world habit: assert
# your assumptions about keys when you build a model. The date table spans complete years because time intelligence (notebook 04) requires an uninterrupted calendar.

# %% [markdown]
# ## 3. Export the tables to load into Power BI
#
# We write CSV files (universally supported) into `exports/`. In Power BI Desktop you would use *Get data > Text/CSV* (verify the current wording) for each file, then check the data types
# it detected (dates as **Date**, keys as **Whole number**, `is_premium` and `is_weekend` as **True/False**).

# %%
fact_sales.to_csv(EXPORTS / "fact_sales.csv", index=False, date_format="%Y-%m-%d")
dim_customer.to_csv(EXPORTS / "dim_customer.csv", index=False, date_format="%Y-%m-%d")
dim_product.to_csv(EXPORTS / "dim_product.csv", index=False)
dim_date.to_csv(EXPORTS / "dim_date.csv", index=False, date_format="%Y-%m-%d")
sorted(p.name for p in EXPORTS.iterdir())

# %% [markdown]
# **Row counts to expect after loading** (Power BI: use a table visual with a *Count* of the key, or "Model view > table hover"; verify):

# %%
pd.Series({"fact_sales": len(fact_sales), "dim_customer": len(dim_customer), "dim_product": len(dim_product), "dim_date": len(dim_date)}, name="rows")

# %% [markdown]
# ## 4. Relationships: which keys, which direction
#
# In the model view you draw (or Power BI auto-detects; check every one) these relationships, each **one-to-many, single direction**:
#
# | From (one side) | To (many side) | Key |
# |---|---|---|
# | `dim_customer[customer_id]` | `fact_sales[customer_id]` | customer |
# | `dim_product[product_id]` | `fact_sales[product_id]` | product |
# | `dim_date[date]` | `fact_sales[order_date]` | order date |
#
# Rules that prevent most trouble:
# 1. The **one side must have unique keys**. Power BI will refuse (or misbehave) otherwise: verify in pandas first.
# 2. Every fact key should exist in the dimension; if not, Power BI shows a **(Blank)** category.
# 3. Keep the direction **single** (dimension filters fact) unless you can explain why not.

# %%
def check_key(dim, dim_key, fact, fact_key):
    return {
        "dim rows": len(dim),
        "dim key unique": bool(dim[dim_key].is_unique),
        "fact rows": len(fact),
        "orphan fact rows": int((~fact[fact_key].isin(dim[dim_key])).sum()),
    }


pd.DataFrame({
    "customer": check_key(dim_customer, "customer_id", fact_sales, "customer_id"),
    "product": check_key(dim_product, "product_id", fact_sales, "product_id"),
    "date": check_key(dim_date, "date", fact_sales, "order_date"),
})

# %% [markdown]
# All keys are unique and there are no orphans, so this model is safe. Let us break it on purpose to see the symptoms.

# %%
# Bug 1: orphan keys. Suppose 3 products were dropped from the product dimension by a bad filter.
broken_products = dim_product[~dim_product["product_id"].isin([1, 2, 3])]
orph = fact_sales[~fact_sales["product_id"].isin(broken_products["product_id"])]
print("fact rows with no matching product:", len(orph), "-> these would be grouped under (Blank) in a visual by product")

# Bug 2: duplicate dimension keys make the join fan out and inflate totals
dup_customers = pd.concat([dim_customer, dim_customer.head(50)], ignore_index=True)
true_lines = len(fact_sales)
fanned = fact_sales.merge(dup_customers, on="customer_id", how="left")
print(f"fact rows: {true_lines}, after joining to a customer table with duplicates: {len(fanned)} (extra rows: {len(fanned) - true_lines})")

# %% [markdown]
# *What just happened?* Orphans do not disappear: Power BI keeps the fact rows and shows them under **(Blank)**, so totals stay right but a mysterious blank bar appears.
# **Duplicate dimension keys are worse**: in a merge they multiply fact rows, and any sum taken over the merged table is overstated. (In Power BI a duplicate key on the "one" side is
# rejected for a one-to-many relationship; people then "fix" it by choosing many-to-many, which hides the problem. Fix the dimension instead.)

# %% [markdown]
# ## 5. Filter direction, emulated in pandas
#
# A **slicer** or filter on a table restricts that table's rows, and the filter **propagates across relationships in the allowed direction**. With single-direction relationships, a filter on
# `dim_product` reaches `fact_sales`, and through it nothing further, because there is no relationship from the fact to `dim_customer` that points that way.
# Let us apply a slicer `category = "Books"` and ask two questions.

# %%
books = dim_product.loc[dim_product["category"] == "Books", "product_id"]
fact_books = fact_sales[fact_sales["product_id"].isin(books)]          # the filter flows dim_product -> fact_sales

print("fact rows after the Books slicer:", len(fact_books), "of", len(fact_sales))
print("units of Books sold:", int(fact_books["quantity"].sum()))
print("customers in a dim_customer slicer, single direction:", len(dim_customer), "(unaffected)")
print("customers if the filter also flowed fact -> customer (both directions):", fact_books["customer_id"].nunique())

# %% [markdown]
# *What just happened?* With **single** direction, a slicer that lists customer names still shows all 2,000 of them after you pick "Books"; with **both** it would list only customers who bought a Book.
# That sounds handy, but bidirectional filtering can create ambiguous paths between tables, slows queries, and changes the meaning of measures. Prefer single direction and, when you need this behaviour
# for one calculation, enable it inside a measure with `CROSSFILTER` or use a visual-level filter. (Names of these options: verify in the docs.)
#
# **Active vs inactive relationships.** Between two tables only one relationship can be *active*. `dim_customer[signup_date]` could also relate to `dim_date[date]` (a **role-playing dimension**: the same
# date table used for order date and for signup date). The second one is created inactive, and you switch it on inside a measure with `USERELATIONSHIP` (notebook 04).

# %% [markdown]
# ## 6. Model hygiene checklist
#
# | Do | Why |
# |---|---|
# | Load only the columns you use | Smaller model, faster refresh |
# | One fact table per grain, dimensions for descriptors | Predictable filtering |
# | Mark a proper **date table** and relate it | Time intelligence needs a continuous calendar (notebook 04) |
# | Hide key columns and raw numeric columns from the report view | People should use measures, not sum a raw column |
# | Name tables and fields for humans (`Customer`, not `dim_customer_v2_final`) | The report author and the reader are not you |
# | Keep measures in a dedicated place and give them clear names | Findability |

# %% [markdown]
# ## Common mistakes
#
# 1. **One giant flat table** (everything joined into one sheet). It works for a tiny report, then duplicates dimension values, blows up size, and makes distinct counts wrong.
# 2. **Auto-detected relationships accepted blindly.** Always inspect key, cardinality and direction.
# 3. **Bidirectional filters everywhere** "to make the slicer work": leads to ambiguity and slow, surprising numbers.
# 4. **No date table** (relying on the auto date/time feature): fine for a quick demo, inadequate for fiscal calendars and reusable time logic.
# 5. **Many-to-many to silence a key error** instead of cleaning the dimension.
# 6. **Ignoring a (Blank) category.** It is a data quality signal, not a cosmetic issue.

# %%
# Demonstration of mistake 1: sum of a dimension attribute on the flat table is overstated
flat = fact_sales.merge(dim_customer, on="customer_id")
print("customers by summing 1 per row of the flat table:", len(flat), "| real number of distinct customers who bought:", flat["customer_id"].nunique())

# %% [markdown]
# ## Interview corner
#
# **1. What is a star schema and why does Power BI prefer it?**
# <details><summary>Answer</summary>
# One fact table at a defined grain surrounded by dimension tables joined on single keys. Filters flow from dimensions to the fact along one-to-many relationships, which the storage engine handles efficiently and which makes DAX filter behaviour easy to reason about. It also avoids duplicated descriptive text and ambiguous paths.
# </details>
#
# **2. What does cross-filter direction control?**
# <details><summary>Answer</summary>
# Whether a filter on one table propagates through a relationship in one direction (from the "one" side to the "many" side, the default and usual choice) or both directions. Both can cause ambiguity and performance cost, so use it sparingly.
# </details>
#
# **3. A visual shows a (Blank) category. What is the likely cause?**
# <details><summary>Answer</summary>
# Fact rows whose key has no match in the dimension (orphans), or nulls in the fact key. Power BI keeps these rows and groups them under Blank. Check referential integrity and fix the source or the dimension.
# </details>
#
# **4. Why should you not use many-to-many relationships casually?**
# <details><summary>Answer</summary>
# They hide non-unique keys instead of fixing them, can make totals ambiguous (results depend on which side filters which), and are harder to debug. Prefer a proper dimension with unique keys, or a bridge table when the relationship is truly many-to-many.
# </details>

# %% [markdown]
# ## Exercises
# `fact_sales`, `dim_customer`, `dim_product`, `dim_date`, `orders`, `items`, `products`, `customers`, `check_key`, `make_dim_date` exist from above.

# %% [markdown]
# ### Exercise 1: Find the orphans
# A junior analyst filtered `dim_product` to products with `unit_price < 500`, calling the result `cheap_products`. How many rows of `fact_sales` would show up under **(Blank)** for product,
# and what share of all `quantity` units do they carry? Store `n_blank_rows` and `blank_units_share` (0 to 1).

# %% tags=["starter"]
cheap_products = dim_product[dim_product["unit_price"] < 500]
n_blank_rows = ...
blank_units_share = ...

# %%
_orph = fact_sales[~fact_sales["product_id"].isin(cheap_products["product_id"])]
assert n_blank_rows == len(_orph), f"expected {len(_orph)} orphan rows, got {n_blank_rows}"
assert abs(blank_units_share - _orph["quantity"].sum() / fact_sales["quantity"].sum()) < 1e-9, "share = orphan units / total units"
print(f"Exercise 1 passed ({n_blank_rows} rows, {blank_units_share:.1%} of units)")

# %% tags=["solution"]
cheap_products = dim_product[dim_product["unit_price"] < 500]
_o = fact_sales[~fact_sales["product_id"].isin(cheap_products["product_id"])]
n_blank_rows = len(_o)
blank_units_share = _o["quantity"].sum() / fact_sales["quantity"].sum()

# %% [markdown]
# ### Exercise 2: Build a date table
# Write `make_calendar(start, end)` returning a DataFrame with columns `date, year, month_number, year_month` (in that order), one row per day from `start` to `end` inclusive.
# It must be continuous with no gaps. *Hint: `pd.date_range`, `.dt.year`, `.dt.month`, `.dt.strftime("%Y-%m")`.*

# %% tags=["starter"]
def make_calendar(start, end):
    ...

# %%
_c = make_calendar("2024-01-01", "2024-12-31")
assert list(_c.columns) == ["date", "year", "month_number", "year_month"], "columns must be date, year, month_number, year_month"
assert len(_c) == 366, "2024 is a leap year: 366 days"
assert _c["date"].is_monotonic_increasing and (_c["date"].diff().dropna() == pd.Timedelta(days=1)).all(), "dates must be continuous"
assert (_c.loc[_c["year_month"] == "2024-02"].shape[0]) == 29
assert _c["date"].iloc[0] == pd.Timestamp("2024-01-01") and _c["date"].iloc[-1] == pd.Timestamp("2024-12-31")
print("Exercise 2 passed")

# %% tags=["solution"]
def make_calendar(start, end):
    d = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    d["year"] = d["date"].dt.year
    d["month_number"] = d["date"].dt.month
    d["year_month"] = d["date"].dt.strftime("%Y-%m")
    return d

# %% [markdown]
# ### Exercise 3: Quantify the fan-out
# `dup_customers` (defined above) has 50 customers listed twice. Join `fact_sales` to it on `customer_id` and store in `overstated_units` how many **units too many** `quantity.sum()` reports
# compared with the correct total. Then store in `dup_key_count` the number of duplicated `customer_id` values in `dup_customers`.

# %% tags=["starter"]
overstated_units = ...
dup_key_count = ...

# %%
_true = int(fact_sales["quantity"].sum())
_dups = set(dup_customers.loc[dup_customers["customer_id"].duplicated(keep=False), "customer_id"])
_extra = int(fact_sales.loc[fact_sales["customer_id"].isin(_dups), "quantity"].sum())
assert dup_key_count == 50, f"there are 50 duplicated keys, got {dup_key_count}"
assert overstated_units == _extra, f"expected {_extra}, got {overstated_units}"
assert overstated_units > 0
print(f"Exercise 3 passed ({overstated_units} extra units on a true total of {_true})")

# %% tags=["solution"]
_m = fact_sales.merge(dup_customers, on="customer_id", how="left")
overstated_units = int(_m["quantity"].sum() - fact_sales["quantity"].sum())
dup_key_count = int(dup_customers["customer_id"].duplicated().sum())

# %% [markdown]
# ### Exercise 4 (open-ended): Design a star schema
# Choose a dataset you know from work (HR headcount, support tickets, website sessions). In a markdown cell state the **grain** of the fact table, list 3 to 4 dimensions,
# name each relationship's key and cardinality, and say which date(s) need a date table (and whether any is role-playing).

# %% [markdown]
# ## Recap
# - Desktop authors, the Service shares; the **semantic model** (tables, relationships, measures) is the real product.
# - Design a **star schema** at a chosen grain: fact table plus dimensions; flatten header attributes onto the line when the grain allows.
# - Keys: unique on the one side, no orphans on the many side (else **(Blank)**); duplicates fan out joins and inflate totals.
# - Prefer one-to-many, single-direction relationships; use inactive relationships plus `USERELATIONSHIP` for role-playing dates.
# - Always validate a model against ground truth computed independently: the row counts and checks above are the start.
#
# **Go deeper:** Microsoft Learn, "Power BI guidance documentation" (star schema, relationships) at learn.microsoft.com/power-bi.
#
# **Next:** [Power Query and data preparation](02_power_query_and_data_prep.ipynb)

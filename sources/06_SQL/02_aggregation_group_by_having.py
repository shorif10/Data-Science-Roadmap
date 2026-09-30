# %% [markdown]
# # Aggregation, GROUP BY and HAVING
#
# **Why this matters.** Nobody wants 5,000 order rows; they want "orders per channel", "cancellation rate by month", "average
# customer age by country". Turning many rows into a few summary numbers is the most common thing analysts do in SQL,
# and the classic source of subtle wrong answers (counting the wrong thing, filtering in the wrong clause).
#
# **You will learn**
# - summarise with `COUNT`, `SUM`, `AVG`, `MIN`, `MAX` and know what `COUNT(*)`, `COUNT(col)` and `COUNT(DISTINCT col)` each count
# - split a table into groups with `GROUP BY` (one key, several keys, an expression such as "month")
# - filter groups with `HAVING` and rows with `WHERE`, and explain the difference
# - compute rates with **conditional aggregation** (`SUM(CASE WHEN ...)`, `FILTER`)
#
# **Prerequisites:** [SELECT, WHERE, ORDER BY, LIMIT](01_select_where_orderby_limit.ipynb).
#
# > Data used: `datasets/shop.db` (synthetic online shop), opened read-only.

# %%
import sqlite3

import pandas as pd

con = sqlite3.connect("file:../datasets/shop.db?mode=ro", uri=True)


def q(sql, params=None):
    return pd.read_sql(sql, con, params=params)


q("SELECT COUNT(*) AS n_orders, MIN(order_date) AS first_order, MAX(order_date) AS last_order FROM orders")

# %% [markdown]
# ## 1. Aggregate functions collapse many rows into one
#
# Without `GROUP BY`, an aggregate treats the **whole table** as one group and returns exactly one row.

# %%
q("""
SELECT COUNT(*)                 AS n_products,
       MIN(unit_price)          AS cheapest,
       MAX(unit_price)          AS priciest,
       ROUND(AVG(unit_price), 2) AS mean_price,
       ROUND(SUM(unit_cost), 2)  AS total_cost_of_one_of_each
FROM products
""")

# %% [markdown]
# Three flavours of counting get confused constantly:
# - `COUNT(*)` counts **rows**.
# - `COUNT(col)` counts rows where `col` is **not NULL**.
# - `COUNT(DISTINCT col)` counts **different non-NULL values**.
#
# Our tables have no NULLs, so let us watch them differ on an inline table where they do.

# %%
q("""
WITH t(customer, coupon) AS (VALUES ('a', 'X1'), ('a', 'X1'), ('b', NULL), ('c', 'Z9'))
SELECT COUNT(*) AS rows_, COUNT(coupon) AS non_null_coupons, COUNT(DISTINCT coupon) AS distinct_coupons,
       AVG(LENGTH(coupon)) AS avg_len_ignores_null
FROM t
""")

# %% [markdown]
# *What just happened?* Four rows, but three have a coupon and only two coupons are different. Every aggregate except
# `COUNT(*)` **silently skips NULLs**. That makes `AVG` the average of the known values, not of all rows: a real trap if
# "NULL" secretly means "zero".
#
# On real data, `COUNT(DISTINCT customer_id)` answers "how many *different* customers ordered?", which is not the same as the number of orders:

# %%
q("SELECT COUNT(*) AS orders, COUNT(DISTINCT customer_id) AS distinct_customers FROM orders")

# %% [markdown]
# ## 2. GROUP BY: one summary row per group
#
# Imagine sorting the rows into piles by `channel`, then summarising each pile. That is `GROUP BY`.

# %%
q("""
SELECT channel, COUNT(*) AS n_orders
FROM orders
GROUP BY channel
ORDER BY n_orders DESC
""")

# %% [markdown]
# **The rule:** every column in `SELECT` must either be in `GROUP BY` or wrapped in an aggregate. Otherwise "which
# row's value should I show for the pile?" has no answer. Postgres and Snowflake reject the query; SQLite quietly
# picks an arbitrary row's value (see below), which is convenient and dangerous.

# %%
# SQLite lets this through; Postgres would raise an error. The 'status' shown per channel is arbitrary.
q("SELECT channel, status, COUNT(*) AS n FROM orders GROUP BY channel")

# %% [markdown]
# Group by several columns to get one row per combination, or by an **expression** such as a month:

# %%
q("""
SELECT channel, status, COUNT(*) AS n
FROM orders
GROUP BY channel, status
ORDER BY channel, n DESC
LIMIT 6
""")

# %%
monthly = q("""
SELECT strftime('%Y-%m', order_date) AS month, COUNT(*) AS n_orders
FROM orders
GROUP BY month
ORDER BY month
""")
monthly.tail(6)

# %% [markdown]
# SQLite (like MySQL and Postgres) lets you `GROUP BY` an alias defined in `SELECT`. Strictly, the standard says you should
# repeat the expression, or refer to the column position (`GROUP BY 1`), which is handy in exploration and fragile in shared code.
# Look at the last rows: December is far busier than the months after it (seasonality; exercise 6 explores it).

# %% [markdown]
# ## 3. WHERE vs HAVING
#
# - `WHERE` filters **rows** before grouping (it cannot see aggregates).
# - `HAVING` filters **groups** after aggregation (it can).
#
# "Which months of 2024 had more than 300 non-cancelled orders?" needs both:

# %%
q("""
SELECT strftime('%Y-%m', order_date) AS month, COUNT(*) AS n_orders
FROM orders
WHERE status <> 'cancelled'                       -- row filter: before grouping
  AND order_date >= '2024-01-01' AND order_date < '2025-01-01'
GROUP BY month
HAVING COUNT(*) > 300                             -- group filter: after aggregation
ORDER BY n_orders DESC
""")

# %% [markdown]
# Prefer `WHERE` whenever a condition talks about a raw column: the database throws rows away early, and the query is cheaper.
# Using `HAVING status <> 'cancelled'` would give the same rows here in SQLite but is wrong style and fails in stricter engines.

# %% [markdown]
# ## 4. Conditional aggregation: rates in one pass
#
# A very common pattern: one scan, several conditions counted side by side. `SUM(CASE WHEN cond THEN 1 ELSE 0 END)` counts the rows meeting `cond`.
# In SQLite a comparison already returns 0/1, so `AVG(cond)` is directly a **rate**.

# %%
q("""
SELECT channel,
       COUNT(*)                                                     AS n_orders,
       SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END)        AS n_cancelled,
       ROUND(100.0 * AVG(status = 'cancelled'), 2)                  AS cancel_pct,
       ROUND(100.0 * AVG(status = 'returned'), 2)                   AS return_pct
FROM orders
GROUP BY channel
ORDER BY cancel_pct DESC
""")

# %% [markdown]
# SQL also has a cleaner standard spelling, `COUNT(*) FILTER (WHERE status = 'cancelled')`, which SQLite (3.30+), Postgres and DuckDB support.
#
# | Task | SQLite | Postgres | MySQL | Snowflake |
# |---|---|---|---|---|
# | conditional count | `COUNT(*) FILTER (WHERE c)` or `SUM(CASE ...)` | `FILTER` | `SUM(CASE ...)` (no `FILTER`) | `COUNT_IF(c)` or `SUM(CASE ...)` |
# | boolean as number | `AVG(a = b)` works (0/1) | needs `AVG((a = b)::int)` | `AVG(a = b)` works | needs `AVG(IFF(...,1,0))` or cast |
#
# `SUM(CASE ...)` is the one form that works **everywhere**, so learn it first.

# %%
q("""
SELECT channel,
       COUNT(*) FILTER (WHERE status = 'delivered') AS delivered,
       COUNT(*) FILTER (WHERE status = 'shipped')   AS in_transit
FROM orders GROUP BY channel
""")

# %% [markdown]
# A rate from a small group is noisy. Always show the **denominator** beside the rate (we did with `n_orders`), and be suspicious of
# "the top segment by rate" when that segment has 6 orders.

# %% [markdown]
# ## 5. Grouping the quantity table
# Aggregates work on any numeric column. `order_items` has one row per product on an order; here are the best-selling products by **units**
# (the product *names* need a join, coming in the next notebook, so we group by id):

# %%
q("""
SELECT product_id, SUM(quantity) AS units, COUNT(*) AS lines, ROUND(AVG(quantity), 2) AS avg_qty_per_line
FROM order_items
GROUP BY product_id
ORDER BY units DESC, product_id
LIMIT 5
""")

# %% [markdown]
# ## Common mistakes
#
# 1. **Aggregating after a join that multiplied your rows** (fan-out). We show it in notebook 03; the symptom is a total that is too large.
# 2. **Putting an aggregate in `WHERE`** (`WHERE COUNT(*) > 5`) fails: use `HAVING`.
# 3. **`COUNT(col)` when you meant `COUNT(*)`**: rows with NULL in `col` disappear from the count.
# 4. **Averaging averages.** The mean of per-group means is not the overall mean unless groups have equal size. Aggregate `SUM` and `COUNT` and divide.
# 5. **Selecting non-grouped columns** and trusting SQLite's arbitrary value.
# 6. **Integer division** in ratios (`SUM(a) / COUNT(*)` on integers): multiply by `1.0`.

# %%
# Mistake 4: the mean of monthly cancel rates is not the overall cancel rate.
by_month = q("SELECT strftime('%Y-%m', order_date) m, AVG(status = 'cancelled') AS rate, COUNT(*) AS n FROM orders GROUP BY m")
overall = q("SELECT AVG(status = 'cancelled') AS rate FROM orders").rate[0]
print(f"average of monthly rates: {by_month.rate.mean():.4f}   overall rate: {overall:.4f}   weighted monthly: {(by_month.rate * by_month.n).sum() / by_month.n.sum():.4f}")

# %% [markdown]
# ## Interview corner
#
# **1. Difference between <code>COUNT(*)</code>, <code>COUNT(col)</code> and <code>COUNT(DISTINCT col)</code>?**
# <details><summary>Answer</summary>
# Rows; rows with non-NULL <code>col</code>; number of distinct non-NULL values. Only <code>COUNT(*)</code> includes NULL rows.
# </details>
#
# **2. Why can't you use an aggregate in <code>WHERE</code>?**
# <details><summary>Answer</summary>
# <code>WHERE</code> runs before grouping, so aggregates do not exist yet. <code>HAVING</code> runs after <code>GROUP BY</code> and can use them.
# </details>
#
# **3. How would you compute conversion or cancellation rate per segment in one query?**
# <details><summary>Answer</summary>
# Conditional aggregation: <code>SUM(CASE WHEN cond THEN 1 ELSE 0 END) * 1.0 / COUNT(*)</code> grouped by the segment. Show the group size as well, since small groups give noisy rates.
# </details>
#
# **4. What happens to NULLs in <code>AVG</code> and in <code>GROUP BY</code>?**
# <details><summary>Answer</summary>
# <code>AVG</code> ignores NULL (it divides by the count of non-NULL values). <code>GROUP BY</code> puts all NULLs together into one group.
# </details>

# %% [markdown]
# ## Exercises

# %%
orders = q("SELECT * FROM orders")
customers = q("SELECT * FROM customers")
items = q("SELECT * FROM order_items")

# %% [markdown]
# ### Exercise 1: Order status mix
# Return one row per order `status` with columns `status`, `n_orders` and `share_pct` (percentage of all orders, rounded to 2 decimals), most frequent first.
# *Hint: a subquery `(SELECT COUNT(*) FROM orders)` gives the total, use `100.0 *` to avoid integer division.*

# %% tags=["starter"]
ex1_sql = """
SELECT ...
"""

# %%
res = q(ex1_sql)
vc = orders.status.value_counts()
assert list(res.columns) == ["status", "n_orders", "share_pct"], f"columns: {list(res.columns)}"
assert res.status.tolist() == vc.index.tolist(), "order the rows by n_orders descending"
assert res.n_orders.tolist() == vc.tolist()
assert ((res.share_pct - (vc / vc.sum() * 100).round(2).values).abs() < 0.011).all(), "share_pct should be a percentage of ALL orders"
print("Exercise 1 passed")

# %% tags=["solution"]
ex1_sql = """
SELECT status,
       COUNT(*) AS n_orders,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM orders), 2) AS share_pct
FROM orders
GROUP BY status
ORDER BY n_orders DESC
"""

# %% [markdown]
# ### Exercise 2: Which channel loses the most orders?
# For each `channel` return `channel`, `n_orders`, and `problem_pct`: the percentage (2 decimals) of orders whose status is `cancelled` **or** `returned`.
# Sort by `problem_pct` descending. *Hint: `IN ('cancelled','returned')` inside a `CASE`, or inside `AVG(...)`.*

# %% tags=["starter"]
ex2_sql = """
SELECT ...
"""

# %%
res = q(ex2_sql)
g = orders.assign(bad=orders.status.isin(["cancelled", "returned"])).groupby("channel").agg(n=("bad", "size"), rate=("bad", "mean"))
g = g.sort_values("rate", ascending=False)
assert list(res.columns) == ["channel", "n_orders", "problem_pct"], f"columns: {list(res.columns)}"
assert res.channel.tolist() == g.index.tolist(), "wrong channel order"
assert res.n_orders.tolist() == g.n.tolist()
assert ((res.problem_pct - (g.rate * 100).round(2).values).abs() < 0.011).all(), "problem_pct differs"
print(f"Exercise 2 passed: worst channel is {res.channel[0]} at {res.problem_pct[0]}%")

# %% tags=["solution"]
ex2_sql = """
SELECT channel,
       COUNT(*) AS n_orders,
       ROUND(100.0 * AVG(status IN ('cancelled', 'returned')), 2) AS problem_pct
FROM orders
GROUP BY channel
ORDER BY problem_pct DESC
"""

# %% [markdown]
# ### Exercise 3: Busy months (WHERE plus HAVING)
# Count **delivered** orders per month (`YYYY-MM`) and keep only months with **at least 250** delivered orders. Return `month`, `n_delivered`, sorted by month.
# *Hint: status goes in `WHERE`, the threshold in `HAVING`.*

# %% tags=["starter"]
ex3_sql = """
SELECT ...
"""

# %%
res = q(ex3_sql)
d = orders[orders.status == "delivered"].assign(month=lambda x: x.order_date.str[:7]).groupby("month").size()
d = d[d >= 250]
assert len(d) > 0, "sanity: threshold should keep some months"
assert list(res.columns) == ["month", "n_delivered"], f"columns: {list(res.columns)}"
assert res.month.tolist() == d.index.tolist(), f"expected months {d.index.tolist()}"
assert res.n_delivered.tolist() == d.tolist()
print(f"Exercise 3 passed ({len(res)} busy months)")

# %% tags=["solution"]
ex3_sql = """
SELECT strftime('%Y-%m', order_date) AS month, COUNT(*) AS n_delivered
FROM orders
WHERE status = 'delivered'
GROUP BY month
HAVING COUNT(*) >= 250
ORDER BY month
"""

# %% [markdown]
# ### Exercise 4: Countries worth a marketing budget
# From `customers`, per `country` return `country`, `n_customers`, `premium_pct` (percentage premium, 1 decimal) and `avg_age` (1 decimal). Keep only countries with **more than 150 customers**
# and sort by `premium_pct` descending, then `country`.

# %% tags=["starter"]
ex4_sql = """
SELECT ...
"""

# %%
res = q(ex4_sql)
g = customers.groupby("country").agg(n=("customer_id", "size"), prem=("is_premium", "mean"), age=("age", "mean"))
g = g[g.n > 150].reset_index().assign(prem=lambda x: (x.prem * 100), age=lambda x: x.age)
g = g.sort_values(["prem", "country"], ascending=[False, True]).reset_index(drop=True)
assert list(res.columns) == ["country", "n_customers", "premium_pct", "avg_age"], f"columns: {list(res.columns)}"
assert res.country.tolist() == g.country.tolist(), "country selection/order differs (HAVING COUNT(*) > 150; sort by premium_pct DESC, country)"
assert ((res.premium_pct - g.prem).abs() < 0.06).all() and ((res.avg_age - g.age).abs() < 0.06).all(), "premium_pct/avg_age differ (round to 1 decimal)"
print("Exercise 4 passed")

# %% tags=["solution"]
ex4_sql = """
SELECT country,
       COUNT(*) AS n_customers,
       ROUND(100.0 * AVG(is_premium), 1) AS premium_pct,
       ROUND(AVG(age), 1) AS avg_age
FROM customers
GROUP BY country
HAVING COUNT(*) > 150
ORDER BY premium_pct DESC, country
"""

# %% [markdown]
# ### Exercise 5: Heavy baskets
# In `order_items`, an order's **basket size** is the sum of `quantity` over its lines. Return a single row with one column, `n_heavy_orders`: how many orders have a basket size of **5 or more**.
# *Hint: aggregate per order in a subquery, then count the rows that pass the filter. (Or `GROUP BY` + `HAVING` inside a subquery.)*

# %% tags=["starter"]
ex5_sql = """
SELECT ...
"""

# %%
res = q(ex5_sql)
exp = int((items.groupby("order_id").quantity.sum() >= 5).sum())
assert exp > 0, "sanity"
assert list(res.columns) == ["n_heavy_orders"] and len(res) == 1, "one row, one column named n_heavy_orders"
assert res.n_heavy_orders[0] == exp, f"expected {exp}, got {res.n_heavy_orders[0]}"
print("Exercise 5 passed:", exp)

# %% tags=["solution"]
ex5_sql = """
SELECT COUNT(*) AS n_heavy_orders
FROM (SELECT order_id
      FROM order_items
      GROUP BY order_id
      HAVING SUM(quantity) >= 5)
"""

# %% [markdown]
# ### Exercise 6 (open-ended): Seasonality
# Using one `GROUP BY` query on `orders`, compare the number of orders per **calendar month (1-12)** across the two full years in the data, and write down which months are peak and which are quiet.
# Do the peaks match what you would expect for a retail shop? Note one reason a raw count could mislead (hint: the data does not start in January).

# %% [markdown]
# ## Recap
# - Aggregates collapse rows; `COUNT(*)` counts rows, `COUNT(col)` skips NULLs, `COUNT(DISTINCT col)` counts unique values.
# - `GROUP BY` makes one row per group; every selected column must be grouped or aggregated (SQLite will not enforce it, others do).
# - `WHERE` filters rows before grouping, `HAVING` filters groups after.
# - Conditional aggregation (`SUM(CASE ...)`, `AVG(cond)`, `FILTER`) computes rates in one pass. Always show the denominator.
# - Do not average averages; aggregate sums and counts, then divide.
#
# **Go deeper:** the SQLite documentation on aggregate functions and `FILTER` (sqlite.org/lang_aggfunc.html); PostgreSQL docs, "Aggregate Functions".
#
# **Next:** [Joins](03_joins.ipynb)

# %% [markdown]
# # Window Functions
#
# **Why this matters.** "Rank each customer within their country", "month-over-month growth", "running total of revenue", "each order's share of the day's
# total". These are everyday analytics requests, and they all need to look at *neighbouring rows while keeping every row*. `GROUP BY` cannot do that: it collapses rows.
# Window functions can, and they are one of the most-asked topics in data scientist and analyst SQL interviews.
#
# **You will learn**
# - read the shape `function() OVER (PARTITION BY ... ORDER BY ... frame)`
# - rank rows with `ROW_NUMBER`, `RANK`, `DENSE_RANK` and understand tie behaviour
# - compare with previous/next rows using `LAG` and `LEAD`
# - compute running totals and moving averages, and avoid the default-frame trap
# - bucket rows with `NTILE` and compute share-of-total with `SUM() OVER ()`
#
# **Prerequisites:** [Subqueries and CTEs](04_subqueries_and_ctes.ipynb).
#
# > Data used: `datasets/shop.db`, read-only. Window functions need SQLite 3.25+ (we have far newer).

# %%
import sqlite3

import pandas as pd

con = sqlite3.connect("file:../datasets/shop.db?mode=ro", uri=True)


def q(sql, params=None):
    return pd.read_sql(sql, con, params=params)


print("SQLite version:", sqlite3.sqlite_version)

# %% [markdown]
# ## 1. The idea: aggregate without collapsing
#
# `GROUP BY category` gives one row per category. An aggregate with `OVER (PARTITION BY category)` computes the same number but **attaches it to every row**.

# %%
q("""
SELECT product_name, category, unit_price,
       ROUND(AVG(unit_price) OVER (PARTITION BY category), 2)                 AS category_avg,
       ROUND(unit_price - AVG(unit_price) OVER (PARTITION BY category), 2)    AS vs_category_avg
FROM products
ORDER BY category, unit_price DESC
LIMIT 6
""")

# %% [markdown]
# *What just happened?* No rows were lost: each product still has its own line, next to the average of its partition. Compare with the correlated subquery of notebook 04:
# same idea, one pass, no self-join. `OVER ()` with nothing inside means "the whole result is one partition", perfect for **share of total**:

# %%
q("""
SELECT category, COUNT(*) AS n_products,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_catalogue
FROM products GROUP BY category
""")

# %% [markdown]
# (Yes, a window function can wrap an aggregate: `GROUP BY` runs first, the window runs over the grouped rows.)
#
# **Anatomy:** `fn(...) OVER (PARTITION BY a ORDER BY b ROWS BETWEEN ... AND ...)`
# - `PARTITION BY`: restart the calculation for each group (optional).
# - `ORDER BY`: the order rows are processed in (needed for ranking, LAG, running totals).
# - the **frame**: which rows around the current one the function sees (only for aggregates and value functions).

# %% [markdown]
# ## 2. Ranking: ROW_NUMBER, RANK, DENSE_RANK
#
# Rank customers by number of orders. Many customers share the same order count, which is exactly where the three functions differ.

# %%
ranked = q("""
WITH cnt AS (SELECT customer_id, COUNT(*) AS n_orders FROM orders GROUP BY customer_id)
SELECT customer_id, n_orders,
       ROW_NUMBER() OVER (ORDER BY n_orders DESC, customer_id) AS row_number,
       RANK()       OVER (ORDER BY n_orders DESC)              AS rank,
       DENSE_RANK() OVER (ORDER BY n_orders DESC)              AS dense_rank
FROM cnt
ORDER BY n_orders DESC, customer_id
""")
ranked.iloc[[0, 1, 2, 3, 4, 5, 6]]

# %% [markdown]
# - `ROW_NUMBER`: always 1, 2, 3, ... ties broken arbitrarily unless you add a tiebreaker (we did: `customer_id`).
# - `RANK`: ties share a rank and then **skip** (1, 2, 2, 4).
# - `DENSE_RANK`: ties share a rank and **no gap** (1, 2, 2, 3).
#
# Pick by question: "exactly one row per group" -> `ROW_NUMBER`; "everyone tied for first" -> `RANK = 1`; "top 3 distinct values" -> `DENSE_RANK <= 3`.
# You cannot filter on a window function in `WHERE` (it is computed after `WHERE`), so wrap it in a CTE/subquery. Snowflake, BigQuery and DuckDB have
# `QUALIFY` for this; SQLite, Postgres and MySQL do not.

# %%
q("""
WITH r AS (
  SELECT product_name, category, unit_price,
         ROW_NUMBER() OVER (PARTITION BY category ORDER BY unit_price DESC) AS rn
  FROM products)
SELECT * FROM r WHERE rn <= 2 ORDER BY category, rn
""")

# %% [markdown]
# That is the **top-N per group** pattern: the most expensive two products in every category. (Snowflake spelling: `QUALIFY ROW_NUMBER() OVER (...) <= 2` with no CTE.)

# %% [markdown]
# ## 3. LAG and LEAD: look at the previous / next row
#
# Monthly order counts, then growth versus the previous month. `LAG(x, 1)` returns `x` from one row earlier in the window's order, `NULL` for the first row.

# %%
mom = q("""
WITH monthly AS (
  SELECT strftime('%Y-%m', order_date) AS month, COUNT(*) AS n_orders
  FROM orders GROUP BY month)
SELECT month, n_orders,
       LAG(n_orders) OVER (ORDER BY month)                                   AS prev_month,
       ROUND(100.0 * (n_orders - LAG(n_orders) OVER (ORDER BY month))
             / LAG(n_orders) OVER (ORDER BY month), 1)                       AS mom_pct
FROM monthly ORDER BY month
""")
mom.iloc[[0, 1, 5, 10, 11, 12, 13]]

# %% [markdown]
# The first month has no predecessor so `prev_month` is NULL (pandas shows `NaN`). Pass a default with `LAG(x, 1, 0)` if you want a number instead, but think
# first: a fake 0 makes growth infinite. `LEAD` is the mirror image (next row). A big December jump followed by a January drop is the seasonality in this shop.
# For "same month last year", use `LAG(x, 12)` **only if no months are missing**; otherwise self-join on a calendar (a gap silently shifts everything).

# %% [markdown]
# ## 4. Running totals, and the frame trap
#
# `SUM(x) OVER (ORDER BY t)` is a running total. The **frame** says which rows count. When you write `ORDER BY` but no frame, the default is
# `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, and `RANGE` treats rows with the same sort value (**peers**) as one unit. Watch:

# %%
q("""
SELECT order_id, order_date,
       COUNT(*) OVER (ORDER BY order_date)                                        AS default_frame,
       COUNT(*) OVER (ORDER BY order_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS rows_frame
FROM orders
ORDER BY order_date, order_id
LIMIT 8
""")

# %% [markdown]
# Several orders share a date, and the default frame gives all of them the *same* total (it includes every peer), while `ROWS` counts one row at a time.
# Neither is a bug; but if you expected a strict row-by-row total, **write `ROWS BETWEEN ... ` explicitly**. Cumulative *daily* revenue is safe either way because
# there is one row per day after grouping.

# %%
q("""
WITH monthly AS (
  SELECT strftime('%Y-%m', o.order_date) AS month, SUM(i.quantity * p.unit_price) AS revenue
  FROM orders o JOIN order_items i ON i.order_id = o.order_id JOIN products p ON p.product_id = i.product_id
  WHERE o.status <> 'cancelled'
  GROUP BY month)
SELECT month,
       ROUND(revenue, 0) AS revenue,
       ROUND(SUM(revenue) OVER (ORDER BY month ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 0) AS cumulative,
       ROUND(AVG(revenue) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 0)        AS moving_avg_3m
FROM monthly ORDER BY month LIMIT 6
""")

# %% [markdown]
# `ROWS BETWEEN 2 PRECEDING AND CURRENT ROW` is a 3-month **moving average** that smooths seasonal noise. The first two rows average fewer than 3 months (the frame is truncated at the start),
# so many analysts blank those out. Frames count *rows*, not calendar time: if a month is missing, "3 rows" spans 4 months.

# %% [markdown]
# ## 5. NTILE, FIRST_VALUE and friends
#
# `NTILE(4)` splits ordered rows into 4 equal-sized buckets (quartiles): the standard tool for "top 25% of customers" and RFM scoring (notebook 08).

# %%
q("""
WITH cr AS (
  SELECT o.customer_id, SUM(i.quantity * p.unit_price) AS revenue
  FROM orders o JOIN order_items i ON i.order_id = o.order_id JOIN products p ON p.product_id = i.product_id
  WHERE o.status <> 'cancelled' GROUP BY o.customer_id),
t AS (SELECT customer_id, revenue, NTILE(4) OVER (ORDER BY revenue DESC) AS quartile FROM cr)
SELECT quartile, COUNT(*) AS n_customers, ROUND(MIN(revenue)) AS min_rev, ROUND(MAX(revenue)) AS max_rev, ROUND(SUM(revenue)) AS total_rev
FROM t GROUP BY quartile ORDER BY quartile
""")

# %% [markdown]
# Buckets have the same **count**, not the same revenue: the top quartile carries far more of the total. That skew is worth quoting in a business summary.
# If the row count is not divisible by 4, the first buckets get one extra row. Ties can straddle a bucket boundary, and `NTILE` splits them arbitrarily.
#
# `FIRST_VALUE(x) OVER (PARTITION BY g ORDER BY t)` grabs the first row's value in each partition, for example each customer's first order date:

# %%
q("""
SELECT DISTINCT customer_id,
       FIRST_VALUE(order_date) OVER (PARTITION BY customer_id ORDER BY order_date, order_id) AS first_order_date
FROM orders ORDER BY customer_id LIMIT 4
""")

# %% [markdown]
# `MIN(order_date) ... GROUP BY customer_id` gives the same here and is simpler; `FIRST_VALUE` shines when you want *another column* from that first row (its channel, say).
# Note the `ORDER BY` inside `OVER` includes a tiebreaker `order_id`, since the default frame is `RANGE` again.

# %% [markdown]
# ## Common mistakes
#
# 1. **Filtering on a window function in `WHERE`.** Wrap in a CTE, or use `QUALIFY` where it exists.
# 2. **No tiebreaker in `ROW_NUMBER`.** Results change between runs, and "top 1 per group" may differ from what you expect.
# 3. **Running total without an explicit `ROWS` frame** when the order key has ties.
# 4. **`LAG(x, 12)` over data with missing periods.** Build a complete calendar first.
# 5. **Using `RANK` when you need one row per group.** Ties produce several rank-1 rows.
# 6. **Forgetting that windows run after `GROUP BY`/`HAVING`**, so `SUM(x) OVER ()` sees only the surviving groups.

# %%
# Mistake 1 in action
try:
    q("SELECT product_name FROM products WHERE ROW_NUMBER() OVER (ORDER BY unit_price) <= 3")
except Exception as e:
    print(type(e).__name__, "->", str(e)[:100])

# %% [markdown]
# ## Interview corner
#
# **1. Difference between <code>ROW_NUMBER</code>, <code>RANK</code> and <code>DENSE_RANK</code>?**
# <details><summary>Answer</summary>
# All number rows in order. <code>ROW_NUMBER</code> is unique; <code>RANK</code> gives ties the same number and skips the next ones (1,2,2,4); <code>DENSE_RANK</code> gives ties the same number without gaps (1,2,2,3).
# </details>
#
# **2. How do you get the top 3 products per category?**
# <details><summary>Answer</summary>
# <code>ROW_NUMBER() OVER (PARTITION BY category ORDER BY metric DESC, tiebreaker)</code> in a CTE, then filter <code>rn &lt;= 3</code> outside (or <code>QUALIFY</code> in Snowflake/BigQuery/DuckDB). Use <code>DENSE_RANK</code> if ties should all be kept.
# </details>
#
# **3. Window function versus GROUP BY?**
# <details><summary>Answer</summary>
# <code>GROUP BY</code> collapses rows to one per group. A window function keeps every row and adds a value computed over related rows, so you can compare each row with its group or neighbours.
# </details>
#
# **4. What does the default frame do when you add ORDER BY, and why does it matter?**
# <details><summary>Answer</summary>
# It is <code>RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW</code>: rows with equal sort keys are peers and share the same running value. Specify <code>ROWS BETWEEN ...</code> when you want a strict row-by-row accumulation.
# </details>

# %% [markdown]
# ## Exercises

# %%
customers = q("SELECT * FROM customers")
products = q("SELECT * FROM products")
orders = q("SELECT * FROM orders")
items = q("SELECT * FROM order_items")
line_rev = (items.merge(orders, on="order_id").merge(products, on="product_id")
            .query("status != 'cancelled'").assign(revenue=lambda d: d.quantity * d.unit_price, month=lambda d: d.order_date.str[:7]))

# %% [markdown]
# ### Exercise 1: The two cheapest products per category
# Return `category`, `product_name`, `unit_price`, `rn` (1 = cheapest) for the **two cheapest products in each category**, ordered by `category`, `rn`.
# *Hint: `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY unit_price, product_id)` in a CTE, then filter.*

# %% tags=["starter"]
ex1_sql = """
WITH ranked AS (
    SELECT ...
)
SELECT ...
"""

# %%
res = q(ex1_sql)
p = products.sort_values(["category", "unit_price", "product_id"]).assign(rn=lambda d: d.groupby("category").cumcount() + 1)
exp = p[p.rn <= 2][["category", "product_name", "unit_price", "rn"]].reset_index(drop=True)
assert list(res.columns) == ["category", "product_name", "unit_price", "rn"], f"columns: {list(res.columns)}"
assert len(res) == len(exp), f"expected {len(exp)} rows (2 per category)"
assert res.equals(exp), "rows differ from the expected two cheapest per category"
print("Exercise 1 passed")

# %% tags=["solution"]
ex1_sql = """
WITH ranked AS (
    SELECT category, product_name, unit_price,
           ROW_NUMBER() OVER (PARTITION BY category ORDER BY unit_price, product_id) AS rn
    FROM products
)
SELECT category, product_name, unit_price, rn
FROM ranked
WHERE rn <= 2
ORDER BY category, rn
"""

# %% [markdown]
# ### Exercise 2: Month-over-month order growth
# Return `month` (`YYYY-MM`), `n_orders`, `mom_pct` (percentage change vs the previous month, rounded to 1 decimal, `NULL` for the first month) for **all** months, ordered by month. Use `LAG`.

# %% tags=["starter"]
ex2_sql = """
SELECT ...
"""

# %%
res = q(ex2_sql)
m = orders.assign(month=orders.order_date.str[:7]).groupby("month").size().rename("n").reset_index()
m["pct"] = (m.n.pct_change() * 100).round(1)
assert list(res.columns) == ["month", "n_orders", "mom_pct"], f"columns: {list(res.columns)}"
assert res.month.tolist() == m.month.tolist() and res.n_orders.tolist() == m.n.tolist()
assert pd.isna(res.mom_pct[0]), "first month has no previous month: mom_pct should be NULL"
assert ((res.mom_pct[1:] - m.pct[1:]).abs() < 0.11).all(), "mom_pct differs: (n - previous) / previous * 100"
print(f"Exercise 2 passed; biggest jump {res.mom_pct.max()}% in {res.month[res.mom_pct.idxmax()]}")

# %% tags=["solution"]
ex2_sql = """
WITH monthly AS (
    SELECT strftime('%Y-%m', order_date) AS month, COUNT(*) AS n_orders
    FROM orders GROUP BY month
)
SELECT month, n_orders,
       ROUND(100.0 * (n_orders - LAG(n_orders) OVER (ORDER BY month)) / LAG(n_orders) OVER (ORDER BY month), 1) AS mom_pct
FROM monthly
ORDER BY month
"""

# %% [markdown]
# ### Exercise 3: Cumulative revenue with a 3-month average
# On non-cancelled orders, return `month`, `revenue`, `cumulative_revenue` and `moving_avg_3m`, all rounded to 2 decimals, one row per month ordered by month.
# Use explicit `ROWS` frames. `moving_avg_3m` averages the current and the two previous months (fewer at the start).

# %% tags=["starter"]
ex3_sql = """
SELECT ...
"""

# %%
res = q(ex3_sql)
m = line_rev.groupby("month").revenue.sum().reset_index()
m["cum"] = m.revenue.cumsum()
m["ma"] = m.revenue.rolling(3, min_periods=1).mean()
assert list(res.columns) == ["month", "revenue", "cumulative_revenue", "moving_avg_3m"], f"columns: {list(res.columns)}"
assert res.month.tolist() == m.month.tolist(), "one row per month"
for got, want in [("revenue", "revenue"), ("cumulative_revenue", "cum"), ("moving_avg_3m", "ma")]:
    assert ((res[got] - m[want]).abs() < 0.02).all(), f"{got} differs"
print("Exercise 3 passed")

# %% tags=["solution"]
ex3_sql = """
WITH monthly AS (
    SELECT strftime('%Y-%m', o.order_date) AS month, SUM(i.quantity * p.unit_price) AS revenue
    FROM orders o
    JOIN order_items i ON i.order_id = o.order_id
    JOIN products p ON p.product_id = i.product_id
    WHERE o.status <> 'cancelled'
    GROUP BY month
)
SELECT month,
       ROUND(revenue, 2) AS revenue,
       ROUND(SUM(revenue) OVER (ORDER BY month ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2) AS cumulative_revenue,
       ROUND(AVG(revenue) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS moving_avg_3m
FROM monthly
ORDER BY month
"""

# %% [markdown]
# ### Exercise 4: How concentrated is revenue?
# Split customers who ordered into **10 equal-size buckets** by non-cancelled revenue (`NTILE(10)`, decile 1 = highest spenders). Return `decile`, `n_customers`, and `revenue_share_pct`
# (that decile's share of total revenue, 1 decimal), ordered by decile. *Hint: `SUM(revenue) OVER ()` for the total, or a scalar subquery.*

# %% tags=["starter"]
ex4_sql = """
SELECT ...
"""

# %%
res = q(ex4_sql)
cr = line_rev.groupby("customer_id").revenue.sum().sort_values(ascending=False)
import numpy as np
parts = np.array_split(cr.values, 10)  # NTILE gives the larger buckets first, like array_split
exp_n = [len(x) for x in parts]
exp_share = [round(100 * x.sum() / cr.sum(), 1) for x in parts]
assert list(res.columns) == ["decile", "n_customers", "revenue_share_pct"], f"columns: {list(res.columns)}"
assert res.decile.tolist() == list(range(1, 11))
assert res.n_customers.tolist() == exp_n, f"bucket sizes should be {exp_n}"
assert ((res.revenue_share_pct - np.array(exp_share)).abs() < 0.11).all(), f"expected shares {exp_share}"
print(f"Exercise 4 passed: top decile earns {res.revenue_share_pct[0]}% of revenue, bottom decile {res.revenue_share_pct[9]}%")

# %% tags=["solution"]
ex4_sql = """
WITH cr AS (
    SELECT o.customer_id, SUM(i.quantity * p.unit_price) AS revenue
    FROM orders o
    JOIN order_items i ON i.order_id = o.order_id
    JOIN products p ON p.product_id = i.product_id
    WHERE o.status <> 'cancelled'
    GROUP BY o.customer_id
),
t AS (SELECT revenue, NTILE(10) OVER (ORDER BY revenue DESC) AS decile FROM cr)
SELECT decile, COUNT(*) AS n_customers,
       ROUND(100.0 * SUM(revenue) / (SELECT SUM(revenue) FROM cr), 1) AS revenue_share_pct
FROM t
GROUP BY decile
ORDER BY decile
"""

# %% [markdown]
# ### Exercise 5 (open-ended): Time between orders
# Use `LAG(order_date)` partitioned by customer to compute the days between consecutive orders (`julianday(a) - julianday(b)` gives days in SQLite).
# Report the median gap in a short markdown note. Is it shorter for premium customers? (You will need a join to `customers`.)

# %% [markdown]
# ## Recap
# - `fn() OVER (PARTITION BY ... ORDER BY ... frame)` computes across related rows **without collapsing them**.
# - `ROW_NUMBER` (unique), `RANK` (ties, gaps), `DENSE_RANK` (ties, no gaps); wrap in a CTE to filter (or `QUALIFY` on Snowflake/BigQuery/DuckDB).
# - `LAG`/`LEAD` for change vs previous/next; beware missing periods.
# - Running totals and moving averages need an explicit `ROWS` frame when the sort key can tie.
# - `NTILE(n)` gives equal-count buckets, and equal-count is not equal-revenue.
#
# **Go deeper:** SQLite docs on window functions (sqlite.org/windowfunctions.html); PostgreSQL tutorial, "Window Functions".
#
# **Next:** [DDL, DML and constraints](06_ddl_dml_and_constraints.ipynb)

# %% [markdown]
# # Subqueries and CTEs
#
# **Why this matters.** Real questions are layered: "customers whose spend is above the *average customer's* spend" needs an average
# first, and then a comparison. SQL handles layers with subqueries and **CTEs** (common table expressions, the `WITH` clause). CTEs
# turn a 60-line unreadable query into named steps a colleague can review, and they are the standard defence against join fan-out.
#
# **You will learn**
# - use a subquery as a value (scalar), as a list (`IN`), as an existence test (`EXISTS`) and as a table (in `FROM`)
# - write a **correlated** subquery and know when it is expensive
# - restructure nested queries into readable `WITH` steps (multiple CTEs)
# - generate rows with a **recursive CTE** (a calendar) to expose missing days
#
# **Prerequisites:** [Joins](03_joins.ipynb).
#
# > Data used: `datasets/shop.db`, read-only.

# %%
import sqlite3

import pandas as pd

con = sqlite3.connect("file:../datasets/shop.db?mode=ro", uri=True)


def q(sql, params=None):
    return pd.read_sql(sql, con, params=params)


# %% [markdown]
# ## 1. Subquery as a value
#
# A query in parentheses that returns **one value** can be used anywhere a number can. "Which products cost more than the average product?"

# %%
q("""
SELECT product_name, category, unit_price
FROM products
WHERE unit_price > (SELECT AVG(unit_price) FROM products)
ORDER BY unit_price DESC
LIMIT 5
""")

# %% [markdown]
# The inner query runs once and produces a single number (a *scalar subquery*). If it returned two rows or columns you would get an error
# (Postgres) or an arbitrary first value (SQLite); be sure it is one value.

# %% [markdown]
# ## 2. Subquery as a list: IN, and as a test: EXISTS
#
# `IN (subquery)` checks membership. `EXISTS (subquery)` asks "is there at least one matching row?" and stops at the first one.

# %%
q("""
SELECT COUNT(*) AS customers_with_a_return
FROM customers
WHERE customer_id IN (SELECT customer_id FROM orders WHERE status = 'returned')
""")

# %%
q("""
SELECT COUNT(*) AS customers_with_a_return
FROM customers c
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id AND o.status = 'returned')
""")

# %% [markdown]
# Same answer. `EXISTS` is a **semi-join**: it never multiplies rows (a plain `JOIN` to orders would repeat a customer once per returned order). Its negation
# `NOT EXISTS` is the safe anti-join. Notice the inner query refers to `c.customer_id` from the outer query: that makes it **correlated**.

# %% [markdown]
# ## 3. Correlated subqueries
#
# A correlated subquery is re-evaluated *for each outer row*, using that row's values. "Products priced above the average of **their own category**":

# %%
q("""
SELECT p.product_name, p.category, p.unit_price,
       (SELECT ROUND(AVG(x.unit_price), 2) FROM products x WHERE x.category = p.category) AS category_avg
FROM products p
WHERE p.unit_price > (SELECT AVG(x.unit_price) FROM products x WHERE x.category = p.category)
ORDER BY p.category, p.unit_price DESC
LIMIT 6
""")

# %% [markdown]
# It reads naturally, but with millions of rows "run a query per row" can be slow (optimisers sometimes rewrite it, not always). The same result with a
# **derived table** (a subquery in `FROM`) computes each category average once:

# %%
q("""
SELECT p.product_name, p.category, p.unit_price, ca.avg_price AS category_avg
FROM products p
JOIN (SELECT category, ROUND(AVG(unit_price), 2) AS avg_price FROM products GROUP BY category) ca
  ON ca.category = p.category
WHERE p.unit_price > ca.avg_price
ORDER BY p.category, p.unit_price DESC
LIMIT 6
""")

# %% [markdown]
# (Tiny differences are possible: this version compares to the **rounded** average. Rounding before comparing is a real-world source of off-by-one rows; round only for display.)
# Window functions (next notebook) solve this pattern more elegantly.

# %% [markdown]
# ## 4. CTEs: give each step a name
#
# `WITH name AS (query)` defines a temporary named result you can use like a table for the rest of the statement. Read it top to bottom like a recipe.
# Task: "Which customers spend more than the average *ordering* customer?"
#
# Step 1: revenue per customer. Step 2: the average of that. Step 3: compare.

# %%
q("""
WITH line_revenue AS (
    SELECT o.customer_id, i.quantity * p.unit_price AS revenue
    FROM orders o
    JOIN order_items i ON i.order_id = o.order_id
    JOIN products p    ON p.product_id = i.product_id
    WHERE o.status <> 'cancelled'
),
customer_revenue AS (
    SELECT customer_id, SUM(revenue) AS revenue
    FROM line_revenue
    GROUP BY customer_id
)
SELECT customer_id, ROUND(revenue, 2) AS revenue,
       ROUND((SELECT AVG(revenue) FROM customer_revenue), 2) AS avg_customer_revenue
FROM customer_revenue
WHERE revenue > (SELECT AVG(revenue) FROM customer_revenue)
ORDER BY revenue DESC
LIMIT 5
""")

# %% [markdown]
# *What just happened?* We built the answer in named layers. Each CTE is small enough to test alone: replace the final `SELECT` with `SELECT * FROM customer_revenue LIMIT 10` to inspect a step.
# That debugging move is the main reason to prefer CTEs over deeply nested subqueries.
#
# CTEs are not automatically faster: databases may inline them or (some versions, some engines) materialise them. Use them for clarity, and measure if speed matters (notebook 07).
#
# **CTEs and fan-out.** To join order-level and line-level facts safely, aggregate each to one row per key in separate CTEs, then join those:

# %%
q("""
WITH order_value AS (
    SELECT i.order_id, SUM(i.quantity * p.unit_price) AS value
    FROM order_items i JOIN products p ON p.product_id = i.product_id
    GROUP BY i.order_id
)
SELECT o.status, COUNT(*) AS n_orders, ROUND(AVG(v.value), 2) AS avg_value
FROM orders o
JOIN order_value v ON v.order_id = o.order_id      -- one row per order on both sides: no fan-out
GROUP BY o.status
ORDER BY avg_value DESC
""")

# %% [markdown]
# ## 5. Recursive CTE: manufacturing rows
#
# A recursive CTE has a starting row and a rule that produces the next row from the previous one, repeated until a condition stops it.
# A classic use is a **calendar**: real tables have missing days (no orders that day means no row), and a `GROUP BY` never shows a zero. Generate the days, then `LEFT JOIN`.

# %%
q("""
WITH RECURSIVE days(d) AS (
    SELECT date('2025-02-01')
    UNION ALL
    SELECT date(d, '+1 day') FROM days WHERE d < '2025-02-07'
)
SELECT days.d, COUNT(o.order_id) AS n_orders
FROM days LEFT JOIN orders o ON o.order_date = days.d
GROUP BY days.d
ORDER BY days.d
""")

# %% [markdown]
# Always include a stopping condition (`WHERE d < ...`); without one the query never ends. Recursive CTEs also walk hierarchies (manager -> reports, category trees).
# The syntax `WITH RECURSIVE` is the same in SQLite, Postgres and MySQL 8; some engines (SQL Server, Oracle) omit the word `RECURSIVE`.

# %% [markdown]
# ## Common mistakes
#
# 1. **A scalar subquery that returns several rows.** Aggregate it, or use `IN`.
# 2. **`NOT IN (subquery)` with NULLs** (notebook 03). Use `NOT EXISTS`.
# 3. **Forgetting the correlation** in an `EXISTS` (`WHERE 1 = 1` inside): it then answers the same for every row.
# 4. **Nesting 4 levels deep** instead of naming steps with CTEs. Nobody can review it.
# 5. **Assuming a CTE is computed once and cached** or, the reverse, that it is always inlined. Behaviour is engine-specific.
# 6. **Recursive CTE with no stop condition.**

# %%
# Mistake 3: an EXISTS without the correlation says "yes" for every customer as soon as ANY order was returned.
q("""
SELECT COUNT(*) AS customers_counted
FROM customers c
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.status = 'returned')      -- missing: AND o.customer_id = c.customer_id
""")

# %% [markdown]
# That is the number of *all* customers, not the number who returned something.

# %% [markdown]
# ## Interview corner
#
# **1. Subquery versus CTE: when do you choose which?**
# <details><summary>Answer</summary>
# Semantically they overlap. A CTE names a step, can be referenced several times, and reads top-down, so use it once logic has more than one layer.
# A short scalar subquery in <code>WHERE</code> is fine inline. Performance depends on the engine and optimiser, not on the syntax.
# </details>
#
# **2. What is a correlated subquery and why can it be slow?**
# <details><summary>Answer</summary>
# It references a column of the outer query so it is logically re-run per outer row (O(n x m) if the optimiser cannot rewrite it). Often you can replace it with a join to a pre-aggregated derived table or with a window function.
# </details>
#
# **3. <code>IN</code> versus <code>EXISTS</code>?**
# <details><summary>Answer</summary>
# Both express a semi-join and modern optimisers usually plan them alike. <code>EXISTS</code> short-circuits and its negation is safe with NULLs, while <code>NOT IN</code> is not.
# </details>
#
# **4. Write a query to show days with zero orders.**
# <details><summary>Answer</summary>
# Generate a calendar (recursive CTE or a date table), <code>LEFT JOIN</code> the fact table on the date and <code>COUNT(fact.key)</code>, so days without matches produce 0.
# </details>

# %% [markdown]
# ## Exercises

# %%
customers = q("SELECT * FROM customers")
products = q("SELECT * FROM products")
orders = q("SELECT * FROM orders")
items = q("SELECT * FROM order_items")

# %% [markdown]
# ### Exercise 1: Above-average spenders
# Using CTEs, return `customer_id` and `revenue` (rounded to 2 decimals) for customers whose total revenue (non-cancelled orders, `quantity * unit_price`)
# is **above the average customer revenue among customers who ordered**. Sort by revenue descending, then `customer_id`.

# %% tags=["starter"]
ex1_sql = """
WITH ...
SELECT ...
"""

# %%
res = q(ex1_sql)
m = items.merge(orders, on="order_id").merge(products, on="product_id")
m = m[m.status != "cancelled"]
rev = (m.quantity * m.unit_price).groupby(m.customer_id).sum()
exp = rev[rev > rev.mean()].round(2).reset_index(name="revenue").sort_values(["revenue", "customer_id"], ascending=[False, True]).reset_index(drop=True)
assert list(res.columns) == ["customer_id", "revenue"], f"columns: {list(res.columns)}"
assert len(res) == len(exp), f"expected {len(exp)} customers, got {len(res)}"
assert res.customer_id.tolist() == exp.customer_id.tolist(), "customers/order differ"
assert ((res.revenue - exp.revenue).abs() < 0.02).all()
print(f"Exercise 1 passed ({len(res)} above-average customers)")

# %% tags=["solution"]
ex1_sql = """
WITH customer_revenue AS (
    SELECT o.customer_id, SUM(i.quantity * p.unit_price) AS revenue
    FROM orders o
    JOIN order_items i ON i.order_id = o.order_id
    JOIN products p    ON p.product_id = i.product_id
    WHERE o.status <> 'cancelled'
    GROUP BY o.customer_id
)
SELECT customer_id, ROUND(revenue, 2) AS revenue
FROM customer_revenue
WHERE revenue > (SELECT AVG(revenue) FROM customer_revenue)
ORDER BY revenue DESC, customer_id
"""

# %% [markdown]
# ### Exercise 2: Price outliers within a category
# Return `product_name`, `category`, `unit_price` for products whose price is **more than 1.25 times the average price of their own category**. Sort by `category`, `unit_price` descending.
# Use a correlated subquery **or** a derived table. Do not round before comparing.

# %% tags=["starter"]
ex2_sql = """
SELECT ...
"""

# %%
res = q(ex2_sql)
avg = products.groupby("category").unit_price.transform("mean")
exp = products[products.unit_price > 1.25 * avg].sort_values(["category", "unit_price"], ascending=[True, False])[["product_name", "category", "unit_price"]].reset_index(drop=True)
assert len(exp) > 0, "sanity"
assert list(res.columns) == ["product_name", "category", "unit_price"], f"columns: {list(res.columns)}"
assert res.equals(exp), f"expected {len(exp)} products: {exp.product_name.tolist()}"
print(f"Exercise 2 passed ({len(res)} products)")

# %% tags=["solution"]
ex2_sql = """
SELECT p.product_name, p.category, p.unit_price
FROM products p
WHERE p.unit_price > 1.25 * (SELECT AVG(x.unit_price) FROM products x WHERE x.category = p.category)
ORDER BY p.category, p.unit_price DESC
"""

# %% [markdown]
# ### Exercise 3: Returners who buy electronics (EXISTS)
# Count customers who (a) have at least one order with status `returned` **and** (b) have bought at least one product of category `Electronics` in any order.
# Return one row with one column `n_customers`. Use two `EXISTS` conditions so no customer is counted twice.

# %% tags=["starter"]
ex3_sql = """
SELECT ...
"""

# %%
res = q(ex3_sql)
returned = set(orders.loc[orders.status == "returned", "customer_id"])
elec_ids = set(products.loc[products.category == "Electronics", "product_id"])
elec_orders = set(items.loc[items.product_id.isin(elec_ids), "order_id"])
elec_customers = set(orders.loc[orders.order_id.isin(elec_orders), "customer_id"])
exp = len(returned & elec_customers)
assert exp > 0, "sanity"
assert list(res.columns) == ["n_customers"] and len(res) == 1
assert res.n_customers[0] == exp, f"expected {exp}, got {res.n_customers[0]}"
print("Exercise 3 passed:", exp)

# %% tags=["solution"]
ex3_sql = """
SELECT COUNT(*) AS n_customers
FROM customers c
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id AND o.status = 'returned')
  AND EXISTS (SELECT 1
              FROM orders o
              JOIN order_items i ON i.order_id = o.order_id
              JOIN products p    ON p.product_id = i.product_id
              WHERE o.customer_id = c.customer_id AND p.category = 'Electronics')
"""

# %% [markdown]
# ### Exercise 4: A complete calendar
# Build a daily table for **1 to 28 February 2024** with a recursive CTE and return `day` (text `YYYY-MM-DD`) and `n_orders` (0 if none), one row per day, ordered by day.
# (2024 is a leap year; stop at the 28th anyway.) *Hint: `LEFT JOIN` orders and `COUNT(o.order_id)`.*

# %% tags=["starter"]
ex4_sql = """
WITH RECURSIVE days(day) AS (
    ...
)
SELECT ...
"""

# %%
res = q(ex4_sql)
idx = pd.date_range("2024-02-01", "2024-02-28").strftime("%Y-%m-%d")
cnt = orders.groupby("order_date").size().reindex(idx, fill_value=0)
assert list(res.columns) == ["day", "n_orders"], f"columns: {list(res.columns)}"
assert res.day.tolist() == list(idx), "need exactly 28 consecutive days"
assert res.n_orders.tolist() == cnt.tolist(), "n_orders differ (COUNT(o.order_id) on a LEFT JOIN)"
print(f"Exercise 4 passed ({(res.n_orders == 0).sum()} days with no orders)")

# %% tags=["solution"]
ex4_sql = """
WITH RECURSIVE days(day) AS (
    SELECT date('2024-02-01')
    UNION ALL
    SELECT date(day, '+1 day') FROM days WHERE day < '2024-02-28'
)
SELECT days.day, COUNT(o.order_id) AS n_orders
FROM days
LEFT JOIN orders o ON o.order_date = days.day
GROUP BY days.day
ORDER BY days.day
"""

# %% [markdown]
# ### Exercise 5 (open-ended): Refactor a nested query
# Take the derived-table query from section 3 and rewrite it as a CTE. Then take the "above-average spenders" solution and add one more CTE that labels each
# customer's country. In markdown, say which version you find easier to read and why.

# %% [markdown]
# ## Recap
# - A subquery can be a value, a list (`IN`), an existence test (`EXISTS`) or a table (in `FROM`).
# - Correlated subqueries run per outer row: readable, sometimes slow; a pre-aggregated join or window function is the alternative.
# - `WITH` CTEs name each step: build layers, test each alone, and pre-aggregate before joining to avoid fan-out.
# - Recursive CTEs generate rows (calendars, sequences, hierarchies); always define the stop condition.
#
# **Go deeper:** SQLite docs on `WITH` (sqlite.org/lang_with.html); PostgreSQL docs, "WITH Queries (Common Table Expressions)".
#
# **Next:** [Window functions](05_window_functions.ipynb)

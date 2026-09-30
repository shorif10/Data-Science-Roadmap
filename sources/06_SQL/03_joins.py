# %% [markdown]
# # Joins
#
# **Why this matters.** Real data is split across tables on purpose: a customer is stored once, their orders many times.
# Almost every business question ("revenue by product category", "customers who never bought") needs you to stitch tables back
# together. Joins are also where the most expensive silent bugs live: a join that multiplies rows inflates revenue and nobody notices.
#
# **You will learn**
# - combine tables with `INNER JOIN` and `LEFT JOIN`, and chain three or more tables
# - find "things with no match" using an **anti-join** (and why `NOT IN` is dangerous)
# - join a table to itself (self-join)
# - recognise and prevent **fan-out** by checking row counts before and after every join
#
# **Prerequisites:** [Aggregation and GROUP BY](02_aggregation_group_by_having.ipynb).
#
# > Data used: `datasets/shop.db`. Relationships: `customers 1---* orders 1---* order_items *---1 products`.

# %%
import sqlite3

import pandas as pd

con = sqlite3.connect("file:../datasets/shop.db?mode=ro", uri=True)


def q(sql, params=None):
    return pd.read_sql(sql, con, params=params)


# %% [markdown]
# ## 1. What a join does
#
# Picture two lists on a desk. For every row on the left, you look for rows on the right whose key matches, and glue them together.
# The **key** here is `orders.customer_id = customers.customer_id`: the *foreign key* on the many side points to the *primary key* on the one side.
#
# Before joining, **know the grain** of each table (what one row means): `orders` is one row per order, `order_items` one row per product line on an order.

# %%
q("""
SELECT o.order_id, o.order_date, c.first_name, c.country
FROM orders AS o
INNER JOIN customers AS c ON c.customer_id = o.customer_id
ORDER BY o.order_id
LIMIT 5
""")

# %% [markdown]
# `AS o` gives the table a short alias; prefix columns with it (`o.order_id`) so the reader and the database know which table you mean
# (and so columns with the same name in both tables do not clash). **Inner join** keeps only rows that have a match on *both* sides.

# %% [markdown]
# ## 2. Row-count discipline
#
# The single best habit: before and after a join, count rows and compare with what you expect. Joining the "many" side to the "one" side
# should keep the many-side count. Let us make that a helper.

# %%
def n(sql):
    return int(q(f"SELECT COUNT(*) AS n FROM ({sql})").n[0])


n_orders = n("SELECT * FROM orders")
n_join = n("SELECT o.* FROM orders o JOIN customers c ON c.customer_id = o.customer_id")
print(f"orders: {n_orders}   orders JOIN customers: {n_join}   (same => customer_id is unique in customers)")

# %% [markdown]
# Same count, so each order matched exactly one customer. Now go one level down: an order has *several* item lines.

# %%
n_lines = n("SELECT o.* FROM orders o JOIN order_items i ON i.order_id = o.order_id")
print(f"orders: {n_orders}   orders JOIN order_items: {n_lines}")

# %% [markdown]
# More rows than orders: an order with 3 lines appears 3 times. This is **fan-out**. That is correct if you want line-level analysis and a bug if you then
# count orders or sum an order-level amount.

# %%
q("""
SELECT o.channel,
       COUNT(*)                    AS wrongly_called_orders,   -- counts lines
       COUNT(DISTINCT o.order_id)  AS real_orders
FROM orders o JOIN order_items i ON i.order_id = o.order_id
GROUP BY o.channel
""")

# %% [markdown]
# *What just happened?* After a one-to-many join, `COUNT(*)` counts the **many** side. The fix is either `COUNT(DISTINCT order_id)`, or
# **aggregate the many side to one row per key first**, then join (the safest habit; you will see it with CTEs in notebook 04).

# %%
q("""
SELECT o.channel, ROUND(AVG(t.n_lines), 3) AS avg_lines_per_order
FROM orders o
JOIN (SELECT order_id, COUNT(*) AS n_lines FROM order_items GROUP BY order_id) t ON t.order_id = o.order_id
GROUP BY o.channel
""")

# %% [markdown]
# ## 3. Chaining joins: revenue by category
#
# Revenue of a line is `quantity * unit_price`. That needs `order_items` (quantity) and `products` (price); the category is also in `products`.
# (The data has one current price per product, not the price at the time of sale, which real systems usually store on the line. Note this limitation whenever
# you compute historical revenue.)

# %%
q("""
SELECT p.category,
       SUM(i.quantity)                             AS units,
       ROUND(SUM(i.quantity * p.unit_price), 2)    AS revenue
FROM order_items i
JOIN products p ON p.product_id = i.product_id
JOIN orders   o ON o.order_id   = i.order_id
WHERE o.status <> 'cancelled'
GROUP BY p.category
ORDER BY revenue DESC
""")

# %% [markdown]
# Each extra join follows the same rule: know the grain, know which side is "one". Here `items -> products` is many-to-one (safe) and `items -> orders` is many-to-one (safe),
# so the grain stays "one row per item line" and the sum is right.

# %% [markdown]
# ## 4. LEFT JOIN: keep everything from the left table
#
# `LEFT JOIN` keeps **every** left row; where there is no match, the right-side columns are `NULL`. "How many orders does each customer have, including those with none?"

# %%
q("""
SELECT c.customer_id, c.first_name, COUNT(o.order_id) AS n_orders
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id
GROUP BY c.customer_id
ORDER BY n_orders, c.customer_id
LIMIT 5
""")

# %% [markdown]
# Note `COUNT(o.order_id)`, **not** `COUNT(*)`: for a customer with no orders the left join produces one row of NULLs, `COUNT(*)` would say 1, but
# `COUNT(o.order_id)` correctly says 0.
#
# A classic trap: a filter on the right table in `WHERE` turns a `LEFT JOIN` back into an `INNER JOIN`, because `NULL` fails the condition.

# %%
wrong = q("""
SELECT COUNT(DISTINCT c.customer_id) AS customers_listed
FROM customers c LEFT JOIN orders o ON o.customer_id = c.customer_id
WHERE o.status = 'delivered'
""").customers_listed[0]
right = q("""
SELECT COUNT(DISTINCT c.customer_id) AS customers_listed
FROM customers c LEFT JOIN orders o ON o.customer_id = c.customer_id AND o.status = 'delivered'
""").customers_listed[0]
print(f"filter in WHERE: {wrong} customers (never-delivered customers vanished)   filter in ON: {right} customers (all kept)")

# %% [markdown]
# Rule of thumb: conditions that describe **which right rows may match** go in `ON`; conditions that filter the final result go in `WHERE`.

# %% [markdown]
# ## 5. Anti-join: rows with no match
#
# "Which customers never ordered?" is `LEFT JOIN` + `WHERE right_key IS NULL`. Two other spellings do the same job:

# %%
a = q("SELECT c.customer_id FROM customers c LEFT JOIN orders o ON o.customer_id = c.customer_id WHERE o.order_id IS NULL")
b = q("SELECT c.customer_id FROM customers c WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id)")
c_ = q("SELECT customer_id FROM customers WHERE customer_id NOT IN (SELECT customer_id FROM orders)")
print(len(a), len(b), len(c_), "| same customers:", set(a.customer_id) == set(b.customer_id) == set(c_.customer_id))

# %% [markdown]
# All agree **here**. But `NOT IN` has a trap: if the subquery returns even one `NULL`, `x NOT IN (...)` is never true and you get **zero rows**, with no error.

# %%
q("""
WITH banned(id) AS (VALUES (1), (NULL))
SELECT COUNT(*) AS rows_returned FROM customers WHERE customer_id NOT IN (SELECT id FROM banned)
""")

# %% [markdown]
# Prefer `NOT EXISTS` or `LEFT JOIN ... IS NULL`; they behave sensibly with NULLs.

# %% [markdown]
# ## 6. Self-join: a table joined to itself
#
# Give the same table two aliases and relate rows to each other. Example: pairs of products in the **same category** whose prices are within 5 of each other
# (useful for "similar item" suggestions). `a.product_id < b.product_id` keeps each pair once and avoids pairing a product with itself.

# %%
q("""
SELECT a.product_name AS product_a, b.product_name AS product_b, a.category, a.unit_price, b.unit_price
FROM products a
JOIN products b ON a.category = b.category AND a.product_id < b.product_id
WHERE ABS(a.unit_price - b.unit_price) < 5
ORDER BY ABS(a.unit_price - b.unit_price)
LIMIT 5
""")

# %% [markdown]
# Other joins exist: `CROSS JOIN` (every combination, rarely wanted), `RIGHT`/`FULL OUTER JOIN` (SQLite supports them from 3.39; older SQLite and some engines do not, and most people
# just swap table order and use `LEFT JOIN`).

# %% [markdown]
# ## Common mistakes
#
# 1. **Not checking row counts** after a join (fan-out): totals silently inflate.
# 2. **`COUNT(*)` after a `LEFT JOIN`** counts the NULL row too. Count a right-side key instead.
# 3. **Right-table filter in `WHERE` after `LEFT JOIN`** (shown above).
# 4. **`NOT IN` with NULLs** (shown above).
# 5. **`SELECT DISTINCT` to "fix" duplicates** created by a bad join: it hides the problem and may still double count sums.
# 6. **Joining on a non-unique key** you assumed was unique. Test with `SELECT key, COUNT(*) ... HAVING COUNT(*) > 1` first.

# %%
# Mistake 6: is product_id really unique in products? Verify instead of assuming.
q("SELECT COUNT(*) AS duplicate_keys FROM (SELECT product_id FROM products GROUP BY product_id HAVING COUNT(*) > 1)")

# %% [markdown]
# ## Interview corner
#
# **1. Difference between <code>INNER JOIN</code> and <code>LEFT JOIN</code>?**
# <details><summary>Answer</summary>
# Inner keeps only matched rows; left keeps all left rows and fills right columns with NULL when nothing matches. Use left when the left side defines the population you report on.
# </details>
#
# **2. Your revenue total doubled after adding a join. What happened and how do you find it?**
# <details><summary>Answer</summary>
# Fan-out: the joined table has several rows per key, so rows on the other side were repeated. Compare row counts before/after, check key uniqueness on the "one" side
# (<code>GROUP BY key HAVING COUNT(*) &gt; 1</code>), and pre-aggregate the many side to one row per key before joining.
# </details>
#
# **3. How do you find customers with no orders? Which way is safest?**
# <details><summary>Answer</summary>
# <code>LEFT JOIN ... WHERE o.id IS NULL</code> or <code>NOT EXISTS</code>. <code>NOT IN</code> returns nothing if the subquery yields any NULL.
# </details>
#
# **4. Where do you put a filter on the right table of a LEFT JOIN?**
# <details><summary>Answer</summary>
# In the <code>ON</code> clause if you want to keep unmatched left rows; in <code>WHERE</code> it removes them (NULL fails the test) and effectively converts the join to an inner join.
# </details>

# %% [markdown]
# ## Exercises

# %%
customers = q("SELECT * FROM customers")
products = q("SELECT * FROM products")
orders = q("SELECT * FROM orders")
items = q("SELECT * FROM order_items")

# %% [markdown]
# ### Exercise 1: Revenue by country
# Revenue = `quantity * unit_price` on non-cancelled orders. Return `country`, `revenue` (rounded to 2 decimals) for every customer country, highest first.
# You need four tables: `customers`, `orders`, `order_items`, `products`.

# %% tags=["starter"]
ex1_sql = """
SELECT ...
"""

# %%
res = q(ex1_sql)
m = (items.merge(orders, on="order_id").merge(customers, on="customer_id").merge(products, on="product_id"))
m = m[m.status != "cancelled"]
exp = (m.assign(rev=m.quantity * m.unit_price).groupby("country").rev.sum().round(2).sort_values(ascending=False))
assert list(res.columns) == ["country", "revenue"], f"columns: {list(res.columns)}"
assert res.country.tolist() == exp.index.tolist(), "country order differs"
assert ((res.revenue - exp.values).abs() < 0.02).all(), "revenue values differ: did you exclude cancelled orders and multiply quantity by unit_price?"
print("Exercise 1 passed; top country:", res.country[0])

# %% tags=["solution"]
ex1_sql = """
SELECT c.country, ROUND(SUM(i.quantity * p.unit_price), 2) AS revenue
FROM order_items i
JOIN orders o    ON o.order_id = i.order_id
JOIN customers c ON c.customer_id = o.customer_id
JOIN products p  ON p.product_id = i.product_id
WHERE o.status <> 'cancelled'
GROUP BY c.country
ORDER BY revenue DESC
"""

# %% [markdown]
# ### Exercise 2: Dormant customers by country
# For each country return `country` and `n_never_ordered` (customers with **no orders at all**), keeping only countries where that number is at least 1, sorted by `n_never_ordered` descending then `country`.
# Use an anti-join (`LEFT JOIN ... IS NULL` or `NOT EXISTS`).

# %% tags=["starter"]
ex2_sql = """
SELECT ...
"""

# %%
res = q(ex2_sql)
never = customers[~customers.customer_id.isin(orders.customer_id)]
exp = never.groupby("country").size().rename("n").reset_index().sort_values(["n", "country"], ascending=[False, True]).reset_index(drop=True)
assert len(exp) > 0
assert list(res.columns) == ["country", "n_never_ordered"], f"columns: {list(res.columns)}"
assert res.country.tolist() == exp.country.tolist() and res.n_never_ordered.tolist() == exp.n.tolist(), "counts or order differ"
print(f"Exercise 2 passed: {res.n_never_ordered.sum()} dormant customers in total")

# %% tags=["solution"]
ex2_sql = """
SELECT c.country, COUNT(*) AS n_never_ordered
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id
WHERE o.order_id IS NULL
GROUP BY c.country
ORDER BY n_never_ordered DESC, c.country
"""

# %% [markdown]
# ### Exercise 3: Beat the fan-out
# Per `channel`, return `channel`, `n_orders` (real distinct orders, non-cancelled) and `avg_order_value` (total revenue of those orders divided by `n_orders`, rounded to 2 decimals).
# Sort by `avg_order_value` descending. *Hint: `COUNT(DISTINCT o.order_id)` after joining lines and products, or pre-aggregate each order's value first.*

# %% tags=["starter"]
ex3_sql = """
SELECT ...
"""

# %%
res = q(ex3_sql)
m = items.merge(orders, on="order_id").merge(products, on="product_id")
m = m[m.status != "cancelled"].assign(rev=lambda x: x.quantity * x.unit_price)
per_order = m.groupby(["channel", "order_id"]).rev.sum().reset_index()
exp = per_order.groupby("channel").agg(n=("order_id", "size"), aov=("rev", "mean")).sort_values("aov", ascending=False)
assert list(res.columns) == ["channel", "n_orders", "avg_order_value"], f"columns: {list(res.columns)}"
assert res.n_orders.tolist() == exp.n.tolist(), f"n_orders should count distinct orders, expected {exp.n.tolist()}"
assert res.channel.tolist() == exp.index.tolist()
assert ((res.avg_order_value - exp.aov.round(2).values).abs() < 0.02).all(), "avg_order_value differs"
print("Exercise 3 passed")

# %% tags=["solution"]
ex3_sql = """
SELECT o.channel,
       COUNT(DISTINCT o.order_id) AS n_orders,
       ROUND(SUM(i.quantity * p.unit_price) / COUNT(DISTINCT o.order_id), 2) AS avg_order_value
FROM orders o
JOIN order_items i ON i.order_id = o.order_id
JOIN products p    ON p.product_id = i.product_id
WHERE o.status <> 'cancelled'
GROUP BY o.channel
ORDER BY avg_order_value DESC
"""

# %% [markdown]
# ### Exercise 4: Substitutes (self-join)
# List every pair of products in the **same category** whose `unit_price` differs by **less than 10**. Return `product_a`, `product_b` (product names, with `product_id` of a smaller than b) and `price_gap` (absolute difference, 2 decimals).
# Sort by `price_gap`, then `product_a`, `product_b`.

# %% tags=["starter"]
ex4_sql = """
SELECT ...
"""

# %%
res = q(ex4_sql)
x = products.merge(products, on="category", suffixes=("_a", "_b"))
x = x[(x.product_id_a < x.product_id_b) & ((x.unit_price_a - x.unit_price_b).abs() < 10)]
x = x.assign(gap=(x.unit_price_a - x.unit_price_b).abs().round(2)).sort_values(["gap", "product_name_a", "product_name_b"]).reset_index(drop=True)
assert len(x) > 0, "sanity"
assert list(res.columns) == ["product_a", "product_b", "price_gap"], f"columns: {list(res.columns)}"
assert len(res) == len(x), f"expected {len(x)} pairs, got {len(res)} (each pair once, never a product with itself)"
assert res.product_a.tolist() == x.product_name_a.tolist() and res.product_b.tolist() == x.product_name_b.tolist()
print(f"Exercise 4 passed ({len(res)} pairs)")

# %% tags=["solution"]
ex4_sql = """
SELECT a.product_name AS product_a, b.product_name AS product_b,
       ROUND(ABS(a.unit_price - b.unit_price), 2) AS price_gap
FROM products a
JOIN products b ON a.category = b.category AND a.product_id < b.product_id
WHERE ABS(a.unit_price - b.unit_price) < 10
ORDER BY price_gap, product_a, product_b
"""

# %% [markdown]
# ### Exercise 5 (open-ended): Prove a fan-out
# Pick any order-level number (for example "orders per customer") and compute it two ways: correctly, and after joining to `order_items` without protecting against fan-out.
# Report how far off the wrong version is (percentage) and which of the fixes from this notebook you would use.

# %% [markdown]
# ## Recap
# - Know the **grain** of each table; join many-side to one-side keys and keep a row-count check next to every join.
# - `INNER` keeps matches only; `LEFT` keeps all left rows (right columns NULL when unmatched); right-table filters go in `ON`, not `WHERE`.
# - Anti-join = `LEFT JOIN ... IS NULL` or `NOT EXISTS`; avoid `NOT IN` when NULLs are possible.
# - Fan-out inflates counts and sums: use `COUNT(DISTINCT key)` or pre-aggregate the many side.
# - Self-joins use two aliases and an inequality (`a.id < b.id`) to make unique pairs.
#
# **Go deeper:** SQLite docs on joins (sqlite.org/lang_select.html); PostgreSQL docs, "Joins Between Tables" in the tutorial.
#
# **Next:** [Subqueries and CTEs](04_subqueries_and_ctes.ipynb)

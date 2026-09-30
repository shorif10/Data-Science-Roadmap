# %% [markdown]
# # SELECT, WHERE, ORDER BY, LIMIT
#
# **Why this matters.** Most company data lives in a database, not in a CSV on your laptop. The first
# thing a data scientist does on a new job is ask a table a question: "which customers signed up last
# quarter?", "what are our ten most expensive products?". SQL is how you ask, and the same four clauses
# you learn here work in SQLite, Postgres, MySQL, Snowflake and BigQuery almost unchanged.
#
# **You will learn**
# - choose columns, compute new ones and rename them with `SELECT ... AS`
# - filter rows with `WHERE` using comparison, `IN`, `BETWEEN`, `LIKE` and `AND`/`OR`/`NOT`
# - handle `NULL` correctly (it is not a value, it is "unknown")
# - sort with `ORDER BY`, take the top of a result with `LIMIT`/`OFFSET`, and remove repeats with `DISTINCT`
# - branch with `CASE WHEN` and know the logical order in which SQL evaluates a query
#
# **Prerequisites:** [Loading and inspecting data with pandas](../05_Pandas/01_loading_and_inspecting_data.ipynb) helps
# (we use `read_sql` to see results as DataFrames), but no SQL knowledge is assumed.
#
# > Data used: `datasets/shop.db`, a synthetic SQLite database of an online shop with four tables:
# > `customers`, `products`, `orders`, `order_items`. See `docs/DATASETS.md`. We open it **read-only**,
# > so nothing you type here can damage it.

# %%
import sqlite3

import pandas as pd

# mode=ro: read-only. Relative path: notebooks run with their module folder as the working directory.
con = sqlite3.connect("file:../datasets/shop.db?mode=ro", uri=True)


def q(sql, params=None):
    """Run a SQL query and return the result as a DataFrame."""
    return pd.read_sql(sql, con, params=params)


q("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")

# %% [markdown]
# `sqlite_master` is SQLite's own catalogue of what exists. (In Postgres/Snowflake you would query
# `information_schema.tables` instead.) Our helper `q()` just sends text to the database and wraps the
# answer in a DataFrame; SQL itself is the text.

# %% [markdown]
# ## 1. SELECT: choosing columns
#
# Think of a table as a spreadsheet. `SELECT` says which columns you want in the answer, `FROM` says which
# table to read. `*` means "all columns" and is fine for a peek, but in real code and views name your columns
# so a schema change does not silently change your output.

# %%
q("SELECT * FROM customers LIMIT 5")

# %%
q("SELECT first_name, country, age FROM customers LIMIT 5")

# %% [markdown]
# You can compute columns and rename them with `AS`. Below we compute the gross margin per unit of each product.

# %%
q("""
SELECT product_name,
       unit_price,
       unit_cost,
       unit_price - unit_cost                          AS margin,
       ROUND(100.0 * (unit_price - unit_cost) / unit_price, 1) AS margin_pct
FROM products
LIMIT 5
""")

# %% [markdown]
# *What just happened?* `SELECT` evaluates an expression per row. Note `100.0`, not `100`: in SQLite (and Postgres)
# dividing two integers does **integer division** (`7/2 = 3`). Forcing one side to a decimal avoids that trap.

# %%
q("SELECT 7 / 2 AS int_division, 7 / 2.0 AS real_division, 7 % 2 AS remainder")

# %% [markdown]
# ## 2. WHERE: choosing rows
#
# `WHERE` keeps only the rows for which the condition is true. Everything else is dropped **before** anything
# is displayed.
#
# | You want | Write |
# |---|---|
# | equality / comparison | `country = 'US'`, `age >= 65`, `country <> 'US'` |
# | several values | `country IN ('US', 'CA')` |
# | a range (both ends **included**) | `age BETWEEN 18 AND 25` |
# | text pattern | `first_name LIKE 'A%'` (`%` = any characters, `_` = exactly one) |
# | combine | `AND`, `OR`, `NOT`; use parentheses when mixing |
#
# String literals use **single** quotes. Double quotes mean "an identifier" (a column or table name).

# %%
q("""
SELECT customer_id, first_name, country, age, is_premium
FROM customers
WHERE country IN ('US', 'CA')
  AND age BETWEEN 18 AND 25
  AND is_premium = 1
""")

# %% [markdown]
# `AND` binds tighter than `OR`, exactly like `*` before `+`. This bites people:

# %%
a = q("SELECT COUNT(*) AS n FROM customers WHERE country = 'US' OR country = 'CA' AND age < 25").n[0]
b = q("SELECT COUNT(*) AS n FROM customers WHERE (country = 'US' OR country = 'CA') AND age < 25").n[0]
print(f"without parentheses: {a}   with parentheses: {b}")

# %% [markdown]
# The first query means "all US customers, plus Canadians under 25". The second means "Americans or Canadians who are under 25".
# Same words, different meaning. When you mix `AND`/`OR`, always parenthesise.
#
# `LIKE` is for simple patterns. In SQLite it is case-insensitive for ASCII letters; in Postgres `LIKE` is
# case-sensitive (use `ILIKE`). Check your dialect.

# %%
q("SELECT product_name, category, unit_price FROM products WHERE product_name LIKE 'Headphones%'")

# %% [markdown]
# ## 3. NULL: the value that is not a value
#
# `NULL` means *unknown or missing*. Any comparison with an unknown is unknown, so `NULL = NULL` is not true.
# You must use `IS NULL` / `IS NOT NULL`. Our shop tables happen to have no `NULL`s, so here is the behaviour
# on a tiny inline table you can see completely:

# %%
q("""
SELECT NULL = NULL      AS null_equals_null,
       NULL IS NULL     AS null_is_null,
       1 + NULL         AS one_plus_null,
       COALESCE(NULL, 0) AS coalesced
""")

# %%
demo = q("""
WITH t(id, discount) AS (VALUES (1, 0.1), (2, NULL), (3, 0.0))
SELECT * FROM t WHERE discount <> 0.1
""")
demo

# %% [markdown]
# Row 2 (unknown discount) is **not** returned by `discount <> 0.1`: "unknown is different from 0.1" is itself unknown,
# and `WHERE` only keeps rows that are definitely true. This silently drops missing rows in real analyses. Use
# `WHERE discount <> 0.1 OR discount IS NULL` if you want them, and `COALESCE(x, default)` to replace `NULL`.
# (`WITH t(...) AS (VALUES ...)` builds a throwaway table; we cover CTEs in notebook 04.)

# %% [markdown]
# ## 4. ORDER BY, LIMIT, OFFSET, DISTINCT
#
# A table has **no inherent order**. Without `ORDER BY` the database may return rows in any order (it often
# looks sorted, until it does not). `LIMIT n` without `ORDER BY` gives you "some n rows", which is fine
# for a peek and wrong for "top 5".

# %%
q("""
SELECT product_name, category, unit_price
FROM products
ORDER BY unit_price DESC
LIMIT 5
""")

# %% [markdown]
# Sort by several keys: the second key breaks ties in the first. `OFFSET` skips rows, which gives simple paging
# (slow on deep pages of big tables, where "keyset pagination" is preferred).

# %%
q("""
SELECT category, product_name, unit_price
FROM products
ORDER BY category ASC, unit_price DESC
LIMIT 5 OFFSET 5
""")

# %%
q("SELECT DISTINCT country FROM customers ORDER BY country")

# %% [markdown]
# `DISTINCT` de-duplicates whole rows of the columns you selected. `SELECT DISTINCT country, is_premium` gives
# unique *pairs*. If you reach for `DISTINCT` to make duplicates disappear after a join, stop: it usually hides a
# join bug (notebook 03).

# %%
q("SELECT DISTINCT country, is_premium FROM customers ORDER BY country, is_premium LIMIT 6")

# %% [markdown]
# ## 5. CASE WHEN: if/else inside SQL
#
# `CASE` creates a new value per row. Here we bucket customers by age.

# %%
q("""
SELECT first_name, age,
       CASE WHEN age < 25 THEN '18-24'
            WHEN age < 40 THEN '25-39'
            WHEN age < 60 THEN '40-59'
            ELSE '60+' END AS age_band
FROM customers
LIMIT 6
""")

# %% [markdown]
# Branches are checked top to bottom and the first match wins, so `age < 40` is only reached for ages of 25+.
# Without `ELSE`, unmatched rows get `NULL`.

# %% [markdown]
# ## 6. Dates in SQLite (and how other databases differ)
#
# SQLite has no real date type: our dates are **text** like `'2024-03-07'`. ISO format sorts correctly as text
# and works with `BETWEEN`, which is why ISO is the only format worth storing. Date functions parse that text.

# %%
q("""
SELECT order_id, order_date,
       strftime('%Y', order_date)      AS year,
       strftime('%m', order_date)      AS month,
       strftime('%w', order_date)      AS weekday_0_is_sunday,
       date(order_date, 'start of month') AS month_start
FROM orders
WHERE order_date BETWEEN '2024-12-20' AND '2024-12-24'
LIMIT 5
""")

# %% [markdown]
# **Dialect cheat sheet** (the concept is the same everywhere; the spelling differs):
#
# | Task | SQLite (here) | Postgres | MySQL | Snowflake / BigQuery |
# |---|---|---|---|---|
# | year of a date | `strftime('%Y', d)` | `EXTRACT(YEAR FROM d)` | `YEAR(d)` | `YEAR(d)` / `EXTRACT(YEAR FROM d)` |
# | first of month | `date(d, 'start of month')` | `DATE_TRUNC('month', d)` | `DATE_FORMAT(d, '%Y-%m-01')` | `DATE_TRUNC('month', d)` / `DATE_TRUNC(d, MONTH)` |
# | string concat | `a \|\| b` | `a \|\| b` | `CONCAT(a, b)` | `a \|\| b` / `CONCAT` |
# | boolean type | integers 0/1 | `BOOLEAN` | `TINYINT(1)` | `BOOLEAN` |
# | row limit | `LIMIT n` | `LIMIT n` | `LIMIT n` | `LIMIT n` (SQL Server: `TOP n`) |
# | case-insensitive match | `LIKE` | `ILIKE` | `LIKE` (collation) | `ILIKE` |

# %% [markdown]
# ## 7. The logical order of a query
#
# You **write** `SELECT` first, but the database **evaluates** in this order:
#
# `FROM` -> `WHERE` -> `GROUP BY` -> `HAVING` -> `SELECT` -> `DISTINCT` -> `ORDER BY` -> `LIMIT`
#
# Consequence: a column alias defined in `SELECT` does not exist yet inside `WHERE`, but it does inside `ORDER BY`.

# %%
try:
    q("SELECT unit_price - unit_cost AS margin FROM products WHERE margin > 100")
except Exception as e:
    print(type(e).__name__, "->", str(e)[:120])

# %%
q("SELECT product_name, unit_price - unit_cost AS margin FROM products WHERE unit_price - unit_cost > 100 ORDER BY margin DESC LIMIT 3")

# %% [markdown]
# (SQLite is unusually lenient and actually lets some queries reuse aliases in `WHERE`; the error above may not appear
# depending on version. Postgres and Snowflake are strict, so write portable SQL: repeat the expression, or wrap it in a subquery/CTE.)

# %% [markdown]
# ## Common mistakes
#
# 1. **`= NULL` instead of `IS NULL`.** It returns no rows and no error.
# 2. **Mixing `AND`/`OR` without parentheses** (shown above).
# 3. **`LIMIT` without `ORDER BY`** when you mean "top N". The result is arbitrary.
# 4. **Double quotes around text.** `WHERE country = "US"` may work in SQLite (it falls back to treating an unknown identifier as a string) and fails in Postgres.
# 5. **Integer division** in ratios: `SUM(a) / SUM(b)` of integer columns truncates. Multiply by `1.0`.
# 6. **`SELECT *` in production queries and views.**

# %%
# Mistake 1 in action: this counts zero rows, silently.
q("SELECT COUNT(*) AS n_equals_null FROM customers WHERE age = NULL")

# %% [markdown]
# ## Interview corner
#
# **1. What is the difference between `WHERE` and `HAVING`?**
# <details><summary>Answer</summary>
# <code>WHERE</code> filters individual rows before grouping; <code>HAVING</code> filters groups after aggregation, so it can use aggregates like
# <code>COUNT(*) &gt; 5</code>. Put every condition you can in <code>WHERE</code>: it shrinks the data before the expensive step. (We meet <code>HAVING</code> in notebook 02.)
# </details>
#
# **2. Why does <code>WHERE x &lt;&gt; 5</code> not return rows where x is NULL?**
# <details><summary>Answer</summary>
# SQL uses three-valued logic (true, false, unknown). <code>NULL &lt;&gt; 5</code> is unknown, and <code>WHERE</code> keeps only rows that evaluate to true.
# Add <code>OR x IS NULL</code> if you want them.
# </details>
#
# **3. Is the order of rows guaranteed without <code>ORDER BY</code>?**
# <details><summary>Answer</summary>
# No. The order depends on storage, indexes and the query plan and can change between runs. Any "first N" or "latest" logic needs an explicit <code>ORDER BY</code>
# (with a tiebreaker column if ties are possible).
# </details>
#
# **4. What is the logical order of query evaluation?**
# <details><summary>Answer</summary>
# FROM, WHERE, GROUP BY, HAVING, SELECT, DISTINCT, ORDER BY, LIMIT. This explains why aliases from SELECT can be used in ORDER BY but (in strict databases) not in WHERE.
# </details>

# %% [markdown]
# ## Exercises
#
# Each exercise asks for a **SQL string**. Write it in the starter cell (as the `..._sql` variable). The check cell runs your SQL
# and compares it with an independent answer computed in pandas.

# %%
# Independent pandas copies of the tables, used only by the checks. Reading whole tables is fine for a 6k-row demo.
customers = q("SELECT * FROM customers")
products = q("SELECT * FROM products")
orders = q("SELECT * FROM orders")

# %% [markdown]
# ### Exercise 1: Premium customers in Germany
# List `customer_id`, `first_name`, `age` of **premium customers (`is_premium = 1`) in Germany (`DE`) aged 30 or over**, oldest first
# (break ties by `customer_id`). *Hint: `WHERE` with `AND`, `ORDER BY age DESC, customer_id`.*

# %% tags=["starter"]
ex1_sql = """
SELECT ...
"""

# %%
res = q(ex1_sql)
exp = (customers[(customers.country == "DE") & (customers.is_premium == 1) & (customers.age >= 30)]
       .sort_values(["age", "customer_id"], ascending=[False, True])[["customer_id", "first_name", "age"]]
       .reset_index(drop=True))
assert list(res.columns) == ["customer_id", "first_name", "age"], f"columns should be customer_id, first_name, age; got {list(res.columns)}"
assert len(res) == len(exp), f"expected {len(exp)} rows, got {len(res)}"
assert res.equals(exp), "rows or order differ: sort by age DESC, then customer_id"
print(f"Exercise 1 passed ({len(res)} customers)")

# %% tags=["solution"]
ex1_sql = """
SELECT customer_id, first_name, age
FROM customers
WHERE country = 'DE' AND is_premium = 1 AND age >= 30
ORDER BY age DESC, customer_id
"""

# %% [markdown]
# ### Exercise 2: Cheap-but-profitable products
# Return `product_name`, `category`, and `margin_pct` (unit margin as a percentage of price, rounded to 1 decimal, column named exactly `margin_pct`)
# for products with **`unit_price` under 100** and **margin above 30%**, sorted by `margin_pct` descending then `product_name`.
# *Hint: repeat the margin expression in `WHERE`; use `100.0 *`.*

# %% tags=["starter"]
ex2_sql = """
SELECT ...
"""

# %%
res = q(ex2_sql)
p = products.assign(margin_pct=((products.unit_price - products.unit_cost) / products.unit_price * 100).round(1))
exp = (p[(p.unit_price < 100) & ((p.unit_price - p.unit_cost) / p.unit_price > 0.30)]
       .sort_values(["margin_pct", "product_name"], ascending=[False, True])[["product_name", "category", "margin_pct"]]
       .reset_index(drop=True))
assert list(res.columns) == ["product_name", "category", "margin_pct"], f"got columns {list(res.columns)}"
assert len(res) == len(exp) and len(exp) > 0, f"expected {len(exp)} rows, got {len(res)}"
assert (res.product_name == exp.product_name).all(), "product order differs"
assert ((res.margin_pct - exp.margin_pct).abs() < 0.11).all(), "margin_pct values differ (check the 100.0 * and ROUND(..., 1))"
print(f"Exercise 2 passed ({len(res)} products)")

# %% tags=["solution"]
ex2_sql = """
SELECT product_name, category,
       ROUND(100.0 * (unit_price - unit_cost) / unit_price, 1) AS margin_pct
FROM products
WHERE unit_price < 100
  AND (unit_price - unit_cost) / unit_price > 0.30
ORDER BY margin_pct DESC, product_name
"""

# %% [markdown]
# ### Exercise 3: A Black-Friday week
# Show `order_id`, `order_date`, `channel` of orders placed **from 2024-11-25 to 2024-12-01 inclusive** whose status is **not** `cancelled`
# and whose channel is `web` or `mobile`. Sort by `order_date`, then `order_id`, and return only the **first 15** rows.
# *Hint: `BETWEEN` includes both ends; `IN`; `<>`; `LIMIT`.*

# %% tags=["starter"]
ex3_sql = """
SELECT ...
"""

# %%
res = q(ex3_sql)
o = orders[(orders.order_date >= "2024-11-25") & (orders.order_date <= "2024-12-01") & (orders.status != "cancelled") & orders.channel.isin(["web", "mobile"])]
exp = o.sort_values(["order_date", "order_id"])[["order_id", "order_date", "channel"]].head(15).reset_index(drop=True)
assert len(o) > 15, "sanity: the window should have more than 15 matching orders"
assert list(res.columns) == ["order_id", "order_date", "channel"], f"got columns {list(res.columns)}"
assert res.equals(exp), "expected the first 15 matching orders sorted by order_date, order_id"
print("Exercise 3 passed")

# %% tags=["solution"]
ex3_sql = """
SELECT order_id, order_date, channel
FROM orders
WHERE order_date BETWEEN '2024-11-25' AND '2024-12-01'
  AND status <> 'cancelled'
  AND channel IN ('web', 'mobile')
ORDER BY order_date, order_id
LIMIT 15
"""

# %% [markdown]
# ### Exercise 4: Age bands with CASE
# For **every customer**, return `customer_id` and `segment` where `segment` is `'premium_senior'` if premium and age >= 60,
# `'premium'` if premium otherwise, `'senior'` if not premium and age >= 60, else `'standard'`. Order by `customer_id`.
# *Hint: order the `WHEN` branches from most to least specific.*

# %% tags=["starter"]
ex4_sql = """
SELECT ...
"""

# %%
res = q(ex4_sql)
import numpy as np
c = customers
seg = np.select([(c.is_premium == 1) & (c.age >= 60), c.is_premium == 1, c.age >= 60], ["premium_senior", "premium", "senior"], "standard")
exp = pd.DataFrame({"customer_id": c.customer_id, "segment": seg}).sort_values("customer_id").reset_index(drop=True)
assert list(res.columns) == ["customer_id", "segment"]
assert len(res) == len(customers), "one row per customer"
assert (res.customer_id.values == exp.customer_id.values).all() and (res.segment.values == exp.segment.values).all(), "segment labels differ"
print("Exercise 4 passed:", res.segment.value_counts().to_dict())

# %% tags=["solution"]
ex4_sql = """
SELECT customer_id,
       CASE WHEN is_premium = 1 AND age >= 60 THEN 'premium_senior'
            WHEN is_premium = 1               THEN 'premium'
            WHEN age >= 60                    THEN 'senior'
            ELSE 'standard' END AS segment
FROM customers
ORDER BY customer_id
"""

# %% [markdown]
# ### Exercise 5 (open-ended): Ask the shop a question
# Write two queries of your own, each using at least three clauses from this notebook (for example `WHERE` + `ORDER BY` + `LIMIT`).
# In a markdown cell, state the business question each answers and one thing you noticed in the result. Try the `products` table by
# category, and `orders` by `status` and `channel`.

# %% [markdown]
# ## Recap
# - `SELECT` picks/computes columns (`AS` renames), `FROM` names the table, `WHERE` filters rows before anything else.
# - `IN`, `BETWEEN` (inclusive), `LIKE`; parenthesise mixed `AND`/`OR`.
# - `NULL` is unknown: use `IS NULL`, `COALESCE`, and remember comparisons with it are never true.
# - No `ORDER BY`, no guaranteed order; `LIMIT` for top-N needs `ORDER BY`.
# - Dates are text in SQLite; date functions differ across databases (see the table).
#
# **Go deeper:** the SQLite documentation (sqlite.org/lang.html) for the exact dialect used here, and the PostgreSQL tutorial
# in the official docs (postgresql.org/docs) for the most standard-compliant open-source dialect.
#
# **Next:** [Aggregation and GROUP BY](02_aggregation_group_by_having.ipynb)

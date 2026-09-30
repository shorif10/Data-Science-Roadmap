# %% [markdown]
# # SQL from Python
#
# **Why this matters.** In a real project SQL does not live in a console: it lives inside Python code (a feature pipeline, an API, a dashboard). Doing that well means three things:
# never gluing user input into query text (the classic security hole), moving results into pandas cleanly, and knowing how much work to push to the database before pulling data out. You will also meet
# **DuckDB**, an in-process analytical database that runs SQL directly on CSV files and DataFrames, and is increasingly a data scientist's favourite tool.
#
# **You will learn**
# - run queries from Python with `sqlite3`, and pass values safely with **parameters** (and see an SQL injection succeed against the unsafe way)
# - load results with `pandas.read_sql`, and use **SQLAlchemy** (the standard Python database toolkit)
# - query CSV files and pandas DataFrames with **DuckDB**
# - decide what to compute in SQL and what to compute in pandas
#
# **Prerequisites:** [SQL analytics patterns](08_sql_analytics_patterns.ipynb) and [pandas loading](../05_Pandas/01_loading_and_inspecting_data.ipynb).
#
# > Data used: `datasets/shop.db` and the CSVs beside it (the same synthetic shop). The database is opened read-only, or copied to a temp dir when we write.

# %%
import sqlite3
import tempfile
import time
from pathlib import Path

import duckdb
import pandas as pd
import sqlalchemy as sa

DB_URI = "file:../datasets/shop.db?mode=ro"
con = sqlite3.connect(DB_URI, uri=True)

# %% [markdown]
# ## 1. The `sqlite3` basics
#
# The standard-library `sqlite3` module follows Python's DB-API, which Postgres (`psycopg`), MySQL and Snowflake connectors copy almost exactly: **connection -> cursor -> execute -> fetch**.

# %%
cur = con.execute("SELECT customer_id, first_name, country FROM customers WHERE country = ? ORDER BY customer_id LIMIT 3", ("CA",))
print([d[0] for d in cur.description])     # column names
print(cur.fetchall())                      # list of tuples

# %% [markdown]
# Set `row_factory = sqlite3.Row` to get rows you can index by column name instead of by position:

# %%
con.row_factory = sqlite3.Row
row = con.execute("SELECT * FROM customers WHERE customer_id = ?", (1,)).fetchone()
print(dict(row))
con.row_factory = None

# %% [markdown]
# ## 2. Parameters: never put values into SQL with f-strings
#
# Suppose a search box on a website feeds a product lookup. The tempting code builds the query with an f-string:

# %%
def find_products_unsafe(name_fragment):
    sql = f"SELECT product_id, product_name, category, unit_price, unit_cost FROM products WHERE product_name LIKE '%{name_fragment}%'"
    return pd.read_sql(sql, con)


print(len(find_products_unsafe("Pro")), "products match 'Pro'  (normal use works fine)")

# %% [markdown]
# A malicious (or just unlucky) user types this into the search box. The single quote closes our string literal, then they append their own `UNION SELECT` that reads a **different table**, and `--` comments out the rest of our query:

# %%
evil = "zzz' UNION SELECT customer_id, first_name, country, age, is_premium FROM customers --"
leaked = find_products_unsafe(evil)
print(f"the 'product search' returned {len(leaked)} rows, which are actually customers:")
leaked.head(3)

# %% [markdown]
# That is **SQL injection**: data typed by a user became *code*. With a writable connection the same trick can modify or delete data. (Python's `sqlite3` refuses to run two statements in one `execute`, which blocks the
# classic `; DROP TABLE ...` variant, but the `UNION` leak above still works, and other drivers may allow stacked statements.) The fix is **parameters**: send the SQL text and the values **separately** so the database
# never parses a value as SQL.

# %%
def find_products(name_fragment):
    return pd.read_sql("SELECT product_id, product_name, category, unit_price FROM products WHERE product_name LIKE '%' || ? || '%'", con, params=(name_fragment,))


print(len(find_products("Pro")), "matches for 'Pro'")
print(len(find_products(evil)), "matches for the attack string (it is just an odd search term now)")

# %% [markdown]
# *What just happened?* With `?` the driver ships the values out-of-band; the text `zzz' UNION ...` is compared with product names as a plain string and matches nothing. Placeholder styles differ between drivers
# (`?` for sqlite3, `%s` for psycopg and MySQL connectors, `:name` and `%(name)s` for named parameters); the principle is identical.
# Named parameters are clearer when there are many:

# %%
pd.read_sql("SELECT COUNT(*) AS n FROM orders WHERE channel = :ch AND order_date >= :since", con, params={"ch": "web", "since": "2025-01-01"})

# %% [markdown]
# **Limits of parameters:** they can only replace **values**, not identifiers (table or column names) or keywords (`ASC`/`DESC`). `ORDER BY ?` would sort by the constant string, not by a column.
# For a dynamic column, check it against an allow-list you wrote yourself:

# %%
ALLOWED_GROUPS = {"country", "is_premium"}


def customers_by(column):
    if column not in ALLOWED_GROUPS:
        raise ValueError(f"cannot group by {column!r}; allowed: {sorted(ALLOWED_GROUPS)}")
    return pd.read_sql(f"SELECT {column}, COUNT(*) AS n FROM customers GROUP BY {column} ORDER BY n DESC", con)  # safe: column came from our allow-list


print(customers_by("is_premium"))
try:
    customers_by("country; DROP TABLE customers")
except ValueError as e:
    print("rejected:", e)

# %% [markdown]
# ## 3. pandas in both directions
#
# `read_sql` runs a query and returns a DataFrame; `parse_dates` converts text columns to datetimes (SQLite stores dates as text, so pandas gets strings unless you ask).
# `DataFrame.to_sql` writes a DataFrame into a table. We write to a temporary database, never to the shop.

# %%
orders = pd.read_sql("SELECT * FROM orders", con, parse_dates=["order_date"])
print(orders.dtypes.to_dict())

# %%
tmp = tempfile.TemporaryDirectory()
scratch = sqlite3.connect(Path(tmp.name) / "scratch.db")
monthly = orders.groupby(orders.order_date.dt.to_period("M").astype(str)).size().rename("n_orders").rename_axis("month").reset_index()
monthly.to_sql("monthly_orders", scratch, index=False, if_exists="replace")
print(pd.read_sql("SELECT * FROM monthly_orders ORDER BY n_orders DESC LIMIT 3", scratch))
print(scratch.execute("SELECT sql FROM sqlite_master WHERE name = 'monthly_orders'").fetchone()[0])

# %% [markdown]
# Notice the table `to_sql` created has **no primary key and no constraints**, just column types. That is why the shop tables lacked them too (notebook 06). For real pipelines create the table
# yourself with proper DDL and use `to_sql(..., if_exists="append")`.
# For big results, `pd.read_sql(sql, con, chunksize=50_000)` yields DataFrames in pieces so you do not need to hold everything in memory.

# %%
scratch.close()
tmp.cleanup()

# %% [markdown]
# ## 4. SQLAlchemy
#
# **SQLAlchemy** is the standard Python toolkit that talks to almost any database through one API: you change only the connection URL (`sqlite:///...`, `postgresql+psycopg://user:pw@host/db`, `snowflake://...`).
# `pandas.read_sql` accepts a SQLAlchemy engine, and `text()` wraps raw SQL with **named bind parameters** (`:name`), safe for the same reason as above.
# (Never hard-code passwords in the URL: read them from environment variables or a secrets manager.)

# %%
engine = sa.create_engine("sqlite:///file:../datasets/shop.db?mode=ro&uri=true")
query = sa.text("SELECT country, COUNT(*) AS n FROM customers WHERE age >= :min_age GROUP BY country ORDER BY n DESC LIMIT 3")
with engine.connect() as conn:
    print(pd.read_sql(query, conn, params={"min_age": 40}))
    print(sa.inspect(conn).get_table_names())

# %% [markdown]
# `engine.connect()` used as a context manager returns the connection to a **pool** when the block ends. In production you create one engine per process and reuse it; never one per query.
# SQLAlchemy also has a Python-expression layer and an ORM (mapping tables to classes); analysts mostly use `text()` and `read_sql`, and application developers use the ORM.

# %%
engine.dispose()

# %% [markdown]
# ## 5. DuckDB: SQL on files and DataFrames
#
# **DuckDB** runs *inside your Python process* (like SQLite) but is built for analytics: it is columnar and fast on aggregations, reads CSV/Parquet directly, and can query a pandas DataFrame **by name** without loading anything into a database.
# Its SQL is close to Postgres, with friendly extras. It is not a replacement for a shared multi-user database; it is a great local analysis engine.

# %%
duck = duckdb.connect()   # in-memory
duck.sql("""
SELECT status, COUNT(*) AS n
FROM read_csv('../datasets/orders.csv')
GROUP BY status ORDER BY n DESC
""").df()

# %% [markdown]
# No import step, no schema: DuckDB detected types from the file. Now query a DataFrame you already have. `register` gives a DataFrame a table name inside the DuckDB connection (in a script, DuckDB can also find a
# variable by name automatically, but registering is explicit and works inside functions).

# %%
customers = pd.read_csv("../datasets/customers.csv")
duck.register("customers_df", customers)
duck.sql("SELECT country, ROUND(AVG(age), 1) AS avg_age, COUNT(*) AS n FROM customers_df GROUP BY country ORDER BY n DESC LIMIT 3").df()

# %% [markdown]
# Joins across files and DataFrames work the same way, and DuckDB includes the dialect features SQLite lacks, such as `QUALIFY` and `date_trunc`:

# %%
duck.sql("""
WITH m AS (
    SELECT date_trunc('month', CAST(order_date AS DATE)) AS month, channel, COUNT(*) AS n
    FROM read_csv('../datasets/orders.csv')
    GROUP BY ALL
)
SELECT * FROM m
QUALIFY ROW_NUMBER() OVER (PARTITION BY month ORDER BY n DESC) = 1     -- top channel per month: filter on a window, no extra wrapper query
ORDER BY month
LIMIT 4
""").df()

# %% [markdown]
# `GROUP BY ALL` groups by every non-aggregated column, another DuckDB/Snowflake convenience (in this DuckDB version it cannot be combined with `QUALIFY` in the same `SELECT`, which is why the grouping sits in a CTE). Passing values safely works here too: `duck.execute("... WHERE country = ?", ["US"]).df()`.
# Trade-offs: DuckDB's dialect has small differences from Postgres/Snowflake, results of `.df()` follow DuckDB's type rules, and it is single-machine (data must fit on your disk/memory budget, though it can spill).

# %% [markdown]
# ## 6. Push work to SQL, or pull data into pandas?
#
# | Do it in SQL when... | Do it in pandas when... |
# |---|---|
# | you filter/aggregate a big table down to a small result (network and memory cost) | the data is already in memory and small |
# | the data lives in a warehouse and the join keys are there | you need Python-only logic (ML, regex packages, custom functions) |
# | you need the database's indexes/partitions | you are doing exploratory reshaping (pivot, rolling, plotting) |
# | you want one reviewed, reproducible query as the source of truth | the transformation is easier to read as Python |
#
# Rule of thumb: **filter, join and aggregate in SQL; reshape, model and plot in Python.** The number of rows crossing the wire is the cost to watch. Measure it:

# %%
t0 = time.perf_counter()
in_sql = pd.read_sql("""
    SELECT o.channel, SUM(i.quantity * p.unit_price) AS revenue
    FROM orders o JOIN order_items i ON i.order_id = o.order_id JOIN products p ON p.product_id = i.product_id
    WHERE o.status <> 'cancelled' GROUP BY o.channel""", con)
t_sql = time.perf_counter() - t0

t0 = time.perf_counter()
o, i, p = (pd.read_sql(f"SELECT * FROM {t}", con) for t in ("orders", "order_items", "products"))
m = o[o.status != "cancelled"].merge(i, on="order_id").merge(p, on="product_id")
in_pd = m.assign(revenue=m.quantity * m.unit_price).groupby("channel", as_index=False).revenue.sum()
t_pd = time.perf_counter() - t0

print(f"SQL side : {len(in_sql)} rows transferred, {t_sql * 1000:.0f} ms")
print(f"pandas   : {len(o) + len(i) + len(p)} rows transferred, {t_pd * 1000:.0f} ms")
print("same answer:", bool((in_sql.set_index('channel').revenue.sort_index() - in_pd.set_index('channel').revenue.sort_index()).abs().max() < 0.01))

# %% [markdown]
# Both are fast here: the tables are tiny and the file is local. Now imagine the tables have 500 million rows on a server across a network: the second approach would try to move everything to your laptop. Count rows transferred, not milliseconds on a toy database.

# %% [markdown]
# ## Common mistakes
#
# 1. **f-strings or `%`/`.format` for values in SQL** = injection risk (shown above). Use parameters; allow-list identifiers.
# 2. **`SELECT *` into pandas** on wide/big tables: pull only the columns and rows you need.
# 3. **Forgetting to close connections** or creating an engine per call. Use context managers; one engine per process.
# 4. **Not committing** writes (Python's `sqlite3` needs `con.commit()`; `with con:` commits or rolls back but does *not* close).
# 5. **Dates arriving as strings.** Use `parse_dates` or convert immediately, and mind time zones.
# 6. **Hard-coding credentials in code or notebooks.** Use environment variables or a secret manager.
# 7. **Doing a join in pandas that the database could do**, moving millions of rows for nothing.

# %% [markdown]
# ## Interview corner
#
# **1. What is SQL injection and how do you prevent it?**
# <details><summary>Answer</summary>
# Untrusted input concatenated into SQL becomes executable SQL. Prevent it by sending values as bound parameters (prepared statements), never by string formatting; allow-list dynamic identifiers; and give the application account least privilege.
# </details>
#
# **2. Why can you not parameterise a table or column name?**
# <details><summary>Answer</summary>
# Placeholders stand for values that are bound after the query is parsed and planned; identifiers and keywords change the plan itself, so they must be part of the SQL text. Validate them against a fixed list.
# </details>
#
# **3. When would you use DuckDB instead of a server database or pandas?**
# <details><summary>Answer</summary>
# For fast SQL analytics over local files (CSV, Parquet) or DataFrames larger than is comfortable for pandas, without setting up a server. Not for concurrent multi-user transactional workloads.
# </details>
#
# **4. How do you decide between doing a transformation in SQL or in pandas?**
# <details><summary>Answer</summary>
# Minimise data moved: filter, join and aggregate close to the data in SQL, then bring the small result into Python for modelling, plotting or logic that SQL expresses poorly. Keep the SQL versioned so results are reproducible.
# </details>

# %% [markdown]
# ## Exercises

# %%
orders_df = pd.read_csv("../datasets/orders.csv")
order_items_df = pd.read_csv("../datasets/order_items.csv")
products_df = pd.read_csv("../datasets/products.csv")

# %% [markdown]
# ### Exercise 1: A safe lookup function
# Complete `orders_by_status(con, status, since)`: return a DataFrame with `order_id, customer_id, order_date, status, channel` of orders with the given status and `order_date >= since`, sorted by `order_id`, **using parameters** (no f-strings with values).
# The check also tries an injection string as the status.

# %% tags=["starter"]
def orders_by_status(con, status, since):
    ...

# %%
res = orders_by_status(con, "returned", "2025-01-01")
exp = orders_df[(orders_df.status == "returned") & (orders_df.order_date >= "2025-01-01")].sort_values("order_id").reset_index(drop=True)
assert len(exp) > 0
assert list(res.columns) == list(orders_df.columns), f"columns: {list(res.columns)}"
assert res.order_id.tolist() == exp.order_id.tolist(), "wrong rows: filter on status AND order_date >= since, sort by order_id"
inj = orders_by_status(con, "returned' OR '1'='1", "2000-01-01")
assert len(inj) == 0, "an injection string must be treated as an (unmatched) status value, so it should return 0 rows"
print(f"Exercise 1 passed ({len(res)} rows)")

# %% tags=["solution"]
def orders_by_status(con, status, since):
    return pd.read_sql(
        "SELECT order_id, customer_id, order_date, status, channel FROM orders WHERE status = ? AND order_date >= ? ORDER BY order_id",
        con, params=(status, since))

# %% [markdown]
# ### Exercise 2: A dynamic but safe breakdown
# Complete `orders_breakdown(con, column)`: count orders grouped by `column`, returning two columns (`column` value and `n`, in that order), sorted by `n` descending then the column value.
# Only `status` and `channel` are allowed; any other name (including an injection attempt) must raise `ValueError` **before** any SQL is run.

# %% tags=["starter"]
def orders_breakdown(con, column):
    ...

# %%
for col in ("status", "channel"):
    res = orders_breakdown(con, col)
    exp = orders_df.groupby(col).size().rename("n").reset_index().sort_values(["n", col], ascending=[False, True]).reset_index(drop=True)
    assert list(res.columns) == [col, "n"], f"columns: {list(res.columns)}"
    assert res[col].tolist() == exp[col].tolist() and res.n.tolist() == exp.n.tolist(), f"wrong counts for {col}"
for bad in ("customer_id", "status; DROP TABLE orders", "1"):
    try:
        orders_breakdown(con, bad)
    except ValueError:
        continue
    raise AssertionError(f"{bad!r} should raise ValueError")
print("Exercise 2 passed")

# %% tags=["solution"]
def orders_breakdown(con, column):
    if column not in {"status", "channel"}:
        raise ValueError(f"column not allowed: {column!r}")
    return pd.read_sql(f"SELECT {column}, COUNT(*) AS n FROM orders GROUP BY {column} ORDER BY n DESC, {column}", con)

# %% [markdown]
# ### Exercise 3: DuckDB on CSV files
# With DuckDB, and **no database and no pandas merge**, compute revenue by product `category` on non-cancelled orders directly from the three CSV files
# (`orders.csv`, `order_items.csv`, `products.csv` in `../datasets/`). Store the DataFrame in `revenue_by_cat` with columns `category`, `revenue` (rounded to 2 decimals), sorted by revenue descending.
# *Hint: `duckdb.sql("... FROM read_csv('../datasets/orders.csv') o JOIN read_csv('../datasets/order_items.csv') i ON ...").df()`.*

# %% tags=["starter"]
revenue_by_cat = ...

# %%
m = order_items_df.merge(orders_df, on="order_id").merge(products_df, on="product_id")
m = m[m.status != "cancelled"]
exp = (m.quantity * m.unit_price).groupby(m.category).sum().round(2).sort_values(ascending=False)
assert isinstance(revenue_by_cat, pd.DataFrame), "revenue_by_cat should be a DataFrame (use .df() on the DuckDB result)"
assert list(revenue_by_cat.columns) == ["category", "revenue"], f"columns: {list(revenue_by_cat.columns)}"
assert revenue_by_cat.category.tolist() == exp.index.tolist(), "category order should be by revenue descending"
assert ((revenue_by_cat.revenue - exp.values).abs() < 0.02).all(), "revenue values differ (non-cancelled orders only; quantity * unit_price)"
print("Exercise 3 passed")

# %% tags=["solution"]
revenue_by_cat = duckdb.sql("""
    SELECT p.category, ROUND(SUM(i.quantity * p.unit_price), 2) AS revenue
    FROM read_csv('../datasets/orders.csv') o
    JOIN read_csv('../datasets/order_items.csv') i ON i.order_id = o.order_id
    JOIN read_csv('../datasets/products.csv') p    ON p.product_id = i.product_id
    WHERE o.status <> 'cancelled'
    GROUP BY p.category
    ORDER BY revenue DESC
""").df()

# %% [markdown]
# ### Exercise 4: Latest order per customer with QUALIFY
# Register the DataFrame `orders_df` in a DuckDB connection and return, in `latest_orders`, **each customer's most recent order** (latest `order_date`; if two share a date, the higher `order_id`): columns `customer_id`, `order_id`, `order_date`,
# sorted by `customer_id`. Use `QUALIFY ROW_NUMBER() OVER (...) = 1`.

# %% tags=["starter"]
latest_orders = ...

# %%
exp = orders_df.sort_values(["customer_id", "order_date", "order_id"]).groupby("customer_id").tail(1).sort_values("customer_id")[["customer_id", "order_id", "order_date"]].reset_index(drop=True)
assert isinstance(latest_orders, pd.DataFrame), "latest_orders should be a DataFrame"
assert list(latest_orders.columns) == ["customer_id", "order_id", "order_date"], f"columns: {list(latest_orders.columns)}"
assert len(latest_orders) == orders_df.customer_id.nunique(), "exactly one row per customer"
assert latest_orders.order_id.tolist() == exp.order_id.tolist(), "wrong order picked: latest order_date, then highest order_id"
print(f"Exercise 4 passed ({len(latest_orders)} customers)")

# %% tags=["solution"]
d = duckdb.connect()
d.register("orders_df", orders_df)
latest_orders = d.sql("""
    SELECT customer_id, order_id, order_date
    FROM orders_df
    QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC, order_id DESC) = 1
    ORDER BY customer_id
""").df()
latest_orders["order_date"] = latest_orders["order_date"].astype(str)

# %% [markdown]
# ### Exercise 5 (open-ended): SQL or pandas?
# Take the top-3-products-per-category question from notebook 08. Implement it (a) fully in SQL and (b) by loading whole tables into pandas. Compare the number of rows transferred and the readability of each,
# and write two sentences on which you would ship in a nightly job that runs against a warehouse table of 100 million order lines.

# %% [markdown]
# ## Recap
# - DB-API pattern: connect, execute, fetch. Pass **values as parameters** (`?`, `:name`, `%s`), never via f-strings; parameters cannot replace identifiers, so allow-list those.
# - `pd.read_sql` (with `params`, `parse_dates`, `chunksize`) and `DataFrame.to_sql` move data both ways; `to_sql` creates constraint-free tables.
# - SQLAlchemy gives one API for many databases: one engine per process, `text()` with bind parameters.
# - DuckDB runs SQL directly on CSV/Parquet and DataFrames, with `QUALIFY`, `GROUP BY ALL` and more; great for local analytics.
# - Push filtering, joins and aggregation to the database; keep modelling and plotting in Python. Watch rows transferred.
#
# **Go deeper:** Python docs for `sqlite3` (docs.python.org), SQLAlchemy docs (docs.sqlalchemy.org), DuckDB docs (duckdb.org/docs), and the OWASP SQL Injection Prevention Cheat Sheet (cheatsheetseries.owasp.org).
#
# **Next module:** [Matplotlib: anatomy of a figure](../07_Matplotlib/01_anatomy_figure_axes.ipynb)
